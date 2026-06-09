"""
Phase 1 HEADLINE benchmark: continuous batching vs (fair) chunked static
batching on the SAME varied-length workload at the SAME capacity B.

Reports throughput (output tok/s), P50/P99 TTFT, P50/P99 end-to-end latency,
and peak GPU memory -- the "Nx throughput at fixed P99" head-to-head.

The metric helpers (percentile / summarize) are factored out so the later
QPS-sweep harness can reuse them unchanged.

Run:  python bench/ab_benchmark.py
NOTE: on CPU this validates the harness only. Continuous batching is a
memory-bound GPU win; on CPU it may tie or LOSE (compute-bound + the per-step
cache-copy churn in _decode_step_batched). Run on the L4 for the real number.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from engine import static_batch_generate
from scheduler import ContinuousBatchingEngine  # noqa: E402


# Varied NATURAL lengths: short factual prompts finish fast (early EOS),
# open-ended ones run long. That spread is what creates head-of-line blocking
# -- the exact thing continuous batching removes. Uniform lengths hide the win.
# Mix spans one-word answers -> short facts -> paragraphs -> long essays so the
# output-length distribution (and thus the P99 tail) is realistic.
WORKLOAD = [
    # very short (early EOS)
    "What is 2+2?",
    "Name the capital of France.",
    "Give me a one-word answer: is the sky blue?",
    "What color is grass? One word.",
    "Reply with just 'yes' or 'no': is water wet?",
    # short
    "List three primary colors.",
    "Name two programming languages.",
    "What does CPU stand for?",
    # medium
    "Explain what a KV cache is in two sentences.",
    "Write a short paragraph about why the sky is blue.",
    "Summarize what an operating system does, briefly.",
    "In a few sentences, what is a hash map?",
    # long (run toward the cap)
    "Describe, step by step and in detail, how a CPU executes one instruction.",
    "Explain the difference between prefill and decode in LLM inference, in detail.",
    "Write a detailed explanation of how virtual memory and page tables work.",
    "Give a thorough, multi-paragraph overview of how TCP establishes a connection.",
]


# ---- reusable metric helpers (the QPS sweep will import these) ----

def percentile(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def summarize(name, ttfts, latencies, total_tokens, wall_clock, peak_mem):
    return {
        "system": name,
        "requests": len(latencies),
        "wall_clock_s": round(wall_clock, 3),
        "throughput_tok_s": round(total_tokens / wall_clock, 2) if wall_clock else None,
        "ttft_p50": round(percentile(ttfts, 50), 3),
        "ttft_p99": round(percentile(ttfts, 99), 3),
        "latency_p50": round(percentile(latencies, 50), 3),
        "latency_p99": round(percentile(latencies, 99), 3),
        "peak_mem_mb": round(peak_mem / 1e6, 1) if peak_mem else None,
    }


def _reset_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_mem():
    return torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None


# ---- the two systems under test ----

def bench_continuous(prompts, B, cap):
    _reset_mem()
    eng = ContinuousBatchingEngine(max_batch_size=B)
    t0 = time.perf_counter()
    for p in prompts:
        eng.submit(p, max_new_tokens=cap)      # all "arrive" at ~t0
    completed = eng.run()
    wall = time.perf_counter() - t0
    # TTFT/latency are measured from each request's own arrival stamp, so they
    # already include queueing delay for requests admitted in later waves.
    ttfts = [r.t_first - r.t_arrival for r in completed]
    lats = [r.t_done - r.t_arrival for r in completed]
    total = sum(len(r.output_ids) for r in completed)
    return summarize("continuous", ttfts, lats, total, wall, _peak_mem())


def bench_chunked_static(prompts, B, cap):
    # FAIR baseline: process in groups of B, each group padded-to-max and run
    # lockstep until the group's SLOWEST finishes, then the next group. No
    # mid-group admission -- that single restriction is what continuous lifts.
    _reset_mem()
    ttfts, lats, total = [], [], 0
    t0 = time.perf_counter()
    for i in range(0, len(prompts), B):
        chunk = prompts[i:i + B]
        chunk_start = time.perf_counter()
        r = static_batch_generate(chunk, max_new_tokens=cap)
        done_at = time.perf_counter()
        total += r["out_tokens"]
        # every request in a chunk shares the chunk's prefill (TTFT) and its
        # lockstep completion time (latency)
        for _ in chunk:
            ttfts.append((chunk_start - t0) + r["ttft_s"])
            lats.append(done_at - t0)
    wall = time.perf_counter() - t0
    return summarize("chunked_static", ttfts, lats, total, wall, _peak_mem())


def print_row(s):
    print(f"{s['system']:>16} | reqs {s['requests']:>3} | "
          f"wall {s['wall_clock_s']:>7}s | tok/s {str(s['throughput_tok_s']):>7} | "
          f"TTFT p50/p99 {s['ttft_p50']}/{s['ttft_p99']}s | "
          f"lat p50/p99 {s['latency_p50']}/{s['latency_p99']}s | "
          f"peakMB {s['peak_mem_mb']}")


if __name__ == "__main__":
    N = 8        # bump to 100+ on GPU for stable P99
    B = 4        # capacity (max concurrent sequences) -- SAME for both systems
    CAP = 64     # max_new_tokens cap; real length variance comes from early EOS

    prompts = [WORKLOAD[i % len(WORKLOAD)] for i in range(N)]

    stat = bench_chunked_static(prompts, B, CAP)
    cont = bench_continuous(prompts, B, CAP)

    print(f"\nworkload={N} reqs, capacity B={B}, max_new_tokens={CAP}\n")
    print_row(stat)
    print_row(cont)

    if cont["throughput_tok_s"] and stat["throughput_tok_s"]:
        x = cont["throughput_tok_s"] / stat["throughput_tok_s"]
        print(f"\ncontinuous / static throughput = {x:.2f}x")
    if not torch.cuda.is_available():
        print("\n[CPU run = harness validation only. Continuous may tie/lose here; "
              "the win is a memory-bound GPU effect. Run on the L4 for the headline.]")

"""
Phase 0 baseline benchmark.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import static_batch_generate  # noqa: E402  (path insert must run first)

# prompts of varied length to test its efficacy
PROMPTS = [
    "Hi.",
    "What is 2+2?",
    "Give me a haiku about autumn.",
    "List three uses for a paperclip.",
    "Explain what a KV cache is in two sentences.",
    "Write a short paragraph about why the sky is blue.",
    "Describe how a CPU executes a single instruction, step by step.",
    "Summarize the plot of Romeo and Juliet, covering the key events of each act.",
]


def make_batch(b):
    """Cycle the prompt pool up to size b so every batch carries a length mix."""
    return [PROMPTS[i % len(PROMPTS)] for i in range(b)]


def sweep(batch_sizes, max_new_tokens=64, repeats=3):
    rows = []
    for b in batch_sizes:
        prompts = make_batch(b)
        # Repeat each point and take the MEDIAN run to smooth out noise.
        runs = [static_batch_generate(prompts, max_new_tokens=max_new_tokens)
                for _ in range(repeats)]
        runs.sort(key=lambda r: r["total_s"])
        r = runs[len(runs) // 2]
        rows.append({
            "batch_size": b,
            "ttft_s": r["ttft_s"],
            "latency_s": r["total_s"],
            "out_tokens": r["out_tokens"],
            "throughput_tok_s": r["throughput_tok_s"],
            "per_seq_throughput": r["throughput_tok_s"] / b,
        })
    return rows


def print_table(rows):
    hdr = (f"{'B':>4} {'ttft_s':>8} {'latency_s':>10} "
           f"{'out_tok':>8} {'tok/s':>9} {'tok/s/seq':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['batch_size']:>4} {r['ttft_s']:>8.3f} {r['latency_s']:>10.3f} "
              f"{r['out_tokens']:>8} {r['throughput_tok_s']:>9.2f} "
              f"{r['per_seq_throughput']:>10.2f}")

# CPU smoke test
if __name__ == "__main__":
    batch_sizes = [1, 2, 4, 8]
    rows = sweep(batch_sizes, max_new_tokens=32, repeats=2)
    print_table(rows)

    out_path = os.path.join(os.path.dirname(__file__), "baseline_static.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out_path}")

    print(
        "\nCPU CAVEAT: CPU inference is compute-bound, so total tok/s may stay "
        "flat or DROP as B grows -- the batching win is a GPU phenomenon "
        "(decode is memory-bound there, so batching amortizes the weight load). "
        "Don't read the throughput story off CPU numbers; use this to validate "
        "the harness, then run the curve on the GPU."
    )

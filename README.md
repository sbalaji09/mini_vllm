# mini-vLLM

A from-scratch, benchmarked LLM **inference server**. I serve a small off-the-shelf model
(Qwen2.5-1.5B-Instruct) but write the serving layer myself — the generation loop, a
continuous-batching scheduler, a paged KV cache, and a custom Triton PagedAttention GPU
kernel. The model is not the point; the systems around it are.

Everything below was **measured** on an NVIDIA L4 (24 GB) via Modal, bf16. Where a number has
a caveat, it's stated — including the places where the naive approach *lost*.

## Headline results

| Component | Result | Notes |
|---|---|---|
| **Continuous batching** vs static | **1.50× throughput, ~1.5× lower P99** | 346 vs 231 tok/s; P99 15.0s vs 22.2s, equal memory |
| **Paged KV cache** vs pad-to-max | **3× (up to ~5×) concurrency**, util **40%→95%** | sequences sustained under a fixed KV budget |
| **Triton PagedAttention kernel** | numerically exact (~1e-7 vs torch); **~99× faster** per-op than the gather it replaces | caveat below |
| End-to-end generation **through the kernel** | coherent text, per-step logits match HF SDPA to ~0.15 (bf16) | the full loop closed |

## The core ideas

- **Prefill vs decode.** Prefill processes the whole prompt in one pass (compute-bound).
  Decode emits one token per step and reloads all model weights from HBM each step
  (memory-bound). The KV cache is what makes decode tractable: cache K and V (reused by every
  future query), not Q (used once).
- **Why batching helps is a *memory* fact.** Decode reloads the weights regardless of batch
  size, so batching amortizes that fixed cost across B sequences — throughput scales with B
  until KV-cache memory runs out. (On CPU, which is compute-bound, batching buys nothing —
  see Phase 1.)
- **Static batching wastes the GPU two ways:** padding (short prompts padded to the longest)
  and head-of-line blocking (short sequences trapped until the longest in the batch finishes).
- **Continuous batching** fixes both by scheduling at the *token* level: admit/retire
  sequences every step so no slot idles. **Paging** fixes KV-cache fragmentation by storing
  K/V in fixed blocks with a per-sequence block table — OS virtual memory for the cache.

## What I built (vs. what's library)

The model weights and forward pass are HuggingFace; **everything else is mine**:

- **Generation loop** — greedy decode with KV-cache feedback (`engine.py`).
- **Continuous-batching scheduler** — token-level admit/decode/retire over a *persistent
  batched KV cache* (`scheduler.py`).
- **Paged KV cache** — block allocator, per-sequence block table, free-block pool
  (`kv_cache.py`); a paged generation engine (`paged_engine.py`).
- **Triton PagedAttention kernel** — flash-decode with online softmax, paged block reads,
  and GQA (`paged_attention.py`); a generation engine that runs through it
  (`paged_kernel_engine.py`).

## The phases (and the honest story)

### Phase 0 — baseline

Single-request greedy loop, a naive static-batch baseline (pad-to-max, lockstep), and a
metrics harness (TTFT, TPOT, throughput, P50/P99, peak memory). The number to beat.

### Phase 1 — continuous batching → 1.50×, after a real fight

The first implementation kept a separate KV cache per sequence and **re-batched them every
step**. On the L4 it *lost*: **0.69× static**. Profiling the engine showed why — **57% of
wall-clock was KV-cache copy churn** (per-step pad/concat/extract), only 35% was the actual
forward. The fix was a **persistent batched cache** grown in place on the hot path and
reshaped only when batch membership changes, plus removing per-row GPU syncs. Result:
**1.50× throughput and ~1.5× lower P99** at equal memory, with the time breakdown flipping to
76% real forward / 18% prefill / 6% cache management. *Measure, diagnose, fix, re-measure.*

### Phase 2 — paged KV cache → 3× concurrency

Fixed 16-token blocks, a per-sequence block table, and a free-block pool. Under a fixed block
budget, paged allocation fits **3× (up to ~5×)** the concurrent sequences of contiguous
pad-to-max, lifting KV-cache utilization from **~40% to ~95%** — internal fragmentation is
bounded to <1 block per sequence, vs. pad-to-max where one long sequence forces every slot to
reserve its length. An end-to-end paged engine (gather blocks → forward → scatter) is
equivalence-verified token-for-token against the single-sequence reference.

**Honest cost:** in pure PyTorch, attention needs contiguous K/V, so the paged engine must
*gather* scattered blocks into a contiguous tensor every step — measured at **2.87× slower**
than the contiguous engine (slower than static, too). So **PyTorch paging is a memory/
concurrency tool, not a throughput tool.** That motivated Phase 4.

### Phase 4 — a Triton PagedAttention kernel → closing the loop

The gather is exactly what a real PagedAttention CUDA/Triton kernel removes by attending over
the blocks **in place**. Built it as a ladder (softmax → contiguous flash-decode → paged →
batched + GQA), each step verified against a torch reference to **~1e-7**. The kernel does
online softmax, reads K/V directly from the block table, and handles GQA.

- **Microbenchmark:** kernel 0.158 ms/step vs gather+attend 15.6 ms/step ≈ **99× faster**.
  *Caveat (cited honestly):* the baseline is the naive per-sequence-loop gather, so this
  bundles copy-avoidance + kernel fusion + Python-overhead removal — not pure memory savings.
- **End-to-end (Tier 2):** a generation engine that owns the decode forward — reusing the
  model's projections/RoPE/norms/MLP but replacing attention with the kernel reading the
  paged pool. Per-step logits match HF SDPA to **~0.15** (bf16), and it generates coherent
  text. Long-sequence token divergence from the reference is **reduction-order**, not a
  correctness bug (same math, different summation order; the per-step match is the proof).

## Benchmark methodology

- **Equal capacity B** for every system — we compare scheduling/cache policy, not batch size.
- **Varied-length workload** (short queries + long generations); length variance is what
  exposes padding waste and head-of-line blocking. Uniform lengths hide the win.
- **Fair static baseline** = chunked at capacity B, each chunk pad-to-max and lockstep.
- **GPU for headline numbers**, CPU/fp32 for correctness and harness validation (the
  throughput story is a memory-bound GPU effect).
- **Correctness gate:** greedy is deterministic, so the continuous/paged engines must match an
  independently-written single-sequence loop token-for-token (verified alone and in
  mixed-length batches).

## Honest limitations

- PyTorch paging is **slower per-op** than the contiguous cache (the gather); the throughput
  recovery requires the custom kernel.
- The Triton kernel and SDPA differ in reduction order, so bf16 generation drifts from the
  reference over long sequences (expected; per-step logits agree).
- **Greedy decoding only**; no sampling.
- **No preemption/recompute** — a sequence that can't get a block mid-prefill leaks its
  partial blocks; budgets are sized generously to avoid it (real systems preempt).
- Per-request (unbatched) prefill; single GPU.

## Repo layout

```text
engine.py              # model load, single-request loop, static-batch baseline   (Phase 0)
scheduler.py           # continuous batching + persistent batched KV cache         (Phase 1)
kv_cache.py            # block allocator, block table, free pool, PagedKVCache      (Phase 2)
paged_engine.py        # paged generation via gather + SDPA                         (Phase 2B)
paged_attention.py     # Triton kernels: softmax -> flash-decode -> paged -> GQA    (Phase 4 T1)
paged_kernel_engine.py # generation through the kernel (own the decode forward)     (Phase 4 T2)
bench/
  metrics.py           # Phase 0 batch-size sweep
  ab_benchmark.py      # static vs continuous vs paged (throughput / P50 / P99)
  paging_experiment.py # paged vs contiguous concurrency under a fixed budget
  modal_run.py         # L4 runner: the A/B throughput benchmark
  modal_kernel.py      # L4 runner: the Triton kernel ladder + microbench
  modal_tier2.py       # L4 runner: generation through the kernel
```

## Running it

```bash
# correctness / harness validation (CPU is fine)
python scheduler.py            # continuous-batching engine
python kv_cache.py             # paged allocator unit tests
python bench/ab_benchmark.py   # static vs continuous vs paged (small, local)

# real numbers + the kernel (needs a GPU; uses Modal)
pip install modal && modal token new
modal run bench/modal_run.py      # throughput: static / continuous / paged
modal run bench/modal_kernel.py   # Triton kernel correctness + microbench
modal run bench/modal_tier2.py    # generate through the kernel
```

**Stack:** Python, PyTorch, HuggingFace transformers 5.x, Triton, Modal (L4).
**Model:** Qwen2.5-1.5B-Instruct (dense, GQA, Apache-2.0) — chosen to be boring so the systems
story stays clean.

"""
Run the Phase 1 HEADLINE benchmark (bench/ab_benchmark.py) on a Modal L4 GPU.

This is the experiment the whole project is built to run: does continuous
batching actually sustain higher throughput than static batching? On CPU it
LOST (~0.8x) because CPU is compute-bound. The L4 is the memory-bound regime
where decode reloads the weights from HBM every step regardless of batch size,
so batching amortizes that load -- and the thesis should finally hold.

Prereqs (local, one-time):
    pip install modal
    modal token new

Run (from the repo root):
    modal run bench/modal_run.py
    modal run bench/modal_run.py --n 200 --b 32 --cap 128
"""
import modal

app = modal.App("mini-vllm-bench")

# Image: CUDA torch (default PyPI wheel) + transformers pinned to match local,
# since the engine depends on the 5.x DynamicCache API (cache.layers[i].keys).
# The three add_local_* lines ship your source into the container at /root.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers==5.10.2")
    .add_local_file("engine.py", "/root/engine.py")
    .add_local_file("scheduler.py", "/root/scheduler.py")
    .add_local_file("kv_cache.py", "/root/kv_cache.py")
    .add_local_file("paged_engine.py", "/root/paged_engine.py")
    .add_local_dir("bench", "/root/bench")
)

# Persist the HF model download (~3GB) across runs so we fetch it only once.
hf_cache = modal.Volume.from_name("mini-vllm-hf-cache", create_if_missing=True)


@app.function(
    gpu="L4",
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=1800,
)
def run_bench(n: int, b: int, cap: int):
    import sys, torch
    sys.path.insert(0, "/root/bench")   # ab_benchmark self-adds /root for engine/scheduler
    import ab_benchmark as ab

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"
    print(f"device={dev}  gpu={gpu}")

    prompts = [ab.WORKLOAD[i % len(ab.WORKLOAD)] for i in range(n)]

    # warmup: run one small batch so CUDA init / kernel autotune isn't charged
    # to the first measured run.
    ab.bench_continuous(prompts[:b], b, 8)

    stat = ab.bench_chunked_static(prompts, b, cap)
    cont = ab.bench_continuous(prompts, b, cap)
    paged = ab.bench_paged(prompts, b, cap, num_blocks=max(512, b * 16))

    print(f"\nworkload={n} reqs, capacity B={b}, max_new_tokens={cap}\n")
    ab.print_row(stat)
    ab.print_row(cont)
    ab.print_row(paged)
    ab.print_breakdown(cont)
    if cont["throughput_tok_s"] and stat["throughput_tok_s"]:
        print(f"\ncontinuous / static throughput = "
              f"{cont['throughput_tok_s'] / stat['throughput_tok_s']:.2f}x")
    if cont["throughput_tok_s"] and paged["throughput_tok_s"]:
        print(f"paged per-op cost: continuous / paged = "
              f"{cont['throughput_tok_s'] / paged['throughput_tok_s']:.2f}x slower "
              f"(the per-step gather; a CUDA kernel would remove it)")

    hf_cache.commit()   # persist the downloaded weights for next run
    return {"static": stat, "continuous": cont, "paged": paged}


@app.local_entrypoint()
def main(n: int = 100, b: int = 16, cap: int = 128):
    run_bench.remote(n, b, cap)

"""
Modal runner for Phase 4 Tier 2 (generate through the Triton kernel).
Run:  modal run bench/modal_tier2.py
T2a = a single custom decode step must match the model's decode logits.
"""
import modal

app = modal.App("mini-vllm-tier2")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "triton", "transformers==5.10.2")
    .add_local_file("engine.py", "/root/engine.py")
    .add_local_file("kv_cache.py", "/root/kv_cache.py")
    .add_local_file("paged_attention.py", "/root/paged_attention.py")
    .add_local_file("paged_kernel_engine.py", "/root/paged_kernel_engine.py")
)

hf_cache = modal.Volume.from_name("mini-vllm-hf-cache", create_if_missing=True)


@app.function(gpu="L4", image=image, volumes={"/root/.cache/huggingface": hf_cache}, timeout=1200)
def run():
    import sys, torch
    sys.path.insert(0, "/root")
    import paged_kernel_engine as pke
    print(f"gpu={torch.cuda.get_device_name(0)}")
    pke.test_decode_step()         # T2a — single step matches the model
    pke.test_generate_matches()    # T2b — full generation through the kernel
    hf_cache.commit()


@app.local_entrypoint()
def main():
    run.remote()

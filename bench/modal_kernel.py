"""
Modal runner for the Phase 4 Triton kernels (GPU-only).

Run:  modal run bench/modal_kernel.py
Each ladder step adds a check here; K0 = the row-softmax warm-up that proves the
Triton toolchain + Modal loop work before we write the hard attention kernel.
"""
import modal

app = modal.App("mini-vllm-kernel")

# torch on Linux+CUDA bundles triton; pin transformers off (not needed for the
# standalone kernel). triton listed explicitly to be safe.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "triton")
    .add_local_file("paged_attention.py", "/root/paged_attention.py")
)


@app.function(gpu="L4", image=image, timeout=900)
def run():
    import sys, torch, triton
    sys.path.insert(0, "/root")
    import paged_attention as pa

    print(f"gpu={torch.cuda.get_device_name(0)}  torch={torch.__version__}  triton={triton.__version__}")
    pa.test_softmax()         # K0
    pa.test_flash_decode()    # K1
    # K2/K3 checks get appended here as the ladder progresses.


@app.local_entrypoint()
def main():
    run.remote()

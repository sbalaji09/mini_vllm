"""
small Triton kernel file that implements a custom-row wise softmax kernel on GPU
"""
import torch
import triton
import triton.language as tl


# defines the GPU kernel
@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, row_stride, BLOCK: tl.constexpr):
    # gets that this Triton program is responsible for and creates column offsets from 0 to BLOCK-1
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    # computes the input and output memory addresses for the start of this row
    row_start = x_ptr + row * row_stride
    out_start = out_ptr + row * row_stride

    # loads one row tile and subtracts the row mask for numerical stability
    x = tl.load(row_start + cols, mask=mask, other=float("-inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)

    # normalizes the row sum to produce softmax probabilities
    y = e / tl.sum(e, axis=0)

    # writes the softmax result back to output for valid coumns only
    tl.store(out_start + cols, y, mask=mask)

# python wrapper around the Triton kernel
def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2 # requires a CUDA 2D tensor
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n_cols)
    softmax_kernel[(n_rows,)](x, out, n_cols, x.stride(0), BLOCK=BLOCK)
    return out


# ---------------- correctness harness ----------------

def test_softmax():
    torch.manual_seed(0)
    x = torch.randn(128, 500, device="cuda", dtype=torch.float32)
    ref = torch.softmax(x, dim=-1)
    got = triton_softmax(x)
    max_err = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, atol=1e-5), f"mismatch, max abs err {max_err}"
    print(f"K0 OK — Triton softmax matches torch (max abs err {max_err:.2e}).")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Triton needs a CUDA GPU — run this on Modal: modal run bench/modal_kernel.py")
    else:
        test_softmax()

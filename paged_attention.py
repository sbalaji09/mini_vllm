"""
standalone kernel for PagedAttention
"""
import torch
import triton
import triton.language as tl


# ---------------- K0: row softmax ----------------
# One program per row. Teaches: program_id, tl.arange offsets, masked load/store,
# and the reductions (tl.max / tl.sum) that online softmax is built from.

@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)          # tile of column offsets [0, BLOCK)
    mask = cols < n_cols                # bounds: BLOCK is padded up to a power of 2
    row_start = x_ptr + row * row_stride
    out_start = out_ptr + row * row_stride

    # TODO (yours): type these understanding each line — they ARE the softmax,
    # and steps 1/2 are exactly what K1's online version generalizes.
    #   1. x = tl.load(row_start + cols, mask=mask, other=float("-inf"))
    #        (masked-out lanes load -inf so they never win the max or add to the sum)
    #   2. x = x - tl.max(x, axis=0)            # numerical stability
    #      e = tl.exp(x)
    #      y = e / tl.sum(e, axis=0)
    #   3. tl.store(out_start + cols, y, mask=mask)
    # (no `raise`/Python exceptions inside @triton.jit — it compiles to GPU code.)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 2
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n_cols)
    softmax_kernel[(n_rows,)](x, out, n_cols, x.stride(0), BLOCK=BLOCK)
    return out


# ---------------- correctness harness (glue) ----------------

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

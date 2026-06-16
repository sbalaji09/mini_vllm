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


# this defines the low-level GPU kernel where one query token attends over a sequence's contiguous K/V, single sequence, and single head
@triton.jit
def flash_decode_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                        seq_len, scale,
                        stride_kn, stride_kd,      # memory strides for indexing K
                        stride_vn, stride_vd,      # memory strides for indexing V
                        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, HEAD_DIM: tl.constexpr):
    
    # creates head dimension offsets and masks out specific lanes
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    q = tl.load(q_ptr + offs_d, mask=mask_d, other=0.0)     # [BLOCK_D]

    # running online-softmax state
    m = -float("inf")                                       # running max
    l = 0.0                                                 # running sum
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)             # running weighted V

    # iterate through K/V in blocks instead of loading the whole sequence at once
    for start in range(0, seq_len, BLOCK_N):
        # creates token offsets for this block and masks out positions past the sequence end
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        # load a block of keys and values
        k = tl.load(k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v = tl.load(v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # computes dot-product attention scores for this specific block
        scores = tl.sum(k*q[None, :], axis=1) * scale
        scores = tl.where(mask_n, scores, -float("inf"))

        # updates running max across all tokens seen so far
        m_new = tl.maximum(m, tl.max(scores, axis=0))

        # computes this block's exponentials and the correction factor for previous accumulated values
        p = tl.exp(scores-m_new)
        corr = tl.exp(m - m_new)

        # updates the softmax denominator and the weighted value sum
        l = l * corr + tl.sum(p, axis=0)
        acc = acc * corr + tl.sum(p[:, None] * v, axis=0)

        # stores the new running max for the next block
        m = m_new   

    o = acc / l
    tl.store(o_ptr + offs_d, o, mask=mask_d)

# python wrapper for flash_decode_kernel
def flash_decode(q, k, v, scale):
    seq_len, head_dim = k.shape
    o = torch.empty(head_dim, device=q.device, dtype=torch.float32)
    BLOCK_D = triton.next_power_of_2(head_dim)
    flash_decode_kernel[(1,)](
        q, k, v, o, seq_len, scale,
        k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        BLOCK_N=64, BLOCK_D=BLOCK_D, HEAD_DIM=head_dim,
    )
    return o


def test_flash_decode():
    torch.manual_seed(0)
    seq_len, head_dim = 200, 128       # 200 is not a multiple of BLOCK_N=64 on purpose
    q = torch.randn(head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float32)
    v = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float32)
    scale = 1.0 / (head_dim ** 0.5)

    scores = (k @ q) * scale                      # [seq_len]
    ref = torch.softmax(scores, dim=0) @ v        # [head_dim]
    got = flash_decode(q, k, v, scale)
    err = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, atol=1e-3), f"mismatch, max abs err {err}"
    print(f"K1 OK — flash-decode attention matches torch (max abs err {err:.2e}).")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Triton needs a CUDA GPU — run this on Modal: modal run bench/modal_kernel.py")
    else:
        test_softmax()
        test_flash_decode()

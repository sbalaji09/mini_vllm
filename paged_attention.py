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

# same softmax as flash_decode_kernel but the K/V per token are directly read out of the paged pool via a block table
@triton.jit
def paged_decode_kernel(q_ptr, k_pool_ptr, v_pool_ptr, o_ptr, bt_ptr,
                        seq_len, scale, kv_head,
                        stride_blk, stride_h, stride_t, stride_d,   # pool strides
                        BLOCK_SIZE: tl.constexpr, BLOCK_D: tl.constexpr, HEAD_DIM: tl.constexpr):
    # creates the offsets and the masks
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    # loads the query vector
    q = tl.load(q_ptr + offs_d, mask=mask_d, other=0.0)

    # initializes online softmax state: running max, running denominator, and weighted value accumulator
    m = -float("inf")
    l = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # computes how many logical blocks the sequence spans and loops through them
    n_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    for j in range(0, n_blocks):
        phys = tl.load(bt_ptr + j)                 # physical block id for logical block j
        offs_t = tl.arange(0, BLOCK_SIZE)
        tok = j * BLOCK_SIZE + offs_t              # global token indices in this block
        mask_t = tok < seq_len

        # computes the base pointer for this physical block and KV head in the key pool
        base_k = k_pool_ptr + phys * stride_blk + kv_head * stride_h
        # loads a tile of keys from paged storage
        k = tl.load(base_k + offs_t[:, None] * stride_t + offs_d[None, :] * stride_d,
                    mask=mask_t[:, None] & mask_d[None, :], other=0.0)
        
        # same computation but in the value pool
        base_v = v_pool_ptr + phys * stride_blk + kv_head * stride_h
        v = tl.load(base_v + offs_t[:, None] * stride_t + offs_d[None, :] * stride_d,
                    mask=mask_t[:, None] & mask_d[None, :], other=0.0)
        
        # online softmax same as earlier
        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(mask_t, scores, -float("inf"))
        m_new = tl.maximum(m, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        corr = tl.exp(m - m_new)
        l = l * corr + tl.sum(p, axis=0)
        acc = acc * corr + tl.sum(p[:, None] * v, axis=0)
        m = m_new

    o = acc / l
    tl.store(o_ptr + offs_d, o, mask=mask_d)

# Python wrapper around the Triton paged decode kernel
def paged_decode(q, k_pool, v_pool, block_table, seq_len, kv_head, scale):
    head_dim = q.shape[0]
    BLK = k_pool.shape[2]
    o = torch.empty(head_dim, device=q.device, dtype=torch.float32)
    BLOCK_D = triton.next_power_of_2(head_dim)
    paged_decode_kernel[(1,)](
        q, k_pool, v_pool, o, block_table, seq_len, scale, kv_head,
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        BLOCK_SIZE=BLK, BLOCK_D=BLOCK_D, HEAD_DIM=head_dim,
    )
    return o


def test_paged_decode():
    torch.manual_seed(0)
    seq_len, head_dim, BLK = 200, 128, 16
    n_seq_blocks = (seq_len + BLK - 1) // BLK
    num_blocks = n_seq_blocks + 5            # pool slack
    scale = 1.0 / (head_dim ** 0.5)

    q = torch.randn(head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float32)
    v = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float32)

    # scatter the sequence into SHUFFLED physical blocks (proves non-contiguity)
    k_pool = torch.zeros(num_blocks, 1, BLK, head_dim, device="cuda", dtype=torch.float32)
    v_pool = torch.zeros_like(k_pool)
    phys = torch.randperm(num_blocks, device="cuda")[:n_seq_blocks].to(torch.int32)
    for t in range(seq_len):
        b, off = phys[t // BLK].item(), t % BLK
        k_pool[b, 0, off, :] = k[t]
        v_pool[b, 0, off, :] = v[t]

    scores = (k @ q) * scale
    ref = torch.softmax(scores, dim=0) @ v
    got = paged_decode(q, k_pool, v_pool, phys, seq_len, kv_head=0, scale=scale)
    err = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, atol=1e-3), f"mismatch, max abs err {err}"
    print(f"K2 OK — paged-decode attention matches torch over shuffled blocks (max abs err {err:.2e}).")

# defines a batched paged-attention decode kernel
@triton.jit
def paged_decode_batched_kernel(q_ptr, k_pool_ptr, v_pool_ptr, o_ptr, bt_ptr, seqlen_ptr,
                                scale, group,
                                stride_qr, stride_qh, stride_qd,    # Q [R, n_q_heads, head_dim]
                                stride_or, stride_oh, stride_od,    # O [R, n_q_heads, head_dim]
                                stride_btr,                         # block_tables [R, max_blocks] row stride
                                stride_blk, stride_h, stride_t, stride_d,   # pool strides
                                BLOCK_SIZE: tl.constexpr, BLOCK_D: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid_r = tl.program_id(0)        # which sequence
    pid_h = tl.program_id(1)        # which query head

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    # maps the query head to its shared KV head
    kv_head = pid_h // group
    q = tl.load(q_ptr + pid_r*stride_qr + pid_h*stride_qh + offs_d*stride_qd,
                mask=mask_d, other=0.0) # loads the query vector
    seq_len = tl.load(seqlen_ptr + pid_r) # loads the sequence's length
    bt_row = bt_ptr + pid_r * stride_btr # gets the start of this sequence's block-table row

    # init softmax state
    m = -float("inf")
    l = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # loops through each logical block of this sequence
    n_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    for j in range(0, n_blocks):
        phys = tl.load(bt_row + j)
        offs_t = tl.arange(0, BLOCK_SIZE)
        tok = j * BLOCK_SIZE + offs_t
        mask_t = tok < seq_len

        # computes the base pointer for this physical block and KV head in the K pool + loads the key block
        base_k = k_pool_ptr + phys * stride_blk + kv_head * stride_h
        k = tl.load(base_k + offs_t[:, None] * stride_t + offs_d[None, :] * stride_d,
                    mask=mask_t[:, None] & mask_d[None, :], other=0.0)
        
        # same, but for the V pool
        base_v = v_pool_ptr + phys * stride_blk + kv_head * stride_h
        v = tl.load(base_v + offs_t[:, None] * stride_t + offs_d[None, :] * stride_d,
                    mask=mask_t[:, None] & mask_d[None, :], other=0.0)

        # computes attention scores for this query against the current key block
        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(mask_t, scores, -float("inf"))
        m_new = tl.maximum(m, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        corr = tl.exp(m - m_new)
        l = l * corr + tl.sum(p, axis=0)
        acc = acc * corr + tl.sum(p[:, None] * v, axis=0)
        m = m_new

    o = acc / l
    
    # normalizes the accumulated value sum to produce the final attention output
    tl.store(o_ptr + pid_r*stride_or + pid_h*stride_oh + offs_d*stride_od, o, mask=mask_d)

# Python wrapper for this custom kernel
def paged_decode_batched(q, k_pool, v_pool, block_tables, seq_lens, n_kv_heads, scale):
    R, n_q_heads, head_dim = q.shape
    group = n_q_heads // n_kv_heads
    BLK = k_pool.shape[2]
    o = torch.empty_like(q)
    BLOCK_D = triton.next_power_of_2(head_dim)
    paged_decode_batched_kernel[(R, n_q_heads)](
        q, k_pool, v_pool, o, block_tables, seq_lens, scale, group,
        q.stride(0), q.stride(1), q.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        block_tables.stride(0),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        BLOCK_SIZE=BLK, BLOCK_D=BLOCK_D, HEAD_DIM=head_dim,
    )
    return o


# ---- glue: build a paged batch, and the gather+attend baseline (what the kernel replaces) ----

def _build_batch(R, n_kv_heads, head_dim, BLK=16, max_len=512, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    seq_lens = torch.randint(20, max_len + 1, (R,), device="cuda", generator=g, dtype=torch.int32)
    max_blocks = (max_len + BLK - 1) // BLK
    total = int(((seq_lens + BLK - 1) // BLK).sum())
    num_blocks = total + 8
    k_pool = torch.randn(num_blocks, n_kv_heads, BLK, head_dim, device="cuda", generator=g)
    v_pool = torch.randn(num_blocks, n_kv_heads, BLK, head_dim, device="cuda", generator=g)
    # assign each sequence a shuffled, non-overlapping set of physical blocks
    perm = torch.randperm(num_blocks, device="cuda", generator=g)
    block_tables = torch.zeros(R, max_blocks, device="cuda", dtype=torch.int32)
    cur = 0
    for r in range(R):
        nb = int((seq_lens[r] + BLK - 1) // BLK)
        block_tables[r, :nb] = perm[cur:cur + nb].to(torch.int32)
        cur += nb
    return k_pool, v_pool, block_tables, seq_lens


def gather_attend(q, k_pool, v_pool, block_tables, seq_lens, n_kv_heads, scale, max_len):
    # the Phase-2B approach: gather scattered blocks -> contiguous, then attend.
    # This is what the kernel makes unnecessary.
    R, n_q_heads, head_dim = q.shape
    group = n_q_heads // n_kv_heads
    BLK = k_pool.shape[2]
    Kc = torch.zeros(R, n_kv_heads, max_len, head_dim, device=q.device, dtype=q.dtype)
    Vc = torch.zeros_like(Kc)
    for r in range(R):
        L = int(seq_lens[r]); nb = (L + BLK - 1) // BLK
        ids = block_tables[r, :nb]
        Kc[r, :, :L, :] = k_pool[ids].permute(1, 0, 2, 3).reshape(n_kv_heads, -1, head_dim)[:, :L, :]
        Vc[r, :, :L, :] = v_pool[ids].permute(1, 0, 2, 3).reshape(n_kv_heads, -1, head_dim)[:, :L, :]
    Kc = Kc.repeat_interleave(group, dim=1)            # GQA expand -> [R, n_q_heads, max_len, D]
    Vc = Vc.repeat_interleave(group, dim=1)
    scores = torch.einsum("rhd,rhld->rhl", q, Kc) * scale
    valid = torch.arange(max_len, device=q.device)[None, :] < seq_lens[:, None]
    scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("rhl,rhld->rhd", p, Vc)


def test_paged_batched():
    torch.manual_seed(0)
    R, n_q_heads, n_kv_heads, head_dim, max_len = 8, 12, 2, 128, 300
    scale = 1.0 / (head_dim ** 0.5)
    k_pool, v_pool, block_tables, seq_lens = _build_batch(R, n_kv_heads, head_dim, max_len=max_len)
    q = torch.randn(R, n_q_heads, head_dim, device="cuda", dtype=torch.float32)
    ref = gather_attend(q, k_pool, v_pool, block_tables, seq_lens, n_kv_heads, scale, max_len)
    got = paged_decode_batched(q, k_pool, v_pool, block_tables, seq_lens, n_kv_heads, scale)
    err = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, atol=1e-3), f"mismatch, max abs err {err}"
    print(f"K3 OK — batched+GQA paged attention matches gather reference (max abs err {err:.2e}).")


def bench_kernel_vs_gather():
    import time
    R, n_q_heads, n_kv_heads, head_dim, max_len = 64, 12, 2, 128, 512
    scale = 1.0 / (head_dim ** 0.5)
    k_pool, v_pool, block_tables, seq_lens = _build_batch(R, n_kv_heads, head_dim, max_len=max_len, seed=1)
    q = torch.randn(R, n_q_heads, head_dim, device="cuda", dtype=torch.float32)

    def run_kernel():
        return paged_decode_batched(q, k_pool, v_pool, block_tables, seq_lens, n_kv_heads, scale)

    def run_gather():
        return gather_attend(q, k_pool, v_pool, block_tables, seq_lens, n_kv_heads, scale, max_len)

    for _ in range(5):           # warmup (compile + autotune)
        run_kernel(); run_gather()
    torch.cuda.synchronize()

    def timed(fn, iters=50):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3   # ms/step

    tk, tg = timed(run_kernel), timed(run_gather)
    print(f"K3 microbench (R={R}): kernel {tk:.3f} ms/step | gather+attend {tg:.3f} ms/step "
          f"| kernel {tg / tk:.2f}x faster (no gather copy)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Triton needs a CUDA GPU — run this on Modal: modal run bench/modal_kernel.py")
    else:
        test_softmax()
        test_flash_decode()
        test_paged_decode()
        test_paged_batched()
        bench_kernel_vs_gather()

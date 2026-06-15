"""
Phase 2 LEVEL A — concurrency experiment (accounting only, no model / no GPU).

Question: under a FIXED block budget, how many sequences fit concurrently under
PAGED allocation vs CONTIGUOUS pad-to-max? Drives the real BlockAllocator /
BlockTable from kv_cache.py, so it also validates them under load.

Honest framing (Phase 2 spec): this is the MEMORY / CONCURRENCY metric, NOT
throughput. Paging in pure PyTorch does not make generation faster.
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kv_cache import BlockAllocator, BlockTable, BLOCK_SIZE  # noqa: E402


def make_workload(n, long_frac=0.3, seed=0):
    # Documented realistic mix: most requests are short, a minority generate
    # long. High length variance is exactly what fragments a pad-to-max cache.
    rng = random.Random(seed)
    lengths = []
    for _ in range(n):
        if rng.random() < long_frac:
            lengths.append(rng.randint(300, 512))   # long generation
        else:
            lengths.append(rng.randint(20, 80))     # short query/answer
    return lengths


def fit_paged(lengths, n_blocks):
    # FIFO admission: admit in order until the next sequence can't get a block.
    alloc = BlockAllocator(n_blocks)
    tables = []
    for L in lengths:
        t = BlockTable(alloc)
        if all(t.append_token() for _ in range(L)):
            tables.append(t)
        else:
            t.free_all()          # roll back the partial sequence
            break
    used = sum(t.length for t in tables)
    reserved = sum(t.reserved_tokens() for t in tables)
    return len(tables), used, reserved


def fit_contiguous(lengths, n_blocks):
    # Pad-to-max: a rectangular cache must reserve L_max slots per sequence.
    # (We use the realized max length — the most CHARITABLE baseline; reserving
    # the generation cap would make paging look even better.)
    T = n_blocks * BLOCK_SIZE
    L_max = max(lengths)
    fit = min(len(lengths), T // L_max)
    used = sum(lengths[:fit])
    reserved = fit * L_max
    return fit, used, reserved


def util(used, reserved):
    return f"{100 * used / reserved:.1f}%" if reserved else "n/a"


def run(lengths, n_blocks, label):
    p_fit, p_used, p_res = fit_paged(lengths, n_blocks)
    c_fit, c_used, c_res = fit_contiguous(lengths, n_blocks)
    print(f"[{label}] budget {n_blocks} blocks ({n_blocks*BLOCK_SIZE} slots) | "
          f"lengths {min(lengths)}..{max(lengths)} mean {sum(lengths)/len(lengths):.0f}")
    print(f"    contiguous (pad-to-max): {c_fit:>4} concurrent | utilization {util(c_used, c_res)}")
    print(f"    paged (16-tok blocks)  : {p_fit:>4} concurrent | utilization {util(p_used, p_res)}")
    print(f"    --> concurrency ratio (paged / contiguous) = {p_fit / c_fit:.2f}x\n")


if __name__ == "__main__":
    N_BLOCKS = 512   # fixed KV budget = 512 blocks * 16 = 8192 token-slots
    base = make_workload(1000, long_frac=0.3)
    run(base, N_BLOCKS, "realistic mix (30% long)")

    # sensitivity: the ratio tracks length variance (Q2). Show it, don't hide it.
    run(make_workload(1000, long_frac=0.5), N_BLOCKS, "higher variance (50% long)")
    run(make_workload(1000, long_frac=0.1), N_BLOCKS, "lower variance (10% long)")

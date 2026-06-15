"""
paged KV cache, LEVEL A (allocator + accounting).
"""

import torch

BLOCK_SIZE = 16   # tokens per block

# fixed pool of physical blocks and a free list
class BlockAllocator:
    # initializes a free list holding every physical block id
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_list = []
        for i in range(num_blocks):
            self.free_list.append(i)

    # returns how many physical blocks are currently free
    def num_free(self) -> int:
        return len(self.free_list)

    # removes and returns one free physical block id
    # returns None if none are free
    def allocate(self) -> int:
        if len(self.free_list) == 0:
            return None
        
        return self.free_list.pop(0)
    
    # adds the given block_ids to the free pool
    def free(self, block_ids):
        # TODO (yours): return a list of physical block ids to the free pool.
        for id in block_ids:
            self.free_list.append(id)

# represents an ordered list of physical block ids where it maps a sequence's logical to physical map
class BlockTable:
    # creates the allocator, block_ids, and length variables
    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.block_ids = []
        self.length = 0 

    # account for one new token landing at the end by allocating a new spot if necessary
    def append_token(self) -> bool:
        if self.length % BLOCK_SIZE == 0:
            block = self.allocator.allocate()
            if block == None:
                return False
            self.block_ids.append(block)
        
        self.length += 1
        return True

    # translates a logical token position to its physical storage in the cache
    def physical(self, logical_pos: int):
        block_id = self.block_ids[logical_pos // BLOCK_SIZE]
        offset = logical_pos % BLOCK_SIZE
        return (block_id, offset)

    # returns the capacity reserved for this sequence
    def reserved_tokens(self) -> int:
        return len(self.block_ids) * BLOCK_SIZE

    # frees all the block_ids in the BlockTable and resets the length and block_ids list
    def free_all(self):
        self.allocator.free(self.block_ids)
        self.block_ids = []
        self.length = 0

# physical KV storage as fixed blocks
class PagedKVCache:
    def __init__(self, num_blocks, n_layers, n_kv_heads, head_dim,
                 dtype=torch.float32, device="cpu"):
        self.allocator = BlockAllocator(num_blocks)
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        # per-layer physical pools: [num_blocks, n_kv_heads, BLOCK_SIZE, head_dim]
        self.k_pool = [torch.zeros(num_blocks, n_kv_heads, BLOCK_SIZE, head_dim,
                                   dtype=dtype, device=device)
                       for _ in range(n_layers)]
        self.v_pool = [torch.zeros(num_blocks, n_kv_heads, BLOCK_SIZE, head_dim,
                                   dtype=dtype, device=device)
                       for _ in range(n_layers)]

    def new_table(self) -> "BlockTable":
        return BlockTable(self.allocator)

    # writes one new token's KV data into paged storage
    def scatter_token(self, table, k, v) -> bool:
        # reserves space for one more token
        if not table.append_token():
            return False
        
        # it finds the offset for where the token lives in physical storage
        p = table.length - 1
        block_id, offset = table.physical(p)

        # for every layer, it then writes the token's key and value into the physical KV pools
        for l in range(self.n_layers):
            self.k_pool[l][block_id, :, offset, :] = k[l]
            self.v_pool[l][block_id, :, offset, :] = v[l]
        return True

    # rebuilds contiguous left-padded KV tensors from paged storage
    def gather(self, tables, max_len):
        R = len(tables)
        k_layers = []
        v_layers = []

        # for each layer, it creates output tensors shaped: [R, n_kv_heads, max_len, head_dim]
        for l in range(self.n_layers):
            # creates the tensors with all zeros of the shape given
            out_k = torch.zeros(
                R,
                self.n_kv_heads,
                max_len,
                self.head_dim,
                dtype=self.k_pool[l].dtype,
                device=self.k_pool[l].device
            )
            out_v = torch.zeros(
                R,
                self.n_kv_heads,
                max_len,
                self.head_dim,
                dtype=self.v_pool[l].dtype,
                device=self.v_pool[l].device,
            )

            # enumerate through all the tables and s being the sequence
            for s, table in enumerate(tables):
                ids = table.block_ids
                if table.length == 0:
                    continue
                
                # fetches the physical blocks using the sequence's block_ids
                k_blocks = self.k_pool[l][ids]
                v_blocks = self.v_pool[l][ids]

                # reshapes the blocks into a normal sequence layout
                k_seq = k_blocks.permute(1, 0, 2, 3).reshape(
                    self.n_kv_heads,
                    -1,
                    self.head_dim
                )
                v_seq = v_blocks.permute(1, 0, 2, 3).reshape(
                    self.n_kv_heads,
                    -1,
                    self.head_dim,
                )

                # this trims away unused slack from the final block
                k_seq = k_seq[:, :table.length, :]
                v_seq = v_seq[:, :table.length, :]
                
                # this writes the real tokens into the right side of the output tensor
                # creates left padding for shorter sequences so all sequences align at their most recent token
                start = max_len - table.length
                out_k[s, :, start:, :] = k_seq
                out_v[s, :, start:, :] = v_seq

            k_layers.append(out_k)
            v_layers.append(out_v)
    
        return k_layers, v_layers


if __name__ == "__main__":
    # --- A1 unit test (glue): exercises your allocator + block table ---
    # Fill the TODOs above, then run `python kv_cache.py`. All asserts must pass.
    alloc = BlockAllocator(num_blocks=4)
    assert alloc.num_free() == 4

    seq = BlockTable(alloc)
    # append 33 tokens -> ceil(33/16) = 3 blocks
    for _ in range(33):
        assert seq.append_token() is True
    assert seq.length == 33
    assert len(seq.block_ids) == 3, f"expected 3 blocks, got {len(seq.block_ids)}"
    assert alloc.num_free() == 1, f"expected 1 free, got {alloc.num_free()}"
    assert seq.reserved_tokens() == 48   # 3 * 16

    # logical->physical mapping
    assert seq.physical(0) == (seq.block_ids[0], 0)
    assert seq.physical(16) == (seq.block_ids[1], 0)
    assert seq.physical(32) == (seq.block_ids[2], 0)
    assert seq.physical(17) == (seq.block_ids[1], 1)

    # budget exhaustion: only 1 block left, so a 2nd sequence past 16 tokens fails
    seq2 = BlockTable(alloc)
    assert seq2.append_token() is True          # grabs the last block
    assert alloc.num_free() == 0
    for _ in range(15):                          # fills block to 16 tokens
        assert seq2.append_token() is True
    assert seq2.append_token() is False          # 17th token needs a block; none free

    # retire returns blocks to the pool
    seq.free_all()
    assert alloc.num_free() == 3
    assert seq.length == 0 and seq.block_ids == []

    print("A1 OK — allocator, block table, mapping, and free-pool all correct.")

    # --- B1 round-trip test (glue): scatter known K/V, gather, assert equal ---
    torch.manual_seed(0)
    NL, H, D = 2, 2, 4
    pool = PagedKVCache(num_blocks=8, n_layers=NL, n_kv_heads=H, head_dim=D)
    lengths = [20, 5]                       # mixed lengths -> exercises left-pad
    ptables = [pool.new_table() for _ in lengths]
    truth = []                             # truth[s] = list of (k, v) per token, each [NL, H, D]
    for s, Ln in enumerate(lengths):
        toks = []
        for _ in range(Ln):
            k = torch.randn(NL, H, D)
            v = torch.randn(NL, H, D)
            assert pool.scatter_token(ptables[s], k, v) is True
            toks.append((k, v))
        truth.append(toks)

    max_len = max(lengths)
    k_layers, v_layers = pool.gather(ptables, max_len)
    for s, Ln in enumerate(lengths):
        for l in range(NL):
            for t in range(Ln):
                col = max_len - Ln + t       # left-padded position of token t
                assert torch.allclose(k_layers[l][s, :, col, :], truth[s][t][0][l])
                assert torch.allclose(v_layers[l][s, :, col, :], truth[s][t][1][l])
    print("B1 OK — scatter + gather round-trip correct (incl. left-pad alignment).")

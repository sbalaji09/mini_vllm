"""
paged KV cache, LEVEL A (allocator + accounting).
"""

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


class BlockTable:
    """One sequence's logical->physical map: an ordered list of physical block
    ids. Mirrors an OS page table for a single address space."""

    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.block_ids = []     # physical blocks owned, in logical order
        self.length = 0         # tokens currently stored in this sequence

    def append_token(self) -> bool:
        # TODO (yours): account for ONE new token landing at logical position
        #   self.length.
        #   - A new physical block is needed exactly when the current length sits
        #     on a block boundary: self.length % BLOCK_SIZE == 0. (This also
        #     covers the very first token, length == 0 -> allocate block #1.)
        #     When needed: block = self.allocator.allocate(); append to block_ids.
        #     If allocation fails (pool empty), return False WITHOUT incrementing.
        #   - Otherwise the token fits in the current last block.
        #   - increment self.length, return True.
        raise NotImplementedError

    def physical(self, logical_pos: int):
        # TODO (yours): translate a logical token position to physical storage:
        #   block_id = self.block_ids[logical_pos // BLOCK_SIZE]
        #   offset   = logical_pos % BLOCK_SIZE
        #   return (block_id, offset)
        raise NotImplementedError

    def reserved_tokens(self) -> int:
        # capacity reserved for this sequence, including last-block slack.
        # (used vs reserved is the fragmentation measurement later.)
        return len(self.block_ids) * BLOCK_SIZE

    def free_all(self):
        # TODO (yours): return ALL of this sequence's blocks to the allocator and
        #   reset block_ids / length (called on retire).
        raise NotImplementedError


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

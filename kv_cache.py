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

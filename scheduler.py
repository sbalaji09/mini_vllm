"""
This file defines the scheduler, which will reuse the model and tokens from engine.py
for the forward pass
"""

import time, itertools
from dataclasses import dataclass, field
import torch
from engine import model, tok

# request dataclass with essential information about request including:
# id, prompt, max_new_tokens, KV cache, last_token, and metrics
@dataclass
class Request:
    id: int
    prompt: str
    max_new_tokens: int = 64
    past_key_values: object = None
    last_token: torch.Tensor = None
    cur_len: int = 0
    output_ids: list = field(default_factory=list)
    finished: bool = False
    t_arrival: float = 0.0
    t_first: float = None
    t_done: float = None

class ContinuousBatchingEngine:
    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size
        self.waiting = [] # FIFO queue of the requests that are waiting to run
        self.running = [] # list of the actively running processes
        self.completed = []
        self._ids = itertools.count()
    
    # submit function takes in a prompt and max_new_tokens, turns it into
    # a Request object and adds it to the waiting queue
    def submit(self, prompt: str, max_new_tokens: int = 64) -> Request:
        r = Request(id=next(self._ids), prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    t_arrival=time.perf_counter())
        self.waiting.append(r)
        return r
    
    
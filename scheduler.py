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
    
    # moves requests from the waiting queue into active execution
    def _admit(self):
        # keep admitting requests while ther eis room in the running set
        while len(self.running) < self.max_batch_size and self.waiting:
            # take the oldest waiting request
            r = self.waiting.pop(0)

            # format the user prompt as a chat prompt so Qwen can intake it
            # then tokenize it into model input IDs
            msgs = [{"role": "user", "content": r.prompt}]
            input_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            out = model(input_ids=input_ids, use_cache=True)

            # store the KV cache, the most likely output token, and record the requests length
            r.past_key_values = out.past_key_values
            r.last_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            r.cur_len = input_ids.shape[-1]

            # save the first generated token ID for the request and record when the first generated
            # token became available
            r.output_ids.append(r.last_token[0].item())
            r.t_first = time.perf_counter()

            # if the last token is eos, then mark it as completed
            # otherwise, append it to running since it still needs to process
            if r.last_token[0].item() == tok.eos_token_id:
                r.finished = True
                r.t_done = time.perf_counter()
                self.completed.append(r)
            else:
                self.running.append(r)

    # this defines one decoding step for all the currently running requests
    def _decode_step(self):
        # this is a fresh list for requests that are still not finished
        new_running = []

        # loops through every request that was active before this step
        for r in self.running:
            # only process not finished requests
            if r.finished == False:
                # run the model on the most recent token for this request
                out = model(input_ids=r.last_token, past_key_values=r.past_key_values, use_cache=True)
                
                # takes the model's prediction for the next token and picks the highest-probability token
                # then updates the output_ids and KV cache based on the new tokens
                last_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                r.output_ids.append(last_token[0].item())
                r.past_key_values = out.past_key_values

                # saves the newly generated token and increments current sequence length
                r.last_token = last_token
                r.cur_len += 1

                # checks whether the request should stop: hit eos or we have generated the maximum allowed number of tokens
                if (r.last_token[0].item() == tok.eos_token_id) or (len(r.output_ids) >= r.max_new_tokens):
                    # mark the request as finished and set the metric times
                    r.finished = True
                    r.t_done = time.perf_counter()
                    self.completed.append(r)
                # if the request is not done yet, keep it active
                else:
                    new_running.append(r)

        # replace the old running list with only the requests that need more tokens
        self.running = new_running
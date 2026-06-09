"""
This file defines the scheduler, which will reuse the model and tokens from engine.py
for the forward pass
"""

import time, itertools
from dataclasses import dataclass, field
import torch
from transformers import DynamicCache
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
            # render template -> string, then tokenize. apply_chat_template with
            # return_tensors returns a BatchEncoding (dict-like) in this
            # transformers version, not a bare tensor, so go through tok(...).
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            input_ids = tok(text, return_tensors="pt")["input_ids"]
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
            else:
                self.running.append(r)

    # this defines one decoding step for all the currently running requests
    def _decode_step(self):
        # this is a fresh list for requests that are still not finished
        new_running = []

        # loops through every request that was active before this step
        for r in self.running:
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
            if (last_token[0].item() == tok.eos_token_id) or (len(r.output_ids) >= r.max_new_tokens):
                # mark the request as finished and set the metric times
                r.finished = True
            # if the request is not done yet, keep it active
            else:
                new_running.append(r)

        # replace the old running list with only the requests that need more tokens
        self.running = new_running
    
    # helper that pads one request's KV cache to a desired sequence length.
    # transformers 5.x: a DynamicCache exposes per-layer tensors via
    # cache.layers[i].keys / .values  (shape [B, n_kv_heads, seq, head_dim]).
    def _pad_one_cache(self, cache, target_len: int):
        padded = []

        # pulls out the key and value tensors from each layer of the KV cache
        for layer in cache.layers:
            key, value = layer.keys, layer.values
            pad_len = target_len - key.shape[-2]

            # if there is something to pad, then pad it by adding zeros and concatenating
            if pad_len > 0:
                kp = key.new_zeros(*key.shape[:-2], pad_len, key.shape[-1])
                vp = value.new_zeros(*value.shape[:-2], pad_len, value.shape[-1])

                key = torch.cat([kp, key], dim=-2)
                value = torch.cat([vp, value], dim=-2)

            padded.append((key, value))

        return padded

    # turns many per-request KV caches into one batched DynamicCache
    def _batch_caches(self, requests):
        max_len = max(r.cur_len for r in requests)

        # pads each request based on the longest request
        padded = [
            self._pad_one_cache(r.past_key_values, max_len)
            for r in requests
        ]

        # build a fresh DynamicCache: for each layer, stack all requests' key
        # and value tensors along the batch dim and register them in order.
        cache = DynamicCache()
        for layer_idx in range(len(padded[0])):
            keys = torch.cat([p[layer_idx][0] for p in padded], dim=0)
            values = torch.cat([p[layer_idx][1] for p in padded], dim=0)
            cache.update(keys, values, layer_idx)

        return cache, max_len

    # pulls one request's KV cache back out of the batched output cache,
    # returning a fresh single-sequence DynamicCache (pad dropped via -keep_len:)
    def _extract_one_cache(self, cache, batch_idx: int, keep_len: int):
        one = DynamicCache()

        # loops through every layer in the model output cache, slicing out this
        # request's row and its real (un-padded) tail of length keep_len
        for layer_idx, layer in enumerate(cache.layers):
            key_i = layer.keys[batch_idx:batch_idx + 1, :, -keep_len:, :].contiguous()
            value_i = layer.values[batch_idx:batch_idx + 1, :, -keep_len:, :].contiguous()
            one.update(key_i, value_i, layer_idx)

        return one
    
    # defines one batched decode step for all requests
    def _decode_step_batched(self):
        if not self.running:
            return
        
        # stores the input ids in a local variable
        batch = self.running
        input_ids = torch.cat([r.last_token for r in batch], dim=0)

        # _batch_caches now returns a ready DynamicCache (5.x API)
        past, max_cache_len = self._batch_caches(batch)

        # creates a batched attention mask with the shape [B, max_cache_len + 1]
        attention_mask = torch.zeros(
            (len(batch), max_cache_len + 1),
            dtype=torch.long,
            device=input_ids.device,
        )

        # creates a tensor for the position ID of each request's new tokens
        position_ids = torch.empty(
            (len(batch), 1),
            dtype=torch.long,
            device=input_ids.device,
        )

        # loops through each request: marks the real cached token and the new token as visible
        for i, r in enumerate(batch):
            attention_mask[i, max_cache_len - r.cur_len:] = 1
            position_ids[i, 0] = r.cur_len
        
        # runs one model forward pass for the entire active batch
        out=model(
            input_ids=input_ids,
            past_key_values=past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )

        # gets the model's next-token prediction for each request
        next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # loops through each request to scatter the batched outputs back into individual request state
        for i, r in enumerate(batch):
            # adds the new generated token to the request and updates values accordingly
            token_id = next_tokens[i].item()

            r.output_ids.append(token_id)
            r.last_token = next_tokens[i:i+1]
            r.cur_len += 1
            r.past_key_values = self._extract_one_cache(out.past_key_values, i, r.cur_len)

            if token_id == tok.eos_token_id or len(r.output_ids) >= r.max_new_tokens:
                r.finished = True

    # this helper removes finished requests from the active running list
    def _retire(self):
        # empty list for requests that aren't done yet
        still_running = []

        for r in self.running:
            # if the request is finished, then add it to the completed list and set its completed time
            if r.finished:
                r.t_done = time.perf_counter()
                self.completed.append(r)
            # otherwise, add it to the still_running list
            else:
                still_running.append(r)
        self.running = still_running
    
    # defines one scheduler iteration
    def step(self):
        # moves new requests into running. does the prefill pass for newly admitted 
        # requests and moves finished requets out of running
        self._admit()
        self._decode_step_batched()
        self._retire()

    # defines the full scheduler loop
    def run(self):
        # runs one scheduler iteration: admit new work, decode running work, and retire finished work
        while self.waiting or self.running:
            self.step()
        return self.completed

if __name__ == "__main__":
    engine = ContinuousBatchingEngine(max_batch_size=2)

    requests = [
        engine.submit("Explain KV cache in one sentence.", max_new_tokens=32),
        engine.submit("What is continuous batching?", max_new_tokens=32),
        engine.submit("Name three benefits of batching.", max_new_tokens=32),
        engine.submit("Explain token decoding briefly.", max_new_tokens=32),
    ]

    completed = engine.run()

    for r in completed:
        text = tok.decode(r.output_ids, skip_special_tokens=True)
        print(f"\nRequest {r.id}")
        print(text)
        print({
            "tokens": len(r.output_ids),
            "ttft_s": r.t_first - r.t_arrival if r.t_first else None,
            "latency_s": r.t_done - r.t_arrival if r.t_done else None,
        })
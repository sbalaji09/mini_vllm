"""
This file defines the scheduler, which will reuse the model and tokens from engine.py
for the forward pass
"""

import time, itertools
from dataclasses import dataclass, field
import torch
from transformers import DynamicCache
from engine import model, tok, DEVICE

# request dataclass with essential information about request including:
# id, prompt, max_new_tokens, KV cache, last_token, and metrics
@dataclass
class Request:
    id: int
    prompt: str
    max_new_tokens: int = 64
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
        self.cache = None
        # timing instrumentation (seconds): where does wall-clock actually go?
        self.t_prefill = 0.0      # per-request prefill forwards in _admit
        self.t_decode_fwd = 0.0   # the batched decode model() forward
        self.t_cache_mgmt = 0.0   # pad/cat/extract churn around the forward

    def _sync(self):
        # GPU kernels are async, so a bare timer around model() would measure
        # only kernel-launch time. Sync so timers reflect real execution.
        # No-op on CPU.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # submit function takes in a prompt and max_new_tokens, turns it into
    # a Request object and adds it to the waiting queue
    def submit(self, prompt: str, max_new_tokens: int = 64) -> Request:
        r = Request(id=next(self._ids), prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    t_arrival=time.perf_counter())
        self.waiting.append(r)
        return r
    
    # moves requests from the waiting queue into active execution
    @torch.no_grad()
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
            input_ids = tok(text, return_tensors="pt")["input_ids"].to(DEVICE)
            self._sync(); _t = time.perf_counter()
            out = model(input_ids=input_ids, use_cache=True)
            self._sync(); self.t_prefill += time.perf_counter() - _t

            # store the KV cache, the most likely output token, and record the requests length
            new_cache = out.past_key_values
            r.last_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            r.cur_len = input_ids.shape[-1]

            # save the first generated token ID for the request and record when the first generated
            # token became available
            r.output_ids.append(r.last_token[0].item())
            r.t_first = time.perf_counter()

            # if the first token is already eos, the request finished during
            # prefill: send it straight to completed (with t_done) so it isn't
            # silently dropped. otherwise keep it running.
            if r.last_token[0].item() == tok.eos_token_id:
                r.finished = True
                r.t_done = time.perf_counter()
                self.completed.append(r)
            
            self._sync()
            _t = time.perf_counter()
            if self.cache is None:
                self.cache = new_cache
                self.running = [r]
            else:
                L = self.cache.layers[0].keys.shape[-2]
                p = new_cache.layers[0].keys.shape[-2]
                Lp = max(L, p)

                old_layers = self._pad_one_cache(self.cache, Lp)
                new_layers = self._pad_one_cache(new_cache, Lp)

                merged = DynamicCache()
                
                for layer_idx in range(len(old_layers)):
                    old_k, old_v = old_layers[layer_idx]
                    new_k, new_v = new_layers[layer_idx]

                    keys = torch.cat([old_k, new_k], dim=0)
                    values = torch.cat([old_v, new_v], dim=0)

                    merged.update(keys, values, layer_idx)
                
                self.cache = merged
            
            self.running.append(r)
            
            self._sync()
            self.t_cache_mgmt += time.perf_counter() - _t


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
    
    @torch.no_grad()
    def _decode_step(self):
        if not self.running:
            return

        R = len(self.running)
        L = max(r.cur_len for r in self.running)
        input_ids = torch.cat([r.last_token for r in self.running], dim=0)

        self._sync()
        _t = time.perf_counter()

        attention_mask = torch.zeros(
            (R, L + 1),
            dtype=torch.long,
            device=input_ids.device,
        )

        position_ids = torch.empty(
            (R, 1),
            dtype=torch.long,
            device=input_ids.device,
        )

        for i, r in enumerate(self.running):
            attention_mask[i, -(r.cur_len + 1):] = 1
            position_ids[i, 0] = r.cur_len

        self._sync()
        self.t_cache_mgmt += time.perf_counter() - _t

        self._sync()
        _t = time.perf_counter()

        out = model(
            input_ids=input_ids,
            past_key_values=self.cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )

        next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        self._sync()
        self.t_decode_fwd += time.perf_counter() - _t

        self.cache = out.past_key_values

        self._sync()
        _t = time.perf_counter()

        eos_hits = next_tokens.squeeze(1) == tok.eos_token_id
        ids = next_tokens.squeeze(1).tolist()
        eos = eos_hits.tolist()

        for i, r in enumerate(self.running):
            r.output_ids.append(ids[i])
            r.last_token = next_tokens[i:i + 1]
            r.cur_len += 1

            if eos[i] or len(r.output_ids) >= r.max_new_tokens:
                r.finished = True

        self._sync()
        self.t_cache_mgmt += time.perf_counter() - _t

    # defines one batched decode step for all requests
    @torch.no_grad()
    def _decode_step_batched(self):
        if not self.running:
            return
        
        # stores the input ids in a local variable
        batch = self.running
        input_ids = torch.cat([r.last_token for r in batch], dim=0)

        # --- CACHE MGMT: build the batched (left-padded) cache + masks (timed) ---
        self._sync(); _t = time.perf_counter()
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
        self._sync(); self.t_cache_mgmt += time.perf_counter() - _t

        # --- DECODE FORWARD: one model() pass for the whole active batch (timed) ---
        self._sync(); _t = time.perf_counter()
        out=model(
            input_ids=input_ids,
            past_key_values=past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        # gets the model's next-token prediction for each request
        next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        self._sync(); self.t_decode_fwd += time.perf_counter() - _t

        # --- CACHE MGMT: scatter outputs + extract each request's cache (timed) ---
        self._sync(); _t = time.perf_counter()
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
        self._sync(); self.t_cache_mgmt += time.perf_counter() - _t

    @torch.no_grad()
    def _retire(self):
        if not self.running:
            return

        keep = [i for i, r in enumerate(self.running) if not r.finished]

        for r in (x for x in self.running if x.finished):
            r.t_done = time.perf_counter()
            self.completed.append(r)

        if len(keep) == len(self.running):
            return

        self._sync()
        _t = time.perf_counter()

        if not keep:
            self.cache = None
            self.running = []
            self._sync()
            self.t_cache_mgmt += time.perf_counter() - _t
            return

        new_max = max(self.running[i].cur_len for i in keep)

        new_cache = DynamicCache()

        for layer_idx, layer in enumerate(self.cache.layers):
            keys = layer.keys[keep, :, -new_max:, :].contiguous()
            values = layer.values[keep, :, -new_max:, :].contiguous()
            new_cache.update(keys, values, layer_idx)

        self.cache = new_cache
        self.running = [self.running[i] for i in keep]

        self._sync()
        self.t_cache_mgmt += time.perf_counter() - _t
    
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
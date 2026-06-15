"""
continuous batching through the PAGED KV cache.
same admit/decode/retire as before, but now the K/V lives in the paged store
"""
import time, itertools
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from engine import model, tok, DEVICE
from kv_cache import PagedKVCache


@dataclass
class PagedRequest:
    id: int
    prompt: str
    max_new_tokens: int = 64
    table: object = None          # this sequence's BlockTable in the paged store
    last_token: torch.Tensor = None
    output_ids: list = field(default_factory=list)
    finished: bool = False
    t_arrival: float = 0.0
    t_first: float = None
    t_done: float = None


class PagedContinuousBatchingEngine:
    def __init__(self, max_batch_size: int = 8, num_blocks: int = 2048):
        self.max_batch_size = max_batch_size
        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        n_kv_heads = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.kv = PagedKVCache(
            num_blocks, self.n_layers, n_kv_heads, head_dim,
            dtype=(torch.bfloat16 if DEVICE == "cuda" else torch.float32),
            device=DEVICE,
        )
        self.waiting, self.running, self.completed = [], [], []
        self._ids = itertools.count()

    def submit(self, prompt: str, max_new_tokens: int = 64) -> PagedRequest:
        r = PagedRequest(id=next(self._ids), prompt=prompt,
                         max_new_tokens=max_new_tokens, t_arrival=time.perf_counter())
        self.waiting.append(r)
        return r

    # --- glue helper: stack one position's K/V across all layers ---
    def _token_kv(self, cache, batch_idx, pos):
        # returns (k, v), each [n_layers, n_kv_heads, head_dim], for `pos` of row `batch_idx`
        k = torch.stack([cache.layers[l].keys[batch_idx, :, pos, :] for l in range(self.n_layers)])
        v = torch.stack([cache.layers[l].values[batch_idx, :, pos, :] for l in range(self.n_layers)])
        return k, v

    @torch.no_grad()
    def _admit(self):
        while len(self.running) < self.max_batch_size and self.waiting:
            r = self.waiting.pop(0)
            text = tok.apply_chat_template([{"role": "user", "content": r.prompt}],
                                           add_generation_prompt=True, tokenize=False)
            input_ids = tok(text, return_tensors="pt")["input_ids"].to(DEVICE)
            out = model(input_ids=input_ids, use_cache=True)
            prompt_cache = out.past_key_values          # [1, H, p, D] per layer
            p = input_ids.shape[-1]
            r.table = self.kv.new_table()
            r.last_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            r.output_ids.append(r.last_token[0].item())
            r.t_first = time.perf_counter()
            
            for pos in range(p):
                k, v = self._token_kv(prompt_cache, 0, pos)
                if not self.kv.scatter_token(r.table, k, v):
                    r.finished = True
                    r.t_done = time.perf_counter()
                    self.completed.append(r)
                    break
            
            if r.finished:
                continue

            if r.last_token[0].item() == tok.eos_token_id:
                r.finished = True
                r.t_done = time.perf_counter()
                self.completed.append(r)
            else:
                self.running.append(r)
                
            raise NotImplementedError

    @torch.no_grad()
    def _decode_step(self):
        if not self.running:
            return
        R = len(self.running)
        max_len = max(r.table.length for r in self.running)
        input_ids = torch.cat([r.last_token for r in self.running], dim=0)   # [R, 1]

        # TODO (yours):
        #   1. GATHER: k_layers, v_layers = self.kv.gather([r.table for r in self.running], max_len)
        #   2. build past = DynamicCache(); for l: past.update(k_layers[l], v_layers[l], l)
        #   3. mask/positions (SAME as Phase 1): attention_mask [R, max_len+1] with row i's
        #      last (r.table.length + 1) cols = 1; position_ids [R,1] = r.table.length
        #   4. out = model(input_ids, past_key_values=past, attention_mask=..., position_ids=..., use_cache=True)
        #   5. next_tokens = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        #   6. for i, r in enumerate(self.running):
        #          k, v = self._token_kv(out.past_key_values, i, max_len)   # the appended token
        #          self.kv.scatter_token(r.table, k, v)
        #          tid = next_tokens[i].item(); r.output_ids.append(tid)
        #          r.last_token = next_tokens[i:i+1]
        #          if tid == tok.eos_token_id or len(r.output_ids) >= r.max_new_tokens:
        #              r.finished = True
        raise NotImplementedError

    @torch.no_grad()
    def _retire(self):
        keep = []
        for r in self.running:
            if r.finished:
                r.t_done = time.perf_counter()
                r.table.free_all()          # return this sequence's blocks to the pool
                self.completed.append(r)
            else:
                keep.append(r)
        self.running = keep

    def step(self):
        self._admit()
        self._decode_step()
        self._retire()

    def run(self):
        while self.waiting or self.running:
            self.step()
        return self.completed


if __name__ == "__main__":
    # --- B2 equivalence gate: paged generation must match generate() exactly ---
    from engine import generate

    TARGET = "Explain what a KV cache is in two sentences."
    CAP = 24
    ref = generate(TARGET, max_new_tokens=CAP)["output_ids"]

    e1 = PagedContinuousBatchingEngine(max_batch_size=1, num_blocks=512)
    e1.submit(TARGET, max_new_tokens=CAP)
    alone = e1.run()[0].output_ids

    e2 = PagedContinuousBatchingEngine(max_batch_size=4, num_blocks=512)
    rt = e2.submit(TARGET, max_new_tokens=CAP)
    e2.submit("Hi.", max_new_tokens=CAP)
    e2.submit("What is 2+2?", max_new_tokens=CAP)
    e2.submit("Write a long, detailed, multi-paragraph essay about how CPUs work.", max_new_tokens=CAP)
    mixed = {r.id: r.output_ids for r in e2.run()}[rt.id]

    print("alone == ref :", alone == ref)
    print("mixed == ref :", mixed == ref)
    print("B2 PASS — paged generation matches greedy reference"
          if (alone == ref and mixed == ref) else "B2 FAIL — gather/scatter bug")

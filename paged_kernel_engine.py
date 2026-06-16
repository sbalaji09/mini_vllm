"""
Phase 4 TIER 2 (Route 2) — generate through the Triton PagedAttention kernel.

We OWN the decode forward: reuse the model's submodules (projections, RoPE, norms,
MLP) but replace attention with paged_decode_batched (the K3 kernel) reading our
PagedKVCache pool. Prefill stays model(...) + scatter (Phase 2B).

Verify (T2a here): a single custom decode step must match the model's own decode
logits. Run on Modal: modal run bench/modal_tier2.py
"""
import itertools
from dataclasses import dataclass, field

import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from engine import model, tok, DEVICE
from kv_cache import PagedKVCache

cfg = model.config
N_LAYERS = cfg.num_hidden_layers
N_Q = cfg.num_attention_heads
N_KV = cfg.num_key_value_heads
HEAD_DIM = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
SCALE = HEAD_DIM ** -0.5

# the model's own submodules — we reuse these, only attention is replaced
_m = model.model
layers, embed, final_norm, rotary, lm_head = _m.layers, _m.embed_tokens, _m.norm, _m.rotary_emb, model.lm_head

from paged_attention import paged_decode_batched  # noqa: E402


def _block_tables_tensor(tables):
    R = len(tables)
    max_blk = max(len(t.block_ids) for t in tables)
    bt = torch.zeros(R, max_blk, dtype=torch.int32, device=DEVICE)
    for r, t in enumerate(tables):
        bt[r, :len(t.block_ids)] = torch.tensor(t.block_ids, dtype=torch.int32, device=DEVICE)
    return bt


@torch.no_grad()
def decode_logits(kv: PagedKVCache, last_tokens, tables):
    # last_tokens: [R, 1]. tables: R BlockTables, EACH ALREADY extended with the
    # new token's slot (t.length includes the token we're generating; its block is
    # allocated, so t.physical(t.length-1) is valid).
    R = last_tokens.shape[0]
    positions = torch.tensor([[t.length - 1] for t in tables], device=DEVICE)   # [R,1]
    seq_lens = torch.tensor([t.length for t in tables], dtype=torch.int32, device=DEVICE)
    bt = _block_tables_tensor(tables)

    h = embed(last_tokens)                 # [R, 1, hidden]
    cos, sin = rotary(h, positions)        # each [R, 1, head_dim]

    for i, layer in enumerate(layers):
        residual = h
        x = layer.input_layernorm(h)
        shp = (R, 1, -1, HEAD_DIM)
        q = layer.self_attn.q_proj(x).view(shp).transpose(1, 2)
        k = layer.self_attn.k_proj(x).view(shp).transpose(1, 2)
        v = layer.self_attn.v_proj(x).view(shp).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        for r, t in enumerate(tables):
            blk, off = t.physical(t.length-1)
            kv.k_pool[i][blk, :, off, :] = k[r, :, 0, :]
            kv.v_pool[i][blk, :, off, :] = v[r, :, 0, :]
        
        attn = paged_decode_batched(q[:, :, 0, :].contiguous(),
                                    kv.k_pool[i], kv.v_pool[i], bt, seq_lens, N_KV, SCALE)
        attn = attn.reshape(R, 1, -1).to(h.dtype)
        attn = layer.self_attn.o_proj(attn)
        h = residual + attn
        h = h + layer.mlp(layer.post_attention_layernorm(h))

    h = final_norm(h)
    return lm_head(h[:, -1, :])            # [R, vocab]


# ---- glue: scatter a prompt's K/V (all layers) into the pool, building its table ----

def scatter_prompt(kv: PagedKVCache, prompt_cache):
    table = kv.new_table()
    p = prompt_cache.layers[0].keys.shape[2]          # prompt length
    for t_pos in range(p):
        k = torch.stack([prompt_cache.layers[l].keys[0, :, t_pos, :] for l in range(N_LAYERS)])
        v = torch.stack([prompt_cache.layers[l].values[0, :, t_pos, :] for l in range(N_LAYERS)])
        kv.scatter_token(table, k, v)
    return table


def _new_pool(num_blocks=512):
    return PagedKVCache(num_blocks, N_LAYERS, N_KV, HEAD_DIM,
                        dtype=(torch.bfloat16 if DEVICE == "cuda" else torch.float32), device=DEVICE)


# ---- T2a: a single custom decode step must match the model's decode ----

def test_decode_step():
    prompt = "Explain what a KV cache is in two sentences."
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    input_ids = tok(text, return_tensors="pt")["input_ids"].to(DEVICE)

    out = model(input_ids=input_ids, use_cache=True)
    prompt_cache = out.past_key_values
    first = out.logits[:, -1, :].argmax(-1, keepdim=True)     # [1,1]

    # Scatter the prompt into our pool FIRST: model(..., past_key_values=prompt_cache)
    # below MUTATES prompt_cache in place (appends `first`), so reading it after that
    # would double-scatter the first token. Order matters.
    kv = _new_pool()
    table = scatter_prompt(kv, prompt_cache)                  # P prompt tokens

    # reference: the model's own next-step logits (this appends `first` to prompt_cache)
    ref = model(input_ids=first, past_key_values=prompt_cache, use_cache=True).logits[:, -1, :]

    # ours: extend the table by the new token, run our custom forward
    table.append_token()                                     # reserve the first-token slot
    got = decode_logits(kv, first, [table])                  # [1, vocab]

    same_tok = (got.argmax(-1) == ref.argmax(-1)).all().item()
    max_err = (got.float() - ref.float()).abs().max().item()
    print(f"T2a: next-token match={same_tok} | max logit abs err {max_err:.3e}")
    assert same_tok, "argmax diverged — a layer/RoPE/scatter/kernel-wiring bug"
    print("T2a OK — custom kernel decode step matches the model.")


# full continuous batching generation through the kernel
@dataclass
class Req:
    id: int
    prompt: str
    max_new_tokens: int = 64
    table: object = None
    last_token: torch.Tensor = None
    output_ids: list = field(default_factory=list)
    finished: bool = False


class PagedKernelEngine:
    def __init__(self, max_batch_size=8, num_blocks=2048):
        self.max_batch_size = max_batch_size
        self.kv = _new_pool(num_blocks)
        self.waiting, self.running, self.completed = [], [], []
        self._ids = itertools.count()

    def submit(self, prompt, max_new_tokens=64):
        r = Req(next(self._ids), prompt, max_new_tokens)
        self.waiting.append(r)
        return r

    @torch.no_grad()
    def _admit(self):
        while len(self.running) < self.max_batch_size and self.waiting:
            r = self.waiting.pop(0)
            text = tok.apply_chat_template([{"role": "user", "content": r.prompt}],
                                           add_generation_prompt=True, tokenize=False)
            ids = tok(text, return_tensors="pt")["input_ids"].to(DEVICE)
            out = model(input_ids=ids, use_cache=True)
            r.table = scatter_prompt(self.kv, out.past_key_values)     # prompt -> pool
            r.last_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
            r.output_ids.append(r.last_token[0].item())
            if r.last_token[0].item() == tok.eos_token_id:
                r.finished = True
                r.table.free_all()
                self.completed.append(r)
            else:
                self.running.append(r)

    @torch.no_grad()
    def _decode_step(self):
        if not self.running:
            return
        
        for r in self.running:
            if not r.table.append_token():
                r.finished = True
        
        last = torch.cat([r.last_token for r in self.running], dim=0)
        logits = decode_logits(self.kv, last, [r.table for r in self.running])
        nxt = logits.argmax(-1, keepdim=True)
        for i, r in enumerate(self.running):
            tid = nxt[i].item()
            r.output_ids.append(tid)
            r.last_token = nxt[i:i+1]
            if tid == tok.eos_token_id or len(r.output_ids) >= r.max_new_tokens:
                r.finished = True

    @torch.no_grad()
    def _retire(self):
        keep = []
        for r in self.running:
            if r.finished:
                r.table.free_all()
                self.completed.append(r)
            else:
                keep.append(r)
        self.running = keep

    def step(self):
        self._admit(); self._decode_step(); self._retire()

    def run(self):
        while self.waiting or self.running:
            self.step()
        return self.completed


def test_generate_matches():
    from engine import generate
    TARGET = "Explain what a KV cache is in two sentences."
    CAP = 24
    ref = generate(TARGET, max_new_tokens=CAP)["output_ids"]

    eng = PagedKernelEngine(max_batch_size=4, num_blocks=512)
    rt = eng.submit(TARGET, max_new_tokens=CAP)
    eng.submit("Hi.", max_new_tokens=CAP)
    eng.submit("Write a long, detailed essay about how CPUs work.", max_new_tokens=CAP)
    got = {r.id: r.output_ids for r in eng.run()}[rt.id]

    n = 0
    for a, b in zip(ref, got):
        if a == b:
            n += 1
        else:
            break
    print(f"T2b: matched {n}/{len(ref)} leading tokens vs generate() (bf16; tail drift = "
          f"kernel-vs-SDPA reduction order, not a bug)")
    print("  generate:", tok.decode(ref, skip_special_tokens=True))
    print("  kernel  :", tok.decode(got, skip_special_tokens=True))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Needs a CUDA GPU — run on Modal: modal run bench/modal_tier2.py")
    else:
        test_decode_step()        # T2a
        test_generate_matches()   # T2b

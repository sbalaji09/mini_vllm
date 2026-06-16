"""
Phase 4 TIER 2 (Route 2) — generate through the Triton PagedAttention kernel.

We OWN the decode forward: reuse the model's submodules (projections, RoPE, norms,
MLP) but replace attention with paged_decode_batched (the K3 kernel) reading our
PagedKVCache pool. Prefill stays model(...) + scatter (Phase 2B).

Verify (T2a here): a single custom decode step must match the model's own decode
logits. Run on Modal: modal run bench/modal_tier2.py
"""
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
        # TODO (yours): one Qwen2 decoder layer with attention replaced by the kernel.
        #   residual = h
        #   x = layer.input_layernorm(h)
        #   shp = (R, 1, -1, HEAD_DIM)
        #   q = layer.self_attn.q_proj(x).view(shp).transpose(1, 2)   # [R, N_Q, 1, HEAD_DIM]
        #   k = layer.self_attn.k_proj(x).view(shp).transpose(1, 2)   # [R, N_KV,1, HEAD_DIM]
        #   v = layer.self_attn.v_proj(x).view(shp).transpose(1, 2)
        #   q, k = apply_rotary_pos_emb(q, k, cos, sin)
        #   # SCATTER this layer's new token K/V into the pool (slot already allocated):
        #   for r, t in enumerate(tables):
        #       blk, off = t.physical(t.length - 1)
        #       kv.k_pool[i][blk, :, off, :] = k[r, :, 0, :]
        #       kv.v_pool[i][blk, :, off, :] = v[r, :, 0, :]
        #   # KERNEL attention (q_in [R, N_Q, HEAD_DIM]); reads the pool in place:
        #   attn = paged_decode_batched(q[:, :, 0, :].contiguous(),
        #                               kv.k_pool[i], kv.v_pool[i], bt, seq_lens, N_KV, SCALE)
        #   attn = attn.reshape(R, 1, -1).to(h.dtype)        # cast fp32 -> model dtype for o_proj
        #   attn = layer.self_attn.o_proj(attn)
        #   h = residual + attn
        #   h = h + layer.mlp(layer.post_attention_layernorm(h))
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


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Needs a CUDA GPU — run on Modal: modal run bench/modal_tier2.py")
    else:
        test_decode_step()

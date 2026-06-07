import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()

@torch.no_grad()
def generate(prompt: str, max_new_tokens: int = 64):
    # Build chat-formatted input ids:
    msgs = [{"role": "user", "content": prompt}]
    input_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")

    t0 = time.perf_counter()
    ttft = None
    generated = []

    # --- PREFILL ---
    # TODO: one forward pass over the FULL input_ids with use_cache=True.
    #   out = model(input_ids=..., use_cache=True)
    #   out.logits has shape [1, seq_len, vocab]; you want the LAST position.
    #   out.past_key_values IS your KV cache — keep it.
    #   Pick the next token (greedy = argmax). Record ttft here.

    # --- DECODE LOOP ---
    # TODO: loop up to max_new_tokens:
    #   feed ONLY the single new token id ([1,1]) plus past_key_values=...
    #   get logits[:, -1, :], argmax, append, update past_key_values.
    #   break on tok.eos_token_id.
    #   (Notice: the redundant O(n^2) work would be here if you passed the
    #    full sequence instead of one token + the cache. Convince yourself why.)

    dt = time.perf_counter() - t0
    text = tok.decode(generated)
    n = len(generated)
    return {
        "text": text,
        "ttft_s": ttft,
        "total_s": dt,
        "out_tokens": n,
        "tpot_s": (dt - ttft) / max(n - 1, 1) if ttft else None,
        "throughput_tok_s": n / dt,
    }

if __name__ == "__main__":
    r = generate("Explain what a KV cache is in two sentences.")
    print(r["text"]); print({k: v for k, v in r.items() if k != "text"})
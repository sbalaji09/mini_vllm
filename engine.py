"""
Given a prompt, this generates up to max_new_tokens and returns the text generated
"""

import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# loads the tokenizer, the pretrained causal language model weights
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)

# puts the model in inference mode
model.eval()

@torch.no_grad()
def generate(prompt: str, max_new_tokens: int = 64):
    # wraps the prompt in a specific format so that Qwen can actually run it
    msgs = [{"role": "user", "content": prompt}]
    input_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")

    t0 = time.perf_counter()
    ttft = None
    generated = []

    # start by passing the input into the model and use the KV cache for the otuput
    out = model(input_ids=input_ids, use_cache=True)
    # save the past key values since that is needed for learning
    past_key_values = out.past_key_values
    
    # get the last token and call argmax to get the greedy first token
    last_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    # this is the time-to-first-token which is prefill latency
    ttft = time.perf_counter() - t0

    # append the last token to the generated list of tokens
    generated.append(last_token.item())

    # iterates through the max number of new tokens
    for _ in range(max_new_tokens-1):
        # breaks out if the most recent token is the end-of-sequence token
        if last_token.item() == tok.eos_token_id:
            break
            
        # runs the model only on the latest token and uses the past_key_values from the first run
        out = model(
            input_ids=last_token, 
            past_key_values=past_key_values, 
            use_cache=True
        )

        # updates the KV cache to include the new processed token
        past_key_values = out.past_key_values
        # looks at model's prediction for next token and chooses the highest-probability token
        last_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # stores the generated token ID so it can be decoded into text later
        generated.append(last_token.item())
    
    # metrics
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
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
    # wraps the prompt in a specific format so that Qwen can actually run it.
    # apply_chat_template(..., return_tensors=) returns a BatchEncoding in
    # transformers 5.x, not a bare tensor, so render to text then tokenize.
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    input_ids = tok(text, return_tensors="pt")["input_ids"]

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
        "output_ids": generated,   # raw token ids, for the correctness gate
        "ttft_s": ttft,
        "total_s": dt,
        "out_tokens": n,
        "tpot_s": (dt - ttft) / max(n - 1, 1) if ttft else None,
        "throughput_tok_s": n / dt,
    }

@torch.no_grad()
def static_batch_generate(prompts: list[str], max_new_tokens: int = 64):
    # we are left padding the batch because then the rightmost tokens (the most important tokens)
    # of each sequence will align
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # apply chat template as STERINGS
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for p in prompts
    ]

    # tokenize all the texts with padding=True (determined left padding)
    enc = tok(texts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"] # these are the padded token ids
    attention_mask = enc["attention_mask"] # matrix where 1 = real token and 0 is a padded token
    B = input_ids.shape[0] # this is the batch size

    t0 = time.perf_counter()
    ttft = None
    outputs = [[] for _ in range(B)]  # per-sequence generated token ids
    finished = torch.zeros(B, dtype=torch.bool) # per-sequence done flag

    # --- BATCHED PREFILL ---
    # count real tokens and 0 index them
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1) # take the padded tokens and put them at a harmless value

    # this is a prefill over all B prompts at once
    # also need to pass the mask and correct positions so the padding doesn't corrupt anything   
    out = model(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        position_ids=position_ids, 
        use_cache=True
    )

    # this gets the first token for every sequence
    last_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = out.past_key_values
    ttft = time.perf_counter() - t0

    # iterate through every sequence
    for i in range(len(last_token)):
        # if this token is the EOS token, then mark it as finished
        if last_token[i].item() == tok.eos_token_id:
            finished[i] = True
        # append this first token to the outputs array
        outputs[i].append(last_token[i].item())



    # --- BATCHED DECODE LOOP ---
    for _ in range(max_new_tokens - 1):
        # loop stops once every sequence has hit EOS
        if finished.all():
            break
        
        # mask grows by one because it can now look at the new token at the end
        attention_mask = torch.cat(
            [attention_mask, torch.ones((B, 1), dtype=attention_mask.dtype)],
            dim=1
        )

        # next_position_ids is the previous position_ids + 1 (slot number for the token to be generated)
        next_position_ids = position_ids[:, -1:] + 1
        position_ids = torch.cat([position_ids, next_position_ids], dim=-1)

        # forward pass again, feeding the position ids including the new token
        out = model(
            input_ids=last_token,
            attention_mask=attention_mask,
            position_ids=next_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # gets the first token of every sequence
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # iterates through every sequence, but only for non finished sequences
        for i in range(B):
            if not finished[i]:
                # records the new token and appends it to outputs
                token_id = next_token[i].item()
                outputs[i].append(token_id)

                # if this is at the end of the sequence, then mark it as finished
                if token_id == tok.eos_token_id:
                    finished[i] = True
        
        # this feeds every sequence's token forward into the next step
        last_token = next_token
        past_key_values = out.past_key_values


    dt = time.perf_counter() - t0
    texts_out = [tok.decode(o, skip_special_tokens=True) for o in outputs]
    total_out = sum(len(o) for o in outputs)
    return {
        "texts": texts_out,
        "ttft_s": ttft,
        "total_s": dt,
        "batch_size": B,
        "out_tokens": total_out,
        "throughput_tok_s": total_out / dt if dt else None,
    }


if __name__ == "__main__":
    r = generate("Explain what a KV cache is in two sentences.")
    print(r["text"]); print({k: v for k, v in r.items() if k != "text"})
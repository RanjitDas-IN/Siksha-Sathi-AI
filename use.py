import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Settings
# -------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8"
LOG_FILE = "conversation_log.txt"
MAX_TOTAL_NEW_TOKENS = 256  # total tokens assistant may produce for a reply
CHUNK_SIZE = 1              # 1 => one token at a time (token-by-token streaming)
TEMPERATURE = 0.7
TOP_P = 0.9

# -------------------------
# Helpers: logging
# -------------------------
if not os.path.exists(LOG_FILE):
    open(LOG_FILE, "w", encoding="utf-8").close()

def log_message(role: str, content: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{role.upper()}: {content}\n\n")

# -------------------------
# Load model & tokenizer
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# Ensure a pad token exists (some models don't set it). If none, set to eos.
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# -------------------------
# Chat loop with simulated streaming
# -------------------------
system_message = {"role": "system", "content": "You are a helpful assistant. Always respond clearly."}
conversation = [system_message]

print("Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("User: ").strip()
    if user_input.lower() in ("exit", "quit"):
        break

    # Append user into conversation and log
    conversation.append({"role": "user", "content": user_input})
    log_message("user", user_input)

    # Build tokenized prompt (keeps explicit roles)
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)              # shape: (1, seq_len)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is None:
        # Fallback: all ones
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    # We'll iteratively extend generated_ids and attention mask
    generated_ids = input_ids.clone()                       # (1, cur_len)
    generated_attention_mask = attention_mask.clone()       # (1, cur_len)

    # Prepare streaming print + logging buffer for assistant
    print("Assistant: ", end="", flush=True)
    assistant_buffer = ""

    total_generated = 0
    stop_early = False

    with torch.inference_mode():
        while total_generated < MAX_TOTAL_NEW_TOKENS and not stop_early:
            # Each call, ask the model to produce CHUNK_SIZE new tokens
            outputs = model.generate(
                input_ids=generated_ids,
                attention_mask=generated_attention_mask,
                max_new_tokens=CHUNK_SIZE,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            # outputs shape: (1, new_total_seq_len)
            start_idx = generated_ids.shape[-1]
            new_tokens = outputs[:, start_idx:]   # keep 2D: shape (1, new_len)

            if new_tokens.shape[-1] == 0:
                # nothing new produced — break to avoid infinite loop
                break

            # Append new tokens to generated_ids (both 2D)
            generated_ids = torch.cat([generated_ids, new_tokens], dim=-1)

            # Extend attention mask with ones for the newly generated tokens
            new_mask = torch.ones((generated_attention_mask.shape[0], new_tokens.shape[-1]),
                                  dtype=generated_attention_mask.dtype, device=device)
            generated_attention_mask = torch.cat([generated_attention_mask, new_mask], dim=-1)

            # Decode only the newly generated tokens for streaming output
            decoded_chunk = tokenizer.decode(new_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            print(decoded_chunk, end="", flush=True)
            assistant_buffer += decoded_chunk

            total_generated += new_tokens.shape[-1]

            # Stop if EOS token was generated in the new tokens
            if (new_tokens == tokenizer.eos_token_id).any():
                stop_early = True
                break

        # End streaming for this reply
    print("\n")  # newline after assistant done

    # Append assistant reply to conversation and log it
    conversation.append({"role": "assistant", "content": assistant_buffer})
    log_message("assistant", assistant_buffer)

#!/usr/bin/env python3
"""
Device-independent token-by-token streaming for Qwen (or other HF models).
- Uses chunk-by-chunk generate (CHUNK_SIZE tokens per call). Default: 1 token.
- Decodes only newly generated tokens each loop iteration.
- Updates attention_mask to match generated_ids.
- Writes user/assistant messages to conversation_log.txt (also supports real-time token logging).
"""

import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Config
# -------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8"
LOG_FILE = "conversation_log.txt"
REALTIME_TOKEN_LOG = True         # if True, append tokens to the log as they are generated
CHUNK_SIZE = 1                    # 1 = token-by-token; increase for larger chunks
MAX_TOTAL_NEW_TOKENS = 512        # safety cap per assistant reply
TEMPERATURE = 0.7
TOP_P = 0.9
SLEEP_BETWEEN_TOKENS = 0.0        # small sleep to make typing feel human (e.g. 0.01)

# -------------------------
# Helpers: logging
# -------------------------
if not os.path.exists(LOG_FILE):
    open(LOG_FILE, "w", encoding="utf-8").close()

def append_to_log(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()

def log_message(role: str, content: str):
    append_to_log(f"{role.upper()}: {content}\n\n")

# -------------------------
# Device selection & model load
# -------------------------
if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"[info] selected device: {device}")

# choose dtype sensibly
torch_dtype = None
if device == "cuda" or device == "mps":
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch_dtype)
model.to(device)
model.eval()

# ensure pad_token exists
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# -------------------------
# Chat state
# -------------------------
system_message = {"role": "system", "content": "You are a helpful assistant. Always respond clearly."}
conversation = [system_message]

print("Type 'exit' or 'quit' to stop.\n")

# -------------------------
# Main loop: token-by-token streaming
# -------------------------
while True:
    try:
        user_input = input("User: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        break

    if user_input.lower() in ("exit", "quit"):
        print("Bye.")
        break

    # record user
    conversation.append({"role": "user", "content": user_input})
    log_message("user", user_input)

    # build prompt using apply_chat_template so roles are preserved
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    # move tensors to device
    input_ids = inputs["input_ids"].to(device)                 # (1, seq_len)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    # we'll iteratively grow these
    generated_ids = input_ids.clone()                          # (1, cur_len)
    generated_attention_mask = attention_mask.clone()          # (1, cur_len)

    print("Assistant: ", end="", flush=True)
    assistant_buffer = ""

    total_generated = 0
    stop_early = False

    # Optionally open the log file for streaming token writes (keeps file open for speed)
    token_log_file = None
    if REALTIME_TOKEN_LOG:
        token_log_file = open(LOG_FILE, "a", encoding="utf-8")

    with torch.inference_mode():
        while total_generated < MAX_TOTAL_NEW_TOKENS and not stop_early:
            # ask model to produce CHUNK_SIZE new tokens
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
            new_tokens = outputs[:, start_idx:]    # <- keeps it 2D (1, new_len)

            if new_tokens.shape[-1] == 0:
                # model returned nothing new; break to avoid infinite loop
                break

            # append new tokens to generated_ids and extend attention mask
            generated_ids = torch.cat([generated_ids, new_tokens], dim=-1)
            new_mask = torch.ones((generated_attention_mask.shape[0], new_tokens.shape[-1]),
                                  dtype=generated_attention_mask.dtype, device=device)
            generated_attention_mask = torch.cat([generated_attention_mask, new_mask], dim=-1)

            # decode only the newly generated tokens
            decoded_chunk = tokenizer.decode(new_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            print(decoded_chunk, end="", flush=True)
            assistant_buffer += decoded_chunk

            # write tokens to log in real time if requested
            if REALTIME_TOKEN_LOG and token_log_file is not None:
                token_log_file.write(decoded_chunk)
                token_log_file.flush()

            total_generated += new_tokens.shape[-1]

            # break if EOS was generated among new tokens
            if (new_tokens == tokenizer.eos_token_id).any():
                stop_early = True
                break

            # optional human-like pacing
            if SLEEP_BETWEEN_TOKENS > 0:
                time.sleep(SLEEP_BETWEEN_TOKENS)

    # close token log file if opened
    if token_log_file is not None:
        token_log_file.write("\n\n")
        token_log_file.close()

    print("\n")  # newline after assistant done

    # append assistant reply to convo and common log (full message)
    conversation.append({"role": "assistant", "content": assistant_buffer})
    log_message("assistant", assistant_buffer)

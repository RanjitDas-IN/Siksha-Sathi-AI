#!/usr/bin/env python3
"""
Realtime token-by-token streaming with realtime log writes and an print_chunk hook
you can replace with a websocket/SSE sender.

- Streams tokens as `decoded_chunk` (the streaming boundary).
- Writes token chunks to LOG_FILE as they arrive (mimics uploaded script).
- Appends full user/assistant messages to LOG_FILE at end of turn.
"""
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


data = """
{
  "id": "19a1168eb167f2e8",
  "threadId": "19a1168eb167f2e8",
  "from": "Ranjit Das <noreply@github.com>",
  "to": "udaysubba2004@gmail.com",
  "subject": "RanjitDas-IN invited you to RanjitDas-IN/Siksha-Sathi-AI",
  "date": "Thu, 23 Oct 2025 07:11:19 -0700",
  "body": "@RanjitDas-IN has invited you to collaborate on the RanjitDas-IN/Siksha-Sathi-AI repository.\r\n\r\n\r\nVisit https://github.com/RanjitDas-IN/Siksha-Sathi-AI/invitations to accept or decline this invitation.\r\n\r\nYou can also head over to https://github.com/RanjitDas-IN/Siksha-Sathi-AI to check out the repository or visit https://github.com/RanjitDas-IN to learn a bit more about @RanjitDas-IN.\r\n\r\nThis invitation will expire in 7 days.\r\n\r\nSome helpful tips:\r\n- If you get a 404 page, make sure you’re signed in as kisxo.\r\n- Too many emails from @RanjitDas-IN? You can block them by visiting\r\n  https://github.com/settings/blocked_users?block_user=RanjitDas-IN or report abuse at\r\n  https://github.com/contact/report-abuse?report=RanjitDas-IN\r\n\r\n---\r\nView it on GitHub:\r\nhttps://github.com/RanjitDas-IN/Siksha-Sathi-AI"
}
"""


def print_chunk(chunk: str):
    print(chunk, end="", flush=True)

# ---------- Config ----------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # change as needed
LOG_FILE = "conversation_log.txt"
CHUNK_SIZE = 1
MAX_TOTAL_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
SLEEP_BETWEEN_TOKENS = 0.0

# ensure log exists
if not os.path.exists(LOG_FILE):
    open(LOG_FILE, "w", encoding="utf-8").close()

def append_to_log(text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()

def log_message(role: str, content: str):
    append_to_log(f"{role.upper()}: {content}\n\n")

# ---------- Model / tokenizer loader ----------
def load_model_and_tokenizer(model_name: str, device: str = None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer, device

# ---------- Streaming generator ----------
def stream_response_generator(
    conversation: list,
    model,
    tokenizer,
    device,
    chunk_size: int = CHUNK_SIZE,
    max_new_tokens: int = MAX_TOTAL_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    sleep_between_tokens: float = SLEEP_BETWEEN_TOKENS,
):
    """
    Yields decoded_chunk strings as they arrive.
    Caller is responsible for forwarding each chunk (e.g., ws.send(chunk)).
    """
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    generated_ids = input_ids.clone()
    generated_attention_mask = attention_mask.clone()

    total_generated = 0
    stop_early = False

    with torch.inference_mode():
        while total_generated < max_new_tokens and not stop_early:
            outputs = model.generate(
                input_ids=generated_ids,
                attention_mask=generated_attention_mask,
                max_new_tokens=chunk_size,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

            start_idx = generated_ids.shape[-1]
            new_tokens = outputs[:, start_idx:]  # (1, new_len)

            if new_tokens.shape[-1] == 0:
                break

            # extend generated_ids and attention mask
            generated_ids = torch.cat([generated_ids, new_tokens], dim=-1)
            new_mask = torch.ones((generated_attention_mask.shape[0], new_tokens.shape[-1]),
                                  dtype=generated_attention_mask.dtype, device=device)
            generated_attention_mask = torch.cat([generated_attention_mask, new_mask], dim=-1)

            # ===== streaming boundary =====
            decoded_chunk = tokenizer.decode(new_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # yield the chunk so caller can forward it to frontend in realtime
            yield decoded_chunk
            # ==============================

            total_generated += new_tokens.shape[-1]

            # stop on EOS
            if (new_tokens == tokenizer.eos_token_id).any():
                stop_early = True
                break

            if sleep_between_tokens > 0:
                time.sleep(sleep_between_tokens)

# ---------- Convenience wrapper that accepts an print_chunk callback ----------
def stream_response_callback(
    conversation: list,
    model,
    tokenizer,
    device,
    print_chunk,
    chunk_size: int = CHUNK_SIZE,
    max_new_tokens: int = MAX_TOTAL_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    sleep_between_tokens: float = SLEEP_BETWEEN_TOKENS,
    realtime_token_log: bool = True,
    log_file_path: str = LOG_FILE,
):
    """
    Calls print_chunk(decoded_chunk) for every token-chunk produced.
    Also writes chunks to the log file in realtime if realtime_token_log=True.
    Returns the final assembled assistant string.
    """
    assistant_buffer = ""

    # optionally open log file once for speed
    token_log_file = None
    if realtime_token_log:
        token_log_file = open(log_file_path, "a", encoding="utf-8")

    try:
        for chunk in stream_response_generator(
            conversation,
            model,
            tokenizer,
            device,
            chunk_size=chunk_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            sleep_between_tokens=sleep_between_tokens,
        ):
            # forward to user-provided callback (e.g., websocket send)
            try:
                print_chunk(chunk)
            except Exception:
                # if frontend send fails, propagate after cleanup
                raise

            assistant_buffer += chunk

            # write chunk to log in realtime (no newline)
            if token_log_file is not None:
                token_log_file.write(chunk)
                token_log_file.flush()
    finally:
        if token_log_file is not None:
            token_log_file.write("\n\n")
            token_log_file.close()

    return assistant_buffer

# ---------- Main realtime loop (console example) ----------
def ai(user_input: str, rag_data: str = '', user_data: str='', chat_history: str=''):

    # print(user_data)
    # return
    model, tokenizer, device = load_model_and_tokenizer(MODEL_NAME)

    system_message = {"role": "system", "content": "You are a helpful assistant. Always respond in english and clearly.", "custom_data": rag_data}
    conversation = [system_message]
    conversation.append({"role": "user", "content": user_input})

    # call streaming wrapper which will also append tokens to LOG_FILE realtime
    final_text = stream_response_callback(
        conversation,
        model,
        tokenizer,
        device,
        print_chunk=print_chunk,
        chunk_size=CHUNK_SIZE,
        max_new_tokens=MAX_TOTAL_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        sleep_between_tokens=SLEEP_BETWEEN_TOKENS,
        realtime_token_log=True,
        log_file_path=LOG_FILE,
    )


user_input=f"""Use this raw email data, and create a summery in proper english (such that an small boy can understand).
### Raw-Email: {data}

"""


# print(user_input)
ai(user_input)


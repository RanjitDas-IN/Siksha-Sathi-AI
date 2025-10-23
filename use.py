import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8"
CONVO_FILE = Path("conversation.txt")

conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Always respond in clear English. Do NOT invent or include 'User:' or 'Assistant:' transcripts inside your replies."
    }
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cpu"
)
model.eval()

if CONVO_FILE.exists():
    try:
        with open(CONVO_FILE, "r", encoding="utf-8") as f:
            old_data = json.load(f)
            if isinstance(old_data, list):
                conversation.extend(old_data)
    except:
        pass

print("Start chatting! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        break

    conversation.append({"role": "user", "content": user_input})

    # Build prompt
    prompt_text = ""
    for turn in conversation:
        prompt_text += f"{turn['content']}\n"

    inputs = tokenizer(prompt_text, return_tensors="pt").to("cpu")

    # -----------------------------
    # Streaming generation
    # -----------------------------
    output_tokens = []
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True
        )

    # decode token by token for streaming
    for i, token_id in enumerate(generated.sequences[0][inputs['input_ids'].shape[1]:]):
        token = tokenizer.decode(token_id, skip_special_tokens=False)
        print(token, end="", flush=True)
        output_tokens.append(token)
    print("\n")  # final newline

    response = "".join(output_tokens)
    conversation.append({"role": "assistant", "content": response})

    with open(CONVO_FILE, "w", encoding="utf-8") as f:
        json.dump(conversation, f, indent=4)

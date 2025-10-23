from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model: conversational/instruct-tuned
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")  # auto GPU if available

# System prompt to constrain behavior
system_prompt = (
    "You are a helpful AI assistant. ONLY chat naturally with the user. "
    "Do not give code, explanations, or multi-choice answers unless explicitly asked. "
    "Keep answers short and friendly."
)

print("Start chatting with Qwen 2.5 1.5B (type 'exit' to quit)")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Build prompt with system instruction
    prompt = f"{system_prompt}\nUser: {user_input}\nAI:"

    # Encode input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and print response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt part to get only AI's answer
    response_clean = response.split("AI:")[-1].strip()
    print(f"Qwen: {response_clean}")

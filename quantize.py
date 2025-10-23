import time
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# 1) Load tokenizer & model on CPU
tokenizer = AutoTokenizer.from_pretrained(model_name)
# load model onto CPU explicitly
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=None, torch_dtype=torch.float32)
model.to("cpu")
model.eval()

# (Optional) limit threads to keep CPU usage sane / reproducible
torch.set_num_threads(4)   # tune to your CPU cores

# 2) Quantize dynamically: targets nn.Linear modules (common and safe)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},       # quantize linear layers
    dtype=torch.qint8        # 8-bit dynamic quantization
)

# 3) Quick size check (save state_dict temporarily to measure bytes)
tmp_before = "orig_state.pth"
tmp_after = "quant_state.pth"
torch.save(model.state_dict(), tmp_before)
torch.save(quantized_model.state_dict(), tmp_after)
size_before = os.path.getsize(tmp_before) / (1024**2)
size_after = os.path.getsize(tmp_after) / (1024**2)
os.remove(tmp_before); os.remove(tmp_after)

print(f"Approx model state_dict size before: {size_before:.2f} MB")
print(f"Approx model state_dict size after : {size_after:.2f} MB")

# 4) Quick inference speed test (simple prompt, small generation)
prompt = "Explain dynamic quantization in one short sentence."
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

# Warm-up (1x)
with torch.inference_mode():
    _ = quantized_model.generate(**inputs, max_new_tokens=8)

# Timed run
runs = 3
times = []
with torch.inference_mode():
    for _ in range(runs):
        t0 = time.time()
        _ = quantized_model.generate(**inputs, max_new_tokens=32, do_sample=False)
        times.append(time.time() - t0)

print(f"Generation times (s) over {runs} runs: {times}")
print(f"Median generation time: {sorted(times)[len(times)//2]:.3f}s")

# 5) Save the quantized model for later reuse
# You can save the quantized model object (pickle) or its state_dict.
# Note: saving the whole model with torch.save(model) will pickle the module.
torch.save(quantized_model.state_dict(), "qwen_quantized_state.pth")
# OR (pickle the full model)
torch.save(quantized_model, "qwen_quantized_full_model.pt")

print("Saved quantized model state to qwen_quantized_state.pth")
print("Saved full pickled quantized model to qwen_quantized_full_model.pt")

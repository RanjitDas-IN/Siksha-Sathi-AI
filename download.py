# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)
# ----------------------------------------------------------------------------------

# Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM-360M")

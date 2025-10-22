from transformers import pipeline

print("Downloading........\nsize(3GB+)")
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)


"""
When you call:

```python
pipe(messages)
```

* The pipeline auto-detects it’s a **chat-style input** (Qwen supports chat templates)
* It tokenizes the `messages`
* Performs forward pass (generates response)
* Returns output like:

  ```python
  [{'generated_text': 'Who am I? I am Qwen, an AI language model developed by Alibaba...'}]
  ```
"""


# ---------------------------------------------------------------------------------------------------------------------------
""" 
Or you can download the quantize version: 
I prefare the quantize version

# !pip install llama-cpp-python

from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="Triangle104/Qwen2.5-1.5B-Q4_K_S-GGUF",
	filename="qwen2.5-1.5b-q4_k_s.gguf",
)

"""
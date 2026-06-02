#!/usr/bin/anaconda/python3/bin/python3

from llama_cpp import Llama
import time

"""
This benchmarks llama.cpp.

its helpful to  run nvidia-smi (with a watch command) while loading the model to monitor VRAM usage.
  * running this shows other processes take GPU VRAM too (the X server, web browsers, Slack, etc.)
  * its very useful to run this - do so, and shut down processes that take VRAM

Point model_path at a local GGUF model below.
"""

model = Llama(
    model_path="/path/to/models/your-model.gguf",
    n_ctx=4096,
    n_gpu_layers=16,      # Adjust based on VRAM
    n_threads=12,          # Set to your CPU core count
    verbose=False
)

prompt = "Why is the sky blue? Explain like a science teacher."

# Warmup
model(prompt, max_tokens=10)

# Benchmark
start = time.time()
output = model(prompt, max_tokens=200)
tokens_gen = len(output['choices'][0]['text'].split())
duration = time.time() - start

print(f"Speed: {tokens_gen/duration:.2f} tokens/sec")
print(f"Tokens: {tokens_gen} | Time: {duration:.2f}s") 

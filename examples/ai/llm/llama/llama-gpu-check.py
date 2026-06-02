#!/usr/bin/anaconda/python3/bin/python3

from llama_cpp import Llama
import time

"""

This script runs a basic GPU check.

Point model_path at a local GGUF model, then adjust the layer counts in the
loop below to find how many layers fit in your GPU's VRAM.
"""

model_path = "/path/to/models/your-model.gguf"


myExit = False

#for layers in [12, 14, 15, 16, 17, 18, 19, 20, 25, 30, 33, 35, 38, 39, 40, 45, 50, 55]:
#for layers in [21, 22, 23, 24, 25, 30]:
for layers in [55, 56, 57, 58, 59, 60]:
    if (myExit): 
        break  
    try:
        start = time.time()
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=layers,
            n_ctx=2048,
            offload_kqv=True,
            verbose=False
        )
        load_time = time.time() - start
        print(f"✅ Loaded {layers} GPU layers in {load_time:.1f}s")
        
        # Quick inference test
        output = llm("Hello", max_tokens=10)
        print(f"Test generation: {output['choices'][0]['text']}")

        del llm #releases from memory
        
    except Exception as e:
        print(f"❌ Failed at {layers} layers: {str(e)}")
        myExit = True 


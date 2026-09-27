#!/usr/bin/env python3
"""CS-21: what a bigger context window costs in VRAM, per model and cache setting (RTX 5090).

Why
---
A delegate worker shares the main model's context window (it is the same loaded model, and llama.cpp
fixes n_ctx at load), and the 8192-token window used so far let two parallel page fetches overflow it
(2026-09-25). This measures how far each first-cut model's window can grow on the 5090, so
max_context_tokens can be chosen from numbers rather than guessed.

What it does
------------
For each model and each cache setting it loads the model in a FRESH child process - exactly as
StreamBase does: every layer on the GPU, LlamaUtils.build_context_kwargs(flash_attn, kv_cache_type) -
at 8K, 16K, 32K and 64K, generates a few tokens (so compute buffers are allocated), and reads the
child's VRAM from nvidia-smi. A size that fails to load ends that series. Settings:
  * "current"  - the model's settings in live_tool_smoke.MODELS (Qwen/Muse: flash attention off, f16);
  * "fa_q8"    - flash attention on, q8_0 cache (what Gemma 4 already uses).
The embedding model (a few hundred MB) is not included; leave room for it.

Usage (llama env; needs the GPU free - it loads each model several times):
    python probe_context_vram.py --out /tmp/ctx_vram.json [--arch qwen35moe] [--sizes 8192 32768]
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()
SRC = settings.SRC

SIZES = [8192, 16384, 32768, 65536]
CHILD = r'''
import json, os, subprocess, sys, time
sys.path.insert(0, {src!r})
from llama_cpp import Llama
from amadeo_utils.ai.llm.llama.llama_utils import LlamaUtils
kwargs = LlamaUtils.build_context_kwargs({flash!r}, {kv!r}, lambda *a: None)
started = time.time()
llm = Llama(model_path={model!r}, n_gpu_layers=-1, n_ctx={ctx}, verbose=False, **kwargs)
loaded = time.time() - started
t0 = time.time()
out = llm.create_completion("The capital of France is", max_tokens=32, temperature=0)
tokens = out["usage"]["completion_tokens"]
speed = tokens / max(1e-6, time.time() - t0)
used = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                      capture_output=True, text=True).stdout
mine = [int(l.split(",")[1]) for l in used.splitlines() if l.split(",")[0].strip() == str(os.getpid())]
total = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.split(",")
print("RESULT " + json.dumps({{"vram_mib": mine[0] if mine else None, "gpu_used_mib": int(total[0]),
                              "gpu_total_mib": int(total[1]), "load_s": round(loaded, 1), "tok_s": round(speed, 1)}}))
'''


def measure(model, ctx, flash, kv):
    """Loads one configuration in a child process. Returns its result dict, or {'error': ...} if it did not load."""
    code = CHILD.format(src=SRC, flash=flash, kv=kv, model=model, ctx=ctx)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    for line in done.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    tail = (done.stderr or "").strip().splitlines()[-3:]
    return {"error": " | ".join(tail)[:300] or f"exit {done.returncode}"}


def main():
    from live_tool_smoke import MODELS
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--arch", choices=sorted(MODELS), action="append")
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    args = parser.parse_args()

    results = []
    for arch in args.arch or sorted(MODELS):
        model, extra = MODELS[arch]
        model = settings.model_path(model)
        settings = {"current": (extra.get("flash_attn", False), extra.get("kv_cache_type", "f16"))}
        if settings["current"] != (True, "q8_0"):
            settings["fa_q8"] = (True, "q8_0")
        for label, (flash, kv) in settings.items():
            for ctx in args.sizes:
                row = {"arch": arch, "setting": label, "flash_attn": flash, "kv_cache_type": kv, "ctx": ctx}
                row.update(measure(model, ctx, flash, kv))
                results.append(row)
                shown = (f"{row['vram_mib'] / 1024:5.1f} GB  (card {row['gpu_used_mib'] / 1024:.1f}/"
                         f"{row['gpu_total_mib'] / 1024:.1f})  {row['tok_s']:5.1f} tok/s  load {row['load_s']}s"
                         if "error" not in row else f"FAILED: {row['error']}")
                print(f"{arch:13} {label:8} ctx {ctx:6}  {shown}", flush=True)
                with open(args.out, "w") as fh:
                    json.dump(results, fh, indent=2)
                if "error" in row:
                    break                           # a bigger window will not fit either
                time.sleep(2)                       # let the driver release the last child's memory


if __name__ == "__main__":
    main()

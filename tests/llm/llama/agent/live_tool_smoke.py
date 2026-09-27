#!/usr/bin/env python3
"""CS-21: a short LIVE run of ToolStream on the GPU, recording what the real models emit.

Why this exists
---------------
Everything else in tests/ stubs the model. That proves the loop, but six fixtures per
architecture in tool_call_fixtures.py are 'inferred' - assembled from template pieces, never
seen from a real model. This runs real generation with only the two harmless built-ins
(get_datetime, calculator) and records, per prompt:

* every raw generation, exactly as the model produced it (before any parsing);
* how parse_tool_calls classified each one;
* which tools ran, how the loop ended, and the reply.

Prompts are synthetic. Output goes only to --out. Uses the GPU (one model at a time).

Usage (llama env)
-------------------------
    python live_tool_smoke.py --arch qwen35moe --out /tmp/live_qwen.json
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()
settings.use_area("llm", "llama", "streams")   # golden_response: the shared config builder

import golden_response as GR                                  # noqa: E402
from amadeo_utils.ai.llm.llama.ToolStream import ToolStream   # noqa: E402
from amadeo_utils.ai.llm.llama import chat_template as CT     # noqa: E402

# Settings copied from the owner's standalone system configs (all full-GPU, 8192 context).
MODELS = {
    "muse-glimmer": ("Muse-Glimmer-30B-Abliterated-Q8_0.gguf", {"flash_attn": False, "kv_cache_type": "f16"}),
    "qwen35moe": ("Qwen3.6-35B-A3B-Aggressive-Q6.gguf", {"flash_attn": False, "kv_cache_type": "f16"}),
    "gemma4": ("gemma-4-31B-it-abliterated.gguf", {"flash_attn": True, "kv_cache_type": "q8_0"}),
}
PROMPTS = [
    ("calculator", "What is 1234 multiplied by 5678? Use the calculator tool rather than working it out yourself."),
    ("datetime", "What is today's date and the current time in New York?"),
    ("no_tool", "In one sentence, why is the sky blue?"),
    ("reasoned_calculator", "!reason What is the square root of 2 times pi, to six decimal places?"),
]


def main():
    parser = argparse.ArgumentParser(description="CS-21 live tool smoke test")
    parser.add_argument("--arch", required=True, choices=sorted(MODELS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model, extra = MODELS[args.arch]
    model = settings.model_path(model)             # MODELS names files inside model_dir
    workdir = tempfile.mkdtemp(prefix="cs21-live-")
    try:
        config = GR.build_args(model, args.arch, 8192, workdir, workdir)
        config.update(extra)
        config.update({"generating_gpu_layers": -1, "embedding_gpu_layers": -1, "max_response_tokens": 512,
                       "tools_allowed": ["get_datetime", "calculator"], "tool_mode": "auto"})
        stream = ToolStream(config)

        raw_log = []
        real = stream.llm_generator.create_chat_completion

        def recording(**kwargs):
            response = real(**kwargs)
            raw = response["choices"][0]["message"]["content"]
            tools = kwargs.get("tools")
            parse = CT.parse_tool_calls(raw, stream.llm_generator, tools=tools or [], thinking=False)
            raw_log.append({"offered": [t["function"]["name"] for t in tools or []], "raw": raw,
                            "parsed_kind": parse.kind, "parsed_calls": parse.calls})
            return response
        stream.llm_generator.create_chat_completion = recording

        out = {"arch": args.arch, "model": os.path.basename(model), "prompts": {}}
        for label, prompt in PROMPTS:
            raw_log.clear()
            sid = f"live_{label}"
            stream.create_session(sid, "live", False, False, False)
            response = stream.get_response({"sessionID": sid, "command": "request", "user_request": prompt})
            result = stream.get_session(sid).get("last_result")
            out["prompts"][label] = {
                "prompt": prompt,
                "reply": response.get("response"),
                "message": response.get("message"),
                "executed": result.executed if result else None,
                "terminated": result.terminated if result else None,
                "generations": list(raw_log),
            }
            stream.remove_session(sid)
            print(f"{args.arch}/{label}: ran {result.executed if result else '?'}, "
                  f"ended {result.terminated if result else '?'}, {len(raw_log)} generation(s)", flush=True)
        stream.cleanup()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CS-20: capture what actually constructing each stream class produces.

The gap this fills
------------------
``golden_dispatch.py`` builds instances with ``object.__new__`` and never runs a
constructor; ``golden_render.py`` drives ``chat_template`` directly. Neither touches
``__init__`` - which is the largest shared block and the riskiest part of the refactor.
This probe constructs both families for real, against real models, and records the whole
resulting attribute surface.

It also renders a prompt through the constructed instance's own generator, which is the
end-to-end proof that the chat handler was installed correctly *by the constructor*
rather than merely being installable.

Run it against a model WITH an embedded chat template and again against one WITHOUT
(Midnight-Rose), because the missing-template case takes the ValueError fallback inside
``__init__`` and that path must survive the refactor.

Determinism
-----------
Lock objects, models and vector databases are recorded by type name rather than repr,
since their reprs embed memory addresses. Nothing here samples or reads the clock.

Usage
-----
    python golden_construct.py --model /path/to.gguf --out before.json

Run in the ``llama`` conda environment. CPU-only by default.
"""
import argparse
import json
import os
import sys
import tempfile

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()

from amadeo_utils.ai.llm.llama.RolePlayStream import RolePlayStream
from amadeo_utils.ai.llm.llama.KnowledgeBaseStream import KnowledgeBaseStream
from amadeo_utils.ai.llm.llama import chat_template as ChatTemplate


CONVERSATION = [
    {"role": "system", "content": "You are Ada, a guide. Be concise."},
    {"role": "user", "content": "What is the boiling point of water?"},
]

# Values that are safe to record verbatim. Anything else is recorded as its type name.
SCALAR_TYPES = (str, int, float, bool, type(None))


def build_args(model, model_type, gpu_layers, ctx, knowledge_base_file=None):
    """Assemble the configuration dictionary both families read in __init__."""
    args = {
        # A REAL legacy model_type matters: for a GGUF with no embedded chat template it is
        # the only thing universal_token_count() can fall back on, and an unrecognised value
        # makes construction fail outright. Midnight-Rose is configured as 'llama-3'.
        "model_type": model_type,
        "generating_model": model,
        "embedding_model": settings.get("embedding_model"),
        "gpu": 0,
        "split_gpus": False,
        "flash_attn": False,
        "kv_cache_type": "f16",
        "embedding_gpu_layers": 0,
        "embedding_max_context_tokens": 512,
        "generating_gpu_layers": gpu_layers,
        "generating_max_context_tokens": ctx,
        "chat_format": None,
        "debug": False,
        "encrypted": False,          # keeps role-play's passphrase prompt non-interactive
        "buffer_context_pcnt": 0.05,
        "system_message": "You are Ada, a guide. Be concise.",
        "base_convo_dir": "/tmp/cs20-probe-convo",
        "system_prompt_dir": "/tmp/cs20-probe-prompts",
    }
    if knowledge_base_file:
        args["knowledge_base_file"] = knowledge_base_file
    return args


def describe(instance):
    """Record an instance's attribute surface in a form that is stable across runs."""
    out = {"attributes": sorted(vars(instance).keys()), "values": {}, "types": {}}

    for name, value in sorted(vars(instance).items()):
        if name == "argsDict":
            # The config is our own input; record only its shape.
            out["types"][name] = "dict"
            out["values"][name] = sorted(value.keys())
        elif isinstance(value, SCALAR_TYPES):
            out["values"][name] = value
            out["types"][name] = type(value).__name__
        elif isinstance(value, list) and all(isinstance(v, SCALAR_TYPES) for v in value):
            out["values"][name] = value
            out["types"][name] = "list"
        elif isinstance(value, dict):
            out["types"][name] = "dict"
            out["values"][name] = sorted(value.keys())
        else:
            # Locks, models, vector databases: reprs carry addresses, so record the type.
            out["types"][name] = type(value).__name__

    # End-to-end proof that the constructor installed the chat handler, not just that the
    # handler is installable.
    try:
        out["rendered_through_instance"] = ChatTemplate.format_chat_prompt(instance.llm_generator, CONVERSATION)
    except Exception as e:
        out["rendered_through_instance"] = f"<RAISED {type(e).__name__}: {e}>"

    return out


def main():
    parser = argparse.ArgumentParser(description="CS-20 construction capture")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-type", default="llama-3",
                        help="legacy model_type hint; load bearing for template-less GGUFs")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--ctx", type=int, default=2048)
    args = parser.parse_args()
    args.model = settings.resolve_model(args.model)   # a name inside model_dir, or a path

    result = {"model": os.path.basename(args.model), "model_path": settings.model_name(args.model),
              "model_type": args.model_type, "gpu_layers": args.gpu_layers, "ctx": args.ctx}

    # A throwaway knowledge base, so the probe never reads the real configured one.
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
        for i, (q, a) in enumerate([("What boils at 100C?", "Water, at sea level."),
                                    ("What is the tallest mountain?", "Mount Everest.")], start=1):
            fh.write(json.dumps({"id": i, "question": q, "answer": a}) + "\n")
        kb_file = fh.name

    try:
        rp = RolePlayStream(build_args(args.model, args.model_type, args.gpu_layers, args.ctx))
        result["RolePlayStream"] = describe(rp)
        rp.cleanup()

        kb = KnowledgeBaseStream(build_args(args.model, args.model_type, args.gpu_layers, args.ctx, kb_file))
        result["KnowledgeBaseStream"] = describe(kb)
        # The knowledge base list is read from a temp path, so record its length, not it.
        result["KnowledgeBaseStream"]["values"]["kbl"] = f"<{len(kb.kbl)} entries>"
        result["KnowledgeBaseStream"]["types"]["kbl"] = "list"
        kb.cleanup()
    finally:
        os.unlink(kb_file)

    settings.write_capture(args.out, result)

    for family in ("RolePlayStream", "KnowledgeBaseStream"):
        d = result[family]
        print(f"{family}: {len(d['attributes'])} attributes; "
              f"thinking_supported={d['values'].get('thinking_supported')}; "
              f"stops={d['values'].get('architecture_stops')}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

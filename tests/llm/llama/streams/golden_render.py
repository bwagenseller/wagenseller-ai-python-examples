#!/usr/bin/env python3
"""CS-20: capture prompt rendering and token counts against a real model.

The companion to ``golden_dispatch.py``. That harness pins the dispatch layer with no
model loaded; this one pins the layer underneath it - the chat-template wiring that both
stream families perform in ``__init__`` and depend on for every generation.

Why this is the invariant worth capturing
-----------------------------------------
The refactor splits ``__init__`` between a base class and its subclasses, and the
chat-template setup (``install_chat_handler`` -> ``supports_thinking`` ->
``model_architecture`` -> ``stop_tokens``) is order-sensitive: the handler must be
installed before the rest is interrogated. If the split reorders or drops a step, the
rendered prompt changes. The **rendered prompt string** is therefore the single highest
-value invariant available, and unlike generated text it is fully deterministic - no
sampling is involved in turning a message list into a prompt.

Token counts are captured alongside it because they are what the session's budgeting
logic is built on, and they are likewise deterministic.

Scope
-----
This deliberately drives ``chat_template`` directly rather than through a stream class,
so it keeps working while the stream classes are being taken apart. ``chat_template.py``
is explicitly out of scope for CS-20; this harness is here to prove the refactor does not
disturb it.

Usage
-----
    python golden_render.py --model /path/to/model.gguf --out render_baseline.json
    # after the refactor, re-run and diff

Run in the ``llama`` conda environment.
"""
import argparse
import json
import os
import sys

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()

# Import order matters: llama_utils sets CUDA_DEVICE_ORDER at import time and must be
# imported before llama_cpp, exactly as the stream classes note in their own imports.
from amadeo_utils.ai.llm.llama.llama_utils import LlamaUtils  # noqa: F401
from amadeo_utils.ai.llm.llama import chat_template as ChatTemplate
from llama_cpp import Llama

# A fixed conversation covering the shapes the real code renders: a system message, a
# user turn, an assistant turn, and a second user turn. Held constant so that any change
# in the rendered output is attributable to the code, not the input.
CONVERSATION = [
    {"role": "system", "content": "You are Ada, a guide. Be concise."},
    {"role": "user", "content": "What is the boiling point of water?"},
    {"role": "assistant", "content": "100 degrees Celsius at sea level."},
    {"role": "user", "content": "And at altitude?"},
]


def capture(llm, label):
    """Record everything the stream classes read out of chat_template, plus the render.

    Args:
        llm: a loaded Llama instance with the chat handler already installed.
        label: name for this capture block (e.g. 'thinking_off').

    Returns:
        dict: the captured values, all deterministic.
    """
    # Each probe is guarded individually. Models with no embedded chat template (e.g.
    # Midnight-Rose) reach this function in the fallback state, where some of these raise
    # and others still answer - and *which* ones do is itself behaviour worth pinning.
    block = {}
    for name, probe in (
        ("architecture", ChatTemplate.model_architecture),
        ("supports_thinking", ChatTemplate.supports_thinking),
        ("thinking_enabled", ChatTemplate.thinking_enabled),
        ("stop_tokens", ChatTemplate.stop_tokens),
    ):
        try:
            block[name] = probe(llm)
        except Exception as e:
            block[name] = f"<RAISED {type(e).__name__}: {e}>"

    # The rendered prompt: the highest-value invariant here. Captured in full rather than
    # hashed, so that a diff shows *what* changed rather than merely that something did.
    try:
        block["rendered_prompt"] = ChatTemplate.format_chat_prompt(llm, CONVERSATION)
    except Exception as e:
        block["rendered_prompt"] = f"<RAISED {type(e).__name__}: {e}>"

    # Per-message token counts, which the session budgeting is built on.
    counts = {}
    for message in CONVERSATION:
        key = f"{message['role']}:{message['content'][:30]}"
        try:
            counts[key] = ChatTemplate.count_message_tokens(llm, message["role"], message["content"])
        except Exception as e:
            counts[key] = f"<RAISED {type(e).__name__}: {e}>"
    block["token_counts"] = counts

    return {label: block}


def main():
    parser = argparse.ArgumentParser(description="CS-20 golden render capture")
    parser.add_argument("--model", required=True, help="path to a .gguf model")
    parser.add_argument("--out", required=True, help="path to write the JSON capture to")
    parser.add_argument("--gpu-layers", type=int, default=-1,
                        help="n_gpu_layers; use 0 to stay on CPU")
    parser.add_argument("--ctx", type=int, default=2048, help="context size")
    args = parser.parse_args()
    args.model = settings.resolve_model(args.model)   # a name inside model_dir, or a path

    llm = Llama(model_path=args.model, n_gpu_layers=args.gpu_layers,
                n_ctx=args.ctx, verbose=False)

    # The full path is recorded as well as the basename so that the verifier can re-run
    # this exact capture later without being told which model it came from.
    result = {"model": os.path.basename(args.model), "model_path": settings.model_name(args.model),
              "gpu_layers": args.gpu_layers, "ctx": args.ctx}

    # Mirror exactly what RolePlayStream.__init__ does, including its fallback. A GGUF
    # with no embedded 'tokenizer.chat_template' (Midnight-Rose, for one) makes
    # install_chat_handler raise ValueError, and production catches it and falls back to
    # llama-cpp-python's own prompt formatting. That fallback is a real supported path -
    # none of the three CS-17 models exercise it - so it is captured rather than skipped.
    try:
        ChatTemplate.install_chat_handler(llm, thinking=False)
        result["chat_handler_installed"] = True
        result["fallback_reason"] = None
    except ValueError as e:
        result["chat_handler_installed"] = False
        result["fallback_reason"] = str(e)

    result.update(capture(llm, "thinking_off"))

    # Toggle reasoning on and re-render. Both families flip this per turn, so both states
    # need pinning - a refactor that broke the toggle would otherwise pass unnoticed.
    try:
        thinking_available = ChatTemplate.supports_thinking(llm)
    except Exception:
        thinking_available = False

    if thinking_available:
        ChatTemplate.set_thinking(llm, True)
        result.update(capture(llm, "thinking_on"))
    else:
        result["thinking_on"] = "<model does not support thinking>"

    settings.write_capture(args.out, result)

    off = result["thinking_off"]
    print(f"model         : {result['model']}")
    print(f"architecture  : {off['architecture']}")
    print(f"thinking      : supported={off['supports_thinking']} enabled={off['thinking_enabled']}")
    print(f"stop tokens   : {off['stop_tokens']}")
    print(f"prompt chars  : {len(off['rendered_prompt'])}")
    print(f"token counts  : {off['token_counts']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

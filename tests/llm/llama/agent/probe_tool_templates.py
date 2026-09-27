#!/usr/bin/env python3
"""CS-21: learn each architecture's native tool-call syntax from its own chat template.

Why this exists
---------------
The tool-loop harness stubs the model call and returns a canned *tool call*. For that to
test anything, the canned text must be exactly what the real model would emit - and the
authoritative source for that is the GGUF's embedded chat template, not documentation. A
template renders a PAST assistant tool call the same way the model is trained to produce
one, so rendering a conversation that contains one shows the syntax to parse.

For every model given, this renders three conversations through the embedded template
(CPU-only, weights mmapped, nothing generated):

* ``tools_offered``   - system + user, with a tool definition passed as ``tools=``.
                        Shows where and how the definitions land in the prompt.
* ``tool_round_trip`` - the same, plus an assistant turn carrying ``tool_calls`` and a
                        ``tool`` role result, ending on a fresh generation prompt. Shows
                        how a call and its result are spelled.
* ``no_tools``        - the plain conversation, as a control: diffing it against
                        ``tools_offered`` isolates exactly what ``tools=`` adds.

It also records whether ``Jinja2ChatFormatter`` passes ``tools`` through to the template
at all - if it did not, ``tools_offered`` would equal ``no_tools``.

Nothing here writes prompt content anywhere but the ``--out`` file, and the prompts are
synthetic.

Usage (llama env)
-------------------------
    python probe_tool_templates.py --out /tmp/tool_templates.json MODEL [MODEL ...]   # names inside model_dir, or paths
"""
import argparse
import json
import os
import sys

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()

# llama_utils must be imported before llama_cpp (CUDA device-order race; see its header).
import amadeo_utils.ai.llm.llama.llama_utils  # noqa: F401
from amadeo_utils.ai.llm.llama import chat_template as ChatTemplate
from llama_cpp import Llama

SYSTEM = "You are Ada, a guide. Be concise."
USER = "What time is it in UTC?"

# One small tool in OpenAI function form, which is what every one of these templates expects.
TOOL = {
    "type": "function",
    "function": {
        "name": "get_datetime",
        "description": "Returns the current date and time.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "IANA timezone name, e.g. UTC"},
            },
            "required": ["timezone"],
        },
    },
}

# Arguments as a dict, NOT the OpenAI-style JSON string: the Muse-Glimmer and Qwen 3.6
# templates raise on a string ("requires tool_call.function.arguments to be a dict"), and
# Gemma 4 renders one as a doubly-braced literal. Found by the first run of this probe.
CALL = {
    "id": "call00000",   # Mistral templates demand exactly 9 alphanumerics
    "type": "function",
    "function": {"name": "get_datetime", "arguments": {"timezone": "UTC"}},
}


def render_all(model_path):
    """Render the three probe conversations through one model's embedded template.

    Args:
        model_path (str): Path to a GGUF.

    Returns:
        dict: Architecture, stop tokens and the three rendered prompts, or an 'error' key.
    """
    llm = Llama(model_path=model_path, n_gpu_layers=0, n_ctx=2048, verbose=False)
    try:
        formatter = ChatTemplate.ChatTemplateFormatter(llm)
    except ValueError as e:
        return {"error": str(e)}

    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    round_trip = base + [
        {"role": "assistant", "content": "", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call00000", "name": "get_datetime",
         "content": json.dumps({"utc": "2026-09-23T14:00:00Z"})},
    ]

    out = {"architecture": ChatTemplate.model_architecture(llm),
           "stop_tokens": ChatTemplate.stop_tokens(llm)}
    for label, messages, kwargs in (("no_tools", base, {}),
                                    ("tools_offered", base, {"tools": [TOOL]}),
                                    ("tool_round_trip", round_trip, {"tools": [TOOL]})):
        try:
            out[label] = formatter.render(messages, **kwargs)
        except Exception as e:           # a template that rejects tools is itself a finding
            out[label] = f"RAISED {type(e).__name__}: {e}"
    out["tools_reach_template"] = out["tools_offered"] != out["no_tools"]
    return out


def main():
    parser = argparse.ArgumentParser(description="CS-21 tool-template probe")
    parser.add_argument("--out", required=True)
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()

    result = {}
    for arg in args.models:
        path = settings.resolve_model(arg)          # a name inside model_dir, or a path
        name = os.path.basename(path)
        print(f"rendering {name} ...", flush=True)
        result[name] = render_all(path)

    # Error messages quote the model's path; record it as <model_dir>/NAME, never this machine's layout.
    settings.write_capture(args.out, result)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

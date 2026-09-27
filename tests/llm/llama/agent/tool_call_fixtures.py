#!/usr/bin/env python3
"""CS-21: canned raw model outputs for the tool loop, per architecture, with expected parses.

What this is
------------
The tool-loop harness stubs the model call. Whatever the stub returns is what the loop
parses, so the canned text has to be exactly what each model really emits - otherwise the
harness would only prove the parser agrees with itself. This module is that canned text,
in one place, for the three architectures in the first cut (owner's decision, 2026-09-23):
Muse-Glimmer (ATEM / Harmony), Qwen 3.6 and Gemma 4.

Each fixture is the RAW generated text: what ``create_chat_completion`` returns in
``choices[0].message.content``, before any reasoning is stripped. Stop strings are absent,
because llama.cpp does not include the matched stop in its output.

Where the text comes from
-------------------------
Every fixture is marked with its provenance:

* ``template`` - the call segment was rendered by the model's own GGUF chat template (a
  past assistant tool call is rendered the way the model is trained to emit one), and
  ``--check`` re-verifies it against the template. These are authoritative.
* ``inferred`` - built from template pieces but never seen whole: prose-then-call,
  truncated calls, calls to unknown tools. Plausible, not proven.
* ``live`` - this exact form was produced by the real model in ``live_tool_smoke.py``
  (2026-09-23, on the 5090). The first live run corrected one inferred form: Muse-Glimmer goes
  from deliberation straight to the next '<|start|>', with no '<|eom|>' between.

Expected results
----------------
Every case has one architecture-independent expectation in ``EXPECTED``:

* ``{"kind": "answer", "text": ...}`` - no tool call; this is the user-facing reply.
* ``{"kind": "calls", "calls": [...]}`` - the calls, with arguments coerced to the types
  in the tool's JSON schema (so Gemma's bare ``14`` and ATEM's ``"14"`` both become ``14``).
* ``{"kind": "malformed"}`` - a call was started but cannot be parsed. The loop must
  surface it as a failure, never execute a partial call, and never show it as an answer.

``unknown_tool`` parses to a perfectly well-formed call to a tool that is never offered.
Rejecting it is the loop's job, not the parser's; see the flag checks in the CS-21 design notes.

Usage
-----
    python tool_call_fixtures.py            # print a summary (runs anywhere, no llama_cpp)
    python tool_call_fixtures.py --check    # re-verify 'template' fixtures (llama env)
"""
import argparse
import os
import sys

# --------------------------------------------------------------------------------------------------
# The tool schemas the fixtures call. Kept deliberately tiny; argument coercion is checked
# against these, so 'days' being an integer here is what makes the mixed-type case meaningful.
# --------------------------------------------------------------------------------------------------

TOOLS = [
    {"type": "function", "function": {
        "name": "get_datetime",
        "description": "Returns the current date and time.",
        "parameters": {"type": "object",
                       "properties": {"timezone": {"type": "string"}},
                       "required": ["timezone"]}}},
    {"type": "function", "function": {
        "name": "get_grades",
        "description": "Returns recent grades.",
        "parameters": {"type": "object",
                       "properties": {"term": {"type": "string"}, "days": {"type": "integer"}},
                       "required": ["term"]}}},
]

ANSWER = "It is 14:00 UTC."

_UTC = {"name": "get_datetime", "arguments": {"timezone": "UTC"}}
_NY = {"name": "get_datetime", "arguments": {"timezone": "America/New_York"}}
_GRADES = {"name": "get_grades", "arguments": {"term": "Q1", "days": 14}}
_ROGUE = {"name": "run_shell", "arguments": {"command": "rm -rf ~"}}

# One expectation per case, shared by every architecture.
EXPECTED = {
    "answer":                 {"kind": "answer", "text": ANSWER},
    "answer_after_reasoning": {"kind": "answer", "text": ANSWER},
    "single_call":            {"kind": "calls", "calls": [_UTC]},
    "call_after_reasoning":   {"kind": "calls", "calls": [_UTC]},
    "prose_then_call":        {"kind": "calls", "calls": [_UTC]},
    "mixed_types":            {"kind": "calls", "calls": [_GRADES]},
    "two_calls":              {"kind": "calls", "calls": [_UTC, _NY]},
    "unknown_tool":           {"kind": "calls", "calls": [_ROGUE]},
    "truncated_call":         {"kind": "malformed"},
}

# Every case also records whether reasoning was on for that generation, because it changes
# how the raw text must be read (Qwen's prompt opens the think block itself when it is on).
THINKING = {case: case.endswith("_after_reasoning") for case in EXPECTED}


# --------------------------------------------------------------------------------------------------
# Muse-Glimmer - ATEM calls inside Harmony messages.
#
# The prompt ends '<|start|>assistant', so the model's first output is the header: ' to=NAME'
# for a call, ' to=self' for deliberation, ' to=user' for the reply. Two calls are two
# separate messages joined by '<|eom|><|start|>assistant' (template-rendered). A call is
# addressed to the TOOL, which is why strip_reasoning() erases it - see the CS-21 design notes.
# --------------------------------------------------------------------------------------------------

def _atem(name, params):
    """One ATEM message addressed to a tool, exactly as the Muse-Glimmer template spells it."""
    body = "".join(f'<atem:parameter name="{k}">{v}</atem:parameter>\n' for k, v in params)
    return (f" to={name}<|message|><atem:function_calls>\n<atem:invoke name=\"{name}\">\n"
            f"{body}</atem:invoke>\n</atem:function_calls>")


# Live form (CS-21 live run): no '<|eom|>' - deliberation runs straight into the next '<|start|>'.
_MUSE_THINK = " to=self<|message|>The user wants the time; I should call the tool.<|start|>assistant"

MUSE = {
    "answer":                 (" to=user<|message|>" + ANSWER, "inferred"),
    "answer_after_reasoning": (_MUSE_THINK + " to=user<|message|>" + ANSWER, "live"),
    "single_call":            (_atem("get_datetime", [("timezone", "UTC")]), "template"),
    "call_after_reasoning":   (_MUSE_THINK + _atem("get_datetime", [("timezone", "UTC")]), "live"),
    "prose_then_call":        (" to=user<|message|>Let me check.<|eom|><|start|>assistant"
                               + _atem("get_datetime", [("timezone", "UTC")]), "inferred"),
    "mixed_types":            (_atem("get_grades", [("term", "Q1"), ("days", 14)]), "template"),
    "two_calls":              (_atem("get_datetime", [("timezone", "UTC")])
                               + "<|eom|><|start|>assistant"
                               + _atem("get_datetime", [("timezone", "America/New_York")]), "template"),
    "unknown_tool":           (_atem("run_shell", [("command", "rm -rf ~")]), "inferred"),
    "truncated_call":         (' to=get_datetime<|message|><atem:function_calls>\n<atem:invoke name="get_datetime">\n'
                               '<atem:parameter name="timezone">UT', "inferred"),
}


# --------------------------------------------------------------------------------------------------
# Qwen 3.6 - '<tool_call><function=NAME><parameter=P>' with values on their own lines.
#
# With reasoning OFF the prompt already contains '<think>\n\n</think>\n\n', so the output
# starts at the call. With reasoning ON the prompt ends '<think>\n' and the output starts
# INSIDE the think block, with no opening tag of its own. Two calls are two consecutive
# <tool_call> blocks separated by a newline (template-rendered). The template's own
# instructions permit prose before a call but not after it.
# --------------------------------------------------------------------------------------------------

def _qwen(name, params):
    """One Qwen 3.6 tool-call block, exactly as its template spells it."""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params)
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


_QWEN_THINK = "The user wants the time; I should call the tool.\n</think>\n\n"

QWEN = {
    "answer":                 (ANSWER, "live"),
    "answer_after_reasoning": (_QWEN_THINK + ANSWER, "live"),
    "single_call":            (_qwen("get_datetime", [("timezone", "UTC")]), "template"),
    "call_after_reasoning":   (_QWEN_THINK + _qwen("get_datetime", [("timezone", "UTC")]), "live"),
    "prose_then_call":        ("Let me check.\n\n" + _qwen("get_datetime", [("timezone", "UTC")]), "inferred"),
    "mixed_types":            (_qwen("get_grades", [("term", "Q1"), ("days", 14)]), "template"),
    "two_calls":              (_qwen("get_datetime", [("timezone", "UTC")]) + "\n"
                               + _qwen("get_datetime", [("timezone", "America/New_York")]), "template"),
    "unknown_tool":           (_qwen("run_shell", [("command", "rm -rf ~")]), "inferred"),
    "truncated_call":         ("<tool_call>\n<function=get_datetime>\n<parameter=timezone>\nUT", "inferred"),
}


# --------------------------------------------------------------------------------------------------
# Gemma 4 - '<|tool_call>call:NAME{...}<tool_call|>'. NOT JSON: keys are bare, strings are
# wrapped in the '<|"|>' token, numbers are bare, and the template sorts keys alphabetically
# (template-rendered: 'days' before 'term'). Two calls are simply adjacent blocks.
#
# With reasoning OFF the prompt ends with an empty, already-closed thought channel. With it ON
# the prompt ends '<|turn>model\n' and the output opens '<|channel>thought\n'.
# --------------------------------------------------------------------------------------------------

def _gemma(name, params):
    """One Gemma 4 tool-call block, exactly as its template spells it (keys sorted)."""
    def value(v):
        return str(v) if isinstance(v, (int, float)) else f'<|"|>{v}<|"|>'
    body = ",".join(f"{k}:{value(v)}" for k, v in sorted(params))
    return f"<|tool_call>call:{name}{{{body}}}<tool_call|>"


_GEMMA_THINK = "<|channel>thought\nThe user wants the time; I should call the tool.<channel|>"

GEMMA = {
    "answer":                 (ANSWER, "live"),
    "answer_after_reasoning": (_GEMMA_THINK + ANSWER, "inferred"),
    "single_call":            (_gemma("get_datetime", [("timezone", "UTC")]), "template"),
    "call_after_reasoning":   (_GEMMA_THINK + _gemma("get_datetime", [("timezone", "UTC")]), "inferred"),
    "prose_then_call":        ("Let me check.\n" + _gemma("get_datetime", [("timezone", "UTC")]), "inferred"),
    "mixed_types":            (_gemma("get_grades", [("term", "Q1"), ("days", 14)]), "template"),
    "two_calls":              (_gemma("get_datetime", [("timezone", "UTC")])
                               + _gemma("get_datetime", [("timezone", "America/New_York")]), "template"),
    "unknown_tool":           (_gemma("run_shell", [("command", "rm -rf ~")]), "inferred"),
    "truncated_call":         ('<|tool_call>call:get_datetime{timezone:<|"|>UT', "inferred"),
}


# Keyed by the GGUF 'general.architecture' string, matching chat_template._SCHEMES.
FIXTURES = {"muse-glimmer": MUSE, "qwen35moe": QWEN, "gemma4": GEMMA}

# The model each architecture is checked against (names inside model_dir).
CHECK_MODELS = {
    "muse-glimmer": "Muse-Glimmer-30B-Abliterated-Q8_0.gguf",      # names inside model_dir
    "qwen35moe": "Qwen3.6-35B-A3B-Aggressive-Q4.gguf",
    "gemma4": "gemma-4-31B-it-abliterated.gguf",
}


def _as_template_calls(expected_calls):
    """Converts EXPECTED calls into the 'tool_calls' shape a chat template renders.

    Arguments stay a dict: the Muse-Glimmer and Qwen 3.6 templates raise on the OpenAI JSON
    string form. Ids are nine alphanumerics, which Mistral-family templates demand.
    """
    return [{"id": f"call{i:05d}", "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"]}}
            for i, c in enumerate(expected_calls)]


def check_against_templates():
    """Re-renders every 'template' fixture through its model's own template and compares.

    For each such case, a conversation ending in an assistant turn that carries the expected
    calls is rendered with reasoning off. The fixture's raw text, minus its leading
    whitespace, must appear verbatim in that render. Runs CPU-only; nothing is generated.

    Returns:
        int: The number of fixtures that did not match.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    from testlib import settings
    settings.use_src()
    # llama_utils must be imported before llama_cpp (CUDA device-order race; see its header).
    import amadeo_utils.ai.llm.llama.llama_utils  # noqa: F401
    from amadeo_utils.ai.llm.llama import chat_template as ChatTemplate
    from llama_cpp import Llama

    base = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    failures = 0
    for arch, fixtures in FIXTURES.items():
        llm = Llama(model_path=settings.model_path(CHECK_MODELS[arch]), n_gpu_layers=0, n_ctx=512, verbose=False)
        formatter = ChatTemplate.ChatTemplateFormatter(llm)
        think_off = ChatTemplate.scheme_for(llm)["think_off"]
        for case, (raw, source) in fixtures.items():
            if source != "template":
                continue
            calls = _as_template_calls(EXPECTED[case]["calls"])
            rendered = formatter.render(base + [{"role": "assistant", "content": "", "tool_calls": calls}],
                                        tools=TOOLS, **think_off)
            ok = raw.strip() in rendered
            failures += 0 if ok else 1
            print(f"  {'PASS' if ok else 'FAIL'}  {arch}/{case}")
        del llm
    return failures


def main():
    parser = argparse.ArgumentParser(description="CS-21 tool-call fixtures")
    parser.add_argument("--check", action="store_true",
                        help="re-verify the 'template' fixtures against the real templates (llama env)")
    args = parser.parse_args()

    # Every architecture must cover every case, or a scheme silently goes untested.
    for arch, fixtures in FIXTURES.items():
        missing = set(EXPECTED) - set(fixtures)
        extra = set(fixtures) - set(EXPECTED)
        assert not missing and not extra, f"{arch}: missing {missing}, unexpected {extra}"

    if args.check:
        failures = check_against_templates()
        print("all template fixtures match" if failures == 0 else f"{failures} fixture(s) drifted")
        sys.exit(failures)

    for arch, fixtures in FIXTURES.items():
        by_source = {}
        for _, source in fixtures.values():
            by_source[source] = by_source.get(source, 0) + 1
        print(f"{arch:14} {len(fixtures)} cases  {by_source}")


if __name__ == "__main__":
    main()

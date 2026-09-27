#!/usr/bin/env python3
"""CS-21: the per-architecture tool-call parsers against the fixtures and edge cases.

What it proves
--------------
``chat_template.parse_tool_calls`` turns each fixture in ``tool_call_fixtures.FIXTURES``
into exactly the result in ``tool_call_fixtures.EXPECTED`` - for all three architectures,
nine cases each - with argument types coerced from the offered schemas. Then a set of edge
cases the fixtures do not reach: Gemma's nested and quoted values, Qwen's Hermes-JSON
fallback, a call drafted inside reasoning (not a call), Muse-Glimmer's namespaced names,
and an architecture with no tool protocol.

Runs anywhere: chat_template does not import llama_cpp at module scope, and every call
passes ``architecture=`` instead of a loaded model.

Usage
-----
    python check_tool_parsers.py      # exits non-zero on any failure
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()

from amadeo_utils.ai.llm.llama import chat_template as CT   # noqa: E402
import tool_call_fixtures as F                               # noqa: E402

failures = []


def check(ok, label, detail=""):
    """Record and print one result."""
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if not ok and detail else ""))
    if not ok:
        failures.append(label)


def as_expected(parse):
    """Reduce a ToolParse to the EXPECTED dict shape for comparison."""
    if parse.kind == CT.TOOL_PARSE_ANSWER:
        return {"kind": "answer", "text": parse.text}
    if parse.kind == CT.TOOL_PARSE_CALLS:
        return {"kind": "calls", "calls": parse.calls}
    return {"kind": parse.kind}


def fixture_checks():
    print("== fixtures: every case, every architecture ==")
    for arch, fixtures in F.FIXTURES.items():
        for case, (raw, _source) in fixtures.items():
            got = as_expected(CT.parse_tool_calls(raw, tools=F.TOOLS, thinking=F.THINKING[case], architecture=arch))
            check(got == F.EXPECTED[case], f"{arch}/{case}", f"got {got}")


def edge_checks():
    print("== edge cases ==")

    def parse(raw, arch, tools=F.TOOLS, thinking=False):
        return CT.parse_tool_calls(raw, tools=tools, thinking=thinking, architecture=arch)

    # Gemma: nested object, array, bool, null, float, and a string holding the characters
    # its own syntax uses (comma, colon, braces, a quote) - legal because strings are token-wrapped.
    q = '<|"|>'
    gemma = (f"<|tool_call>call:f{{a:[{q}x{q},{q}y{q}],b:true,c:null,d:-2.5,"
             f"e:{{f:{q}g, h: {{i}} \"j\"{q}}}}}<tool_call|>")
    got = parse(gemma, "gemma4", tools=[])
    check(got.kind == "calls" and got.calls[0]["arguments"] ==
          {"a": ["x", "y"], "b": True, "c": None, "d": -2.5, "e": {"f": 'g, h: {i} "j"'}},
          "gemma4: nested / array / bool / null / float / syntax chars inside a string", f"got {got}")

    got = parse('<|tool_call>call:f{a:<|"|>unterminated}<tool_call|>', "gemma4")
    check(got.kind == "malformed", "gemma4: unterminated string inside a closed block is malformed", f"got {got}")

    got = parse('<|tool_call>call:get_grades{days:<|"|>14<|"|>,term:<|"|>Q1<|"|>}<tool_call|>', "gemma4")
    check(got.calls == [{"name": "get_grades", "arguments": {"term": "Q1", "days": 14}}],
          "gemma4: a quoted integer is coerced by schema", f"got {got}")

    got = parse('<|tool_call>call:get_datetime{timezone:5}<tool_call|>', "gemma4")
    check(got.calls == [{"name": "get_datetime", "arguments": {"timezone": "5"}}],
          "gemma4: a bare number for a string parameter becomes a string", f"got {got}")

    # Captured LIVE from Gemma 4 with reasoning on (CS-21): it imitated the history marker instead
    # of calling the tool - no '<|tool_call>call:' opener, but the closer is there. Must not be an answer.
    live = ('<|channel>thought\nThe user is asking for the square root of (2 * pi).\nI will use the `calculator` '
            'tool.<channel|>[TOOL CALL - not from the user] calculator{expression:<|"|>sqrt(2 * pi)<|"|>}<tool_call|>')
    got = parse(live, "gemma4", thinking=True)
    check(got.kind == "malformed", "gemma4 (live capture): marker imitation with an orphaned closer is malformed",
          f"got {got}")
    got = parse("Here you go.\n</tool_call>", "qwen35moe")
    check(got.kind == "malformed", "qwen35moe: an orphaned </tool_call> is malformed", f"got {got}")

    # Qwen: Hermes JSON inside <tool_call> is accepted.
    got = parse('<tool_call>\n{"name": "get_grades", "arguments": {"term": "Q1", "days": "14"}}\n</tool_call>', "qwen35moe")
    check(got.calls == [{"name": "get_grades", "arguments": {"term": "Q1", "days": 14}}],
          "qwen35moe: Hermes-JSON fallback, with coercion", f"got {got}")

    got = parse('<tool_call>\nnot a call at all\n</tool_call>', "qwen35moe")
    check(got.kind == "malformed", "qwen35moe: closed block with unparseable body is malformed", f"got {got}")

    # Qwen: a call drafted INSIDE reasoning is deliberation, not a call.
    drafted = ("Maybe <tool_call>\n<function=get_datetime>\n<parameter=timezone>\nUTC\n</parameter>\n"
               "</function>\n</tool_call> but no, I know it.\n</think>\n\n" + F.ANSWER)
    got = parse(drafted, "qwen35moe", thinking=True)
    check(got.kind == "answer" and got.text == F.ANSWER, "qwen35moe: a call drafted inside <think> is ignored",
          f"got {got}")

    # Qwen: multi-line parameter value keeps its inner newlines, loses only the framing ones.
    got = parse("<tool_call>\n<function=f>\n<parameter=body>\nline one\nline two\n</parameter>\n</function>\n</tool_call>",
                "qwen35moe", tools=[])
    check(got.calls == [{"name": "f", "arguments": {"body": "line one\nline two"}}],
          "qwen35moe: multi-line value keeps inner newlines", f"got {got}")

    # Qwen: reasoning on and budget ran out mid-thought - no answer, no call.
    got = parse("still thinking about <tool_call>", "qwen35moe", thinking=True)
    check(got.kind == "answer" and got.text == "", "qwen35moe: truncated reasoning yields an empty answer",
          f"got {got}")

    # Gemma: a call inside the thought channel is ignored.
    got = parse('<|channel>thought\n<|tool_call>call:get_datetime{timezone:<|"|>UTC<|"|>}<tool_call|><channel|>'
                + F.ANSWER, "gemma4", thinking=True)
    check(got.kind == "answer" and got.text == F.ANSWER, "gemma4: a call inside the thought channel is ignored",
          f"got {got}")

    # Muse-Glimmer: namespaced name resolves to an offered tool; unoffered namespace stays as-is.
    ns = (' to=functions.get_datetime<|message|><atem:function_calls>\n<atem:invoke name="functions.get_datetime">\n'
          '<atem:parameter name="timezone">UTC</atem:parameter>\n</atem:invoke>\n</atem:function_calls>')
    got = parse(ns, "muse-glimmer")
    check(got.calls == [{"name": "get_datetime", "arguments": {"timezone": "UTC"}}],
          "muse-glimmer: 'functions.get_datetime' resolves to the offered get_datetime", f"got {got}")
    got = parse(ns.replace("get_datetime", "rm_rf"), "muse-glimmer")
    check(got.calls and got.calls[0]["name"] == "functions.rm_rf",
          "muse-glimmer: an unoffered namespaced name is left for the caller to refuse", f"got {got}")

    # Muse-Glimmer: two invokes inside ONE function_calls block.
    both = (' to=get_datetime<|message|><atem:function_calls>\n'
            '<atem:invoke name="get_datetime">\n<atem:parameter name="timezone">UTC</atem:parameter>\n</atem:invoke>\n'
            '<atem:invoke name="get_datetime">\n<atem:parameter name="timezone">America/New_York</atem:parameter>\n</atem:invoke>\n'
            '</atem:function_calls>')
    got = parse(both, "muse-glimmer")
    check(len(got.calls) == 2, "muse-glimmer: two invokes in one block are two calls", f"got {got}")

    # Muse-Glimmer: a tool-addressed header with nothing after it (budget ran out) is malformed.
    # The template's own recipient notation: '# Valid recipients: "self", "get_datetime.*", "user"'. Muse copies the
    # '.*' into the call (seen live 2026-09-25: 9 refused calls in one turn, then the round cap).
    glob = (' to=get_datetime.*<|message|><atem:function_calls>\n<atem:invoke name="get_datetime.*">\n'
            '<atem:parameter name="timezone">UTC</atem:parameter>\n</atem:invoke>\n</atem:function_calls>')
    got = parse(glob, "muse-glimmer")
    check(got.kind == "calls" and got.calls[0]["name"] == "get_datetime" and got.calls[0]["arguments"] == {"timezone": "UTC"},
          "muse-glimmer: the template's 'name.*' notation resolves to the tool", f"got {got}")
    got = parse(glob.replace("get_datetime", "rm_rf"), "muse-glimmer")
    check(got.kind == "calls" and got.calls[0]["name"] == "rm_rf",
          "muse-glimmer: 'unoffered.*' becomes the bare unoffered name, for the caller to refuse", f"got {got}")
    got = parse(" to=get_datetime.*<|message|>", "muse-glimmer")
    check(got.kind == "malformed", "muse-glimmer: a 'name.*' header with no call body is malformed", f"got {got}")

    got = parse(" to=get_datetime<|message|>", "muse-glimmer")
    check(got.kind == "malformed", "muse-glimmer: tool header with no call body is malformed", f"got {got}")

    # Muse-Glimmer: deliberation that merely MENTIONS a tool in prose is not a call.
    got = parse(" to=self<|message|>I could use get_datetime here.<|eom|><|start|>assistant to=user<|message|>"
                + F.ANSWER, "muse-glimmer", thinking=True)
    check(got.kind == "answer" and got.text == F.ANSWER, "muse-glimmer: mentioning a tool in reasoning is not a call",
          f"got {got}")

    # A model with no tool protocol: everything is an answer, even text that looks like a call.
    got = parse('<tool_call>\n{"name": "get_datetime", "arguments": {}}\n</tool_call>', "gemma3")
    check(got.kind == "answer", "gemma3 (no tool scheme): output is always an answer", f"got {got}")
    check(CT.tool_scheme_for(architecture="gemma3") is None and CT.tool_scheme_for(architecture="llama") is None,
          "no tool scheme for gemma3 / llama in the first cut")


def main():
    fixture_checks()
    edge_checks()
    print()
    print("ALL CHECKS PASS" if not failures else f"{len(failures)} check(s) FAILED")
    sys.exit(len(failures))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CS-21: prove that only the tool family can offer the model tools.

What it proves
--------------
the CS-21 design notes's hard constraint: "role-play never uses tools" must be true by construction.
After the generate_raw() split that means four things, each checked here:

1. **Static - the two existing families cannot reach tools.** Parsing (not grepping)
   RolePlayStream.py and KnowledgeBaseStream.py: neither calls ``generate_raw``, neither
   calls ``create_chat_completion``, and no call in either passes a ``tools=`` keyword.
2. **Static - generation has one home.** ``create_chat_completion`` is called exactly once
   in StreamBase.py, inside ``generate_raw``.
3. **Static - generate_once has no way in.** Its signature has no ``tools`` parameter.
4. **Runtime - tools are forwarded only when set.** With the model call stubbed and no model
   loaded, ``generate_once`` and ``generate_raw(tools=None)`` send exactly the keyword set
   the pre-CS-21 code sent, and ``generate_raw(tools=[...])`` adds ``tools`` and nothing
   else (in particular never ``tool_choice``, which would switch llama-cpp-python into
   grammar-constrained output). It also checks generate_raw returns the text UNSTRIPPED,
   using a real Muse-Glimmer tool call, which strip_reasoning() would erase.

Checks 1-3 run anywhere. Check 4 imports StreamBase, which imports llama_cpp, so it runs in
the llama env; elsewhere it is reported as skipped rather than passed.

Usage
-----
    python check_tools_isolation.py       # exits non-zero on any failure
"""
import ast
import os
import sys
import threading

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()
SRC = settings.SRC
LLAMA_DIR = os.path.join(SRC, "amadeo_utils", "ai", "llm", "llama")
FAMILIES = ("RolePlayStream.py", "KnowledgeBaseStream.py")

# The exact keywords generate_once passed to create_chat_completion before CS-21. If this
# set changes for the non-tool path, role-play's call to the model has changed.
PRE_CS21_KWARGS = {"messages", "max_tokens", "stream", "repeat_penalty", "stop"}

failures = []


def check(ok, label):
    """Record and print one result."""
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        failures.append(label)


def calls_in(tree):
    """Yield (callee_name, call_node) for every call whose callee is a name or attribute."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name:
                yield name, node


def parse(filename):
    with open(os.path.join(LLAMA_DIR, filename), encoding="utf-8") as fh:
        return ast.parse(fh.read())


def static_checks():
    print("== static: the existing families cannot reach tools ==")
    for filename in FAMILIES:
        tree = parse(filename)
        calls = list(calls_in(tree))
        check(not any(n == "generate_raw" for n, _ in calls), f"{filename}: never calls generate_raw")
        check(not any(n == "create_chat_completion" for n, _ in calls), f"{filename}: never calls create_chat_completion")
        check(not any(kw.arg == "tools" for _, c in calls for kw in c.keywords), f"{filename}: no call passes tools=")

    print("== static: generation has exactly one home ==")
    tree = parse("StreamBase.py")
    ccc = []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            ccc += [fn.name for n, _ in calls_in(fn) if n == "create_chat_completion"]
    check(ccc == ["generate_raw"], f"StreamBase.py: create_chat_completion called only in generate_raw (found in {ccc})")

    # The tool family is the one class allowed to offer tools - but only through generate_raw,
    # never by calling the model itself.
    tool_calls = [n for n, _ in calls_in(parse("ToolStream.py"))]
    check("create_chat_completion" not in tool_calls and "generate_raw" in tool_calls,
          "ToolStream.py: reaches the model only through generate_raw")

    once = next(fn for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef) and fn.name == "generate_once")
    params = [a.arg for a in once.args.args + once.args.kwonlyargs]
    check("tools" not in params and once.args.kwarg is None,
          "StreamBase.generate_once: no tools parameter and no **kwargs")


# Real Muse-Glimmer output for a tool call (tool_call_fixtures.MUSE['single_call']).
MUSE_CALL = (' to=get_datetime<|message|><atem:function_calls>\n<atem:invoke name="get_datetime">\n'
             '<atem:parameter name="timezone">UTC</atem:parameter>\n</atem:invoke>\n</atem:function_calls>')


def runtime_checks():
    print("== runtime: tools are forwarded only when set (model call stubbed) ==")
    sys.path.insert(0, SRC)
    try:
        from amadeo_utils.ai.llm.llama.StreamBase import StreamBase
    except ImportError as e:
        print(f"  SKIP  runtime checks - cannot import StreamBase here ({e}); run in the llama env")
        return

    # A StreamBase with no model: only what generate_raw touches. Same technique as
    # CS-20's golden_dispatch.py (object.__new__, attributes set by hand).
    class FakeGenerator:
        """Records each call's keywords; answers with a canned Muse-Glimmer tool call."""
        def __init__(self):
            self.calls = []
            # set_thinking() / split_stops() read the architecture from here.
            self.metadata = {"general.architecture": "muse-glimmer"}

        def create_chat_completion(self, **kwargs):
            self.calls.append(kwargs)
            return {"choices": [{"message": {"content": MUSE_CALL}}]}

    stream = object.__new__(StreamBase)
    stream.llm_generator = FakeGenerator()
    stream.generating_gpu_lock = threading.Lock()
    stream.models_released = False
    stream.architecture_stops = ["<|eot|>", "<|return|>"]
    stream.thinking_supported = True
    stream.argsDict = {"generating_max_context_tokens": 2048, "repeat_penalty": 1.1}

    # set_thinking needs an installed handler; there is no model, so neutralise it for this test only.
    import amadeo_utils.ai.llm.llama.chat_template as CT
    CT.set_thinking = lambda llm, thinking: False

    tool = {"type": "function", "function": {"name": "get_datetime", "parameters": {"type": "object"}}}
    args = ([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], [], 64, False, "sess", 0)

    answer = stream.generate_once(*args)
    check(set(stream.llm_generator.calls[-1]) == PRE_CS21_KWARGS,
          f"generate_once sends exactly the pre-CS-21 keywords {sorted(PRE_CS21_KWARGS)}")
    check(answer == "", "generate_once strips a Muse-Glimmer tool call to '' (the reason generate_raw exists)")

    raw, _ = stream.generate_raw(*args)
    check(set(stream.llm_generator.calls[-1]) == PRE_CS21_KWARGS, "generate_raw(tools=None) sends no tools keyword")

    raw, _ = stream.generate_raw(*args, tools=[tool])
    sent = stream.llm_generator.calls[-1]
    check(set(sent) == PRE_CS21_KWARGS | {"tools"} and sent["tools"] == [tool],
          "generate_raw(tools=[...]) adds tools and nothing else (no tool_choice)")
    check(raw == MUSE_CALL, "generate_raw returns the tool call unstripped")


def main():
    static_checks()
    runtime_checks()
    print()
    print("ALL CHECKS PASS" if not failures else f"{len(failures)} check(s) FAILED")
    sys.exit(len(failures))


if __name__ == "__main__":
    main()

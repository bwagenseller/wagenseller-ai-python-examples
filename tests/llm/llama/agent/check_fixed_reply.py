#!/usr/bin/env python3
"""CS-24: a tool's fixed reply ("Done.") replaces the model's words when the whole turn succeeded with one.

Why
---
Brent, 2026-10-17: "just have the agent simply say 'done' if it was successful - or explain the issue if something
happened". Asking the model doesn't hold: the personas dress every answer up ("Ho ho ho! I have turned on both..."),
and they ignore prompt rules. So a tool can return a fixed reply with a clean success (home_control: only when nothing
was skipped, failed or clamped), and the loop uses it as the answer when every call in the turn gave one.

What it proves
--------------
With the model call stubbed (see check_call_check.make_stream):
* one successful call with a reply: the answer is the reply, written and spoken, and that's what the history saves;
* two such calls: the reply is said once;
* the model's own answer is kept when any call in the turn had no reply (a status, the weather), failed, or was
  refused; when there were no calls; and in a worker;
* through the real script contract (tool_script_main -> load_script_tool -> execute), a ToolAnswer's reply reaches
  ToolOutcome.reply, and a tool that gives none leaves it None.

Usage (llama env - ToolStream imports llama_cpp):
    python check_fixed_reply.py
"""
import os
import sys
import tempfile
import textwrap
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # up to tests/
from testlib import settings  # noqa: E402
settings.use_src()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_call_check import make_stream  # noqa: E402

failures = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if not ok and detail else ""))
    if not ok:
        failures.append(label)


def main():
    try:
        from amadeo_utils.ai.llm.llama.ToolStream import ToolStream
        from amadeo_utils.ai.llm.llama import chat_template as ChatTemplate
        from amadeo_utils.ai.llm.llama.llama_utils import LlamaUtils
    except ImportError as e:            # llama_cpp is imported by the module; run this in the llama env
        print(f"needs the llama env: {e}")
        sys.exit(2)
    from amadeo_utils.ai.llm.tools import registry as R
    from amadeo_utils.ai.llm.tools.script_tools import load_script_tool

    # The real parser needs a model; the scripted replies are already parsed.
    ChatTemplate.parse_tool_calls = lambda raw, llm=None, tools=None, thinking=False: raw
    LlamaUtils.universal_token_count = staticmethod(lambda *a, **k: 1)

    def answer(text):
        return ChatTemplate.ToolParse(ChatTemplate.TOOL_PARSE_ANSWER, [], text)

    def call(*names):
        return ChatTemplate.ToolParse(ChatTemplate.TOOL_PARSE_CALLS, [{"name": n, "arguments": {}} for n in names], "")

    schema = {"type": "object", "properties": {}}
    registry = R.ToolRegistry()
    registry.register(R.Tool("house", "acts on the house", schema, lambda: "ok", physical=True))
    registry.register(R.Tool("weather", "forecast", schema, lambda: "ok"))
    policy = R.ToolPolicy(registry, ["house", "weather"], R.MODE_AUTO)
    messages = [{"role": "system", "content": "You are a test."}, {"role": "user", "content": "x"}]
    session = {"session_id": "check"}
    persona = "Ho ho ho! I have turned on both lamps for you!"

    def run(script, refuse=(), fail=(), worker=False):
        stream, _ = make_stream(ToolStream, ChatTemplate, script, refuse, fail, replies={"house": "Done."})
        result = stream.run_tool_loop(session, messages, policy, R.Taint(), False, 256, 0, 5, time.monotonic(),
                                      worker=worker, request="" if worker else "turn on the lamps")
        return stream, result

    # 1. One clean success: "Done." is the answer, shown, spoken and saved.
    stream, result = run([call("house"), answer(persona)])
    check(result.answer == "Done.", "one clean success: the answer is the tool's reply", result.answer)
    check(stream.compose_reply(result, spoken=True) == "Done.", "...and that is all that is read aloud",
          stream.compose_reply(result, spoken=True))
    saved = {"chat_history": [], "continuous_save": False,
             "db": type("DB", (), {"add_document": lambda self, q, a: None})()}
    stream.persist_turn(saved, "turn on the lamps", result)
    check(saved["chat_history"][-1]["content"] == "Done." and saved["chat_history"][1].get("tool_calls"),
          "the history saves the tool call, then 'Done.'", str(saved["chat_history"]))

    # 2. Two successful calls with the same reply: said once.
    _, result = run([call("house", "house"), answer(persona)])
    check(result.answer == "Done.", "two clean successes: 'Done.' once", result.answer)

    # 3. The model's words stay whenever anything needs them.
    _, result = run([call("house", "weather"), answer("Done, and it will rain.")])
    check(result.answer == "Done, and it will rain.", "a call with no reply in the same round: the model answers",
          result.answer)
    _, result = run([call("house"), call("weather"), answer("Done, and it will rain.")])
    check(result.answer == "Done, and it will rain.", "a call with no reply in a later round: the model answers",
          result.answer)
    _, result = run([call("house"), answer("The lamp didn't respond.")], fail=("house",))
    check(result.answer == "The lamp didn't respond.", "a failed call: the model explains", result.answer)
    _, result = run([call("house"), answer("I can't do that.")], refuse=("house",))
    check(result.answer == "I can't do that.", "a refused call: the model explains", result.answer)
    _, result = run([answer("Hello!")])
    check(result.answer == "Hello!", "no calls: the model answers", result.answer)
    _, result = run([call("house"), answer("worker answer")], worker=True)
    check(result.answer == "worker answer", "a worker's loop keeps its own answer", result.answer)

    # 4. The real script contract carries the reply.
    with tempfile.TemporaryDirectory() as tmp:
        def script(name, reply):
            path = os.path.join(tmp, f"{name}.py")
            with open(path, "w") as fh:
                fh.write(textwrap.dedent(f"""
                    from amadeo_utils.ai.llm.tools.script_tools import tool_script_main, ToolAnswer
                    DEFINITION = {{"name": "{name}", "description": "d", "parameters": {{"type": "object"}}}}
                    tool_script_main(DEFINITION, lambda arguments, config: ToolAnswer({{"done": 1}}, "did it", {reply!r}))
                """))
            return path

        outcome = R.execute(load_script_tool(sys.executable, script("with_reply", "Done.")), {}, 10000)
        check(outcome.ok and outcome.reply == "Done." and outcome.summary == "did it",
              "contract: a ToolAnswer reply reaches ToolOutcome.reply, summary intact", str(outcome))
        outcome = R.execute(load_script_tool(sys.executable, script("no_reply", None)), {}, 10000)
        check(outcome.ok and outcome.reply is None and outcome.summary == "did it",
              "contract: no reply -> None", str(outcome))

    print("\nALL CHECKS PASS" if not failures else f"\n{len(failures)} check(s) FAILED")
    sys.exit(len(failures))


if __name__ == "__main__":
    main()

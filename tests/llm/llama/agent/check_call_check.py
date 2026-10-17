#!/usr/bin/env python3
"""CS-24: the tool call check (registry.CallCheck) and what the tool loop does with it.

Why
---
Live, 2026-10-17: the voice agents said "the office lights are on" and "the doghouse light is on" without calling
home_control. Each made-up answer was saved to the history, and the next turn copied it. A tool can now declare
patterns for requests it must handle. When the model answers such a request without calling the tool, the loop asks
it once to reconsider. If it still doesn't call, the user is told and the saved answer says nothing ran.

What it proves
--------------
With the model call stubbed (scripted replies, no model file):
* a matching request answered without the tool gets exactly one nudge, and the nudge names the tool;
* if the model then calls the tool, nothing is reported as skipped;
* if it answers again without the tool, the tool is reported as skipped, the user gets the tool's notice (spoken
  too), and the history records that nothing ran;
* no nudge when the tool was called straight away, when the request doesn't match, when the tool isn't offered,
  when the call was refused, or when no request is given (a worker);
* load_script_tool reads 'call_check' from a script's definition, and refuses a malformed one or a bad pattern;
* home_control's own patterns match device requests and leave everyday make-believe alone.

Usage (llama env - ToolStream imports llama_cpp):
    python check_call_check.py
"""
import importlib.util
import os
import sys
import tempfile
import textwrap
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # up to tests/
from testlib import settings  # noqa: E402
settings.use_src()

failures = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if not ok and detail else ""))
    if not ok:
        failures.append(label)


def make_stream(ToolStream, ChatTemplate, script, refuse=(), fail=(), replies=None):
    """
    A ToolStream with no model: generate_raw plays 'script' (a list of ToolParse results, one per generation),
    parse_tool_calls passes them through, and _run_one_call records the call instead of running anything.
    Also used by check_fixed_reply.py.

    Args:
        refuse: Tool names whose calls are refused.
        fail: Tool names whose calls fail.
        replies: {tool name: fixed reply} for successful calls (see registry.Summarized).

    Returns:
        (stream, seen) - seen['prompts'] holds the messages each generation was given.
    """
    stream = ToolStream.__new__(ToolStream)
    stream.argsDict = {'generating_max_context_tokens': 100_000, 'buffer_context_pcnt': 0.1}
    stream._clock = time.monotonic
    stream.max_turn_seconds = 60
    stream.generating_gpu_lock = threading.Lock()
    stream.llm_generator = None
    stream.model_type = None
    stream.max_result_chars, stream.tool_result_share = 4000, 0.5
    stream.worker_notes = "off"
    stream._prompt_tokens = lambda messages, defs: 0
    scripted = list(script)
    seen = {'prompts': []}

    def generate_raw(messages, stops, max_tokens, reason, session_id, used_tokens, tools=None):
        seen['prompts'].append([dict(m) for m in messages])
        return scripted.pop(0), []

    def run_one_call(session, name, arguments, policy, taint, result, *rest):
        if name in refuse:
            result.refused.append(name)
            return "refused"
        result.executed.append(name)
        if name in fail:
            result.notices.append(f"The tool '{name}' failed: it broke")
            result.calls.append({"round": result.current_round, "name": name, "arguments": arguments, "result": "FAILED"})
            return "it broke"
        result.replies.append((replies or {}).get(name))      # as the real _run_one_call does on success
        result.calls.append({"round": result.current_round, "name": name, "arguments": arguments, "result": "done"})
        return "done"

    stream.generate_raw = generate_raw
    stream._run_one_call = run_one_call
    return stream, seen


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

    def call(name, **arguments):
        return ChatTemplate.ToolParse(ChatTemplate.TOOL_PARSE_CALLS, [{"name": name, "arguments": arguments}], "")

    schema = {"type": "object", "properties": {}}
    registry = R.ToolRegistry()
    registry.register(R.Tool("house", "acts on the house", schema, lambda: "ok", physical=True,
                             call_check=R.CallCheck([r"\bturn\w*(\W+\w+){0,5}?\W+(on|off)\b"],
                                                    "Nothing in the house was changed or checked.")))
    registry.register(R.Tool("calculator", "arithmetic", schema, lambda: "ok"))
    policy = R.ToolPolicy(registry, ["house", "calculator"], R.MODE_AUTO)
    no_house = R.ToolPolicy(registry, ["calculator"], R.MODE_AUTO)
    messages = [{"role": "system", "content": "You are a test."}, {"role": "user", "content": "x"}]
    session = {"session_id": "check"}

    def run(script, request, use_policy=policy, refuse=(), worker=False):
        stream, seen = make_stream(ToolStream, ChatTemplate, script, refuse)
        result = stream.run_tool_loop(session, messages, use_policy, R.Taint(), False, 256, 0, 5, time.monotonic(),
                                      worker=worker, request=request)
        return stream, result, seen

    # 1. Made up twice: one nudge, then reported as skipped.
    stream, result, seen = run([answer("The office lights are on!"), answer("Ho ho ho, they are on!")],
                               "turn on the office lights")
    check(result.generations == 2, "made up: exactly one nudge (two generations)", str(result.generations))
    nudge = seen['prompts'][1][-1]["content"] if len(seen['prompts']) > 1 else ""
    check(nudge.startswith(ToolStream.NOTICE_PREFIX) and "'house'" in nudge, "the nudge is a tool notice naming the tool", nudge)
    check(len(seen['prompts']) > 1 and seen['prompts'][1][-2] == {"role": "assistant", "content": "The office lights are on!"},
          "the nudge follows the model's own first answer")
    check(result.skipped == ["house"], "still not called: reported as skipped", str(result.skipped))
    check("Nothing in the house was changed or checked." in result.notices, "the user gets the tool's own notice",
          str(result.notices))
    check(result.answer == "Ho ho ho, they are on!", "the second answer is the one shown", result.answer)
    spoken = stream.compose_reply(result, spoken=True)
    check(spoken.endswith("Note: Nothing in the house was changed or checked."), "the notice is read aloud", spoken)

    # The saved history must not show a device changed with no tool behind it.
    saved = {"chat_history": [], "continuous_save": False,
             "db": type("DB", (), {"add_document": lambda self, q, a: None})()}
    stream.persist_turn(saved, "turn on the office lights", result)
    stored = saved["chat_history"][-1]["content"]
    check("house did not run" in stored and stored.startswith("Ho ho ho"), "the saved answer says nothing ran", stored)

    # 2. Nudged, then it calls the tool: nothing skipped.
    _, result, _ = run([answer("Done!"), call("house", action="turn_on"), answer("Turned on 2 lamps.")],
                       "turn on the office lights")
    check(result.executed == ["house"] and not result.skipped and not result.notices,
          "nudged, then called: ran, nothing skipped, no notice", f"{result.executed} {result.skipped} {result.notices}")
    check(result.generations == 3, "nudged, then called: three generations", str(result.generations))

    # 3. Called straight away: no nudge.
    _, result, seen = run([call("house", action="turn_on"), answer("Turned on 2 lamps.")], "turn on the office lights")
    check(result.generations == 2 and not result.skipped, "called first time: no nudge", str(result.generations))
    check(not any(ToolStream.NOTICE_PREFIX in str(m.get("content")) for m in seen['prompts'][-1]),
          "called first time: no notice in the prompt")

    # 4. A request that doesn't match: no nudge.
    _, result, _ = run([answer("Why did the elf go to school?")], "tell me a joke")
    check(result.generations == 1 and not result.skipped, "unrelated request: no nudge", str(result.generations))

    # 5. The tool isn't offered (not allowed for this server): no nudge - it could not have been called.
    _, result, _ = run([answer("I can't do that.")], "turn on the office lights", use_policy=no_house)
    check(result.generations == 1 and not result.skipped, "tool not offered: no nudge", str(result.generations))

    # 6. The call was refused: the user already hears about that, so no nudge on top.
    _, result, _ = run([call("house", action="turn_on"), answer("I couldn't.")], "turn on the office lights",
                       refuse=("house",))
    check(result.generations == 2 and not result.skipped, "refused call: counts as tried, no nudge",
          f"{result.generations} {result.skipped}")

    # 7. No request (a worker's loop): no check at all.
    _, result, _ = run([answer("The office lights are on!")], "", worker=True)
    check(result.generations == 1 and not result.skipped, "no request (worker): no nudge", str(result.generations))

    # 8. load_script_tool: the definition field, and a malformed one stopping the server.
    with tempfile.TemporaryDirectory() as tmp:
        def script(name, call_check):
            path = os.path.join(tmp, f"{name}.py")
            with open(path, "w") as fh:
                fh.write(textwrap.dedent(f"""
                    import json
                    print(json.dumps({{"name": "{name}", "description": "d", "parameters": {{"type": "object"}},
                                       "call_check": {call_check!r}}}))
                """))
            return path

        tool = load_script_tool(sys.executable, script("good", {"patterns": [r"\blamp\b"], "notice": "Nothing ran."}))
        check(tool.call_check is not None and tool.call_check.matches("the LAMP please") and
              tool.call_check.notice == "Nothing ran.", "load_script_tool reads call_check (case-insensitive)")
        tool = load_script_tool(sys.executable, script("none", None))
        check(tool.call_check is None, "no call_check: none")
        for label, spec in (("bad regex", {"patterns": ["(unclosed"]}),
                            ("misspelt key", {"pattern": ["lamp"]}),
                            ("patterns not a list", {"patterns": "lamp"})):
            try:
                load_script_tool(sys.executable, script("bad", spec))
                check(False, f"a {label} stops the server")
            except ValueError:
                check(True, f"a {label} stops the server")

    # 9. home_control's own patterns.
    path = os.path.join(settings.REPO_ROOT, "scripts", "ai", "ai-tools", "home_control", "home_control.py")
    spec = importlib.util.spec_from_file_location("home_control_definition", path)
    home_control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(home_control)
    house = R.CallCheck(**home_control.DEFINITION["call_check"])
    for request in ("Turn on the office lights", "can you turn the porch lights off", "Which lights are on upstairs?",
                    "Turn them off please.", "dim the kitchen lights to 30", "set the heat to 68",
                    "is the garage door open", "lock the front door", "close the blinds", "what's the thermostat at",
                    "switch off the lamp", "is the hall light on", "lights off"):
        check(house.matches(request), f"home_control matches: {request!r}")
    for request in ("I want Christmas lights on my list", "what's the weather", "tell me a joke",
                    "turn to page five", "I'm feeling down", "can you open the door to the North Pole",
                    "put lights on the tree"):
        check(not house.matches(request), f"home_control leaves alone: {request!r}")

    print("\nALL CHECKS PASS" if not failures else f"\n{len(failures)} check(s) FAILED")
    sys.exit(len(failures))


if __name__ == "__main__":
    main()

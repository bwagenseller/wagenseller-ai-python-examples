#!/usr/bin/env python3
"""CS-22: StreamBase's stateless 'one_shot' command, in all three stream families (no model loaded).

Why
---
The conversational pipeline asks the LLM server short questions ("which agent was addressed?") that must never land
in anyone's chat history or vector database. 'one_shot' is that path: a system prompt and a question in, a short
answer out, nothing remembered. It lives in StreamBase, so role-play, the knowledge base and the tool agent all
have it.

What it proves (RolePlayStream, KnowledgeBaseStream and ToolStream built with object.__new__ - no model; the one
generation call, generate_once(), is replaced with a recorder)
--------------------------------------------------------------------------------------------------------------------
1. Dispatch: 'one_shot' is answered with no session at all, and no session is created, looked up or touched;
   get_response() and create_session_from_request() are never called.
2. The generation gets exactly [system, user] messages, no conversational stops, reasoning OFF, and the answer
   comes back trimmed.
3. max_tokens: defaults to ONE_SHOT_DEFAULT_TOKENS, is clamped to 1..ONE_SHOT_MAX_TOKENS, and a non-integer is
   refused.
4. Refusals, with no generation: missing / blank / non-string system_prompt or user_request, or either one longer
   than ONE_SHOT_MAX_CHARS.
5. A server shutting down (generate_once raises RuntimeError) gives an error reply, not an exception.

Usage:  python check_one_shot.py      (the llama env: StreamBase imports llama_cpp)
"""
import os
import sys
import threading

# testlib finds the library under test (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()

from amadeo_utils.ai.llm.llama.StreamBase import StreamBase  # noqa: E402
from amadeo_utils.ai.llm.llama.RolePlayStream import RolePlayStream  # noqa: E402
from amadeo_utils.ai.llm.llama.KnowledgeBaseStream import KnowledgeBaseStream  # noqa: E402
from amadeo_utils.ai.llm.llama.ToolStream import ToolStream  # noqa: E402

failures = []


def check(label, got, expected):
    """Records one comparison and prints PASS / FAIL."""
    if got == expected:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}: got {got!r}, expected {expected!r}")
        failures.append(label)


def build(cls, answer="  Rick.  ", raises=None):
    """
    A stream with no model loaded. generate_once records its arguments and returns `answer` (or raises `raises`);
    every session method records that it was called, which must never happen for 'one_shot'.

    Returns:
        (stream, generations, session_calls)
    """
    stream = object.__new__(cls)
    stream.sessions = {}
    stream.sessions_lock = threading.Lock()
    stream.session_locks = {}
    generations, session_calls = [], []

    def generate_once(messages, local_stop, max_response_tokens, reason_used, session_id, used_tokens):
        generations.append({'messages': messages, 'stop': local_stop, 'max_tokens': max_response_tokens, 'reason': reason_used})
        if raises:
            raise raises
        return answer

    stream.generate_once = generate_once
    for name in ('get_session', 'get_session_and_lock', 'create_session_from_request', 'get_response', 'remove_session'):
        setattr(stream, name, lambda *a, _name=name, **k: session_calls.append(_name))
    return stream, generations, session_calls


def ask(stream, **fields):
    """Sends a 'one_shot' request through handle_client_request, as the server would. Returns the reply dict."""
    request = dict({'command': 'one_shot', 'sessionID': 'transient-1'}, **fields)
    reply, data = stream.handle_client_request(request, None)
    check(f"{type(stream).__name__}: no binary data", data, None)
    return reply


GOOD = {'system_prompt': 'Answer with one name.', 'user_request': 'Rick and Frasier, hi.'}

# 1 + 2, for every family
for cls in (RolePlayStream, KnowledgeBaseStream, ToolStream):
    stream, generations, session_calls = build(cls)
    reply = ask(stream, **GOOD)
    name = cls.__name__
    check(f"{name}: answered, trimmed", (reply['success'], reply['type'], reply['response']), (True, 'llm_response', 'Rick.'))
    check(f"{name}: no session touched", (session_calls, stream.sessions), ([], {}))
    check(f"{name}: [system, user] messages", generations[0]['messages'],
          [{'role': 'system', 'content': GOOD['system_prompt']}, {'role': 'user', 'content': GOOD['user_request']}])
    check(f"{name}: no stops, reasoning off", (generations[0]['stop'], generations[0]['reason']), ([], False))

# 3. max_tokens
stream, generations, _ = build(RolePlayStream)
ask(stream, **GOOD)
check("max_tokens defaults", generations[-1]['max_tokens'], StreamBase.ONE_SHOT_DEFAULT_TOKENS)
ask(stream, max_tokens=10_000, **GOOD)
check("max_tokens clamped down", generations[-1]['max_tokens'], StreamBase.ONE_SHOT_MAX_TOKENS)
ask(stream, max_tokens=0, **GOOD)
check("max_tokens clamped up", generations[-1]['max_tokens'], 1)
ask(stream, max_tokens="7", **GOOD)
check("numeric string accepted", generations[-1]['max_tokens'], 7)
count = len(generations)
reply = ask(stream, max_tokens="lots", **GOOD)
check("non-integer max_tokens refused, nothing generated", (reply['success'], len(generations)), (False, count))

# 4. refusals
too_long = 'x' * (StreamBase.ONE_SHOT_MAX_CHARS + 1)
for label, fields in (("missing system_prompt", {'user_request': 'hi'}),
                      ("missing user_request", {'system_prompt': 'hi'}),
                      ("blank user_request", {'system_prompt': 'hi', 'user_request': '   '}),
                      ("non-string system_prompt", {'system_prompt': 5, 'user_request': 'hi'}),
                      ("over-long user_request", {'system_prompt': 'hi', 'user_request': too_long}),
                      ("over-long system_prompt", {'system_prompt': too_long, 'user_request': 'hi'})):
    stream, generations, session_calls = build(RolePlayStream)
    reply = ask(stream, **fields)
    check(f"{label}: refused, nothing generated, no session",
          (reply['success'], reply['type'], generations, session_calls), (False, 'error', [], []))

# 5. shutting down
stream, _, _ = build(RolePlayStream, raises=RuntimeError("the models have been released - the server is shutting down"))
reply = ask(stream, **GOOD)
check("server shutting down: an error reply", (reply['success'], 'shutting down' in reply['message']), (False, True))

print(f"\n{len(failures)} failure(s)")
sys.exit(len(failures))

#!/usr/bin/env python3
"""CS-22: asking the LLM which agent was addressed - the question and the reading of the answer (pure logic).

Why
---
When several agents are in play and punctuation can't settle who is being spoken to, the server asks the LLM
(stateless 'one_shot'). A model may answer "Rick.", "I think Rick", "Dr. Crane" or "Rick or Frasier"; only an answer
that names exactly one candidate may be trusted, and everything else must fall back to the first agent named.

What it proves (amadeo_utils.ai.combined.conversational_ai.routing)
------------------------------------------------------------------
1. build_routing_prompt: lists the candidates by display name, mentions who spoke last only when known, and quotes
   the transcript in the question.
2. parse_routing_reply: accepts a display name, name or wake word, with punctuation or extra words around it;
   rejects blank answers, answers naming nobody, and answers naming two candidates or a non-candidate. Only the
   reply's first non-empty line is read, so an explanation after the name can't spoil a clear answer.

Usage:  python check_routing.py      (any Python 3 with the repo's src on the path)
"""
import os
import sys

# testlib finds the library under test (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testlib import settings  # noqa: E402
settings.use_src()

from amadeo_utils.ai.combined.conversational_ai.routing import build_routing_prompt, parse_routing_reply  # noqa: E402
from amadeo_utils.ai.combined.conversational_ai.wake_words import build_agents  # noqa: E402

failures = []


def check(label, got, expected):
    """Records one comparison and prints PASS / FAIL."""
    if got == expected:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}: got {got!r}, expected {expected!r}")
        failures.append(label)


FALLBACK = {'voice': 'default', 'system_prompt_id': 'default', 'continuous_save': False, 'load_previous': True}
ROSE, CRANE, RICK = build_agents({'agents': [
    {'name': 'rose', 'wake_words': ['rose', 'hey rose']},
    {'name': 'crane', 'display_name': 'Frasier', 'wake_words': ['doctor crane', 'dr crane', 'frasier']},
    {'name': 'rick', 'wake_words': ['rick']},
]}, FALLBACK)

# 1. the question
system, user = build_routing_prompt("Rick and Frasier, what do you both think?", [RICK, CRANE])
check("candidates listed by display name", 'Rick, Frasier' in system, True)
check("asks for exactly one name", 'exactly one name' in system, True)
check("no 'spoke last' when nobody did", 'spoke last' in system, False)
check("the transcript is quoted in the question", '"Rick and Frasier, what do you both think?"' in user, True)
system, _ = build_routing_prompt("Rick said something odd.", [RICK, CRANE], last_speaker=CRANE)
check("who spoke last is mentioned when known", 'Frasier spoke last' in system, True)


# 2. the answer
def parsed(reply, candidates=(RICK, CRANE)):
    """The name of the agent parse_routing_reply() picks, or None."""
    agent = parse_routing_reply(reply, list(candidates))
    return agent['name'] if agent else None

check("a bare display name", parsed("Frasier"), 'crane')
check("a name with punctuation", parsed("Rick."), 'rick')
check("a name inside a sentence", parsed("The user is speaking to Rick"), 'rick')
check("the agent's name (not display name)", parsed("crane"), 'crane')
check("a wake word ('Dr. Crane')", parsed("Dr. Crane"), 'crane')
check("the same agent twice is still one answer", parsed("Frasier (Dr. Crane)"), 'crane')
check("two candidates: rejected", parsed("Rick or Frasier"), None)
check("no name: rejected", parsed("I'm not sure."), None)
check("blank: rejected", parsed("   "), None)
check("a non-candidate: rejected", parsed("Rose"), None)
check("a candidate plus a non-candidate: the candidate", parsed("Rick, not Rose"), 'rick')
check("only the first line counts (a chatty model explains after the name)",
      parsed("Frasier\n\nThe user is asking Frasier what Rick would say."), 'crane')
check("leading blank lines are skipped", parsed("\n\n  Rick\nbecause Frasier..."), 'rick')
check("two names on the first line: still rejected", parsed("Rick, Frasier\n\nThe user is addressing both."), None)

print(f"\n{len(failures)} failure(s)")
sys.exit(len(failures))

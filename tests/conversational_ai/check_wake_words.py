#!/usr/bin/env python3
"""CS-22: wake words and agent configs for the conversational AI pipeline (pure logic, no audio or GPU).

Why
---
The conversational client used to send every sentence it heard to the LLM. Now each agent has wake words, and the
server decides from the transcript which agent (if any) was spoken to. The matching rules are easy to get subtly
wrong, so they are pinned down here.

What it proves (amadeo_utils.ai.combined.conversational_ai.wake_words)
---------------------------------------------------------------------
1. Normalising: case and punctuation are ignored, so "Dr. Crane," matches "dr crane".
2. Whole words only: "rose" does not match "roses" or "primrose".
3. Position: a wake word must START within the first N words (it may run past word N); 0 means anywhere.
4. First said wins: "Rick, is Dr. Crane right?" is for Rick, although "dr crane" is longer. Only two wake words
   starting at the same word fall back to the longer one; "hey rose" still beats "rose" (it starts earlier).
5. select_agent order: a wake word beats a continuation (switching agents mid-conversation); a continuation goes to
   the active agent; an always-listening agent gets what is left; otherwise nobody. Blank speech never wakes anyone
   and is never a continuation.
6. build_agents: agent_defaults are merged under each agent; the old single-agent config becomes one always-listening
   agent; bad configs (duplicate names, a wake word on two agents, wrong types, empty agents) are refused;
   allow_unknown_speakers / allowed_speakers default to true / empty and must be a boolean / a list of names.
7. Several agents in play (named, plus the active agent mid-conversation): one clearly spoken to by punctuation
   ("Rick, ...", "Hey Rick ...", "..., Rick?") answers with no LLM call; otherwise 'ambiguous' with the candidates.

Usage:  python check_wake_words.py      (any Python 3 with the repo's src on the path)
"""
import os
import sys

# testlib finds the library under test (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testlib import settings  # noqa: E402
settings.use_src()

from amadeo_utils.ai.combined.conversational_ai.wake_words import (  # noqa: E402
    normalise, find_wake_word, select_agent, build_agents)

failures = []


def check(label, got, expected):
    """Records one comparison and prints PASS / FAIL."""
    if got == expected:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}: got {got!r}, expected {expected!r}")
        failures.append(label)


def raises(label, exc_type, fn):
    """Records whether fn() raises exc_type."""
    try:
        fn()
    except exc_type:
        print(f"PASS  {label}")
        return
    except Exception as e:      # the wrong exception
        print(f"FAIL  {label}: raised {type(e).__name__}: {e}")
        failures.append(label)
        return
    print(f"FAIL  {label}: nothing raised")
    failures.append(label)


FALLBACK = {'voice': 'default', 'system_prompt_id': 'default', 'continuous_save': False, 'load_previous': True}

# The example config from the ticket (made-up names)
CONFIG = {
    'agent_defaults': {'voice': 'heart', 'continuous_save': True, 'load_previous': True},
    'agents': [
        {'name': 'rose', 'wake_words': ['rose', 'hey rose'], 'system_prompt_id': 'assistant-rose'},
        {'name': 'crane', 'wake_words': ['doctor crane', 'dr crane', 'frasier', 'frazier'],
         'system_prompt_id': 'assistant-frasier', 'voice': 'frasier', 'load_previous': False},
    ],
}
AGENTS = build_agents(CONFIG, FALLBACK)


def woken(transcript, max_position=10):
    """The (agent name, wake word) woken by a transcript, or None."""
    match = find_wake_word(transcript, AGENTS, max_position)
    return (match[0]['name'], match[1]) if match else None


# 1. normalising
check("normalise strips case and punctuation", normalise("Dr. Crane, are you there?"), ['dr', 'crane', 'are', 'you', 'there'])
check("'Dr. Crane,' wakes crane", woken("Dr. Crane, what do you think?"), ('crane', 'dr crane'))
check("'ROSE!' wakes rose", woken("ROSE! Turn on the lights."), ('rose', 'rose'))

# 2. whole words only
check("'roses' does not wake rose", woken("The roses need water."), None)
check("a possessive ('Rose's') does not wake rose", woken("What's Rose's favourite colour?"), None)
check("'primrose' does not wake rose", woken("Primrose is a flower."), None)

# 3. position
check("wake word at word 10 counts", woken("one two three four five six seven eight nine rose"), ('rose', 'rose'))
check("wake word at word 11 does not", woken("one two three four five six seven eight nine ten rose"), None)
check("two-word wake word may start at word 10 and run past it",
      woken("one two three four five six seven eight nine dr crane"), ('crane', 'dr crane'))
check("max_position 0 means anywhere", woken(" ".join(["word"] * 40) + " rose", max_position=0), ('rose', 'rose'))
check("blank transcript wakes nobody", woken(""), None)

# 4. the first wake word said wins; the longer one only breaks a tie at the same word
check("'hey rose' beats 'rose'", woken("Hey Rose, what time is it?"), ('rose', 'hey rose'))
TRIO = build_agents({'agents': [
    {'name': 'rick', 'wake_words': ['rick']},
    {'name': 'crane', 'wake_words': ['dr crane', 'frasier']},
    {'name': 'doc', 'wake_words': ['dr']},
]}, FALLBACK)
first = find_wake_word("Rick, do you think that Dr. Crane is right about that?", TRIO)
check("first name said wins over a longer, later one", (first[0]['name'], first[1]), ('rick', 'rick'))
first = find_wake_word("Dr. Crane, do you agree with Rick?", TRIO)
check("...and the other way round", (first[0]['name'], first[1]), ('crane', 'dr crane'))
first = find_wake_word("Dr. Crane, hello", TRIO)
check("same starting word: the longer wake word wins ('dr crane' over 'dr')", first[0]['name'], 'crane')
first = find_wake_word("Dr. Who, hello", TRIO)
check("same starting word, only the shorter one matches", first[0]['name'], 'doc')

# 5. select_agent
def picked(transcript, continuation=False, active_agent='', agents=AGENTS):
    """The (agent name or None, reason) that select_agent() gives."""
    selection = select_agent(transcript, agents, continuation, active_agent)
    return (selection.agent['name'] if selection.agent else None, selection.reason)

check("no wake word, no conversation: nobody", picked("What time is it?"), (None, 'not_addressed'))
check("continuation goes to the active agent", picked("And tomorrow?", True, 'rose'), ('rose', 'continuation'))
check("mid-conversation with crane, naming Rick first switches to Rick",
      (select_agent("Rick, do you think that Dr. Crane is right about that?", TRIO, True, 'crane').agent['name']), 'rick')
check("a wake word spoken TO another agent beats a continuation (switch agents)", picked("Frasier, what do you think?", True, 'rose'), ('crane', 'vocative'))
check("continuation without an active agent: nobody", picked("And tomorrow?", True, ''), (None, 'not_addressed'))
check("continuation naming an unknown agent: nobody", picked("And tomorrow?", True, 'ghost'), (None, 'not_addressed'))
check("blank speech is never a continuation", picked("   ", True, 'rose'), (None, 'not_addressed'))

always_on = build_agents({'system_prompt_id': 'solo'}, FALLBACK)
check("old single-agent config is always listening", picked("What time is it?", agents=always_on), ('default', 'always_on'))
check("always-listening agent still gets blank speech (old behaviour)", picked("", agents=always_on), ('default', 'always_on'))
mixed = build_agents({'agents': [{'name': 'rose', 'wake_words': ['rose']}, {'name': 'house'}]}, FALLBACK)
check("mixed: wake word picks its agent", picked("Rose, hello", agents=mixed), ('rose', 'wake_word'))
check("mixed: anything else goes to the always-listening agent", picked("hello", agents=mixed), ('house', 'always_on'))

# 6. build_agents
rose, crane = AGENTS
check("rose inherits the defaults", (rose['voice'], rose['continuous_save'], rose['load_previous']), ('heart', True, True))
check("crane overrides voice and load_previous", (crane['voice'], crane['continuous_save'], crane['load_previous']), ('frasier', True, False))
check("fallback fills keys nobody set", build_agents({'agents': [{'name': 'x', 'wake_words': ['x']}]}, FALLBACK)[0]['system_prompt_id'], 'default')
old = build_agents({'voice': 'bella', 'system_prompt_id': 'assistant-rose', 'load_previous': False}, FALLBACK)
check("old config: one agent named default with its keys",
      (len(old), old[0]['name'], old[0]['voice'], old[0]['system_prompt_id'], old[0]['load_previous'], old[0]['wake_words']),
      (1, 'default', 'bella', 'assistant-rose', False, []))
check("display_name defaults to the name in title case", (rose['display_name'], crane['display_name']), ('Rose', 'Crane'))
check("display_name can be set per agent",
      build_agents({'agents': [{'name': 'crane', 'display_name': 'Frasier'}]}, FALLBACK)[0]['display_name'], 'Frasier')
check("agent_defaults cannot set a display_name",
      build_agents({'agent_defaults': {'display_name': 'Z'}, 'agents': [{'name': 'a'}]}, FALLBACK)[0]['display_name'], 'A')
check("allow_unknown_speakers defaults to true, and can be set per agent or in agent_defaults",
      [a['allow_unknown_speakers'] for a in build_agents({'agent_defaults': {'allow_unknown_speakers': False},
                                                          'agents': [{'name': 'a'}, {'name': 'b', 'allow_unknown_speakers': True}]}, FALLBACK)]
      + [rose['allow_unknown_speakers']], [False, True, True])
raises("allow_unknown_speakers must be a boolean", TypeError,
       lambda: build_agents({'agents': [{'name': 'a', 'allow_unknown_speakers': 'no'}]}, FALLBACK))
check("allowed_speakers defaults to empty, and can be set per agent or in agent_defaults",
      [a['allowed_speakers'] for a in build_agents({'agent_defaults': {'allowed_speakers': ['Sam']},
                                                    'agents': [{'name': 'a'}, {'name': 'b', 'allowed_speakers': ['Kim', 'Kevin']}]}, FALLBACK)]
      + [rose['allowed_speakers']], [['Sam'], ['Kim', 'Kevin'], []])
raises("allowed_speakers must be a list", TypeError,
       lambda: build_agents({'agents': [{'name': 'a', 'allowed_speakers': 'Sam'}]}, FALLBACK))
raises("allowed_speakers must hold non-empty names", ValueError,
       lambda: build_agents({'agents': [{'name': 'a', 'allowed_speakers': ['Sam', ' ']}]}, FALLBACK))
check("agent_defaults cannot set a name", build_agents({'agent_defaults': {'name': 'z'}, 'agents': [{'name': 'a'}]}, FALLBACK)[0]['name'], 'a')

raises("empty agents list refused", ValueError, lambda: build_agents({'agents': []}, FALLBACK))
raises("agent without a name refused", ValueError, lambda: build_agents({'agents': [{'wake_words': ['x']}]}, FALLBACK))
raises("duplicate agent names refused", ValueError, lambda: build_agents({'agents': [{'name': 'a'}, {'name': 'a'}]}, FALLBACK))
raises("same wake word on two agents refused (after normalising)", ValueError,
       lambda: build_agents({'agents': [{'name': 'a', 'wake_words': ['Dr. Crane']}, {'name': 'b', 'wake_words': ['dr crane']}]}, FALLBACK))
raises("wake_words must be a list", TypeError, lambda: build_agents({'agents': [{'name': 'a', 'wake_words': 'rose'}]}, FALLBACK))
raises("a blank wake word is refused", ValueError, lambda: build_agents({'agents': [{'name': 'a', 'wake_words': ['!!']}]}, FALLBACK))
raises("wrong type in agent_defaults refused", TypeError, lambda: build_agents({'agent_defaults': {'load_previous': 'yes'}, 'agents': [{'name': 'a'}]}, FALLBACK))

# 7. several agents in play: punctuation rules, then 'ambiguous' for the LLM
PEOPLE = build_agents({'agents': [
    {'name': 'rose', 'wake_words': ['rose', 'hey rose']},
    {'name': 'crane', 'display_name': 'Frasier', 'wake_words': ['doctor crane', 'dr crane', 'frasier', 'frazier']},
    {'name': 'rick', 'wake_words': ['rick']},
]}, FALLBACK)


def decided(transcript, continuation=False, active_agent=''):
    """(agent name, reason, candidate names) from select_agent() over PEOPLE."""
    sel = select_agent(transcript, PEOPLE, continuation, active_agent)
    return (sel.agent['name'] if sel.agent else None, sel.reason, [a['name'] for a in sel.candidates])

check("leading vocative beats a later mention", decided("Rick, do you think that Dr. Crane is right about that?"), ('rick', 'vocative', []))
check("vocative after a sentence about someone else", decided("Frasier said something odd. Rick, what do you think?"), ('rick', 'vocative', []))
check("trailing vocative ('..., Rick?')", decided("I agree with Rose. What do you think, Rick?"), ('rick', 'vocative', []))
check("a greeting needs no comma ('Hey Rick what...')", decided("Hey Rick what do you think about Rose"), ('rick', 'vocative', []))
check("a filler needs the comma ('And Rick, ...')", decided("And Rick, what did Rose mean?"), ('rick', 'vocative', []))
check("filler then comma ('So, Frasier, ...')", decided("So, Frasier, tell me what Rose meant."), ('crane', 'vocative', []))
check("'Dr.' is not a sentence end", decided("Dr. Crane, do you agree with Rick?"), ('crane', 'vocative', []))
check("punctuation inside quotes still counts", decided('"Rick," she said, "is Rose right?"'), ('rick', 'vocative', []))
check("a filler without a comma is not a vocative ('And Rick said...')",
      decided("And Rick said Rose was wrong"), ('rick', 'ambiguous', ['rick', 'rose']))
check("two names spoken to together: ambiguous", decided("Rick and Frasier, what do you both think?"), ('rick', 'ambiguous', ['rick', 'crane']))
check("both names spoken to: ambiguous",
      decided("Frasier, as usual, was wrong. What do you think, Rick?"), ('crane', 'ambiguous', ['crane', 'rick']))
check("mid-conversation, a mere mention of another agent: the active one is in play too",
      decided("Rick said something odd earlier, what do you think?", True, 'crane'), ('rick', 'ambiguous', ['rick', 'crane']))
check("mid-conversation, speaking to another agent: no LLM needed",
      decided("Rick, what do you think?", True, 'crane'), ('rick', 'vocative', []))
check("mid-conversation, naming the agent you are talking to: no switch, no LLM",
      decided("Frasier, you're right.", True, 'crane'), ('crane', 'wake_word', []))
check("one agent named twice is still one agent", decided("Rose, Rose, are you there?"), ('rose', 'wake_word', []))
check("names only after the window wake nobody",
      decided("one two three four five six seven eight nine ten Rick and Rose"), (None, 'not_addressed', []))
check("a name in the window brings later names into play",
      decided("Rick said that one two three four five six seven eight Rose is right"), ('rick', 'ambiguous', ['rick', 'rose']))

print(f"\n{len(failures)} failure(s)")
sys.exit(len(failures))

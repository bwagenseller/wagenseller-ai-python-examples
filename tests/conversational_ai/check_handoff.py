#!/usr/bin/env python3
"""CS-22: handoff notes and speaker tags - telling an agent who is talking, and what was said to the other agents since
it last spoke (pure logic).

Why
---
Each agent has its own LLM session and history. Talk to Rose, then say "Hey Frasier, what do you think about that?",
and Frasier has no idea what "that" is. The server now puts the turns an agent missed at the front of its request.
Which turns count as "missed", and how the note is kept short, are pinned down here. Separately, an agent once
called the user by another agent's name, because nothing told it who was talking - so requests carry a speaker tag,
and the note (saved in the agent's history) names the speaker of every turn instead of saying "I".

What it proves (amadeo_utils.ai.combined.conversational_ai.handoff)
------------------------------------------------------------------
1. missed_turns: everything after the agent's own last turn; everything if it never spoke; nothing if it answered
   the latest turn. Malformed turns (they arrive over the network) are skipped, not fatal.
2. build_handoff_note: the note names agents by display name (title case if unknown) and each turn's speaker (the
   default speaker, then 'The user', if the turn has none), quotes both sides, uses only the most recent max_turns,
   and is '' when nothing was missed or notes are off (max_turns 0). '##' in quotes can't hide part of it.
3. shorten: long text is cut at a word boundary with '...', and whitespace is collapsed.
4. speaker_tag / clean_name: '[Brent, to Santa]: ', '[Brent]: ' with no addressee, '' with no speaker; brackets and
   '##' in a name are removed.

Usage:  python check_handoff.py      (any Python 3 with the repo's src on the path)
"""
import os
import sys

# testlib finds the library under test (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testlib import settings  # noqa: E402
settings.use_src()

from amadeo_utils.ai.combined.conversational_ai.handoff import shorten, missed_turns, build_handoff_note, speaker_tag, clean_name  # noqa: E402

failures = []


def check(label, got, expected):
    """Records one comparison and prints PASS / FAIL."""
    if got == expected:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}: got {got!r}, expected {expected!r}")
        failures.append(label)


def turn(agent, user, reply):
    """One turn of a conversation, as the client sends it."""
    return {'agent': agent, 'user': user, 'reply': reply}


NAMES = {'rose': 'Rose', 'crane': 'Frasier'}
T1 = turn('rose', 'Should I repaint the deck?', 'Yes, before the first frost.')
T2 = turn('rose', 'What colour?', 'Grey would suit the house.')
T3 = turn('crane', 'Hey Frasier, what do you think?', 'Grey? How pedestrian.')

# 1. missed_turns
check("never spoke: missed everything", missed_turns([T1, T2], 'crane'), [T1, T2])
check("answered the latest turn: missed nothing", missed_turns([T1, T2], 'rose'), [])
check("missed only what came after its own last turn", missed_turns([T1, T3, T2], 'crane'), [T2])
check("back to Rose after Frasier: she missed his turn", missed_turns([T1, T2, T3], 'rose'), [T3])
check("no turns: nothing missed", missed_turns([], 'rose'), [])
check("None: nothing missed", missed_turns(None, 'rose'), [])
check("malformed turns are skipped",
      missed_turns(['junk', {'agent': 'rose'}, {'agent': 1, 'user': 'a', 'reply': 'b'}, T1], 'crane'), [T1])

# 2. build_handoff_note
check("Frasier is told what Rose said, and by whom",
      build_handoff_note([T1], 'crane', NAMES, default_speaker='Alex'),
      '(Since you last spoke: Alex said to Rose: "Should I repaint the deck?" Rose replied: "Yes, before the first frost.")\n')
check("several turns are joined in order",
      build_handoff_note([T1, T2], 'crane', NAMES, default_speaker='Alex'),
      '(Since you last spoke: Alex said to Rose: "Should I repaint the deck?" Rose replied: "Yes, before the first frost."'
      ' Then Alex said to Rose: "What colour?" Rose replied: "Grey would suit the house.")\n')
check("Rose is told what Frasier (by display name) said",
      build_handoff_note([T1, T3], 'rose', NAMES, default_speaker='Alex'),
      '(Since you last spoke: Alex said to Frasier: "Hey Frasier, what do you think?" Frasier replied: "Grey? How pedestrian.")\n')
check("a turn's own speaker wins over the default",
      'Sam said to Rose:' in build_handoff_note([dict(T1, speaker='Sam')], 'crane', NAMES, default_speaker='Alex'), True)
check("no speaker anywhere: 'The user'", 'The user said to Rose:' in build_handoff_note([T1], 'crane', NAMES), True)
check("a non-string speaker falls back to the default",
      'Alex said to Rose:' in build_handoff_note([dict(T1, speaker=7)], 'crane', NAMES, default_speaker='Alex'), True)
check("the note never says 'I' (the agent must not guess who 'I' is)",
      ' I said' in build_handoff_note([T1, T3], 'rose', NAMES, default_speaker='Alex'), False)
check("no note when the agent missed nothing", build_handoff_note([T1, T2], 'rose', NAMES), '')
check("only the most recent max_turns",
      build_handoff_note([T1, T2], 'crane', NAMES, max_turns=1).count(' said to '), 1)
check("max_turns 1 keeps the LATEST turn", 'What colour?' in build_handoff_note([T1, T2], 'crane', NAMES, max_turns=1), True)
check("max_turns 0 turns notes off", build_handoff_note([T1], 'crane', NAMES, max_turns=0), '')
check("unknown display name: title case", 'said to Rose:' in build_handoff_note([T1], 'crane', {}), True)
check("empty display name: title case", 'said to Rose:' in build_handoff_note([T1], 'crane', {'rose': ''}), True)
long_reply = turn('rose', 'Tell me everything.', 'word ' * 500)
note = build_handoff_note([long_reply], 'crane', NAMES, max_chars=50)
check("long replies are shortened in the note", len(note) < 200 and '..."' in note, True)
note = build_handoff_note([turn('rose', 'a ## b', 'c ## d')], 'crane', NAMES)
check("'##' in a quote can't hide part of the note", note.count('##'), 0)

# 3. shorten
check("short text unchanged", shorten("Hello there.", 50), "Hello there.")
check("cut at a word boundary", shorten("one two three four", 10), "one two...")
check("whitespace collapsed", shorten("  a \n b\t c ", 50), "a b c")
check("max_chars 0 means no limit", shorten("x " * 1000, 0), ("x " * 1000).strip())
check("one long word is still cut", shorten("a" * 20, 5), "aaaaa...")

# 4. speaker_tag / clean_name
check("speaker and addressee", speaker_tag('Brent', 'Santa'), '[Brent, to Santa]: ')
check("no addressee (one agent)", speaker_tag('Brent'), '[Brent]: ')
check("no speaker: no tag", speaker_tag('', 'Santa'), '')
check("None speaker: no tag", speaker_tag(None, 'Santa'), '')
check("brackets and '##' are taken out of names", speaker_tag('[Br##ent]', 'Sa]nta'), '[Brent, to Santa]: ')
check("clean_name collapses whitespace", clean_name('  Mary \n Ann '), 'Mary Ann')

print(f"\n{len(failures)} failure(s)")
sys.exit(len(failures))

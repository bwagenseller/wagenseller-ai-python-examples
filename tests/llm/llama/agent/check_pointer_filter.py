#!/usr/bin/env python3
"""CS-21: which replies count as "only points at the worker's answer" (ToolStream._is_pointer_only).

Why
---
After a web lookup the main model kept adding "The top news headlines for today are shown above."
directly under the answer it pointed at (seen live, 2026-09-25). The instruction now tells it not
to, and a reply that is NOTHING but such a pointer is dropped in code. The first pattern tried
would also have dropped real replies ("As shown above, stay indoors tonight."), so the filter
now needs the WHOLE reply to be one pointer sentence. The positives are the pointer lines models
actually wrote in live runs; the negatives are replies that must survive.

Runs in the llama env (no model):  python check_pointer_filter.py
"""
import os
import sys

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()
SRC = settings.SRC

POINTERS = [                                     # all dropped
    "The answer is shown above.",
    "The top news headlines for today are shown above.",                 # Qwen, live
    "The top-rated X (Twitter) posts of the day are shown above.",       # Qwen, live
    "The requested tips and strategies have been provided above.",       # Gemma 4, live
    "The tips are listed above.",
    "See the answer above.",
    "The results are displayed above",
    "(The results above were shown to the user.)",                       # Qwen after recall, live
    "The answer has been shown to the user.",
    "I am afraid I cannot see the specific trending topics myself, Kevin, but the answer has been provided to you above.",  # Gemma 4, live 2026-09-26
    "I cannot see the results myself, but they have been shown to you.",
    "It's shown above.",
]
# (reply, player_name): a persona prompt makes the models address the user by name (live, 2026-09-26).
NAMED_POINTERS = [                               # all dropped
    ("The top headline on Twitter right now is shown above, Kevin.", "Kevin"),       # Qwen 3.6, live
    ("Kevin, the answer is shown above.", "Kevin"),                                  # Muse-Glimmer, live
    ("Kevin, the assistant's answer is shown above.", "Kevin"),                      # Muse-Glimmer, live
    ("kevin, the answer is shown above", "Kevin"),
]
NAMED_KEEP = [                                   # all kept
    ("Kevin, want me to check the radar too?", "Kevin"),
    ("The answer is shown above, Alex.", "Kevin"),          # not the session's name
    ("The answer is shown above, Kevin.", ""),               # no name in the session: nothing is removed
    ("Kevin, the answer is shown above. Want the radar too?", "Kevin"),
]
KEEP = [                                         # all kept
    "Want me to look up tomorrow's forecast as well?",
    "As shown above, the storm moved east; stay indoors tonight.",
    "Based on the above, bring an umbrella tomorrow.",
    "The answer is shown above. Want me to check the radar too?",
    "Math test on the 29th is the big one - shown above with the rest.",
    "(The results above were shown to the user.) Want me to fetch another page?",
    "The forecast has been provided to you above; bring an umbrella tomorrow.",
    "I cannot see the answer, but it has been shown to you. Want me to search again?",
    "",
]


def main():
    from amadeo_utils.ai.llm.llama.ToolStream import ToolStream
    failures = []
    for text in POINTERS:
        ok = ToolStream._is_pointer_only(text)
        print(f"  {'PASS' if ok else 'FAIL'}  dropped: {text!r}")
        failures += [] if ok else [text]
    for text in KEEP:
        ok = not ToolStream._is_pointer_only(text)
        print(f"  {'PASS' if ok else 'FAIL'}  kept:    {text!r}")
        failures += [] if ok else [text]
    for text, name in NAMED_POINTERS:
        ok = ToolStream._is_pointer_only(text, name)
        print(f"  {'PASS' if ok else 'FAIL'}  dropped: {text!r} (player {name!r})")
        failures += [] if ok else [text]
    for text, name in NAMED_KEEP:
        ok = not ToolStream._is_pointer_only(text, name)
        print(f"  {'PASS' if ok else 'FAIL'}  kept:    {text!r} (player {name!r})")
        failures += [] if ok else [text]
    long_pointer = "The " + "very " * 40 + "detailed answer is shown above."
    ok = not ToolStream._is_pointer_only(long_pointer)
    print(f"  {'PASS' if ok else 'FAIL'}  kept: a pointer sentence longer than POINTER_MAX_CHARS")
    failures += [] if ok else ["long pointer"]
    print("\nALL CHECKS PASS" if not failures else f"\n{len(failures)} check(s) FAILED")
    sys.exit(len(failures))


if __name__ == "__main__":
    main()

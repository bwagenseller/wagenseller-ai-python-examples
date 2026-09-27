#!/usr/bin/env python3
"""CS-21: ToolStream.speakable - what a spoken session's reply looks like before text-to-speech reads it.

Why
---
2026-09-26, from the voice pipeline: the agent's grades answer was a markdown list (bullets,
bold, percentages) and text-to-speech read the symbols aloud. Spoken sessions now get a rule asking
the main model for plain speech (SPOKEN_MAIN_RULE), and speakable() is the safety net that
guarantees no markup is spoken whatever a model writes. Text sessions are not touched.

What it proves (no model): a grades reply shaped like the real one becomes plain sentences with the facts intact;
headings, numbered lists, tables, links, bare URLs, code and nested emphasis are handled; plain
text, arithmetic ("5 * 3"), snake_case and ordinary punctuation pass through unchanged.

Usage (llama env - ToolStream imports llama_cpp; no model is loaded):  python check_speakable.py
"""
import os
import sys

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()
SRC = settings.SRC
failures = []

# The shape of a real spoken grades reply (bullets, bold, percentages, a closing question); the names and numbers
# are made up.
GRADES_REPLY = """Well, Kevin, here is a snapshot of your current academic standing for Trimester 1:

*   **Art:** B+ (87.50%)
*   **Chorus:** B (84.10%)
*   **Math:** B (81.20%)
*   **Science:** B (83.00%)
*   **History:** C (76.40%)
*   **English:** C- (71.30%)
*   **Phys Ed:** A (94.00%)

Band is currently marked as N/A.

I notice you have a few missing assignments, particularly in History and English. I would suggest giving those some attention before they become a more significant issue! How do you feel about these grades?"""


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if not ok and detail else ""))
    if not ok:
        failures.append(label)


def main():
    from amadeo_utils.ai.llm.llama.ToolStream import ToolStream
    speak = ToolStream.speakable

    out = speak(GRADES_REPLY)
    print("== a grades reply ==\n" + out + "\n")
    symbols = [s for s in ("*", "#", "_", "`", "|") if s in out]
    check(not symbols, "no markdown symbols left", str(symbols))
    for fact in ("Art: B+ (87.50%).", "Phys Ed: A (94.00%).", "English: C- (71.30%).", "How do you feel about these grades?",
                 "Well, Kevin, here is a snapshot"):
        check(fact in out, f"kept: {fact!r}", out)

    print("== other markdown ==")
    cases = [
        ("# Weather today\nIt is **sunny**.", "Weather today. It is sunny."),        # one paragraph: one spoken line
        ("# Weather today\n\nIt is **sunny**.", "Weather today.\nIt is sunny."),     # paragraphs stay separate
        ("1. First step\n2) Second step", "First step. Second step."),
        ("| Course | Grade |\n|---|---|\n| Math | B |", "Course, Grade. Math, B."),
        ("See [the NWS site](https://weather.gov) for more.", "See the NWS site for more."),
        ("Details at https://example.com/page today.", "Details at today."),
        ("Run `get_weather` now.", "Run get_weather now."),
        ("***very*** important and _quite_ so", "very important and quite so"),
        ("> quoted line", "quoted line"),
        ("Before\n\n---\n\nAfter", "Before\nAfter"),
        ("```\nprint(1)\n```", "print(1)"),
    ]
    for given, want in cases:
        got = speak(given)
        check(got == want, f"{given[:40]!r} -> {want!r}", f"got {got!r}")

    print("== left alone ==")
    for plain in ("It is 7:42 AM on Saturday.", "5 * 3 is 15.", "The variable snake_case_name stays.",
                  "Wait - really? Yes: 3 missing.", ""):
        check(speak(plain) == plain, f"unchanged: {plain!r}", repr(speak(plain)))

    print("\nALL CHECKS PASS" if not failures else f"\n{len(failures)} check(s) FAILED")
    sys.exit(len(failures))


if __name__ == "__main__":
    main()

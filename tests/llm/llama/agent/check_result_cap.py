#!/usr/bin/env python3
"""CS-21: the room-left rule for tool-result caps (ToolStream.result_cap_chars), as plain arithmetic.

Why
---
A round's results used to get max_tool_result_tokens EACH, whatever room was left. Seen live
(2026-09-25): a worker fetched two pages in parallel in an 8K window, the two 3000-token results
overflowed it, and the loop had to cut them down for a tools-off final pass. Now one round's
results share tool_result_share of the room left, evenly, each capped at max_tool_result_tokens
and floored at MIN_RESULT_TOKENS. golden_tool_loop.py's 'parallel_results_share_the_room' checks
the effect in the real loop; this checks the numbers.

Runs anywhere (no model):  python check_result_cap.py
"""
import os
import sys

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()
SRC = settings.SRC
failures = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if not ok and detail else ""))
    if not ok:
        failures.append(label)


def main():
    try:
        from amadeo_utils.ai.llm.llama.ToolStream import ToolStream
    except ImportError as e:            # llama_cpp is imported by the module; run this in the llama env
        print(f"needs the llama env: {e}")
        sys.exit(2)
    stream = ToolStream.__new__(ToolStream)          # no model: result_cap_chars only reads these two attributes
    stream.max_result_chars = 3000 * 4
    stream.tool_result_share = 0.5
    floor = ToolStream.MIN_RESULT_TOKENS * 4

    check(stream.result_cap_chars(100_000, 1) == 12_000, "plenty of room: one result keeps the full cap")
    check(stream.result_cap_chars(100_000, 2) == 12_000, "plenty of room: parallel results keep the full cap too")
    check(stream.result_cap_chars(6000, 1) == 3000 * 4, "6000 left, one call: half the room = 3000 tokens")
    check(stream.result_cap_chars(6000, 2) == 1500 * 4, "6000 left, two calls: 1500 tokens each")
    check(stream.result_cap_chars(6000, 3) == 1000 * 4, "6000 left, three calls: 1000 tokens each")
    check(stream.result_cap_chars(300, 2) == floor, "almost no room: floored at MIN_RESULT_TOKENS")
    check(stream.result_cap_chars(-500, 1) == floor, "negative room (already over): still the floor, never negative")
    check(stream.result_cap_chars(6000, 0) == 3000 * 4, "zero calls is treated as one")
    stream.tool_result_share = 1.0
    check(stream.result_cap_chars(4000, 2) == 2000 * 4, "tool_result_share 1.0: the whole room, split evenly")
    stream.tool_result_share = 0.25
    check(stream.result_cap_chars(8000, 1) == 2000 * 4, "tool_result_share 0.25: a quarter of the room")
    print("\nALL CHECKS PASS" if not failures else f"\n{len(failures)} check(s) FAILED")
    sys.exit(len(failures))


if __name__ == "__main__":
    main()

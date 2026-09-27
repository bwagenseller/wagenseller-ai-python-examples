#!/usr/bin/env python3
"""CS-21: get_datetime resolves everyday zone names correctly, defaults to local time, and reads unambiguously.

Why
---
2026-09-26, found live: asked for Eastern time, the agent answered an hour off. Python
accepts 'EST' as a real zone - a FIXED UTC-5 with no daylight saving - so in summer it is an hour
behind New York; 'EDT', 'ET', 'Eastern' and city names were rejected outright; and the tool
defaulted to UTC. Now everyday names map to the DST-aware zone they mean, cities resolve, the
default is the server's local zone, and the answer carries the zone's own abbreviation, the IANA
name and the UTC offset.

What it proves (no model; compares against the live New York clock, so it holds all year)
-----------------------------------------------------------------------------------------
* Everyday names -> the right zone: EST/EDT/ET/Eastern/"Eastern Standard Time"/"eastern daylight
  time" -> America/New_York, and likewise Central, Mountain, Arizona, Pacific, Alaska, Hawaii,
  UTC/GMT/Z; cities ("new york", London, Tokyo); exact IANA names; nonsense and non-strings refused.
* 'EST' now answers with New York's CURRENT offset (EDT in summer), not a fixed -05:00, and says
  how it read the name.
* The default: an explicit default_timezone, else this machine's own zone; an unknown default
  stops the tool from being built (a config mistake stops the server).
* The output: weekday, date, 12-hour clock with abbreviation, IANA name, UTC offset, ISO form.

Usage (any env with the repo's src; no model):  python check_datetime_tool.py
"""
import os
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

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
    from amadeo_utils.ai.llm.tools import builtin_tools as B

    print("== everyday names, cities and IANA names ==")
    expected = {
        "America/New_York": ["EST", "EDT", "ET", "Eastern", "eastern", "Eastern Standard Time", "eastern daylight time",
                             "US/Eastern", "America/New_York", "New York", "new york"],
        "America/Chicago": ["CST", "CDT", "Central", "Central Time"],
        "America/Denver": ["MST", "MDT", "Mountain"],
        "America/Phoenix": ["Arizona"],
        "America/Los_Angeles": ["PST", "PDT", "PT", "Pacific", "Pacific Time", "Los Angeles"],
        "America/Anchorage": ["AKST", "Alaska"],
        "Pacific/Honolulu": ["HST", "Hawaii"],
        "UTC": ["UTC", "GMT", "Z", "utc"],
        "Europe/London": ["Europe/London", "London"],
        "Asia/Tokyo": ["Tokyo"],
        "Europe/Paris": ["Europe/Paris"],
    }
    for zone, names in expected.items():
        got = {n: B.resolve_timezone(n)[1] for n in names}
        wrong = {n: z for n, z in got.items() if z != zone}
        check(not wrong, f"{zone}: {', '.join(names)}", str(wrong))
    for bad in ("Mars", "Eastern Mars", "Nowhere/Land", 7):
        try:
            B.resolve_timezone(bad)
            check(False, f"refused: {bad!r}", "it was accepted")
        except ValueError:
            check(True, f"refused: {bad!r}")
    check(B.resolve_timezone(None, "America/Denver")[1] == "America/Denver" and
          B.resolve_timezone("  ", "America/Denver")[1] == "America/Denver", "no zone -> the default")

    print("== 'EST' now answers with New York's CURRENT offset ==")
    ny = datetime.now(ZoneInfo("America/New_York"))
    ny_offset = ny.strftime("%z")
    ny_offset = f"UTC{ny_offset[:3]}:{ny_offset[3:]}"
    est = B.get_datetime("EST")
    check(f"{ny.tzname()} (America/New_York, {ny_offset})" in est,
          f"'EST' -> {ny.tzname()} {ny_offset}, America/New_York (was a fixed UTC-05:00)", est)
    check("was read as America/New_York" in est, "and it says how it read the name", est)
    check("Note:" not in B.get_datetime("America/New_York"), "an exact IANA name adds no note")

    print("== the default ==")
    tool = B.make_get_datetime_tool("America/Denver")
    out = tool.function()
    check("(America/Denver," in out and "America/Denver" in tool.description, "an explicit default_timezone is used",
          out)
    local = B.local_timezone()
    out = B.make_get_datetime_tool(None).function()
    check(f"({local}," in out, f"no default_timezone -> this machine's own zone ({local})", out)
    try:
        B.make_get_datetime_tool("Nowhere/Land")
        check(False, "an unknown default_timezone stops the tool being built", "it was built")
    except ValueError:
        check(True, "an unknown default_timezone stops the tool being built")
    check("never convert" in tool.description.lower() and "EACH timezone" in tool.description,
          "the description tells the model to call it per zone, never convert itself")

    print("== the output reads unambiguously ==")
    out = B.get_datetime("Tokyo")
    shape = re.fullmatch(r"\w+day, \d{4}-\d{2}-\d{2}, \d{1,2}:\d{2}:\d{2} (AM|PM) \S+ \(Asia/Tokyo, UTC\+09:00\); "
                         r"ISO \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+09:00\. Note: 'Tokyo' was read as Asia/Tokyo\.", out)
    check(bool(shape), "weekday, date, 12-hour time + abbreviation, IANA name, offset, ISO, note", out)

    print("\nALL CHECKS PASS" if not failures else f"\n{len(failures)} check(s) FAILED")
    sys.exit(len(failures))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CS-24: home_control against the REAL Home Assistant - read only ('list' and 'status'). A person reads the output.

It never turns anything on or off: only 'list' and 'status' are sent, through the real tool contract, with the
tool's real config ('home_control.json' in ai_tools_config_dir, from tests/local_settings.json). Not in a suite
(see tests/README.md: live_* scripts are run by hand).

Usage (in the agent-tools env, on a machine that can read the token file):
    PYTHONPATH=<repo>/src <agent-tools python> live_home_control.py ["status target" ...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testlib import settings  # noqa: E402
settings.use_src()
from amadeo_utils.ai.llm.tools import registry as R                  # noqa: E402
from amadeo_utils.ai.llm.tools.script_tools import load_script_tool   # noqa: E402

SCRIPT = os.path.join(settings.REPO_ROOT, "scripts", "ai", "ai-tools", "home_control", "home_control.py")


def main():
    config = os.path.join(settings.get("ai_tools_config_dir"), "home_control.json")
    tool = load_script_tool(sys.executable, SCRIPT, config)
    outcome = R.execute(tool, {"action": "list"}, 200000)
    print(f"list: ok={outcome.ok} summary={outcome.summary!r}")
    if outcome.ok:
        for room, items in json.loads(outcome.content)["rooms"].items():
            print(f"  {room}")
            for item in items:
                print(f"    {item['name']:40} {item['kind']:14} {item['state']:12} {item.get('blocked', '')}")
    else:
        print("  " + outcome.content)
    for target in sys.argv[1:] or ["thermostat"]:
        outcome = R.execute(tool, {"action": "status", "target": target}, 20000)
        print(f"status {target!r}: ok={outcome.ok} summary={outcome.summary!r}")
        print("  " + outcome.content[:1000])


if __name__ == "__main__":
    main()

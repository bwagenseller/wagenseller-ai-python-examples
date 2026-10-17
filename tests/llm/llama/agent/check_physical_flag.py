#!/usr/bin/env python3
"""CS-24: the 'physical' tool flag in ToolPolicy (tools/registry.py).

What it proves
--------------
A tool flagged physical (home_control: it acts on the house):
* is offered to the main loop in every mode that offers direct tools ('auto', 'direct'), and after private data
  has entered the conversation (it is not outbound);
* is NEVER offered to a delegate worker, and a worker's call to it is refused - a worker reads untrusted pages;
* in 'direct' mode, is switched off once untrusted text has reached the model, with a refusal naming the house;
* needs no approval;
* and an unflagged tool is unaffected by all of this.

Usage (any env with the library on its path):
    python check_physical_flag.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # up to tests/
from testlib import settings  # noqa: E402
settings.use_src()
from amadeo_utils.ai.llm.tools import registry as R  # noqa: E402

failures = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if not ok and detail else ""))
    if not ok:
        failures.append(label)


def main():
    schema = {"type": "object", "properties": {}}
    registry = R.ToolRegistry()
    registry.register(R.Tool("home_control", "acts on the house", schema, lambda: "ok", physical=True))
    registry.register(R.Tool("get_weather", "forecast", schema, lambda: "ok", outbound=True))
    registry.register(R.Tool("web_search", "search", schema, lambda: "ok", outbound=True, untrusted_output=True))
    registry.register(R.Tool("get_grades", "grades", schema, lambda: "ok", private=True))
    allowed = ["home_control", "get_weather", "web_search", "get_grades"]
    clean, private, untrusted = R.Taint(), R.Taint(private=True), R.Taint(untrusted=True)

    auto = R.ToolPolicy(registry, allowed, R.MODE_AUTO)
    check("home_control" in auto.offered_names(clean), "auto: the main loop is offered home_control")
    check("home_control" in auto.offered_names(private), "auto: still offered after private data (not outbound)")
    check("home_control" in auto.offered_names(untrusted),
          "auto: the quarantine keeps untrusted text out of the main model, so it stays on")

    worker = auto.worker_policy()
    check("home_control" not in worker.offered_names(clean), "a delegate worker is never offered home_control",
          str(worker.offered_names(clean)))
    check("web_search" in worker.offered_names(clean), "...while the worker keeps web_search")
    check(worker.refusal("home_control", clean) is not None, "a worker's call to home_control is refused")

    direct = R.ToolPolicy(registry, allowed, R.MODE_DIRECT)
    check("home_control" in direct.offered_names(clean), "direct: offered before anything untrusted is read")
    check("home_control" not in direct.offered_names(untrusted), "direct: switched off after untrusted text")
    reason = direct.refusal("home_control", untrusted) or ""
    check("acts on the house" in reason, "direct: the refusal names the house", reason)
    check("home_control" in direct.offered_names(private), "direct: private data alone does not switch it off")

    check(not registry.get("home_control").requires_approval, "physical forces no approval")
    check(registry.get("get_weather").acts_on_something is False and registry.get("home_control").acts_on_something,
          "acts_on_something: physical yes, an ordinary tool no")
    check("get_weather" in worker.offered_names(clean) and "get_grades" not in worker.offered_names(clean),
          "unchanged: workers keep outbound tools and never get private ones")

    print("\nALL CHECKS PASS" if not failures else f"\n{len(failures)} check(s) FAILED")
    sys.exit(len(failures))


if __name__ == "__main__":
    main()

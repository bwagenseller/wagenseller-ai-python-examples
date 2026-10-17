#!/usr/bin/env python3
"""CS-24: the home_control tool against a fake Home Assistant (fake_home_assistant.py; a made-up house).

What it proves
--------------
* The tool describes itself as 'physical' only (not outbound, private, untrusted_output or internal_commands),
  and the server's loader accepts that flag.
* Resolution: exact name, alias, area (by name, by alias, with and without a kind word, "all the lights"), fuzzy
  for a misspelling; an ambiguous name and an unknown one are errors naming the candidates, and nothing is called.
* The skips: a hidden switch (the original behind "Show as light") never doubles a match; disabled, config and
  alexa_media entities and 'exclude' entries are unreachable, even by name.
* The rules: locks, alarm panels and non-window covers are refused unless listed; listed ones work; a window
  covering works by default; an opaque scene is refused, an inspectable one whose members all pass works, one with
  a blocked member is refused; a group with an excluded member is refused and named when an area includes it; a
  script is refused even when listed. Every refusal leaves the fake's call log empty.
* Values: brightness, fan and cover levels clamped to 0-100 (level 0 on a light is turn_off); an on/off light cannot
  be dimmed; the thermostat is clamped to its own min/max and the result says so; non-numbers are refused.
* status works on blocked entities (read only) and on a sensor-only room; list marks blocked devices.
* A failing device call is reported as a failure; a bad token and a group-readable token file are refused.

Usage (in the agent-tools env):
    PYTHONPATH=<repo>/src <agent-tools python> check_home_control.py
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testlib import settings  # noqa: E402
settings.use_src()
from amadeo_utils.ai.llm.tools import registry as R                  # noqa: E402
from amadeo_utils.ai.llm.tools.script_tools import load_script_tool   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fake_home_assistant as fake                                    # noqa: E402

SCRIPT = os.path.join(settings.REPO_ROOT, "scripts", "ai", "ai-tools", "home_control", "home_control.py")
TOKEN = "made-up-test-token"
failures = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if not ok and detail else ""))
    if not ok:
        failures.append(label)


def main():
    work = tempfile.mkdtemp(prefix="cs24-home-")
    try:
        with fake.FakeHomeAssistant(token=TOKEN) as ha:
            run_checks(work, ha)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("\nALL CHECKS PASS" if not failures else f"\n{len(failures)} check(s) FAILED")
    sys.exit(len(failures))


def write_config(work, name, ha_url, token=TOKEN, token_mode=0o600, **overrides):
    """A token file and a tool config for the fake; returns the config's path."""
    token_file = os.path.join(work, f"{name}.token")
    with open(token_file, "w") as fh:
        fh.write(token + "\n")
    os.chmod(token_file, token_mode)
    config = dict(fake.CONFIG, base_url=ha_url, secrets_file=token_file, **overrides)
    path = os.path.join(work, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(config, fh)
    return path


def run_checks(work, ha):
    print("== definition")
    tool = load_script_tool(sys.executable, SCRIPT, write_config(work, "main", ha.url))
    check(tool.name == "home_control" and tool.physical, "home_control describes itself as physical")
    check(not (tool.outbound or tool.private or tool.untrusted_output) and tool.internal_commands == "none",
          "not outbound, private, untrusted_output or internal_commands")
    check(not tool.requires_approval, "needs no approval")

    def call(arguments, tool_=tool):
        """Runs one call through the real contract; returns (ok, parsed result or error text, summary, calls)."""
        before = len(ha.calls)
        outcome = R.execute(tool_, arguments, 20000)
        try:
            content = json.loads(outcome.content) if outcome.ok else outcome.content
        except ValueError:
            content = outcome.content
        return outcome.ok, content, outcome.summary, ha.calls[before:]

    def ids(calls):
        return sorted(e for c in calls for e in c["target"]["entity_id"])

    print("== resolution")
    ok, res, summary, calls = call({"action": "turn_off", "target": "Kitchen Ceiling"})
    check(ok and ids(calls) == ["light.kitchen_ceiling"] and calls[0]["service"] == "turn_off",
          "exact name -> one light; the hidden switch with the same name does not make it ambiguous", f"{res} {calls}")
    check(summary == "Turned off Kitchen Ceiling", "summary says what happened", summary)
    ok, res, _, calls = call({"action": "turn_on", "target": "the lamp by the sink"})
    check(ok and ids(calls) == ["light.counter_lamp"], "alias (with 'the')", f"{res} {calls}")
    ok, res, summary, calls = call({"action": "turn_off", "target": "kitchen lights"})
    check(ok and ids(calls) == ["light.counter_lamp", "light.kitchen_ceiling"], "area + kind: kitchen lights only",
          f"{res} {calls}")
    check(summary == "Turned off 2 in Kitchen", "area summary", summary)
    ok, res, _, calls = call({"action": "turn_off", "target": "everything in the kitchen"})
    check(ok and ids(calls) == ["light.counter_lamp", "light.kitchen_ceiling", "switch.coffee_maker"],
          "area alone: lights and switches; not the sensor, hidden, disabled, config or alexa_media switches",
          f"{res} {calls}")
    ok, res, _, calls = call({"action": "turn_off", "target": "cookhouse"})
    check(ok and len(calls) == 3, "area alias", f"{res} {calls}")
    ok, res, _, calls = call({"action": "turn_on", "target": "kitchen cieling"})
    check(ok and ids(calls) == ["light.kitchen_ceiling"], "fuzzy match for a misspelling", f"{res} {calls}")
    ok, res, _, calls = call({"action": "turn_on", "target": "reading lamp"})
    check(not ok and "Reading Lamp (Den)" in res and "Reading Lamp (Porch)" in res and not calls,
          "ambiguous name -> error naming both, nothing called", res)
    ok, res, _, calls = call({"action": "turn_on", "target": "hot tub"})
    check(not ok and "nothing is called" in res and not calls, "unknown name -> error, nothing called", res)
    ok, res, _, calls = call({"action": "turn_on"})
    check(not ok and "target is required" in res and not calls, "missing target -> error", res)

    print("== skipped and excluded entities")
    for target in ("server power", "kitchen speaker do not disturb", "kitchen old plug", "kitchen firmware beta"):
        ok, res, _, calls = call({"action": "turn_off", "target": target})
        check(not ok and not calls, f"unreachable by name: {target}", f"{res} {calls}")

    print("== the rules")
    ok, res, _, calls = call({"action": "unlock", "target": "front door"})
    check(not ok and "listed in the tool config" in res and not calls, "unlisted lock refused", res)
    ok, res, _, calls = call({"action": "unlock", "target": "back door"})
    check(ok and calls == [{"domain": "lock", "service": "unlock", "service_data": {},
                            "target": {"entity_id": ["lock.back_door"]}}], "listed lock works", f"{res} {calls}")
    ok, res, _, calls = call({"action": "open", "target": "garage door"})
    check(not ok and "non-window covers" in res and not calls, "garage door (class 'door') refused", res)
    ok, res, _, calls = call({"action": "open", "target": "side garage door"})
    check(ok and calls and calls[0]["service"] == "open_cover", "listed garage door opens", f"{res} {calls}")
    ok, res, summary, calls = call({"action": "open", "target": "garage"})
    check(ok and ids(calls) == ["cover.side_garage_door"] and any("Garage Door" in s for s in res["skipped"])
          and "skipped 1" in summary, "area: the blocked door is skipped and named, the listed one opens",
          f"{res} {summary} {calls}")
    ok, res, _, calls = call({"action": "close", "target": "den blinds"})
    check(ok and calls and calls[0]["service"] == "close_cover", "window covering works by default", f"{res} {calls}")
    ok, res, _, calls = call({"action": "turn_off", "target": "home alarm"})
    check(not ok and not calls, "alarm panel cannot be disarmed", res)
    ok, res, _, calls = call({"action": "turn_on", "target": "movie night"})
    check(not ok and "not visible" in res and not calls, "opaque scene refused", res)
    ok, res, _, calls = call({"action": "turn_on", "target": "evening"})
    check(ok and calls and calls[0]["domain"] == "scene", "inspectable scene with allowed members works",
          f"{res} {calls}")
    ok, res, _, calls = call({"action": "turn_on", "target": "lock up"})
    check(not ok and "members is blocked" in res and not calls, "scene with a lock member refused", res)
    ok, res, _, calls = call({"action": "turn_on", "target": "porch group"})
    check(not ok and not calls, "group with an excluded member refused", res)
    ok, res, _, calls = call({"action": "turn_off", "target": "all the lights"})
    check(ok and "light.porch_group" not in ids(calls) and "light.kitchen_ceiling" in ids(calls)
          and any("Porch Group" in s for s in res["skipped"]), "whole house: the group is skipped and named",
          f"{res} {calls}")
    ok, res, _, calls = call({"action": "turn_on", "target": "unlock everything"})
    check(not ok and not calls, "script refused, even though it is in explicit_allow", res)

    spec = importlib.util.spec_from_file_location("home_control", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    house = module.House(fake.STATES, fake.ENTITIES, fake.DEVICES, fake.AREAS, fake.CONFIG)
    allow = module.id_set(fake.CONFIG, "explicit_allow")
    check(house.refusal("script.unlock_everything", allow) and house.refusal("automation.night_mode", allow),
          "the rules: script and automation blocked outright")

    print("== values")
    ok, res, _, calls = call({"action": "set_level", "target": "kitchen ceiling", "level": 40})
    check(ok and calls[0]["service_data"] == {"brightness_pct": 40}, "dim to 40%", f"{res} {calls}")
    ok, res, _, calls = call({"action": "set_level", "target": "kitchen ceiling", "level": 0})
    check(ok and calls[0]["service"] == "turn_off", "level 0 on a light is turn_off", f"{res} {calls}")
    ok, res, _, calls = call({"action": "set_level", "target": "den fan", "level": 250})
    check(ok and calls[0]["service_data"] == {"percentage": 100} and "note" in res, "fan level clamped to 100",
          f"{res} {calls}")
    ok, res, _, calls = call({"action": "set_level", "target": "den blinds", "level": -5})
    check(ok and calls[0]["service_data"] == {"position": 0}, "cover position clamped to 0", f"{res} {calls}")
    ok, res, _, calls = call({"action": "set_level", "target": "counter lamp", "level": 50})
    check(not ok and "not dimmed" in res and not calls, "an on/off light cannot be dimmed", res)
    ok, res, summary, calls = call({"action": "set_temperature", "target": "thermostat", "temperature": 120})
    check(ok and calls[0]["service_data"] == {"temperature": 90} and "range" in res.get("note", "")
          and summary == "Set Thermostat to 90", "thermostat clamped to its max, and the result says so",
          f"{res} {summary} {calls}")
    ok, res, _, calls = call({"action": "set_temperature", "target": "thermostat", "temperature": "warm"})
    check(not ok and "must be a number" in res and not calls, "non-numeric temperature refused", res)
    ok, res, _, calls = call({"action": "set_temperature", "target": "den fan", "temperature": 70})
    check(not ok and not calls, "set_temperature on a fan refused", res)

    print("== status and list")
    ok, res, summary, calls = call({"action": "status", "target": "front door"})
    check(ok and res["devices"][0]["state"] == "locked" and not calls, "status reads a blocked lock (no call)",
          f"{res} {calls}")
    ok, res, _, _ = call({"action": "status", "target": "garage door"})
    check(ok and sorted(d["kind"] for d in res["devices"]) == ["binary_sensor", "cover"],
          "status of a name shared by a cover and its sensor shows both", res)
    labels = house._list([e for e in house.entities if e.name == "Garage Door"])
    check(labels == "Garage Door (Garage, cover) or Garage Door (Garage, binary sensor)",
          "an ambiguity error tells same-named devices apart by kind", labels)
    ok, res, _, _ = call({"action": "status", "target": "attic"})
    check(ok and [d["name"] for d in res["devices"]] == ["Attic Leak"], "status of a sensor-only room", res)
    ok, res, _, _ = call({"action": "status", "target": "server power"})
    check(not ok, "status cannot read an excluded entity", res)
    ok, res, _, calls = call({"action": "list"})
    rooms = res["rooms"] if ok else {}
    names = {i["name"]: i for items in rooms.values() for i in items}
    check(ok and "Kitchen Ceiling" in names and "Server Power" not in names and "Kitchen Speaker Do Not Disturb"
          not in names and "Kitchen Temperature" not in names and not calls,
          "list: controllable only; excluded, alexa_media and sensors absent", json.dumps(res)[:300])
    check("blocked" in names.get("Front Door", {}) and "blocked" not in names.get("Back Door", {"blocked": 1}),
          "list marks the unlisted lock as blocked, not the listed one")

    print("== failures")
    ok, res, _, calls = call({"action": "turn_on", "target": "failing plug"})
    check(not ok and "did not respond" in res and len(calls) == 1, "a failing device call is reported", res)
    bad = load_script_tool(sys.executable, SCRIPT, write_config(work, "bad", ha.url, token="wrong"))
    ok, res, _, calls = call({"action": "list"}, bad)
    check(not ok and "refused the access token" in res and "wrong" not in res, "bad token refused (and not echoed)",
          res)
    loose = load_script_tool(sys.executable, SCRIPT, write_config(work, "loose", ha.url, token_mode=0o640))
    ok, res, _, calls = call({"action": "list"}, loose)
    check(not ok and "readable by its group" in res, "group-readable token file refused", res)
    down = load_script_tool(sys.executable, SCRIPT, write_config(work, "down", "http://127.0.0.1:9"))
    ok, res, _, _ = call({"action": "list"}, down)
    check(not ok and "cannot reach Home Assistant" in res, "Home Assistant down -> a clear error", res)


if __name__ == "__main__":
    main()

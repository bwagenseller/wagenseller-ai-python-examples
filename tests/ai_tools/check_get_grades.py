#!/usr/bin/env python3
"""CS-21: the get_grades tool, offline - no browser, no login, no credentials.

What it proves
--------------
The tool runs through the real tool-script contract (load_script_tool + registry.execute: minimal
environment, arguments on stdin) against a STAND-IN for infinitecampus.py. The stand-in executes the
REAL infinitecampus.py source - so the real parse_grades / parse_assignments / build_json_output are
what is tested - with playwright and dotenv replaced by fakes and only login_and_fetch swapped for
fixture data. Like the real one, the fake login prints its progress to stdout.

* It describes itself as private and not outbound, and the model's only arguments are term and days.
* Secrets file: not configured, missing, or readable by group/others -> refused before anything runs;
  0600 -> accepted, and SECRETS_FILE reaches infinitecampus.py's dotenv_values; the password it reads
  never lands in os.environ (so Playwright and Chromium do not inherit it). A group may read it only
  when the config names that group as secrets_trusted_group; world-readable is always refused.
* Arguments: an unknown term is refused, lower case is accepted, days is capped at 60, a non-number
  is refused.
* The answer survives infinitecampus.py printing to stdout (it is moved to stderr).
* Grades only for the requested term; assignments only inside the window; missing work flagged;
  long teacher comments capped; HTML-escaped names unescaped; work due after today marked upcoming.
* A failed login is an ERROR ("could not log in"), never an empty grade list.
* The history summary holds counts only - no course, grade, assignment or comment text.
* infinitecampus.py's --secrets-file: read, and wins over SECRETS_FILE; a missing file exits 2. Headless by
  default; --show-browser opens a visible browser.
* browser "headless" sets HEADLESS; max_login_attempts is passed and clamped to 3; "xvfb" either
  runs under a fresh DISPLAY (if Xvfb is installed) or fails cleanly saying how to install it.

Usage (agent-tools env):
    PYTHONPATH=<repo>/src <agent-tools python> check_get_grades.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testlib import settings  # noqa: E402
settings.use_src()
SRC = settings.SRC
ROOT = settings.REPO_ROOT
from amadeo_utils.ai.llm.tools import registry as R                  # noqa: E402
from amadeo_utils.ai.llm.tools.script_tools import load_script_tool   # noqa: E402

TOOL = os.path.join(ROOT, "scripts", "ai", "ai-tools", "get_grades", "get_grades.py")
REAL_IC = os.path.join(ROOT, "scripts", "tools", "infinite_campus", "infinitecampus.py")
failures = []

# The stand-in. MODE and RECORD are filled in per copy; the tool's minimal environment carries no test settings.
STUB = r'''
import json, os, sys, types
MODE, RECORD, REAL = {mode!r}, {record!r}, {real!r}

playwright = types.ModuleType("playwright")
sync_api = types.ModuleType("playwright.sync_api")
def sync_playwright():
    raise RuntimeError("the stand-in never opens a browser")
sync_api.sync_playwright = sync_playwright
dotenv = types.ModuleType("dotenv")
LOADED = []
def dotenv_values(path=None, *args, **kwargs):
    # A minimal reader of KEY=VALUE lines, standing in for python-dotenv's (which is not in this env).
    LOADED.append(path)
    values = {{}}
    for line in open(path) if path else []:
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.strip().partition("=")
            values[key.strip()] = value.strip()
    return values
dotenv.dotenv_values = dotenv_values
sys.modules.update({{"playwright": playwright, "playwright.sync_api": sync_api, "dotenv": dotenv}})

exec(compile(open(REAL).read(), REAL, "exec"))      # the real module body: constants, parsers, JSON builder

from datetime import datetime, timedelta
def _due(days_ago):
    # infinitecampus.py subtracts 5 hours from the UTC due date; add them back so the local date is 'days_ago'.
    return (datetime.now() - timedelta(days=days_ago) + timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

GRADES = [{{"termName": "T1", "courses": [
    {{"courseName": "Zz Algebra II", "teacherDisplay": "Teacher A",
      "gradingTasks": [{{"taskName": "Trimester Grade", "progressScore": "B+", "progressPercent": 88.5}}]}},
    {{"courseName": "Aa Biology", "teacherDisplay": "Teacher B",
      "gradingTasks": [{{"taskName": "Trimester Grade", "progressScore": "A", "progressPercent": 95.25}}]}}]}},
    {{"termName": "T2", "courses": [
    {{"courseName": "Other Term Course", "teacherDisplay": "Teacher C", "gradingTasks": []}}]}}]
ASSIGNMENTS = [
    {{"assignmentName": "Recent quiz", "courseName": "Aa Biology", "dueDate": _due(2), "score": 9, "totalPoints": 10}},
    {{"assignmentName": "Sources &amp; Notes", "courseName": "Zz Algebra II", "dueDate": _due(1), "score": 4, "totalPoints": 4}},
    {{"assignmentName": "Upcoming test", "courseName": "Zz Algebra II", "dueDate": _due(-4)}},
    {{"assignmentName": "Missing lab", "courseName": "Aa Biology", "dueDate": _due(3), "missing": True,
      "comments": "Please turn this in. " * 40}},
    {{"assignmentName": "Old homework", "courseName": "Zz Algebra II", "dueDate": _due(40), "score": 5, "totalPoints": 5}},
]

def login_and_fetch(max_retries=3):
    print("Login attempt 1 of %d..." % max_retries)          # stdout, exactly like the real one
    print("  Entering Microsoft credentials...")
    with open(RECORD, "w") as fh:
        json.dump({{"max_retries": max_retries, "headless": HEADLESS, "display": os.environ.get("DISPLAY"),
                   "secrets_file_env": os.environ.get("SECRETS_FILE"), "dotenv_values_path": LOADED,
                   "password_in_module": INFINITE_CAMPUS_MS_SSO_PASSWORD,
                   "password_in_environ": os.environ.get("INFINITE_CAMPUS_MS_SSO_PASSWORD")}}, fh)
    if MODE == "fail":
        return None, None
    return GRADES, ASSIGNMENTS
'''


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if not ok and detail else ""))
    if not ok:
        failures.append(label)


def main():
    work = tempfile.mkdtemp(prefix="cs21-grades-")
    try:
        run_checks(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("\nALL CHECKS PASS" if not failures else f"\n{len(failures)} check(s) FAILED")
    sys.exit(len(failures))


def run_checks(work):
    record = os.path.join(work, "record.json")
    stubs = {}
    for mode in ("ok", "fail"):
        stubs[mode] = os.path.join(work, f"ic_stub_{mode}.py")
        with open(stubs[mode], "w") as fh:
            fh.write(STUB.format(mode=mode, record=record, real=REAL_IC))
    secrets = os.path.join(work, "ic.env")
    with open(secrets, "w") as fh:
        fh.write("# fake - read by the stand-in's dotenv_values only\nINFINITE_CAMPUS_MS_SSO_PASSWORD=fake-password\n")
    os.chmod(secrets, 0o600)

    def tool_with(**config):
        """The real tool through the real contract, with this config."""
        path = os.path.join(work, "config.json")
        base = {"secrets_file": secrets, "browser": "headless", "infinitecampus_script": stubs["ok"]}
        base.update(config)
        with open(path, "w") as fh:
            json.dump({k: v for k, v in base.items() if v is not None}, fh)
        return load_script_tool(sys.executable, TOOL, path)

    def call(tool, arguments):
        if os.path.exists(record):
            os.remove(record)
        out = R.execute(tool, arguments, 1000000)
        recorded = json.load(open(record)) if os.path.exists(record) else None
        return out, (json.loads(out.content) if out.ok else None), recorded

    print("== definition ==")
    tool = tool_with()
    check(tool.private and not tool.outbound and not tool.untrusted_output, "private, not outbound, not untrusted_output")
    described = json.loads(os.popen(f"{sys.executable} {TOOL} --describe").read())
    check(set(described["parameters"]["properties"]) == {"term", "days"}, "the model's only arguments are term and days",
          str(described["parameters"]["properties"].keys()))

    print("== secrets file ==")
    out, _, rec = call(tool_with(secrets_file=None), {})
    check(not out.ok and "secrets_file" in out.content and rec is None, "no secrets_file configured -> refused, nothing run",
          out.content)
    out, _, rec = call(tool_with(secrets_file=os.path.join(work, "nope.env")), {})
    check(not out.ok and "cannot read its secrets file" in out.content and rec is None, "missing secrets file -> refused",
          out.content)
    for mode in (0o640, 0o604, 0o660):
        os.chmod(secrets, mode)
        out, _, rec = call(tool_with(), {})
        check(not out.ok and "refuses to run" in out.content and rec is None,
              f"secrets file mode {mode:o} -> refused before anything runs (no trusted group)", out.content)
    # A trusted group: the file's own group may read it when the config names that group; "other" never may.
    own_group = __import__("grp").getgrgid(os.stat(secrets).st_gid).gr_name
    os.chmod(secrets, 0o660)
    out, _, rec = call(tool_with(secrets_trusted_group=own_group), {"term": "T1"})
    check(out.ok and rec is not None, f"mode 660 accepted when secrets_trusted_group names the file's group ({own_group})",
          out.content[:200])
    out, _, rec = call(tool_with(secrets_trusted_group="root" if own_group != "root" else "daemon"), {"term": "T1"})
    check(not out.ok and "not this tool's secrets_trusted_group" in out.content and rec is None,
          "mode 660 refused when secrets_trusted_group names a different group", out.content)
    out, _, rec = call(tool_with(secrets_trusted_group="no-such-group-cs21"), {"term": "T1"})
    check(not out.ok and rec is None, "mode 660 refused when secrets_trusted_group does not exist", out.content)
    os.chmod(secrets, 0o664)
    out, _, rec = call(tool_with(secrets_trusted_group=own_group), {"term": "T1"})
    check(not out.ok and "readable by every user" in out.content and rec is None,
          "world-readable is refused even with a trusted group", out.content)
    os.chmod(secrets, 0o600)
    out, data, rec = call(tool_with(), {"term": "T1"})
    check(out.ok and rec and rec["secrets_file_env"] == secrets and rec["dotenv_values_path"] == [secrets],
          "mode 600 accepted; SECRETS_FILE reaches infinitecampus.py's dotenv_values", out.content[:200] + f" rec={rec}")
    check(rec and rec["password_in_module"] == "fake-password" and rec["password_in_environ"] is None,
          "the password is read from the file but never copied into os.environ (inherited by the browser)", str(rec))

    print("== arguments ==")
    out, _, rec = call(tool_with(), {"term": "T9"})
    check(not out.ok and "term must be one of" in out.content, "an unknown term is refused", out.content)
    out, data, _ = call(tool_with(), {"term": "t1"})
    check(out.ok and data["term"] == "T1", "a lower-case term is accepted")
    out, data, _ = call(tool_with(), {"term": "T1", "days": 500})
    check(out.ok and data["days"] == 60, "days is capped at 60", out.content[:200])
    out, _, _ = call(tool_with(), {"term": "T1", "days": "lots"})
    check(not out.ok and "whole number" in out.content, "a non-number days is refused", out.content)
    out, data, _ = call(tool_with(), {})
    check(out.ok and data["term"] == "T3", "no term -> infinitecampus.py's TARGET_TERM", out.content[:120])
    out, data, _ = call(tool_with(default_term="T1"), {})
    check(out.ok and data["term"] == "T1", "no term -> the config's default_term when set", out.content[:120])

    print("== the result ==")
    out, data, rec = call(tool_with(), {"term": "T1", "days": 14})
    check(out.ok, "the answer parses although infinitecampus.py printed to stdout", out.content[:200])
    if out.ok:
        courses = [g["course"] for g in data["grades"]]
        check(courses == ["Aa Biology", "Zz Algebra II"], "grades only for the requested term, sorted", str(courses))
        check(data["grades"][0]["grade"] == "A" and data["grades"][0]["percent"] == "95.25%",
              "grade and percent come from the real parser", str(data["grades"][0]))
        items = [i for day in data["assignments"] for i in day["items"]]
        check(sorted(i["name"] for i in items) == ["Missing lab", "Recent quiz", "Sources & Notes", "Upcoming test"],
              "assignments only inside the window (plus upcoming ones)", str([i["name"] for i in items]))
        check(not any("&amp;" in json.dumps(day) for day in data["assignments"]),
              "HTML-escaped names are unescaped ('&amp;' -> '&')")
        flags = {i["name"]: day.get("upcoming") for day in data["assignments"] for i in day["items"]}
        check(flags == {"Recent quiz": False, "Sources & Notes": False, "Missing lab": False, "Upcoming test": True},
              "days after today are marked upcoming; past days are not", str(flags))
        lab = next(i for i in items if i["name"] == "Missing lab")
        check(lab["missing"] is True and len(lab["comments"]) <= 301 and lab["comments"].endswith("…"),
              "missing work flagged; a long teacher comment is capped", f"len={len(lab['comments'])}")
        summary = out.summary or ""
        leaked = [w for w in ("Biology", "Algebra", "Recent quiz", "Missing lab", "Sources", "Upcoming test", "95", "B+",
                              "Please turn", "Teacher")
                  if w in summary]
        check(summary == "T1: grades for 2 courses; 3 assignments due in the last 14 days, 1 missing; 1 upcoming"
              and not leaked,
              f"the history summary is counts only: {summary!r}", f"leaked={leaked}")
    check(rec and rec["headless"] is True and rec["max_retries"] == 2,
          "browser 'headless' sets HEADLESS; default login attempts is 2", str(rec))
    _, _, rec = call(tool_with(max_login_attempts=9), {"term": "T1"})
    check(rec and rec["max_retries"] == 3, "max_login_attempts is clamped to 3", str(rec))

    print("== failure ==")
    out, _, rec = call(tool_with(infinitecampus_script=stubs["fail"]), {"term": "T1"})
    check(not out.ok and "could not log in" in out.content and rec is not None,
          "a failed login is an error, never an empty grade list", out.content)
    out, _, _ = call(tool_with(browser="firefox"), {"term": "T1"})
    check(not out.ok and "browser must be one of" in out.content, "an unknown browser setting is refused", out.content)

    print("== infinitecampus.py itself: a failed login prints nothing and exits 1 ==")
    runner = os.path.join(work, "run_main.py")
    with open(runner, "w") as fh:
        fh.write("import runpy, sys\n"
                 "ns = runpy.run_path(sys.argv[1])\n"               # the stand-in: real source, fake browser, failing login
                 "sys.argv = ['infinitecampus.py'] + sys.argv[2:]\n"
                 "ns['main']()\n")
    for flags in (["--json"], ["--email"], ["--print"]):
        env = dict(os.environ, SECRETS_FILE=secrets, PYTHONPATH=SRC)
        done = subprocess.run([sys.executable, runner, stubs["fail"], *flags], capture_output=True, text=True, env=env,
                              timeout=60)
        # stdout holds only the stand-in's own progress lines ("Login attempt...") - nothing from main()'s output modes
        leaked = [l for l in done.stdout.splitlines() if not l.startswith(("Login attempt", "  Entering"))]
        check(done.returncode == 1 and not leaked and "Could not fetch anything" in done.stderr,
              f"{flags[0]}: exit 1, no summary/JSON/email, the reason on stderr",
              f"rc={done.returncode} leaked={leaked[:3]} stderr={done.stderr[-200:]}")

    print("== infinitecampus.py --secrets-file ==")
    flag_env = os.path.join(work, "from_flag.env")
    with open(flag_env, "w") as fh:
        fh.write("INFINITE_CAMPUS_MS_SSO_LOGIN=login-from-flag\n")
    probe = os.path.join(work, "run_probe.py")
    with open(probe, "w") as fh:
        fh.write("import runpy, sys\n"
                 "ns = runpy.run_path(sys.argv[1])\n"
                 "sys.argv = ['infinitecampus.py'] + sys.argv[2:]\n"
                 "try:\n"
                 "    ns['main']()\n"
                 "except SystemExit as e:\n"
                 "    print('LOGIN=' + str(ns['load_settings'].__globals__.get('INFINITE_CAMPUS_MS_SSO_LOGIN')), file=sys.stderr)\n"
                 "    print('HEADLESS=' + str(ns['load_settings'].__globals__.get('HEADLESS')), file=sys.stderr)\n"
                 "    raise\n")
    env = dict(os.environ, SECRETS_FILE=secrets, PYTHONPATH=SRC)
    done = subprocess.run([sys.executable, probe, stubs["fail"], "--json", "--secrets-file", flag_env],
                          capture_output=True, text=True, env=env, timeout=60)
    check("LOGIN=login-from-flag" in done.stderr, "--secrets-file is read, and wins over SECRETS_FILE",
          done.stderr[-300:])
    check("HEADLESS=True" in done.stderr, "infinitecampus.py runs headless by default", done.stderr[-200:])
    done = subprocess.run([sys.executable, probe, stubs["fail"], "--json", "--show-browser"],
                          capture_output=True, text=True, env=env, timeout=60)
    check("HEADLESS=False" in done.stderr, "--show-browser opens a visible browser", done.stderr[-200:])
    done = subprocess.run([sys.executable, probe, stubs["fail"], "--json", "--secrets-file", os.path.join(work, "nope.env")],
                          capture_output=True, text=True, env=env, timeout=60)
    check(done.returncode == 2 and "secrets file not found" in done.stderr,
          "a --secrets-file that does not exist: exit 2, with the reason", f"rc={done.returncode} {done.stderr[-200:]}")

    print("== xvfb ==")
    out, data, rec = call(tool_with(browser="xvfb"), {"term": "T1"})
    if shutil.which("Xvfb"):
        check(out.ok and rec and rec["headless"] is False and (rec["display"] or "").startswith(":"),
              f"browser 'xvfb' runs a visible browser on a private display ({rec and rec['display']})", out.content[:200])
    else:
        check(not out.ok and "sudo apt install xvfb" in out.content and rec is None,
              "browser 'xvfb' without Xvfb installed fails cleanly, saying how to fix it", out.content)


if __name__ == "__main__":
    main()

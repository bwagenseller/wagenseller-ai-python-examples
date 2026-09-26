#!/usr/bin/env python3
"""
get_grades - grades and recent assignments from Infinite Campus (CS-21 tool).

Runs as a tool script (see amadeo_utils/ai/llm/tools/script_tools.py): the tool server calls it with
the arguments on stdin and reads one JSON answer from stdout.

It reuses scripts/tools/infinite_campus/infinitecampus.py rather than copying it: that script is
loaded by path and its login_and_fetch / parse_grades / parse_assignments / build_json_output are
called directly. That script is not modified. Two things it does are handled here instead:

* It prints login progress to stdout, which would corrupt this tool's one-line answer, so stdout is
  pointed at stderr for as long as its code runs.
* If every login attempt fails it returns (None, None), which its parsers would turn into "no
  grades". Here that is an error - the model must never tell the user there are no grades when the
  truth is that the login failed.

Credentials
-----------
The credentials never enter the model's process or its context. infinitecampus.py reads them from
the .env file named by SECRETS_FILE; the tool server gives tool scripts a minimal environment, so
this script sets SECRETS_FILE itself from 'secrets_file' in its config. It refuses to run if that
file can be read by anyone but its owner (mode 0600) - unless the config names one
'secrets_trusted_group', in which case members of that group (and only when it is the file's
group) may read it too. Access for all other users is always refused.

What the model controls: only 'term' (one of the configured terms) and 'days' (1-60). There is no
argument that reaches infinitecampus.py's --email mode, or anything else.

Security flags: 'private' - the result is a child's grades. Returning it private-taints the session
in 'direct' mode (outbound tools are then disabled). Not 'outbound': the model's arguments only
choose which term and how many days, and the fetch goes to the one configured school site.
The one-line history summary holds counts only, never grades, course names or comments.

The browser
-----------
Tool scripts get no DISPLAY, so a visible browser cannot open from here. 'browser' in the config:
  "headless" - Chromium without a window (the default; the site may refuse it);
  "xvfb"     - a normal browser inside a private virtual display that this script starts and stops
               (needs the Xvfb package: sudo apt install xvfb).

Config (get_grades.json in your tool config folder - outside the repo):
    {
        "secrets_file": "/home/you/.secrets/infinite_campus.env",   # required; mode 0600 (or 0640/0660, below)
        "secrets_trusted_group": "",                                 # optional: a group allowed to read it
        "browser": "headless",                                       # or "xvfb"
        "max_login_attempts": 2,                                     # each can take ~60 s
        "terms": ["T1", "T2", "T3"],
        "default_term": "T1",                                        # else infinitecampus.TARGET_TERM
        "infinitecampus_script": ""                                  # optional; default: the repo copy
    }

Python: the env must have playwright (with its chromium installed) and python-dotenv - the same env
infinitecampus.py runs in. Set it as this tool's "python" in the tool server config.

Usage:
    get_grades.py --describe
    echo '{"term": "T1", "days": 14}' | get_grades.py --config get_grades.json
"""
import contextlib
import ctypes
import grp
import html
import importlib.util
import os
import signal
import stat
import subprocess
import sys
from datetime import datetime

from amadeo_utils.ai.llm.tools.script_tools import ToolAnswer, ToolError, tool_script_main

DEFAULT_DAYS = 14
MAX_DAYS = 60
DEFAULT_TERMS = ("T1", "T2", "T3")
DEFAULT_LOGIN_ATTEMPTS = 2
MAX_LOGIN_ATTEMPTS = 3
MAX_COMMENT_CHARS = 300         # teacher comments are free text; keep the result inside the per-result cap
BROWSERS = ("headless", "xvfb")
XVFB_START_SECONDS = 10
# scripts/ai/ai-tools/get_grades/get_grades.py -> scripts/tools/infinite_campus/infinitecampus.py
DEFAULT_IC_SCRIPT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "..", "..", "tools", "infinite_campus", "infinitecampus.py"))

DEFINITION = {
    "name": "get_grades",
    "description": "Gets the student's current grades for a school term, their recent assignments (scores, "
                   "missing work, teacher comments) and upcoming assignments, from Infinite Campus. Takes a few "
                   "seconds to a minute (it logs in).",
    "parameters": {
        "type": "object",
        "properties": {
            "term": {"type": "string", "enum": list(DEFAULT_TERMS),
                     "description": "The trimester: T1, T2 or T3. Default: the current one."},
            "days": {"type": "integer",
                     "description": f"How many days back to list assignments (1-{MAX_DAYS}). Default {DEFAULT_DAYS}."},
        },
        "required": [],
    },
    "flags": {"private": True},
    "timeout_s": 170,           # two login attempts at ~60-80 s each; the turn cap is 180 s
}


# ------------------------------------------------------------------------------------------ checks

def secrets_path(config):
    """
    The .env holding the credentials, checked before anything reads it.

    Only its owner may have access - or, if the config names a 'secrets_trusted_group', also that one group,
    when it is the file's group. Access for "other" users is always refused.

    Raises:
        ToolError: if it is not configured, missing, or readable beyond its owner (and trusted group).
    """
    path = os.path.expanduser(str(config.get("secrets_file") or ""))
    if not path:
        # Without this, infinitecampus.py's load_dotenv(None) would search upward from the working directory.
        raise ToolError("get_grades is not configured: 'secrets_file' is missing from its config")
    try:
        info = os.stat(path)
    except OSError:
        raise ToolError("get_grades cannot read its secrets file (missing, or not readable by the tool server's user)")
    if info.st_mode & stat.S_IRWXO:
        raise ToolError("get_grades refuses to run: its secrets file is readable by every user (chmod o-rwx it)")
    if info.st_mode & stat.S_IRWXG and not group_is_trusted(info.st_gid, config.get("secrets_trusted_group")):
        raise ToolError("get_grades refuses to run: its secrets file is readable by its group, and that group is not "
                        "this tool's secrets_trusted_group (chmod 600 it, or set secrets_trusted_group)")
    return path


def group_is_trusted(gid, trusted_name):
    """True if 'trusted_name' is set and is the group with this gid."""
    if not trusted_name:
        return False
    try:
        return grp.getgrnam(str(trusted_name)).gr_gid == gid
    except KeyError:
        return False


def read_arguments(arguments, config, default_term):
    """
    The model's two choices, validated.

    Returns:
        tuple[str, int]: term, days.

    Raises:
        ToolError: for a term not in the configured list, or days that is not a whole number.
    """
    terms = [str(t).upper() for t in (config.get("terms") or DEFAULT_TERMS)]
    term = str(arguments.get("term") or config.get("default_term") or default_term or terms[0]).strip().upper()
    if term not in terms:
        raise ToolError(f"term must be one of: {', '.join(terms)}")
    try:
        days = int(arguments.get("days", DEFAULT_DAYS))
    except (TypeError, ValueError):
        raise ToolError("days must be a whole number")
    return term, max(1, min(MAX_DAYS, days))


# ------------------------------------------------------------------------------------------ the browser

def _die_with_parent():
    """preexec_fn for Xvfb: the kernel sends it SIGTERM if this script dies, even by SIGKILL (a tool timeout)."""
    PR_SET_PDEATHSIG = 1
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


@contextlib.contextmanager
def virtual_display():
    """
    A private Xvfb display for the duration of the block; DISPLAY points at it inside.

    Xvfb picks a free display number itself (-displayfd) and is stopped afterwards - or by the kernel,
    if this script is killed first.

    Raises:
        ToolError: if Xvfb is not installed or does not start.
    """
    read_fd, write_fd = os.pipe()
    try:
        xvfb = subprocess.Popen(["Xvfb", "-displayfd", str(write_fd), "-nolisten", "tcp", "-screen", "0", "1280x1024x24"],
                                pass_fds=(write_fd,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                preexec_fn=_die_with_parent)
    except OSError:
        os.close(read_fd)
        os.close(write_fd)
        raise ToolError("browser is 'xvfb' but Xvfb is not installed (sudo apt install xvfb)")
    os.close(write_fd)
    try:
        import select
        ready, _, _ = select.select([read_fd], [], [], XVFB_START_SECONDS)
        number = os.read(read_fd, 32).decode().strip() if ready else ""
        if not number.isdigit():
            raise ToolError("the virtual display (Xvfb) did not start")
        previous = os.environ.get("DISPLAY")
        os.environ["DISPLAY"] = f":{number}"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("DISPLAY", None)
            else:
                os.environ["DISPLAY"] = previous
    finally:
        os.close(read_fd)
        xvfb.terminate()
        try:
            xvfb.wait(timeout=5)
        except subprocess.TimeoutExpired:
            xvfb.kill()


# ------------------------------------------------------------------------------------------ the tool

def load_infinitecampus(path, secrets):
    """
    Imports infinitecampus.py by path. It reads SECRETS_FILE and the credentials at import time, so that is set
    first. Its import-time output (if any) goes to stderr.

    Raises:
        ToolError: if the script is missing or cannot be imported (e.g. playwright not in this env).
    """
    if not os.path.isfile(path):
        raise ToolError("get_grades cannot find infinitecampus.py")
    os.environ["SECRETS_FILE"] = secrets
    spec = importlib.util.spec_from_file_location("infinitecampus", path)
    module = importlib.util.module_from_spec(spec)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            spec.loader.exec_module(module)
    except ImportError as e:
        raise ToolError(f"get_grades is missing a library in this Python env ({e.name or e})")
    return module


def tidy_output(output, today):
    """
    Cleans infinitecampus.py's JSON structure in place for the model.

    * Infinite Campus returns names HTML-escaped ("Primary &amp; Secondary Sources"): unescaped.
    * Teacher comments are free text: capped at MAX_COMMENT_CHARS so a long one cannot crowd out the rest.
    * parse_assignments keeps everything due on or after the window start - including work due AFTER today. Each
      day is marked 'upcoming' (due after today) so the model can tell "was due" from "is coming up".

    Args:
        output (dict): build_json_output's result.
        today (datetime.date): the local date the result is for.
    """
    for grade in output.get("grades", []):
        for key in ("course", "teacher"):
            if isinstance(grade.get(key), str):
                grade[key] = html.unescape(grade[key])
    for day in output.get("assignments", []):
        try:
            day["upcoming"] = datetime.strptime(day.get("date", ""), "%A %m/%d/%Y").date() > today
        except ValueError:
            pass                                     # an unreadable date is left unmarked rather than guessed
        for item in day.get("items", []):
            for key in ("name", "course", "comments"):
                if isinstance(item.get(key), str):
                    item[key] = html.unescape(item[key])
            comment = item.get("comments") or ""
            if len(comment) > MAX_COMMENT_CHARS:
                item["comments"] = comment[:MAX_COMMENT_CHARS].rstrip() + "…"


def summary_of(output, days):
    """The one line kept in chat history: counts only - no grades, course names or comments (private)."""
    past = [item for day in output.get("assignments", []) if not day.get("upcoming")
            for item in day.get("items", [])]
    upcoming = sum(len(day.get("items", [])) for day in output.get("assignments", []) if day.get("upcoming"))
    missing = sum(1 for item in past if item.get("missing"))
    return (f"{output.get('term')}: grades for {len(output.get('grades', []))} courses; {len(past)} assignments "
            f"due in the last {days} days, {missing} missing; {upcoming} upcoming")


def handle(arguments, config):
    """The tool: check the config, log in through infinitecampus.py, return its JSON structure."""
    secrets = secrets_path(config)
    browser = str(config.get("browser") or "headless").strip().lower()
    if browser not in BROWSERS:
        raise ToolError(f"get_grades is misconfigured: browser must be one of {', '.join(BROWSERS)}")
    try:
        attempts = max(1, min(MAX_LOGIN_ATTEMPTS, int(config.get("max_login_attempts", DEFAULT_LOGIN_ATTEMPTS))))
    except (TypeError, ValueError):
        raise ToolError("get_grades is misconfigured: max_login_attempts must be a whole number")

    ic = load_infinitecampus(os.path.expanduser(config.get("infinitecampus_script") or DEFAULT_IC_SCRIPT), secrets)
    term, days = read_arguments(arguments, config, getattr(ic, "TARGET_TERM", None))

    # infinitecampus.py reads its module-level HEADLESS when it launches the browser. The module was loaded privately
    # by this process, so setting it here changes nothing for anyone else.
    ic.HEADLESS = browser == "headless"
    display = virtual_display() if browser == "xvfb" else contextlib.nullcontext()
    with display, contextlib.redirect_stdout(sys.stderr):
        grades_data, assignments_data = ic.login_and_fetch(max_retries=attempts)
    if grades_data is None and assignments_data is None:
        raise ToolError("could not log in to Infinite Campus (every attempt failed) - no grades were fetched")

    with contextlib.redirect_stdout(sys.stderr):
        output = ic.build_json_output(ic.parse_grades(grades_data, term), ic.parse_assignments(assignments_data, days),
                                      term)
    tidy_output(output, datetime.now().date())
    output["days"] = days
    return ToolAnswer(output, summary_of(output, days))


if __name__ == "__main__":
    tool_script_main(DEFINITION, handle)

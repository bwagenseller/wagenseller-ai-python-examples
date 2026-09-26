"""
Tools that run as separate scripts, usually in the 'agent-tools' conda env (CS-21).

Why scripts
-----------
The tool server runs in the 'llama' env, because it needs llama_cpp. Tools with their own
dependencies (weather, fetching pages, web search) run as scripts in another env instead, which
also means: hostile web content is parsed outside the LLM server's process, a hung tool can be
killed (a thread cannot), and the tool sees none of the server's environment.

The contract - both halves live in this file
--------------------------------------------
A tool script calls ``tool_script_main(DEFINITION, handler)``. It then supports:

  script.py --describe             prints its DEFINITION as JSON: name, description, parameters,
                                   its security flags and its timeout. The script declares its own
                                   flags, because whoever writes a tool is the one who knows it sends
                                   data out or reads private data.
  script.py --config FILE          runs one call. The ARGUMENTS arrive as JSON on stdin - never on
                                   the command line, where 'ps' would show them to every user on the
                                   machine. FILE is the tool's own config (API endpoints, home
                                   location...), kept in a config folder outside the repository.

It answers on stdout with one JSON object: {"ok": true, "result": ..., "summary": "..."} (the
summary optional - see ToolAnswer) or {"ok": false, "error": "..."}. A clean failure is ok=false with a message written for the model and the user; anything
else (a crash, a non-JSON answer, a non-zero exit) is reported as the tool failing.

The server half, load_script_tool(), runs --describe once at startup and returns an ordinary
registry.Tool whose function runs the script per call.

This module deliberately imports nothing beyond the standard library at module scope, so tool
scripts can use it from any environment.
"""
import argparse
import json
import os
import subprocess
import sys
from typing import Any, Callable, Dict, Optional


class ToolError(Exception):
    """Raised by a tool's handler for an expected failure; its message is shown to the model and the user."""


class ToolAnswer:
    """
    What a handler returns when it also wants to write its own one-line summary for the chat history (see
    registry.Summarized). A handler that returns anything else gets the generic summary.

    Attributes:
        result: The full result, sent to the model this turn.
        summary (str): One line kept for later turns. For a tool with untrusted output, describe the result - a URL, a
            count - and never quote text from it.
    """

    def __init__(self, result: Any, summary: str):
        self.result = result
        self.summary = summary


# ------------------------------------------------------------------------------------------ script side

def tool_script_main(definition: Dict[str, Any], handler: Callable[[Dict[str, Any], Dict[str, Any]], Any]):
    """
    The whole command-line surface of a tool script. Call it from the script's __main__.

    Args:
        definition: {'name', 'description', 'parameters', 'flags': {...}, 'timeout_s'} - see the module docstring.
        handler: called as handler(arguments, config); returns anything JSON-serialisable, or raises ToolError.
    """
    parser = argparse.ArgumentParser(description=definition.get("description", ""))
    parser.add_argument("--describe", action="store_true", help="print this tool's definition as JSON and exit")
    parser.add_argument("--config", default="", help="this tool's JSON config file")
    args = parser.parse_args()

    if args.describe:
        print(json.dumps(definition))
        return

    try:
        config = {}
        if args.config:
            with open(args.config, encoding="utf-8") as fh:
                config = json.load(fh)
        raw = sys.stdin.read()
        arguments = json.loads(raw) if raw.strip() else {}
        if not isinstance(arguments, dict):
            raise ToolError("arguments must be a JSON object")
        produced = handler(arguments, config)
        if isinstance(produced, ToolAnswer):
            answer = {"ok": True, "result": produced.result, "summary": str(produced.summary)}
        else:
            answer = {"ok": True, "result": produced}
    except ToolError as e:
        answer = {"ok": False, "error": str(e)}
    except (OSError, json.JSONDecodeError) as e:
        answer = {"ok": False, "error": f"{definition.get('name', 'tool')} could not start: {type(e).__name__}: {e}"}
    print(json.dumps(answer, ensure_ascii=False, default=str))


# ------------------------------------------------------------------------------------------ server side

def _child_environment(extra_pythonpath: str) -> Dict[str, str]:
    """
    The whole environment a tool script gets: enough to run Python and find its cache, nothing else. The server's
    own environment can hold tokens and keys; a tool that fetches arbitrary URLs has no business seeing them.
    """
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LC_ALL", "TZ") if key in os.environ}
    env["PYTHONPATH"] = extra_pythonpath
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _src_root() -> str:
    """The directory amadeo_utils is imported from, so scripts can import this module in their own env."""
    here = os.path.dirname(os.path.abspath(__file__))              # .../src/amadeo_utils/ai/llm/tools
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def load_script_tool(python: str, script: str, config: Optional[str] = None):
    """
    Builds a registry.Tool for a tool script, by asking the script to describe itself.

    Args:
        python: The interpreter to run it with, e.g. '~/miniforge3/envs/agent-tools/bin/python'.
        script: The tool script's path.
        config: The tool's own config file, or None.

    Returns:
        registry.Tool

    Raises:
        ValueError: if the script cannot describe itself, or describes itself inconsistently.
    """
    from amadeo_utils.ai.llm.tools.registry import Tool, Summarized   # deferred: scripts import this module without the registry

    python, script = os.path.expanduser(python), os.path.expanduser(script)
    config = os.path.expanduser(config) if config else None
    env = _child_environment(_src_root())
    try:
        described = subprocess.run([python, script, "--describe"], capture_output=True, text=True, timeout=30,
                                   env=env, stdin=subprocess.DEVNULL)
        definition = json.loads(described.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        raise ValueError(f"tool script [{script}] could not describe itself: {type(e).__name__}: {e}")
    for key in ("name", "description", "parameters"):
        if key not in definition:
            raise ValueError(f"tool script [{script}] describes itself without '{key}'")

    timeout = float(definition.get("timeout_s", 20))
    command = [python, script] + (["--config", config] if config else [])

    def run(**arguments):
        """One call: arguments on stdin, one JSON answer on stdout. The process is killed if it overruns."""
        try:
            finished = subprocess.run(command, input=json.dumps(arguments), capture_output=True, text=True,
                                      timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"did not finish within {timeout:g} seconds and was stopped")
        try:
            answer = json.loads(finished.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            # stderr may hold a traceback with argument values in it: report that it failed, not what it said.
            raise RuntimeError(f"exited with code {finished.returncode} without a usable answer")
        if not answer.get("ok"):
            raise RuntimeError(answer.get("error") or "failed without saying why")
        if answer.get("summary"):
            return Summarized(answer.get("result"), answer["summary"])
        return answer.get("result")

    flags = dict(definition.get("flags") or {})
    # A typo in a flag name must stop the server, not silently leave the tool unflagged.
    unknown = set(flags) - {"outbound", "private", "untrusted_output", "internal_commands", "needs_approval"}
    if unknown:
        raise ValueError(f"tool script [{script}] declares unknown flag(s) {sorted(unknown)}")
    # The registry's own timeout is the backstop; the subprocess timeout above is the one that fires and kills.
    return Tool(name=definition["name"], description=definition["description"], parameters=definition["parameters"],
                function=run, timeout_s=timeout + 5, **flags)

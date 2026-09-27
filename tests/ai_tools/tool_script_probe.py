#!/usr/bin/env python3
"""CS-21 test helper: a tool script that reports on its own surroundings, for check_ai_tools.py.

Depending on its arguments it returns its environment's variable NAMES and its argv, sleeps
(to be killed by the timeout), fails cleanly, or crashes. Never used by the tool server.
"""
import os
import sys
import time

from amadeo_utils.ai.llm.tools.script_tools import ToolError, tool_script_main

DEFINITION = {"name": "probe", "description": "test probe",
              "parameters": {"type": "object", "properties": {"mode": {"type": "string"}, "secret_arg": {"type": "string"}}},
              "flags": {}, "timeout_s": 1}


def handle(arguments, config):
    mode = arguments.get("mode")
    if mode == "sleep":
        time.sleep(30)
    if mode == "fail":
        raise ToolError("clean failure message")
    if mode == "crash":
        raise RuntimeError(f"traceback containing {arguments.get('secret_arg')}")
    return {"env": sorted(os.environ), "argv": sys.argv, "pid": os.getpid()}


if __name__ == "__main__":
    tool_script_main(DEFINITION, handle)

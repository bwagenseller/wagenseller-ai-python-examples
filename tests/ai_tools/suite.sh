#!/usr/bin/env bash
#
# ai_tools: the agent's tool scripts (scripts/ai/ai-tools/) - their contract, guards and live behaviour.
#
# The tools run in their own conda env (agent_tools_python), as the agent server runs them. check_ai_tools.py also
# calls the real services (NWS, SearXNG, a few web pages) unless OFFLINE=1; those checks need ai_tools_config_dir.
# check_get_grades.py uses a stand-in login: no browser, no credentials.

source "$(dirname "$0")/../testlib/suite_lib.sh"
cd "$(dirname "$0")" || exit 1

TOOLS_PY="$(need_python agent_tools_python)"
if [ -z "$TOOLS_PY" ]; then
    skip "every tool-script check" "agent_tools_python is not set or not executable"
    finish
fi

run "get_grades (stand-in login)" env PYTHONPATH="$SRC" "$TOOLS_PY" check_get_grades.py
if [ "$OFFLINE" -eq 1 ]; then
    run "tool scripts (offline)" env PYTHONPATH="$SRC" "$TOOLS_PY" check_ai_tools.py --offline
else
    run "tool scripts (incl. live NWS, SearXNG, web)" env PYTHONPATH="$SRC" "$TOOLS_PY" check_ai_tools.py
fi

finish

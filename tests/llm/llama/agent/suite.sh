#!/usr/bin/env bash
#
# llm/llama/agent: the tool-calling agent family (ToolStream) - its tool loop, parsers, filters and safeguards.
#
# Everything here runs with the model call stubbed or no model at all, except that golden_tool_loop.py loads the
# three tool-capable models (CPU) for their real chat templates. The live_* and probe_* scripts are NOT run here:
# their output is sampled or measured, so a person reads it (see tests/README.md).
# Needs: llama_python; model_dir and embedding_model for golden_tool_loop.py.

source "$(dirname "$0")/../../../testlib/suite_lib.sh"
cd "$(dirname "$0")" || exit 1

PY="$(need_python llama_python)"
if [ -z "$PY" ]; then
    skip "every agent check" "llama_python is not set or not executable"
    finish
fi

run "tool-call parsers, every architecture"               "$PY" check_tool_parsers.py
run "credential filter"                                    "$PY" check_secret_filter.py
run "role-play / knowledge base cannot reach tools"        "$PY" check_tools_isolation.py
run "approval protocol over TCP"                           "$PY" check_approval_protocol.py
run "result cap shares the room left"                      "$PY" check_result_cap.py
run "every tool config key reaches the server"             "$PY" check_tool_config_loader.py
run "pointer-only replies after a delegate are dropped"    "$PY" check_pointer_filter.py
run "get_datetime: zone names, local default, output"      "$PY" check_datetime_tool.py
run "spoken replies: markdown made speakable"              "$PY" check_speakable.py
run "tool-loop scenarios x 3 architectures"                "$PY" golden_tool_loop.py --out "$LOG_DIR/tool_loop.json"

finish

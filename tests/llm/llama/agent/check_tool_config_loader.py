#!/usr/bin/env python3
"""CS-21: every tool-server config key survives ToolStream.get_args_dict.

Why
---
2026-09-25: tool_result_share and the three worker_notes* keys were listed in
TOOL_CONFIG_FIELDS (so their TYPES were checked) but never copied into the settings dict, so a
config's value was silently replaced by the default. The scenario driver sets those values on
the stream directly and could not see it. This loads a real config file through the real loader
with a NON-default value for every key in TOOL_CONFIG_FIELDS and checks each one arrives - so a
key added later without its copy line fails here.

Runs in the llama env (the loader's module imports llama_cpp):  python check_tool_config_loader.py
"""
import json
import os
import sys
import tempfile

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()
SRC = settings.SRC
# The example config always comes from this repository, whichever library copy is under test (AMADEO_TEST_SRC).
EXAMPLE = os.path.join(settings.REPO_ROOT, "scripts", "ai", "llm", "llama", "llama_stream",
                       "example_amadeo_agent_server_config.json")

# A value for each tool key that differs from its default, with the value the loader should hand on.
NON_DEFAULT = {
    "knowledge_base_file": "/tmp/cs21-kb.jsonl",
    "tools_allowed": ["calculator"],
    "script_tools": [{"script": "/x.py", "python": "/py", "config": "/c.json"}],
    "tool_mode": "direct",
    "max_tool_rounds": 4,
    "max_turn_seconds": 42.5,
    "max_tool_result_tokens": 1234,
    "tool_result_share": 0.3,
    "worker_notes": "always",
    "worker_notes_trigger": 0.6,
    "worker_notes_tokens": 250,
    "system_prompt_dir": "/tmp/cs21-prompts",
    "default_timezone": "Europe/London",
    "client_idle_timeout_seconds": 0.0,
    "auto_approve": True,
    "approval_timeout_seconds": 33,
    "tool_audit_log": "/tmp/cs21-audit.jsonl",
    "encrypted": True,
}


def main():
    from amadeo_utils.ai.llm.llama.ToolStream import ToolStream
    failures = []
    missing_values = sorted(set(ToolStream.TOOL_CONFIG_FIELDS) - set(NON_DEFAULT))
    if missing_values:
        failures.append(f"no test value for new key(s) {missing_values} - add them to NON_DEFAULT")
    with tempfile.TemporaryDirectory() as work:
        prompt = os.path.join(work, "prompt.txt")
        with open(prompt, "w") as fh:
            fh.write("test prompt\n")
        config = json.load(open(EXAMPLE))
        config.update(NON_DEFAULT, default_system_prompt_file=prompt)
        path = os.path.join(work, "config.json")
        with open(path, "w") as fh:
            json.dump(config, fh)
        sys.argv = ["amadeo_agent_server.py", "--json", path]
        settings = ToolStream.get_args_dict()
        prompt_path_seen = [prompt]
        # The same config, but with the pre-2026-09-26 key name: must be refused, with a message.
        old = dict(config)
        old["system_prompt_file"] = old.pop("default_system_prompt_file")
        old_path = os.path.join(work, "old.json")
        with open(old_path, "w") as fh:
            json.dump(old, fh)
        import logging
        captured = []
        handler = logging.Handler()
        handler.emit = lambda record: captured.append(record.getMessage())
        logging.getLogger("amadeo_utils.ai.llm.llama.ToolStream").addHandler(handler)
        sys.argv = ["amadeo_agent_server.py", "--json", old_path]
        old_key_result = [ToolStream.get_args_dict()]
        old_key_log = [" ".join(captured)]
    if not settings:
        failures.append("the loader rejected the config")
    for key in sorted(ToolStream.TOOL_CONFIG_FIELDS):
        want, got = NON_DEFAULT.get(key), settings.get(key)
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {key}: {got!r}" + ("" if ok else f" (config said {want!r})"))
        if not ok:
            failures.append(key)
    # A shared server setting (LlamaUtils.map_server_system_config), not a tool one - but the tool server needs it too.
    ok = settings.get("client_idle_timeout_seconds") == NON_DEFAULT["client_idle_timeout_seconds"]
    print(f"  {'PASS' if ok else 'FAIL'}  client_idle_timeout_seconds (shared): {settings.get('client_idle_timeout_seconds')!r}")
    if not ok:
        failures.append("client_idle_timeout_seconds")
    # The renamed prompt key: 'default_system_prompt_file' is loaded into the internal 'system_prompt_file', and a
    # config still using the old name is refused with a message naming the new one.
    ok = settings.get("system_prompt_file") == prompt_path_seen[0]
    print(f"  {'PASS' if ok else 'FAIL'}  default_system_prompt_file loaded as the server's prompt")
    if not ok:
        failures.append("default_system_prompt_file")
    old_style = old_key_result[0]
    ok = old_style == {} and "is now called 'default_system_prompt_file'" in old_key_log[0]
    print(f"  {'PASS' if ok else 'FAIL'}  a config with the old 'system_prompt_file' key is refused, naming the new key")
    if not ok:
        failures.append("old system_prompt_file refused")
    print("\nALL CHECKS PASS" if not failures else f"\nFAILED: {failures}")
    sys.exit(len(failures))


if __name__ == "__main__":
    main()

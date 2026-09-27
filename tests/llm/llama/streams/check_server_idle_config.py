#!/usr/bin/env python3
"""CS-21: client_idle_timeout_seconds reaches all three llama stream servers; the client refuses unknown modes.

Why
---
2026-09-25: the idle timeout (AmadeoServer, 300 s) became configurable, first for the tool server
only; the owner asked for it on the role-play and knowledge-base servers too. It is now a shared
server setting (LlamaUtils.map_server_system_config for the knowledge-base and tool servers, and
the role-play server's own loader), validated by LlamaUtils.resolve_idle_timeout. In the same
change LlamaStreamClient stopped sending an unrecognised 'mode' as a knowledge-base session.

What it proves
--------------
* For each server's REAL loader, from a real JSON file: key absent -> 300; 0 -> 0 (never);
  1234 -> 1234; a negative value or a string is refused - never silently loaded.
* Each server script passes the setting to AmadeoServer.
* The client: a JSON 'mode' of "tool" and a command-line '--mode roleplay' both stop it with exit
  code 2 and a message; a valid mode gets as far as trying to connect.

Runs in the llama env (the stream modules import llama_cpp):  python check_server_idle_config.py
"""
import json
import os
import socket
import subprocess
import sys
import tempfile

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()
SRC = settings.SRC
STREAM_DIR = os.path.join(settings.REPO_ROOT, "scripts", "ai", "llm", "llama", "llama_stream")
failures = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if not ok and detail else ""))
    if not ok:
        failures.append(label)


def load(loader, example, work, value, prompt_key):
    """Writes the example config with client_idle_timeout_seconds = value (or absent) and loads it via 'loader'."""
    config = json.load(open(os.path.join(STREAM_DIR, example)))
    config.pop("client_idle_timeout_seconds", None)
    if value != "absent":
        config["client_idle_timeout_seconds"] = value
    prompt = os.path.join(work, "prompt.txt")
    with open(prompt, "w") as fh:
        fh.write("test prompt\n")
    if prompt_key == "system_prompt_dir":
        config["system_prompt_dir"] = work
    else:
        config[prompt_key] = prompt
    path = os.path.join(work, "config.json")
    with open(path, "w") as fh:
        json.dump(config, fh)
    sys.argv = ["server.py", "--json", path]
    try:
        settings = loader()
    except (ValueError, TypeError) as e:
        return f"refused: {type(e).__name__}"
    if not settings:
        return "refused: empty settings"
    if settings.get("generating_model", "").endswith(config["model"]) is False:
        return "fell back to defaults"
    return settings.get("client_idle_timeout_seconds", "missing")


def loaders_check():
    from amadeo_utils.ai.llm.llama.KnowledgeBaseStream import KnowledgeBaseStream
    from amadeo_utils.ai.llm.llama.RolePlayStream import RolePlayStream
    from amadeo_utils.ai.llm.llama.ToolStream import ToolStream
    servers = [("knowledge base", KnowledgeBaseStream.get_args_dict, "example_knowledge_base_server_config.json", "system_prompt_file"),
               ("role play", RolePlayStream.get_args_dict, "example_role_play_server_config.json", "system_prompt_dir"),
               ("agent", ToolStream.get_args_dict, "example_amadeo_agent_server_config.json", "default_system_prompt_file")]
    for name, loader, example, prompt_key in servers:
        print(f"== {name} server loader ==")
        with tempfile.TemporaryDirectory() as work:
            got = {v: load(loader, example, work, v, prompt_key) for v in ("absent", 0, 1234, -5, "never")}
        check(got["absent"] == 300, "absent -> 300 seconds (AmadeoServer's old default)", repr(got["absent"]))
        check(got[0] == 0, "0 -> 0 (never)", repr(got[0]))
        check(got[1234] == 1234, "1234 -> 1234", repr(got[1234]))
        check(str(got[-5]).startswith("refused"), "a negative value is refused, not loaded", repr(got[-5]))
        check(str(got["never"]).startswith("refused") or got["never"] == "fell back to defaults",
              "a string is not accepted as a timeout", repr(got["never"]))


def scripts_check():
    print("== the server scripts pass it to AmadeoServer ==")
    for script in ("role_play_server.py", "knowledge_base_server.py", "amadeo_agent_server.py"):
        text = open(os.path.join(STREAM_DIR, script)).read()
        check("client_timeout=argsDict.get('client_idle_timeout_seconds'" in text, f"{script} passes client_timeout")


def client_mode_check():
    print("== the client refuses an unknown mode ==")
    env = dict(os.environ, PYTHONPATH=SRC)
    client = os.path.join(STREAM_DIR, "LlamaStreamClient.py")
    with socket.socket() as s:                       # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]
    with tempfile.TemporaryDirectory() as work:
        for mode, expect_refused in (("tool", True), ("roleplay", True), ("tools", True), ("agent", False)):
            path = os.path.join(work, f"{mode}.json")
            with open(path, "w") as fh:
                json.dump({"host": "127.0.0.1", "port": dead_port, "mode": mode, "user_id": "t",
                           "spoken_response": False}, fh)
            done = subprocess.run([sys.executable, client, "--json", path], capture_output=True, text=True,
                                  env=env, timeout=30, stdin=subprocess.DEVNULL)
            out = done.stdout + done.stderr
            if expect_refused:
                check(done.returncode == 2 and "Unknown mode" in out and "Attempting to connect" not in out,
                      f"JSON mode {mode!r}: refused with exit 2 before connecting", out[-300:])
            else:
                check("Unknown mode" not in out and "Attempting to connect" in out,
                      f"JSON mode {mode!r}: accepted, and it tries to connect", out[-300:])
        done = subprocess.run([sys.executable, client, "--mode", "roleplay"], capture_output=True, text=True,
                              env=env, timeout=30, stdin=subprocess.DEVNULL)
        check(done.returncode == 2 and "invalid choice" in (done.stdout + done.stderr),
              "command-line --mode roleplay: refused with exit 2", (done.stdout + done.stderr)[-300:])


def main():
    loaders_check()
    scripts_check()
    client_mode_check()
    print("\nALL CHECKS PASS" if not failures else f"\n{len(failures)} check(s) FAILED")
    sys.exit(len(failures))


if __name__ == "__main__":
    main()

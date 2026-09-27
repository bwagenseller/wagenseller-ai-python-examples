#!/usr/bin/env python3
"""CS-21: client-chosen names that become paths - system_prompt_id and user_id - are refused when unsafe.

Why
---
2026-09-25: the agent server gained role-play's client-chosen prompts (system_prompt_id ->
'<system_prompt_dir>/<id>.txt'). Reading role-play's code for it showed two path-traversal gaps,
both closed here with the owner's approval: the prompt id was joined straight into a file path
('../../elsewhere/file' loaded any readable .txt as the system prompt, which the model could then
repeat), and the user id straight into the conversation folder ('../../x' saved conversations
outside it). Both servers now use LlamaUtils.is_safe_name / safe_prompt_path; an unsafe name fails
the session (fatal_errors) and never becomes a path component.

What it proves
--------------
* The helpers: plain names pass; paths, '..', absolute names, empty and non-strings are refused.
* A REAL RolePlayStream (CPU; the same setup as golden_save_load.py): a valid id loads its prompt
  exactly as before; a safe id with no file still falls back to the base prompt with no error (the
  old behaviour, unchanged); '../...' prompt ids and user ids fail the session, and the conversation
  folder never contains the unsafe name.
The agent server's side is covered by golden_tool_loop scenarios prompt_* / unsafe_user_id_refused.

Usage (llama env; CPU):  CUDA_VISIBLE_DEVICES= python check_prompt_ids.py --model /path/to.gguf --model-type llama-3
"""
import argparse
import os
import shutil
import sys
import tempfile

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()
failures = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if not ok and detail else ""))
    if not ok:
        failures.append(label)


def helper_checks():
    from amadeo_utils.ai.llm.llama.llama_utils import LlamaUtils as LU
    print("== LlamaUtils.is_safe_name / safe_prompt_path ==")
    for name in ("default", "voice", "santa-lite", "Bob", "John Smith", "v1.2_final"):
        check(LU.is_safe_name(name), f"safe: {name!r}")
    for name in ("", "../x", "../../etc/passwd", "/etc/passwd", "a/b", "..", "x..y", ".hidden", "-dash", None, 7, "x" * 65):
        check(not LU.is_safe_name(name), f"refused: {name!r}")
    check(LU.safe_prompt_path("/p", "voice") == "/p/voice.txt", "safe_prompt_path joins a safe id")
    check(LU.safe_prompt_path("/p", "../voice") is None and LU.safe_prompt_path("", "voice") is None,
          "safe_prompt_path refuses an unsafe id, and a missing folder")


def roleplay_checks(model, model_type):
    import golden_response as GR
    from amadeo_utils.ai.llm.llama.RolePlayStream import RolePlayStream
    print("== RolePlayStream (real, CPU) ==")
    work = tempfile.mkdtemp(prefix="cs21-promptids-")
    prompt_dir, convo_dir = os.path.join(work, "prompts"), os.path.join(work, "convo")
    os.makedirs(prompt_dir)
    os.makedirs(convo_dir)
    with open(os.path.join(prompt_dir, "default.txt"), "w") as fh:
        fh.write("DEFAULTPROMPTSENTINEL " + GR.SYSTEM_PROMPT_TEXT)
    secret = os.path.join(work, "outside.txt")                  # a .txt a traversal could have reached
    with open(secret, "w") as fh:
        fh.write("OUTSIDESECRET")
    try:
        rp = RolePlayStream(GR.build_args(model, model_type, 2048, prompt_dir, convo_dir))
        cases = [("ok", "tester", "default"), ("missing", "tester", "nope"), ("traverse", "tester", "../outside"),
                 ("user", "../../elsewhere", "default")]
        for sid, user, prompt in cases:
            rp.create_session(sid, user, "", prompt, False, False, False)
        s = {sid: rp.get_session(sid) for sid, _, _ in cases}

        check(not s["ok"]["fatal_errors"] and "DEFAULTPROMPTSENTINEL" in s["ok"]["system_message"]
              and s["ok"]["convo_dir"] == os.path.join(convo_dir, "tester", "default"),
              "a valid id loads its prompt, into <base>/<user>/<id> - as before", str(s["ok"]["fatal_errors"]))
        check(not s["missing"]["fatal_errors"] and "DEFAULTPROMPTSENTINEL" not in s["missing"]["system_message"],
              "a safe id with no file: base prompt, no error - the old behaviour, unchanged", s["missing"]["fatal_errors"])
        check("system_prompt_id is invalid" in s["traverse"]["fatal_errors"]
              and "OUTSIDESECRET" not in s["traverse"]["system_message"] and ".." not in s["traverse"]["convo_dir"],
              "a '../' prompt id fails the session and reads nothing outside the folder",
              f"{s['traverse']['fatal_errors']!r} {s['traverse']['convo_dir']}")
        check("user_id is invalid" in s["user"]["fatal_errors"] and ".." not in s["user"]["convo_dir"]
              and s["user"]["convo_dir"].startswith(convo_dir),
              "a '../' user id fails the session; its folder stays inside the conversation root",
              f"{s['user']['fatal_errors']!r} {s['user']['convo_dir']}")
        rp.cleanup()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-type", default="llama-3")
    args = parser.parse_args()
    args.model = settings.resolve_model(args.model)   # a name inside model_dir, or a path
    helper_checks()
    roleplay_checks(args.model, args.model_type)
    print("\nALL CHECKS PASS" if not failures else f"\n{len(failures)} check(s) FAILED")
    sys.exit(len(failures))


if __name__ == "__main__":
    main()

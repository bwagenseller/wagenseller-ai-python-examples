#!/usr/bin/env python3
"""CS-21: capture role-play's save -> load round trip, to prove moving it into StreamBase changed nothing.

Why this exists
---------------
CS-20's golden_response.py sends '!save' and '!load', but only on fresh sessions: '!save' writes
files nothing reads back, and '!load' finds nothing to load. Moving save/load into StreamBase
(CS-21) needs the real round trip pinned, including role-play's own step - re-seeding its
INITIAL_KNOWLEDGE_BASE_DOCUMENTS - which stays in RolePlayStream.

What it records, for two cases (a saved conversation reloaded with load_previous=True, and the
same directory opened with load_previous=False):

* the chat history that came back, exactly;
* the vector database's size and its sorted user_text column (so re-seeded documents show up);
* which files the save wrote.

Run it against the pre-change sources (``AMADEO_TEST_SRC=<copy>``) and the current ones, and diff.
Honours AMADEO_TEST_SRC like the CS-20 harnesses. llama env; CPU only.

Usage
-----
    python golden_save_load.py --model /path/to.gguf --model-type llama-3 --out result.json
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()

# Reuse golden_response.py's configuration builder and history seeding (same folder), so this runs the same setup.
import golden_response as GR                                     # noqa: E402
from amadeo_utils.ai.llm.llama.RolePlayStream import RolePlayStream  # noqa: E402


def snapshot(session):
    """What a reload restored: history, and the vector database's contents."""
    df = session["db"].df
    return {
        "chat_history": session["chat_history"],
        "db_rows": len(df),
        "db_user_text": sorted(t.strip() for t in df["user_text"]),
    }


def main():
    parser = argparse.ArgumentParser(description="CS-21 save/load round-trip capture")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-type", default="llama-3")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.model = settings.resolve_model(args.model)   # a name inside model_dir, or a path

    workdir = tempfile.mkdtemp(prefix="cs21-saveload-")
    prompt_dir, convo_dir = os.path.join(workdir, "prompts"), os.path.join(workdir, "convo")
    os.makedirs(prompt_dir)
    os.makedirs(convo_dir)
    with open(os.path.join(prompt_dir, "default.txt"), "w", encoding="utf-8") as fh:
        fh.write(GR.SYSTEM_PROMPT_TEXT)

    result = {"model": os.path.basename(args.model)}
    try:
        rp = RolePlayStream(GR.build_args(args.model, args.model_type, 2048, prompt_dir, convo_dir))

        # A first session: seed it, save it.
        rp.create_session("s1", "saver", "", "default", False, False, False)
        s1 = rp.get_session("s1")
        rp.load_chat_history(s1, False)              # what get_response does on a first turn
        GR.seed_long_history(rp, "s1", 3)
        rp.save(s1)
        saved_dir = s1["convo_dir"]
        result["files_written"] = sorted(os.listdir(saved_dir))
        rp.remove_session("s1")

        # Reload it into a fresh session, as '!load' / load_previous=True does.
        rp.create_session("s2", "saver", "", "default", False, False, True)
        s2 = rp.get_session("s2")
        rp.load_chat_history(s2, True)
        result["reloaded"] = snapshot(s2)
        rp.remove_session("s2")

        # The same directory, opened WITHOUT restoring.
        rp.create_session("s3", "saver", "", "default", False, False, False)
        s3 = rp.get_session("s3")
        rp.load_chat_history(s3, False)
        result["not_reloaded"] = snapshot(s3)
        rp.remove_session("s3")

        rp.cleanup()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    settings.write_capture(args.out, result)
    print(f"reloaded {len(result['reloaded']['chat_history'])} history entries, "
          f"{result['reloaded']['db_rows']} db rows; wrote {args.out}")


if __name__ == "__main__":
    main()

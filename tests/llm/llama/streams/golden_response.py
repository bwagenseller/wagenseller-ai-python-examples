#!/usr/bin/env python3
"""CS-20: capture get_response's behaviour deterministically.

The gap this fills
------------------
``golden_dispatch.py`` stubs ``get_response`` out entirely and ``golden_construct.py``
never calls it, so the largest and most delicate method in either family is the one thing
the suite does not cover. Extracting helpers out of it without this would be unverifiable.

How it is made deterministic
----------------------------
Generation is sampled, so its text cannot be diffed. Instead the model call itself is
stubbed: ``llm_generator.create_chat_completion`` is replaced with a recorder that
captures its arguments and returns a canned completion. That makes every run identical
AND captures the single most valuable artifact - **the exact ``messages_for_llm`` list the
family assembled**, plus the stop strings and token budget it chose. Those are what the
helpers actually touch.

The canned completion deliberately contains a ``<think>`` block so that the
``strip_reasoning`` path is exercised rather than skipped.

Everything time-varying is normalised out: ``elapsed_time`` is zeroed and any embedded
'!date' timestamp is replaced with a placeholder.

Each scenario uses its own ``user_id``, so each gets its own vector-database directory and
scenarios cannot contaminate one another through persisted documents.

Usage
-----
    python golden_response.py --model /path/to.gguf --out before.json

Honours AMADEO_TEST_SRC, like golden_construct.py, so a baseline can be taken from the pristine
pre-refactor sources. Run in the ``llama`` conda environment.
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()

from amadeo_utils.ai.llm.llama.RolePlayStream import RolePlayStream
from amadeo_utils.ai.llm.llama.KnowledgeBaseStream import KnowledgeBaseStream


# Canned model output. The <think> block makes strip_reasoning do real work.
CANNED_CONTENT = "<think>weighing it up</think>Water boils at 100 degrees Celsius."

SYSTEM_PROMPT_TEXT = "You are Ada, a guide. Be concise.\n"

# Scenarios are (label, user_request). Role-play-only commands are marked.
SHARED_SCENARIOS = [
    ("plain", "What boils water?"),
    ("help", "!help"),
    ("history_review", "!history What boils water?"),
    ("vector_test", "!vectortest What boils water?"),
    ("reason", "!reason What boils water?"),
    ("think", "!remember What boils water?"),
]

ROLE_PLAY_ONLY_SCENARIOS = [
    ("save", "!save"),
    ("load", "!load"),
    ("strike", "!strike"),
    ("crystal", "!crystal What boils water?"),
    ("length_preset_short", "!short What boils water?"),
    ("ignore_me", "!ignoreme What boils water?"),
]

# CS-21: the same requests against a session whose history NO LONGER FITS the context window.
# Every scenario above starts with an empty history, so role-play always took its "full history
# fits" branch and its vector-database branch was never captured - the very code CS-21 moves
# into a shared helper. Seeding a long history forces that branch in both families.
OVERFLOW_SCENARIOS = [
    ("overflow_plain", "What boils water?"),
    ("overflow_history_review", "!history What boils water?"),
    ("overflow_think", "!remember What boils water?"),
]

# 24 pairs at a nominal 120 tokens each is ~2900 tokens: well past a 2048-token window.
SEED_PAIRS = 24

# CS-21: a history that is NOT empty but still fits (8 pairs, ~960 tokens), so role-play's
# whole-history branch is captured with real content in it. Found missing by a mutation test:
# moving the "does it fit" threshold by 1000 tokens changed no capture at all, because every
# other scenario had either no history or far too much.
MIDSIZE_SCENARIOS = [
    ("midsize_plain", "What boils water?"),
    ("midsize_history_review", "!history What boils water?"),
]
MIDSIZE_PAIRS = 8


def seed_midsize_history(stream, session_id):
    """Like seed_long_history, but small enough that the whole history still fits the window."""
    seed_long_history(stream, session_id, MIDSIZE_PAIRS)


def seed_long_history(stream, session_id, pairs=SEED_PAIRS):
    """Fill a session's chat history and vector database with more than its window can hold.

    The token counts are fixed nominal values rather than measured: the assembly code only sums
    them, so fixed values keep the capture independent of the tokenizer while still forcing the
    overflow. The vector database computes its own counts and embeddings, as in real use.
    """
    session = stream.get_session(session_id)
    for i in range(pairs):
        user = f"Tell me fact number {i} about water and its boiling point at altitude."
        assistant = (f"Fact {i}: water boils at a lower temperature the higher you go, "
                     f"because the air pressure drops. ") * 3
        session["db"].add_document(user, assistant)
        session["chat_history"].append({"role": "user", "content": user, "token_count": 40})
        session["chat_history"].append({"role": "assistant", "content": assistant, "token_count": 80})


ISO_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def normalise(obj):
    """Strip everything that legitimately varies between runs."""
    if isinstance(obj, dict):
        return {k: (0.0 if k == "elapsed_time" else normalise(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalise(v) for v in obj]
    if isinstance(obj, str):
        return ISO_TIMESTAMP.sub("<TIMESTAMP>", obj)
    return obj


def build_args(model, model_type, ctx, prompt_dir, convo_dir, knowledge_base_file=None):
    """Configuration for either family. Mirrors a real server config's keys."""
    args = {
        "model_type": model_type,
        "generating_model": model,
        "embedding_model": settings.get("embedding_model"),
        "gpu": 0,
        "split_gpus": False,
        "flash_attn": False,
        "kv_cache_type": "f16",
        "embedding_gpu_layers": 0,
        "embedding_max_context_tokens": 512,
        "generating_gpu_layers": 0,
        "generating_max_context_tokens": ctx,
        "chat_format": None,
        "debug": False,
        "encrypted": False,
        "buffer_context_pcnt": 0.05,
        "system_message": SYSTEM_PROMPT_TEXT.strip(),
        "base_convo_dir": convo_dir,
        "system_prompt_dir": prompt_dir,
        "max_response_tokens": 160,
        "repeat_penalty": 1.1,
        "max_vector_database_pcnt": 0.2,
        "top_k": 4,
        "min_vector_db_score": 0.46,
        "response_token_presets": {"veryshort": 32, "short": 64, "medium": 128,
                                   "normal": 256, "long": 512, "verylong": 1024},
        "reasoning_budget_tokens": 2048,
        "suppressed_reasoning_tokens": 0,
    }
    if knowledge_base_file:
        args["knowledge_base_file"] = knowledge_base_file
    return args


def install_recorder(stream):
    """Replace the model call with a recorder, so runs are deterministic.

    Returns the list that each call's arguments are appended to. Capturing
    'messages' here is the point of the whole harness: it is the assembled prompt,
    which is exactly what the helpers being extracted are responsible for building.
    """
    calls = []

    def fake_create_chat_completion(**kwargs):
        calls.append({
            "messages": kwargs.get("messages"),
            "max_tokens": kwargs.get("max_tokens"),
            "stop": kwargs.get("stop"),
            "repeat_penalty": kwargs.get("repeat_penalty"),
            "stream": kwargs.get("stream"),
        })
        return {"choices": [{"message": {"content": CANNED_CONTENT}}]}

    stream.llm_generator.create_chat_completion = fake_create_chat_completion
    return calls


def run_scenarios(stream, family, scenarios, make_session, seed=None, probe_missing_session=True):
    """Drive one family through every scenario, each on its own isolated session.

    'seed', if given, is called with (stream, session_id) after the session is created and
    before the request - used to pre-load a long history (CS-21).
    """
    out = {}
    for label, user_request in scenarios:
        session_id = f"sess_{label}"
        user_id = f"user_{label}"          # own vector-db directory, so scenarios stay isolated
        make_session(stream, session_id, user_id)
        if seed is not None:
            seed(stream, session_id)

        calls = install_recorder(stream)
        try:
            response = stream.get_response({
                "sessionID": session_id,
                "command": "request",
                "user_request": user_request,
            })
            record = {"response": normalise(response)}
        except Exception as e:
            record = {"raised": f"{type(e).__name__}: {e}"}

        record["generator_calls"] = normalise(calls)
        session = stream.get_session(session_id)
        if session is not None:
            record["session_after"] = {
                "used_tokens": session.get("used_tokens"),
                "full_history_fits": session.get("full_history_fits"),
                "chat_history_len": len(session.get("chat_history", [])),
            }
        stream.remove_session(session_id)
        out[label] = record

    if not probe_missing_session:
        return out

    # A request against a session that does not exist, which must not reach the model.
    calls = install_recorder(stream)
    out["no_such_session"] = {
        "response": normalise(stream.get_response({
            "sessionID": "does_not_exist", "command": "request", "user_request": "hello"})),
        "generator_calls": normalise(calls),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="CS-20 get_response capture")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-type", default="llama-3")
    parser.add_argument("--ctx", type=int, default=2048)
    args = parser.parse_args()
    args.model = settings.resolve_model(args.model)   # a name inside model_dir, or a path

    workdir = tempfile.mkdtemp(prefix="cs20-response-")
    prompt_dir = os.path.join(workdir, "prompts")
    convo_dir = os.path.join(workdir, "convo")
    os.makedirs(prompt_dir)
    os.makedirs(convo_dir)
    with open(os.path.join(prompt_dir, "default.txt"), "w", encoding="utf-8") as fh:
        fh.write(SYSTEM_PROMPT_TEXT)

    kb_file = os.path.join(workdir, "kb.jsonl")
    with open(kb_file, "w", encoding="utf-8") as fh:
        for i, (q, a) in enumerate([("What boils at 100C?", "Water, at sea level."),
                                    ("Tallest mountain?", "Mount Everest.")], start=1):
            fh.write(json.dumps({"id": i, "question": q, "answer": a}) + "\n")

    result = {"model": os.path.basename(args.model), "model_path": settings.model_name(args.model),
              "model_type": args.model_type, "ctx": args.ctx}

    try:
        rp = RolePlayStream(build_args(args.model, args.model_type, args.ctx, prompt_dir, convo_dir))
        result["RolePlayStream"] = run_scenarios(
            rp, "RolePlayStream", SHARED_SCENARIOS + ROLE_PLAY_ONLY_SCENARIOS,
            lambda s, sid, uid: s.create_session(sid, uid, "", "default", False, False, False))
        result["RolePlayStream"].update(run_scenarios(
            rp, "RolePlayStream", OVERFLOW_SCENARIOS,
            lambda s, sid, uid: s.create_session(sid, uid, "", "default", False, False, False),
            seed=seed_long_history, probe_missing_session=False))
        result["RolePlayStream"].update(run_scenarios(
            rp, "RolePlayStream", MIDSIZE_SCENARIOS,
            lambda s, sid, uid: s.create_session(sid, uid, "", "default", False, False, False),
            seed=seed_midsize_history, probe_missing_session=False))
        rp.cleanup()

        kb = KnowledgeBaseStream(build_args(args.model, args.model_type, args.ctx,
                                            prompt_dir, convo_dir, kb_file))
        result["KnowledgeBaseStream"] = run_scenarios(
            kb, "KnowledgeBaseStream", SHARED_SCENARIOS,
            lambda s, sid, uid: s.create_session(sid, uid, False))
        result["KnowledgeBaseStream"].update(run_scenarios(
            kb, "KnowledgeBaseStream", OVERFLOW_SCENARIOS,
            lambda s, sid, uid: s.create_session(sid, uid, False),
            seed=seed_long_history, probe_missing_session=False))
        result["KnowledgeBaseStream"].update(run_scenarios(
            kb, "KnowledgeBaseStream", MIDSIZE_SCENARIOS,
            lambda s, sid, uid: s.create_session(sid, uid, False),
            seed=seed_midsize_history, probe_missing_session=False))
        kb.cleanup()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    settings.write_capture(args.out, result)

    for family in ("RolePlayStream", "KnowledgeBaseStream"):
        data = result.get(family, {})
        reached = sum(1 for v in data.values() if v.get("generator_calls"))
        raised = [k for k, v in data.items() if "raised" in v]
        print(f"{family}: {len(data)} scenarios, {reached} reached the model"
              + (f", raised in: {raised}" if raised else ""))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

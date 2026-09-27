#!/usr/bin/env python3
"""CS-21: capture what the knowledge-base server's config loader produces, to prove sharing it changed nothing.

The system half of LlamaUtils.get_args_dict_knowledge_base_server was lifted into
SERVER_SYSTEM_REQUIRED_FIELDS / SERVER_SYSTEM_OPTIONAL_FIELDS / map_server_system_config so the
tool server can share it. This loads a set of configs - complete, minimal (optional keys absent),
missing a required key, a field of the wrong type, and a bad token setting - and records the
resulting settings dictionary (or the exception) for each. Run it against the pre-change and
current sources (AMADEO_TEST_SRC) and diff: identical output means identical behaviour, including the
fall-back-to-defaults paths.

Usage (llama env):  python golden_kb_server_config.py --out result.json
"""
import argparse
import copy
import json
import os
import sys
import tempfile

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()
from amadeo_utils.ai.llm.llama.llama_utils import LlamaUtils   # noqa: E402

# The example config always comes from the real tree, whichever library copy is under test.
EXAMPLE = os.path.join(settings.REPO_ROOT, "scripts", "ai", "llm", "llama", "llama_stream",
                       "example_knowledge_base_server_config.json")


def local_placeholders(text: str) -> str:
    """
    Replaces this machine's default paths with named placeholders ("<BASE_MODEL_DIR>/...").

    A config missing a required key makes the loader fall back to SubjectiveConstants' defaults, which
    subjective_constants_local.py (gitignored) overrides per machine. Recording them as placeholders keeps the
    baseline valid on every machine and free of any machine's directory layout. Longest values first, so a
    directory is not replaced inside a longer path that has its own constant.
    """
    from amadeo_utils.ai.llm.llama.subjective_constants import SubjectiveConstants as SC
    paths = {}
    for name in dir(SC):
        value = getattr(SC, name)
        if not name.startswith("_") and isinstance(value, str) and value.startswith("/"):
            paths[name] = value
    for name, value in sorted(paths.items(), key=lambda item: -len(item[1])):
        text = text.replace(value, f"<{name}>")
    return text


def variants(base):
    """(label, config) pairs covering every branch of the loader."""
    minimal = {k: v for k, v in base.items() if k in LlamaUtils.load_knowledge_base_server_json_config.__doc__ and k in (
        "host", "port", "base_model_dir", "base_embedding_dir", "model", "embedding_model", "knowledge_base_file",
        "system_prompt_file", "gpu_layers", "embedding_gpu_layers", "max_context_tokens", "embedding_max_context_tokens",
        "max_response_tokens", "repeat_penalty", "max_vector_database_pcnt", "buffer_context_pcnt", "top_k",
        "min_vector_db_score")}
    missing = copy.deepcopy(base); del missing["top_k"]
    wrong_type = copy.deepcopy(base); wrong_type["port"] = "65450"
    bad_tokens = copy.deepcopy(base); bad_tokens["reasoning_budget_tokens"] = -5
    full = copy.deepcopy(base); full.update({"gpu": 1, "split_gpus": True, "flash_attn": True, "kv_cache_type": "q8_0",
                                            "debug": True, "model_type": "gemma4", "reasoning_budget_tokens": 1024})
    return [("example", base), ("minimal", minimal), ("full", full), ("missing_required", missing),
            ("wrong_type", wrong_type), ("bad_tokens", bad_tokens)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = json.load(open(EXAMPLE))
    workdir = tempfile.mkdtemp(prefix="cs21-kbcfg-")
    prompt = os.path.join(workdir, "prompt.txt")
    open(prompt, "w").write("You are a test prompt.\n")
    base["system_prompt_file"] = prompt
    out = {}
    for label, config in variants(base):
        path = os.path.join(workdir, f"{label}.json")
        json.dump(config, open(path, "w"))
        sys.argv = ["loader", "--json", path]
        try:
            out[label] = LlamaUtils.get_args_dict_knowledge_base_server("127.0.0.1", 65450, lambda *_: None)
        except Exception as e:
            out[label] = f"raised {type(e).__name__}: {e}"
    # The temporary directory's random name is the only thing that varies between runs.
    text = json.dumps(out, indent=2, sort_keys=True, default=str).replace(workdir, "<TMP>")
    text = local_placeholders(text)
    open(args.out, "w").write(text + "\n")
    print(f"{len(out)} variants; wrote {args.out}")


if __name__ == "__main__":
    main()

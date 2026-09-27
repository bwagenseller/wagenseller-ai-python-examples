#!/usr/bin/env bash
#
# Capture a render baseline (baselines/render_baseline_*.json) for EVERY model in model_dir.
#
# Deliberately not limited to the three models from CS-17. The library is used with
# Gemma-3, Gemma-4, three Qwen generations, Mistral-Nemo and Llama-70B derivatives, and
# those older families exercise cases the new ones do not - a Llama derivative reports no
# thinking support and an EMPTY stop-token list, for instance. A refactor verified only
# against the three newest models could break the rest silently.
#
# Runs CPU-only (--gpu-layers 0). Rendering a prompt and counting tokens needs the
# model's metadata and tokenizer, not its weights, and llama.cpp mmaps the file - so even
# a 40 GB model captures in about 13 seconds and nothing competes for the GPU.
#
# Usage:
#   tests/llm/llama/streams/capture_render_baselines.sh
#
# Add a model to the directory, re-run this, and suite.sh picks it up automatically - it reads the
# model's name back out of each baseline file. Only capture baselines from code you trust: a baseline
# is the definition of "correct" that every later run is held to.

set -u

cd "$(dirname "$0")" || exit 1
PY="${PY:-$(python3 ../../../testlib/settings.py --get llama_python)}"
MODEL_DIR="${MODEL_DIR:-$(python3 ../../../testlib/settings.py --get model_dir)}"
if [ -z "$PY" ] || [ -z "$MODEL_DIR" ]; then
    echo "llama_python and model_dir must be set in tests/local_settings.json"
    exit 1
fi

shopt -s nullglob
models=("$MODEL_DIR"/*.gguf)
if [ ${#models[@]} -eq 0 ]; then
    echo "no .gguf files found in $MODEL_DIR"
    exit 1
fi

echo "capturing render baselines for ${#models[@]} models (CPU-only)"
echo

failures=0
for model in "${models[@]}"; do
    base="$(basename "$model" .gguf)"
    # Slug for the filename: lowercase, non-alphanumerics collapsed to underscores.
    slug="$(echo "$base" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/_*$//')"
    out="baselines/render_baseline_${slug}.json"

    if "$PY" golden_render.py --model "$model" --gpu-layers 0 --out "$out" >/dev/null 2>&1; then
        arch=$("$PY" -c "import json;print(json.load(open('$out'))['thinking_off']['architecture'])" 2>/dev/null)
        think=$("$PY" -c "import json;print(json.load(open('$out'))['thinking_off']['supports_thinking'])" 2>/dev/null)
        printf "  OK    %-44s arch=%-14s thinking=%s\n" "$base" "$arch" "$think"
    else
        printf "  FAIL  %-44s (see: %s golden_render.py --model %s --gpu-layers 0 --out /tmp/x.json)\n" \
               "$base" "$PY" "$model"
        failures=$((failures + 1))
    fi
done

echo
echo "captured $(( ${#models[@]} - failures )) of ${#models[@]}; $failures failed"
exit "$failures"

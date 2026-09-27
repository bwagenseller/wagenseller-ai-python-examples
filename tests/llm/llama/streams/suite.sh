#!/usr/bin/env bash
#
# llm/llama/streams: the role-play and knowledge-base stream families and their shared StreamBase.
#
# Most checks here are golden: a harness captures what the code does (dispatch, rendered prompts, token counts,
# construction, the assembled get_response prompt, save/load, the config loaders) and the capture must match a
# baseline in baselines/ byte for byte. Baselines record models by their name inside model_dir, so they work on
# any machine that has the same model files; a check whose model is missing is SKIPPED, not failed.
#
# Slow part (~4 minutes, skipped with QUICK=1): the render / construct / get_response captures for every model.
# Needs: llama_python, model_dir, embedding_model (tests/local_settings.json).

source "$(dirname "$0")/../../../testlib/suite_lib.sh"
cd "$(dirname "$0")" || exit 1

PY="$(need_python llama_python)"
if [ -z "$PY" ]; then
    skip "every stream check" "llama_python is not set or not executable"
    finish
fi
MODEL_DIR="$(setting model_dir)"

fields() {
    # $1 = baseline file, the rest = keys (key=default); prints their values space-separated.
    "$PY" -c '
import json, sys
d = json.load(open(sys.argv[1]))
out = []
for spec in sys.argv[2:]:
    key, _, default = spec.partition("=")
    out.append(str(d.get(key, default)))
print(" ".join(out))' "$@" 2>/dev/null
}

model_here() {
    # True when the model named in a baseline exists in this machine's model_dir.
    [ -n "$MODEL_DIR" ] && [ -n "$1" ] && [ -e "$MODEL_DIR/$1" ]
}

run "idle-timeout setting reaches all three servers; client refuses bad modes" "$PY" check_server_idle_config.py

OUT="$LOG_DIR/kb_server_config.json"
compare "knowledge-base server config loader" baselines/kb_server_config_baseline.json \
    "$PY" golden_kb_server_config.py --out "$OUT"

OUT="$LOG_DIR/dispatch.json"
compare "dispatch + session lifecycle (no model loaded)" baselines/dispatch_baseline.json \
    "$PY" golden_dispatch.py --out "$OUT"

# The save/load round trip and the prompt-id checks use the model from the MN-Violet-Lotus construction baseline.
read -r SL_MODEL SL_TYPE < <(fields baselines/construct_baseline_mn_violet_lotus_12b.json model_path model_type)
if model_here "${SL_MODEL:-}"; then
    OUT="$LOG_DIR/save_load.json"
    compare "save/load round trip" baselines/save_load_baseline_mn_violet_lotus_12b.json \
        "$PY" golden_save_load.py --model "$SL_MODEL" --model-type "$SL_TYPE" --out "$OUT"
    run "unsafe prompt / user ids refused (helpers + real role-play stream)" \
        "$PY" check_prompt_ids.py --model "$SL_MODEL" --model-type "$SL_TYPE"
else
    skip "save/load round trip + prompt ids" "model ${SL_MODEL:-?} not in model_dir"
fi

if [ "$QUICK" -eq 1 ]; then
    skip "render / construct / get_response baselines" "QUICK"
    finish
fi

# Prompt rendering + token counts, CPU only, one baseline per model (capture_render_baselines.sh makes them).
for baseline in baselines/render_baseline_*.json; do
    label="${baseline#baselines/render_baseline_}"; label="${label%.json}"
    read -r model gpu ctx < <(fields "$baseline" model_path gpu_layers=0 ctx=2048)
    if ! model_here "${model:-}"; then skip "render/$label" "model ${model:-?} not in model_dir"; continue; fi
    OUT="$LOG_DIR/render_$label.json"
    compare "render/$label" "$baseline" "$PY" golden_render.py --model "$model" --gpu-layers "$gpu" --ctx "$ctx" --out "$OUT"
done

# Real construction, end to end. Midnight-Rose is included on purpose: it has no embedded chat template, so it
# exercises the fallback inside StreamBase.__init__.
for baseline in baselines/construct_baseline_*.json; do
    label="${baseline#baselines/construct_baseline_}"; label="${label%.json}"
    read -r model mtype gpu ctx < <(fields "$baseline" model_path model_type gpu_layers=0 ctx=2048)
    if ! model_here "${model:-}"; then skip "construct/$label" "model ${model:-?} not in model_dir"; continue; fi
    OUT="$LOG_DIR/construct_$label.json"
    compare "construct/$label" "$baseline" "$PY" golden_construct.py --model "$model" --model-type "$mtype" \
        --gpu-layers "$gpu" --ctx "$ctx" --out "$OUT"
done

# get_response with the model call stubbed: the assembled messages, stops and token budget.
for baseline in baselines/response_baseline_*.json; do
    label="${baseline#baselines/response_baseline_}"; label="${label%.json}"
    read -r model mtype ctx < <(fields "$baseline" model_path model_type ctx=2048)
    if ! model_here "${model:-}"; then skip "response/$label" "model ${model:-?} not in model_dir"; continue; fi
    OUT="$LOG_DIR/response_$label.json"
    compare "response/$label" "$baseline" "$PY" golden_response.py --model "$model" --model-type "$mtype" \
        --ctx "$ctx" --out "$OUT"
done

finish

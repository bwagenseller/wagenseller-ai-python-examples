#!/usr/bin/env bash
#
# asr: speech-to-text support in amadeo_utils/ai/asr/ - speaker identification (CS-23).
#
# The matching rules run in any Python 3 (no model). The model check runs the WhisperX ASR server's request handler
# with a real embedding model on the CPU, in the stt env, against Kokoro stock voices rendered once by
# make_synthetic_voices.py (the speaker_id_voices_dir setting) - never a real person's voice.

source "$(dirname "$0")/../testlib/suite_lib.sh"
cd "$(dirname "$0")" || exit 1

run "speaker ID: threshold, margin, too-short, location order, profiles, config" python3 check_speaker_id.py

STT_PY="$(need_python stt_python)"
VOICES="$(setting speaker_id_voices_dir)"
if [ -z "$STT_PY" ]; then
    skip "speaker ID end to end through the ASR server (real model)" "stt_python is not set or not executable"
elif [ -z "$VOICES" ] || [ ! -d "$VOICES" ]; then
    skip "speaker ID end to end through the ASR server (real model)" "speaker_id_voices_dir is not set (render it with make_synthetic_voices.py in the kokoro env)"
elif [ "$QUICK" = 1 ] || [ "$OFFLINE" = 1 ]; then
    skip "speaker ID end to end through the ASR server (real model)" "--quick / --offline (the models may need downloading)"
else
    run "speaker ID end to end through the ASR server (real model)" "$STT_PY" check_speaker_id_model.py
fi

finish

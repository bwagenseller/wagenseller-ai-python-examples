#!/usr/bin/env bash
#
# conversational_ai: the conversational AI pipeline (scripts/ai/combos/conversational_ai/ and its library in
# amadeo_utils/ai/combined/conversational_ai/) - wake words, agents, handoff notes, the server's ASR gate and the client's config.
#
# No models, audio or sockets. The wake-word and server checks run in any Python 3. The client-config check loads
# the real client script, which imports sounddevice / pygame / webrtcvad, so it needs media_python.

source "$(dirname "$0")/../testlib/suite_lib.sh"
cd "$(dirname "$0")" || exit 1

run "wake words and agent configs" python3 check_wake_words.py
run "handoff notes between agents" python3 check_handoff.py
run "LLM routing: the question and reading the answer" python3 check_routing.py
run "server ASR gate, routing, handoff, per-agent LLM sessions, cleanup" python3 check_asr_gate.py

MEDIA_PY="$(need_python media_python)"
if [ -z "$MEDIA_PY" ]; then
    skip "client config loader" "media_python is not set or not executable"
else
    run "client config loader" "$MEDIA_PY" check_client_config.py
fi

finish

#!/usr/bin/env bash
#
# client_server: AmadeoServer / AmadeoClient - the socket layer every Amadeo server and client shares.
#
# check_idle_timeout.py --dead-peer (a peer that vanishes without closing) needs root for iptables and is not run
# here; run it by hand when that path changes.
# Needs: llama_python (the stream servers it starts import llama_cpp).

source "$(dirname "$0")/../testlib/suite_lib.sh"
cd "$(dirname "$0")" || exit 1

PY="$(need_python llama_python)"
if [ -z "$PY" ]; then
    skip "idle timeout, keepalive and client reconnect" "llama_python is not set or not executable"
    finish
fi
run "idle timeout, keepalive and client reconnect (real sockets)" "$PY" check_idle_timeout.py

finish

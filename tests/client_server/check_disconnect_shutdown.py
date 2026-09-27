#!/usr/bin/env python3
"""CS-22: AmadeoServer runs its shutdown hook however a persistent session ends, not only on terminate_session.

Why
---
AmadeoServer's 'additional_shutdown' hook frees what a session holds: the conversational AI server's ASR / LLM
connections, and each LLM stream server's session (history, vector DB). It only ran on 'terminate_session', so a
client that just went away - closed its socket, crashed, lost the network, or idled past the server's limit - left
all of that behind for the life of the server.

What it proves (the real AmadeoServer in a subprocess, the real AmadeoClient and raw sockets)
---------------------------------------------------------------------------------------------
1. terminate_session: the hook runs, exactly once (not again when the connection then closes).
2. The client closes its socket without terminating: the hook runs, once.
3. The server's idle limit closes the connection: the hook runs, once.
4. A one-off (non-persistent) request: the hook does NOT run - there was no session to end.
5. A connection that closes before establishing a session: the hook does NOT run.
6. A hook that raises does not take the server down: the next client is still served.

Usage:  python check_disconnect_shutdown.py      (any Python 3 with the repo's src on the path)
"""
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time

# testlib finds the library under test (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testlib import settings  # noqa: E402
settings.use_src()
SRC = settings.SRC
sys.path.insert(0, SRC)
failures = []

# A real AmadeoServer whose shutdown hook prints the session it was given. argv: port, client_timeout. The hook
# raises for a session whose id starts with 'boom', to show a failing hook is contained.
SERVER = r'''
import logging, sys
sys.path.insert(0, {src!r})
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
from amadeo_utils.server.amadeo_server import AmadeoServer
port, timeout = int(sys.argv[1]), float(sys.argv[2])

def handler(request, data):
    return {{"success": True, "type": "system_message", "message": "ok", "file_size": 0}}, None

def shutdown(session_id):
    print("HOOK " + session_id, flush=True)
    if session_id.startswith("boom"):
        raise RuntimeError("hook failed on purpose")

AmadeoServer("127.0.0.1", port, client_timeout=timeout, additional_client_functionality=handler,
             additional_shutdown=shutdown).start_server()
'''


def check(label, got, expected):
    """Records one comparison and prints PASS / FAIL."""
    if got == expected:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}: got {got!r}, expected {expected!r}")
        failures.append(label)


def free_port():
    """A TCP port nothing is listening on."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """The real AmadeoServer in a subprocess; its output is collected to a file."""

    def __init__(self, timeout):
        self.port = free_port()
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        self.proc = subprocess.Popen([sys.executable, "-u", "-c", SERVER.format(src=SRC), str(self.port), str(timeout)],
                                     stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)

    def text(self):
        """Everything the server has printed so far."""
        self.log.flush()
        with open(self.log.name) as fh:
            return fh.read()

    def hook_count(self, session_id):
        """How many times the shutdown hook has run for session_id."""
        return self.text().count(f"HOOK {session_id}\n")

    def wait_for_hook(self, session_id, seconds=5):
        """Waits (up to seconds) for the hook to run for session_id, then a moment more to catch a second run."""
        deadline = time.time() + seconds
        while time.time() < deadline and not self.hook_count(session_id):
            time.sleep(0.1)
        time.sleep(0.5)
        return self.hook_count(session_id)

    def stop(self):
        """Stops the server and removes its log."""
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        os.unlink(self.log.name)


def send(sock, request):
    """Sends one request the way AmadeoClient does (4-byte length, JSON) and reads the reply's JSON."""
    body = json.dumps(dict(request, file_size=0)).encode()
    sock.sendall(struct.pack("!I", len(body)) + body)
    length = struct.unpack("!I", sock.recv(4))[0]
    reply = b""
    while len(reply) < length:
        reply += sock.recv(length - len(reply))
    return json.loads(reply)


def establish(port, session_id):
    """Opens a raw persistent connection asking for session_id; returns the socket."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    reply = send(sock, {"persistent": True, "command": "establish_connection", "sessionID": session_id})
    assert reply.get("success") and reply.get("sessionID") == session_id, reply
    return sock


server = Server(timeout=0)
try:
    # 1. terminate_session, through the real client
    from amadeo_utils.client.amadeo_client import AmadeoClient
    client = AmadeoClient("127.0.0.1", server.port, additional_server_response_functionality=lambda r, d: None,
                          session_id="terminated", persistent_request_timeout=5)
    client.establish_persistent_connection()
    client.close_connection()
    check("terminate_session: the hook runs exactly once", server.wait_for_hook("terminated"), 1)

    # 2. the client just closes its socket
    establish(server.port, "vanished").close()
    check("closed without terminate: the hook runs once", server.wait_for_hook("vanished"), 1)

    # 4. a one-off request
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    send(sock, {"persistent": False, "command": "request", "sessionID": "oneoff"})
    sock.close()
    check("one-off request: no hook", server.wait_for_hook("oneoff", seconds=1), 0)

    # 5. connected, but never established a session
    sock = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    sock.close()
    time.sleep(0.5)
    check("no session established: no hook at all", server.text().count("HOOK "), 2)

    # 6. a failing hook is contained
    establish(server.port, "boom1").close()
    check("a failing hook ran", server.wait_for_hook("boom1"), 1)
    establish(server.port, "after-boom").close()
    check("the server still serves the next client", server.wait_for_hook("after-boom"), 1)
finally:
    server.stop()

# 3. the idle limit closes the connection
server = Server(timeout=1)
try:
    sock = establish(server.port, "idled")
    check("idle limit: the hook runs once", server.wait_for_hook("idled", seconds=4), 1)
    sock.close()
finally:
    server.stop()

print(f"\n{len(failures)} failure(s)")
sys.exit(len(failures))

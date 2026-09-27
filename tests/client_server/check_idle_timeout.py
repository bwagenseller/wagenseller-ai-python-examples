#!/usr/bin/env python3
"""CS-21: the server's idle timeout, 'never' with TCP keepalive, and the client's reconnect.

Why
---
2026-09-25, found on a live tool server: AmadeoServer closed any client idle for 300 s
(CLIENT_TIMEOUT) and dropped its session - the conversation with it - and LlamaStreamClient said
nothing: the next request silently got no reply. For an always-on assistant the connection must
survive idleness. Now client_timeout 0 = never, with TCP keepalive so a client that VANISHED is
still detected, and the client reports a lost connection, reconnects, and resends the request only
when it cannot have run.

What it proves (the real AmadeoServer in a subprocess, the real AmadeoClient / LlamaStreamClient)
------------------------------------------------------------------------------------------
1. timeout 1 s: an idle client is closed and its session cleaned up (the default behaviour, kept).
2. timeout 0: an idle client survives well past 1 s, and its server-side connection carries a
   keepalive timer (read from the kernel with `ss`, not from our own code).
3. --dead-peer (needs root, for iptables): with keepalive shortened to 1 s x 2 probes, a client whose
   packets are dropped is declared dead and its session cleaned up within seconds.
4. LlamaStreamClient end to end: after the server closes an idle connection, the next request prints
   "Lost the connection", reconnects with a new session, and the request is answered - sent exactly
   once to the server that answered it.
5. recover_connection alone: after a TIMEOUT (the server may still be working) it reconnects but does
   NOT resend; after a closed connection it resends.
6. establish_connection: a requested sessionID of None gets a new id (it used to be granted as a session
   named None); a non-string id is refused; a requested string id is still granted.

Usage:  python check_idle_timeout.py [--dead-peer]      (any env with the repo's src on the path)
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testlib import settings  # noqa: E402
settings.use_src()
SRC = settings.SRC
ROOT = os.path.dirname(SRC)
CLIENT_SCRIPT = os.path.join(ROOT, "scripts", "ai", "llm", "llama", "llama_stream", "LlamaStreamClient.py")
sys.path.insert(0, SRC)
failures = []

# A real AmadeoServer with an echo handler. argv: port, client_timeout, keepalive idle, interval, count.
SERVER = r'''
import logging, sys
sys.path.insert(0, {src!r})
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
from amadeo_utils.server.amadeo_server import AmadeoServer
port, timeout, idle, interval, count = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
AmadeoServer.KEEPALIVE_IDLE, AmadeoServer.KEEPALIVE_INTERVAL, AmadeoServer.KEEPALIVE_COUNT = idle, interval, count

def handler(request, data):
    if request.get("command") == "request":
        print("HANDLED request " + request.get("user_request", ""), flush=True)
        return {{"success": True, "type": "llm_response", "response": "ECHO:" + request.get("user_request", ""),
                "elapsed_time": 0, "file_size": 0}}, None
    return {{"success": True, "type": "system_message", "message": "session ok", "file_size": 0}}, None

AmadeoServer("127.0.0.1", port, client_timeout=timeout, additional_client_functionality=handler).start_server()
'''


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if not ok and detail else ""))
    if not ok:
        failures.append(label)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """The real AmadeoServer in a subprocess; its log is collected to a file."""

    def __init__(self, timeout, keepalive=(60, 10, 6)):
        self.port = free_port()
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        self.proc = subprocess.Popen([sys.executable, "-u", "-c", SERVER.format(src=SRC), str(self.port), str(timeout),
                                      *map(str, keepalive)], stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)

    def text(self):
        self.log.flush()
        with open(self.log.name) as fh:
            return fh.read()

    def wait_for(self, fragment, seconds):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if fragment in self.text():
                return True
            time.sleep(0.2)
        return False

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        os.unlink(self.log.name)


def connected_client(port):
    from amadeo_utils.client.amadeo_client import AmadeoClient
    client = AmadeoClient("127.0.0.1", port, additional_server_response_functionality=lambda r, d: None,
                          persistent_request_timeout=10)
    assert client.establish_persistent_connection(), "could not connect"
    return client


def keepalive_timer_on(port):
    """The kernel's view of the server-side connection: `ss -o` shows 'timer:(keepalive,...)' only with SO_KEEPALIVE."""
    out = subprocess.run(["ss", "-tno", "state", "established", f"( sport = :{port} )"], capture_output=True, text=True).stdout
    return "keepalive" in out, out.strip()


def case_idle_timeout():
    print("== 1. an idle limit still closes an idle client ==")
    server = Server(timeout=1)
    try:
        client = connected_client(server.port)
        time.sleep(2.5)
        response, _ = client.send_persistent_request("request", user_request="late")
        check(response is None and not isinstance(client.last_error, (socket.timeout, TimeoutError)),
              "after 2.5 s idle with a 1 s limit the request fails as 'connection closed', not as a timeout",
              f"response={response} last_error={client.last_error!r}")
        check(server.wait_for("timed out after 1.0 idle seconds", 3) and "Cleaned up session" in server.text(),
              "the server logs the idle timeout and cleans up the session", server.text()[-400:])
    finally:
        server.stop()


def case_never():
    print("== 2. client_timeout 0: an idle client survives, with keepalive on ==")
    server = Server(timeout=0)
    try:
        client = connected_client(server.port)
        on, detail = keepalive_timer_on(server.port)
        check(on, "the server-side connection has a TCP keepalive timer (kernel, via ss)", detail)
        time.sleep(3)
        response, _ = client.send_persistent_request("request", user_request="still here")
        check(response is not None and response.get("response") == "ECHO:still here",
              "after 3 s idle the connection still works", f"response={response} last_error={client.last_error!r}")
        check("timed out" not in server.text(), "the server did not time the client out")
    finally:
        server.stop()
    control = Server(timeout=5)
    try:
        client = connected_client(control.port)         # keep a reference: a collected client closes its socket
        on, detail = keepalive_timer_on(control.port)
        check(not on, "with an idle limit set, keepalive is NOT turned on (other servers unchanged)", detail)
    finally:
        control.stop()


def case_dead_peer():
    print("== 3. keepalive catches a client that vanished (iptables drops its packets) ==")
    server = Server(timeout=0, keepalive=(1, 1, 2))
    rule = ["INPUT", "-i", "lo", "-p", "tcp", "--dport", str(server.port), "-j", "DROP"]
    try:
        client = connected_client(server.port)          # keep a reference: a collected client closes its socket
        subprocess.run(["sudo", "iptables", "-I", *rule], check=True)
        started = time.time()
        caught = server.wait_for("stopped answering keepalive probes", 15)
        check(caught and "Cleaned up session" in server.text(),
              f"declared dead and cleaned up after {time.time() - started:.1f} s (keepalive 1 s idle, 1 s x 2 probes)",
              server.text()[-400:])
    finally:
        subprocess.run(["sudo", "iptables", "-D", *rule])
        server.stop()


def case_client_reconnects():
    print("== 4. LlamaStreamClient reports a closed connection, reconnects, and resends ==")
    server = Server(timeout=1)
    config = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"host": "127.0.0.1", "port": server.port, "mode": "agent", "user_id": "tester",
               "spoken_response": False, "continuous_save": False, "load_previous": False}, config)
    config.close()
    env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get("PYTHONPATH", ""))
    client = subprocess.Popen([sys.executable, "-u", CLIENT_SCRIPT, "--json", config.name], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        client.stdin.write("hello\n"); client.stdin.flush()
        server.wait_for("HANDLED request hello", 5)
        time.sleep(2.5)                                    # past the 1 s idle limit: the server closes the connection
        client.stdin.write("again\n"); client.stdin.flush()
        server.wait_for("HANDLED request again", 8)
        client.stdin.write("!exit\n"); client.stdin.flush()
        out, _ = client.communicate(timeout=15)
        check("ECHO:hello" in out, "the first request is answered", out[-600:])
        check("Lost the connection to the server" in out, "the client SAYS the connection was lost", out[-600:])
        check("Reconnected with a new session" in out, "it reconnects with a new session", out[-600:])
        check("ECHO:again" in out and server.text().count("HANDLED request again") == 1,
              "the lost request is answered after reconnecting - sent once", out[-600:])
    finally:
        if client.poll() is None:
            client.kill()
        server.stop()
        os.unlink(config.name)


def case_recover_logic():
    print("== 5. recover_connection: resend after a closed connection, never after a timeout ==")
    sys.path.insert(0, os.path.dirname(CLIENT_SCRIPT))
    import LlamaStreamClient as module

    class FakeSocketClient:
        persistent_request_timeout = 600

        def __init__(self, error):
            self.last_error, self.sent = error, []

        def close_connection(self):
            pass

        def establish_persistent_connection(self):
            return True

        def send_persistent_request(self, command, **kwargs):
            self.sent.append(command)
            return {"success": True}, None

    for error, label, resend in ((socket.timeout("timed out"), "after a timeout", False),
                                 (ValueError("Invalid response header"), "after the server closed it", True)):
        client = object.__new__(module.LlamaStreamClient)
        client.argsDict = {"mode": "agent", "user_id": "t", "player_name": "", "system_prompt_id": "default",
                           "spoken_response": False,
                           "continuous_save": False,
                           "load_previous": False}
        client.socket_client = FakeSocketClient(error)
        carry_on = client.recover_connection("do the thing")
        check(carry_on and client.socket_client.sent == (["create_llm_session", "request"] if resend else ["create_llm_session"]),
              f"{label}: reconnect{' and resend' if resend else ', no resend'}", str(client.socket_client.sent))


def case_session_ids():
    print("== 6. establish_connection: None is a new session, not one named None ==")
    from amadeo_utils.client.amadeo_client import AmadeoClient
    server = Server(timeout=5)
    try:
        def establish(requested):
            client = AmadeoClient("127.0.0.1", server.port, additional_server_response_functionality=lambda r, d: None)
            client.client_socket = socket.create_connection(("127.0.0.1", server.port), timeout=5)
            client.send_request_data({"persistent": True, "sessionID": requested, "requestID": "r1",
                                      "command": "establish_connection", "message": "", "file_size": 0}, None)
            response, _ = client.receive_response()
            return client, response
        c1, r = establish(None)
        check(r.get("success") and isinstance(r.get("sessionID"), str) and len(r.get("sessionID")) > 8,
              "a requested sessionID of None gets a newly generated id", str(r))
        c2, r = establish(123)
        check(not r.get("success") and "Invalid sessionID" in r.get("message", ""),
              "a non-string sessionID is refused", str(r))
        c3, r = establish("my-own-session")
        check(r.get("success") and r.get("sessionID") == "my-own-session", "a requested string id is still granted", str(r))
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dead-peer", action="store_true", help="also run case 3 (needs sudo for iptables)")
    args = parser.parse_args()
    case_idle_timeout()
    case_never()
    if args.dead_peer:
        case_dead_peer()
    case_client_reconnects()
    case_recover_logic()
    case_session_ids()
    print("\nALL CHECKS PASS" if not failures else f"\n{len(failures)} check(s) FAILED")
    sys.exit(len(failures))


if __name__ == "__main__":
    main()

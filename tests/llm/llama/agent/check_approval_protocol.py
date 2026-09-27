#!/usr/bin/env python3
"""CS-21: the mid-turn approval question, over a real TCP connection.

What it proves
--------------
The server half is the real ``ToolStream.approve_tool_call`` (no model is loaded - it only
needs the socket), the client half is the real ``AmadeoClient.send_persistent_request``
with an interim handler, joined by a real localhost connection. For each case the server
receives a request, asks one approval question part-way through, then sends the request's
real response, which the client must still receive.

Cases (owner's decision, 2026-09-24): only a case-insensitive 'y' or 'yes' approves; anything else is
no; no answer within the timeout is no; a client with no interim handler is answered "no"
automatically; a turn with no client connection at all is no. The question must name the
tool and carry the exact arguments.

Usage (llama env):  python check_approval_protocol.py
"""
import os
import socket
import sys
import threading
import time

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()
from amadeo_utils.ai.llm.llama.ToolStream import ToolStream   # noqa: E402
from amadeo_utils.client.amadeo_client import AmadeoClient     # noqa: E402

ARGS = {"path": "/tmp/notes.txt", "text": "buy milk"}
failures = []


def run_case(label, handler, expect_decision, approval_timeout=2.0, client_delay=0.0):
    """One request: the server asks one question, then answers; returns what both ends saw."""
    stream = object.__new__(ToolStream)               # no model: approve_tool_call needs only these
    stream.approval_timeout = approval_timeout
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    seen = {}

    def server():
        conn, _ = listener.accept()
        conn.settimeout(10)
        seen["request"] = ToolStream._receive_frame(conn)
        seen["decision"] = stream.approve_tool_call({"_client_socket": conn, "session_id": "s1"}, "write_note", ARGS)
        ToolStream._send_frame(conn, {"success": True, "type": "llm_response", "response": f"done:{seen['decision']}",
                                      "message": "", "file_size": 0})
        time.sleep(0.2)
        conn.close()

    thread = threading.Thread(target=server, daemon=True)
    thread.start()

    questions = []

    def recording_handler(message):
        questions.append(message)
        time.sleep(client_delay)
        return handler(message) if handler else None

    client = AmadeoClient("127.0.0.1", port, additional_server_response_functionality=lambda r, d: None,
                          interim_response_functionality=recording_handler if handler is not None or client_delay else None)
    client.client_socket = socket.create_connection(("127.0.0.1", port))
    client.client_socket.settimeout(10)
    client.is_persistent, client.session_id = True, "s1"
    response, _ = client.send_persistent_request("request", user_request="write it down")
    thread.join(5)
    listener.close()

    ok = seen.get("decision") == expect_decision
    if handler is not None and client_delay == 0:
        q = questions[0] if questions else {}
        ok = ok and q.get("type") == "approval_request" and q.get("tool") == "write_note" and q.get("arguments") == ARGS \
            and q.get("timeout_seconds") == approval_timeout
    # Whatever happened mid-turn, the client must still get the request's REAL response afterwards - except after a
    # server-side timeout, where the late answer is still in flight (see the note in the summary).
    if client_delay == 0:
        ok = ok and response is not None and response.get("response") == f"done:{expect_decision}"
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: decision={seen.get('decision')!r}"
          + ("" if ok else f"  response={response!r} questions={questions!r}"))
    if not ok:
        failures.append(label)


def answer(text):
    return lambda message: {"command": "approval_response", "answer": text}


def main():
    run_case("'y' approves", answer("y"), "approve")
    run_case("'YES' approves (case-insensitive)", answer("YES"), "approve")
    run_case("' yes ' approves (whitespace trimmed)", answer(" yes "), "approve")
    run_case("'yep' is no", answer("yep"), "deny")
    run_case("empty answer is no", answer(""), "deny")
    run_case("a reply that is not an approval_response is no", lambda m: {"command": "request", "answer": "y"}, "deny")
    run_case("client with no interim handler answers no automatically", None, "deny")
    # The server's own deadline is only a backstop for a hung client: it waits APPROVAL_GRACE_SECONDS past the
    # client's. Shrink the grace so the backstop case runs in a second.
    ToolStream.APPROVAL_GRACE_SECONDS = 0.2
    run_case("hung client: the server's backstop deadline is no", answer("y"), "timeout", approval_timeout=0.3, client_delay=1.5)

    stream = object.__new__(ToolStream)
    stream.approval_timeout = 1.0
    decision = stream.approve_tool_call({"session_id": "local"}, "write_note", ARGS)
    ok = decision == "timeout"
    print(f"  {'PASS' if ok else 'FAIL'}  no client connection (local caller) is no: decision={decision!r}")
    if not ok:
        failures.append("local")

    print("\nALL CHECKS PASS" if not failures else f"\n{len(failures)} check(s) FAILED")
    sys.exit(len(failures))


if __name__ == "__main__":
    main()

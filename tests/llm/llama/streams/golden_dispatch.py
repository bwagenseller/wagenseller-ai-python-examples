#!/usr/bin/env python3
"""CS-20: capture the dispatch and session-lifecycle behaviour of both stream families.

This is the golden-output harness described in the CS-20 design notes. It exists to prove
that extracting a shared base class from ``RolePlayStream`` and ``KnowledgeBaseStream``
changes nothing. Run it on current ``main`` to record a baseline, run it again after the
refactor, and diff the two JSON files: the diff must be empty.

What it covers
--------------
``handle_client_request`` is the method being lifted into the base as a template method,
so its branch behaviour is the single highest-value thing to pin down. Every dispatch
path is exercised and three things are recorded per case: the response dictionary, the
second tuple element, and which of ``create_session`` / ``get_response`` was called with
which arguments. The session-lifecycle methods (``get_session``,
``get_session_and_lock``, ``remove_session``) lift verbatim and are exercised too,
including the contracts their docstrings claim - (None, None) for a missing session, and
a second removal being a no-op rather than a KeyError.

How it avoids the GPU
---------------------
Instances are built with ``object.__new__`` so ``__init__`` never runs and no model is
ever loaded; only the attributes the dispatch and lifecycle code actually touch are
injected. ``create_session`` and ``get_response`` are replaced per instance with
recorders, because both are heavy (a VectorDB, a real generation) and because what
matters here is *that they were called, and with what*, not what they return. Everything
below the dispatch layer is therefore out of scope for this harness by design - prompt
rendering and token counts are captured separately, against a real model.

Determinism
-----------
Nothing here samples, sleeps, or reads the clock. Lock objects are recorded as a
presence flag rather than by identity, since their ``repr`` embeds a memory address that
would differ between runs. Output is JSON with sorted keys.

Usage
-----
    python golden_dispatch.py --out baseline.json        # on current main, before refactoring
    python golden_dispatch.py --out after.json           # after the refactor
    diff baseline.json after.json                        # must be empty

Must run where ``llama_cpp`` imports - i.e. the ``llama`` conda environment.
"""
import argparse
import json
import os
import sys
import threading

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()

from amadeo_utils.ai.llm.llama.RolePlayStream import RolePlayStream
from amadeo_utils.ai.llm.llama.KnowledgeBaseStream import KnowledgeBaseStream

# Marker returned by the stubbed create_session. RolePlayStream reads 'system_message' off
# this returned dict; KnowledgeBaseStream reads it off self.argsDict instead. That
# asymmetry is a real divergence between the two families, so both values are distinct
# and both are recorded - whichever one surfaces in the response tells us which code path
# ran, and the refactor must not swap them.
STUB_SESSION_MESSAGE = "<FROM-create_session-RETURN>"
STUB_ARGS_MESSAGE = "<FROM-argsDict>"


def build(cls):
    """Build a stream instance with no model loaded and recording stubs in place.

    ``__init__`` loads two Llama models, so it is bypassed entirely with
    ``object.__new__``. Only the attributes the dispatch and lifecycle code reads are
    injected; if the refactor makes those methods depend on something new, this harness
    will fail loudly with an AttributeError, which is itself a useful signal.

    Args:
        cls: RolePlayStream or KnowledgeBaseStream.

    Returns:
        tuple: (instance, calls) where calls is the list the stubs append to.
    """
    inst = object.__new__(cls)
    inst.argsDict = {"system_message": STUB_ARGS_MESSAGE}
    inst.sessions = {}
    inst.sessions_lock = threading.Lock()
    inst.session_locks = {}
    inst.models_released = False

    calls = []

    def fake_create_session(*args, **kwargs):
        """Stand in for the real create_session: record the call, register the session."""
        calls.append({"method": "create_session", "args": list(args), "kwargs": dict(kwargs)})
        # Register a minimal session so that a follow-up request in the same case sees an
        # existing session, exactly as the real method would.
        session_id = args[0] if args else kwargs.get("session_id")
        with inst.sessions_lock:
            inst.sessions[session_id] = {"session_id": session_id,
                                         "system_message": STUB_SESSION_MESSAGE}
            inst.session_locks[session_id] = threading.Lock()
        return inst.sessions[session_id]

    def fake_get_response(request):
        """Stand in for the real get_response: record the call, return a fixed dict."""
        calls.append({"method": "get_response", "request": request})
        return {"stubbed": "get_response"}

    inst.create_session = fake_create_session
    inst.get_response = fake_get_response
    return inst, calls


# Every dispatch branch in handle_client_request, named so a diff points at the case.
# 'pre' seeds a session before the request, to reach the "already exists" branches.
CASES = [
    ("create_new_full", {"sessionID": "s1", "command": "create_llm_session",
                         "user_id": "u1", "system_prompt_id": "sp1", "player_name": "Ada",
                         "spoken_response": False, "continuous_save": True,
                         "load_previous": False}, False),
    ("create_new_defaults", {"sessionID": "s2", "command": "create_llm_session"}, False),
    ("create_when_exists", {"sessionID": "s3", "command": "create_llm_session",
                            "user_request": "hello"}, True),
    ("request_with_text", {"sessionID": "s4", "command": "request",
                           "user_request": "hello"}, True),
    ("request_no_text", {"sessionID": "s5", "command": "request"}, True),
    ("request_empty_text", {"sessionID": "s6", "command": "request",
                            "user_request": ""}, True),
    ("unknown_command", {"sessionID": "s7", "command": "banana",
                         "user_request": "hello"}, True),
    ("missing_command", {"sessionID": "s8", "user_request": "hello"}, True),
    ("create_no_user_request", {"sessionID": "s9", "command": "create_llm_session"}, False),
    ("no_session_id", {"command": "request", "user_request": "hello"}, False),
    ("none_session_id", {"sessionID": None, "command": "request",
                         "user_request": "hello"}, False),
]


def dispatch_matrix(cls):
    """Run every dispatch case against a fresh instance and record what happened."""
    out = {}
    for name, request, seed_session in CASES:
        inst, calls = build(cls)
        if seed_session:
            # Pre-register the session so the "already exists" branches are reachable
            # without going through the stubbed create_session.
            sid = request.get("sessionID")
            inst.sessions[sid] = {"session_id": sid, "system_message": STUB_SESSION_MESSAGE}
            inst.session_locks[sid] = threading.Lock()

        try:
            response, extra = inst.handle_client_request(dict(request))
            result = {"response": response, "second_value": extra}
        except Exception as e:  # a raised exception is itself behaviour worth pinning
            result = {"raised": f"{type(e).__name__}: {e}"}

        result["calls"] = calls
        result["sessions_after"] = sorted(inst.sessions.keys())
        out[name] = result
    return out


def lifecycle(cls):
    """Exercise the three session-lifecycle methods, including their documented edge cases."""
    inst, _ = build(cls)
    results = {}

    # Seed one session directly, bypassing create_session.
    inst.sessions["live"] = {"session_id": "live"}
    inst.session_locks["live"] = threading.Lock()

    results["get_session_existing"] = inst.get_session("live")
    results["get_session_missing"] = inst.get_session("ghost")

    session, lock = inst.get_session_and_lock("live")
    # Record the lock as a presence flag: its repr contains a memory address, which would
    # differ between runs and make the golden file non-deterministic.
    results["get_session_and_lock_existing"] = {"session": session, "lock_present": lock is not None}

    session, lock = inst.get_session_and_lock("ghost")
    results["get_session_and_lock_missing"] = {"session": session, "lock_present": lock is not None}

    inst.remove_session("live")
    results["after_remove_get_session"] = inst.get_session("live")
    results["after_remove_sessions"] = sorted(inst.sessions.keys())
    results["after_remove_locks"] = sorted(inst.session_locks.keys())

    # A second removal must be a quiet no-op, not a KeyError - the docstring says a double
    # shutdown for the same session is expected.
    try:
        inst.remove_session("live")
        results["double_remove"] = "no_error"
    except Exception as e:
        results["double_remove"] = f"{type(e).__name__}: {e}"

    try:
        inst.remove_session("never_existed")
        results["remove_unknown"] = "no_error"
    except Exception as e:
        results["remove_unknown"] = f"{type(e).__name__}: {e}"

    return results


def main():
    parser = argparse.ArgumentParser(description="CS-20 golden dispatch/lifecycle capture")
    parser.add_argument("--out", required=True, help="path to write the JSON capture to")
    args = parser.parse_args()

    capture = {}
    for cls in (RolePlayStream, KnowledgeBaseStream):
        capture[cls.__name__] = {
            "dispatch": dispatch_matrix(cls),
            "lifecycle": lifecycle(cls),
        }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(capture, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    # A short console summary so a run is readable without opening the JSON.
    for family, data in sorted(capture.items()):
        errors = [k for k, v in data["dispatch"].items() if "raised" in v]
        print(f"{family}: {len(data['dispatch'])} dispatch cases, "
              f"{len(data['lifecycle'])} lifecycle assertions"
              + (f", raised in: {errors}" if errors else ""))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

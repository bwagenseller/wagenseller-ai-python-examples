#!/usr/bin/env python3
"""CS-25: optional encryption of system prompts (no model loaded).

Why
---
Role-play and the agent already encrypt chat history when 'encrypted' is set, but the prompts themselves were read
as plaintext. A prompt may now be stored as '<name>.txt.enc' (scripts/utils/amadeo_encryption.py, with the salt
'amadeo.salt' beside it) and is decrypted in memory with the same conversation passphrase. Encryption stays
OPTIONAL: a plaintext prompt must behave exactly as before.

What it proves (every file is made up, in a temporary folder)
---------------------------------------------------------------------------------------------------------------------
1. LlamaUtils helpers:
   * a plaintext prompt loads as before, needs no passphrase, and a missing one still falls back to the default;
   * '<name>.txt' that only exists as '<name>.txt.enc' is found and decrypted, as is '<name>.txt.enc' named directly;
   * no passphrase: get_system_message defers (None) when asked to, and raises otherwise - never the default prompt;
   * a wrong passphrase raises ValueError, and a missing salt raises FileNotFoundError WITHOUT creating a new salt;
   * plaintext beside its encrypted copy: the plaintext wins (the path as named comes first);
   * decrypting writes nothing to the prompt folder.
2. ToolStream.resolve_system_prompt (built with object.__new__, token counting stubbed): an encrypted client prompt
   is decrypted; a wrong passphrase fails the session; a plaintext prompt is unchanged.
3. RolePlayStream.create_session (same): an encrypted prompt is decrypted; a wrong prompt passphrase refuses the
   session through 'fatal_errors' rather than running on the default prompt.
4. LlamaUtils.str_to_bool: '--encrypted False' is now False (argparse's type=bool made it True).
5. VectorDB: encrypted history whose 'convo.salt' is missing is refused with an error naming the salt, and no new
   salt is created in its place; with the salt restored, the same passphrase opens it again.
6. protect_process_memory (in a child process, since it is one-way): the process becomes non-dumpable with a core
   limit of 0, and another process of the same user can no longer read its /proc/<pid>/environ. RolePlayStream's
   post_model_init applies it BEFORE asking for the passphrase when 'encrypted' is set, and not at all otherwise.

Usage:  python check_prompt_encryption.py      (the llama env: the streams import llama_cpp; needs 'cryptography')
"""
import argparse
import os
import subprocess
import sys
import tempfile
import threading

# testlib finds the library under test (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()

from amadeo_utils.ai.llm.llama.llama_utils import LlamaUtils  # noqa: E402
from amadeo_utils.ai.llm.llama.ToolStream import ToolStream  # noqa: E402
from amadeo_utils.ai.llm.llama.RolePlayStream import RolePlayStream  # noqa: E402
from amadeo_utils.misc_utils.FileEncryption import AmadeoEncryption  # noqa: E402
from amadeo_utils.ai.llm.vector_database.VectorDB import VectorDB  # noqa: E402

PASSPHRASE = "correct horse battery staple"
SECRET_PROMPT = "You are Captain Nimbus, a weather balloon who answers in rhyme. ##You are talking to @@player@@.##"
PLAIN_PROMPT = "You are a plain, unencrypted helper."

failures = []


def check(label, got, expected):
    """Records one comparison and prints PASS / FAIL."""
    if got == expected:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}: got {got!r}, expected {expected!r}")
        failures.append(label)


def raises(label, exc_type, func, *args, **kwargs):
    """Records that func(*args, **kwargs) raises exc_type."""
    try:
        func(*args, **kwargs)
    except exc_type as e:
        print(f"PASS  {label} ({type(e).__name__})")
        return
    except Exception as e:  # noqa: BLE001 - any other exception is the failure being reported
        print(f"FAIL  {label}: raised {type(e).__name__}: {e}")
        failures.append(label)
        return
    print(f"FAIL  {label}: nothing was raised")
    failures.append(label)


def make_prompt_dir(root: str) -> str:
    """
    A prompt folder holding 'secret.txt.enc' (encrypted with PASSPHRASE and the folder's amadeo.salt, exactly as
    amadeo_encryption.py would) and 'plain.txt'.
    """
    prompt_dir = os.path.join(root, "prompts")
    os.makedirs(prompt_dir)
    AmadeoEncryption(PASSPHRASE, salt_path=os.path.join(prompt_dir, "amadeo.salt")).encrypt_and_save(
        SECRET_PROMPT, os.path.join(prompt_dir, "secret.txt.enc"))
    with open(os.path.join(prompt_dir, "plain.txt"), "w", encoding="utf-8") as f:
        f.write(PLAIN_PROMPT)
    return prompt_dir


def check_helpers(root: str, prompt_dir: str):
    """Part 1: the LlamaUtils helpers."""
    plain = os.path.join(prompt_dir, "plain.txt")
    secret = os.path.join(prompt_dir, "secret.txt")
    quiet = lambda *_: None  # noqa: E731 - swallow log lines

    check("plaintext prompt loads with no passphrase", LlamaUtils.get_system_message(plain, quiet), PLAIN_PROMPT)
    check("missing plaintext prompt still falls back to the default",
          LlamaUtils.get_system_message(os.path.join(prompt_dir, "nope.txt"), quiet), LlamaUtils.BASE_SYSTEM_MESSAGE)
    check("plaintext prompt is not 'encrypted'", LlamaUtils.is_encrypted_prompt(plain), False)
    check("'secret.txt' resolves to its '.enc' twin", LlamaUtils.find_prompt_file(secret), secret + ".enc")
    check("'secret.txt' is 'encrypted'", LlamaUtils.is_encrypted_prompt(secret), True)

    before = sorted(os.listdir(prompt_dir))
    check("encrypted prompt decrypts by its plain name",
          LlamaUtils.get_system_message(secret, quiet, passphrase=PASSPHRASE), SECRET_PROMPT)
    check("encrypted prompt decrypts by its '.enc' name",
          LlamaUtils.load_system_prompt(secret + ".enc", quiet, PASSPHRASE), SECRET_PROMPT)
    check("decrypting wrote nothing to the prompt folder", sorted(os.listdir(prompt_dir)), before)

    check("no passphrase + defer_encrypted -> None",
          LlamaUtils.get_system_message(secret, quiet, defer_encrypted=True), None)
    raises("no passphrase, not deferred -> ValueError (no silent default)", ValueError,
           LlamaUtils.get_system_message, secret, quiet)
    raises("wrong passphrase -> ValueError (no silent default)", ValueError,
           LlamaUtils.get_system_message, secret, quiet, passphrase="not the passphrase")

    # A prompt copied somewhere without its salt: refuse, and do not mint a new salt in its place.
    lonely = os.path.join(root, "lonely")
    os.makedirs(lonely)
    with open(secret + ".enc", "rb") as src, open(os.path.join(lonely, "secret.txt.enc"), "wb") as dst:
        dst.write(src.read())
    raises("missing salt -> FileNotFoundError", FileNotFoundError,
           LlamaUtils.get_system_message, os.path.join(lonely, "secret.txt"), quiet, passphrase=PASSPHRASE)
    check("missing salt: no new salt was created", os.path.exists(os.path.join(lonely, "amadeo.salt")), False)

    # Plaintext left beside the encrypted copy: the path as named wins, with a warning.
    both = os.path.join(prompt_dir, "both.txt")
    with open(both, "w", encoding="utf-8") as f:
        f.write(PLAIN_PROMPT)
    AmadeoEncryption(PASSPHRASE, salt_path=os.path.join(prompt_dir, "amadeo.salt")).encrypt_and_save(
        SECRET_PROMPT, both + ".enc")
    warnings = []
    check("plaintext beside its '.enc' copy wins",
          LlamaUtils.load_system_prompt(both, warnings.append, PASSPHRASE), PLAIN_PROMPT)
    check("...and a warning is logged", len(warnings), 1)
    os.remove(both)
    os.remove(both + ".enc")


def check_tool_stream(prompt_dir: str):
    """Part 2: ToolStream.resolve_system_prompt, with no model."""
    stream = object.__new__(ToolStream)
    stream.argsDict = {'system_prompt_dir': prompt_dir, 'system_message': PLAIN_PROMPT}
    stream.system_message, stream.system_tokens = PLAIN_PROMPT, 1
    stream.generating_gpu_lock = threading.Lock()
    stream.llm_generator, stream.model_type = None, None

    stream.passphrase = PASSPHRASE
    message, _, key, error = stream.resolve_system_prompt("secret", "Ada")
    check("ToolStream: encrypted prompt decrypted, no error", (key, error), ("secret", None))
    check("ToolStream: decrypted prompt personalised", message.startswith(SECRET_PROMPT.replace("##You are talking to @@player@@.##", "You are talking to Ada.")), True)

    stream.passphrase = "not the passphrase"
    _, _, key, error = stream.resolve_system_prompt("secret", "Ada")
    check("ToolStream: wrong passphrase fails the session", (key, error is not None), ("_invalid_prompt_", True))

    stream.passphrase = ""
    message, _, key, error = stream.resolve_system_prompt("plain", "")
    check("ToolStream: plaintext prompt unchanged without a passphrase", (key, error, message.startswith(PLAIN_PROMPT)),
          ("plain", None, True))


def check_role_play_stream(root: str, prompt_dir: str):
    """Part 3: RolePlayStream.create_session, with no model."""
    def build(passphrase):
        stream = object.__new__(RolePlayStream)
        stream.argsDict = {'system_prompt_dir': prompt_dir, 'base_convo_dir': os.path.join(root, "convos"),
                           'buffer_context_pcnt': 0.1, 'generating_max_context_tokens': 4096, 'debug': False}
        stream.sessions, stream.session_locks = {}, {}
        stream.sessions_lock, stream.generating_gpu_lock, stream.embedding_gpu_lock = threading.Lock(), threading.Lock(), threading.Lock()
        stream.llm_generator = stream.llm_embedder = stream.model_type = None
        stream.passphrase = passphrase
        return stream

    session = build(PASSPHRASE).create_session("s1", "user1", "Ada", "secret", False, False, False)
    check("RolePlayStream: encrypted prompt decrypted and personalised",
          (session['system_message'], session['fatal_errors']),
          (SECRET_PROMPT.replace("##You are talking to @@player@@.##", "You are talking to Ada."), ''))

    # The conversation and the prompts have separate salts, so a passphrase that opens (here: creates) the
    # conversation can still be the wrong one for the prompt.
    session = build("not the passphrase").create_session("s2", "user2", "Ada", "secret", False, False, False)
    check("RolePlayStream: wrong prompt passphrase refuses the session",
          "could not be decrypted" in session['fatal_errors'], True)

    session = build("").create_session("s3", "user3", "", "plain", False, False, False)
    check("RolePlayStream: plaintext prompt unchanged without a passphrase",
          (session['system_message'], session['fatal_errors']), (PLAIN_PROMPT, ''))


def check_str_to_bool():
    """Part 4: the '--encrypted' flag parses false as false."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--encrypted", type=LlamaUtils.str_to_bool, default=False)
    check("--encrypted False -> False", parser.parse_args(["--encrypted", "False"]).encrypted, False)
    check("--encrypted true -> True", parser.parse_args(["--encrypted", "true"]).encrypted, True)
    check("no --encrypted -> False", parser.parse_args([]).encrypted, False)
    raises("--encrypted maybe -> refused", SystemExit, parser.parse_args, ["--encrypted", "maybe"])


def check_missing_convo_salt(root: str):
    """Part 5: a lost convo.salt is reported as such, not as a wrong passphrase, and is never replaced."""
    convo_dir = os.path.join(root, "salt_convo")
    lock = threading.Lock()

    def open_db(passphrase=PASSPHRASE):
        return VectorDB(None, lock, None, lock, None, convo_dir, False, passphrase)

    open_db().save_session([{"role": "user", "content": "Is the balloon still up?"}])
    salt = os.path.join(convo_dir, "convo.salt")
    with open(salt, "rb") as f:
        original = f.read()
    os.remove(salt)

    try:
        open_db()
        print("FAIL  missing convo.salt: nothing was raised")
        failures.append("missing convo.salt raises")
    except ValueError as e:
        check("missing convo.salt: the error names the salt, not the passphrase",
              ("convo.salt" in str(e), "Incorrect passphrase" in str(e)), (True, False))
    check("missing convo.salt: no new salt was created", os.path.exists(salt), False)

    with open(salt, "wb") as f:
        f.write(original)
    check("restored convo.salt: history opens again", open_db().load_session()[0]["content"], "Is the balloon still up?")
    check("plaintext conversation needs no salt", open_db("").save_dir == convo_dir and not os.path.exists(os.path.join(convo_dir, "chat_history.json")), True)


def check_process_protection():
    """Part 6: the process protection, and that role-play applies it only when encrypted."""
    import amadeo_utils.ai.llm.llama.RolePlayStream as rps_module

    # One-way, so it runs in a child: report the state before and after, then wait to be inspected.
    child_code = (
        "import resource, sys, time\n"
        "from amadeo_utils.misc_utils.process_protection import protect_process_memory, is_process_dumpable\n"
        "before = is_process_dumpable()\n"
        "ok = protect_process_memory()\n"
        "print(before, ok, is_process_dumpable(), resource.getrlimit(resource.RLIMIT_CORE), flush=True)\n"
        "time.sleep(30)\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    child = subprocess.Popen([sys.executable, "-c", child_code], stdout=subprocess.PIPE, text=True, env=env)
    try:
        report = child.stdout.readline().strip()
        check("protect_process_memory: dumpable before, applied, not dumpable after, core limit 0",
              report, "True True False (0, 0)")
        try:
            with open(f"/proc/{child.pid}/environ", "rb") as f:
                f.read()
            readable = True
        except PermissionError:
            readable = False
        check("protect_process_memory: same-user read of /proc/<pid>/environ is refused", readable, False)
    finally:
        child.kill()
        child.wait()

    # RolePlayStream.post_model_init: protection first, then the passphrase prompt - and only when encrypted.
    calls = []
    real_protect, real_getpass = rps_module.protect_process_memory, rps_module.getpass.getpass
    rps_module.protect_process_memory = lambda *a, **k: calls.append("protect") or True
    rps_module.getpass.getpass = lambda *a, **k: calls.append("getpass") or PASSPHRASE
    try:
        for encrypted, expected in ((True, ["protect", "getpass"]), (False, [])):
            calls.clear()
            stream = object.__new__(RolePlayStream)
            stream.argsDict = {'encrypted': encrypted}
            stream.post_model_init()
            check(f"RolePlayStream.post_model_init encrypted={encrypted}: calls {expected}", calls, expected)
    finally:
        rps_module.protect_process_memory, rps_module.getpass.getpass = real_protect, real_getpass


def main() -> int:
    # Token counting needs a model; every stream check here only needs it to return a number.
    LlamaUtils.universal_token_count = staticmethod(lambda *args, **kwargs: 1)
    with tempfile.TemporaryDirectory() as root:
        prompt_dir = make_prompt_dir(root)
        check_helpers(root, prompt_dir)
        check_tool_stream(prompt_dir)
        check_role_play_stream(root, prompt_dir)
        check_missing_convo_salt(root)
    check_str_to_bool()
    check_process_protection()
    print(f"\n{len(failures)} failure(s)")
    return len(failures)


if __name__ == "__main__":
    sys.exit(main())

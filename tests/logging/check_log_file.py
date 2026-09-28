#!/usr/bin/env python3
"""Server log files: amadeo_utils.logging_utils.add_log_file, and the 'log_file' setting in the server configs.

Why
---
Every long-running server logs to the screen (a tmux pane). A 'log_file' in a server's JSON config sends the same
lines to a file as well, one file a day, kept 90 days. Without it, nothing changes. Server logs can hold what was said
near a microphone, so the file must not be world-readable, and a server must never refuse to start over its log.

What it proves (plain Python - no models)
-----------------------------------------
1. No path (None or '') adds nothing: screen only, as before.
2. With a path: every line reaches both the screen and the file; missing folders are created; the file is plain
   text (the console's colour codes stripped) and readable by owner and group only (0660).
3. Calling it again with the same file adds nothing (no doubled lines).
4. A file that cannot be opened is reported, and logging carries on to the screen only.
5. Rotation: a new file each midnight, and only the last 90 dated files are kept.
6. The config loaders that run in plain Python pass 'log_file' through: the conversational server's, and the
   shared llama server fields (knowledge-base and agent servers), and the role-play server's own loader - and a
   config without it gives ''. (The ASR, Kokoro and F5 loaders need their model environments; they follow the same
   pattern.)

Usage:  python check_log_file.py      (any Python 3 with the repo's src on the path)
"""
import io
import json
import logging
import os
import stat
import sys
import tempfile

# testlib finds the library under test (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testlib import settings  # noqa: E402
settings.use_src()

from amadeo_utils.logging_utils import add_log_file, LOG_RETENTION_DAYS  # noqa: E402

failures = []


def check(label, got, expected):
    """Records one comparison and prints PASS / FAIL."""
    if got == expected:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}: got {got!r}, expected {expected!r}")
        failures.append(label)


def fresh_logger(name):
    """A logger of its own with a 'screen' (a StringIO) attached, so the checks never touch the root logger."""
    logger = logging.getLogger(f"check_log_file.{name}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    screen = io.StringIO()
    logger.addHandler(logging.StreamHandler(screen))
    return logger, screen


with tempfile.TemporaryDirectory() as folder:
    # 1. no path
    logger, screen = fresh_logger('none')
    check("no log_file (None or ''): nothing added, screen only",
          (add_log_file(None, logger=logger), add_log_file('', logger=logger), len(logger.handlers)), (None, None, 1))

    # 2. screen and file
    logger, screen = fresh_logger('both')
    path = os.path.join(folder, 'new', 'folder', 'server.log')
    handler = add_log_file(path, logger=logger)
    logger.info("\033[1;34mWhisperXServer: loaded.\033[0m")
    logger.warning("plain line")
    handler.flush()
    with open(path, encoding='utf-8') as f:
        text = f.read()
    check("missing folders are created and the file is written", os.path.isfile(path), True)
    check("every line reaches the file...", ('WhisperXServer: loaded.' in text, 'plain line' in text), (True, True))
    check("...and still reaches the screen, colours and all", '\033[1;34mWhisperXServer: loaded.' in screen.getvalue(), True)
    check("the file has no colour codes", '\033[' in text, False)
    check("the file uses the console's line format", ' - WARNING - ' in text and '<module>:' in text, True)
    check("the file is readable by owner and group only", stat.S_IMODE(os.stat(path).st_mode) & 0o007, 0)

    # 3. again with the same file
    check("the same file twice adds nothing (no doubled lines)",
          (add_log_file(path, logger=logger) is handler, len(logger.handlers)), (True, 2))

    # 4. a file that cannot be opened
    logger2, screen2 = fresh_logger('broken')
    blocker = os.path.join(folder, 'not-a-folder')
    open(blocker, 'w').close()
    result = add_log_file(os.path.join(blocker, 'server.log'), logger=logger2)
    logger2.info("still logging")
    check("an unopenable file: reported, nothing added, the screen carries on",
          (result, len(logger2.handlers), 'Could not open log file' in screen2.getvalue(), 'still logging' in screen2.getvalue()),
          (None, 1, True, True))

    # 5. rotation: daily, 90 kept
    check("a new file each midnight, 90 days kept", (handler.when, handler.backupCount, LOG_RETENTION_DAYS), ('MIDNIGHT', 90, 90))
    for day in range(1, 96):     # 95 dated files from earlier days
        open(f"{path}.2026-01-01".replace('01-01', f"{(day - 1) // 28 + 1:02d}-{(day - 1) % 28 + 1:02d}"), 'w').close()
    handler.doRollover()
    dated = [f for f in os.listdir(os.path.dirname(path)) if f.startswith('server.log.')]
    check("after a rollover only the last 90 dated files are left", len(dated), 90)
    handler.close()

    # 6. the config loaders
    from amadeo_utils.ai.combined.conversational_ai.conversational_ai import ConversationalAiServer
    from amadeo_utils.ai.llm.llama.llama_utils import LlamaUtils

    def write(name, data):
        """Writes a JSON config and returns its path."""
        p = os.path.join(folder, name)
        with open(p, 'w') as f:
            json.dump(data, f)
        return p

    saved_argv = sys.argv
    try:
        sys.argv = ['server', '--json', write('convo.json', {'host': 'h', 'port': 1, 'log_file': '/logs/convo.log'})]
        with_file = ConversationalAiServer.get_args_dict_server()
        sys.argv = ['server', '--json', write('convo-plain.json', {'host': 'h', 'port': 1})]
        without = ConversationalAiServer.get_args_dict_server()
    finally:
        sys.argv = saved_argv
    check("conversational server: log_file from its config; '' without one", (with_file['log_file'], without['log_file']), ('/logs/convo.log', ''))

    check("llama servers: log_file is a shared optional server field", LlamaUtils.SERVER_SYSTEM_OPTIONAL_FIELDS.get('log_file'), str)
    base = {k: (1 if t in (int, float) else 'x') for k, t in LlamaUtils.SERVER_SYSTEM_REQUIRED_FIELDS.items()}
    check("llama servers: map_server_system_config passes log_file on; '' without one",
          (LlamaUtils.map_server_system_config(dict(base, log_file='/logs/kb.log'))['log_file'],
           LlamaUtils.map_server_system_config(base)['log_file']), ('/logs/kb.log', ''))

    # role-play has its own loader; its required fields, with made-up values
    role_play = {'host': 'h', 'port': 1, 'base_model_dir': '/m', 'base_embedding_dir': '/e', 'model': 'x.gguf',
                 'embedding_model': 'e.gguf', 'base_convo_dir': '/c', 'system_prompt_dir': '/p', 'gpu_layers': 1,
                 'embedding_gpu_layers': 1, 'max_context_tokens': 1, 'embedding_max_context_tokens': 1,
                 'max_response_tokens': 1, 'repeat_penalty': 1.1, 'max_vector_database_pcnt': 0.2,
                 'buffer_context_pcnt': 0.05, 'top_k': 1, 'min_vector_db_score': 0.5}
    saved_argv = sys.argv
    try:
        sys.argv = ['server', '--json', write('rp.json', dict(role_play, log_file='/logs/rp.log'))]
        with_file = LlamaUtils.get_args_dict_role_play_server('h', 1, lambda message: None)
        sys.argv = ['server', '--json', write('rp-plain.json', role_play)]
        without = LlamaUtils.get_args_dict_role_play_server('h', 1, lambda message: None)
    finally:
        sys.argv = saved_argv
    check("role-play server: log_file from its config; '' without one", (with_file.get('log_file'), without.get('log_file')), ('/logs/rp.log', ''))

print(f"\n{len(failures)} failure(s)")
sys.exit(len(failures))

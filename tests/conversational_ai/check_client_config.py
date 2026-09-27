#!/usr/bin/env python3
"""CS-22: the conversational AI client's config loader (get_args_dict_streaming_client).

Why
---
1. The loader fell back to PORT (65400) for vad_frame_duration, vad_aggressiveness, silence_duration, pipeline and
   voice when a JSON config left them out - a 65400 ms VAD frame, and a pipeline silently reset.
2. 'response_timeout_seconds' in a JSON config was silently ignored (load_json_config never read it).
3. The config now lists agents (agent_defaults + agents); the old single-agent form must keep working.

What it proves (the real client script, loaded from scripts/ - nothing is recorded or played)
------------------------------------------------------------------------------------------
1. A minimal JSON config (host + port only) gets the class defaults, not PORT.
2. response_timeout_seconds, conversation_window_seconds and wake_word_max_position are read from JSON.
3. agent_defaults + agents build the agent list; an old-style config becomes one always-listening agent.
4. A config with a bad agent list is reported and the command-line defaults are used instead (the loader's
   long-standing fallback), rather than the client crashing.
5. Command-line arguments become one always-listening agent.

Usage:  python check_client_config.py      (a Python with sounddevice, pygame, webrtcvad and numpy - the media env)
"""
import importlib.util
import json
import logging
import os
import sys
import tempfile

# testlib finds the library under test (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testlib import settings  # noqa: E402
settings.use_src()

CLIENT_SCRIPT = os.path.join(os.path.dirname(settings.SRC), "scripts", "ai", "combos", "conversational_ai", "conversational-ai-client.py")

logging.disable(logging.WARNING)    # the loader warns on the bad config on purpose
spec = importlib.util.spec_from_file_location("conversational_ai_client", CLIENT_SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Client = module.ConversationalAiPipelineClient

failures = []


def check(label, got, expected):
    """Records one comparison and prints PASS / FAIL."""
    if got == expected:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}: got {got!r}, expected {expected!r}")
        failures.append(label)


def load(config=None, argv=()):
    """Runs the loader on a JSON config (written to a temp file) and/or command-line arguments."""
    args = list(argv)
    path = None
    if config is not None:
        fd, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump(config, f)
        args += ['--json', path]
    sys.argv = ['conversational-ai-client.py'] + args
    try:
        return Client.get_args_dict_streaming_client()
    finally:
        if path:
            os.remove(path)


# 1. the PORT fallback bug
a = load({'host': 'h', 'port': 1})
check("missing VAD / pipeline fields get their real defaults",
      (a['vad_frame_duration'], a['vad_aggressiveness'], a['silence_duration'], a['pipeline']),
      (Client.VAD_FRAME_DURATION_MS, Client.VAD_AGGRESSIVENESS, Client.SILENCE_DURATION_TO_END_BUFFER_MS, Client.PIPELINE))
check("missing voice gets the default voice, not PORT", a['agents'][0]['voice'], Client.VOICE)

# 2. newly read keys
a = load({'host': 'h', 'port': 1, 'response_timeout_seconds': 42.5, 'conversation_window_seconds': 10, 'wake_word_max_position': 3})
check("timeouts and wake-word position come from JSON",
      (a['response_timeout_seconds'], a['conversation_window_seconds'], a['wake_word_max_position']), (42.5, 10, 3))
a = load({'host': 'h', 'port': 1})
check("window defaults to 25 s, position to 10", (a['conversation_window_seconds'], a['wake_word_max_position']), (25, 10))
check("handoff notes default to 3 turns / 600 chars", (a['handoff_max_turns'], a['handoff_max_chars']), (3, 600))
a = load({'host': 'h', 'port': 1, 'handoff_max_turns': 0, 'handoff_max_chars': 100})
check("handoff settings come from JSON", (a['handoff_max_turns'], a['handoff_max_chars']), (0, 100))
check("llm_routing is on unless the JSON turns it off", (load({'host': 'h', 'port': 1})['llm_routing'],
      load({'host': 'h', 'port': 1, 'llm_routing': False})['llm_routing']), (True, False))

# 3. agents
a = load({'host': 'h', 'port': 1,
          'agent_defaults': {'voice': 'heart', 'continuous_save': True, 'load_previous': True},
          'agents': [{'name': 'rose', 'wake_words': ['rose'], 'system_prompt_id': 'assistant-rose'},
                     {'name': 'crane', 'wake_words': ['dr crane'], 'system_prompt_id': 'assistant-frasier',
                      'voice': 'frasier', 'load_previous': False}]})
check("agents built from agent_defaults + agents",
      [(x['name'], x['voice'], x['load_previous']) for x in a['agents']], [('rose', 'heart', True), ('crane', 'frasier', False)])
a = load({'host': 'h', 'port': 1, 'voice': 'bella', 'system_prompt_id': 'solo'})
check("old-style config: one always-listening agent",
      [(x['name'], x['voice'], x['system_prompt_id'], x['wake_words']) for x in a['agents']], [('default', 'bella', 'solo', [])])

# 4. a bad agent list falls back to the command line
a = load({'host': 'json-host', 'port': 1, 'agents': [{'name': 'a'}, {'name': 'a'}]}, argv=['--host', 'cli-host'])
check("duplicate agent names: JSON rejected, command line used", (a['host'], a['agents'][0]['name']), ('cli-host', 'default'))

# 5. command line
a = load(argv=['--voice', 'bella', '--system-prompt-id', 'solo'])
check("command line: one always-listening agent",
      [(x['name'], x['voice'], x['system_prompt_id'], x['wake_words']) for x in a['agents']], [('default', 'bella', 'solo', [])])

print(f"\n{len(failures)} failure(s)")
sys.exit(len(failures))

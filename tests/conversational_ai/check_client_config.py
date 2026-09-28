#!/usr/bin/env python3
"""CS-22 / CS-23: the conversational AI client's config loader (get_args_dict_streaming_client).

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
6. Voice recognition (CS-23): voice_recognition (default off) and location_id (default none) come from JSON or the
   command line; so do save_known_field_clips / save_unknown_field_clips (default off: a client opts its microphone
   in); allow_unknown_speakers is per agent (default true) and can be set in agent_defaults;
   allowed_speakers is per agent (default empty: anyone).

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

# 6. voice recognition
a = load({'host': 'h', 'port': 1})
check("voice recognition is off and there is no location unless configured", (a['voice_recognition'], a['location_id']), (False, ''))
a = load({'host': 'h', 'port': 1, 'voice_recognition': True, 'location_id': 'kitchen',
          'agent_defaults': {'allow_unknown_speakers': False},
          'agents': [{'name': 'rose'}, {'name': 'crane', 'allow_unknown_speakers': True}]})
check("voice_recognition and location_id come from JSON", (a['voice_recognition'], a['location_id']), (True, 'kitchen'))
check("allow_unknown_speakers per agent, from agent_defaults or the agent",
      [(x['name'], x['allow_unknown_speakers']) for x in a['agents']], [('rose', False), ('crane', True)])
check("allow_unknown_speakers defaults to true", load({'host': 'h', 'port': 1})['agents'][0]['allow_unknown_speakers'], True)
a = load({'host': 'h', 'port': 1, 'voice_recognition': True,
          'agents': [{'name': 'rose'}, {'name': 'crane', 'allowed_speakers': ['Sam']}]})
check("allowed_speakers per agent, defaulting to empty", [(x['name'], x['allowed_speakers']) for x in a['agents']],
      [('rose', []), ('crane', ['Sam'])])
a = load({'host': 'json-host', 'port': 1, 'voice_recognition': 'yes'}, argv=['--host', 'cli-host'])
check("a non-boolean voice_recognition rejects the JSON", a['host'], 'cli-host')
a = load(argv=['--voice-recognition', '--location-id', 'office'])
check("command line: --voice-recognition and --location-id", (a['voice_recognition'], a['location_id']), (True, 'office'))

# field clips: opt-in per client
a = load({'host': 'h', 'port': 1})
check("field clips are off unless the client opts in", (a['save_known_field_clips'], a['save_unknown_field_clips']), (False, False))
a = load({'host': 'h', 'port': 1, 'voice_recognition': True, 'save_unknown_field_clips': True})
check("save_*_field_clips come from JSON, each on its own", (a['save_known_field_clips'], a['save_unknown_field_clips']), (False, True))
a = load({'host': 'json-host', 'port': 1, 'save_known_field_clips': 'yes'}, argv=['--host', 'cli-host'])
check("a non-boolean save_known_field_clips rejects the JSON", (a['host'], a['save_known_field_clips']), ('cli-host', False))
a = load(argv=['--save-known-field-clips'])
check("command line: --save-known-field-clips", (a['save_known_field_clips'], a['save_unknown_field_clips']), (True, False))

print(f"\n{len(failures)} failure(s)")
sys.exit(len(failures))

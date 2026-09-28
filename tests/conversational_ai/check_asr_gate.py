#!/usr/bin/env python3
"""CS-22 / CS-23: the conversational AI server's wake-word gate, voice recognition, per-agent LLM sessions and cleanup.

Why
---
The server transcribes every chunk of speech, then decides - without transcribing again - whether it was meant for
an agent. Only speech meant for an agent may reach the LLM, and by then the request must carry exactly one agent.
Separately, the LLM server fixes the system prompt when a session is created, so each agent needs its own LLM
session; and the server used to leak its ASR / LLM connections when a client left.

What it proves (ConversationalAiServer, with a stub SessionWorker and stub clients - no sockets, no models)
----------------------------------------------------------------------------------------------------------
1. A request's agent list, continuation flag and active agent land in the backpack; an old-style request (no
   'agents') becomes one always-listening agent.
2. The ASR stage: no wake word -> 'not_addressed' reply, nothing sent to the LLM, worker shut down. A wake word ->
   an llm-send job for that one agent (with its prompt, save/load flags), and the backpack cut down to one agent.
   A continuation goes to the active agent; another agent's wake word overrides it. Blank speech is not addressed.
   Every request is tagged with its speaker ('[Kevin, to Frasier]: ...'), after any handoff note.
3. The TTS stage's reply to the client carries agent_name, system_prompt_id and speaker.
4. LLM sessions are shared per (user_id, system_prompt_id) - the pair the LLM server saves the history under - so
   two clients (Pis) on the same user_id talk to ONE session and cannot overwrite each other's saved history.
   Different agents (prompts) or user_ids get their own; a session is named '<creating session>-<agent>'; a client
   with different settings joins with a warning; a failed connection is not kept.
5. end_session() closes the session's ASR client, and each LLM session only once its last client has left.
6. Voice recognition (CS-23): the request's voice_recognition / location_id reach the ASR server; the ASR's speaker
   overrides the request's everywhere (the tag, handoff notes, the reply); an unrecognized voice is refused by an
   agent with allow_unknown_speakers false - unless it is continuing a conversation with that SAME agent; too little
   speech keeps the last speaker mid-conversation; an ASR server that cannot judge fails closed (unrecognized);
   '@@NAME@@' gets the household name, not player_name; with recognition off, the ASR's speaker is ignored.
   The client's field-clip opt-in (save_known / save_unknown_field_clips, a real true only) reaches the ASR server
   with recognition on.
   An agent with allowed_speakers answers only those people (ignoring case): anyone else is refused as
   speaker_not_allowed and an unrecognized voice as unknown_speaker - strictly, even continuing a conversation with
   that same agent; with recognition off the list does nothing; a malformed list refuses everyone.

Usage:  python check_asr_gate.py      (any Python 3 with the repo's src on the path)
"""
import os
import sys
import threading

# testlib finds the library under test (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testlib import settings  # noqa: E402
settings.use_src()

import amadeo_utils.ai.combined.conversational_ai.conversational_ai as convo  # noqa: E402
from amadeo_utils.ai.combined.conversational_ai.conversational_ai import ConversationalAiServer  # noqa: E402

failures = []


def check(label, got, expected):
    """Records one comparison and prints PASS / FAIL."""
    if got == expected:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}: got {got!r}, expected {expected!r}")
        failures.append(label)


class StubWorker:
    """Stands in for SessionWorker: a backpack, and a record of queued jobs, client replies and shutdowns."""

    def __init__(self, pipeline='basic_conversational'):
        self.backpack = {}
        self.jobs = []
        self.sent = []
        self.shut_down = False
        self.pipeline = pipeline

    def save_in_backpack(self, key, item):
        self.backpack[key] = item

    def get_from_backpack(self, key):
        return self.backpack.get(key)

    def add_work(self, job):
        self.jobs.append(job)

    def send_to_client(self, response_json, raw_data=None):
        self.sent.append((response_json, raw_data))

    def shutdown(self):
        self.shut_down = True

    def get_pipeline(self):
        return self.pipeline


def bare_server():
    """A ConversationalAiServer without its socket server or real connections."""
    server = ConversationalAiServer.__new__(ConversationalAiServer)
    server.session_to_asr_client_map = {}
    server.session_to_asr_client_lock = threading.Lock()
    server.llm_sessions = {}
    server.llm_sessions_lock = threading.Lock()
    server.llm_host, server.llm_port, server.llm_response_timeout = 'localhost', 1, 1
    server.asr_host, server.asr_port = 'localhost', 1
    server.household_name = 'everyone here'
    return server


AGENTS = [
    {'name': 'rose', 'wake_words': ['rose', 'hey rose'], 'system_prompt_id': 'assistant-rose', 'voice': 'heart',
     'continuous_save': True, 'load_previous': True},
    {'name': 'crane', 'display_name': 'Frasier', 'wake_words': ['dr crane', 'frasier'], 'system_prompt_id': 'assistant-frasier',
     'voice': 'frasier', 'continuous_save': True, 'load_previous': False},
]


def intake(request):
    """Runs handle_client_request with a stub worker; returns the worker."""
    server = bare_server()
    worker = StubWorker()
    server._create_worker = lambda request_id, pipeline, client_info: worker
    request = dict(request, pipeline='basic_conversational', sessionID='S1', _client_info={})
    server.handle_client_request(request, b'audio')
    return worker


def heard(transcript, continuation=False, active_agent='', agents=AGENTS, recent_turns=None, llm_routing=True,
          voice_recognition=None, asr_speaker=None):
    """
    Runs a request through intake and then the ASR stage with the given transcript; returns the worker.
    voice_recognition (if not None) goes in the request; asr_speaker (a dict of speaker fields) goes in the ASR reply.
    """
    request = {'agents': agents, 'continuation': continuation, 'active_agent': active_agent,
               'wake_word_max_position': 10, 'user_id': 'u1', 'player_name': 'Kevin',
               'recent_turns': recent_turns or [], 'llm_routing': llm_routing}
    if voice_recognition is not None:
        request.update(voice_recognition=voice_recognition, location_id='den')
    worker = intake(request)
    worker.jobs.clear()     # the asr-send job from intake
    response = {'success': True, 'type': 'transcription', 'transcription': transcript, **(asr_speaker or {})}
    ConversationalAiServer.handle_asr_worker_drone(bare_server(), worker,
                                                   {'sessionID': 'S1', 'requestID': 'R1', 'response': response})
    return worker


# 1. intake
w = intake({'agents': AGENTS, 'continuation': True, 'active_agent': 'rose', 'user_id': 'u1'})
check("intake keeps the agent list", [a['name'] for a in w.backpack['agents']], ['rose', 'crane'])
check("intake keeps continuation / active agent", (w.backpack['continuation'], w.backpack['active_agent']), (True, 'rose'))
check("intake queues the ASR job", [j['command'] for j in w.jobs], ['asr-send'])
w = intake({'voice': 'bella', 'system_prompt_id': 'solo', 'load_previous': False})
check("old-style request becomes one always-listening agent",
      [(a['name'], a['wake_words'], a['system_prompt_id'], a['voice'], a['load_previous']) for a in w.backpack['agents']],
      [('default', [], 'solo', 'bella', False)])

# 2. ASR stage
w = heard("What time is it?")
check("no wake word: reply is not_addressed", (w.sent[0][0]['success'], w.sent[0][0]['type']), (False, 'not_addressed'))
check("no wake word: nothing queued for the LLM, worker shut down", (w.jobs, w.shut_down), ([], True))

w = heard("Dr. Crane, how are you?")
job = w.jobs[0]
check("wake word: one llm-send job for that agent",
      (len(w.jobs), job['command'], job['agent_name'], job['system_prompt_id'], job['load_previous'], job['user_request']),
      (1, 'llm-send', 'crane', 'assistant-frasier', False, "[Kevin, to Frasier]: Dr. Crane, how are you?"))
check("wake word: backpack cut down to one agent", [a['name'] for a in w.backpack['agents']], ['crane'])
check("wake word: that agent's voice is what TTS will use", w.backpack['voice'], 'frasier')
check("wake word: nothing sent to the client yet", w.sent, [])

check("continuation goes to the active agent", heard("And tomorrow?", True, 'rose').jobs[0]['agent_name'], 'rose')
check("another agent's wake word overrides the continuation", heard("Frasier, your view?", True, 'rose').jobs[0]['agent_name'], 'crane')
check("blank speech in a conversation is not addressed", heard("", True, 'rose').sent[0][0]['type'], 'not_addressed')
w = heard("", agents=[dict(AGENTS[0], name='solo', wake_words=[])])
check("blank speech to an always-listening agent keeps the old reply", w.jobs[0]['user_request'], "I didn't quite get that.")

# 2b. handoff notes and speaker tags
ROSE_TURN = {'agent': 'rose', 'user': 'Should I repaint the deck?', 'reply': 'Yes, before the first frost.'}
w = heard("Hey Frasier, what do you think about that?", True, 'rose', recent_turns=[ROSE_TURN])
check("switching agents: the LLM gets a handoff note, then who said what to whom",
      w.jobs[0]['user_request'],
      '(Since you last spoke: Kevin said to Rose: "Should I repaint the deck?" Rose replied: "Yes, before the first frost.")\n'
      '[Kevin, to Frasier]: Hey Frasier, what do you think about that?')
check("switching agents: the backpack (and so the client) keeps only what was said",
      w.backpack['transcription'], 'Hey Frasier, what do you think about that?')
check("same agent continuing: no note, just the tag",
      heard("And the fence?", True, 'rose', recent_turns=[ROSE_TURN]).jobs[0]['user_request'], '[Kevin, to Rose]: And the fence?')
check("request without recent_turns: no note", heard("Rose, hello").jobs[0]['user_request'], '[Kevin, to Rose]: Rose, hello')
w = heard("hello", agents=[dict(AGENTS[0], name='default', wake_words=[])])
check("one agent: the tag names only the speaker", w.jobs[0]['user_request'], '[Kevin]: hello')
check("'##' in what was heard can't hide the words", heard("Rose, a ## b").jobs[0]['user_request'], '[Kevin, to Rose]: Rose, a   b')
w = intake({'agents': AGENTS, 'player_name': 'Kevin', 'speaker': 'Sam'})
check("an explicit speaker wins over player_name", w.backpack['speaker'], 'Sam')
check("no speaker: player_name stands in", intake({'agents': AGENTS, 'player_name': 'Kevin'}).backpack['speaker'], 'Kevin')
check("old client with no names: no speaker", intake({'agents': AGENTS}).backpack['speaker'], '')

# 2c. several agents in play: rules first, then the LLM (a stub here), then the first agent named
class StubRouter:
    """Stands in for AmadeoClient on the routing call: records each 'one_shot' request and gives a set reply."""
    reply = None        # the (response, raw) the next call returns
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    def send_transient_request(self, command, message='', binary_data=None, **kwargs):
        StubRouter.calls.append(dict(kwargs, command=command))
        return StubRouter.reply


def routed(transcript, reply, continuation=False, active_agent='', llm_routing=True):
    """Runs the ASR stage with the LLM stubbed to give `reply`; returns (agent chosen, LLM calls made)."""
    StubRouter.calls, StubRouter.reply = [], reply
    saved, convo.AmadeoClient = convo.AmadeoClient, StubRouter
    try:
        worker = heard(transcript, continuation, active_agent, llm_routing=llm_routing)
    finally:
        convo.AmadeoClient = saved
    return worker.jobs[0]['agent_name'], StubRouter.calls

ok = lambda name: ({'success': True, 'response': name}, None)
agent, calls = routed("Rose and Frasier, what do you both think?", ok("Frasier"))
check("ambiguous: the LLM's choice answers", agent, 'crane')
check("ambiguous: exactly one stateless one_shot call", [c['command'] for c in calls], ['one_shot'])
check("the routing prompt lists the candidates", 'Rose, Frasier' in calls[0]['system_prompt'], True)
check("the routing question quotes what was said", 'Rose and Frasier, what do you both think?' in calls[0]['user_request'], True)
agent, calls = routed("Rose, what did Frasier mean?", ok("Frasier"))
check("two named, one clearly spoken to: no LLM call", (agent, len(calls)), ('rose', 0))
agent, calls = routed("Frasier said something odd, what do you think?", ok("Rose"), True, 'rose')
check("mid-conversation mention: the LLM can keep the active agent", agent, 'rose')
check("mid-conversation: the prompt says who spoke last", 'Rose spoke last' in calls[0]['system_prompt'], True)
check("unusable answer: the first agent named", routed("Rose and Frasier, hi.", ok("Rose or Frasier"))[0], 'rose')
check("LLM unreachable: the first agent named", routed("Rose and Frasier, hi.", (None, None))[0], 'rose')
check("LLM refuses: the first agent named", routed("Rose and Frasier, hi.", ({'success': False, 'message': 'x'}, None))[0], 'rose')
agent, calls = routed("Rose and Frasier, hi.", ok("Frasier"), llm_routing=False)
check("llm_routing off: first agent named, no call", (agent, len(calls)), ('rose', 0))

# 3. TTS reply carries the agent
w = heard("Rose, hello")
w.backpack['llm_response'] = 'Hello!'
ConversationalAiServer.handle_tts_worker_drone(bare_server(), w, {'sessionID': 'S1', 'requestID': 'R1', 'response': {}, 'byte_data': b'wav'})
reply = w.sent[0][0]
check("TTS reply names the agent, its prompt and the speaker",
      (reply['agent_name'], reply['system_prompt_id'], reply['speaker'], reply['success']), ('rose', 'assistant-rose', 'Kevin', True))


# 6. voice recognition
UNRECOGNIZED = 'Unrecognized voice'
GATED = [AGENTS[0], dict(AGENTS[1], allow_unknown_speakers=False)]    # Frasier only answers known voices


def voice(name, status='identified'):
    """ASR speaker fields: who the ASR server recognized."""
    return {'speaker': name, 'speaker_status': status, 'speaker_score': 0.8, 'speaker_step': 'location'}


UNKNOWN = voice(UNRECOGNIZED, 'unknown')

w = intake({'agents': AGENTS, 'voice_recognition': True, 'location_id': 'den'})
check("intake keeps voice_recognition and location_id", (w.backpack['voice_recognition'], w.backpack['location_id']), (True, 'den'))
w = intake({'agents': AGENTS, 'voice_recognition': 'yes', 'location_id': 7})
check("only a real true turns recognition on; a non-string location is dropped",
      (w.backpack['voice_recognition'], w.backpack['location_id']), (False, ''))
w = intake({'agents': AGENTS, 'voice_recognition': True, 'save_known_field_clips': True, 'save_unknown_field_clips': 'yes'})
check("intake keeps the client's field-clip opt-in, only for a real true",
      (w.backpack['save_known_field_clips'], w.backpack['save_unknown_field_clips']), (True, False))
check("a client that says nothing opts out of field clips",
      (intake({'agents': AGENTS}).backpack['save_known_field_clips'], intake({'agents': AGENTS}).backpack['save_unknown_field_clips']), (False, False))


class StubAsrClient:
    """Stands in for AmadeoClient on the ASR connection: records what each transcribe request carried."""
    sent = []

    def __init__(self, *args, **kwargs):
        pass

    def establish_persistent_connection(self):
        return True

    def update_request_id(self, request_id):
        pass

    def send_persistent_request(self, command, message='', binary_data=None, **kwargs):
        StubAsrClient.sent.append(dict(kwargs, command=command))
        return {}, None


def asr_request(request):
    """Runs intake and the ASR send for a request; returns the fields the ASR server was sent."""
    StubAsrClient.sent = []
    saved, convo.AmadeoClient = convo.AmadeoClient, StubAsrClient
    try:
        worker = intake(request)
        ConversationalAiServer._handle_asr_interaction(bare_server(), worker, worker.jobs[0])
    finally:
        convo.AmadeoClient = saved
    return StubAsrClient.sent[0]

sent = asr_request({'agents': AGENTS, 'voice_recognition': True, 'location_id': 'den'})
check("recognition on: the ASR server is asked for the speaker, with the location",
      (sent['command'], sent.get('voice_recognition'), sent.get('location_id')), ('transcribe', True, 'den'))
sent = asr_request({'agents': AGENTS})
check("recognition off: the ASR request carries no speaker fields", ('voice_recognition' in sent, 'location_id' in sent), (False, False))
sent = asr_request({'agents': AGENTS, 'voice_recognition': True, 'location_id': 'den', 'save_unknown_field_clips': True})
check("recognition on: the client's field-clip opt-in reaches the ASR server",
      (sent.get('save_known_field_clips'), sent.get('save_unknown_field_clips')), (False, True))
sent = asr_request({'agents': AGENTS, 'save_known_field_clips': True, 'save_unknown_field_clips': True})
check("recognition off: no field-clip opt-in is sent (nothing is judged, nothing saved)",
      ('save_known_field_clips' in sent, 'save_unknown_field_clips' in sent), (False, False))

w = heard("Rose, hello", voice_recognition=True, asr_speaker=voice('Sam'))
check("the recognized speaker replaces player_name in the tag", w.jobs[0]['user_request'], '[Sam, to Rose]: Rose, hello')
check("...and in the backpack the reply is built from", (w.backpack['speaker'], w.backpack['speaker_source']), ('Sam', 'voice'))
w = heard("Hey Frasier, what do you think about that?", True, 'rose', recent_turns=[ROSE_TURN],
          voice_recognition=True, asr_speaker=voice('Sam'))
check("...and in the handoff note, for turns that name no speaker",
      w.jobs[0]['user_request'].splitlines()[0].startswith('(Since you last spoke: Sam said to Rose:'), True)
check("recognition on: '@@NAME@@' gets the household name, not player_name", w.jobs[0]['player_name'], 'everyone here')
check("recognition off: '@@NAME@@' gets player_name as before", heard("Rose, hi").jobs[0]['player_name'], 'Kevin')
w = heard("Rose, hello", voice_recognition=False, asr_speaker=voice('Sam'))
check("recognition off: an ASR speaker is ignored", w.jobs[0]['user_request'], '[Kevin, to Rose]: Rose, hello')

w = heard("Rose, hello", voice_recognition=True, asr_speaker=UNKNOWN)
check("an unknown voice is tagged as such", w.jobs[0]['user_request'], f'[{UNRECOGNIZED}, to Rose]: Rose, hello')

w = heard("Frasier, tell me a secret", agents=GATED, voice_recognition=True, asr_speaker=UNKNOWN)
check("gated agent, unknown voice, new conversation: refused with unknown_speaker",
      (w.sent[0][0]['success'], w.sent[0][0]['type'], w.sent[0][0]['agent_name']), (False, 'unknown_speaker', 'crane'))
check("...nothing sent to the LLM, worker shut down", (w.jobs, w.shut_down), ([], True))
w = heard("And another?", True, 'crane', agents=GATED, voice_recognition=True, asr_speaker=UNKNOWN)
check("gated agent, unknown voice, continuing with that same agent: goes through, tagged unrecognized",
      (w.jobs[0]['agent_name'], w.jobs[0]['user_request']), ('crane', f'[{UNRECOGNIZED}, to Frasier]: And another?'))
w = heard("Frasier, what about you?", True, 'rose', agents=GATED, voice_recognition=True, asr_speaker=UNKNOWN)
check("gated agent, unknown voice, continuing with a DIFFERENT agent: refused",
      (w.sent[0][0]['type'], w.jobs), ('unknown_speaker', []))
w = heard("Frasier, tell me a secret", agents=GATED, voice_recognition=True, asr_speaker=voice('Sam'))
check("gated agent, known voice: answers", w.jobs[0]['agent_name'], 'crane')
check("ungated agent, unknown voice: answers", heard("Rose, hi", agents=GATED, voice_recognition=True, asr_speaker=UNKNOWN).jobs[0]['agent_name'], 'rose')
w = heard("Frasier, tell me a secret", agents=GATED, voice_recognition=False)
check("recognition off: a gated agent answers as before", w.jobs[0]['agent_name'], 'crane')

SAM_TURN = dict(ROSE_TURN, speaker='Sam')
w = heard("Yes.", True, 'rose', recent_turns=[SAM_TURN], voice_recognition=True, asr_speaker=voice('', 'too_short'))
check("too short, mid-conversation: whoever spoke last", (w.jobs[0]['user_request'], w.backpack['speaker_source']),
      ('[Sam, to Rose]: Yes.', 'last_speaker'))
w = heard("Rose?", voice_recognition=True, asr_speaker=voice('', 'too_short'))
check("too short, new conversation: an unrecognized voice", w.backpack['speaker'], UNRECOGNIZED)
w = heard("Frasier?", agents=GATED, voice_recognition=True, asr_speaker=voice('', 'too_short'))
check("too short can't open a gated agent", w.sent[0][0]['type'], 'unknown_speaker')
w = heard("Rose, hello", voice_recognition=True, asr_speaker=voice('', 'disabled'))
check("ASR server can't recognize voices: fails closed (unrecognized)", (w.backpack['speaker'], w.backpack['speaker_source']),
      (UNRECOGNIZED, 'fallback'))
w = heard("Rose, hello", voice_recognition=True)
check("older ASR server that reports no speaker: unrecognized too", w.backpack['speaker'], UNRECOGNIZED)

w = heard("Rose, hello", voice_recognition=True, asr_speaker=voice('Sam'))
w.backpack['llm_response'] = 'Hello, Sam!'
ConversationalAiServer.handle_tts_worker_drone(bare_server(), w, {'sessionID': 'S1', 'requestID': 'R1', 'response': {}, 'byte_data': b'wav'})
check("the reply to the client carries the recognized speaker and how it was decided",
      (w.sent[0][0]['speaker'], w.sent[0][0]['speaker_source']), ('Sam', 'voice'))

# allowed_speakers: strict - no continuation exception
LISTED = [AGENTS[0], dict(AGENTS[1], allowed_speakers=['Sam', 'kim'])]    # Frasier only answers Sam and Kim
w = heard("Frasier, tell me a secret", agents=LISTED, voice_recognition=True, asr_speaker=voice('Sam'))
check("allowed_speakers: a listed speaker is answered", w.jobs[0]['agent_name'], 'crane')
w = heard("Frasier, tell me a secret", agents=LISTED, voice_recognition=True, asr_speaker=voice('Kim'))
check("allowed_speakers: names match ignoring case", w.jobs[0]['agent_name'], 'crane')
w = heard("Frasier, tell me a secret", agents=LISTED, voice_recognition=True, asr_speaker=voice('Kevin'))
check("allowed_speakers: an unlisted speaker is refused with speaker_not_allowed",
      (w.sent[0][0]['success'], w.sent[0][0]['type'], w.sent[0][0]['agent_name'], w.sent[0][0]['speaker'], w.jobs, w.shut_down),
      (False, 'speaker_not_allowed', 'crane', 'Kevin', [], True))
w = heard("And another?", True, 'crane', agents=LISTED, voice_recognition=True, asr_speaker=voice('Kevin'))
check("allowed_speakers: an unlisted speaker is refused even continuing with that same agent",
      (w.sent[0][0]['type'], w.jobs), ('speaker_not_allowed', []))
w = heard("And another?", True, 'crane', agents=LISTED, voice_recognition=True, asr_speaker=UNKNOWN)
check("allowed_speakers: an unrecognized voice is refused (as unknown_speaker) even continuing with that same agent",
      (w.sent[0][0]['type'], w.jobs), ('unknown_speaker', []))
w = heard("Rose, hi", agents=LISTED, voice_recognition=True, asr_speaker=voice('Kevin'))
check("allowed_speakers only restricts its own agent", w.jobs[0]['agent_name'], 'rose')
w = heard("Frasier, tell me a secret", agents=LISTED, voice_recognition=False)
check("allowed_speakers: with recognition off, the list does nothing", w.jobs[0]['agent_name'], 'crane')
w = heard("Frasier, tell me a secret", agents=[AGENTS[0], dict(AGENTS[1], allowed_speakers=[])], voice_recognition=True, asr_speaker=voice('Kevin'))
check("allowed_speakers empty: no restriction", w.jobs[0]['agent_name'], 'crane')
w = heard("Frasier, tell me a secret", agents=[AGENTS[0], dict(AGENTS[1], allowed_speakers='Sam')], voice_recognition=True, asr_speaker=voice('Sam'))
check("allowed_speakers malformed (from the network): refuses everyone", (w.sent[0][0]['type'], w.jobs), ('speaker_not_allowed', []))


# 4. per-agent LLM clients
class StubClient:
    """Stands in for AmadeoClient; records the session id asked for and whether it was closed."""
    made = []

    def __init__(self, *args, session_id='', **kwargs):
        self.session_id, self.closed = session_id, False
        StubClient.made.append(self)

    def establish_persistent_connection(self):
        return True

    def send_persistent_request(self, *args, **kwargs):
        return {}, None

    def close_connection(self):
        self.closed = True


class StubCreate:
    """Records what create_llm_session asked for; StubClient sends every request through it."""
    sent = []


def stub_send(self, command, *args, **kwargs):
    StubCreate.sent.append(dict(kwargs, command=command, session_id=self.session_id))
    return {}, None


StubClient.send_persistent_request = stub_send


class DeadClient(StubClient):
    """An AmadeoClient whose connection fails."""

    def establish_persistent_connection(self):
        return False


real_client = convo.AmadeoClient
convo.AmadeoClient = StubClient
try:
    server = bare_server()
    # Two clients (kitchen = S1, office = S2) on the same user_id, both with Rose and Frasier
    rose_client, rose_lock = server._get_or_create_llm_client('S1', 'R1', 'rose', 'u1', 'Kevin', 'assistant-rose')
    crane_client, _ = server._get_or_create_llm_client('S1', 'R2', 'crane', 'u1', 'Kevin', 'assistant-frasier')
    again, _ = server._get_or_create_llm_client('S1', 'R3', 'rose', 'u1', 'Kevin', 'assistant-rose')
    office_rose, office_lock = server._get_or_create_llm_client('S2', 'R4', 'rose', 'u1', 'Kevin', 'assistant-rose')
    other_user, _ = server._get_or_create_llm_client('S3', 'R5', 'rose', 'u2', 'Kevin', 'assistant-rose')
    check("each agent (prompt) gets its own LLM session", rose_client is not crane_client, True)
    check("an agent's LLM session is reused", again is rose_client, True)
    check("two clients on the same user_id share one session - and one lock", (office_rose is rose_client, office_lock is rose_lock), (True, True))
    check("another user_id gets its own session", other_user is not rose_client, True)
    check("LLM session ids are '<creating session>-<agent>'", [c.session_id for c in StubClient.made], ['S1-rose', 'S1-crane', 'S3-rose'])
    check("create_llm_session is sent once per shared session",
          [(x['session_id'], x['user_id'], x['system_prompt_id']) for x in StubCreate.sent if x['command'] == 'create_llm_session'],
          [('S1-rose', 'u1', 'assistant-rose'), ('S1-crane', 'u1', 'assistant-frasier'), ('S3-rose', 'u2', 'assistant-rose')])
    check("the shared session knows both clients", sorted(server.llm_sessions[('u1', 'assistant-rose')].client_sessions), ['S1', 'S2'])

    logging_warnings = []
    handler = type('H', (convo.logging.Handler,), {'emit': lambda self, record: logging_warnings.append(record.getMessage())})()
    convo.logger.addHandler(handler)
    server._get_or_create_llm_client('S4', 'R6', 'rose', 'u1', 'Sam', 'assistant-rose')
    convo.logger.removeHandler(handler)
    check("a client with other settings joins the shared session, with a warning",
          any('joins the shared LLM session' in m for m in logging_warnings), True)

    convo.AmadeoClient = DeadClient
    dead, _ = server._get_or_create_llm_client('S5', 'R7', 'rose', 'u9', 'Kevin', 'assistant-rose')
    check("a session whose connection failed is not kept (the next request retries)", ('u9', 'assistant-rose') in server.llm_sessions, False)
    convo.AmadeoClient = StubClient

    # 5. cleanup
    asr_s1, asr_s2 = StubClient(), StubClient()
    server.session_to_asr_client_map = {'S1': (asr_s1, threading.Lock()), 'S2': (asr_s2, threading.Lock())}
    server.end_session('S1')
    check("end_session closes the session's ASR client", asr_s1.closed, True)
    check("...and an LLM session only it used", crane_client.closed, True)
    check("...but not one another client still uses", (rose_client.closed, sorted(server.llm_sessions[('u1', 'assistant-rose')].client_sessions)), (False, ['S2', 'S4']))
    check("end_session leaves other sessions' ASR clients alone", asr_s2.closed, False)
    server.end_session('S2')
    server.end_session('S4')
    check("the shared session closes when its last client leaves", (rose_client.closed, ('u1', 'assistant-rose') in server.llm_sessions), (True, False))
    check("other users' sessions stay open", (other_user.closed, list(server.llm_sessions)), (False, [('u2', 'assistant-rose')]))
    server.end_session('S-unknown')     # must not raise
    check("end_session on an unknown session is harmless", True, True)
finally:
    convo.AmadeoClient = real_client

print(f"\n{len(failures)} failure(s)")
sys.exit(len(failures))

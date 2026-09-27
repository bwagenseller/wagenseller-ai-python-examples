#!/usr/bin/env python3
"""CS-22: the conversational AI server's wake-word gate, per-agent LLM sessions and session cleanup.

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
   Every request is tagged with its speaker ('[Alex, to Frasier]: ...'), after any handoff note.
3. The TTS stage's reply to the client carries agent_name, system_prompt_id and speaker.
4. LLM clients are keyed by (session, agent) and ask the LLM server for '<session>-<agent>' ids.
5. end_session() closes the session's ASR client and every one of its LLM clients, and nothing else.

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
    server.session_to_llm_client_map = {}
    server.session_to_llm_client_lock = threading.Lock()
    server.llm_host, server.llm_port, server.llm_response_timeout = 'localhost', 1, 1
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


def heard(transcript, continuation=False, active_agent='', agents=AGENTS, recent_turns=None, llm_routing=True):
    """Runs a request through intake and then the ASR stage with the given transcript; returns the worker."""
    worker = intake({'agents': agents, 'continuation': continuation, 'active_agent': active_agent,
                     'wake_word_max_position': 10, 'user_id': 'u1', 'player_name': 'Alex',
                     'recent_turns': recent_turns or [], 'llm_routing': llm_routing})
    worker.jobs.clear()     # the asr-send job from intake
    response = {'success': True, 'type': 'transcription', 'transcription': transcript}
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
      (1, 'llm-send', 'crane', 'assistant-frasier', False, "[Alex, to Frasier]: Dr. Crane, how are you?"))
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
      '(Since you last spoke: Alex said to Rose: "Should I repaint the deck?" Rose replied: "Yes, before the first frost.")\n'
      '[Alex, to Frasier]: Hey Frasier, what do you think about that?')
check("switching agents: the backpack (and so the client) keeps only what was said",
      w.backpack['transcription'], 'Hey Frasier, what do you think about that?')
check("same agent continuing: no note, just the tag",
      heard("And the fence?", True, 'rose', recent_turns=[ROSE_TURN]).jobs[0]['user_request'], '[Alex, to Rose]: And the fence?')
check("request without recent_turns: no note", heard("Rose, hello").jobs[0]['user_request'], '[Alex, to Rose]: Rose, hello')
w = heard("hello", agents=[dict(AGENTS[0], name='default', wake_words=[])])
check("one agent: the tag names only the speaker", w.jobs[0]['user_request'], '[Alex]: hello')
check("'##' in what was heard can't hide the words", heard("Rose, a ## b").jobs[0]['user_request'], '[Alex, to Rose]: Rose, a   b')
w = intake({'agents': AGENTS, 'player_name': 'Alex', 'speaker': 'Sam'})
check("an explicit speaker wins over player_name", w.backpack['speaker'], 'Sam')
check("no speaker: player_name stands in", intake({'agents': AGENTS, 'player_name': 'Alex'}).backpack['speaker'], 'Alex')
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
      (reply['agent_name'], reply['system_prompt_id'], reply['speaker'], reply['success']), ('rose', 'assistant-rose', 'Alex', True))


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


real_client = convo.AmadeoClient
convo.AmadeoClient = StubClient
try:
    server = bare_server()
    rose_client, _ = server._get_or_create_llm_client('S1', 'R1', 'rose', 'u1', 'Alex', 'assistant-rose')
    crane_client, _ = server._get_or_create_llm_client('S1', 'R2', 'crane', 'u1', 'Alex', 'assistant-frasier')
    again, _ = server._get_or_create_llm_client('S1', 'R3', 'rose', 'u1', 'Alex', 'assistant-rose')
    other, _ = server._get_or_create_llm_client('S2', 'R4', 'rose', 'u1', 'Alex', 'assistant-rose')
    check("each agent gets its own LLM client", rose_client is not crane_client, True)
    check("an agent's LLM client is reused", again is rose_client, True)
    check("LLM session ids are '<session>-<agent>'", [c.session_id for c in StubClient.made], ['S1-rose', 'S1-crane', 'S2-rose'])

    # 5. cleanup
    asr_s1, asr_s2 = StubClient(), StubClient()
    server.session_to_asr_client_map = {'S1': (asr_s1, threading.Lock()), 'S2': (asr_s2, threading.Lock())}
    server.end_session('S1')
    check("end_session closes the session's ASR and LLM clients", (asr_s1.closed, rose_client.closed, crane_client.closed), (True, True, True))
    check("end_session leaves other sessions alone", (asr_s2.closed, other.closed, list(server.session_to_llm_client_map)), (False, False, [('S2', 'rose')]))
    server.end_session('S-unknown')     # must not raise
    check("end_session on an unknown session is harmless", True, True)
finally:
    convo.AmadeoClient = real_client

print(f"\n{len(failures)} failure(s)")
sys.exit(len(failures))

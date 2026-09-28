from amadeo_utils.client.amadeo_client import AmadeoClient
from amadeo_utils.server.amadeo_server import AmadeoServer
import logging
import json
from amadeo_utils.colored_text import ColoredText
from typing import Dict, Any
import threading
import time
import os
import argparse
from amadeo_utils.server.session_worker import SessionWorker
from amadeo_utils.ai.combined.conversational_ai.wake_words import select_agent, WAKE_WORD_MAX_POSITION
from amadeo_utils.ai.combined.conversational_ai.handoff import build_handoff_note, speaker_tag, HANDOFF_MAX_TURNS, HANDOFF_MAX_CHARS, HIDDEN_DELIMITER
from amadeo_utils.ai.combined.conversational_ai.routing import build_routing_prompt, parse_routing_reply, display_name, ROUTING_MAX_TOKENS
from amadeo_utils.ai.combined.conversational_ai.speakers import resolve_speaker, refuses_unknown_speaker, refuses_unlisted_speaker, HOUSEHOLD_NAME
from amadeo_utils.ai.asr.speaker_id import UNRECOGNIZED_SPEAKER

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

VALID_PIPELINES = ['reflection', 'translate_text', 'translate_voice', 'revoice', 'basic_conversational']


class SharedLlmSession:
    """
    One LLM session (a persistent connection to the LLM server, holding one agent's chat history) and the client
    sessions using it. See ConversationalAiServer.llm_sessions.
    """

    def __init__(self, client, lock, settings):
        """
        Args:
            client: the AmadeoClient holding the persistent LLM connection.
            lock: taken for each request, so requests from different clients take turns.
            settings: what the session was created with (player_name, continuous_save, load_previous), to warn a
                later client whose own settings differ.
        """
        self.client = client
        self.lock = lock
        self.settings = settings
        self.client_sessions = set()    # this server's sessionIDs of the clients using it

class ConversationalAiServer:

    HOST = 'localhost'
    PORT = 65400
    ASR_HOST = 'localhost'
    ASR_PORT = 65432
    TTS_HOST = 'localhost'
    TTS_PORT = 8888
    LLM_HOST = 'localhost'
    LLM_PORT = 65440
    # How long to wait for the LLM server's reply to one request. 120 s suits role-play and the knowledge base; the
    # agent server can take longer (web lookups, a login, several tool rounds - up to its max_turn_seconds, 180 by
    # default), so raise it in the config ('llm_response_timeout_seconds') when this server talks to the agent server.
    LLM_RESPONSE_TIMEOUT_SECONDS = 120

    def __init__(self, argsDict: dict):
        self.args_dict = argsDict
        self.session_workers = {}  # session_id -> SessionWorker
        self.workers_lock = threading.Lock()

        self.request_to_session_map = {}  # Simple dictionary that maps requestIDs to sessionIDs
        self.request_to_session_lock = threading.Lock()

        self.session_to_asr_client_map = {}  # Simple dictionary that maps sessionIDs to a tuple (asr clients, asr client locks); asr clients use a persistent connection, so we keep that connection for the whole session
        self.session_to_asr_client_lock = threading.Lock() # use this lock to interact with the asr client as well

        # Maps (user_id, system_prompt_id) to a SharedLlmSession: one persistent LLM connection, and so one LLM session
        # and one chat history, for everyone talking to that agent under that user_id - whichever client (Pi) they
        # are at. Keyed like this because:
        #  * the LLM server fixes the system prompt when a session is created, so each agent (prompt) needs its own
        #    session - one shared connection would keep answering as whichever agent spoke first; and
        #  * the LLM server saves a session's history to <user_id>/<system_prompt_id> by rewriting the whole file from
        #    that session's copy in memory, which it loads only once. Two sessions on the same file would each
        #    overwrite the other's turns (last save wins) and never hear what the other was told. One shared session
        #    means one copy and one writer, and every client hears the whole household's conversation with the agent.
        # Requests to a shared session take turns on its lock.
        self.llm_sessions = {}
        self.llm_sessions_lock = threading.Lock()  # guards self.llm_sessions (each session has its own lock for requests)

        # Register global handlers for all sessions
        SessionWorker.register_global_handler('ping', self._handle_ping)
        SessionWorker.register_global_handler('status', self._handle_status)

        self.asr_host = argsDict['asr_host']
        self.asr_port = argsDict['asr_port']
        self.tts_host = argsDict['tts_host']
        self.tts_port = argsDict['tts_port']
        self.llm_host = argsDict['llm_host']
        self.llm_port = argsDict['llm_port']
        self.llm_response_timeout = argsDict.get('llm_response_timeout_seconds', ConversationalAiServer.LLM_RESPONSE_TIMEOUT_SECONDS)
        # What '@@NAME@@' in a system prompt becomes when voice recognition is on (see speakers.HOUSEHOLD_NAME)
        self.household_name = argsDict.get('household_name', HOUSEHOLD_NAME)

        self.server = AmadeoServer(argsDict['host'], argsDict['port'],
                                 synchronous=False,
                                 additional_client_functionality=self.handle_client_request,
                                 additional_shutdown=self.end_session)


    def handle_client_request(self, request, client_binary_data):
        """
        The entrypoint of the code is here.

        Traditionally, this would accept the request, perform whatever needed to be done, and then return a dictionary and optional data to AmadeoServer - and AmadeoServer would take it from there
        This will no longer work, as we arent simply running it through a model (for example, ASR) and then returning a result - we are sending the request away for processing, THEN sending the results
        of that to ANOTHER service, THEN maybe sending that result to another service, THEN finally returning a result of some sort.

        Because of this, this method will now return 'basic_dictionary_that_will_not_be_used, None' to conform with what is expected, but a worker thread will actually handle the rest.

        This is achieved by setting the response host/port in the dictionary 'self.response_handlers' (with the sessionID as key), and then creating a 'job' and starting the queue process (with self.asr_queue).

        """

        session_id = request.get('sessionID') # comes from AmadeoServer

        # This is a new request - its entirely possible that a new request comes in while the old is being processed. Thus, the request_id is king for the lifespan, although we still need the session_id
        request_id = AmadeoServer.generate_session_id()
        request['requestID'] = request_id

        pipeline = request.get('pipeline')

        logger.info(f"{ColoredText.BLUE_TEXT}Handling client pipeline request '{pipeline}' for sessionID {session_id} - new requestID {request_id} created.{ColoredText.END_TEXT}")

        # Create session worker
        worker = self._create_worker(request_id, pipeline, request['_client_info'])

        # save the sessionID and requestID in the backpack
        worker.save_in_backpack('sessionID', session_id)
        worker.save_in_backpack('requestID', request_id)

        if pipeline == 'reflection' or pipeline == 'revoice':
            # outfit this job to interact with the ASR server

            if pipeline == 'revoice':
                # if this is a revoice, add a few more things
                voice = request.get('voice')
                worker.save_in_backpack('voice', voice)

            job = {
                'command': 'asr-send',
                'pipeline': pipeline,
                'sessionID': session_id,
                'requestID': request_id,
                'byte_data': client_binary_data,
                'request': request
            }
            worker.add_work(job)
        elif pipeline == 'basic_conversational':
            # Settings that belong to the user, whichever agent answers
            worker.save_in_backpack('user_id', request.get('user_id'))
            worker.save_in_backpack('player_name', request.get('player_name', ''))
            # Who is talking this turn, for the speaker tag and handoff notes (see handoff.py). The client sends its
            # player_name; with voice recognition on, the ASR stage replaces it with the voice it recognized (see
            # speakers.py). An older client sends nothing, and player_name stands in.
            worker.save_in_backpack('speaker', request.get('speaker') or request.get('player_name') or '')
            # Voice recognition (CS-23): the ASR server works out the speaker from the audio, comparing the samples
            # enrolled at this client's location first. The client's speaker above is used only when it is off.
            worker.save_in_backpack('voice_recognition', request.get('voice_recognition') is True)
            location_id = request.get('location_id')
            worker.save_in_backpack('location_id', location_id if isinstance(location_id, str) else '')
            # Whether this client lets its microphone be recorded as field clips (known / unrecognized voices); the ASR
            # server's own config must allow it too. Off unless the client says a real true.
            worker.save_in_backpack('save_known_field_clips', request.get('save_known_field_clips') is True)
            worker.save_in_backpack('save_unknown_field_clips', request.get('save_unknown_field_clips') is True)

            # Every agent the client knows about. The ASR stage picks one of them (or none) once it has the
            # transcript, and cuts this list down to that one agent before the LLM stage.
            agents = request.get('agents')
            if not isinstance(agents, list) or not agents:
                # A client from before wake words: one always-listening agent built from the top-level keys
                agents = [{
                    'name': 'default',
                    'wake_words': [],
                    'system_prompt_id': request.get('system_prompt_id', 'default'),
                    'voice': request.get('voice'),
                    'continuous_save': request.get('continuous_save', False),
                    'load_previous': request.get('load_previous', True)
                }]
            worker.save_in_backpack('agents', agents)

            # Is this the next turn of a conversation that is already under way, and with whom?
            worker.save_in_backpack('continuation', bool(request.get('continuation', False)))
            worker.save_in_backpack('active_agent', request.get('active_agent', ''))
            worker.save_in_backpack('wake_word_max_position', request.get('wake_word_max_position', WAKE_WORD_MAX_POSITION))

            # The turns of the conversation so far (oldest first), so an agent can be told what was said to the others
            # since it last spoke (see handoff.py)
            worker.save_in_backpack('recent_turns', request.get('recent_turns') or [])
            worker.save_in_backpack('handoff_max_turns', request.get('handoff_max_turns', HANDOFF_MAX_TURNS))
            worker.save_in_backpack('handoff_max_chars', request.get('handoff_max_chars', HANDOFF_MAX_CHARS))

            # May the LLM be asked which agent was addressed, when several are in play and the rules can't tell?
            worker.save_in_backpack('llm_routing', bool(request.get('llm_routing', True)))

            job = {
                'command': 'asr-send',
                'pipeline': pipeline,
                'sessionID': session_id,
                'requestID': request_id,
                'byte_data': client_binary_data,
                'request': request
            }
            worker.add_work(job)
        elif pipeline == 'terminate':
            self.remove_asr_client(session_id)
        else:
            logger.warning(f"{ColoredText.YELLOW_TEXT}Unknown command from sessionID {session_id}: {pipeline}{ColoredText.END_TEXT}")

        return {
            'success': True,
            'message': 'Pipeline queued',
            'sessionID': session_id
        }, None

    def _create_worker(self, request_id, pipeline, client_info):
        with self.workers_lock:
            if request_id not in self.session_workers:
                session_handlers = {
                    'asr-send': self._handle_asr_interaction,
                    'asr-receive': self.handle_asr_worker_drone,
                    'llm-send': self._handle_llm_interaction,
                    'llm-receive': self.handle_llm_worker_drone,
                    'tts-send': self._handle_tts_interaction,
                    'tts-receive': self.handle_tts_worker_drone
                    # Add other session-specific handlers here
                }

                logger.info(f"{ColoredText.BLUE_TEXT}Creating a worker for requestID {request_id}.{ColoredText.END_TEXT}")
                self.session_workers[request_id] = SessionWorker(
                    session_id=request_id,
                    client_socket=client_info['socket'],
                    address=client_info['address'],
                    parent_server=self,
                    pipeline=pipeline,
                    command_handlers=session_handlers,
                    max_workers=3
                )
            return self.session_workers[request_id]

    def _get_worker(self, request_id):
        with self.workers_lock:
            return self.session_workers[request_id]


    def _get_or_create_llm_client(self, session_id:str, request_id:str, agent_name:str, user_id:str, player_name:str, system_prompt_id:str, continuous_save:bool = False, load_previous:bool = False):
        """
        Finds the LLM session shared by everyone talking to this agent under this user_id, creating it on first use,
        and records that this client session uses it (see self.llm_sessions for why sessions are shared).

        The settings below are sent to the LLM server only when the session is created, and are fixed from then on:
        a later client whose settings differ (e.g. another player_name) joins the existing session, with a warning.

        Args:
            session_id: this server's session ID for the client.
            request_id: the request that needs the LLM.
            agent_name: the agent answering (names the LLM session and appears in the logs).
            user_id, system_prompt_id: which shared session - the pair the LLM server keeps the history under.
            player_name, continuous_save, load_previous: sent to the LLM server when the session is created.

        Returns:
            (llm client, its lock) - hold the lock for each request.
        """
        key = (user_id, system_prompt_id)
        settings = {'player_name': player_name, 'continuous_save': continuous_save, 'load_previous': load_previous}
        with self.llm_sessions_lock:
            shared = self.llm_sessions.get(key)
            if shared is None:
                logger.info(f"{ColoredText.BLUE_TEXT}Creating the shared LLM session for user '{user_id}', prompt '{system_prompt_id}' (agent '{agent_name}', first used by sessionID {session_id}), to LLM host {self.llm_host} and LLM port {self.llm_port}.{ColoredText.END_TEXT}")
                # The LLM server refuses a sessionID that is already in use, so the session is named after the client
                # session that created it and the agent - unique for as long as it lives
                llm_client = AmadeoClient( self.llm_host, self.llm_port, additional_server_response_functionality=self.handle_llm_server_response, session_id = f"{session_id}-{agent_name}", request_id = request_id, persistent_request_timeout=self.llm_response_timeout)

                if not llm_client.establish_persistent_connection():
                    # Not kept: the next request tries again, rather than every client of this user being stuck with
                    # a dead connection
                    logger.error(f"{ColoredText.RED_TEXT}Failed to establish LLM connection for worker with sessionID {session_id}!{ColoredText.END_TEXT}")
                    return llm_client, threading.Lock()

                # Send using the persistent request method
                # the first request to the LLM must establish some parameters
                response, raw_data = llm_client.send_persistent_request(
                    command="create_llm_session",
                    message="Request to LLM",
                    binary_data=None,
                    user_id=user_id,
                    player_name=player_name,
                    system_prompt_id=system_prompt_id,
                    spoken_response=True,
                    continuous_save=continuous_save,
                    load_previous=load_previous
                )

                shared = SharedLlmSession(llm_client, threading.Lock(), settings)
                self.llm_sessions[key] = shared
            elif session_id not in shared.client_sessions and settings != shared.settings:
                logger.warning(f"{ColoredText.YELLOW_TEXT}sessionID {session_id} joins the shared LLM session for user '{user_id}', prompt '{system_prompt_id}', which was created with {shared.settings}; its own {settings} are not used.{ColoredText.END_TEXT}")
            shared.client_sessions.add(session_id)
            return shared.client, shared.lock


    def _get_or_create_asr_client(self, session_id):
        """
        Finds the asr_client by session ID and returns both the asr client and its lock in a tuple
        """
        with self.session_to_asr_client_lock:
            if session_id not in self.session_to_asr_client_map:
                logger.info(f"{ColoredText.BLUE_TEXT}Creating an ASR client for sessionID {session_id} to ASR host {self.asr_host} and ASR port {self.asr_port}.{ColoredText.END_TEXT}")
                asr_client = AmadeoClient( self.asr_host, self.asr_port, additional_server_response_functionality=self.handle_asr_server_response, session_id = session_id)


                asr_client_lock = threading.Lock() # use this lock to interact with the asr client as well

                if not asr_client.establish_persistent_connection():
                    logger.error(f"{ColoredText.RED_TEXT}Failed to establish ASR connection for worker with sessionID {session_id}!{ColoredText.END_TEXT}")

                self.session_to_asr_client_map[session_id] = (asr_client, asr_client_lock)
            asr_client, asr_client_lock = self.session_to_asr_client_map[session_id]
            return asr_client, asr_client_lock


    def remove_session(self, session_id):
        """Called by SessionWorker during cleanup"""
        with self.session_to_asr_client_lock:
            # Yes, we know its resultID and not sessionID here - THE WORKERS HERE ARE DEFINED BY resultID
            logger.info(f"{ColoredText.BLUE_TEXT}Removed resultID {session_id} from workers threads.{ColoredText.END_TEXT}")
            self.session_workers.pop(session_id, None)

    def remove_asr_client(self, session_id):
        """
        Closes and forgets the ASR client for a session.

        Args:
            session_id: the session whose ASR connection is no longer needed.
        """
        with self.session_to_asr_client_lock:
            entry = self.session_to_asr_client_map.pop(session_id, None)
        if entry:
            asr_client, _ = entry       # the map holds (client, lock)
            asr_client.close_connection()
            logger.info(f"{ColoredText.BLUE_TEXT}Shut down and removed ASR client for session {session_id}.{ColoredText.END_TEXT}")

    def remove_llm_clients(self, session_id):
        """
        This client session no longer needs its LLM sessions: forget it on each shared session it used, and close
        the ones no other client is still using.

        Args:
            session_id: the session whose LLM connections are no longer needed.
        """
        with self.llm_sessions_lock:
            closing = []
            for key, shared in list(self.llm_sessions.items()):
                shared.client_sessions.discard(session_id)
                if not shared.client_sessions:
                    closing.append((key, self.llm_sessions.pop(key)))
        for (user_id, system_prompt_id), shared in closing:
            # Wait for a request still under way on it (a reply being generated) before closing the connection
            with shared.lock:
                shared.client.close_connection()
            logger.info(f"{ColoredText.BLUE_TEXT}Shut down and removed the LLM session for user '{user_id}', prompt '{system_prompt_id}' (its last client, session {session_id}, left).{ColoredText.END_TEXT}")

    def _route_with_llm(self, session_id, request_id, transcript, candidates, last_speaker):
        """
        Asks the LLM which of several agents the user is speaking to.

        Uses the LLM server's stateless 'one_shot' command over a transient connection: nothing is stored in any
        agent's history, and it needs no LLM session of its own. It runs in this request's worker thread, so the
        request simply waits for it; it queues behind any generation already on the GPU.

        Args:
            session_id, request_id: for logging.
            transcript: what the user said.
            candidates: the agents in play (see wake_words.select_agent()).
            last_speaker: the agent that answered the previous turn, or None.

        Returns:
            The chosen agent, or None if the LLM could not be reached or its answer did not name exactly one candidate.
        """
        system_prompt, user_request = build_routing_prompt(transcript, candidates, last_speaker)
        started = time.time()
        router = AmadeoClient(self.llm_host, self.llm_port)
        response, _ = router.send_transient_request('one_shot', 'Which agent was addressed?', system_prompt=system_prompt,
                                                    user_request=user_request, max_tokens=ROUTING_MAX_TOKENS)
        elapsed = time.time() - started
        names = [display_name(a) for a in candidates]

        if not response or not response.get('success'):
            message = response.get('message', '') if response else 'no reply'
            logger.warning(f"{ColoredText.YELLOW_TEXT}Worker for sessionID {session_id} amd requestID {request_id}: could not ask the LLM which of {names} was addressed ({message}); the first agent named answers.{ColoredText.END_TEXT}")
            return None

        reply = response.get('response', '')
        chosen = parse_routing_reply(reply, candidates)
        if chosen is None:
            logger.warning(f"{ColoredText.YELLOW_TEXT}Worker for sessionID {session_id} amd requestID {request_id}: the LLM's answer {reply!r} does not name exactly one of {names}; the first agent named answers.{ColoredText.END_TEXT}")
        else:
            logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id}: the LLM chose '{chosen['name']}' from {names} in {elapsed:.2f} s (answer {reply!r}).{ColoredText.END_TEXT}")
        return chosen

    def end_session(self, session_id):
        """
        AmadeoServer's shutdown hook: runs when a client's session ends - by 'terminate_session', or by the client
        disconnecting, vanishing or timing out. Closes the session's ASR and LLM connections, which otherwise stayed
        open for the life of the server. Safe to run twice.

        Args:
            session_id: the session that ended.
        """
        self.remove_asr_client(session_id)
        self.remove_llm_clients(session_id)

    def _handle_tts_interaction(self, worker, job):
        """
        job = {
            'command': 'tts-send',
            'pipeline': pipeline,
            'sessionID': session_id,
            'requestID': request_id,
            'voice': voice,
            'text': transcription
        }
        """
        session_id = job['sessionID']
        request_id = job['requestID']

        tts_client = AmadeoClient(self.tts_host, self.tts_port, additional_server_response_functionality = self.handle_tts_server_response)
        tts_client.send_transient_request('service_tts', '', text=job['text'], voice=job['voice'], requestID=request_id, sessionID = session_id)

        logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id} sent text to TTS.{ColoredText.END_TEXT}")
        # Response is handled automatically by handle_server_response callback

    def _handle_asr_interaction(self, worker, job):
        """
        Handles the asr action via a SessionWorker. Here, the SessionWorker will send the audio to the ASR server for processing

        Here is an example Job object for ASR, for reference
        job = {
            'command': 'asr',
            'sessionID': session_id,
            'requestID': request_id,
            'byte_data': client_binary_data,
            'request': request
        }

        """
        # This runs async in the worker's thread pool
        # Can call worker.save_results() to store intermediate results
        # Can access self.model, self.call_llm_service(), etc.

        session_id = job['sessionID']
        request_id = job['requestID']
        speech_segment_bytes = job['byte_data']

        # save the audio in the backpack
        worker.save_in_backpack('original_audio', speech_segment_bytes)


        asr_client, asr_client_lock = self._get_or_create_asr_client(session_id)
        with asr_client_lock:
            # we re-use self.session_to_asr_client_lock for the asr_client too

            asr_client.update_request_id(request_id)
            # Voice recognition happens in the ASR server, in the same pass as the transcription, so it is asked for
            # here (only the conversational pipeline sets it)
            speaker_fields = {}
            if worker.get_from_backpack('voice_recognition'):
                speaker_fields = {'voice_recognition': True, 'location_id': worker.get_from_backpack('location_id') or '',
                                  'save_known_field_clips': bool(worker.get_from_backpack('save_known_field_clips')),
                                  'save_unknown_field_clips': bool(worker.get_from_backpack('save_unknown_field_clips'))}
            # Send using the new persistent request method with binary audio data
            # this already includes sessionID and requestID
            response, raw_data = asr_client.send_persistent_request(
                command="transcribe",
                message="Audio chunk for transcription",
                binary_data=speech_segment_bytes, # Send as binary data after JSON
                **speaker_fields
            )

        logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id} sent transcription to ASR.{ColoredText.END_TEXT}")
        # Response is handled automatically by handle_server_response callback

    def _handle_llm_interaction(self, worker, job):
        """
        job = {
            'command': 'llm-send',
            'pipeline': 'conversational',
            'sessionID': session_id,
            'requestID': request_id,
            'user_request': transcription,
            'agent_name': agent_name,
            'user_id': user_id,
            'system_prompt_id': system_prompt_id,
            'player_name': player_name
        }
        """
        session_id = job['sessionID']
        request_id = job['requestID']
        llm_client, llm_client_lock = self._get_or_create_llm_client(session_id, request_id, job.get('agent_name', 'default'), job.get('user_id', 'Bob'), job.get('player_name', ''), job.get('system_prompt_id', 'default'), job.get('continuous_save', False), job.get('load_previous', False))
        with llm_client_lock:
            # the shared session's own lock: requests from different clients take turns

            llm_client.update_request_id(request_id)

            # Send using the persistent request method
            response, raw_data = llm_client.send_persistent_request(
                command="request",
                message="Request to LLM",
                binary_data=None,
                user_request=job['user_request']
            )

        logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id} sent transcription to LLM.{ColoredText.END_TEXT}")
        # Response is handled automatically by handle_server_response callback


    def handle_asr_server_response(self, response, raw_data):
        """
        Callback function to handle server responses from the ASR client. We quickly hand off to the worker, as its best to put the processing away from the main class running the server.

        The ASR server will pass back the sessionID, and if we include one, a requestID - and we did. Use this to find the worker and send it to the worker instead.
        """

        if response:
            session_id = response['sessionID']
            request_id = response['requestID']

            worker = self._get_worker(request_id)

            logger.info(f"{ColoredText.BLUE_TEXT}Got ASR server response for sessionID {session_id} amd requestID {request_id} - sending to worker.{ColoredText.END_TEXT}")

            # outfit this job to interact with the ASR server
            job = {
                'command': 'asr-receive',
                'pipeline': worker.get_pipeline,
                'sessionID': session_id,
                'requestID': request_id,
                'response': response
            }
            worker.add_work(job)
        else:
            logger.error(f"{ColoredText.RED_TEXT}ASR Server error: no response.{ColoredText.END_TEXT}")

    def handle_llm_server_response(self, response, raw_data):
        """Callback for LLM responses"""
        if response:
            request_id = response['requestID']

            worker = self._get_worker(request_id)
            # The LLM session is '<sessionID>-<agent>' (one per agent), so take this server's own sessionID from the
            # backpack rather than from the LLM's reply
            session_id = worker.get_from_backpack('sessionID')

            job = {
                'command': 'llm-receive',
                'pipeline': worker.get_pipeline(),
                'sessionID': session_id,
                'requestID': request_id,
                'response': response
            }
            worker.add_work(job)
        else:
            logger.error(f"{ColoredText.RED_TEXT}LLM Server error: no response.{ColoredText.END_TEXT}")

    def handle_tts_server_response(self, response, raw_data):
        """
        Callback function to handle server responses from the ASR client. We quickly hand off to the worker, as its best to put the processing away from the main class running the server.

        The ASR server will pass back the sessionID, and if we include one, a requestID - and we did. Use this to find the worker and send it to the worker instead.
        """

        if response:
            session_id = response['sessionID']
            request_id = response['requestID']

            worker = self._get_worker(request_id)

            logger.info(f"{ColoredText.BLUE_TEXT}Got TTS server response for sessionID {session_id} amd requestID {request_id} - sending to worker.{ColoredText.END_TEXT}")

            # outfit this job to interact with the ASR server
            job = {
                'command': 'tts-receive',
                'pipeline': worker.get_pipeline,
                'sessionID': session_id,
                'requestID': request_id,
                'response': response,
                'byte_data': raw_data
            }
            worker.add_work(job)
        else:
            logger.error(f"{ColoredText.RED_TEXT}TTS Server error: no response.{ColoredText.END_TEXT}")


    def handle_asr_worker_drone(self, worker, job):

        session_id = job['sessionID']
        request_id = job['requestID']
        response = job['response'] # response is, at least, not empty at this point - when it was packed it was verified to exist

        pipeline = worker.get_pipeline()

        logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id} processing ASR request for pipeline {pipeline}.{ColoredText.END_TEXT}")
        heard = ''  # exactly what the ASR heard; blank if it heard nothing or failed (the wake-word check needs to know)
        if response.get("success"):
            if response.get('type') == 'transcription':
                transcription = response.get("transcription")
                if transcription and transcription.strip():
                    # if the transcription is not blank or None, just pass
                    heard = transcription
                else:
                    logger.info(f"{ColoredText.BLUE_TEXT} TEXT IS BLANK for sessionID {session_id} amd requestID {request_id} for pipeline {pipeline}.{ColoredText.END_TEXT}")
                    transcription = "I didn't quite get that."
            elif response.get('type') == 'garbage_transcription':
                logger.info(f"{ColoredText.BLUE_TEXT} Garbage transcription for sessionID {session_id} amd requestID {request_id} for pipeline {pipeline} - ignoring.{ColoredText.END_TEXT}")

                to_client = {
                    'success': False,
                    'sessionID': session_id,
                    'requestID': request_id,
                    'file_size': 0,
                    'message': "Garbage transcription; ignoring."
                }
                worker.send_to_client(to_client, None)
                worker.shutdown()
                return
        else:
            logger.warning(f"{ColoredText.YELLOW_TEXT} ASR Response failed for sessionID {session_id} amd requestID {request_id} for pipeline {pipeline}.{ColoredText.END_TEXT}")
            transcription = "I didn't quite get that."


        if pipeline == 'reflection':

            client_binary_data = worker.get_from_backpack('original_audio')

            to_client = {
                'sessionID': session_id,
                'requestID': request_id,
                'success': True,
                'transcription': transcription
            }

            worker.send_to_client(to_client, client_binary_data)

            logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id} sending ASR transcription and audio back to client.{ColoredText.END_TEXT}")
            worker.shutdown()
        elif pipeline == 'revoice':
            voice = worker.get_from_backpack('voice')
            worker.save_in_backpack('transcription', transcription)

            job = {
                'command': 'tts-send',
                'pipeline': pipeline,
                'sessionID': session_id,
                'requestID': request_id,
                'voice': voice,
                'text': transcription
            }
            worker.add_work(job)
        elif pipeline == 'basic_conversational':
            # Was this speech meant for one of the agents? The ASR has already run, so deciding costs nothing extra.
            selection = select_agent(heard, worker.get_from_backpack('agents'),
                                     continuation=worker.get_from_backpack('continuation'),
                                     active_agent=worker.get_from_backpack('active_agent'),
                                     max_position=worker.get_from_backpack('wake_word_max_position'))
            agent, reason = selection.agent, selection.reason

            if reason == 'ambiguous':
                # Several agents in play and the rules cannot tell which is spoken to: ask the LLM, if allowed. If it
                # is not, or its answer is unusable, the first agent named answers (selection.agent).
                routed = None
                if worker.get_from_backpack('llm_routing'):
                    last_speaker = None
                    if worker.get_from_backpack('continuation'):
                        last_speaker = next((a for a in selection.candidates if a.get('name') == worker.get_from_backpack('active_agent')), None)
                    routed = self._route_with_llm(session_id, request_id, heard, selection.candidates, last_speaker)
                if routed is not None:
                    agent, reason = routed, 'llm_routed'
                else:
                    reason = 'first_named'

            if agent is None:
                # Nobody was addressed: tell the client, so it can go back to listening. Its conversation window (if
                # any) carries on as it was.
                logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id}: no wake word and no conversation under way - not sent to the LLM. Heard: '{heard}'{ColoredText.END_TEXT}")
                to_client = {
                    'success': False,
                    'type': 'not_addressed',
                    'sessionID': session_id,
                    'requestID': request_id,
                    'file_size': 0,
                    'transcription': heard,
                    'message': "Audio received, but no conversation is taking place."
                }
                worker.send_to_client(to_client, None)
                worker.shutdown()
                return

            # Who is talking. With voice recognition on, this is the voice the ASR server recognized (or an
            # unrecognized voice) in place of the client's player_name; everything below - the tag, the handoff
            # notes, the saved history, the reply to the client - follows from it (see speakers.py).
            voice_recognition = worker.get_from_backpack('voice_recognition')
            continuation = worker.get_from_backpack('continuation')
            active_agent = worker.get_from_backpack('active_agent')
            speaker, speaker_source = resolve_speaker(voice_recognition, response if isinstance(response, dict) else {},
                                                      worker.get_from_backpack('speaker'), continuation,
                                                      worker.get_from_backpack('recent_turns'))
            worker.save_in_backpack('speaker', speaker)
            worker.save_in_backpack('speaker_source', speaker_source)
            if voice_recognition:
                logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id}: speaker '{speaker}' ({speaker_source}; ASR status {response.get('speaker_status')!r}, score {response.get('speaker_score')!r}).{ColoredText.END_TEXT}")

            # An agent restricted to certain people (allowed_speakers) refuses anyone else - strictly, even
            # mid-conversation. An unrecognized voice is refused as unknown_speaker, a recognized but unlisted one as
            # speaker_not_allowed, so the client can say which.
            if refuses_unlisted_speaker(voice_recognition, speaker, agent) and speaker != UNRECOGNIZED_SPEAKER:
                logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id}: '{agent['name']}' does not answer '{speaker}' (not in its allowed_speakers) - not sent to the LLM. Heard: '{heard}'{ColoredText.END_TEXT}")
                to_client = {
                    'success': False,
                    'type': 'speaker_not_allowed',
                    'sessionID': session_id,
                    'requestID': request_id,
                    'file_size': 0,
                    'transcription': heard,
                    'agent_name': agent['name'],
                    'speaker': speaker,
                    'message': f"{display_name(agent)} does not answer {speaker}."
                }
                worker.send_to_client(to_client, None)
                worker.shutdown()
                return

            # An agent that only answers known voices refuses an unrecognized one - unless it is already talking with
            # it (a continuation with that same agent). An agent with allowed_speakers refuses an unrecognized voice
            # even then.
            if refuses_unlisted_speaker(voice_recognition, speaker, agent) or refuses_unknown_speaker(voice_recognition, speaker, agent, continuation, active_agent):
                logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id}: '{agent['name']}' does not answer unrecognized voices - not sent to the LLM. Heard: '{heard}'{ColoredText.END_TEXT}")
                to_client = {
                    'success': False,
                    'type': 'unknown_speaker',
                    'sessionID': session_id,
                    'requestID': request_id,
                    'file_size': 0,
                    'transcription': heard,
                    'agent_name': agent['name'],
                    'speaker': speaker,
                    'message': f"Voice not recognized; {display_name(agent)} only answers voices it knows."
                }
                worker.send_to_client(to_client, None)
                worker.shutdown()
                return

            logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id}: agent '{agent['name']}' answers ({reason}).{ColoredText.END_TEXT}")

            # If this agent missed part of the conversation (the user was talking to another agent), say what it
            # missed at the front of the request. The client still gets back - and logs - only what the user said.
            all_agents = worker.get_from_backpack('agents')
            note = build_handoff_note(worker.get_from_backpack('recent_turns'), agent['name'],
                                      {a.get('name'): a.get('display_name', '') for a in all_agents},
                                      max_turns=worker.get_from_backpack('handoff_max_turns'),
                                      max_chars=worker.get_from_backpack('handoff_max_chars'),
                                      default_speaker=speaker)
            if note:
                logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id}: handing '{agent['name']}' what it missed: {note.strip()}{ColoredText.END_TEXT}")

            # Say who is talking (and, with several agents, to whom), so the agent never has to guess - it is saved
            # in the agent's history with the words. Only for real speech: the canned "I didn't quite get that." is
            # not something anyone said. The delimiter is taken out of what was heard so it can't hide the words.
            user_request = transcription
            if heard:
                tag = speaker_tag(speaker, display_name(agent) if len(all_agents) > 1 else '')
                user_request = tag + transcription.replace(HIDDEN_DELIMITER, ' ')

            # From here on there is exactly one agent: cut the list down, and put its settings where the LLM and TTS
            # stages look for them
            worker.save_in_backpack('agents', [agent])
            worker.save_in_backpack('agent_name', agent['name'])
            worker.save_in_backpack('system_prompt_id', agent.get('system_prompt_id', 'default'))
            worker.save_in_backpack('voice', agent.get('voice'))
            worker.save_in_backpack('transcription', transcription)

            job = {
                'command': 'llm-send',
                'pipeline': pipeline,
                'sessionID': session_id,
                'requestID': request_id,
                'user_request': note + user_request,
                'agent_name': agent['name'],
                'user_id': worker.get_from_backpack('user_id'),
                'system_prompt_id': agent.get('system_prompt_id', 'default'),
                # Fills '@@NAME@@' in the system prompt when the LLM session is created. With voice recognition on,
                # the agent is talking to whoever is in the room, so the prompt names the household rather than one
                # person; the speaker tag on each turn says who is actually talking.
                'player_name': self.household_name if voice_recognition else worker.get_from_backpack('player_name'),
                'continuous_save': agent.get('continuous_save', False),
                'load_previous': agent.get('load_previous', False)
            }
            worker.add_work(job)


    def handle_tts_worker_drone(self, worker, job):
        """
        job = {
            'command': 'tts-receive',
            'pipeline': worker.get_pipeline,
            'sessionID': session_id,
            'requestID': request_id,
            'response': response,
            'byte_data': raw_data
        }
        """
        session_id = job['sessionID']
        request_id = job['requestID']
        raw_data = job['byte_data']
        response = job['response'] # response is, at least, not empty at this point - when it was packed it was verified to exist
        transcription = worker.get_from_backpack('transcription')

        pipeline = worker.get_pipeline()

        logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id} processing TTS request for pipeline {pipeline}.{ColoredText.END_TEXT}")

        # if you want to handle other audio types or errors, do so here
        #if response.get("success"):
        #    if response.get('type') == 'audio':
        #    else:
        #        transcription = "I didn't quite get that."
        #else:
        #    transcription = "I didn't quite get that."


        if pipeline == 'revoice':

            to_client = {
                'sessionID': session_id,
                'requestID': request_id,
                'success': True,
                'transcription': transcription
            }

            worker.send_to_client(to_client, raw_data)

            logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id} for pipeline {pipeline} sending TTS transcription and audio back to client.{ColoredText.END_TEXT}")
            worker.shutdown()
        elif pipeline == 'basic_conversational':
            transcription = worker.get_from_backpack('transcription')
            llm_response = worker.get_from_backpack('llm_response')

            # agent_name / system_prompt_id go back so the client can name this agent in a continuation
            to_client = {
                'sessionID': session_id,
                'requestID': request_id,
                'success': True,
                'transcription': transcription,
                'llm_response': llm_response,
                'agent_name': worker.get_from_backpack('agent_name'),
                'system_prompt_id': worker.get_from_backpack('system_prompt_id'),
                # who the server took to be talking, so the client can record it in its recent turns
                'speaker': worker.get_from_backpack('speaker'),
                # how that was decided (see speakers.py: request, voice, last_speaker or fallback)
                'speaker_source': worker.get_from_backpack('speaker_source') or 'request'
            }

            worker.send_to_client(to_client, raw_data)
            logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id} for pipeline {pipeline} sending TTS transcription and audio back to client.{ColoredText.END_TEXT}")
            worker.shutdown()


    def handle_llm_worker_drone(self, worker, job):
        """
        job = {
            'command': 'llm-receive',
            'pipeline': 'conversational',
            'sessionID': session_id,
            'requestID': request_id,
            'response': response
        }
        """
        session_id = job['sessionID']
        request_id = job['requestID']
        response = job['response']

        pipeline = worker.get_pipeline()

        if response.get("success"):
            llm_response = response.get("response", "I'm sorry, I didn't catch that.")
        else:
            llm_response = "I'm sorry, could you repeat that?"

        if pipeline == 'basic_conversational':
            voice = worker.get_from_backpack('voice')
            worker.save_in_backpack('llm_response', llm_response)

            job = {
                'command': 'tts-send',
                'pipeline': pipeline,
                'sessionID': session_id,
                'requestID': request_id,
                'voice': voice,
                'text': llm_response
            }
            worker.add_work(job)



    def _handle_ping(self, worker, job):
        """Global handler example"""
        logger.info(f"Ping from session {worker.session_id}")

    def _handle_status(self, worker, job):
        """Global handler example"""
        active_count = worker.get_active_command_count()
        logger.info(f"Session {worker.session_id} has {active_count} active commands")


    @staticmethod
    def load_json_config(filepath: str) -> dict:
        """
        Loads a JSON file and scrapes specific entries into a dictionary.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict: A dictionary containing the scraped configuration fields:


        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file content is not valid JSON.
            KeyError: If any of the required fields are missing from the JSON.
            TypeError: If a field's value is not of the expected type.
        """
        required_fields = {
            'host': str,
            'port': int,
        }

        # Add optional fields with their types
        optional_fields = {
            'asr_host': str,
            'asr_port': int,
            'llm_host': str,
            'llm_port': int,
            'tts_host': str,
            'tts_port': int,
            'llm_response_timeout_seconds': (int, float),
            'household_name': str,
            'log_file': str         # also log to this file (see amadeo_utils.logging_utils); missing = screen only
        }

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Error: The file '{filepath}' was not found.")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error: Invalid JSON format in '{filepath}': {e}", e.doc, e.pos)
        except Exception as e:
            # Catch other potential file reading errors
            raise IOError(f"Error reading file '{filepath}': {e}")

        scraped_data = {}
        # Process required fields (your existing code)
        for field, expected_type in required_fields.items():
            if field not in data:
                raise KeyError(f"Error: Required field '{field}' missing from JSON in '{filepath}'.")

            value = data[field]
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Error: Field '{field}' in '{filepath}' has unexpected type "
                    f"'{type(value).__name__}', expected '{expected_type.__name__}'."
                )
            scraped_data[field] = value

        # Process optional fields (new code)
        for field, expected_type in optional_fields.items():
            if field in data:  # Only process if present
                value = data[field]
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Error: Optional field '{field}' in '{filepath}' has unexpected type "
                        f"'{type(value).__name__}', expected '{expected_type.__name__}'."
                    )
                scraped_data[field] = value

        return scraped_data


    @staticmethod
    def get_args_dict_server() -> dict:
        """
        Gets args dictionary for the conversational AI server.
        """

        # Set up command-line argument parsing
        parser = argparse.ArgumentParser(description='Conversational AI Suite - Multiple paths',formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('--host', default=ConversationalAiServer.HOST, help='Server host address (default: localhost)')
        parser.add_argument('--port', type=int, default=ConversationalAiServer.PORT, help=f"Server port number (default: {ConversationalAiServer.PORT})")

        parser.add_argument('--asr-host', default=ConversationalAiServer.ASR_HOST, help='The ASR server host address (default: localhost)')
        parser.add_argument('--asr-port', type=int, default=ConversationalAiServer.ASR_PORT, help=f"The ASR server port number (default: {ConversationalAiServer.ASR_PORT})")

        parser.add_argument('--tts-host', default=ConversationalAiServer.TTS_HOST, help='The TTS server host address (default: localhost)')
        parser.add_argument('--tts-port', type=int, default=ConversationalAiServer.TTS_PORT, help=f"The TTS server port number (default: {ConversationalAiServer.TTS_PORT})")

        parser.add_argument('--llm-host', default=ConversationalAiServer.LLM_HOST, help='The LLM server host address (default: localhost)')
        parser.add_argument('--llm-port', type=int, default=ConversationalAiServer.LLM_PORT, help=f"The LLM server port number (default: {ConversationalAiServer.LLM_PORT})")
        parser.add_argument('--llm-response-timeout-seconds', type=float, default=ConversationalAiServer.LLM_RESPONSE_TIMEOUT_SECONDS, help=f"How long to wait for the LLM server's reply to one request (default: {ConversationalAiServer.LLM_RESPONSE_TIMEOUT_SECONDS}). Raise it for the agent server, whose turns can take longer.")

        parser.add_argument('--household-name', default=HOUSEHOLD_NAME, help=f"With a client's voice recognition on, what '@@NAME@@' in a system prompt becomes - the agent is talking to whoever is in the room, not one person (default: '{HOUSEHOLD_NAME}').")

        parser.add_argument("--json", type=str, default="", help="If this points to a valid JSON file, the ENTIRE parameter settings are pulled from that file, and the defaults - and other arguments passed from the command line - are ignored. If the JSON load fails for whatever reason, though, the defaults WILL be engaged. Just remember that if there is a dash in the arg name, its going to be an underscore in the JSON.")

        argDict = {}

        try:
            args = parser.parse_args()
            use_default_arg_config = True  # This is only flipped if we successfully load from a JSON file

            json_config_file = args.json

            if json_config_file and os.path.exists(json_config_file):
                try:
                    config_dict = ConversationalAiServer.load_json_config(json_config_file)


                    argDict['host'] = config_dict.get('host', ConversationalAiServer.HOST)
                    argDict['port'] = config_dict.get('port', ConversationalAiServer.PORT)

                    argDict['asr_host'] = config_dict.get('asr_host', ConversationalAiServer.ASR_HOST)
                    argDict['asr_port'] = config_dict.get('asr_port', ConversationalAiServer.ASR_PORT)

                    argDict['tts_host'] = config_dict.get('tts_host', ConversationalAiServer.TTS_HOST)
                    argDict['tts_port'] = config_dict.get('tts_port', ConversationalAiServer.TTS_PORT)

                    argDict['llm_host'] = config_dict.get('llm_host', ConversationalAiServer.LLM_HOST)
                    argDict['llm_port'] = config_dict.get('llm_port', ConversationalAiServer.LLM_PORT)
                    argDict['llm_response_timeout_seconds'] = config_dict.get('llm_response_timeout_seconds', ConversationalAiServer.LLM_RESPONSE_TIMEOUT_SECONDS)
                    argDict['household_name'] = config_dict.get('household_name', HOUSEHOLD_NAME)
                    argDict['log_file'] = config_dict.get('log_file', '')

                    logger.info(f"Config loaded from JSON {json_config_file}.")

                    use_default_arg_config = False

                except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Could not load JSON config [{json_config_file}] - there are errors. Will attempt to load other defaults or args. Error: {e}.")

            elif json_config_file:
                logger.warning(f"Could not load JSON config [{json_config_file}] - file does not exist. Loading from defaults or other parameters sent.")

            if use_default_arg_config:
                argDict['host'] = args.host
                argDict['port'] = args.port

                argDict['asr_host'] = args.asr_host
                argDict['asr_port'] = args.asr_port

                argDict['tts_host'] = args.tts_host
                argDict['tts_port'] = args.tts_port

                argDict['llm_host'] = args.llm_host
                argDict['llm_port'] = args.llm_port
                argDict['llm_response_timeout_seconds'] = args.llm_response_timeout_seconds
                argDict['household_name'] = args.household_name
                argDict['log_file'] = ''    # a log file is only configured through --json: screen only

        except SystemExit as e:
            argDict = {}
            if e.code == 0:
                # --help was used, so print no error
                print(f"Thank you!")
            else:
                logger.error(f"Invalid arguments.")

        # A socket timeout of 0 would make every read non-blocking, and a negative one is an error: a reply timeout
        # must be a positive number of seconds.
        timeout = argDict.get('llm_response_timeout_seconds')
        if argDict and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0):
            logger.error(f"llm_response_timeout_seconds must be a positive number of seconds, not {timeout!r}.")
            argDict = {}

        return argDict
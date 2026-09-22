import gc
import logging
import os
import sys
import threading
import time

from typing import Dict, Any, List, NamedTuple, Optional

# IMPORTANT: llama_utils MUST be imported before llama_cpp. Importing llama_cpp loads the llama.cpp shared library,
# which registers its GGML CUDA backend and pins the device ordering for the life of the process; llama_utils sets
# CUDA_DEVICE_ORDER at import time so that '--gpu N' means the Nth card as 'nvidia-smi -L' lists it. Flip these two
# lines and the GPU selection silently reverts to CUDA's own 'fastest first' ordering.
from amadeo_utils.ai.llm.llama.llama_utils import LlamaUtils
from amadeo_utils.ai.llm.llama import chat_template as ChatTemplate
from llama_cpp import Llama

from amadeo_utils.colored_text import ColoredText

"""
Shared base for the Llama stream classes.

What this owns
--------------
Session lifecycle and the locking discipline around it - the machinery that is identical
for every kind of stream, whatever it does with the model. 'RolePlayStream' and
'KnowledgeBaseStream' extend this, and any future family (a tool-calling one, say) should
too.

What this deliberately does NOT own
-----------------------------------
Prompt assembly. That is where the families genuinely differ, and pulling it in here would
mean flags and 'if' branches in shared code - which is the coupling this class exists to
remove. A subclass decides what goes into a prompt; the base only decides who holds which
lock while it happens.

Contract for subclasses
-----------------------
'__init__' here builds the locks, validates the required files, loads both models and
installs the chat template, so a subclass usually needs no constructor of its own. It
shapes itself to a family through three hooks:

  * validate_required_files()      - extend to check files beyond the two models.
  * post_model_init()              - family setup that needs the loaded models.
  * create_session_from_request()  - build a session from a client request (required).
  * get_response()                 - generate one answer (required).

After construction these attributes exist, and the methods here rely on them:

  * self.argsDict            dict  - the configuration dictionary
  * self.sessions            dict  - session_id -> session dictionary
  * self.sessions_lock       Lock  - guards the STRUCTURE of self.sessions/self.session_locks
  * self.session_locks       dict  - session_id -> Lock guarding that session's CONTENTS
  * self.generating_gpu_lock Lock  - guards the generative model
  * self.embedding_gpu_lock  Lock  - guards the embedding model
  * self.models_released     bool  - False until cleanup() runs
  * self.llm_generator             - the generative model, released by cleanup()
  * self.llm_embedder              - the embedding model, released by cleanup()
  * self.model_type                - from argsDict['model_type']
  * self.thinking_supported  bool  - whether this architecture has a reasoning mode
  * self.architecture_stops  list  - stop strings that end an assistant turn

Lock ordering
-------------
There are four kinds of lock and they must always be acquired in this order. Acquiring
them in any other order between two threads is how you deadlock a server:

  1. self.sessions_lock        - held only for the handful of instructions it takes to look
                                 something up or swap it out; never held across model work
                                 or across a session lock.
  2. self.session_locks[id]    - guards one session's contents. Held for the length of one
                                 request, which can be tens of seconds.
  3. self.generating_gpu_lock  - one generation at a time, machine wide.
  4. self.embedding_gpu_lock   - as above, for embedding.

The two GPU locks are separate rather than one so that a request embedding its result does
not block an unrelated request that is generating. Nothing holds both at once except
cleanup(), which is why the order between them (3 before 4) is only load bearing there.
"""

logger = logging.getLogger(__name__)


class RequestContext(NamedTuple):
    """
    What get_response() needs in hand before it takes a session lock.

    Either 'error' is set - in which case the caller returns it immediately and nothing else is valid - or the session
    and its lock are set. 'start_time' and 'user_input' are always populated, because an error response still has to
    report its own elapsed time.
    """
    session: Optional[Dict[str, Any]]
    session_lock: Optional[threading.Lock]
    start_time: float
    user_input: Optional[str]
    error: Optional[Dict[str, Any]]


class StreamBase:
    """Session lifecycle and lock discipline shared by all Llama stream families."""

    # ----------------------------------------------------------------------------------------------------- Construction

    def __init__(self, argsDict: dict):
        """
        Builds the locks, validates the required files, loads both models and installs the chat template.

        A subclass normally does not need its own constructor: it extends validate_required_files() for any extra file
        it cannot start without, and implements post_model_init() for setup that needs the loaded models.

        Args:
            argsDict: the configuration dictionary for this stream.
        """
        self.argsDict = argsDict
        self.sessions = {}

        # The four-level lock ordering is documented in full in this module's docstring. In short: sessions_lock ->
        # session_locks[id] -> generating_gpu_lock -> embedding_gpu_lock, and never in any other order.
        self.sessions_lock = threading.Lock()
        self.session_locks = {}

        self.generating_gpu_lock = threading.Lock()
        self.embedding_gpu_lock = threading.Lock()

        # Flipped by cleanup() so that a request arriving during shutdown is refused rather than handed a model that is
        # in the middle of being freed. Guarded by the two GPU locks.
        self.models_released = False

        self.model_type = self.argsDict['model_type']

        self.validate_required_files()

        # Work out which card each model goes on. The embedding model is tiny, so it is always pinned to the single
        # nominated GPU even when the generative model is being spread across all of them - splitting a 100 MB model
        # would buy nothing and would put its tensors on a card the generator wants for its own layers.
        gpu_index = self.argsDict.get('gpu', LlamaUtils.GPU_INDEX)
        embedder_gpu_kwargs = LlamaUtils.build_gpu_kwargs(gpu_index, False, 'embedding', logger.info)
        generator_gpu_kwargs = LlamaUtils.build_gpu_kwargs(gpu_index, self.argsDict.get('split_gpus', False), 'generative', logger.info)
        # Flash attention and KV cache precision for the generative model only; the embedder is tiny and keeps the
        # library defaults. See LlamaUtils.build_context_kwargs for why these exist and what they cost.
        generator_context_kwargs = LlamaUtils.build_context_kwargs(self.argsDict.get('flash_attn', LlamaUtils.FLASH_ATTN), self.argsDict.get('kv_cache_type', LlamaUtils.KV_CACHE_TYPE), logger.info)

        # Initialize the EMBEDDING model
        self.llm_embedder = Llama(
            model_path=self.argsDict['embedding_model'],
            n_gpu_layers=self.argsDict['embedding_gpu_layers'],
            embedding=True,  # ESSENTIAL for embedding models
            verbose=self.argsDict['debug'],
            n_ctx=self.argsDict['embedding_max_context_tokens'], # Embedding models don't need huge context for individual texts, but set a reasonable one
            **embedder_gpu_kwargs
        )

        logger.info(f"{ColoredText.GREEN_TEXT}{type(self).__name__}: Embedding model [{self.argsDict['embedding_model']}] loaded with [{self.argsDict['embedding_gpu_layers']}] GPU layers and a context size of [{self.argsDict['embedding_max_context_tokens']}].{ColoredText.END_TEXT}")

        # Initialize the GENERATIVE model
        self.llm_generator = Llama(
            model_path=self.argsDict['generating_model'],
            n_gpu_layers=self.argsDict['generating_gpu_layers'],
            embedding=False, # NOT needed for a generative model
            n_ctx=self.argsDict['generating_max_context_tokens'], # This is the context window for the chat model
            chat_format=self.argsDict['chat_format'],  # you should usually leave this None unless you have a real need
            verbose=self.argsDict['debug'],
            **generator_gpu_kwargs,
            **generator_context_kwargs
        )
        logger.info(f"{ColoredText.GREEN_TEXT}{type(self).__name__}: Generative text model [{self.argsDict['generating_model']}] loaded with [{self.argsDict['generating_gpu_layers']}] GPU layers and a context size of [{self.argsDict['generating_max_context_tokens']}].{ColoredText.END_TEXT}")

        # Take over prompt formatting from llama-cpp-python so that reasoning can be switched off.
        # See the equivalent block in role_play.py for why this is necessary; in short, the library freezes its own
        # formatter and 'create_chat_completion()' has no '**kwargs', so a template variable like 'enable_thinking'
        # cannot otherwise be reached. Reasoning starts OFF, which matters most here: left on, Qwen 3.6 spends the whole
        # response budget deliberating and never reaches an answer, and a spoken session would pay for every token of
        # deliberation in latency before the user hears a word. The REASON_PREFIX command turns it on for one turn.
        #
        # This runs before any other thread exists, and the generator is only ever touched under generating_gpu_lock
        # afterwards, so the handler is installed once here and switched per turn inside the lock.
        #
        # The except is load bearing: a GGUF with no embedded 'tokenizer.chat_template' (older merges such as
        # Midnight-Rose) makes install_chat_handler raise ValueError, and those models must still run on
        # llama-cpp-python's own formatting via argsDict['chat_format']. Removing this fallback breaks every model
        # without an embedded template.
        self.thinking_supported = False
        try:
            ChatTemplate.install_chat_handler(self.llm_generator, thinking=False)
            self.thinking_supported = ChatTemplate.supports_thinking(self.llm_generator)
            architecture = ChatTemplate.model_architecture(self.llm_generator)
            logger.info(f"{ColoredText.GREEN_TEXT}{type(self).__name__}: Using the model's embedded chat template (architecture [{architecture}]); reasoning is {'available and currently suppressed' if self.thinking_supported else 'not applicable to this model'}.{ColoredText.END_TEXT}")
        except ValueError as e:
            logger.warning(f"{ColoredText.YELLOW_TEXT}{type(self).__name__}: {e} Falling back to llama-cpp-python's own prompt formatting.{ColoredText.END_TEXT}")

        # Stop strings that end an assistant turn for THIS architecture; merged with the conversational stops at
        # generation time rather than replacing them.
        self.architecture_stops = ChatTemplate.stop_tokens(self.llm_generator)

        self.post_model_init()

    # ---------------------------------------------------------------------------------------- Subclass extension points
    #
    # Everything a new family has to supply, in one place. The first two have workable defaults and are optional; the
    # last two raise NotImplementedError and must be provided.
    #
    #   validate_required_files()     - called by __init__ BEFORE the models load. Override to check extra files.
    #   post_model_init()             - called by __init__ as its LAST step, once both models exist.
    #   create_session_from_request() - called by handle_client_request when a client opens a session.
    #   get_response()                - called by handle_client_request for every ordinary turn.

    def validate_required_files(self):
        """
        Checks the files this stream cannot start without, exiting if any is missing.

        The base checks the two models every family needs. A subclass that needs more - a knowledge base file, say -
        overrides this, calls super().validate_required_files() first, and then adds its own checks.

        Returns:

        """
        # check to see if both models exist - if not, exit
        if not os.path.exists(self.argsDict['generating_model']):
            logger.error(f"{ColoredText.RED_TEXT}{type(self).__name__}: The model [{self.argsDict['generating_model']}] does not exist - exiting.{ColoredText.END_TEXT}")
            sys.exit(0)
        elif not os.path.exists(self.argsDict['embedding_model']):
            logger.error(f"{ColoredText.RED_TEXT}{type(self).__name__}: The model [{self.argsDict['embedding_model']}] does not exist - exiting.{ColoredText.END_TEXT}")
            sys.exit(0)

    def post_model_init(self):
        """
        Hook for family setup that needs the loaded models. Runs as the last step of construction.

        The default does nothing. Role-play uses it to collect the conversation passphrase; the knowledge base uses it
        to count its system-message tokens, which requires the generator.

        Returns:

        """
        pass

    def create_session_from_request(self, session_id: str, request: Dict[str, Any]) -> str:
        """
        Subclass hook: pull this family's parameters off the request, create the session, and return the system message
        that the 'create_llm_session' reply should carry.

        This is the ONLY part of handle_client_request that differs between families, which is why it is the seam. A
        role-play session needs a player name, a system prompt id and the save/load flags; a knowledge-base session
        needs none of those. Each subclass documents its own request fields here.

        Args:
            session_id: The session to create.
            request: The full client request dictionary.

        Returns:
            str: the system message to return to the client.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement create_session_from_request()")

    def get_response(self, request: Dict[str, Any]):
        """
        Subclass hook: generate the answer to one request.

        Args:
            request: The full client request dictionary.

        Returns:
            dict: the response dictionary to send to the client.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement get_response()")

    # ------------------------------------------------------------------------------------------------ Session lifecycle

    def get_session(self, session_id):
        """
        Looks up a session by id.

        Args:
            session_id: The session to look up.

        Returns:
            dict: the session dictionary, or None if there is no such session.
        """
        with self.sessions_lock:
            return self.sessions.get(session_id)

    def get_session_and_lock(self, session_id):
        """
        Looks up a session AND the lock that guards it, as a single atomic step.

        This exists because fetching the two separately is a race: a caller that gets the session, and only then reaches
        for self.session_locks[session_id], can have remove_session delete the lock in between and take a KeyError to
        the face. Since the pair is returned under one acquisition of sessions_lock, the caller always ends up with a
        lock object that genuinely belongs to the session it was handed - even if the session is torn down a moment
        later, in which case the caller simply does its work against a dictionary nobody will read again.

        Args:
            session_id: The session to look up.

        Returns:
            tuple: (session_dict, session_lock), or (None, None) if there is no such session.
        """
        with self.sessions_lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None, None
            return session, self.session_locks[session_id]

    def remove_session(self, session_id):
        """
        When used with AmadeoServer, set this to 'additional_shutdown' so it will run when the socket is closed. If not using AmadeoServer, run this at the end of the session.

        Args:
            session_id:

        Returns:

        """
        logger.info(f"{ColoredText.BLUE_TEXT}session_id {session_id} ended - removing from dictionary.{ColoredText.END_TEXT}")

        # Detach the session from the structure first, holding sessions_lock only for the pop itself. Once it is out of
        # both dictionaries no new request can find it, so there is nothing to be gained by continuing to hold the
        # structure lock - and a great deal to lose: the wait below can easily run to tens of seconds if the user
        # disconnected mid-generation, and holding sessions_lock across that wait would stall every OTHER user's request
        # dispatch behind this one disconnect.
        with self.sessions_lock:
            session = self.sessions.pop(session_id, None)
            session_lock = self.session_locks.pop(session_id, None)

        if session_lock is None:
            return  # never existed, or a second shutdown for the same session - either way there is nothing to wait on

        # A request that grabbed this session before the pop is still working on it. Wait for it to finish so that we do
        # not return - and let the caller tear the connection down - while a generation is still writing to the session.
        with session_lock:
            pass

    def cleanup(self):
        """
        Releases LLMs from memory. Call this right before shutdown.

        Both GPU locks are taken so that this cannot free a model out from under a request that is mid-generation or
        mid-embedding; the locks are acquired in the documented order (generating, then embedding - see the module
        docstring) and held across the whole release. 'models_released' is set while they are still held, so any request
        that was waiting on a lock finds the flag set the moment it gets in and bails out instead of calling into a
        freed model.

        This does NOT tear down live sessions - AmadeoServer calls remove_session for each of those as its connections
        close. Sessions still holding a VectorDB that references these models will fail if used after this point, which
        is why this belongs at shutdown and nowhere else.

        Returns:

        """

        with self.generating_gpu_lock:
            with self.embedding_gpu_lock:
                if self.models_released:
                    return  # already cleaned up; a second call must not del a second time

                self.models_released = True

                del self.llm_embedder
                del self.llm_generator

                gc.collect()  # Force garbage collection

        # Named after the concrete subclass so the log line reads exactly as it did when this method lived in each of
        # them ('RolePlayStream.cleanup: ...', 'KnowledgeBaseStream.cleanup: ...').
        logger.info(f"{ColoredText.BLUE_TEXT}{type(self).__name__}.cleanup: Generative and embedding models released.{ColoredText.END_TEXT}")

    # ------------------------------------------------------------------------------------------------- Request dispatch

    def handle_client_request(self, request: Dict[str, Any], data: bytes = None):
        """
        This method is designed specifically to handle a request from a server - this class can stay running alongside a server class, but the server class will call this method when it gets a request (the server class will handle stuff like sockets etc etc, but this will handle the SPECIFIC
        tasks related to the LLM). This method (and other methods in other classes that implement this) expects a dictionary and data (bytes, which can represent all kinds of media files), although the data portion of that may not be used (depending on the case; in the case of LLMs, this is not used).
        This should return a dictionary (that will be turned into JSON) and byte data (if applicable, but in our case its not).

        To see the basics of what is expected for the server, see the main description for 'amadeo_server.AmadeoServer', although there are some additional ones specific to a Llama implementation with a vector database:
        * command == 'create_llm_session' (used for the first request from the LLM ONLY - this returns the system message)
            * the fields this consumes vary by family - see the subclass's create_session_from_request()
        * command == 'request' (used for all LLM requests after the first one)
            * 'user_request' - The current request from the user. The LLM will generate a direct response to this.
            * no other fields needed
        * command (anything else) (anything else counts as 'request', with a warning in the log)
            * 'user_request' - The current request from the user. The LLM will generate a direct response to this.
            * no other fields needed

        To see the base dictionary fields will be sent to the client. see the main description for 'amadeo_server.AmadeoServer'; here are ADDITIONAL fields that are sent:
        * response - the response as generated by the LLM

        Args:
            request: A dictionary that will contain fields. It should ALWAYS contain 'user_request', which represents the user's request of the LLM. The first call to this should include the 'system_prompt', but if its not sent in subseuqent turns its OK - its set on the first turn.
            data: bytes - This will always be ignored.

        Returns:
            Tuple[dict, None] - The dictionary (that will be converted to JSON and sent to the client), None (Since this has to fit the format of what we may send to a client, that is (JSON, media_data) - and since this returns no media, its always None)
        """

        session_id = request.get('sessionID') # comes from AmadeoServer - at this point, we know its a legit session_id
        command = request.get('command', 'UNKNOWN')
        user_request = request.get('user_request')

        # just see if this session exists
        if self.get_session(session_id):
            sessionExists = True
        else:
            sessionExists = False

        if command != 'create_llm_session' and not user_request:
            # If there is no user request, fail immediately
            logger.warning(f"{ColoredText.GREEN_TEXT}session_id {session_id} made a request, but there was no request contents.{ColoredText.END_TEXT}")
            response = {
                'success': False,
                'type': 'error',
                "response": '',
                "message": "No user request made.",
                "elapsed_time": 0.0,
                'file_size': 0
                }
            return response, None

        else:
            if command == 'create_llm_session' and sessionExists:
                logger.warning(f"{ColoredText.GREEN_TEXT}session_id {session_id} requested to be established, but it was already established - ignoring establishment request and processing LLM request.{ColoredText.END_TEXT}")

                return self.get_response(request), None
            elif command == 'create_llm_session' and not sessionExists:
                system_message = self.create_session_from_request(session_id, request)
                # Spoken and text sessions get the same reply: confirmation that the session exists, carrying the system
                # message. A spoken session once tried to generate a greeting here instead, but passed get_response() a
                # bare string where it expects the request dictionary, so it raised AttributeError and the client never got
                # an answer to its 'create_llm_session' request. The greeting was not worth fixing: the voice pipeline
                # ignores this reply, so the greeting would never be heard, yet it would hold the GPU and leave a
                # synthetic exchange at the top of the session's chat history.
                response = {
                    'success': True,
                    'type': 'system_message',
                    "response": '',
                    "message": system_message,
                    "elapsed_time": 0.0,
                    'file_size': 0
                }
                return response, None
            else:
                if command != 'request':
                    logger.warning(f"{ColoredText.GREEN_TEXT}session_id {session_id} requested command {command} - setting to 'request'.{ColoredText.END_TEXT}")
                    command = 'request'

                return self.get_response(request), None

    # ------------------------------------------------------------------------------------ Shared pieces of get_response
    #
    # These are helpers, not template methods: the subclass's get_response() stays in charge of the order of
    # operations and calls up into these for the mechanical parts. That split is deliberate. The two families assemble
    # a prompt in genuinely different ways - role-play sends the whole chat history when it fits and only falls back to
    # vector search when it does not, while the knowledge base always abridges - and driving that from the base would
    # mean strategy flags in shared code. Each family keeps its own strategy; only the machinery is shared.

    def begin_request(self, request: Dict[str, Any]) -> RequestContext:
        """
        Does the work every request needs before its session lock is taken.

        Starts the clock, pulls the session id and user request out of the request dictionary, fetches the session and
        its lock together, and applies the two guards that can refuse a request outright - an unknown session, and a
        server that is shutting down.

        Args:
            request: The client request dictionary.

        Returns:
            RequestContext: with 'error' set if the request must be refused, otherwise with the session and its lock.
        """
        #start the clock
        start_time = time.time()

        session_id = request.get('sessionID') # comes from AmadeoServer - at this point, we know its a legit session_id
        user_input = request.get('user_request')

        logger.info(f"{ColoredText.BLUE_TEXT}Handling request from session_id '{session_id}'.{ColoredText.END_TEXT}")

        # mySessionDict requires the use of its session lock - we are CONSTANTLY using things from this dictionary here,
        # so just lock the whole thing. The dictionary and its lock are fetched together, in one acquisition of
        # sessions_lock, because fetching them separately races with remove_session - see get_session_and_lock.
        mySessionDict, session_lock = self.get_session_and_lock(session_id)

        # If the session_id was not found, immediately exit
        if not mySessionDict:
            return RequestContext(None, None, start_time, user_input, {
                'success': False,
                'type': 'error',
                "response": '',
                "message": f"session_id {session_id} not found - maybe it recently closed?",
                "elapsed_time": time.time() - start_time,
                'file_size': 0
            })

        # Cheap early bail during shutdown, so a request arriving after cleanup() does not grind through history
        # assembly and vector searches only to be refused at the generation step. This read is deliberately unlocked -
        # it is an optimisation, not the guard; the load bearing check is inside generating_gpu_lock further down.
        if self.models_released:
            return RequestContext(None, None, start_time, user_input, {
                'success': False,
                'type': 'error',
                "response": '',
                "message": "The server is shutting down and the models have been released.",
                "elapsed_time": time.time() - start_time,
                'file_size': 0
            })

        return RequestContext(mySessionDict, session_lock, start_time, user_input, None)

    def format_history_dump(self, items: List[Dict[str, Any]]) -> str:
        """
        Renders chat history for the '!history' command, which shows the user what WOULD have been sent to the model.

        This only builds a string and hands it back to the caller, which returns it to the client. It deliberately
        writes nothing to the log: the text contains the conversation itself, and conversation content is never logged.

        Args:
            items: chat history entries, each with 'role', 'token_count' and 'content'.

        Returns:
            str: the formatted, coloured dump.
        """
        dumped_items = ''
        for item in items:
            dumped_items += f"{ColoredText.YELLOW_TEXT}role: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{item['role']} {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}token count: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{item['token_count']} {ColoredText.END_TEXT}\n"
            dumped_items += f"{ColoredText.YELLOW_TEXT}content: {ColoredText.END_TEXT}{ColoredText.CYAN_TEXT}{item['content']}{ColoredText.END_TEXT}\n\n"
        return dumped_items

    def generate_once(self, messages: List[Dict[str, Any]], local_stop: List[str], max_response_tokens: int,
                      reason_used: bool, session_id: str, used_tokens: int) -> str:
        """
        Runs one generation and returns the cleaned answer.

        The caller builds 'messages' and the conversational part of 'local_stop', because both are family specific -
        role-play adds the player's name as a stop so the model cannot speak for the user. Everything from there on is
        the same for every family and lives here: merging in this architecture's own stops, splitting them, holding the
        GPU lock for the call, and stripping reasoning out of the result.

        Args:
            messages: the assembled conversation to send.
            local_stop: the caller's conversational stop strings.
            max_response_tokens: token budget for this turn's reply.
            reason_used: whether the model should reason on this turn.
            session_id: for logging only.
            used_tokens: for logging only.

        Returns:
            str: the answer, with any reasoning removed and any trailing stop string truncated.

        Raises:
            RuntimeError: if the models were released while this request was queued.
        """
        # The hard-coded stops the caller passed cover ChatML and Llama-3 only. Gemma 4 ends a turn with '<turn|>' and
        # Muse Glimmer with '<|eot|>', neither of which appears there, so add this architecture's own.
        local_stop = local_stop + [stop for stop in self.architecture_stops if stop not in local_stop]

        # Conversational stops would fire inside a reasoning model's deliberation and end the turn empty, so
        # for such models they are applied to the stripped answer instead. See ChatTemplate.split_stops.
        generation_stop, answer_stop = ChatTemplate.split_stops(self.llm_generator, local_stop)

        logger.info(f"{ColoredText.BLUE_TEXT}Sending to the LLM generator for session_id {session_id} ... used_tokens: {used_tokens} generating_max_context_tokens: {self.argsDict['generating_max_context_tokens']} used_max_response_tokens: {max_response_tokens} reasoning: {reason_used and self.thinking_supported}{ColoredText.END_TEXT}")

        with self.generating_gpu_lock:
            # Checked INSIDE the lock: cleanup() sets this while holding the same lock, so a request that
            # was queued behind a shutdown finds it set here rather than calling into a freed model.
            if self.models_released:
                raise RuntimeError("the models have been released - the server is shutting down")

            # Set INSIDE the lock. Reasoning mode is a property of the shared generator, not of a session,
            # so setting it outside would let one session's '!reason' turn leak into whichever other session
            # happened to generate next.
            ChatTemplate.set_thinking(self.llm_generator, reason_used)

            llama_response = self.llm_generator.create_chat_completion(
                messages=messages,
                max_tokens=max_response_tokens,
                stream=False,
                repeat_penalty = self.argsDict['repeat_penalty'],
                stop=generation_stop
            )

        # Get the full response content directly
        full_response_content = llama_response["choices"][0]["message"]["content"]

        # Strip any reasoning before the response goes any further, so that deliberation never reaches the client (or a
        # text-to-speech voice), the vector database or the chat history, where it would be replayed to the model as
        # though it were part of the conversation. This applies on a '!reason' turn too: only the answer is returned.
        # 'thinking' is passed explicitly rather than read back from the model: the lock has been released by now, so
        # another session may already have changed the generator's setting.
        full_response_content = ChatTemplate.strip_reasoning(full_response_content, self.llm_generator, thinking=reason_used)
        return ChatTemplate.truncate_at_stops(full_response_content, answer_stop)

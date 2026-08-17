import os
import sys
import logging
import getpass

# IMPORTANT: llama_utils MUST be imported before llama_cpp. Importing llama_cpp loads the llama.cpp shared library,
# which registers its GGML CUDA backend and pins the device ordering for the life of the process; llama_utils sets
# CUDA_DEVICE_ORDER at import time so that '--gpu N' means the Nth card as 'nvidia-smi -L' lists it. Flip these two
# lines and the GPU selection silently reverts to CUDA's own 'fastest first' ordering.
from amadeo_utils.ai.llm.llama.llama_utils import LlamaUtils
from llama_cpp import Llama

from typing import Dict, Any, Optional
from amadeo_utils.ai.llm.vector_database.VectorDB import VectorDB
from amadeo_utils.colored_text import ColoredText
import threading
from datetime import datetime
import time
import gc

"""
This is an implementation of Llama.cpp. It was primarily built for responding from a server (handle_client_request acts as a callback function for a larger server script), but you could use it independently if you really wanted to as well, although it would be a bit clunky. 
"""

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class RolePlayStream:

    HOST = '127.0.0.1'
    PORT = 65440

    HELP_PREFIX = "!help"
    SAVE_PREFIX = "!save"
    LOAD_PREFIX = "!load"
    THINK_PREFIX = "!remember"
    SEE_PAST_PREFIX = "!history"
    STRIKE_PREFIX = "!strike"
    CRYSTAL_BALL_PREFIX = "!crystal"
    VECTOR_TEST_PREFIX = "!vectortest"
    DATETIME_PREFIX = "!date"
    IGNORE_ME_PREFIX = "!ignoreme"
    IGNORE_YOU_PREFIX = "!ignoreyou"
    HIDDEN_INSTRUCTION_DELIMITER = "##"
    SYSTEM_PROMPT_PLAYER_IDENTIFICATION_DELIMITER = "@@"
    SYSTEM_PROMPT_PLAYER_IDENTIFICATION_LINE_DELIMITER = "##"

    SPEECH_SAVE_PREFIX = "Save."
    SPEECH_LOAD_PREFIX = "Load."
    SPEECH_THINK_PREFIX = "remember"
    SPEECH_STRIKE_PREFIX = "Strike."


    VERYSHORT_PREFIX = "!veryshort"
    SHORT_PREFIX = "!short"
    MEDIUM_PREFIX = "!medium"
    NORMAL_PREFIX = "!normal"
    LONG_PREFIX = "!long"
    VERYLONG_PREFIX = "!verylong"

    HISTORY_SINGLETON = "### Relevant Conversation History:\n"
    HISTORY_REQUEST = "### Relevant Conversation History - Request:\n"
    HISTORY_RESPONSE = "### Relevant Conversation History - Response:\n"


    # --- Initial Knowledge Base Documents ---
    # These are always added to the DB, whether loading a session or starting new.
    # This ensures foundational knowledge is present.
    INITIAL_KNOWLEDGE_BASE_DOCUMENTS = [
        "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure (sea level). This temperature changes with altitude.",
        "Artificial intelligence (AI) is a field of computer science that aims to create intelligent machines capable of reasoning, learning, and problem-solving.",
        "The moon is Earth's only natural satellite. It influences tides and stabilizes Earth's axial tilt.",
        "Mount Everest is the Earth's highest mountain above sea level, located in the Himalayas."
    ]

    #####################################################################################################################################################################################################################################################################
    """
    Constructor for RolePlayStream
    """
    def __init__(self, argsDict: dict):
        self.argsDict = argsDict
        self.sessions = {}

        # ---------------------------------------------------------------------------------------- Locking, in one place
        #
        # There are four kinds of lock here, and they must always be acquired in this order. Acquiring them in any other
        # order between two threads is how you deadlock a server:
        #
        #   1. self.sessions_lock        - guards the STRUCTURE of self.sessions / self.session_locks (which sessions
        #                                  exist). Held only for the handful of instructions it takes to look something
        #                                  up or swap it out; never held across model work or across a session lock.
        #   2. self.session_locks[id]    - guards the CONTENTS of one session's dictionary. Held for the length of one
        #                                  request, which can be tens of seconds.
        #   3. self.generating_gpu_lock  - guards self.llm_generator. One generation at a time, machine wide.
        #   4. self.embedding_gpu_lock   - guards self.llm_embedder.
        #
        # The two GPU locks are separate rather than one lock so that a request embedding its result does not block an
        # unrelated request that is generating. Nothing currently holds both at once except cleanup(), which is why the
        # order between them (3 before 4) is only stated here rather than being load bearing anywhere else.
        self.sessions_lock = threading.Lock()
        self.session_locks = {}

        self.generating_gpu_lock = threading.Lock()
        self.embedding_gpu_lock = threading.Lock()

        # Flipped by cleanup() so that a request arriving during shutdown is refused rather than handed a model that is
        # in the middle of being freed. Guarded by the two GPU locks.
        self.models_released = False

        self.model_type = self.argsDict['model_type']

        # check to see if both models exist - if not, exit
        if not os.path.exists(self.argsDict['generating_model']):
            logger.error(f"{ColoredText.RED_TEXT}RolePlayStream: The model [{self.argsDict['generating_model']}] does not exist - exiting.{ColoredText.END_TEXT}")
            sys.exit(0)
        elif not os.path.exists(self.argsDict['embedding_model']):
            logger.error(f"{ColoredText.RED_TEXT}RolePlayStream: The model [{self.argsDict['embedding_model']}] does not exist - exiting.{ColoredText.END_TEXT}")
            sys.exit(0)

        # Work out which card each model goes on. The embedding model is tiny, so it is always pinned to the single
        # nominated GPU even when the generative model is being spread across all of them - splitting a 100 MB model
        # would buy nothing and would put its tensors on a card the generator wants for its own layers.
        gpu_index = self.argsDict.get('gpu', LlamaUtils.GPU_INDEX)
        embedder_gpu_kwargs = LlamaUtils.build_gpu_kwargs(gpu_index, False, 'embedding', logger.info)
        generator_gpu_kwargs = LlamaUtils.build_gpu_kwargs(gpu_index, self.argsDict.get('split_gpus', False), 'generative', logger.info)

        # Initialize the EMBEDDING model
        self.llm_embedder = Llama(
            model_path=self.argsDict['embedding_model'],
            n_gpu_layers=self.argsDict['embedding_gpu_layers'],
            embedding=True,  # ESSENTIAL for embedding models
            verbose=self.argsDict['debug'],
            n_ctx=self.argsDict['embedding_max_context_tokens'], # Embedding models don't need huge context for individual texts, but set a reasonable one
            **embedder_gpu_kwargs
        )

        logger.info(f"{ColoredText.GREEN_TEXT}RolePlayStream: Embedding model [{self.argsDict['embedding_model']}] loaded with [{self.argsDict['embedding_gpu_layers']}] GPU layers and a context size of [{self.argsDict['embedding_max_context_tokens']}].{ColoredText.END_TEXT}")

        # Initialize the GENERATIVE model
        self.llm_generator = Llama(
            model_path=self.argsDict['generating_model'],
            n_gpu_layers=self.argsDict['generating_gpu_layers'],
            embedding=False, # NOT needed for a generative model
            n_ctx=self.argsDict['generating_max_context_tokens'], # This is the context window for the chat model
            chat_format=self.argsDict['chat_format'],  # you should usually leave this None unless you have a real need
            verbose=self.argsDict['debug'],
            # chat_handler is often useful for proper prompt formatting with chat models,
            # but has been removed for compatibility. Ensure your generative model
            # is fine-tuned for conversational input without explicit chat handler.
            **generator_gpu_kwargs
        )
        logger.info(f"{ColoredText.GREEN_TEXT}RolePlay: Generative text model [{self.argsDict['generating_model']}] loaded with [{self.argsDict['generating_gpu_layers']}] GPU layers and a context size of [{self.argsDict['generating_max_context_tokens']}].{ColoredText.END_TEXT}")

        if self.argsDict['encrypted']:
            self.passphrase = getpass.getpass("🔑 Enter conversation passphrase: ")
        else:
            self.passphrase = ""


    def create_session(self, session_id: str, user_id: str, player_name: str, system_prompt_id: str, spoken_response: bool, continuous_save: bool, load_previous: bool):
        """
        returns the created dictionary.
        Args:
            session_id:
            user_id:
            player_name:
            system_prompt_id:
            spoken_response:
            continuous_save:
            load_previous:

        Returns:

        """
        logger.info(f"{ColoredText.BLUE_TEXT} Adding user_id {user_id} with session_id [{session_id}] and system_prompt_id [{system_prompt_id}] to the dictionary.{ColoredText.END_TEXT}")
        with (self.sessions_lock):
            if session_id not in self.sessions:
                self.sessions[session_id] = {}
                self.sessions[session_id]['session_id'] = session_id
                self.sessions[session_id]['system_prompt_id'] = system_prompt_id
                self.sessions[session_id]['user_id'] = user_id
                self.sessions[session_id]['player_name'] = player_name
                self.sessions[session_id]['spoken_response'] = spoken_response
                self.sessions[session_id]['continuous_save'] = continuous_save
                self.sessions[session_id]['load_previous'] = load_previous
                self.sessions[session_id]['used_tokens'] = 0
                self.sessions[session_id]['max_useable_tokens'] = (1 - self.argsDict['buffer_context_pcnt']) * self.argsDict['generating_max_context_tokens']  # shave a bit off the top to accommodate the buffer
                self.sessions[session_id]['full_history_fits'] = True
                self.sessions[session_id]['convo_dir'] = os.path.join(self.argsDict['base_convo_dir'], user_id, system_prompt_id)
                self.sessions[session_id]['fatal_errors'] = ''
                self.sessions[session_id]['db'] = VectorDB(self.llm_embedder, self.embedding_gpu_lock, self.llm_generator, self.generating_gpu_lock, self.model_type, self.sessions[session_id]['convo_dir'], self.argsDict['debug'], self.passphrase)
                self.sessions[session_id]['chat_history'] = []

                system_message = LlamaUtils.get_system_message(os.path.join(self.argsDict['system_prompt_dir'], system_prompt_id + '.txt'))

                if not player_name:
                    # If there is no given player name, take the line out of the system prompt that identifies the player
                    # We want to use the version of the input that does not have any hidden instructions (marked by the delimiter)
                    system_message = LlamaUtils.remove_instructions(system_message, RolePlayStream.SYSTEM_PROMPT_PLAYER_IDENTIFICATION_LINE_DELIMITER)
                else:
                    # remove the delimiters and put the players name in place of the placeholder
                    system_message = LlamaUtils.replace_instructions(system_message, RolePlayStream.SYSTEM_PROMPT_PLAYER_IDENTIFICATION_DELIMITER, player_name)
                    system_message = LlamaUtils.remove_instruction_delimiters(system_message, RolePlayStream.SYSTEM_PROMPT_PLAYER_IDENTIFICATION_LINE_DELIMITER, False)
                self.sessions[session_id]['system_message'] = system_message
                with self.generating_gpu_lock: # Careful - now if we use self.sessions_lock AND self.generating_gpu_lock it MUST be in that order!
                    self.sessions[session_id]['system_tokens'] = LlamaUtils.universal_token_count(self.llm_generator, "system", self.sessions[session_id]['system_message'], self.model_type)

                # now do some user validation
                if not user_id:
                    self.sessions[session_id]['fatal_errors'] += ' user_id is invalid.'
                if not system_prompt_id:
                    self.sessions[session_id]['fatal_errors'] += ' system_prompt_id is invalid.'

                # Create the lock for this session
                self.session_locks[session_id] = threading.Lock()
            logger.info(f"{ColoredText.BLUE_TEXT} Added session_id [{session_id}]: user_id {user_id}, player_name {player_name}, system_prompt_id [{system_prompt_id}], spoken_response [{spoken_response}], continuous_save [{continuous_save}], load_previous [{load_previous}].{ColoredText.END_TEXT}")
            return self.sessions[session_id]

    def get_session(self, session_id):
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

    def handle_client_request(self, request: Dict[str, Any], data:bytes = None):
        """
        This method is designed specifically to handle a request from a server - this class can stay running alongside a server class, but the server class will call this method when it gets a request (the server class will handle stuff like sockets etc etc, but this will handle the SPECIFIC
        tasks related to the LLM). This method (and other methods in other classes that implement this) expects a dictionary and data (bytes, which can represent all kinds of media files), although the data portion of that may not be used (depending on the case; in the case of LLMs, this is not used).
        This should return a dictionary (that will be turned into JSON) and byte data (if applicable, but in our case its not).

        To see the basics of what is expected for the server, see the main description for 'amadeo_server.AmadeoServer', although there are some additional ones specific to a Llama implementation with a vector database:
        * command == 'create_llm_session' (used for the first request from the LLM ONLY - this returns the system message)
            * user_id - something that identifies the user. This will be used as part of a directory name, which may store the user chat log
            * system_prompt_id - identifies the system prompt
            * player_name - The name of the user as far as the LLM is concerned. This can be different from user_id
            * spoken_response - Boolean. True if this will be run through a TTS (text to speech), False otherwise. If you are just getting back text, ste to False.
            * continuous_save - Boolean. True if you wish to save after every interaction, False otherwise. Saving means you can end the conversation and pick up at a later time / data, exctly where you left off.
            * load_previous - Boolean. If, on the first iteration, we should load any previous conversation, if it exists.
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
                user_id = request.get('user_id', 'UNKNOWN')
                system_prompt_id = request.get('system_prompt_id', 'default')
                player_name = request.get('player_name', '')
                spoken_response = request.get('spoken_response', True) # we pay a higher penalty if this is false and we need a spoken response, rather than if we wished for a text response and got spoken response instead
                continuous_save = request.get('continuous_save', False)
                load_previous = request.get('load_previous', True)

                retDict = self.create_session(session_id, user_id, player_name, system_prompt_id, spoken_response, continuous_save, load_previous)
                if retDict['spoken_response']:
                    # Simulate a greeting, which Really we should never get to this as spoken responses cannot review a vector test, but just in case...
                    response =self.get_response("Hello! Please use a short phrase to respond.")
                else:
                    response = {
                        'success': True,
                        'type': 'system_message',
                        "response": '',
                        "message": retDict['system_message'],
                        "elapsed_time": 0.0,
                        'file_size': 0
                    }
                    return response, None
            else:
                if command != 'request':
                    logger.warning(f"{ColoredText.GREEN_TEXT}session_id {session_id} requested command {command} - setting to 'request'.{ColoredText.END_TEXT}")
                    command = 'request'

                return self.get_response(request), None

    def get_response(self, request: Dict[str, Any]):
        """

        Args:
            request: A dictionary that will contain fields. It should ALWAYS contain 'user_request', which represents the user's request of the LLM. The first call to this should include the 'system_prompt', but if its not sent in subseuqent turns its OK - its set on the first turn.

        Returns:
            Dict - A dictionary that contains the following:
                success - Boolean (if the request was successfully processed).
                type - Either 'llm_response', 'system_message', or 'error'
                response - The response from the LLM (or a simulated response)
                message - If there is a message NOT generated by the LLM (or, not 'simulated' by the LLM if this is returned as speech), that message is here. typically error messages.
                elapsed_time - The time, in seconds, it took to process this request
                file_size - Will always be 0, as this will never return a file


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
            response = {
                'success': False,
                'type': 'error',
                "response": '',
                "message": f"session_id {session_id} not found - maybe it recently closed?",
                "elapsed_time": time.time() - start_time,
                'file_size': 0
            }
            return response

        # Cheap early bail during shutdown, so a request arriving after cleanup() does not grind through history
        # assembly and vector searches only to be refused at the generation step. This read is deliberately unlocked -
        # it is an optimisation, not the guard; the load bearing check is inside generating_gpu_lock further down.
        if self.models_released:
            return {
                'success': False,
                'type': 'error',
                "response": '',
                "message": "The server is shutting down and the models have been released.",
                "elapsed_time": time.time() - start_time,
                'file_size': 0
            }

        # The lock is really for mySessionDict
        with (session_lock):
            logger.info(f"Request received for session_id {mySessionDict['session_id']} - processing.")

            # Immediately check and see if thee are fatal errors
            if mySessionDict['fatal_errors']:
                logger.warning(f"session_id {mySessionDict['session_id']} prompt request rejected - {mySessionDict['fatal_errors']}.")
                response = {
                    'success': False,
                    'type': 'error',
                    "response": '',
                    "message": f"session_id {mySessionDict['session_id']} prompt request rejected - {mySessionDict['fatal_errors']}.",
                    "elapsed_time": time.time() - start_time,
                    'file_size': 0
                }
                return response


            # determine if the user wants to do anything special
            if mySessionDict['spoken_response']:
                # if there is a spoken response
                think_used = LlamaUtils.report_keyword(user_input, self.SPEECH_THINK_PREFIX)

                ignore_user_in_vector_db = False
                ignore_assistant_in_vector_db = False
                crystal_ball = False
                vector_test = False
                chat_history_review = False
                date_given = False

            else:
                # if there is a text response
                vector_test, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.VECTOR_TEST_PREFIX)
                think_used, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.THINK_PREFIX)
                crystal_ball, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.CRYSTAL_BALL_PREFIX)
                chat_history_review, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.SEE_PAST_PREFIX)
                date_given, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.DATETIME_PREFIX)
                ignore_user_in_vector_db, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.IGNORE_ME_PREFIX)
                ignore_assistant_in_vector_db, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.IGNORE_YOU_PREFIX)


            # Check for actual commands that do not interact with the LLM itself - save, load, strike, help. Once done, immediately send the response
            if (mySessionDict['spoken_response'] and user_input.lower() == RolePlayStream.SPEECH_SAVE_PREFIX.lower()) or (user_input == RolePlayStream.SAVE_PREFIX):
                # save the session
                self.save(mySessionDict)

                return {
                    'success': True,
                    'type': 'llm_response',
                    "response": "That is burned into my memory.",
                    "message": '',
                    "elapsed_time": time.time() - start_time,
                    'file_size': 0
                }
            elif (mySessionDict['spoken_response'] and user_input.lower() == RolePlayStream.SPEECH_LOAD_PREFIX.lower()) or (user_input == RolePlayStream.LOAD_PREFIX):
                # load chat history
                mySessionDict['chat_history'] = self.load_chat_history(mySessionDict, True)

                return {
                    'success': True,
                    'type': 'llm_response',
                    "response": "I am not sure what just happened.",
                    "message": '',
                    "elapsed_time": time.time() - start_time,
                    'file_size': 0
                }
            elif (mySessionDict['spoken_response'] and user_input.lower() == RolePlayStream.SPEECH_STRIKE_PREFIX.lower()) or (user_input == RolePlayStream.STRIKE_PREFIX):
                # strike the last request / response from the record
                self.strike_from_record(mySessionDict)

                # If we have elected to continuously save after each LLM response, do so now (technically not a response, but we are removing the last response)
                if mySessionDict['continuous_save']:
                    self.save(mySessionDict)

                return {
                    'success': True,
                    'type': 'llm_response',
                    "response": "I forgot what you just said.",
                    "message": '',
                    "elapsed_time": time.time() - start_time,
                    'file_size': 0
                }
            elif user_input == RolePlayStream.HELP_PREFIX:
                return {
                    'success': True,
                    'type': 'system_message',
                    "response": '',
                    "message": self.get_help(),
                    "elapsed_time": time.time() - start_time,
                    'file_size': 0
                }


            # if there is nothing in the chat history, try to load a previous session, if it exists
            if not mySessionDict['chat_history']:
                mySessionDict['chat_history'] = self.load_chat_history(mySessionDict, mySessionDict['load_previous'])

            # determine the max response tokens, IF it changed
            used_max_response_tokens, user_input = self.adjust_response_tokens(user_input, self.argsDict['max_response_tokens'])

            # Add the system tokens and the tokens allotted for the current assistant response
            used_tokens = mySessionDict['system_tokens'] + used_max_response_tokens

            # add the date time if requested
            if date_given: user_input = user_input + f" For reference, the datetime is {datetime.now().isoformat(timespec='seconds')}."

            # Now that we have cleared out most of the prompts, we can generate the token count and embedding based off the most recent prompt
            with self.generating_gpu_lock:
                user_input_tokens = LlamaUtils.universal_token_count(self.llm_generator, "user", LlamaUtils.remove_instruction_delimiters(user_input, self.HIDDEN_INSTRUCTION_DELIMITER), self.model_type) # get the token count, minus any instruction delimiter

            # Add the user input tokens, so now we have user input tokens and system message tokens
            used_tokens += user_input_tokens # we save the token count with any hidden instructions


            # IF we wanted a vector test, we are now in a position to do so - so do that now and exit immediately
            if vector_test:
                max_vector_db_tokens = .85 * (mySessionDict['max_useable_tokens'] - used_tokens)  # this used to be 'max_vector_database_pcnt * max_useable_tokens', but long system prompts messed with this, so we capture this now, taking into account used_tokens
                temp_top_k = 25  # set this very high to accommodate more returns
                temp_min_vector_db_score = .05

                dumped_items, dumped_tokens = self.get_relevant_items_from_db(mySessionDict, user_input, ignore_user_in_vector_db, ignore_assistant_in_vector_db, temp_min_vector_db_score, max_vector_db_tokens, temp_top_k)
                if mySessionDict['spoken_response']:
                    # Really we should never get to this as spoken responses cannot review a vector test, but just in case...
                    response = {
                        'success': True,
                        'type': 'llm_response',
                        "response": "I'm sorry, I was lost in thought. What did you say, again?",
                        "message": '',
                        "elapsed_time": time.time() - start_time,
                        'file_size': 0
                    }
                else:
                    response = {
                        'success': True,
                        'type': 'llm_response',
                        "response": dumped_items,
                        "message": "",
                        "elapsed_time": time.time() - start_time,
                        'file_size': 0
                    }
                return response


            # Construct messages list for GENERATOR LLM, including system message, context, and chat history
            # Initialize messages_for_llm with the system message
            messages_for_llm = [{"role": "system", "content": mySessionDict['system_message']}]

            if mySessionDict['full_history_fits']:
                total_history_tokens = sum(d['token_count'] for d in mySessionDict['chat_history'])
                # if the total tokens for the history is less than the max context, just use the entire history
                if used_tokens + total_history_tokens <= mySessionDict['max_useable_tokens']:
                    # remove 'token_count'
                    formatted_chat_history = [
                        {'role': d['role'], 'content': d['content']}
                        for d in mySessionDict['chat_history']
                    ]
                    messages_for_llm.extend(formatted_chat_history)

                    used_tokens += total_history_tokens

                    # If we wish to see the chat history, print it
                    if chat_history_review:
                        dumped_items = ''
                        for item in mySessionDict['chat_history']:
                            dumped_items += f"{ColoredText.YELLOW_TEXT}role: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{item['role']} {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}token count: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{item['token_count']} {ColoredText.END_TEXT}\n"
                            dumped_items += f"{ColoredText.YELLOW_TEXT}content: {ColoredText.END_TEXT}{ColoredText.CYAN_TEXT}{item['content']}{ColoredText.END_TEXT}\n\n"
                        if mySessionDict['spoken_response']:
                            # Really we should never get to this as spoken responses cannot review the chat history, but just in case...
                            response = {
                                'success': True,
                                'type': 'llm_response',
                                "response": "I'm sorry, I was lost in thought. What did you say, again?",
                                "message": '',
                                "elapsed_time": time.time() - start_time,
                                'file_size': 0
                            }
                        else:
                            response = {
                                'success': True,
                                'type': 'llm_response',
                                "response": dumped_items,
                                "message": "",
                                "elapsed_time": time.time() - start_time,
                                'file_size': 0
                            }
                        return response


                else:
                    mySessionDict['full_history_fits'] = False


            # if the full history does not fit, we must use the vector database
            if not mySessionDict['full_history_fits']:
                # Search the vector database for relevant context


                # we need to set some things depending on if the user wants the LLM to 'really think'
                if think_used:
                    logger.info(f"{ColoredText.CYAN_TEXT}Going far back in memory for session_id {mySessionDict['session_id']}...{ColoredText.END_TEXT}")
                    max_vector_db_tokens = .85 * (mySessionDict['max_useable_tokens'] - used_tokens) # this used to be 'max_vector_database_pcnt * max_useable_tokens', but long system prompts messed with this, so we capture this now, taking into account used_tokens
                    temp_top_k = 25 # set this very high to accommodate more returns
                    temp_min_vector_db_score = .05

                else:
                    # normal run
                    max_vector_db_tokens = self.argsDict['max_vector_database_pcnt'] * (mySessionDict['max_useable_tokens'] - used_tokens) # this used to be 'max_vector_database_pcnt * max_useable_tokens', but long system prompts messed with this, so we capture this now, taking into account used_tokens
                    temp_top_k = self.argsDict['top_k']
                    temp_min_vector_db_score = self.argsDict['min_vector_db_score']


                # determine if there were relevant items from the vector DB
                db_items, db_tokens = self.get_relevant_items_from_db(mySessionDict, user_input, ignore_user_in_vector_db, ignore_assistant_in_vector_db, temp_min_vector_db_score, max_vector_db_tokens, temp_top_k)

                # if there were DB items
                if db_items:
                    messages_for_llm.extend(db_items)

                    # add in the token count from the vector db results
                    used_tokens += db_tokens


                # Finally, add on the chat history - used_tokens is now the sum of the new user request, the system message, the preemptive assistant response, and the vector db entries
                abridged_chat_history, abridged_chat_history_tokens = LlamaUtils.fit_to_token_limit(mySessionDict['chat_history'], mySessionDict['max_useable_tokens'] - used_tokens)

                # Add in the abridged chat history tokens
                used_tokens += abridged_chat_history_tokens

                # If we wish to see the chat history, send it
                if chat_history_review:
                    dumped_items = ''
                    for item in abridged_chat_history:
                        dumped_items += f"{ColoredText.YELLOW_TEXT}role: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{item['role']} {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}token count: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{item['token_count']} {ColoredText.END_TEXT}\n"
                        dumped_items += f"{ColoredText.YELLOW_TEXT}content: {ColoredText.END_TEXT}{ColoredText.CYAN_TEXT}{item['content']}{ColoredText.END_TEXT}\n\n"
                    if mySessionDict['spoken_response']:
                        # Really we should never get to this as spoken responses cannot review the chat history, but just in case...
                        response = {
                            'success': True,
                            'type': 'llm_response',
                            "response": "I'm sorry, I was lost in thought. What did you say, again?",
                            "message": '',
                            "elapsed_time": time.time() - start_time,
                            'file_size': 0
                        }
                    else:
                        response = {
                            'success': True,
                            'type': 'llm_response',
                            "response": dumped_items,
                            "message": "",
                            "elapsed_time": time.time() - start_time,
                            'file_size': 0
                        }
                    return response


                # remove 'token_count'
                formatted_chat_history = [
                    {'role': d['role'], 'content': d['content']}
                    for d in abridged_chat_history
                ]

                # store in messages_for_llm
                messages_for_llm.extend(formatted_chat_history)

            # Finally, append the most recent content; remember to remove any instruction delimiters if they exist (but leave the instructions intact)
            messages_for_llm.append({"role": "user", "content": LlamaUtils.remove_instruction_delimiters(user_input, RolePlayStream.HIDDEN_INSTRUCTION_DELIMITER)})


            if not chat_history_review:
                try:
                    # Generate response from the GENERATOR LLM

                    # if the user has a name, make that a stop point - otherwise do not.
                    # This is important, as sometimes the LLM goes off the rails and tries to speak for you - so you need to stop that
                    if mySessionDict['player_name']:
                        local_stop = ["[INST]", "<|im_end|>", "<|start_header_id|>", "User:", "Assistant:", mySessionDict['player_name'] + ":"]
                    else:
                        local_stop = ["[INST]", "<|im_end|>", "<|start_header_id|>", "User:", "Assistant:"]

                    logger.info(f"{ColoredText.BLUE_TEXT}Sending to the LLM generator for session_id {mySessionDict['session_id']} ... used_tokens: {used_tokens} generating_max_context_tokens: {self.argsDict['generating_max_context_tokens']} used_max_response_tokens: {used_max_response_tokens}{ColoredText.END_TEXT}")

                    with self.generating_gpu_lock:
                        # Checked INSIDE the lock: cleanup() sets this while holding the same lock, so a request that
                        # was queued behind a shutdown finds it set here rather than calling into a freed model.
                        if self.models_released:
                            raise RuntimeError("the models have been released - the server is shutting down")

                        llama_response = self.llm_generator.create_chat_completion(
                            messages=messages_for_llm,
                            max_tokens=used_max_response_tokens,
                            stream=False,
                            repeat_penalty = self.argsDict['repeat_penalty'],
                            stop=local_stop
                        )

                    # Get the full response content directly
                    full_response_content = llama_response["choices"][0]["message"]["content"]

                    # if there was a response AND we didnt look into the crystal ball (i.e. we want to save this interaction), continue
                    if full_response_content.strip() and not crystal_ball:
                        full_response_content = full_response_content.strip()

                        with self.generating_gpu_lock:
                            response_tokens = LlamaUtils.universal_token_count(self.llm_generator, "assistant", full_response_content, self.model_type) # get the token count for the assistant response

                        # We want to use the version of the input that does not have any hidden instructions (marked by the delimiter)
                        cleaned_user_input = LlamaUtils.remove_instructions(user_input, RolePlayStream.HIDDEN_INSTRUCTION_DELIMITER)

                        with self.generating_gpu_lock:
                            cleaned_user_input_tokens = LlamaUtils.universal_token_count(self.llm_generator, "user", cleaned_user_input, self.model_type) # get the token count, minus any instructions. This will be stored to the vector database

                        # The user request and assistant response were initially separate, having the request and response stored separately; however, Gemini said it would be best if they were combined, both for the embedding AND the text
                        # Gemini also said we want to use VECTOR_DB_USER_REQUEST and VECTOR_DB_AGENT_RESPONSE in the embedding - as it would help it - but when we retrieve it, we want to split on VECTOR_DB_AGENT_RESPONSE and then remove VECTOR_DB_USER_REQUEST, and put them both into the chat history separately explained to me that
                        # Do not forget that we want to completely remove any hidden instructions before we save to the database
                        mySessionDict['db'].add_document(cleaned_user_input, full_response_content)

                        # Update chat history with user input and assistant response for future turns
                        mySessionDict['chat_history'].append({"role": "user", "content": cleaned_user_input, "token_count": cleaned_user_input_tokens})
                        mySessionDict['chat_history'].append({"role": "assistant", "content": full_response_content, "token_count": response_tokens})

                        # If we have elected to continuously save after each LLM response, do so now
                        if mySessionDict['continuous_save']:
                            self.save(mySessionDict)

                        # finally, make a dictionary that will be returned to the client
                        response = {
                            'success': True,
                            'type': 'llm_response',
                            "response": full_response_content,
                            "message": '',
                            "elapsed_time": time.time() - start_time,
                            'file_size': 0
                        }
                    elif crystal_ball:
                        logger.info(f"{ColoredText.BLUE_TEXT}session_id {mySessionDict['session_id']} uses the crystal ball, costing us a Morty.{ColoredText.END_TEXT}")
                        response = {
                            'success': True,
                            'type': 'llm_response',
                            "response": '*You peer into the crystal ball* ' + full_response_content,
                            "message": '',
                            "elapsed_time": time.time() - start_time,
                            'file_size': 0
                        }
                    else:
                        logger.warning(f"{ColoredText.GREEN_TEXT}The LLM goofed for session_id {mySessionDict['session_id']} and didn't return a proper response.{ColoredText.END_TEXT}")
                        if mySessionDict['spoken_response']:
                            # While this IS a failure, mark as a success and just simulate the LLM asking you to repeat, as this will be spoken and not in text
                            response = {
                                'success': True,
                                'type': 'llm_response',
                                "response": 'Sorry, you are breaking up; what did you say, again?',
                                "message": '',
                                "elapsed_time": time.time() - start_time,
                                'file_size': 0
                            }
                        else:
                            # this is a little different - it actually marks this as a failure and puts the response in the 'message' instead. This is because this is a text response, and we
                            # can deal with errors a bit better with text
                            response = {
                                'success': False,
                                'type': 'error',
                                "response": '',
                                "message": "The LLM goofed and didn't return a proper response; please try again.",
                                "elapsed_time": time.time() - start_time,
                                'file_size': 0
                            }

                except Exception as e:
                    logger.error(f"{ColoredText.RED_TEXT}Uncaught exception when attempting to generate text for session_id {mySessionDict['session_id']}: [{e}].{ColoredText.END_TEXT}")
                    response = {
                        'success': False,
                        'type': 'error',
                        "response": '',
                        "message": f"Uncaught exception when attempting to generate text: [{e}]",
                        "elapsed_time": time.time() - start_time,
                        'file_size': 0
                    }
            else:
                # We simply wanted to see the chat history - however, we somehow got here and we shouldnt have, as seeing the chat history was handled above
                # This is simply a safety net
                if mySessionDict['spoken_response']:
                    response = {
                        'success': True,
                        'type': 'llm_response',
                        "response": 'Sorry, someone was talking in the background; what did you say, again?',
                        "message": '',
                        "elapsed_time": time.time() - start_time,
                        'file_size': 0
                    }
                else:
                    # this is a little different - it actually marks this as a failure and puts the response in the 'message' instead. This is because this is a text response, and we
                    # can deal with errors a bit better with text
                    response = {
                        'success': False,
                        'type': 'error',
                        "response": '',
                        "message": "You have reached the chat history section, but this should have been handled.",
                        "elapsed_time": time.time() - start_time,
                        'file_size': 0
                    }
                logger.warning(f"{ColoredText.GREEN_TEXT}session_id {mySessionDict['session_id']} somehow reached the 'else' in the chat history and they shouldn't have (the return should have happened already).{ColoredText.END_TEXT}")

        return response


    """
    Adjusts the response tokens, as necessary; It returns the new response token count; in addition, it changes the prompt to request the response to use up to the token count, no more. 
    It uses the hidden instruction delimiter so it wont show in the chat log  
    """
    def adjust_response_tokens(self, local_text: str, max_response_tokens:int)->(int,str):

        #### #figure out if we want to override the base of max_response_tokens
        override_tokens = 0
        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlayStream.VERYSHORT_PREFIX)
        if max_response_override: override_tokens = 32

        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlayStream.SHORT_PREFIX)
        if max_response_override: override_tokens = 64

        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlayStream.MEDIUM_PREFIX)
        if max_response_override: override_tokens = 128

        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlayStream.NORMAL_PREFIX)
        if max_response_override: override_tokens = 256

        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlayStream.LONG_PREFIX)
        if max_response_override: override_tokens = 512

        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlayStream.VERYLONG_PREFIX)
        if max_response_override: override_tokens = 1024

        # if this was never set - or it was set to max_response_tokens - take the default
        if override_tokens == 0 or (override_tokens == max_response_tokens):
            override_tokens = max_response_tokens
        else:
            local_text = f"{local_text}{RolePlayStream.HIDDEN_INSTRUCTION_DELIMITER} Use up to {override_tokens} tokens in your response; try to fill the entire token count, if it makes sense; do not mention the change in tokens or change in response pattern.{RolePlayStream.HIDDEN_INSTRUCTION_DELIMITER}"

        return override_tokens, local_text


    def get_relevant_items_from_db(self, sessionDict: Dict, local_prompt:str, ignore_user_in_vector_db: bool, ignore_assistant_in_vector_db: bool, local_min_confidence_score: float, local_max_tokens, local_top_k: int):
        """
        This MUST be called from within a lock on self.session_locks[session_id]!

        :param local_prompt:
        :param ignore_user_in_vector_db:
        :param ignore_assistant_in_vector_db:
        :param local_min_confidence_score:
        :param local_max_tokens:
        :param local_top_k:
        :param print_lines:
        :return:
        """

        retVal = []

        logger.info(f"{ColoredText.BLUE_TEXT}RolePlayStream.get_relevant_items_from_db: Searching Vector database for relevant context for session_id {sessionDict['session_id']}; top_k = {local_top_k}, max_vector_db_tokens = {local_max_tokens} ...{ColoredText.END_TEXT}")

        # Retrieve top K documents based on similarity
        # also, COMPLETELY remove any hidden instructions from the prompt, and then turn the prompt into an embedding
        retrieved_results = sessionDict['db'].search(LlamaUtils.remove_instructions(local_prompt, self.HIDDEN_INSTRUCTION_DELIMITER), ignore_user_in_vector_db, ignore_assistant_in_vector_db, k=local_top_k)  # Get top K relevant documents

        logger.info(f"{ColoredText.BLUE_TEXT}RolePlayStream.get_relevant_items_from_db: Vector Database search complete for session_id {sessionDict['session_id']} ...{ColoredText.END_TEXT}")

        temp_vdb_token_count = 0

        # Format retrieved context for the GENERATOR LLM
        if retrieved_results:
            for column_header, user_request, user_token_count, assistant_response, assistant_token_count, score in retrieved_results:
                # if the score is acceptable AND the token count will not put us over local_max_tokens
                if (score > local_min_confidence_score) and ((temp_vdb_token_count + user_token_count + assistant_token_count) <= local_max_tokens):
                    temp_vdb_token_count += user_token_count + assistant_token_count

                    retVal.append({"role": "user", "content": RolePlayStream.HISTORY_REQUEST + user_request})
                    retVal.append({"role": "assistant", "content": RolePlayStream.HISTORY_RESPONSE + assistant_response})

        else:
            logger.info(f"{ColoredText.YELLOW_TEXT}RolePlayStream.get_relevant_items_from_db: No chat history found in vector database for session_id {sessionDict['session_id']}.{ColoredText.END_TEXT}")

        return retVal, temp_vdb_token_count


    def strike_from_record(self, sessionDict: Dict):
        """
        This MUST be called from within a lock on self.session_locks[session_id]!

        :return:
        """
        sessionDict['db'].strike_last_from_record()

        # strike from local chat history
        if len(sessionDict['chat_history']) >= 2:  # Check if there are at least two items to remove
            last_response = sessionDict['chat_history'].pop()  # Removes last response from LLM
            last_request = sessionDict['chat_history'].pop()  # Removes last request from you

            logger.info(f"{ColoredText.BLUE_TEXT}Removed previous pair from conversation history for session_id {sessionDict['session_id']}.{ColoredText.END_TEXT}")
        else:
            logger.info(f"{ColoredText.BLUE_TEXT}Chat history not long enough to stroke last conversation for session_id {sessionDict['session_id']}.{ColoredText.END_TEXT}")

    ################################################################################ Save and Load ####################################################################################################################
    def save(self, sessionDict: Dict):
        """
        This MUST be called from within a lock on self.session_locks[session_id]!

        :param local_chat_history:
        :return:
        """
        sessionDict['db'].save_session(sessionDict['chat_history'])


    def load_chat_history(self, sessionDict: Dict, load_previous: bool)->list:
        """
        This MUST be called from within a lock on self.session_locks[session_id]!

        (Re)Load chat history
        """
        sessionDict['chat_history'] = []
        with (self.generating_gpu_lock):
            static_response_tokens = LlamaUtils.universal_token_count(self.llm_generator, "assistant", VectorDB.ASSISTANT_RESPONSE, self.model_type)


        if os.path.exists(sessionDict['convo_dir']):
            if load_previous:
                sessionDict['chat_history'] = sessionDict['db'].load_session()  # db.df will be updated internally by load_session

                # Re-add initial knowledge base documents if they are not already in the loaded DB.
                # This ensures they are always present, even if a partial DB was saved/loaded.
                # A more robust check might involve comparing document hashes or IDs.
                # For simplicity, we just add them again here; duplicates will exist if already loaded,
                # but for small datasets and demonstration, this is acceptable.
                logger.info(f"{ColoredText.BLUE_TEXT}\nRolePlayStream.load_chat_history: Ensuring initial knowledge base documents are present in DB for session_id {sessionDict['session_id']}...{ColoredText.END_TEXT}")
                initial_kb_texts = [doc.strip() for doc in RolePlayStream.INITIAL_KNOWLEDGE_BASE_DOCUMENTS]
                current_db_texts = set(sessionDict['db'].df['user_text'].apply(lambda x: x.strip()))  # Strip to normalize for comparison

                docs_to_add = []
                responses_to_add = []

                for doc_text in RolePlayStream.INITIAL_KNOWLEDGE_BASE_DOCUMENTS:
                    if doc_text.strip() not in current_db_texts:

                        # create a string that will correctly store this as a request / response (the LLM expects this)
                        docs_to_add.append(doc_text.strip())
                        responses_to_add.append(VectorDB.ASSISTANT_RESPONSE)

                if docs_to_add:
                    sessionDict['db'].add_documents(docs_to_add, responses_to_add)
                    logger.info(f"{ColoredText.BLUE_TEXT}RolePlayStream.load_chat_history: Added {len(docs_to_add)} missing initial knowledge base documents for session_id {sessionDict['session_id']}.{ColoredText.END_TEXT}")
                else:
                    logger.info(f"{ColoredText.BLUE_TEXT}RolePlayStream.load_chat_history: All initial knowledge base documents already present or DB was loaded fully for session_id {sessionDict['session_id']}.{ColoredText.END_TEXT}")

            else:
                logger.info(f"{ColoredText.BLUE_TEXT}\nRolePlayStream.load_chat_history: Populating Vector Database with initial knowledge base documents (new session) for session_id {sessionDict['session_id']} ...{ColoredText.END_TEXT}")
                for doc_text in RolePlayStream.INITIAL_KNOWLEDGE_BASE_DOCUMENTS:
                    sessionDict['db'].add_document(doc_text.strip(), VectorDB.ASSISTANT_RESPONSE)

                logger.info(f"{ColoredText.BLUE_TEXT}RolePlayStream.load_chat_history: Vector Database initially populated with {len(RolePlayStream.INITIAL_KNOWLEDGE_BASE_DOCUMENTS)} documents for session_id {sessionDict['session_id']}.{ColoredText.END_TEXT}")

        logger.info(f"{ColoredText.BLUE_TEXT}RolePlayStream.load_chat_history: Current Vector Database size: {len(sessionDict['db'].df)} documents.{ColoredText.END_TEXT}")

        return sessionDict['chat_history']

    def cleanup(self):
        """
        Releases LLMs from memory. Call this right before shutdown.

        Both GPU locks are taken so that this cannot free a model out from under a request that is mid-generation or
        mid-embedding; the locks are acquired in the documented order (generating, then embedding - see __init__) and
        held across the whole release. 'models_released' is set while they are still held, so any request that was
        waiting on a lock finds the flag set the moment it gets in and bails out instead of calling into a freed model.

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

        logger.info(f"{ColoredText.BLUE_TEXT}RolePlayStream.cleanup: Generative and embedding models released.{ColoredText.END_TEXT}")

    @staticmethod
    def get_help() -> str:
        retVal = ''
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.SAVE_PREFIX}' to save the current session.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.LOAD_PREFIX}' to reload the last saved session (this will clear current unsaved progress).{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.SEE_PAST_PREFIX}' to see the chat history that WOULD have been sent to the LLM; note it does not and is just for you to review it.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.STRIKE_PREFIX}' to remove the last chat request/response from the chat history and vector database.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.CRYSTAL_BALL_PREFIX}' followed by your prompt to see what the LLM would say to a zany question or comment; the request nor response are saved in the chat history, so after the LLM initially responds, it will be like you never asked the question. Careful, though, it does consume a Morty!{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.THINK_PREFIX}' followed by your prompt to get the LLM to really dig deep in its memory; what this really means is the 'long term' chat history of the vector database will have ample amount of room to try to find the answer from previous conversations. This is useful if you are asking for information that is well outside of the context history window. Note that if the entire chat history fits within the context, the database will not be used (as there is no need, its all there).{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.DATETIME_PREFIX}' to print the current datetime in a line (something like 'For reference, the datetime is YYYY-MM-DD HH:II:SS); useful for tracking dates.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.VECTOR_TEST_PREFIX}' followed by your prompt tests the vector database; it will show you everything that would have been selected from the vector database. This does not contact the LLM.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.IGNORE_ME_PREFIX}' followed by your prompt alters the items found matching in the vector database by ignoring the user input / p[rompt / request via replacing it with a generic 'Update me.'; useful if you want to save on tokens while having the assistant use its previous responses (good for having it remember parts of a roleplay world it created).{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.IGNORE_YOU_PREFIX}' followed by your prompt alters the items found matching in the vector database by ignoring the assistant response via replacing it with a generic 'Fascinating.'; useful if you want the assistant to focus on what you said and ignore its response (for example, if you use it for journaling). Also useful if the assistant is chatty and fills responses with nonsense filler or questions, and you want it to focus on what you said.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.VERYSHORT_PREFIX}', '{RolePlayStream.SHORT_PREFIX}', '{RolePlayStream.MEDIUM_PREFIX}', '{RolePlayStream.NORMAL_PREFIX}', '{RolePlayStream.LONG_PREFIX}', or '{RolePlayStream.VERYLONG_PREFIX}'  followed by your prompt to temporarily set the max-response-tokens. Values: '{RolePlayStream.VERYSHORT_PREFIX}'=32, '{RolePlayStream.SHORT_PREFIX}'=64, '{RolePlayStream.MEDIUM_PREFIX}'=128, '{RolePlayStream.NORMAL_PREFIX}'=256, '{RolePlayStream.LONG_PREFIX}'=512, '{RolePlayStream.VERYLONG_PREFIX}'=1024. Omit these to use the default.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Sometimes, you want to send instructions for this round of chat to the LLM, bout you dont want the instructions saved to the vector database _or_ the chat history; in those cases, wrap instructions in the '{RolePlayStream.HIDDEN_INSTRUCTION_DELIMITER}' delimiter like so: 'Tell me about Artificial intelligence{RolePlayStream.HIDDEN_INSTRUCTION_DELIMITER} , but please use no more than 50 characters{RolePlayStream.HIDDEN_INSTRUCTION_DELIMITER}.' This way the instructions will not be saved (so it wont influence future generations).{ColoredText.END_TEXT}\n"

        retVal += f"{ColoredText.BLUE_TEXT}* ...and, finally, type '{RolePlayStream.HELP_PREFIX}' for this menu again!{ColoredText.END_TEXT}\n"

        return retVal

    @staticmethod
    def get_args_dict() -> dict:
        """
        Gets the args dictionary for the role-play server.

        This is a thin wrapper around LlamaUtils.get_args_dict_role_play_server - every entry point in this tree shares
        one set of argument definitions and one set of config loaders, so that adding a setting (or fixing the help text
        on one) only has to happen in a single place. All this class contributes is its own default bind address, which
        it owns because the two servers must not default to the same port.

        :return: The merged dictionary of system and role-play server settings. Empty if argument parsing failed or '--help' was used.
        """

        return LlamaUtils.get_args_dict_role_play_server(RolePlayStream.HOST, RolePlayStream.PORT, logger.info)

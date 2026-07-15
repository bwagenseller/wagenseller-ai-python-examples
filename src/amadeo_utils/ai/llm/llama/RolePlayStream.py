import os
import sys
import logging
import getpass
from llama_cpp import Llama
from typing import Dict, Any, Optional
from amadeo_utils.ai.llm.vector_database.VectorDB import VectorDB
from amadeo_utils.ai.llm.llama.llama_utils import LlamaUtils
from amadeo_utils.colored_text import ColoredText
from amadeo_utils.ai.llm.llama.subjective_constants import SubjectiveConstants
import threading
from datetime import datetime
import time
import argparse
import json
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

        self.sessions_lock = threading.Lock()  # For adding/removing sessions
        self.session_locks = {}  # Per-session locks; if we use self.session_locks AND self.generating_gpu_lock it MUST be in that order!

        self.generating_gpu_lock = threading.Lock() # if we use self.sessions_lock AND self.generating_gpu_lock it MUST be in that order!
        self.embedding_gpu_lock = threading.Lock()

        self.model_type = self.argsDict['model_type']

        # check to see if both models exist - if not, exit
        if not os.path.exists(self.argsDict['generating_model']):
            logger.error(f"{ColoredText.RED_TEXT}RolePlayStream: The model [{self.argsDict['generating_model']}] does not exist - exiting.{ColoredText.END_TEXT}")
            sys.exit(0)
        elif not os.path.exists(self.argsDict['embedding_model']):
            logger.error(f"{ColoredText.RED_TEXT}RolePlayStream: The model [{self.argsDict['embedding_model']}] does not exist - exiting.{ColoredText.END_TEXT}")
            sys.exit(0)

        # Initialize the EMBEDDING model
        self.llm_embedder = Llama(
            model_path=self.argsDict['embedding_model'],
            n_gpu_layers=self.argsDict['embedding_gpu_layers'],
            embedding=True,  # ESSENTIAL for embedding models
            verbose=self.argsDict['debug'],
            n_ctx=self.argsDict['embedding_max_context_tokens'] # Embedding models don't need huge context for individual texts, but set a reasonable one
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

    def remove_session(self, session_id):
        """
        When used with AmadeoServer, set this to 'additional_shutdown' so it will run when the socket is closed. If not using AmadeoServer, run this at the end of the session.

        Args:
            session_id:

        Returns:

        """
        # Use main lock when modifying STRUCTURE (removing sessions/locks)
        logger.info(f"{ColoredText.BLUE_TEXT}session_id {session_id} ended - removing from dictionary.{ColoredText.END_TEXT}")
        with self.sessions_lock:
            if session_id in self.sessions:
                # Acquire session lock before deleting to ensure no one is using it
                with self.session_locks[session_id]:
                    del self.sessions[session_id]
                del self.session_locks[session_id]

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

        # mySessionDict requires the use of self.session_locks[session_id] - we are CONSTANTLY using things from this dictionary here, so just lock the whole thing
        mySessionDict = self.get_session(session_id)

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

        # The lock is really for mySessionDict
        with (self.session_locks[session_id]):
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

        Returns:

        """

        del self.llm_embedder
        del self.llm_generator

        gc.collect()  # Force garbage collection

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
        Gets args dictionary for a traditional vector database, meant to save the conversation for later.
        """

        parser = argparse.ArgumentParser(description='Run a LLM, as you see fit.')
        parser.add_argument("-ho", "--host", default=RolePlayStream.HOST, help="The hostname/IP that the server will bind to.")
        parser.add_argument("-p", "--port", type=int, default=RolePlayStream.PORT, help="The port that the server will listen on for requests.")

        parser.add_argument("-bmd", "--base-model-dir", default=SubjectiveConstants.BASE_MODEL_DIR,help="The location of the base model that will generate text.")
        parser.add_argument("-bed", "--base-embedding-dir", default=SubjectiveConstants.BASE_EMBEDDING_DIR,help="The location of the embedding model.")
        parser.add_argument("-m", "--model", default=SubjectiveConstants.MODEL,help="The filename of your text generation model. Please, just the filename, not the directory.")
        parser.add_argument("-mt", "--model-type", default=LlamaUtils.MODEL_TYPE,help="The model type: ['llama-2', 'llama-3', 'alpaca', 'qwen', 'command-r', 'vicuna', 'oasst_llama', 'baichuan-2', 'baichuan', 'openbuddy', 'redpajama-incite', 'snoozy', 'phind', 'intel', 'open-orca', 'mistrallite', 'zephyr', 'pygmalion', 'chatml', 'mistral-instruct', 'chatglm3', 'openchat', 'saiga', 'gemma', 'functionary', 'functionary-v2', 'functionary-v1', 'chatml-function-calling'].")
        parser.add_argument("-cf", "--chat-format", default=LlamaUtils.CHAT_FORMAT,help="The chat format type; you should usually leave this None and let the system figure it out (with the exception of 'command-r' - in that case, use 'llama-3'). Values: ['llama-2', 'llama-3', 'alpaca', 'qwen', 'vicuna', 'oasst_llama', 'baichuan-2', 'baichuan', 'openbuddy', 'redpajama-incite', 'snoozy', 'phind', 'intel', 'open-orca', 'mistrallite', 'zephyr', 'pygmalion', 'chatml', 'mistral-instruct', 'chatglm3', 'openchat', 'saiga', 'gemma', 'functionary', 'functionary-v2', 'functionary-v1', 'chatml-function-calling'].")
        parser.add_argument("-em", "--embedding-model", default=SubjectiveConstants.EMBEDDING_MODEL,help="The filename of your embedding model. Please, just the filename, not the directory.")

        parser.add_argument("-bcd", "--base-convo-dir", default=SubjectiveConstants.BASE_CONVO_DIR,help="The base directory where all conversation histories are stored.")
        parser.add_argument("-spf", "--system-prompt-dir", default=SubjectiveConstants.SYSTEM_PROMPT_DIR,help="A directory that contains system prompt files; this is just the directory, but the prompts themselves are in text files in this directory. These files represent the initial message you send to the LLM to 'set the tone' of the entire conversation. This is critical! Be creative!")

        parser.add_argument("-gl", "--gpu-layers", type=int, default=LlamaUtils.GPU_LAYERS,help="How many GPU layers do you want for the text generator model? -1 means try to get them all, but be warned: if the GPU layers are too high, the model will not fit in VRAM this will fail. This number is usually between 10 and 70, IF -1 does not work.")
        parser.add_argument("-egl", "--embedding-gpu-layers", type=int, default=LlamaUtils.EMBEDDING_GPU_LAYERS,help="Similar to --gpu-layers but for the embedding model (see that description). You will usually want -1 for this. If -1 doesn't work, you can try some value between 10 and 100, but if -1 doesnt work, you probably have much bigger problems as embedding models are very small.")
        parser.add_argument("-mct", "--max-context-tokens", type=int, default=LlamaUtils.MAX_CTX,help="Known as 'n_ctx' in the llama binary, this is the max token count for the entire conversation, including vector database retrieval, chat history, current prompt, and LLM response. This should usually be a power of 2, and common choices are 512, 2048, 4096, or 8192, but it can go higher. Llama 3 models typically have n_ctx of 8192 or more. Just know that the space this tames is n_ctx^2 and it does count against your VRAM.")
        parser.add_argument("-emct", "--embedding-max-context-tokens", type=int, default=LlamaUtils.EMBEDDING_MAX_CTX,help="Similar to '--max-context-tokens' (see that description), but for the embedding model. This does not have to be too big as it only has to accommodate either one request or response; 512 is usually more than enough for this.")
        parser.add_argument("-mrt", "--max-response-tokens", type=int, default=LlamaUtils.MAX_RESPONSE_TOKENS,help="The maximum number of response tokens the LLM will use in its response.")

        parser.add_argument("-rpt", "--repeat-penalty", type=float, default=LlamaUtils.REPEAT_PENALITY,help="A number starting from 1; this is a penality for the LLM to repeat phrases. 1.1 to 1.3 works well; 1.5+ starts making things weird and incoherent.")

        parser.add_argument("-mvdbp", "--max-vector-database-pcnt", type=float, default=LlamaUtils.MAX_VECTOR_DB_PCNT,help="A number from 0 to 1; it represents the percentage of the share of the overall chat history that is occupied by items from the vector database. Note that if the entire chat history fits into max-context-tokens (n_ctx) the vector database will not be used ")
        parser.add_argument("-bcp", "--buffer-context-pcnt", type=float, default=LlamaUtils.BUFFER_CTX_PCNT,help="A number from 0 to 1; it represents the percentage of the share of the overall context tokens we want to use as a 'buffer'; since We cant fully guess the number of tokens in the chat history we send to the LLM, we approximate as best we can. This number is a buffer to help ensure that we do not hit this limit, as the LLM WILL fail if we do.")

        parser.add_argument("-tk", "--top-k", type=int, default=LlamaUtils.TOP_K, help="The number of results returned by the vector database 'search'.")
        parser.add_argument("-mvdbs", "--min-vector-db-score", type=float, default=LlamaUtils.MIN_VECTOR_DB_SCORE, help="Every match from the vector database has a confidence score from 0 to 1; this indicates the minimum score you wish to have in the results from the vector database.")

        parser.add_argument("-en", "--encrypted", type=bool,default=False, help="Is the conversation history encrypted? If set to true, you will have to enter a passphrase.")

        parser.add_argument("-d", "--debug", action='store_true',help="Do you want to see some additional log lines while using the LLM?")
        parser.add_argument("-j", "--json", type=str, default="", help="If this points to a valid JSON file, the ENTIRE parameter settings are pulled from that file, and the defaults - and other arguments passed from the command line - are ignored. If the JSON load fails for whatever reason, though, the defaults WILL be engaged.")

        argDict = {}

        try:
            args = parser.parse_args()
            use_default_arg_config = True  # This is only flipped if we successfully load from a JSON file

            json_config_file = args.json

            if json_config_file and os.path.exists(json_config_file):
                try:
                    config_dict = RolePlayStream.load_json_config(json_config_file)

                    argDict['host'] = config_dict['host']
                    argDict['port'] = config_dict['port']

                    argDict['generating_model'] = os.path.join(config_dict['base_model_dir'], config_dict['model'])
                    argDict['embedding_model'] = os.path.join(config_dict['base_embedding_dir'], config_dict['embedding_model'])

                    argDict['base_convo_dir'] = config_dict['base_convo_dir']
                    argDict['system_prompt_dir'] = config_dict['system_prompt_dir']

                    argDict['generating_gpu_layers'] = config_dict['gpu_layers']
                    argDict['embedding_gpu_layers'] = config_dict['embedding_gpu_layers']
                    argDict['generating_max_context_tokens'] = config_dict['max_context_tokens']
                    argDict['embedding_max_context_tokens'] = config_dict['embedding_max_context_tokens']
                    argDict['max_response_tokens'] = config_dict['max_response_tokens']  # Maximum tokens the generative model is allowed to generate

                    argDict['repeat_penalty'] = config_dict['repeat_penalty']

                    argDict['max_vector_database_pcnt'] = config_dict['max_vector_database_pcnt']
                    argDict['buffer_context_pcnt'] = config_dict['buffer_context_pcnt']

                    argDict['top_k'] = config_dict['top_k']
                    argDict['min_vector_db_score'] = config_dict['min_vector_db_score']

                    argDict['model_type'] = config_dict.get('model_type', LlamaUtils.MODEL_TYPE)
                    argDict['chat_format'] = config_dict.get('chat_format', LlamaUtils.CHAT_FORMAT)
                    argDict['debug'] = config_dict.get('debug', False)

                    argDict['encrypted'] = config_dict.get('encrypted', False)

                    logger.info(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_args_dict: Config loaded from JSON file {json_config_file}; prompt directory is '{argDict['system_prompt_dir']}' and base convo directory is '{argDict['base_convo_dir']}'.{ColoredText.END_TEXT}")
                    use_default_arg_config = False


                except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.error(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict: Could not load JSON config [{json_config_file}] - there are errors. Will attempt to load other defaults or args. Error: {e}.{ColoredText.END_TEXT}")


            elif json_config_file:
                logger.warning(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict: Could not load JSON config [{json_config_file}] - file does not exist. Loading from defaults or other parameters sent.{ColoredText.END_TEXT}")

            if use_default_arg_config:

                argDict['host'] = args.host
                argDict['port'] = args.port

                argDict['generating_model'] = os.path.join(args.base_model_dir, args.model)
                argDict['embedding_model'] = os.path.join(args.base_embedding_dir, args.embedding_model)

                argDict['base_convo_dir'] = args.base_convo_dir

                argDict['system_prompt_dir'] = args.system_prompt_dir

                argDict['generating_gpu_layers'] = args.gpu_layers
                argDict['embedding_gpu_layers'] = args.embedding_gpu_layers
                argDict['generating_max_context_tokens'] = args.max_context_tokens
                argDict['embedding_max_context_tokens'] = args.embedding_max_context_tokens
                argDict['max_response_tokens'] = args.max_response_tokens  # Maximum tokens the generative model is allowed to generate

                argDict['repeat_penalty'] = args.repeat_penalty

                argDict['max_vector_database_pcnt'] = args.max_vector_database_pcnt
                argDict['buffer_context_pcnt'] = args.buffer_context_pcnt

                argDict['top_k'] = args.top_k
                argDict['min_vector_db_score'] = args.min_vector_db_score

                argDict['debug'] = args.debug
                argDict['encrypted'] = args.encrypted
                argDict['model_type'] = args.model_type
                argDict['chat_format'] = args.chat_format

                logger.info(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_args_dict: Config loaded from args / defaults; prompt is from '{argDict['system_prompt_dir']}' and base convo directory is '{argDict['base_convo_dir']}'.{ColoredText.END_TEXT}")

        except SystemExit as e:
            argDict = {}
            if e.code == 0:
                # --help was used, so print no error
                print(f"{ColoredText.BLUE_TEXT}Thank you!{ColoredText.END_TEXT}")
            else:
                logger.error(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict: Invalid arguments.{ColoredText.END_TEXT}")

        return argDict


    @staticmethod
    def load_json_config(filepath: str) -> dict:
        """
        Loads a JSON file and scrapes specific entries into a dictionary.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict: A dictionary containing the scraped configuration fields. All fields are required. An example of a JSON doc:
            {

                "host": "127.0.0.1",
                "port": 65440,

                "base_model_dir": "/home/kevin/ai/models/llama.cpp",
                "base_embedding_dir": "/home/kevin/ai/models/llama.cpp/embedding_models",
                "model": "llama-3-70b.Q4_K_M.gguf",
                "model_type": "Llama3",
                "embedding_model": "nomic-embed-text-v1.5.Q5_K_M.gguf",

                "base_convo_dir": "/home/kevin/ai/chat_history",
                "system_prompt_dir": "/home/kevin/ai/",

                "gpu_layers": 57,
                "embedding_gpu_layers": -1,
                "max_context_tokens": 4096,
                "embedding_max_context_tokens": 512,
                "max_response_tokens": 512,

                "repeat_penalty": 1.1,

                "max_vector_database_pcnt": 0.2,
                "buffer_context_pcnt": 0.05,

                "top_k": 4,
                "min_vector_db_score": 0.46,

                "encrypted": false,

                "debug": false
            }

        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file content is not valid JSON.
            KeyError: If any of the required fields are missing from the JSON.
            TypeError: If a field's value is not of the expected type.
        """
        required_fields = {
            'host': str,
            'port': int,

            'base_model_dir': str,
            'base_embedding_dir': str,
            'model': str,
            'embedding_model': str,

            'base_convo_dir': str,
            "system_prompt_dir": str,

            'gpu_layers': int,
            'embedding_gpu_layers': int,
            'max_context_tokens': int,
            'embedding_max_context_tokens': int,
            'max_response_tokens': int,

            'repeat_penalty': float,

            'max_vector_database_pcnt': float,
            'buffer_context_pcnt': float,

            'top_k': int,
            'min_vector_db_score': float
        }

        # Add optional fields with their types
        optional_fields = {
            'model_type': str,
            'chat_format': Optional[str],
            'encrypted': bool,
            'debug': bool
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
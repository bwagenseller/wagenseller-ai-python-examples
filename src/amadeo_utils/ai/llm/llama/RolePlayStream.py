import os
import logging
import getpass

# IMPORTANT: llama_utils MUST be imported before anything that pulls in llama_cpp - which now includes StreamBase,
# since the base class loads the models. Importing llama_cpp loads the llama.cpp shared library, which registers its
# GGML CUDA backend and pins the device ordering for the life of the process; llama_utils sets CUDA_DEVICE_ORDER at
# import time so that '--gpu N' means the Nth card as 'nvidia-smi -L' lists it. Reorder these lines and the GPU
# selection silently reverts to CUDA's own 'fastest first' ordering.
from amadeo_utils.ai.llm.llama.llama_utils import LlamaUtils
from amadeo_utils.ai.llm.llama.StreamBase import StreamBase

from typing import Dict, Any, Optional
from amadeo_utils.ai.llm.vector_database.VectorDB import VectorDB
from amadeo_utils.colored_text import ColoredText
import threading
from datetime import datetime
import time

"""
This is an implementation of Llama.cpp. It was primarily built for responding from a server (handle_client_request acts as a callback function for a larger server script), but you could use it independently if you really wanted to as well, although it would be a bit clunky. 
"""

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class RolePlayStream(StreamBase):

    HOST = '127.0.0.1'
    PORT = 65440

    HELP_PREFIX = "!help"
    SAVE_PREFIX = "!save"
    LOAD_PREFIX = "!load"
    THINK_PREFIX = "!remember"
    REASON_PREFIX = "!reason"
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

    def post_model_init(self):
        """
        Collects the conversation passphrase, if this user's chat logs are encrypted.

        Runs as the last step of construction. It is asked for here rather than in create_session() because it is
        prompted once per process, interactively, before any session exists.

        Returns:

        """
        if self.argsDict['encrypted']:
            self.passphrase = getpass.getpass("\U0001F511 Enter conversation passphrase: ")
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
                # user_id and system_prompt_id come from the client and become directory / file names: only safe names
                # are used as such (see LlamaUtils.is_safe_name). An unsafe one fails the session below, and meanwhile
                # stands in as a fixed placeholder so no path is ever built from it (e.g. '../../somewhere').
                prompt_path = LlamaUtils.safe_prompt_path(self.argsDict['system_prompt_dir'], system_prompt_id)
                safe_user = user_id if LlamaUtils.is_safe_name(user_id) else '_invalid_user_'
                safe_prompt = system_prompt_id if prompt_path else '_invalid_prompt_'
                self.sessions[session_id]['convo_dir'] = os.path.join(self.argsDict['base_convo_dir'], safe_user, safe_prompt)
                self.sessions[session_id]['fatal_errors'] = ''
                self.sessions[session_id]['db'] = VectorDB(self.llm_embedder, self.embedding_gpu_lock, self.llm_generator, self.generating_gpu_lock, self.model_type, self.sessions[session_id]['convo_dir'], self.argsDict['debug'], self.passphrase)
                self.sessions[session_id]['chat_history'] = []

                system_message = LlamaUtils.get_system_message(prompt_path) if prompt_path else LlamaUtils.BASE_SYSTEM_MESSAGE

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
                if not user_id or not LlamaUtils.is_safe_name(user_id):
                    self.sessions[session_id]['fatal_errors'] += ' user_id is invalid.'
                if not system_prompt_id or not prompt_path:
                    self.sessions[session_id]['fatal_errors'] += ' system_prompt_id is invalid.'

                # Create the lock for this session
                self.session_locks[session_id] = threading.Lock()
            logger.info(f"{ColoredText.BLUE_TEXT} Added session_id [{session_id}]: user_id {user_id}, player_name {player_name}, system_prompt_id [{system_prompt_id}], spoken_response [{spoken_response}], continuous_save [{continuous_save}], load_previous [{load_previous}].{ColoredText.END_TEXT}")
            return self.sessions[session_id]

    def create_session_from_request(self, session_id: str, request: Dict[str, Any]) -> str:
        """
        Pulls the role-play session's parameters off the request and creates the session.

        Request fields consumed:
        * user_id - something that identifies the user. This will be used as part of a directory name, which may store the user chat log
        * system_prompt_id - identifies the system prompt
        * player_name - The name of the user as far as the LLM is concerned. This can be different from user_id
        * spoken_response - Boolean. True if this will be run through a TTS (text to speech), False otherwise. If you are just getting back text, ste to False.
        * continuous_save - Boolean. True if you wish to save after every interaction, False otherwise. Saving means you can end the conversation and pick up at a later time / data, exctly where you left off.
        * load_previous - Boolean. If, on the first iteration, we should load any previous conversation, if it exists.

        Args:
            session_id: The session to create.
            request: The full client request dictionary.

        Returns:
            str: the system message for this session, taken from the session that was just built.
        """
        user_id = request.get('user_id', 'UNKNOWN')
        system_prompt_id = request.get('system_prompt_id', 'default')
        player_name = request.get('player_name', '')
        spoken_response = request.get('spoken_response', True) # we pay a higher penalty if this is false and we need a spoken response, rather than if we wished for a text response and got spoken response instead
        continuous_save = request.get('continuous_save', False)
        load_previous = request.get('load_previous', True)

        retDict = self.create_session(session_id, user_id, player_name, system_prompt_id, spoken_response, continuous_save, load_previous)

        # Role-play reads the system message off the session that was just created, because create_session() rewrites it
        # per player - substituting the player name, or stripping the identification line when there is none. The
        # knowledge base has no such per-session rewriting and reads argsDict instead. The two are NOT interchangeable,
        # and the difference is preserved deliberately rather than unified.
        return retDict['system_message']

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

        # Everything a request needs before its session lock is taken - the clock, the session and its lock, and the
        # two guards that refuse a request outright (unknown session, server shutting down) - is identical for every
        # family. See StreamBase.begin_request.
        ctx = self.begin_request(request)
        if ctx.error:
            return ctx.error

        # Unpacked into the names the rest of this method already uses, so nothing below needs to change.
        mySessionDict, session_lock = ctx.session, ctx.session_lock
        start_time, user_input = ctx.start_time, ctx.user_input

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

                # A spoken session never reveals reasoning - there is nothing sensible to do with
                # deliberation in a voice pipeline except wait longer to hear the answer.
                reason_used = False

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
                reason_used, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.REASON_PREFIX)
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
                    "message": self.get_help(self.argsDict.get('response_token_presets')),
                    "elapsed_time": time.time() - start_time,
                    'file_size': 0
                }


            # if there is nothing in the chat history, try to load a previous session, if it exists
            if not mySessionDict['chat_history']:
                mySessionDict['chat_history'] = self.load_chat_history(mySessionDict, mySessionDict['load_previous'])

            # determine the max response tokens, IF it changed
            used_max_response_tokens, user_input = self.adjust_response_tokens(user_input, self.argsDict['max_response_tokens'])

            # Room for the model's deliberation, which shares max_tokens with the answer - see the matching note in
            # role_play.py. Added after the length prefix's hidden instruction, so it never asks for a longer answer.
            used_max_response_tokens += LlamaUtils.turn_token_allowance(reason_used, self.thinking_supported, mySessionDict['system_tokens'], used_max_response_tokens, mySessionDict['max_useable_tokens'], self.argsDict)

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


            # Construct messages list for GENERATOR LLM: system message, then either the whole chat history (while it
            # still fits) or vector-database context plus as much recent history as fits, then this request. The
            # budgeting is shared with the other families; only the vector search is role-play's own, since it honours
            # the '!ignoreme' / '!ignoreyou' options. See StreamBase.assemble_context.
            assembled = self.assemble_context(
                mySessionDict, mySessionDict['system_message'], user_input, used_tokens,
                mySessionDict['max_useable_tokens'], think_used,
                lambda min_score, max_tokens, top_k: self.get_relevant_items_from_db(
                    mySessionDict, user_input, ignore_user_in_vector_db, ignore_assistant_in_vector_db,
                    min_score, max_tokens, top_k),
                use_full_history_when_it_fits=True)
            messages_for_llm, used_tokens = assembled.messages, assembled.used_tokens

            # If we wish to see the chat history, send what WOULD have gone to the model instead of generating
            if chat_history_review:
                dumped_items = self.format_history_dump(assembled.history_used)
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


            if not chat_history_review:
                try:
                    # Generate response from the GENERATOR LLM

                    # if the user has a name, make that a stop point - otherwise do not.
                    # This is important, as sometimes the LLM goes off the rails and tries to speak for you - so you need to stop that
                    if mySessionDict['player_name']:
                        local_stop = ["[INST]", "<|im_end|>", "<|start_header_id|>", "User:", "Assistant:", mySessionDict['player_name'] + ":"]
                    else:
                        local_stop = ["[INST]", "<|im_end|>", "<|start_header_id|>", "User:", "Assistant:"]

                    # Everything from merging this architecture's stops through stripping reasoning out of the result
                    # is identical for every family; only 'messages_for_llm' and the conversational stops above are
                    # family specific. See StreamBase.generate_once.
                    full_response_content = self.generate_once(
                        messages_for_llm, local_stop, used_max_response_tokens,
                        reason_used, mySessionDict['session_id'], used_tokens)

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
        # The budget behind each length prefix comes from the system config's 'response_token_presets', so it can differ
        # per model; any preset the config leaves out keeps LlamaUtils.RESPONSE_TOKEN_PRESETS.
        presets = self.argsDict.get('response_token_presets', LlamaUtils.RESPONSE_TOKEN_PRESETS)
        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlayStream.VERYSHORT_PREFIX)
        if max_response_override: override_tokens = presets['veryshort']

        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlayStream.SHORT_PREFIX)
        if max_response_override: override_tokens = presets['short']

        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlayStream.MEDIUM_PREFIX)
        if max_response_override: override_tokens = presets['medium']

        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlayStream.NORMAL_PREFIX)
        if max_response_override: override_tokens = presets['normal']

        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlayStream.LONG_PREFIX)
        if max_response_override: override_tokens = presets['long']

        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlayStream.VERYLONG_PREFIX)
        if max_response_override: override_tokens = presets['verylong']

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
    def load_chat_history(self, sessionDict: Dict, load_previous: bool)->list:
        """
        This MUST be called from within a lock on self.session_locks[session_id]!

        (Re)Load chat history
        """
        with (self.generating_gpu_lock):
            static_response_tokens = LlamaUtils.universal_token_count(self.llm_generator, "assistant", VectorDB.ASSISTANT_RESPONSE, self.model_type)

        # Resetting the history and restoring a saved conversation is shared with the other families - see
        # StreamBase.load_chat_history. What follows is role-play's own: its initial knowledge-base documents.
        super().load_chat_history(sessionDict, load_previous)

        if os.path.exists(sessionDict['convo_dir']):
            if load_previous:
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

    @staticmethod
    def get_help(presets: dict = None) -> str:
        """
        Lists the commands. The length-prefix budgets shown are the ones actually in force, since a system config may
        override them.

        :param presets: The run's 'response_token_presets'; LlamaUtils.RESPONSE_TOKEN_PRESETS if not given.
        """
        presets = presets or LlamaUtils.RESPONSE_TOKEN_PRESETS
        retVal = ''
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.SAVE_PREFIX}' to save the current session.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.LOAD_PREFIX}' to reload the last saved session (this will clear current unsaved progress).{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.SEE_PAST_PREFIX}' to see the chat history that WOULD have been sent to the LLM; note it does not and is just for you to review it.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.STRIKE_PREFIX}' to remove the last chat request/response from the chat history and vector database.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.CRYSTAL_BALL_PREFIX}' followed by your prompt to see what the LLM would say to a zany question or comment; the request nor response are saved in the chat history, so after the LLM initially responds, it will be like you never asked the question. Careful, though, it does consume a Morty!{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.REASON_PREFIX}' followed by your prompt to let the model reason out loud for that one turn. Reasoning is normally suppressed, as it costs both time and context to generate text you never see. The deliberation is never saved to the chat history or the vector database. The turn gets extra token room for the deliberation on top of the normal response budget, so there is no need to add a length prefix - and better not to, since those ask the model to fill the whole budget and it will spend it planning.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.THINK_PREFIX}' followed by your prompt to get the LLM to really dig deep in its memory; what this really means is the 'long term' chat history of the vector database will have ample amount of room to try to find the answer from previous conversations. This is useful if you are asking for information that is well outside of the context history window. Note that if the entire chat history fits within the context, the database will not be used (as there is no need, its all there).{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.DATETIME_PREFIX}' to print the current datetime in a line (something like 'For reference, the datetime is YYYY-MM-DD HH:II:SS); useful for tracking dates.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.VECTOR_TEST_PREFIX}' followed by your prompt tests the vector database; it will show you everything that would have been selected from the vector database. This does not contact the LLM.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.IGNORE_ME_PREFIX}' followed by your prompt alters the items found matching in the vector database by ignoring the user input / p[rompt / request via replacing it with a generic 'Update me.'; useful if you want to save on tokens while having the assistant use its previous responses (good for having it remember parts of a roleplay world it created).{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.IGNORE_YOU_PREFIX}' followed by your prompt alters the items found matching in the vector database by ignoring the assistant response via replacing it with a generic 'Fascinating.'; useful if you want the assistant to focus on what you said and ignore its response (for example, if you use it for journaling). Also useful if the assistant is chatty and fills responses with nonsense filler or questions, and you want it to focus on what you said.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{RolePlayStream.VERYSHORT_PREFIX}', '{RolePlayStream.SHORT_PREFIX}', '{RolePlayStream.MEDIUM_PREFIX}', '{RolePlayStream.NORMAL_PREFIX}', '{RolePlayStream.LONG_PREFIX}', or '{RolePlayStream.VERYLONG_PREFIX}'  followed by your prompt to temporarily set the max-response-tokens. Values: '{RolePlayStream.VERYSHORT_PREFIX}'={presets['veryshort']}, '{RolePlayStream.SHORT_PREFIX}'={presets['short']}, '{RolePlayStream.MEDIUM_PREFIX}'={presets['medium']}, '{RolePlayStream.NORMAL_PREFIX}'={presets['normal']}, '{RolePlayStream.LONG_PREFIX}'={presets['long']}, '{RolePlayStream.VERYLONG_PREFIX}'={presets['verylong']}. Omit these to use the default.{ColoredText.END_TEXT}\n"
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

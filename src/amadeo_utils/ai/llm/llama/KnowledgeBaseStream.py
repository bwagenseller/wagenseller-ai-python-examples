import os
import sys
import logging
import numpy as np

# IMPORTANT: llama_utils MUST be imported before anything that pulls in llama_cpp - which now includes StreamBase,
# since the base class loads the models. Importing llama_cpp loads the llama.cpp shared library, which registers its
# GGML CUDA backend and pins the device ordering for the life of the process; llama_utils sets CUDA_DEVICE_ORDER at
# import time so that '--gpu N' means the Nth card as 'nvidia-smi -L' lists it. Reorder these lines and the GPU
# selection silently reverts to CUDA's own 'fastest first' ordering.
from amadeo_utils.ai.llm.llama.llama_utils import LlamaUtils
from amadeo_utils.ai.llm.llama.StreamBase import StreamBase

from typing import Dict, Any, Optional, List, Callable
from amadeo_utils.ai.llm.vector_database.VectorDB import VectorDB
from amadeo_utils.colored_text import ColoredText
import threading
from datetime import datetime
import time
import json

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeBaseStream(StreamBase):

    HOST = '127.0.0.1'
    PORT = 65440

    HELP_PREFIX = "!help"
    THINK_PREFIX = "!remember"
    REASON_PREFIX = "!reason"
    SEE_PAST_PREFIX = "!history"
    VECTOR_TEST_PREFIX = "!vectortest"
    HIDDEN_INSTRUCTION_DELIMITER = "##"

    SPEECH_THINK_PREFIX = "remember"

    HISTORY_SINGLETON = "### Relevant Conversation History:\n"
    HISTORY_REQUEST = "### Relevant Conversation History - Request:\n"
    HISTORY_RESPONSE = "### Relevant Conversation History - Response:\n"


    def validate_required_files(self):
        """
        Checks the two models (via the base) and then the knowledge base file itself.

        Returns:

        """
        super().validate_required_files()

        if not os.path.exists(self.argsDict['knowledge_base_file']):
            logger.error(f"{ColoredText.RED_TEXT}KnowledgeBase: The knowledge base file [{self.argsDict['knowledge_base_file']}] does not exist - exiting.{ColoredText.END_TEXT}")
            sys.exit(0)

        # at this point, the knowledge base file does exist; get the 'knowledge base list', which is the contents of the knowledge base in JSONL format
        self.kbl = KnowledgeBaseStream.read_knowledge_base_file(self.argsDict['knowledge_base_file'])
        if len(self.kbl) == 0:
            logger.error(f"{ColoredText.RED_TEXT}KnowledgeBase: The knowledge base file [{self.argsDict['knowledge_base_file']}] was empty - exiting.{ColoredText.END_TEXT}")
            sys.exit(0)

    def post_model_init(self):
        """
        Counts the system message's tokens and works out how much of the context a session may use.

        Returns:

        """
        # Get the system tokens. This runs during construction, before the server has started and therefore before any
        # other thread exists, so it is the one place the generator is touched without generating_gpu_lock held. Taking
        # the lock here anyway keeps the rule 'never touch llm_generator unlocked' true without exception, which is
        # cheaper to maintain than an exception everyone has to remember.
        with self.generating_gpu_lock:
            self.system_tokens = (LlamaUtils.universal_token_count(self.llm_generator, "system", self.argsDict['system_message'], self.model_type))

        self.max_useable_tokens = (1 - self.argsDict['buffer_context_pcnt']) * self.argsDict['generating_max_context_tokens']  # shave a bit off the top to accommodate the buffer

    def create_session(self, session_id: str, user_id: str, spoken_response: bool):
        """
        returns the created dictionary.
        Args:
            session_id:
            user_id:
            spoken_response:

        Returns:
        """
        logger.info(f"{ColoredText.BLUE_TEXT} Adding user_id {user_id} with session_id [{session_id}] to the dictionary.{ColoredText.END_TEXT}")
        with (self.sessions_lock):
            if session_id not in self.sessions:
                self.sessions[session_id] = {}
                self.sessions[session_id]['session_id'] = session_id
                self.sessions[session_id]['user_id'] = user_id
                self.sessions[session_id]['spoken_response'] = spoken_response
                self.sessions[session_id]['used_tokens'] = 0
                self.sessions[session_id]['full_history_fits'] = True
                self.sessions[session_id]['fatal_errors'] = ''
                self.sessions[session_id]['db'] = VectorDB(self.llm_embedder, self.embedding_gpu_lock, self.llm_generator, self.generating_gpu_lock, self.model_type, '', self.argsDict['debug'])
                self.sessions[session_id]['chat_history'] = []

                self.load_knowledge_base(self.sessions[session_id])

                # now do some user validation
                if not user_id:
                    self.sessions[session_id]['fatal_errors'] += ' user_id is invalid.'

                # Create the lock for this session
                self.session_locks[session_id] = threading.Lock()
            logger.info(f"{ColoredText.BLUE_TEXT} Added session_id [{session_id}]: user_id {user_id}, spoken_response [{spoken_response}]")
            return self.sessions[session_id]


    def create_session_from_request(self, session_id: str, request: Dict[str, Any]) -> str:
        """
        Pulls the knowledge-base session's parameters off the request and creates the session.

        Request fields consumed:
        * user_id - something that identifies the user. This will be used as part of a directory name, which may store the user chat log
        * spoken_response - Boolean. True if this will be run through a TTS (text to speech), False otherwise. If you are just getting back text, ste to False.

        Args:
            session_id: The session to create.
            request: The full client request dictionary.

        Returns:
            str: the configured system message, read from argsDict.
        """
        user_id = request.get('user_id', 'UNKNOWN')
        spoken_response = request.get('spoken_response', True) # we pay a higher penalty if this is false and we need a spoken response, rather than if we wished for a text response and got spoken response instead

        self.create_session(session_id, user_id, spoken_response)

        # Read from argsDict, not from the session dictionary. create_session() does no per-player rewriting here, so in
        # practice the two carry the same text - but they are read from different places, and role-play genuinely needs
        # the session's own copy. Preserved as-is rather than unified.
        return self.argsDict['system_message']

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

                # A spoken session never reasons - the reply is never shown anyway, and deliberation would only make
                # the caller wait longer to hear the answer.
                reason_used = False

                vector_test = False
                chat_history_review = False

            else:
                # if there is a text response
                vector_test, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.VECTOR_TEST_PREFIX)
                think_used, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.THINK_PREFIX)
                reason_used, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.REASON_PREFIX)
                chat_history_review, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.SEE_PAST_PREFIX)


            # Check for actual commands that do not interact with the LLM itself - save, load, strike, help. Once done, immediately send the response
            if user_input == KnowledgeBaseStream.HELP_PREFIX:
                return {
                    'success': True,
                    'type': 'system_message',
                    "response": '',
                    "message": self.get_help(),
                    "elapsed_time": time.time() - start_time,
                    'file_size': 0
                }

            # A model's deliberation comes out of the same max_tokens as its answer: room for a '!reason' turn to think
            # ('reasoning_budget_tokens'), or for a model that thinks even when told not to (Muse Glimmer's
            # 'suppressed_reasoning_tokens'). Without it, Muse Glimmer can spend the whole budget on hidden deliberation
            # and never reach an answer.
            used_max_response_tokens = self.argsDict['max_response_tokens'] + LlamaUtils.turn_token_allowance(reason_used, self.thinking_supported, self.system_tokens, self.argsDict['max_response_tokens'], self.max_useable_tokens, self.argsDict)

            # Add the system tokens and the tokens allotted for the current assistant response
            used_tokens = self.system_tokens + used_max_response_tokens

            # Now that we have cleared out most of the prompts, we can generate the token count and embedding based off the most recent prompt
            with self.generating_gpu_lock:
                user_input_tokens = LlamaUtils.universal_token_count(self.llm_generator, "user", LlamaUtils.remove_instruction_delimiters(user_input, self.HIDDEN_INSTRUCTION_DELIMITER), self.model_type) # get the token count, minus any instruction delimiter

            # Add the user input tokens, so now we have user input tokens and system message tokens
            used_tokens += user_input_tokens # we save the token count with any hidden instructions


            # IF we wanted a vector test, we are now in a position to do so - so do that now and exit immediately
            if vector_test:
                max_vector_db_tokens = .85 * (self.max_useable_tokens - used_tokens)  # this used to be 'max_vector_database_pcnt * max_useable_tokens', but long system prompts messed with this, so we capture this now, taking into account used_tokens
                temp_top_k = 25  # set this very high to accommodate more returns
                temp_min_vector_db_score = .05

                dumped_items, dumped_tokens = self.get_relevant_items_from_db(mySessionDict, user_input, temp_min_vector_db_score, max_vector_db_tokens, temp_top_k)
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
            messages_for_llm = [{"role": "system", "content": self.argsDict['system_message']}]

            # we need to set some things depending on if the user wants the LLM to 'really think'
            if think_used:
                logger.info(f"{ColoredText.CYAN_TEXT}Going far back in memory for session_id {mySessionDict['session_id']}...{ColoredText.END_TEXT}")
                max_vector_db_tokens = .85 * (self.max_useable_tokens - used_tokens) # this used to be 'max_vector_database_pcnt * max_useable_tokens', but long system prompts messed with this, so we capture this now, taking into account used_tokens
                temp_top_k = 25 # set this very high to accommodate more returns
                temp_min_vector_db_score = .05

            else:
                # normal run
                max_vector_db_tokens = self.argsDict['max_vector_database_pcnt'] * (self.max_useable_tokens - used_tokens) # this used to be 'max_vector_database_pcnt * max_useable_tokens', but long system prompts messed with this, so we capture this now, taking into account used_tokens
                temp_top_k = self.argsDict['top_k']
                temp_min_vector_db_score = self.argsDict['min_vector_db_score']


            # determine if there were relevant items from the vector DB
            db_items, db_tokens = self.get_relevant_items_from_db(mySessionDict, user_input, temp_min_vector_db_score, max_vector_db_tokens, temp_top_k)

            # if there were DB items
            if db_items:
                messages_for_llm.extend(db_items)

                # add in the token count from the vector db results
                used_tokens += db_tokens


            # Finally, add on the chat history - used_tokens is now the sum of the new user request, the system message, the preemptive assistant response, and the vector db entries
            abridged_chat_history, abridged_chat_history_tokens = LlamaUtils.fit_to_token_limit(mySessionDict['chat_history'], self.max_useable_tokens - used_tokens)

            # Add in the abridged chat history tokens
            used_tokens += abridged_chat_history_tokens

            # If we wish to see the chat history, send it
            if chat_history_review:
                dumped_items = self.format_history_dump(abridged_chat_history)
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
            messages_for_llm.append({"role": "user", "content": LlamaUtils.remove_instruction_delimiters(user_input, KnowledgeBaseStream.HIDDEN_INSTRUCTION_DELIMITER)})


            if not chat_history_review:
                try:
                    # Generate response from the GENERATOR LLM

                    local_stop = ["[INST]", "<|im_end|>", "<|start_header_id|>", "User:", "Assistant:"]

                    # Everything from merging this architecture's stops through stripping reasoning out of the result
                    # is identical for every family; only 'messages_for_llm' and the conversational stops above are
                    # family specific. See StreamBase.generate_once.
                    full_response_content = self.generate_once(
                        messages_for_llm, local_stop, used_max_response_tokens,
                        reason_used, mySessionDict['session_id'], used_tokens)

                    # if there was a response AND we didnt look into the crystal ball (i.e. we want to save this interaction), continue
                    if full_response_content.strip():
                        full_response_content = full_response_content.strip()

                        with self.generating_gpu_lock:
                            response_tokens = LlamaUtils.universal_token_count(self.llm_generator, "assistant", full_response_content, self.model_type) # get the token count for the assistant response

                        # We want to use the version of the input that does not have any hidden instructions (marked by the delimiter)
                        cleaned_user_input = LlamaUtils.remove_instructions(user_input, KnowledgeBaseStream.HIDDEN_INSTRUCTION_DELIMITER)

                        with self.generating_gpu_lock:
                            cleaned_user_input_tokens = LlamaUtils.universal_token_count(self.llm_generator, "user", cleaned_user_input, self.model_type) # get the token count, minus any instructions. This will be stored to the vector database

                        # Update chat history with user input and assistant response for future turns
                        mySessionDict['chat_history'].append({"role": "user", "content": cleaned_user_input, "token_count": cleaned_user_input_tokens})
                        mySessionDict['chat_history'].append({"role": "assistant", "content": full_response_content, "token_count": response_tokens})


                        # finally, make a dictionary that will be returned to the client
                        response = {
                            'success': True,
                            'type': 'llm_response',
                            "response": full_response_content,
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
    Reads a Knowledge Base file and returns a list of dictionaries that represent the knowledge base. The file must be in JSON Lines (JSONL) format, containing a list of dictionaries with fields 'id', 'question', and 'answer'.

    Args:
        filepath (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries, where each dictionary represents
              one JSON object (line) from the file.
    """
    @staticmethod
    def read_knowledge_base_file(filepath:str) -> List[str]:
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    # Skip empty lines if any
                    if line.strip():
                        data.append(json.loads(line.strip()))
        except FileNotFoundError:
            logger.error(f"{ColoredText.RED_TEXT}KnowledgeBase.read_knowledge_base_file: The file '{filepath}' did not exist.{ColoredText.END_TEXT}")
        except json.JSONDecodeError as e:
            logger.error(f"{ColoredText.RED_TEXT}KnowledgeBase.read_knowledge_base_file: There was an error decoding JSON. Please ensure each line is a valid JSON object: a list of dictionaries with fields 'id', 'question', and 'answer', with one entry per line. Error on line: {line.strip()}. Error: {e}.{ColoredText.END_TEXT}")
        except Exception as e:
            logger.error(f"{ColoredText.RED_TEXT}KnowledgeBase.read_knowledge_base_file: An unexpected error occurred: {e}.{ColoredText.END_TEXT}")
        return data


    def load_knowledge_base(self, sessionDict: Dict)->int:
        """
        This MUST be called from within a lock on self.session_locks[session_id] OR this needs to be done during construction!

        Load knowledge base

        Returns: the row count.
        """

        logger.info(f"{ColoredText.BLUE_TEXT}\nKnowledgeBase.load_knowledge_base: Populating Vector Database with knowledge base documents...{ColoredText.END_TEXT}")
        for doc_text in self.kbl:
            sessionDict['db'].add_document(doc_text['question'].strip(), doc_text['answer'].strip())

        vector_db_size = len(sessionDict['db'].df)
        logger.info(f"{ColoredText.BLUE_TEXT}KnowledgeBase.load_knowledge_base: Current Vector Database size: {vector_db_size} documents.{ColoredText.END_TEXT}")

        return vector_db_size


    def get_relevant_items_from_db(self, sessionDict: Dict, local_prompt:str, local_min_confidence_score: float, local_max_tokens, local_top_k: int):
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

        logger.info(f"{ColoredText.BLUE_TEXT}KnowledgeBaseStream.get_relevant_items_from_db: Searching Vector database for relevant context for session_id {sessionDict['session_id']}; top_k = {local_top_k}, max_vector_db_tokens = {local_max_tokens} ...{ColoredText.END_TEXT}")

        # Retrieve top K documents based on similarity
        # also, COMPLETELY remove any hidden instructions from the prompt, and then turn the prompt into an embedding
        retrieved_results = sessionDict['db'].search(LlamaUtils.remove_instructions(local_prompt, self.HIDDEN_INSTRUCTION_DELIMITER), False, False, k=local_top_k)  # Get top K relevant documents

        logger.info(f"{ColoredText.BLUE_TEXT}KnowledgeBaseStream.get_relevant_items_from_db: Vector Database search complete for session_id {sessionDict['session_id']} ...{ColoredText.END_TEXT}")

        temp_vdb_token_count = 0

        # Format retrieved context for the GENERATOR LLM
        if retrieved_results:
            for column_header, user_request, user_token_count, assistant_response, assistant_token_count, score in retrieved_results:
                # if the score is acceptable AND the token count will not put us over local_max_tokens
                if (score > local_min_confidence_score) and ((temp_vdb_token_count + user_token_count + assistant_token_count) <= local_max_tokens):
                    temp_vdb_token_count += user_token_count + assistant_token_count

                    retVal.append({"role": "user", "content": KnowledgeBaseStream.HISTORY_REQUEST + user_request})
                    retVal.append({"role": "assistant", "content": KnowledgeBaseStream.HISTORY_RESPONSE + assistant_response})

        else:
            logger.info(f"{ColoredText.YELLOW_TEXT}KnowledgeBaseStream.get_relevant_items_from_db: No chat history found in vector database for session_id {sessionDict['session_id']}.{ColoredText.END_TEXT}")

        return retVal, temp_vdb_token_count


    @staticmethod
    def get_help() -> str:
        retVal = ''
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{KnowledgeBaseStream.SEE_PAST_PREFIX}' to see the chat history that WOULD have been sent to the LLM; note it does not and is just for you to review it.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{KnowledgeBaseStream.THINK_PREFIX}' followed by your prompt to get the LLM to really dig deep in its memory; what this really means is the 'long term' chat history of the vector database will have ample amount of room to try to find the answer from previous conversations. This is useful if you are asking for information that is well outside of the context history window. Note that if the entire chat history fits within the context, the database will not be used (as there is no need, its all there).{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{KnowledgeBaseStream.REASON_PREFIX}' followed by your prompt to let the model reason before it answers, for that one turn - useful for a question that needs several knowledge base entries combined. Only the answer is returned; the deliberation is discarded, and never saved to the chat history. Expect a slower reply, and other sessions wait while it generates. Not to be confused with '{KnowledgeBaseStream.THINK_PREFIX}', which searches the knowledge base more widely but does not change how the model answers; the two can be combined. Spoken sessions never reason.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Type '{KnowledgeBaseStream.VECTOR_TEST_PREFIX}' followed by your prompt tests the vector database; it will show you everything that would have been selected from the vector database. This does not contact the LLM.{ColoredText.END_TEXT}\n"
        retVal += f"{ColoredText.BLUE_TEXT}* Sometimes, you want to send instructions for this round of chat to the LLM, bout you dont want the instructions saved to the vector database _or_ the chat history; in those cases, wrap instructions in the '{KnowledgeBaseStream.HIDDEN_INSTRUCTION_DELIMITER}' delimiter like so: 'Tell me about Artificial intelligence{KnowledgeBaseStream.HIDDEN_INSTRUCTION_DELIMITER} , but please use no more than 50 characters{KnowledgeBaseStream.HIDDEN_INSTRUCTION_DELIMITER}.' This way the instructions will not be saved (so it wont influence future generations).{ColoredText.END_TEXT}\n"

        retVal += f"{ColoredText.BLUE_TEXT}* ...and, finally, type '{KnowledgeBaseStream.HELP_PREFIX}' for this menu again!{ColoredText.END_TEXT}\n"

        return retVal


    @staticmethod
    def get_args_dict() -> dict:
        """
        Gets the args dictionary for the knowledge base server.

        This is a thin wrapper around LlamaUtils.get_args_dict_knowledge_base_server - every entry point in this tree
        shares one set of argument definitions and one set of config loaders, so that adding a setting (or fixing the
        help text on one) only has to happen in a single place. All this class contributes is its own default bind
        address, which it owns because the two servers must not default to the same port.

        Note that the knowledge base half of this is shared verbatim with the interactive knowledge base script, so a
        single knowledge base JSON drives both.

        :return: The merged dictionary of system, knowledge base, and server settings, plus the loaded 'system_message'. Empty if argument parsing failed or '--help' was used.
        """

        return LlamaUtils.get_args_dict_knowledge_base_server(KnowledgeBaseStream.HOST, KnowledgeBaseStream.PORT, logger.info)

#!/usr/bin/env python

import numpy as np
import os
import sys
from llama_cpp import Llama
import getpass
from amadeo_utils.ai.llm.vector_database.VectorDB import (VectorDB)
from amadeo_utils.ai.llm.llama.llama_utils import LlamaUtils
from amadeo_utils.colored_text import ColoredText
from datetime import datetime
import gc
import threading



"""
Ensure you have 'llama-cpp-python', 'pyarrow', and 'fastparquet' installed:
pip install llama-cpp-python pyarrow fastparquet

"""

class RolePlay:

    #####################################################################################################################################################################################################################################################################

    HELP_PREFIX = "!help"
    SAVE_PREFIX = "!save"
    QUIT_PREFIX = "!quit"
    EXIT_PREFIX = "!exit"
    LOAD_PREFIX = "!load"
    THINK_PREFIX = "!think"
    SEE_PAST_PREFIX = "!history"
    STRIKE_PREFIX = "!strike"
    CRYSTAL_BALL_PREFIX = "!crystal"
    VECTOR_TEST_PREFIX = "!vectortest"
    DATETIME_PREFIX = "!date"
    IGNORE_ME_PREFIX = "!ignoreme"
    IGNORE_YOU_PREFIX = "!ignoreyou"
    HIDDEN_INSTRUCTION_DELIMITER = "##"

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

    argsDict = {}
    #llm_embedder
    #llm_generator
    #db
    system_tokens = 0
    max_useable_tokens = 0
    used_tokens = 0
    full_history_fits = True
    chat_history = []

    #####################################################################################################################################################################################################################################################################
    """
    Constructor for RolePlay
    """
    def __init__(self, argsDict: dict):
        self.argsDict = argsDict
        self.used_tokens = 0
        self.full_history_fits = True

        # check to see if both models exist - if not, exit
        if not os.path.exists(self.argsDict['generating_model']):
            print(f"{ColoredText.RED_TEXT}RolePlay: The model [{self.argsDict['generating_model']}] does not exist - exiting.{ColoredText.END_TEXT}")
            sys.exit(0)
        elif not os.path.exists(self.argsDict['embedding_model']):
            print(f"{ColoredText.RED_TEXT}RolePlay: The model [{self.argsDict['embedding_model']}] does not exist - exiting.{ColoredText.END_TEXT}")
            sys.exit(0)

        # Initialize the EMBEDDING model
        self.llm_embedder = Llama(
            model_path=self.argsDict['embedding_model'],
            n_gpu_layers=self.argsDict['embedding_gpu_layers'],
            embedding=True,  # ESSENTIAL for embedding models
            verbose=self.argsDict['debug'],
            n_ctx=self.argsDict['embedding_max_context_tokens'] # Embedding models don't need huge context for individual texts, but set a reasonable one
        )

        print(f"{ColoredText.GREEN_TEXT}RolePlay: Embedding model [{self.argsDict['embedding_model']}] loaded with [{self.argsDict['embedding_gpu_layers']}] GPU layers and a context size of [{self.argsDict['embedding_max_context_tokens']}].{ColoredText.END_TEXT}")

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
        print(f"{ColoredText.GREEN_TEXT}RolePlay: Generative text model [{self.argsDict['generating_model']}] loaded with [{self.argsDict['generating_gpu_layers']}] GPU layers and a context size of [{self.argsDict['generating_max_context_tokens']}].{ColoredText.END_TEXT}")

        self.model_type = self.argsDict['model_type']

        # the locks really are not needed here, as the is the only script running this and there is no threading - however, VectorDB uses locks as other scripts DO call that in a threaded environment, so we need them for that purpose
        self.generating_gpu_lock = threading.Lock()
        self.embedding_gpu_lock = threading.Lock()

        if self.argsDict['encrypted']:
            self.passphrase = getpass.getpass("🔑 Enter conversation passphrase: ")
        else:
            self.passphrase = ""

        # Initialize db
        self.db = VectorDB(self.llm_embedder, self.embedding_gpu_lock, self.llm_generator, self.generating_gpu_lock, self.model_type, self.argsDict['convo_dir'], self.argsDict['debug'], self.passphrase)

        # Get the system tokens
        self.system_tokens = (LlamaUtils.universal_token_count(self.llm_generator, "system", self.argsDict['system_message'], self.model_type))

        self.max_useable_tokens = (1 - self.argsDict['buffer_context_pcnt']) * self.argsDict['generating_max_context_tokens']  # shave a bit off the top to accommodate the buffer


    ################################################################################ User Input ####################################################################################################################

    def handle_user_input(self, user_input: str):
        # determine if the user wants to do anything special
        vector_test, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.VECTOR_TEST_PREFIX)
        think_used, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.THINK_PREFIX)
        crystal_ball, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.CRYSTAL_BALL_PREFIX)
        chat_history_review, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.SEE_PAST_PREFIX)
        date_given, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.DATETIME_PREFIX)
        ignore_user_in_vector_db, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.IGNORE_ME_PREFIX)
        ignore_assistant_in_vector_db, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.IGNORE_YOU_PREFIX)

        # determine the max response tokens, IF it changed
        used_max_response_tokens, user_input = self.adjust_response_tokens(user_input)

        # Add the system tokens and the tokens allotted for the current assistant response
        used_tokens = self.system_tokens + used_max_response_tokens

        # add the date time if requested
        if date_given: user_input = user_input + f" For reference, the datetime is {datetime.now().isoformat(timespec='seconds')}."


        # Now that we have cleared out most of the prompts, we can generate the token count and embedding based off the most recent prompt
        user_input_tokens = (LlamaUtils.universal_token_count(self.llm_generator, "user", LlamaUtils.remove_instruction_delimiters(user_input, RolePlay.HIDDEN_INSTRUCTION_DELIMITER), self.model_type)) # get the token count, minus any instruction delimiter

        # Add the user input tokens, so now we have user input tokens and system message tokens
        used_tokens += user_input_tokens # we save the token count with any hidden instructions


        # IF we wanted a vector test, we are now in a position to do si - so do that now and exit immediately
        if vector_test:
            max_vector_db_tokens = .85 * (self.max_useable_tokens - used_tokens)  # this used to be 'max_vector_database_pcnt * max_useable_tokens', but long system prompts messed with this, so we capture this now, taking into account used_tokens
            temp_top_k = 25  # set this very high to accommodate more returns
            temp_min_vector_db_score = .05

            dumped_items, dumped_tokens = self.get_relevant_items_from_db(user_input, ignore_user_in_vector_db, ignore_assistant_in_vector_db, temp_min_vector_db_score, max_vector_db_tokens, temp_top_k, True)
            return


        # Construct messages list for GENERATOR LLM, including system message, context, and chat history
        # Initialize messages_for_llm with the system message
        messages_for_llm = [{"role": "system", "content": self.argsDict['system_message']}]

        if self.full_history_fits:
            total_history_tokens = sum(d['token_count'] for d in chat_history)
            # if the total tokens for the history is less than the max context, just use the entire history
            if used_tokens + total_history_tokens <= self.max_useable_tokens:
                # remove 'token_count'
                formatted_chat_history = [
                    {'role': d['role'], 'content': d['content']}
                    for d in chat_history
                ]
                messages_for_llm.extend(formatted_chat_history)

                used_tokens += total_history_tokens

                # If we wish to see the chat history, print it
                if chat_history_review:
                    for item in chat_history:
                        print(f"{ColoredText.YELLOW_TEXT}role: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{item['role']} {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}token count: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{item['token_count']} {ColoredText.END_TEXT}"
                              f"{ColoredText.YELLOW_TEXT}content: {ColoredText.END_TEXT}{ColoredText.CYAN_TEXT}{item['content']}{ColoredText.END_TEXT}")
            else:
                self.full_history_fits = False

        # if the full history does not fit, we must use the vector database
        if not self.full_history_fits:
            # Search the vector database for relevant context


            # we need to set some things depending on if the user wants the LLM to 'really think'
            if think_used:
                print(f"{ColoredText.CYAN_TEXT}Going far back in memory...{ColoredText.END_TEXT}")
                max_vector_db_tokens = .85 * (self.max_useable_tokens - used_tokens) # this used to be 'max_vector_database_pcnt * max_useable_tokens', but long system prompts messed with this, so we capture this now, taking into account used_tokens
                temp_top_k = 25 # set this very high to accommodate more returns
                temp_min_vector_db_score = .05

            else:
                # normal run
                max_vector_db_tokens = self.argsDict['max_vector_database_pcnt'] * (self.max_useable_tokens - used_tokens) # this used to be 'max_vector_database_pcnt * max_useable_tokens', but long system prompts messed with this, so we capture this now, taking into account used_tokens
                temp_top_k = self.argsDict['top_k']
                temp_min_vector_db_score = self.argsDict['min_vector_db_score']


            # determine if there were relevant items from the vector DB
            db_items, db_tokens = self.get_relevant_items_from_db(user_input, ignore_user_in_vector_db, ignore_assistant_in_vector_db, temp_min_vector_db_score, max_vector_db_tokens, temp_top_k, chat_history_review)

            # if there were DB items
            if db_items:
                messages_for_llm.extend(db_items)

                # add in the token count from the vector db results
                used_tokens += db_tokens


            # Finally, add on the chat history - used_tokens is now the sum of the new user request, the system message, the preemptive assistant response, and the vector db entries
            abridged_chat_history, abridged_chat_history_tokens = LlamaUtils.fit_to_token_limit(chat_history, self.max_useable_tokens - used_tokens)

            # Add in the abridged chat history tokens
            used_tokens += abridged_chat_history_tokens

            # If we wish to see the chat history, print it
            if chat_history_review:
                for item in abridged_chat_history:
                    print(f"{ColoredText.YELLOW_TEXT}role: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{item['role']} {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}token count: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{item['token_count']} {ColoredText.END_TEXT}"
                          f"{ColoredText.YELLOW_TEXT}content: {ColoredText.END_TEXT}{ColoredText.CYAN_TEXT}{item['content']}{ColoredText.END_TEXT}")

            # remove 'token_count'
            formatted_chat_history = [
                {'role': d['role'], 'content': d['content']}
                for d in abridged_chat_history
            ]

            # store in messages_for_llm
            messages_for_llm.extend(formatted_chat_history)


        # Finally, append the most recent content; remember to remove any instruction delimiters if they exist (but leave the instructions intact)
        messages_for_llm.append({"role": "user", "content": LlamaUtils.remove_instruction_delimiters(user_input, RolePlay.HIDDEN_INSTRUCTION_DELIMITER)})

        if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}Your final message to the model: {LlamaUtils.remove_instruction_delimiters(user_input, RolePlay.HIDDEN_INSTRUCTION_DELIMITER)}{ColoredText.END_TEXT}")

        if not chat_history_review:
            try:
                # Generate response from the GENERATOR LLM
                print("\n", end="", flush=True)

                # if the user has a name, kame that a stop point - otherwise do not.
                # This is important, as sometimes the LLM goes off the rails and tries to speak for you - so you need to stop that

                if self.argsDict['player_name']:
                    local_stop = ["[INST]", "<|im_end|>", "<|start_header_id|>", "User:", "Assistant:", self.argsDict['player_name'] + ":"]
                else:
                    local_stop = ["[INST]", "<|im_end|>", "<|start_header_id|>", "User:", "Assistant:"]


                if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}Sending to the LLM generator... used_tokens: {used_tokens} generating_max_context_tokens: {self.argsDict['generating_max_context_tokens']} used_max_response_tokens: {used_max_response_tokens} generating_gpu_layers: {self.argsDict['generating_gpu_layers']} {ColoredText.END_TEXT}")

                response_stream = self.llm_generator.create_chat_completion( # Using llm_generator here
                    messages=messages_for_llm,
                    max_tokens=used_max_response_tokens,
                    stream=True,
                    stop=local_stop
                )

                if crystal_ball: print(f"{ColoredText.BLUE_TEXT}You peer into the future and see what WOULD have happened!{ColoredText.END_TEXT}")

                full_response_content = ""
                for chunk in response_stream:
                    if "content" in chunk["choices"][0]["delta"]:
                        content = chunk["choices"][0]["delta"]["content"]
                        print(content, end="", flush=True)
                        full_response_content += content
                print() # Ensure a new line after the streamed response

                # if there was a response AND we didnt look into the crystal ball (i.e. we want to save this interaction), continue
                if full_response_content.strip() and not crystal_ball:
                    full_response_content = full_response_content.strip()
                    response_tokens = (LlamaUtils.universal_token_count(self.llm_generator, "assistant", full_response_content, self.model_type)) # get the token count for the assistant response

                    # We want to use the version of the input that does not have any hidden instructions (marked by the delimiter)
                    cleaned_user_input = LlamaUtils.remove_instructions(user_input, RolePlay.HIDDEN_INSTRUCTION_DELIMITER)
                    cleaned_user_input_tokens = (LlamaUtils.universal_token_count(self.llm_generator, "user", cleaned_user_input, self.model_type)) # get the token count, minus any instructions. This will be stored to the vector database

                    # The user request and assistant response were initially separate, having the request and response stored separately; however, Gemini said it would be best if they were combined, both for the embedding AND the text
                    # Gemini also said we want to use VECTOR_DB_USER_REQUEST and VECTOR_DB_AGENT_RESPONSE in the embedding - as it would help it - but when we retrieve it, we want to split on VECTOR_DB_AGENT_RESPONSE and then remove VECTOR_DB_USER_REQUEST, and put them both into the chat history separately explained to me that
                    # Do not forget that we want to completely remove any hidden instructions before we save to the database
                    self.db.add_document(cleaned_user_input, full_response_content)

                    if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}Stored combined user request and assistant response in long-term memory.{ColoredText.END_TEXT}")

                    # Update chat history with user input and assistant response for future turns
                    chat_history.append({"role": "user", "content": cleaned_user_input, "token_count": cleaned_user_input_tokens})
                    chat_history.append({"role": "assistant", "content": full_response_content, "token_count": response_tokens})
                elif crystal_ball:
                    print(f"{ColoredText.BLUE_TEXT}You can pretend that didn't just happen. I hope your peer into the future was worth it - it cost us a Morty!{ColoredText.END_TEXT}")
                else:
                    print(f"{ColoredText.GREEN_TEXT}LLM goofed and didn't return a proper response - please try again.{ColoredText.END_TEXT}")


            except Exception as e:
                print(f"{ColoredText.RED_TEXT}Uncaught exception when attempting to generate text: [{e}].{ColoredText.END_TEXT}")
        else:
            # We simply wanted to see the chat history
            print(f"{ColoredText.BLUE_TEXT}Chat history end.{ColoredText.END_TEXT}")


    def cleanup(self):
        # releases LLMs from memory
        del self.llm_embedder
        del self.llm_generator

        gc.collect()  # Force garbage collection

    """
    Adjusts the response tokens, as necessary; It returns the new response token count; in addition, it changes the prompt to request the response to use up to the token count, no more. 
    It uses the hidden instruction delimiter so it wont show in the chat log  
    """
    def adjust_response_tokens(self, local_text: str)->(int,str):

        #### #figure out if we want to override the base of max_response_tokens
        override_tokens = 0
        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlay.VERYSHORT_PREFIX)
        if max_response_override: override_tokens = 32

        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlay.SHORT_PREFIX)
        if max_response_override: override_tokens = 64

        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlay.MEDIUM_PREFIX)
        if max_response_override: override_tokens = 128

        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlay.NORMAL_PREFIX)
        if max_response_override: override_tokens = 256

        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlay.LONG_PREFIX)
        if max_response_override: override_tokens = 512

        max_response_override, local_text = LlamaUtils.report_and_remove_keyword(local_text, RolePlay.VERYLONG_PREFIX)
        if max_response_override: override_tokens = 1024

        # if this was never set - or it was set to self.argsDict['max_response_tokens'] - take the default
        if override_tokens == 0 or (override_tokens == self.argsDict['max_response_tokens']):
            override_tokens = self.argsDict['max_response_tokens']
        else:
            local_text = f"{local_text}{RolePlay.HIDDEN_INSTRUCTION_DELIMITER} Please use up to {override_tokens} tokens in your response; try to fill the entire token count, if it makes sense; do not mention the change in tokens or change in response pattern.{RolePlay.HIDDEN_INSTRUCTION_DELIMITER}"

        return override_tokens, local_text

    def get_relevant_items_from_db(self, local_prompt:str, ignore_user_in_vector_db: bool, ignore_assistant_in_vector_db: bool, local_min_confidence_score: float, local_max_tokens, local_top_k: int, print_lines: bool):

        retVal = []
        if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}RolePlay.get_relevant_items_from_db: Searching Vector database for relevant context; top_k = {local_top_k}, max_vector_db_tokens = {local_max_tokens} ...{ColoredText.END_TEXT}")

        # Retrieve top K documents based on similarity
        # also, COMPLETELY remove any hidden instructions from the prompt, and then turn the prompt into an embedding
        retrieved_results = self.db.search(LlamaUtils.remove_instructions(local_prompt, self.HIDDEN_INSTRUCTION_DELIMITER), ignore_user_in_vector_db, ignore_assistant_in_vector_db, k=local_top_k)  # Get top K relevant documents

        if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}RolePlay.get_relevant_items_from_db: Vector Database search complete...{ColoredText.END_TEXT}")

        temp_vdb_token_count = 0

        # Format retrieved context for the GENERATOR LLM
        if retrieved_results:
            for column_header, user_request, user_token_count, assistant_response, assistant_token_count, score in retrieved_results:
                # if the score is acceptable AND the token count will not put us over local_max_tokens
                if (score > local_min_confidence_score) and ((temp_vdb_token_count + user_token_count + assistant_token_count) <= local_max_tokens):
                    temp_vdb_token_count += user_token_count + assistant_token_count

                    retVal.append({"role": "user", "content": RolePlay.HISTORY_REQUEST + user_request})
                    retVal.append({"role": "assistant", "content": RolePlay.HISTORY_RESPONSE + assistant_response})
                    if print_lines: print(
                        f"{ColoredText.YELLOW_TEXT}Retrieved From Column: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{column_header} {ColoredText.END_TEXT}"
                        f"{ColoredText.YELLOW_TEXT}role: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}user {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}token count: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{user_token_count} {ColoredText.END_TEXT}"
                        f"{ColoredText.YELLOW_TEXT}score: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{score} {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}content: {ColoredText.END_TEXT}{ColoredText.MAGENTA_TEXT}{user_request}{ColoredText.END_TEXT}")
                    if print_lines: print(
                        f"{ColoredText.YELLOW_TEXT}role: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}assistant {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}token count: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{assistant_token_count} {ColoredText.END_TEXT}"
                        f"{ColoredText.YELLOW_TEXT}score: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{score} {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}content: {ColoredText.END_TEXT}{ColoredText.MAGENTA_TEXT}{assistant_response}{ColoredText.END_TEXT}")

        else:
            if self.argsDict['debug'] or print_lines: print(f"{ColoredText.YELLOW_TEXT}RolePlay.get_relevant_items_from_db: No chat history found in vector database.{ColoredText.END_TEXT}")

        return retVal, temp_vdb_token_count




    def strike_from_record(self):
        self.db.strike_last_from_record()

        # strike from local chat history
        if len(self.chat_history) >= 2:  # Check if there are at least two items to remove
            last_response = self.chat_history.pop()  # Removes last response from LLM
            last_request = self.chat_history.pop()  # Removes last request from you

            if self.argsDict['debug']:
                print(
                    f"{ColoredText.BLUE_TEXT}Removed previous pair from conversation history - USER: {last_request['content']} \nASSISTANT: {last_response['content']}\n{ColoredText.END_TEXT}")
            else:
                print(f"{ColoredText.BLUE_TEXT}Removed previous pair from conversation history.{ColoredText.END_TEXT}")
        else:
            print(
                f"{ColoredText.BLUE_TEXT}Chat history not long enough to stroke last conversation.{ColoredText.END_TEXT}")

    ################################################################################ Save and Load ####################################################################################################################
    def save(self, local_chat_history):
        self.db.save_session(local_chat_history)

    """
    (Re)Load chat history
    """
    def load_chat_history(self, load_previous: bool)->list:
        self.chat_history = []
        static_response_tokens = (LlamaUtils.universal_token_count(self.llm_generator, "assistant", VectorDB.ASSISTANT_RESPONSE, self.model_type))

        if os.path.exists(self.argsDict['convo_dir']):
            # if we know we want to re-load, just do it - otherwise, ask
            if load_previous: load_option = 'y'
            else : load_option = input(f"Load previous session from '{self.argsDict['convo_dir']}'? (y/n): ").strip().lower()
            if load_option == 'y':
                self.chat_history = self.db.load_session()  # db.df will be updated internally by load_session

                # Re-add initial knowledge base documents if they are not already in the loaded DB.
                # This ensures they are always present, even if a partial DB was saved/loaded.
                # A more robust check might involve comparing document hashes or IDs.
                # For simplicity, we just add them again here; duplicates will exist if already loaded,
                # but for small datasets and demonstration, this is acceptable.
                if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}\nRolePlay.load_chat_history: Ensuring initial knowledge base documents are present in DB...{ColoredText.END_TEXT}")
                initial_kb_texts = [doc.strip() for doc in RolePlay.INITIAL_KNOWLEDGE_BASE_DOCUMENTS]
                current_db_texts = set(self.db.df['user_text'].apply(lambda x: x.strip()))  # Strip to normalize for comparison

                docs_to_add = []
                responses_to_add = []

                for doc_text in RolePlay.INITIAL_KNOWLEDGE_BASE_DOCUMENTS:
                    if doc_text.strip() not in current_db_texts:

                        # create a string that will correctly store this as a request / response (the LLM expects this)
                        docs_to_add.append(doc_text.strip())
                        responses_to_add.append(VectorDB.ASSISTANT_RESPONSE)

                if docs_to_add:
                    self.db.add_documents(docs_to_add, responses_to_add)
                    if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}RolePlay.load_chat_history: Added {len(docs_to_add)} missing initial knowledge base documents.{ColoredText.END_TEXT}")
                else:
                    if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}RolePlay.load_chat_history: All initial knowledge base documents already present or DB was loaded fully.{ColoredText.END_TEXT}")

            else:
                if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}\nRolePlay.load_chat_history: Populating Vector Database with initial knowledge base documents (new session)...{ColoredText.END_TEXT}")
                for doc_text in RolePlay.INITIAL_KNOWLEDGE_BASE_DOCUMENTS:
                    self.db.add_document(doc_text.strip(), VectorDB.ASSISTANT_RESPONSE)

                if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}RolePlay.load_chat_history: Vector Database initially populated with {len(RolePlay.INITIAL_KNOWLEDGE_BASE_DOCUMENTS)} documents.{ColoredText.END_TEXT}")

        if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}RolePlay.load_chat_history: Current Vector Database size: {len(self.db.df)} documents.{ColoredText.END_TEXT}")

        return self.chat_history


    @staticmethod
    def print_help():
        print(f"{ColoredText.BLUE_TEXT}* Type '{RolePlay.EXIT_PREFIX}' or '{RolePlay.QUIT_PREFIX}' to end the conversation.{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Type '{RolePlay.SAVE_PREFIX}' to save the current session.{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Type '{RolePlay.LOAD_PREFIX}' to reload the last saved session (this will clear current unsaved progress).{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Type '{RolePlay.SEE_PAST_PREFIX}' to see the chat history that WOULD have been sent to the LLM; note it does not and is just for you to review it.{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Type '{RolePlay.STRIKE_PREFIX}' to remove the last chat request/response from the chat history and vector database.{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Type '{RolePlay.CRYSTAL_BALL_PREFIX}' followed by your prompt to see what the LLM would say to a zany question or comment; the request nor response are saved in the chat history, so after the LLM initially responds, it will be like you never asked the question. Careful, though, it does consume a Morty!{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Type '{RolePlay.THINK_PREFIX}' followed by your prompt to get the LLM to really dig deep in its memory; what this really means is the 'long term' chat history of the vector database will have ample amount of room to try to find the answer from previous conversations. This is useful if you are asking for information that is well outside of the context history window. Note that if the entire chat history fits within the context, the database will not be used (as there is no need, its all there).{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Type '{RolePlay.DATETIME_PREFIX}' to print the current datetime in a line (something like 'For reference, the datetime is YYYY-MM-DD HH:II:SS); useful for tracking dates.{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Type '{RolePlay.VECTOR_TEST_PREFIX}' followed by your prompt tests the vector database; it will show you everything that would have been selected from the vector database. This does not contact the LLM.{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Type '{RolePlay.IGNORE_ME_PREFIX}' followed by your prompt alters the items found matching in the vector database by ignoring the user input / p[rompt / request via replacing it with a generic 'Update me.'; useful if you want to save on tokens while having the assistant use its previous responses (good for having it remember parts of a roleplay world it created).{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Type '{RolePlay.IGNORE_YOU_PREFIX}' followed by your prompt alters the items found matching in the vector database by ignoring the assistant response via replacing it with a generic 'Fascinating.'; useful if you want the assistant to focus on what you said and ignore its response (for example, if you use it for journaling). Also useful if the assistant is chatty and fills responses with nonsense filler or questions, and you want it to focus on what you said.{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Type '{RolePlay.VERYSHORT_PREFIX}', '{RolePlay.SHORT_PREFIX}', '{RolePlay.MEDIUM_PREFIX}', '{RolePlay.NORMAL_PREFIX}', '{RolePlay.LONG_PREFIX}', or '{RolePlay.VERYLONG_PREFIX}'  followed by your prompt to temporarily set the max-response-tokens. Values: '{RolePlay.VERYSHORT_PREFIX}'=32, '{RolePlay.SHORT_PREFIX}'=64, '{RolePlay.MEDIUM_PREFIX}'=128, '{RolePlay.NORMAL_PREFIX}'=256, '{RolePlay.LONG_PREFIX}'=512, '{RolePlay.VERYLONG_PREFIX}'=1024. Omit these to use the default.{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Sometimes, you want to send instructions for this round of chat to the LLM, bout you dont want the instructions saved to the vector database _or_ the chat history; in those cases, wrap instructions in the '{RolePlay.HIDDEN_INSTRUCTION_DELIMITER}' delimiter like so: 'Tell me about Artificial intelligence{RolePlay.HIDDEN_INSTRUCTION_DELIMITER} , but please use no more than 50 characters{RolePlay.HIDDEN_INSTRUCTION_DELIMITER}.' This way the instructions will not be saved (so it wont influence future generations).{ColoredText.END_TEXT}")

        print(f"{ColoredText.BLUE_TEXT}* ...and, finally, type '{RolePlay.HELP_PREFIX}' for this menu again!{ColoredText.END_TEXT}")


def splitAndSaveUserText(someText: str):
    if VectorDB.VECTOR_DB_USER_REQUEST in someText and VectorDB.VECTOR_DB_AGENT_RESPONSE in someText:
        # The request and response was combined, so we must uncombine it and take out the stuff we added
        parts = someText.split(VectorDB.VECTOR_DB_AGENT_RESPONSE, 1)
        user_part = parts[0].replace(VectorDB.VECTOR_DB_USER_REQUEST, "").strip()
        assistant_part = parts[1].strip()
        return user_part
    else:
        return someText

def splitAndSaveAssistantText(someText: str):

    if VectorDB.VECTOR_DB_USER_REQUEST in someText and VectorDB.VECTOR_DB_AGENT_RESPONSE in someText:
        # The request and response was combined, so we must uncombine it and take out the stuff we added
        parts = someText.split(VectorDB.VECTOR_DB_AGENT_RESPONSE, 1)
        user_part = parts[0].replace(VectorDB.VECTOR_DB_USER_REQUEST, "").strip()
        assistant_part = parts[1].strip()
        return assistant_part
    else:
        return "I See."


def getUserTokens(someText: str, model_type: str = "llama-3"):
    return LlamaUtils.universal_token_count(rp.llm_generator, "user", someText, model_type)

def getAssistantTokens(someText: str, model_type: str = "llama-3"):
    return LlamaUtils.universal_token_count(rp.llm_generator, "assistant", someText, model_type)

def getEmbedding(someText: str):
    return np.array(rp.llm_embedder.embed(f"{someText}"), dtype=np.float32)


# --- Main Chatbot Logic ---
if __name__ == "__main__":

    # get the arguments dictionary
    argsDict = LlamaUtils.get_args_dict()

    # if the dictionary was not passed back, just exit
    if not argsDict:
        sys.exit(0)

    # set rp
    rp = RolePlay(argsDict)


    # load the chat history, if it exists - if not, start anew
    chat_history = rp.load_chat_history(False)

    print(f"{ColoredText.BLUE_TEXT}\n--- Starting Interactive Chatbot ---{ColoredText.END_TEXT}")
    rp.print_help()
    print(f"\n{ColoredText.CYAN_TEXT}System message: {ColoredText.END_TEXT}{ColoredText.BLUE_TEXT}{argsDict['system_message']}\n{ColoredText.END_TEXT}")



    """
    # THIS WAS A SPOT WHERE I WAS CHANGING DATAFRAMES - I left it in in case I have to do it again 
    
    #del rp.db.df['token_count']
    #del rp.db.df['text']
    
    print(f"\n{ColoredText.RED_TEXT}DataFram Processing...{ColoredText.END_TEXT}")
    rp.db.df['user_text'] = rp.db.df['text'].apply(splitAndSaveUserText)
    rp.db.df['user_embedding'] = rp.db.df['user_text'].apply(getEmbedding)
    rp.db.df['user_token_count'] = rp.db.df['user_text'].apply(getUserTokens)

    rp.db.df['assistant_text'] = rp.db.df['text'].apply(splitAndSaveAssistantText)
    rp.db.df['assistant_embedding'] = rp.db.df['assistant_text'].apply(getEmbedding)
    rp.db.df['assistant_token_count'] = rp.db.df['assistant_text'].apply(getAssistantTokens)
    
    rp.db.printDF()
    print(f"\n{ColoredText.RED_TEXT}DataFram Processing complete{ColoredText.END_TEXT}")
    """


    while True:
        user_input = input("\n>>: ").strip()

        if user_input.lower().strip() in [RolePlay.EXIT_PREFIX, RolePlay.QUIT_PREFIX]:
            print(f"{ColoredText.BLUE_TEXT}Exiting....{ColoredText.END_TEXT}")
            break
        elif user_input.lower().strip() == RolePlay.SAVE_PREFIX:
            rp.save(chat_history)
            continue # Skip to next loop iteration
        elif user_input.lower().strip() == RolePlay.HELP_PREFIX:
            rp.print_help()
            continue  # Skip to next loop iteration
        elif user_input.lower().strip() == RolePlay.STRIKE_PREFIX:
            # strike it from the vector database
            rp.strike_from_record()

            continue
        elif user_input.lower().strip() == RolePlay.LOAD_PREFIX:
            print(f"{ColoredText.BLUE_TEXT}Loading from previous save...{ColoredText.END_TEXT}")
            chat_history = rp.load_chat_history(True)

            continue # Skip to next loop iteration


        rp.handle_user_input(user_input)


    # END WHILE

    rp.cleanup()



"""
    except FileNotFoundError:
        print(f"\nError: One or both model files not found. Please check paths:")
        print(f"Embedding model: '{embedding_model}'")
        print(f"Generative model: '{generating_model}'")
        print("Ensure you have downloaded the correct .gguf files for each role.")
    except ValueError as ve:
        print(f"\nConfiguration Error: {ve}")
        print("This usually means your embedding model is not producing fixed-size embeddings,")
        print("or there's a mismatch between loaded and new embeddings.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during Llama model initialization or chat loop: {e}")
        print("Ensure 'llama-cpp-python', 'pyarrow', 'fastparquet' are installed,")
        print("and your models are compatible and correctly specified. Also check n_ctx values.")
"""



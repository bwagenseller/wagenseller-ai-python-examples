import numpy as np
import os
import sys

# Line editing for the '>>:' prompt. Importing readline is all it takes: input() then supports the arrow keys, Home/End,
# Ctrl-A/Ctrl-E and word jumps for fixing a typo mid-line, and Up/Down to recall this session's earlier inputs. Without
# it the arrow keys just insert escape codes such as '^[[D'. The history stays in memory only - nothing typed is written
# to disk. Optional, because some Python builds lack the module; the script works the same without it.
try:
    import readline  # noqa: F401 - imported for its side effect on input()
except ImportError:
    pass

# IMPORTANT: llama_utils MUST be imported before llama_cpp. Importing llama_cpp loads the llama.cpp shared library,
# which registers its GGML CUDA backend and pins the device ordering for the life of the process; llama_utils sets
# CUDA_DEVICE_ORDER at import time so that '--gpu N' means the Nth card as 'nvidia-smi -L' lists it. Flip these two
# lines and the GPU selection silently reverts to CUDA's own 'fastest first' ordering.
from amadeo_utils.ai.llm.llama.llama_utils import LlamaUtils
from amadeo_utils.ai.llm.llama import chat_template as ChatTemplate
from llama_cpp import Llama

from amadeo_utils.ai.llm.vector_database.VectorDB import VectorDB
from amadeo_utils.colored_text import ColoredText
import json
import gc
import threading



"""
Dependencies: see the 'llm' extra in pyproject.toml, which holds the exact pins and the install command. In short,
llama-cpp-python must come from the prebuilt CUDA wheel with '--only-binary' - a plain 'pip install llama-cpp-python'
can silently fall back to a CPU-only source build - and pyarrow is the only parquet engine needed (fastparquet is not).

"""

class KnowledgeBase:

    #####################################################################################################################################################################################################################################################################

    HELP_PREFIX = "!help"
    QUIT_PREFIX = "!quit"
    EXIT_PREFIX = "!exit"
    THINK_PREFIX = "!think"
    REASON_PREFIX = "!reason"
    SEE_PAST_PREFIX = "!history"
    VECTOR_TEST_PREFIX = "!vectortest"
    HIDDEN_INSTRUCTION_DELIMITER = "##"

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
    chat_history = []

    #####################################################################################################################################################################################################################################################################
    """
    Constructor for KnowledgeBase
    """
    def __init__(self, argsDict: dict):
        self.argsDict = argsDict
        self.used_tokens = 0

        self.model_type = self.argsDict['model_type']

        # check to see if both models exist - if not, exit
        if not os.path.exists(self.argsDict['generating_model']):
            print(f"{ColoredText.RED_TEXT}KnowledgeBase: The model [{self.argsDict['generating_model']}] does not exist - exiting.{ColoredText.END_TEXT}")
            sys.exit(0)
        elif not os.path.exists(self.argsDict['embedding_model']):
            print(f"{ColoredText.RED_TEXT}KnowledgeBase: The model [{self.argsDict['embedding_model']}] does not exist - exiting.{ColoredText.END_TEXT}")
            sys.exit(0)
        elif not os.path.exists(self.argsDict['knowledge_base_file']):
            print(f"{ColoredText.RED_TEXT}KnowledgeBase: The knowledge base file [{self.argsDict['knowledge_base_file']}] does not exist - exiting.{ColoredText.END_TEXT}")
            sys.exit(0)


        # Work out which card each model goes on. The embedding model is tiny, so it is always pinned to the single
        # nominated GPU even when the generative model is being spread across all of them - splitting a 100 MB model
        # would buy nothing and would put its tensors on a card the generator wants for its own layers.
        gpu_index = self.argsDict.get('gpu', LlamaUtils.GPU_INDEX)
        embedder_gpu_kwargs = LlamaUtils.build_gpu_kwargs(gpu_index, False, 'embedding')
        generator_gpu_kwargs = LlamaUtils.build_gpu_kwargs(gpu_index, self.argsDict.get('split_gpus', False), 'generative')
        # Flash attention and KV cache precision for the generative model only; the embedder is tiny and keeps the
        # library defaults. See LlamaUtils.build_context_kwargs for why these exist and what they cost.
        generator_context_kwargs = LlamaUtils.build_context_kwargs(self.argsDict.get('flash_attn', LlamaUtils.FLASH_ATTN), self.argsDict.get('kv_cache_type', LlamaUtils.KV_CACHE_TYPE))

        # try
        # Initialize the EMBEDDING model
        self.llm_embedder = Llama(
            model_path=self.argsDict['embedding_model'],
            n_gpu_layers=self.argsDict['embedding_gpu_layers'],
            embedding=True,  # ESSENTIAL for embedding models
            verbose=self.argsDict['debug'],
            n_ctx=self.argsDict['embedding_max_context_tokens'], # Embedding models don't need huge context for individual texts, but set a reasonable one
            **embedder_gpu_kwargs
        )

        print(f"{ColoredText.GREEN_TEXT}KnowledgeBase: Embedding model [{self.argsDict['embedding_model']}] loaded with [{self.argsDict['embedding_gpu_layers']}] GPU layers and a context size of [{self.argsDict['embedding_max_context_tokens']}].{ColoredText.END_TEXT}")

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
            **generator_gpu_kwargs,
            **generator_context_kwargs
        )
        print(f"{ColoredText.GREEN_TEXT}KnowledgeBase: Generative text model [{self.argsDict['generating_model']}] loaded with [{self.argsDict['generating_gpu_layers']}] GPU layers and a context size of [{self.argsDict['generating_max_context_tokens']}].{ColoredText.END_TEXT}")

        # Take over prompt formatting from llama-cpp-python so that reasoning can be switched off.
        #
        # The library builds its own formatter per model and freezes it, and 'create_chat_completion()' has no
        # '**kwargs', so there is no supported way to reach a template variable like 'enable_thinking' through it.
        # Installing our own handler keeps the model's embedded template - the prompt format is unchanged - while
        # letting the reasoning variables be reassigned between turns. Reasoning starts OFF: left on, a model such as
        # Qwen 3.6 spends the whole response budget deliberating and never reaches an answer. The REASON_PREFIX command
        # turns it back on for a single turn.
        #
        # A GGUF without an embedded template has nothing to install, in which case we leave the library's own handling
        # alone and simply have no reasoning control.
        self.thinking_supported = False
        try:
            ChatTemplate.install_chat_handler(self.llm_generator, thinking=False)
            self.thinking_supported = ChatTemplate.supports_thinking(self.llm_generator)
            architecture = ChatTemplate.model_architecture(self.llm_generator)
            print(f"{ColoredText.GREEN_TEXT}KnowledgeBase: Using the model's embedded chat template (architecture [{architecture}]); reasoning is {'available and currently suppressed' if self.thinking_supported else 'not applicable to this model'}.{ColoredText.END_TEXT}")
        except ValueError as e:
            print(f"{ColoredText.YELLOW_TEXT}KnowledgeBase: {e} Falling back to llama-cpp-python's own prompt formatting.{ColoredText.END_TEXT}")

        # Stop strings that end an assistant turn for THIS architecture. Merged with the conversational stops at
        # generation time rather than replacing them.
        self.architecture_stops = ChatTemplate.stop_tokens(self.llm_generator)

        # the locks really are not needed here, as the is the only script running this and there is no threading - however, VectorDB uses locks as other scripts DO call that in a threaded environment, so we need them for that purpose
        self.generating_gpu_lock = threading.Lock()
        self.embedding_gpu_lock = threading.Lock()

        # Initialize db
        self.db = VectorDB(self.llm_embedder, self.embedding_gpu_lock, self.llm_generator, self.generating_gpu_lock, self.model_type, '', self.argsDict['debug'])

        # Get the system tokens
        self.system_tokens = (LlamaUtils.universal_token_count(self.llm_generator, "system", self.argsDict['system_message'], self.model_type))

        self.max_useable_tokens = (1 - self.argsDict['buffer_context_pcnt']) * self.argsDict['generating_max_context_tokens']  # shave a bit off the top to accommodate the buffer


    ################################################################################ User Input ####################################################################################################################

    def handle_user_input(self, user_input: str):
        # determine if the user wants to do anything special
        vector_test, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.VECTOR_TEST_PREFIX)
        think_used, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.THINK_PREFIX)
        reason_used, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.REASON_PREFIX)
        chat_history_review, user_input = LlamaUtils.report_and_remove_keyword(user_input, self.SEE_PAST_PREFIX)

        # determine the max response tokens, IF it changed
        used_max_response_tokens = self.argsDict['max_response_tokens']

        # A model's deliberation comes out of the same max_tokens as its answer: room for a '!reason' turn to think
        # ('reasoning_budget_tokens'), or for a model that thinks even when told not to (Muse Glimmer's
        # 'suppressed_reasoning_tokens'). Without it, Muse Glimmer can spend the whole budget on hidden deliberation
        # and never reach an answer.
        used_max_response_tokens += LlamaUtils.turn_token_allowance(reason_used, self.thinking_supported, self.system_tokens, used_max_response_tokens, self.max_useable_tokens, self.argsDict)

        # Add the system tokens and the tokens allotted for the current assistant response
        used_tokens = self.system_tokens + used_max_response_tokens

        # Now that we have cleared out most of the prompts, we can generate the token count and embedding based off the most recent prompt
        user_input_tokens = (LlamaUtils.universal_token_count(self.llm_generator, "user", LlamaUtils.remove_instruction_delimiters(user_input, KnowledgeBase.HIDDEN_INSTRUCTION_DELIMITER), self.model_type)) # get the token count, minus any instruction delimiter

        # Add the user input tokens, so now we have user input tokens and system message tokens
        used_tokens += user_input_tokens # we save the token count with any hidden instructions


        # IF we wanted a vector test, we are now in a position to do si - so do that now and exit immediately
        if vector_test:
            max_vector_db_tokens = .85 * (self.max_useable_tokens - used_tokens)  # this used to be 'max_vector_database_pcnt * max_useable_tokens', but long system prompts messed with this, so we capture this now, taking into account used_tokens
            temp_top_k = 25  # set this very high to accommodate more returns
            temp_min_vector_db_score = .05

            dumped_items, dumped_tokens = self.get_relevant_items_from_db(user_input, False, False, temp_min_vector_db_score, max_vector_db_tokens, temp_top_k, True)
            return


        # Construct messages list for GENERATOR LLM, including system message, context, and chat history
        # Initialize messages_for_llm with the system message
        messages_for_llm = [{"role": "system", "content": self.argsDict['system_message']}]

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
        db_items, db_tokens = self.get_relevant_items_from_db(user_input, False, False, temp_min_vector_db_score, max_vector_db_tokens, temp_top_k, chat_history_review)

        # if there were DB items
        if db_items:
            messages_for_llm.extend(db_items)

            # add in the token count from the vector db results
            used_tokens += db_tokens


        # Finally, add on the chat history - used_tokens is now the sum of the new user request, the system message, the preemptive assistant response, and the vector db entries
        abridged_chat_history, abridged_chat_history_tokens = LlamaUtils.fit_to_token_limit(self.chat_history, self.max_useable_tokens - used_tokens)

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
        messages_for_llm.append({"role": "user", "content": LlamaUtils.remove_instruction_delimiters(user_input, KnowledgeBase.HIDDEN_INSTRUCTION_DELIMITER)})

        if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}Your final message to the model: {LlamaUtils.remove_instruction_delimiters(user_input, KnowledgeBase.HIDDEN_INSTRUCTION_DELIMITER)}{ColoredText.END_TEXT}")

        if not chat_history_review:
            try:
                # Generate response from the GENERATOR LLM
                print("\n", end="", flush=True)

                local_stop = ["[INST]", "<|im_end|>", "<|start_header_id|>", "User:", "Assistant:"]

                # The hard-coded stops above cover ChatML and Llama-3 only. Gemma 4 ends a turn with '<turn|>' and
                # Muse Glimmer with '<|eot|>', neither of which appears above, so add this architecture's own.
                local_stop = local_stop + [stop for stop in self.architecture_stops if stop not in local_stop]

                # For a model with a reasoning mode, the conversational stops ('User:', 'Assistant:') must not reach
                # llama.cpp: they would fire inside the model's reasoning notes, where such markers are common, and end
                # the turn before any answer exists. They are enforced on the answer instead - live by the display
                # filter below, and on the stored text after stripping. Other models are unaffected.
                generation_stop, answer_stop = ChatTemplate.split_stops(self.llm_generator, local_stop)

                # Reveal the model's reasoning for this turn only, if asked for.
                ChatTemplate.set_thinking(self.llm_generator, reason_used)
                if reason_used and self.thinking_supported:
                    print(f"{ColoredText.BLUE_TEXT}Reasoning is enabled for this turn; the deliberation below is shown but will not be saved.{ColoredText.END_TEXT}")

                if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}Sending to the LLM generator... used_tokens: {used_tokens} generating_max_context_tokens: {self.argsDict['generating_max_context_tokens']} used_max_response_tokens: {used_max_response_tokens} generating_gpu_layers: {self.argsDict['generating_gpu_layers']} {ColoredText.END_TEXT}")

                response_stream = self.llm_generator.create_chat_completion( # Using llm_generator here
                    messages=messages_for_llm,
                    max_tokens=used_max_response_tokens,
                    stream=True,
                    stop=generation_stop
                )

                # Filter what reaches the screen, not what is kept. Some models deliberate no matter what they are told
                # (Muse Glimmer's reasoning control is only a 'Reasoning strength' instruction), and without this their
                # deliberation and protocol headers would stream live on every turn. On a '!reason' turn the filter
                # shows the deliberation, then the reply, with the markers removed.
                display_filter = ChatTemplate.ReasoningStreamFilter(self.llm_generator, thinking=reason_used, answer_stops=answer_stop)

                full_response_content = ""
                for chunk in response_stream:
                    if "content" in chunk["choices"][0]["delta"]:
                        content = chunk["choices"][0]["delta"]["content"]
                        print(display_filter.feed(content), end="", flush=True)
                        full_response_content += content
                        if display_filter.finished:
                            break   # an answer stop was reached; leaving the loop ends generation, as a stop string would
                print(display_filter.flush()) # release anything held back, and end the line

                # Strip any reasoning before the response goes any further, so that the deliberation never reaches the
                # chat history, where it would be fed back to the model as though it were part of the conversation.
                # It costs nothing when there is nothing to remove.
                full_response_content = ChatTemplate.strip_reasoning(full_response_content, self.llm_generator, thinking=reason_used)
                full_response_content = ChatTemplate.truncate_at_stops(full_response_content, answer_stop)

                # if there was a response, continue
                if full_response_content.strip():
                    full_response_content = full_response_content.strip()
                    response_tokens = (LlamaUtils.universal_token_count(self.llm_generator, "assistant", full_response_content, self.model_type)) # get the token count for the assistant response

                    # We want to use the version of the input that does not have any hidden instructions (marked by the delimiter)
                    cleaned_user_input = LlamaUtils.remove_instructions(user_input, KnowledgeBase.HIDDEN_INSTRUCTION_DELIMITER)
                    cleaned_user_input_tokens = (LlamaUtils.universal_token_count(self.llm_generator, "user", cleaned_user_input, self.model_type)) # get the token count, minus any instructions. This will be stored to the vector database

                    # Update chat history with user input and assistant response for future turns
                    self.chat_history.append({"role": "user", "content": cleaned_user_input, "token_count": cleaned_user_input_tokens})
                    self.chat_history.append({"role": "assistant", "content": full_response_content, "token_count": response_tokens})
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


    def get_relevant_items_from_db(self, local_prompt:str, ignore_user_in_vector_db: bool, ignore_assistant_in_vector_db: bool, local_min_confidence_score: float, local_max_tokens, local_top_k: int, print_lines: bool):

        retVal = []
        if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}KnowledgeBase.get_relevant_items_from_db: Searching Vector database for relevant context; top_k = {local_top_k}, max_vector_db_tokens = {local_max_tokens} ...{ColoredText.END_TEXT}")

        # Retrieve top K documents based on similarity
        # also, COMPLETELY remove any hidden instructions from the prompt, and then turn the prompt into an embedding
        retrieved_results = self.db.search(LlamaUtils.remove_instructions(local_prompt, self.HIDDEN_INSTRUCTION_DELIMITER), ignore_user_in_vector_db, ignore_assistant_in_vector_db, k=local_top_k)  # Get top K relevant documents

        if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}KnowledgeBase.get_relevant_items_from_db: Vector Database search complete...{ColoredText.END_TEXT}")

        temp_vdb_token_count = 0

        # Format retrieved context for the GENERATOR LLM
        if retrieved_results:
            for column_header, user_request, user_token_count, assistant_response, assistant_token_count, score in retrieved_results:
                # if the score is acceptable AND the token count will not put us over local_max_tokens
                if (score > local_min_confidence_score) and ((temp_vdb_token_count + user_token_count + assistant_token_count) <= local_max_tokens):
                    temp_vdb_token_count += user_token_count + assistant_token_count

                    retVal.append({"role": "user", "content": KnowledgeBase.HISTORY_REQUEST + user_request})
                    retVal.append({"role": "assistant", "content": KnowledgeBase.HISTORY_RESPONSE + assistant_response})
                    if print_lines: print(
                        f"{ColoredText.YELLOW_TEXT}Retrieved From Column: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{column_header} {ColoredText.END_TEXT}"
                        f"{ColoredText.YELLOW_TEXT}role: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}user {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}token count: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{user_token_count} {ColoredText.END_TEXT}"
                        f"{ColoredText.YELLOW_TEXT}score: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{score} {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}content: {ColoredText.END_TEXT}{ColoredText.MAGENTA_TEXT}{user_request}{ColoredText.END_TEXT}")
                    if print_lines: print(
                        f"{ColoredText.YELLOW_TEXT}role: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}assistant {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}token count: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{assistant_token_count} {ColoredText.END_TEXT}"
                        f"{ColoredText.YELLOW_TEXT}score: {ColoredText.END_TEXT}{ColoredText.GREEN_TEXT}{score} {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}content: {ColoredText.END_TEXT}{ColoredText.MAGENTA_TEXT}{assistant_response}{ColoredText.END_TEXT}")

        else:
            if self.argsDict['debug'] or print_lines: print(f"{ColoredText.YELLOW_TEXT}KnowledgeBase.get_relevant_items_from_db: No chat history found in vector database.{ColoredText.END_TEXT}")

        return retVal, temp_vdb_token_count



    ################################################################################ Save and Load ####################################################################################################################
    """
    Reads a Knowledge Base file and returns a list of dictionaries that represent the knowledge base. The file must be in JSON Lines (JSONL) format, containing a list of dictionaries with fields 'id', 'question', and 'answer'.

    Args:
        filepath (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries, where each dictionary represents
              one JSON object (line) from the file.
    """

    def read_knowledge_base_file(self, filepath):
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    # Skip empty lines if any
                    if line.strip():
                        data.append(json.loads(line.strip()))
        except FileNotFoundError:
            print(f"{ColoredText.RED_TEXT}KnowledgeBase.read_knowledge_base_file: The file '{filepath}' did not exist.{ColoredText.END_TEXT}")
        except json.JSONDecodeError as e:
            print(f"{ColoredText.RED_TEXT}KnowledgeBase.read_knowledge_base_file: There was an error decoding JSON. Please ensure each line is a valid JSON object: a list of dictionaries with fields 'id', 'question', and 'answer', with one entry per line. Error on line: {line.strip()}. Error: {e}.{ColoredText.END_TEXT}")
        except Exception as e:
            print(f"{ColoredText.RED_TEXT}KnowledgeBase.read_knowledge_base_file: An unexpected error occurred: {e}.{ColoredText.END_TEXT}")
        return data


    """
    Load knowledge base
    
    Returns: the row count.
    """
    def load_knowledge_base(self)->int:

        kbf = self.read_knowledge_base_file(self.argsDict['knowledge_base_file'])

        if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}\nKnowledgeBase.load_knowledge_base: Populating Vector Database with knowledge base documents...{ColoredText.END_TEXT}")
        for doc_text in kbf:
            self.db.add_document(doc_text['question'].strip(), doc_text['answer'].strip())

        if self.argsDict['debug']: print(f"{ColoredText.BLUE_TEXT}KnowledgeBase.load_knowledge_base: Current Vector Database size: {len(self.db.df)} documents.{ColoredText.END_TEXT}")

        return len(self.db.df)


    @staticmethod
    def print_help():
        print(f"{ColoredText.BLUE_TEXT}* Type '{KnowledgeBase.EXIT_PREFIX}' or '{KnowledgeBase.QUIT_PREFIX}' to end the conversation.{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Type '{KnowledgeBase.SEE_PAST_PREFIX}' to see the chat history that WOULD have been sent to the LLM; note it does not and is just for you to review it.{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Type '{KnowledgeBase.THINK_PREFIX}' followed by your prompt to get the LLM to really dig deep in its memory; what this really means is the 'long term' chat history of the vector database will have ample amount of room to try to find the answer from previous conversations. This is useful if you are asking for information that is well outside of the context history window. Note that if the entire chat history fits within the context, the database will not be used (as there is no need, its all there).{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Type '{KnowledgeBase.REASON_PREFIX}' followed by your prompt to let the model reason out loud for that one turn, which can help with a question that needs several knowledge base entries combined. Reasoning is normally suppressed, as it costs time and context to generate text you never see. The deliberation is printed but is never saved to the chat history. Not to be confused with '{KnowledgeBase.THINK_PREFIX}', which searches the knowledge base more widely but does not change how the model answers; the two can be combined.{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Type '{KnowledgeBase.VECTOR_TEST_PREFIX}' followed by your prompt tests the vector database; it will show you everything that would have been selected from the vector database. This does not contact the LLM.{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}* Sometimes, you want to send instructions for this round of chat to the LLM, bout you dont want the instructions saved to the vector database _or_ the chat history; in those cases, wrap instructions in the '{KnowledgeBase.HIDDEN_INSTRUCTION_DELIMITER}' delimiter like so: 'Tell me about Artificial intelligence{KnowledgeBase.HIDDEN_INSTRUCTION_DELIMITER} , but please use no more than 50 characters{KnowledgeBase.HIDDEN_INSTRUCTION_DELIMITER}.' This way the instructions will not be saved (so it wont influence future generations).{ColoredText.END_TEXT}")

        print(f"{ColoredText.BLUE_TEXT}* ...and, finally, type '{KnowledgeBase.HELP_PREFIX}' for this menu again!{ColoredText.END_TEXT}")


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


def getUserTokens(someText: str, model_type: str = "llama3"):
    return (LlamaUtils.universal_token_count(rp.llm_generator, "user", someText, model_type))

def getAssistantTokens(someText: str, model_type: str = "llama3"):
    return (LlamaUtils.universal_token_count(rp.llm_generator, "assistant", someText, model_type))

def getEmbedding(someText: str):
    return np.array(rp.llm_embedder.embed(f"{someText}"), dtype=np.float32)


# --- Main Chatbot Logic ---
if __name__ == "__main__":

    # get the arguments dictionary
    argsDict = LlamaUtils.get_args_dict_knowledge_base()

    # if the dictionary was not passed back, just exit
    if not argsDict:
        sys.exit(0)

    # set rp
    rp = KnowledgeBase(argsDict)


    # load the chat history, if it exists - if not, start anew
    knowledge_base_entries = rp.load_knowledge_base()

    if knowledge_base_entries == 0:
        print(f"{ColoredText.RED_TEXT}No knowledge base entries - exiting.{ColoredText.END_TEXT}")
        sys.exit(0)

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

        if user_input.lower().strip() in [KnowledgeBase.EXIT_PREFIX, KnowledgeBase.QUIT_PREFIX]:
            print(f"{ColoredText.BLUE_TEXT}Exiting....{ColoredText.END_TEXT}")
            break
        elif user_input.lower().strip() == KnowledgeBase.HELP_PREFIX:
            rp.print_help()
            continue  # Skip to next loop iteration


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
        print("Ensure the 'llm' extra from pyproject.toml is installed (llama-cpp-python from its CUDA wheel, plus pyarrow),")
        print("and your models are compatible and correctly specified. Also check n_ctx values.")
"""



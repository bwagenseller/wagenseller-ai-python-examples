import re
import argparse
import json
import os
from amadeo_utils.ai.llm.llama.subjective_constants import SubjectiveConstants
from amadeo_utils.colored_text import ColoredText
from typing import Callable, Optional


class LlamaUtils:

    ################################################################################################################### Constants ####################################################################################################################


    CONVO_NAME = "my_chat"

    MODEL_TYPE = "llama-3"
    CHAT_FORMAT = None

    BASE_SYSTEM_MESSAGE = "You are Ada, a helpful AI automatron."

    GPU_LAYERS = 57  # Adjust based on your GPU availability for generator
    EMBEDDING_GPU_LAYERS = -1  # Adjust based on your GPU availability for embedder
    MAX_CTX = 4096  #
    EMBEDDING_MAX_CTX = 512
    MAX_RESPONSE_TOKENS = 512

    REPEAT_PENALITY = 1.1

    MAX_VECTOR_DB_PCNT = .2  # A number from 0 to 1; it represents the percentage of the share of the overall chat history that is occupied by items from the vector database. Note that if the entire chat history fits into n_ctx the vector database will not be used
    BUFFER_CTX_PCNT = .05  # A number from 0 to 1; it represents the percentage of the share of the overall context tokens we want to use as a 'buffer'; since We cant fully guess the number of tokens in the chat history we send to the LLM, we approximate as best we can. This number is a buffer to help ensure that we do not hit this limit, as the LLM WILL fail if we do.

    TOP_K = 3  # the number of results returned by the vector database 'search'
    MIN_VECTOR_DB_SCORE = .46

    # THREAD_COUNT = 8 #-1 means 'use all cores', 0 means default (Llama finds the number of threads to half of the number of CPU cores)


    ################################################################################################################### Loading System Prompt ####################################################################################################################

    def load_system_prompt(filepath: str, log_func: Callable[[str], None] = print):
        """
        Loads the entire content of a file into a single string.

        Args:
            filepath (str): The path to the file to be loaded.
            log_func (Callable): The function that prints whatever we are targeting. This will usually be 'print' or some form of 'logger.info'. 'logger'error' etc.

        Returns:
            str: The entire content of the file as a single string. Returns an empty string if the file is empty.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            IOError: For other input/output errors (e.g., permissions).
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            log_func(f"Error: The file '{filepath}' was not found; system prompt could not be loaded!")
            raise  # Re-raise the exception after printing
        except IOError as e:
            log_func(f"Error reading file '{filepath}'; system prompt could not be loaded: {e}")
            raise  # Re-raise the exception after printing

    ################################################################################################################### Parsing Arguments From Command Line ####################################################################################################################

    @staticmethod
    def get_args_dict(log_func: Callable[[str], None] = print) -> dict:
        """
        Gets args dictionary for a traditional vector database, meant to save the conversation for later.

        :param log_func: The function that prints whatever we are targeting. This will usually be 'print' or some form of 'logger.info'. 'logger'error' etc.
        :return:
        """

        parser = argparse.ArgumentParser(description='Run a LLM, as you see fit.')
        parser.add_argument("-bmd", "--base-model-dir", default=SubjectiveConstants.BASE_MODEL_DIR,help="The location of the base model that will generate text.")
        parser.add_argument("-bed", "--base-embedding-dir", default=SubjectiveConstants.BASE_EMBEDDING_DIR,help="The location of the embedding model.")
        parser.add_argument("-m", "--model", default=SubjectiveConstants.MODEL,help="The filename of your text generation model. Please, just the filename, not the directory.")
        parser.add_argument("-mt", "--model-type", default=LlamaUtils.MODEL_TYPE,help="The model type: ['llama-2', 'llama-3', 'alpaca', 'qwen', 'command-r', 'vicuna', 'oasst_llama', 'baichuan-2', 'baichuan', 'openbuddy', 'redpajama-incite', 'snoozy', 'phind', 'intel', 'open-orca', 'mistrallite', 'zephyr', 'pygmalion', 'chatml', 'mistral-instruct', 'chatglm3', 'openchat', 'saiga', 'gemma', 'functionary', 'functionary-v2', 'functionary-v1', 'chatml-function-calling'].")
        parser.add_argument("-cf", "--chat-format", default=LlamaUtils.CHAT_FORMAT,help="The chat format type; you should usually leave this None and let the system figure it out (with the exception of 'command-r' - in that case, use 'llama-3'). Values: ['llama-2', 'llama-3', 'alpaca', 'qwen', 'vicuna', 'oasst_llama', 'baichuan-2', 'baichuan', 'openbuddy', 'redpajama-incite', 'snoozy', 'phind', 'intel', 'open-orca', 'mistrallite', 'zephyr', 'pygmalion', 'chatml', 'mistral-instruct', 'chatglm3', 'openchat', 'saiga', 'gemma', 'functionary', 'functionary-v2', 'functionary-v1', 'chatml-function-calling'].")
        parser.add_argument("-em", "--embedding-model", default=SubjectiveConstants.EMBEDDING_MODEL,help="The filename of your embedding model. Please, just the filename, not the directory.")

        parser.add_argument("-pn", "--player-name", default=SubjectiveConstants.BASE_PLAYER_NAME,help="Give your username - What did you say your name was in the prompt")
        parser.add_argument("-bcd", "--base-convo-dir", default=SubjectiveConstants.BASE_CONVO_DIR,help="The base directory where all conversation histories are stored.")
        parser.add_argument("-cn", "--convo-name", default=LlamaUtils.CONVO_NAME,help="What do you want this chat session to be named?")
        parser.add_argument("-obcd", "--override-base-convo-dir", action='store_true',help="Do you want to use the override base directory? This is a second directory that is immutable and is meant for temporary conversation storage")
        parser.add_argument("-spf", "--system-prompt-file", default=SubjectiveConstants.SYSTEM_PROMPT_FILE,help="A filename (full, absolute path to the file) that contains your system prompt; this is a text file. This is the initial message you send to the LLM to 'set the tone' of the entire conversation. This is critical! Be creative!")

        parser.add_argument("-gl", "--gpu-layers", type=int, default=LlamaUtils.GPU_LAYERS,help="How many GPU layers do you want for the text generator model? -1 means try to get them all, but be warned: if the GPU layers are too high, the model will not fit in VRAM this will fail. This number is usually between 10 and 70, IF -1 does not work.")
        parser.add_argument("-egl", "--embedding-gpu-layers", type=int, default=LlamaUtils.EMBEDDING_GPU_LAYERS,help="Similar to --gpu-layers but for the embedding model (see that description). You will usually want -1 for this. If -1 doesn't work, you can try some value between 10 and 100, but if -1 doesnt work, you probably have much bigger problems as embedding models are very small.")
        parser.add_argument("-mct", "--max-context-tokens", type=int, default=LlamaUtils.MAX_CTX,help="Known as 'n_ctx' in the llama binary, this is the max token count for the entire conversation, including vector database retrieval, chat history, current prompt, and LLM response. This should usually be a power of 2, and common choices are 512, 2048, 4096, or 8192, but it can go higher. Llama 3 models typically have n_ctx of 8192 or more. Just know that the space this tames is n_ctx^2 and it does count against your VRAM.")
        parser.add_argument("-emct", "--embedding-max-context-tokens", type=int, default=LlamaUtils.EMBEDDING_MAX_CTX,help="Similar to '--max-context-tokens' (see that description), but for the embedding model. This does not have to be too big as it only has to accommodate either one request or response; 512 is usually more than enough for this.")
        parser.add_argument("-mrt", "--max-response-tokens", type=int, default=LlamaUtils.MAX_RESPONSE_TOKENS,help="The maximum number of response tokens the LLM will use in its response.")

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
                    config_dict = LlamaUtils.load_json_config(json_config_file)

                    argDict['generating_model'] = os.path.join(config_dict['base_model_dir'], config_dict['model'])
                    argDict['embedding_model'] = os.path.join(config_dict['base_embedding_dir'], config_dict['embedding_model'])

                    if config_dict['override_base_convo_dir']:
                        argDict['convo_dir'] = os.path.join(SubjectiveConstants.BASE_CONVO_DIR_OVERRIDE, config_dict['convo_name'])
                    else:
                        argDict['convo_dir'] = os.path.join(config_dict['base_convo_dir'], config_dict['convo_name'])

                    argDict['system_prompt_file'] = config_dict['system_prompt_file']

                    argDict['generating_gpu_layers'] = config_dict['gpu_layers']
                    argDict['embedding_gpu_layers'] = config_dict['embedding_gpu_layers']
                    argDict['generating_max_context_tokens'] = config_dict['max_context_tokens']
                    argDict['embedding_max_context_tokens'] = config_dict['embedding_max_context_tokens']
                    argDict['max_response_tokens'] = config_dict['max_response_tokens']  # Maximum tokens the generative model is allowed to generate

                    argDict['max_vector_database_pcnt'] = config_dict['max_vector_database_pcnt']
                    argDict['buffer_context_pcnt'] = config_dict['buffer_context_pcnt']

                    argDict['top_k'] = config_dict['top_k']
                    argDict['min_vector_db_score'] = config_dict['min_vector_db_score']

                    argDict['model_type'] = config_dict.get('model_type', LlamaUtils.MODEL_TYPE)
                    argDict['player_name'] = config_dict.get('player_name', SubjectiveConstants.BASE_PLAYER_NAME)
                    argDict['encrypted'] = config_dict.get('encrypted', False)
                    argDict['debug'] = config_dict.get('debug', False)
                    argDict['chat_format'] = config_dict.get('chat_format', LlamaUtils.CHAT_FORMAT)


                    log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_args_dict: Config loaded from JSON file {json_config_file}; prompt is from '{argDict['system_prompt_file']}' and convo directory is '{argDict['convo_dir']}'.{ColoredText.END_TEXT}")
                    use_default_arg_config = False


                except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
                    log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict: Could not load JSON config [{json_config_file}] - there are errors. Will attempt to load other defaults or args. Error: {e}.{ColoredText.END_TEXT}")


            elif json_config_file:
                log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict: Could not load JSON config [{json_config_file}] - file does not exist. Loading from defaults or other parameters sent.{ColoredText.END_TEXT}")

            if use_default_arg_config:

                argDict['generating_model'] = os.path.join(args.base_model_dir, args.model)
                argDict['embedding_model'] = os.path.join(args.base_embedding_dir, args.embedding_model)

                if args.override_base_convo_dir:
                    argDict['convo_dir'] = os.path.join(SubjectiveConstants.BASE_CONVO_DIR_OVERRIDE, args.convo_name)
                else:
                    argDict['convo_dir'] = os.path.join(args.base_convo_dir, args.convo_name)

                argDict['system_prompt_file'] = args.system_prompt_file

                argDict['generating_gpu_layers'] = args.gpu_layers
                argDict['embedding_gpu_layers'] = args.embedding_gpu_layers
                argDict['generating_max_context_tokens'] = args.max_context_tokens
                argDict['embedding_max_context_tokens'] = args.embedding_max_context_tokens
                argDict['max_response_tokens'] = args.max_response_tokens  # Maximum tokens the generative model is allowed to generate

                argDict['max_vector_database_pcnt'] = args.max_vector_database_pcnt
                argDict['buffer_context_pcnt'] = args.buffer_context_pcnt

                argDict['top_k'] = args.top_k
                argDict['min_vector_db_score'] = args.min_vector_db_score

                argDict['player_name'] = args.player_name
                argDict['debug'] = args.debug
                argDict['encrypted'] = args.encrypted
                argDict['model_type'] = args.model_type
                argDict['chat_format'] = args.chat_format

                log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_args_dict: Config loaded from args / defaults; prompt is from '{argDict['system_prompt_file']}' and convo directory is '{argDict['convo_dir']}'.{ColoredText.END_TEXT}")

            argDict['system_message'] = LlamaUtils.get_system_message(argDict['system_prompt_file'])

        except SystemExit as e:
            argDict = {}
            if e.code == 0:
                # --help was used, so print no error
                print(f"{ColoredText.BLUE_TEXT}Thank you!{ColoredText.END_TEXT}")
            else:
                log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict: Invalid arguments.{ColoredText.END_TEXT}")

        return argDict


    @staticmethod
    def get_args_dict_knowledge_base(log_func: Callable[[str], None] = print) -> dict:
        """
        Gets args dictionary for a knowledge base - this is not meant to be a continuous chatbot, just a simple 'answer these questions' without caring too much about the conversation.

        :param log_func: The function that prints whatever we are targeting. This will usually be 'print' or some form of 'logger.info'. 'logger'error' etc.
        :return:

        """

        parser = argparse.ArgumentParser(description='Run a LLM, as you see fit.')
        parser.add_argument("-bmd", "--base-model-dir", default=SubjectiveConstants.BASE_MODEL_DIR,help="The location of the base model that will generate text.")
        parser.add_argument("-bed", "--base-embedding-dir", default=SubjectiveConstants.BASE_EMBEDDING_DIR,help="The location of the embedding model.")
        parser.add_argument("-m", "--model", default=SubjectiveConstants.MODEL,help="The filename of your text generation model. Please, just the filename, not the directory.")
        parser.add_argument("-mt", "--model-type", default=LlamaUtils.MODEL_TYPE,help="The model type: ['llama-2', 'llama-3', 'alpaca', 'qwen', 'command-r', 'vicuna', 'oasst_llama', 'baichuan-2', 'baichuan', 'openbuddy', 'redpajama-incite', 'snoozy', 'phind', 'intel', 'open-orca', 'mistrallite', 'zephyr', 'pygmalion', 'chatml', 'mistral-instruct', 'chatglm3', 'openchat', 'saiga', 'gemma', 'functionary', 'functionary-v2', 'functionary-v1', 'chatml-function-calling'].")
        parser.add_argument("-cf", "--chat-format", default=LlamaUtils.CHAT_FORMAT,help="The chat format type; you should usually leave this None and let the system figure it out (with the exception of 'command-r' - in that case, use 'llama-3'). Values: ['llama-2', 'llama-3', 'alpaca', 'qwen', 'vicuna', 'oasst_llama', 'baichuan-2', 'baichuan', 'openbuddy', 'redpajama-incite', 'snoozy', 'phind', 'intel', 'open-orca', 'mistrallite', 'zephyr', 'pygmalion', 'chatml', 'mistral-instruct', 'chatglm3', 'openchat', 'saiga', 'gemma', 'functionary', 'functionary-v2', 'functionary-v1', 'chatml-function-calling'].")
        parser.add_argument("-em", "--embedding-model", default=SubjectiveConstants.EMBEDDING_MODEL,help="The filename of your embedding model. Please, just the filename, not the directory.")

        parser.add_argument("-kbf", "--knowledge-base-file", type=str, default="", help="The JSON Lines (JSONL) file that contains your knowledge base. This must be in JSONL format - a list of dictionaries with fields 'id', 'question', and 'answer'.")
        parser.add_argument("-spf", "--system-prompt-file", default=SubjectiveConstants.SYSTEM_PROMPT_FILE,help="A filename (full, absolute path to the file) that contains your system prompt; this is a text file. This is the initial message you send to the LLM to 'set the tone' of the entire conversation. This is critical! Be creative!")

        parser.add_argument("-gl", "--gpu-layers", type=int, default=LlamaUtils.GPU_LAYERS,help="How many GPU layers do you want for the text generator model? -1 means try to get them all, but be warned: if the GPU layers are too high, the model will not fit in VRAM this will fail. This number is usually between 10 and 70, IF -1 does not work.")
        parser.add_argument("-egl", "--embedding-gpu-layers", type=int, default=LlamaUtils.EMBEDDING_GPU_LAYERS,help="Similar to --gpu-layers but for the embedding model (see that description). You will usually want -1 for this. If -1 doesn't work, you can try some value between 10 and 100, but if -1 doesnt work, you probably have much bigger problems as embedding models are very small.")
        parser.add_argument("-mct", "--max-context-tokens", type=int, default=LlamaUtils.MAX_CTX,help="Known as 'n_ctx' in the llama binary, this is the max token count for the entire conversation, including vector database retrieval, chat history, current prompt, and LLM response. This should usually be a power of 2, and common choices are 512, 2048, 4096, or 8192, but it can go higher. Llama 3 models typically have n_ctx of 8192 or more. Just know that the space this tames is n_ctx^2 and it does count against your VRAM.")
        parser.add_argument("-emct", "--embedding-max-context-tokens", type=int, default=LlamaUtils.EMBEDDING_MAX_CTX,help="Similar to '--max-context-tokens' (see that description), but for the embedding model. This does not have to be too big as it only has to accommodate either one request or response; 512 is usually more than enough for this.")
        parser.add_argument("-mrt", "--max-response-tokens", type=int, default=LlamaUtils.MAX_RESPONSE_TOKENS,help="The maximum number of response tokens the LLM will use in its response.")

        parser.add_argument("-mvdbp", "--max-vector-database-pcnt", type=float, default=LlamaUtils.MAX_VECTOR_DB_PCNT,help="A number from 0 to 1; it represents the percentage of the share of the overall chat history that is occupied by items from the vector database. Note that if the entire chat history fits into max-context-tokens (n_ctx) the vector database will not be used ")
        parser.add_argument("-bcp", "--buffer-context-pcnt", type=float, default=LlamaUtils.BUFFER_CTX_PCNT,help="A number from 0 to 1; it represents the percentage of the share of the overall context tokens we want to use as a 'buffer'; since We cant fully guess the number of tokens in the chat history we send to the LLM, we approximate as best we can. This number is a buffer to help ensure that we do not hit this limit, as the LLM WILL fail if we do.")

        parser.add_argument("-tk", "--top-k", type=int, default=LlamaUtils.TOP_K, help="The number of results returned by the vector database 'search'.")
        parser.add_argument("-mvdbs", "--min-vector-db-score", type=float, default=LlamaUtils.MIN_VECTOR_DB_SCORE, help="Every match from the vector database has a confidence score from 0 to 1; this indicates the minimum score you wish to have in the results from the vector database.")

        parser.add_argument("-d", "--debug", action='store_true',help="Do you want to see some additional log lines while using the LLM?")
        parser.add_argument("-j", "--json", type=str, default="", help="If this points to a valid JSON file, the ENTIRE parameter settings are pulled from that file, and the defaults - and other arguments passed from the command line - are ignored. If the JSON load fails for whatever reason, though, the defaults WILL be engaged.")

        argDict = {}

        try:
            args = parser.parse_args()
            use_default_arg_config = True  # This is only flipped if we successfully load from a JSON file

            json_config_file = args.json

            if json_config_file and os.path.exists(json_config_file):
                try:
                    config_dict = LlamaUtils.load_json_config_knowledge_base(json_config_file)

                    argDict['generating_model'] = os.path.join(config_dict['base_model_dir'], config_dict['model'])
                    argDict['embedding_model'] = os.path.join(config_dict['base_embedding_dir'], config_dict['embedding_model'])

                    argDict['knowledge_base_file'] = config_dict['knowledge_base_file']
                    argDict['system_prompt_file'] = config_dict['system_prompt_file']

                    argDict['generating_gpu_layers'] = config_dict['gpu_layers']
                    argDict['embedding_gpu_layers'] = config_dict['embedding_gpu_layers']
                    argDict['generating_max_context_tokens'] = config_dict['max_context_tokens']
                    argDict['embedding_max_context_tokens'] = config_dict['embedding_max_context_tokens']
                    argDict['max_response_tokens'] = config_dict['max_response_tokens']  # Maximum tokens the generative model is allowed to generate

                    argDict['max_vector_database_pcnt'] = config_dict['max_vector_database_pcnt']
                    argDict['buffer_context_pcnt'] = config_dict['buffer_context_pcnt']

                    argDict['top_k'] = config_dict['top_k']
                    argDict['min_vector_db_score'] = config_dict['min_vector_db_score']

                    argDict['model_type'] = config_dict.get('model_type', LlamaUtils.MODEL_TYPE)
                    argDict['debug'] = config_dict.get('debug', False)
                    argDict['chat_format'] = config_dict.get('chat_format', LlamaUtils.CHAT_FORMAT)

                    log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_args_dict_knowledge_base: Config loaded from JSON file {json_config_file}; prompt is from '{argDict['system_prompt_file']}'.{ColoredText.END_TEXT}")
                    use_default_arg_config = False


                except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
                    log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict_knowledge_base: Could not load JSON config [{json_config_file}] - there are errors. Will attempt to load other defaults or args. Error: {e}.{ColoredText.END_TEXT}")


            elif json_config_file:
                log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict_knowledge_base: Could not load JSON config [{json_config_file}] - file does not exist. Loading from defaults or other parameters sent.{ColoredText.END_TEXT}")

            if use_default_arg_config:

                argDict['generating_model'] = os.path.join(args.base_model_dir, args.model)
                argDict['embedding_model'] = os.path.join(args.base_embedding_dir, args.embedding_model)

                argDict['knowledge_base_file'] = args.knowledge_base_file
                argDict['system_prompt_file'] = args.system_prompt_file

                argDict['generating_gpu_layers'] = args.gpu_layers
                argDict['embedding_gpu_layers'] = args.embedding_gpu_layers
                argDict['generating_max_context_tokens'] = args.max_context_tokens
                argDict['embedding_max_context_tokens'] = args.embedding_max_context_tokens
                argDict['max_response_tokens'] = args.max_response_tokens  # Maximum tokens the generative model is allowed to generate

                argDict['max_vector_database_pcnt'] = args.max_vector_database_pcnt
                argDict['buffer_context_pcnt'] = args.buffer_context_pcnt

                argDict['top_k'] = args.top_k
                argDict['min_vector_db_score'] = args.min_vector_db_score

                argDict['debug'] = args.debug
                argDict['model_type'] = args.model_type
                argDict['chat_format'] = args.chat_format

                log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_args_dict_knowledge_base: Config loaded from args / defaults; prompt is from '{argDict['system_prompt_file']}'.{ColoredText.END_TEXT}")

            argDict['system_message'] = LlamaUtils.get_system_message(argDict['system_prompt_file'])

        except SystemExit as e:
            argDict = {}
            if e.code == 0:
                # --help was used, so print no error
                log_func(f"{ColoredText.BLUE_TEXT}Thank you!{ColoredText.END_TEXT}")
            else:
                log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict_knowledge_base: Invalid arguments.{ColoredText.END_TEXT}")

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
                "base_model_dir": "/home/kevin/ai/models/llama.cpp",
                "base_embedding_dir": "/home/kevin/ai/models/llama.cpp/embedding_models",
                "model": "llama-3-70b.Q4_K_M.gguf",
                "model_type": "llama3",
                "embedding_model": "nomic-embed-text-v1.5.Q5_K_M.gguf",

                "player_name": "Kevin",
                "base_convo_dir": "/home/kevin/ai/chat_history",
                "convo_name": "my_chat",
                "override_base_convo_dir": true,
                "system_prompt_file": "/home/kevin/ai/system_prompt.txt",

                "gpu_layers": 57,
                "embedding_gpu_layers": -1,
                "max_context_tokens": 4096,
                "embedding_max_context_tokens": 512,
                "max_response_tokens": 512,

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
            'base_model_dir': str,
            'base_embedding_dir': str,
            'model': str,
            'embedding_model': str,

            'base_convo_dir': str,
            'convo_name': str,
            'override_base_convo_dir': bool,
            "system_prompt_file": str,

            'gpu_layers': int,
            'embedding_gpu_layers': int,
            'max_context_tokens': int,
            'embedding_max_context_tokens': int,
            'max_response_tokens': int,

            'max_vector_database_pcnt': float,
            'buffer_context_pcnt': float,

            'top_k': int,
            'min_vector_db_score': float
        }

        # Add optional fields with their types
        optional_fields = {
            'model_type': str,
            'chat_format': Optional[str],
            'player_name': str,
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


    """
    Loads a JSON file for a Knowledge Base and scrapes specific entries into a dictionary.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        dict: A dictionary containing the scraped configuration fields. All fields are required. An example of a JSON doc:
        {   
            "base_model_dir": "/home/kevin/ai/models/llama.cpp",
            "base_embedding_dir": "/home/kevin/ai/models/llama.cpp/embedding_models",
            "model": "llama-3-70b.Q4_K_M.gguf",
            "model_type": "llama3", 
            "embedding_model": "nomic-embed-text-v1.5.Q5_K_M.gguf",

            "knowledge_base_file": "/home/kevin/ai/knowledge_base.jsonl",        
            "system_prompt_file": "/home/kevin/ai/system_prompt.txt",

            "gpu_layers": 57,
            "embedding_gpu_layers": -1,
            "max_context_tokens": 4096,
            "embedding_max_context_tokens": 512,
            "max_response_tokens": 512,

            "max_vector_database_pcnt": 0.2,
            "buffer_context_pcnt": 0.05,

            "top_k": 4,
            "min_vector_db_score": 0.46,

            "debug": false
        }

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file content is not valid JSON.
        KeyError: If any of the required fields are missing from the JSON.
        TypeError: If a field's value is not of the expected type.
    """

    @staticmethod
    def load_json_config_knowledge_base(filepath: str) -> dict:
        """
        Loads a JSON file and scrapes specific entries into a dictionary.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict: A dictionary containing the scraped configuration fields:
                  'model_path' (str), 'n_ctx' (int), 'debug' (bool), and 'min_confidence' (float).

        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file content is not valid JSON.
            KeyError: If any of the required fields are missing from the JSON.
            TypeError: If a field's value is not of the expected type.
        """
        required_fields = {
            'base_model_dir': str,
            'base_embedding_dir': str,
            'model': str,
            'embedding_model': str,

            'knowledge_base_file': str,
            "system_prompt_file": str,

            'gpu_layers': int,
            'embedding_gpu_layers': int,
            'max_context_tokens': int,
            'embedding_max_context_tokens': int,
            'max_response_tokens': int,

            'max_vector_database_pcnt': float,
            'buffer_context_pcnt': float,

            'top_k': int,
            'min_vector_db_score': float
        }

        # Add optional fields with their types
        optional_fields = {
            'model_type': str,
            'chat_format': Optional[str],
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


    @staticmethod
    def get_system_message(prompt_file: str, log_func: Callable[[str], None] = print)->str:
        """
        Accepts the file that contains the system prompt and attempts to extract it as a string.

        :param prompt_file:
        :param log_func: The function that prints whatever we are targeting. This will usually be 'print' or some form of 'logger.info'. 'logger'error' etc.
        :return:
        """
        # System message load
        system_message = ""
        try:
            system_message = LlamaUtils.load_system_prompt(prompt_file)
            log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_system_message: Loaded system message{ColoredText.END_TEXT}")
        except Exception as e:
            log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_system_message: Could not load system message file; loading default prompt. Error: [{e}]{ColoredText.END_TEXT}")

        if not system_message:
            system_message = LlamaUtils.BASE_SYSTEM_MESSAGE

        return system_message


    ################################################################################################################### Keyword Processing ####################################################################################################################
    @staticmethod
    def report_keyword(input_string:str, target:str):
        """
        Determines if a larger string contains a specific substring;.

        Args:
            input_string (str): Input string to process
            target (str): the target substring

        Returns:
            Boolean: True if the substring existed, False otherwise
        """
        # Convert to lowercase for case-insensitive search
        lower_str = input_string.lower()
        target = target.lower()

        # Find position of first occurrence
        idx = lower_str.find(target)

        if idx == -1:
            return False
        else:
            return True


    @staticmethod
    def report_and_remove_keyword(input_string:str, target:str):
        """
        Determines if a larger string contains a specific substring; if found, it removes the first occurrence of the substring (case-insensitive) from the input string.

        Args:
            input_string (str): Input string to process
            target (str): the target substring

        Returns:
            1. Boolean: True if the substring existed, False otherwise
            2. str: Modified string with first substring removed, or original if not found
        """
        # Convert to lowercase for case-insensitive search
        lower_str = input_string.lower()
        target = target.lower()
        target_len = len(target)

        # Find position of first occurrence
        idx = lower_str.find(target)

        if idx == -1:
            return False, input_string  # Return original if not found

        # Remove the substring at found position
        return True, input_string[:idx] + input_string[idx + target_len:]

    @staticmethod
    def remove_instruction_delimiters(input_string:str, delimiter:str, escape_characters: bool = True):
        """
        Sometimes, you want to send instructions to the LLM in your prompt, but you do not want to save the instructions in the prompt history (to save tokens, to not influence future prompts, etc). You can do this by including the instructions encased in
        a delimiter (i.e. ##These are hidden instructions that get passed to the LLM but not stored in the chat history.##). This method is used to remove the delimiters _before_ you send the prompt to the LLM, the instructions itself will be present, but
        delimiters will not.

        In other words, it simply removes the delimiters so your prompt can be (cleanly) sent to the LLM.

        You use this method on the string before you send the prompt to the LLM; you then run 'remove_instructions' on an _unmodified_ version of the prompt (i.e. a version of your prompt that was not run through this method) before you save it to your chat history / vector database.

        Args:
            input_string (str): Your prompt, including 'hidden' instructions encased in your favorite delimiter (e.g., "##", "**", "[[", etc.).
            delimiter (str): The string that marks the start and end of the hidden instructions that will not be saved to the chat history (e.g., "##", "**", "[[", etc.).
            escape_characters (bool): Sometimes, if special characters are used, you need to escape them; if they are not used this may not work. Give the option of escaping the characters.

        Returns:
            str: Your prompt with the instruction delimiters removed (but the instructions remain), with leading/trailing whitespace stripped.
        """
        # Escape the delimiter, but only if we opt to: This is crucial! If the delimiter contains special regex characters (like '.', '*', '+'), re.escape() will escape them so they are treated as literal characters.
        if escape_characters:
            escaped_delimiter = re.escape(delimiter)
            return input_string.replace(escaped_delimiter, "").strip()
        else:
            return input_string.replace(delimiter, "").strip()

    @staticmethod
    def remove_instructions(input_string: str, delimiter: str) -> str:
        """
        Sometimes, you want to send instructions to the LLM in your prompt, but you do not want to save the instructions in the prompt history (to save tokens, to not influence future prompts, etc). You can do this by including the instructions encased in
        a delimiter (i.e. ##These are hidden instructions that get passed to the LLM but not stored in the chat history.##). This method is used to clean the _entire_ instruction, including delimiters, before its saved to your chat history.

        In other words, it removes substrings enclosed by a specified delimiter string, including the delimiters themselves. Uses a non-greedy regex to correctly handle multiple pairs.

        You use this method on the string before you save your prompt to your chat history / vector database; you then run 'remove_instruction_delimiters' on an _unmodified_ version of the prompt (i.e. a version of your prompt that was not run through this method) before you send the prompt to the LLM.

        Args:
            input_string (str): Your prompt, including 'hidden' instructions encased in your favorite delimiter (e.g., "##", "**", "[[", etc.).
            delimiter (str): The string that marks the start and end of the hidden instructions that will not be saved to the chat history (e.g., "##", "**", "[[", etc.).

        Returns:
            str: Your prompt with the hidden instructions removed, with leading/trailing whitespace stripped.
        """
        return LlamaUtils.replace_instructions(input_string, delimiter, "")

    @staticmethod
    def replace_instructions(input_string: str, delimiter: str, replace_with: str) -> str:
        """
        There are cases where you want to substitute an arbitrary section of a string with another section. To do this, the subsection is identified with a delimiter. This will be replaced by 'replace_with'

        Args:
            input_string (str): The larger string you want to target.
            delimiter (str): The string that marks the start and end of the hidden instructions (e.g., "##", "**", "[[", etc.).
            replace_with (str): The string you wish to replace the contents of the delimiter with.

        Returns:
            str: The string the hidden instructions replaced, with leading/trailing whitespace stripped.
        """
        # 1. Escape the delimiter: This is crucial!
        #    If the delimiter contains special regex characters (like '.', '*', '+'),
        #    re.escape() will escape them so they are treated as literal characters.
        escaped_delimiter = re.escape(delimiter)

        # 2. Construct the regex pattern using the escaped delimiter:
        #    - {escaped_delimiter} : Matches the literal delimiter.
        #    - (.*?)              : Non-greedy match for any characters in between.
        pattern = rf"{escaped_delimiter}(.*?){escaped_delimiter}"

        # 3. Use re.sub to replace all occurrences of the pattern with an empty string.
        cleaned_string = re.sub(pattern, replace_with, input_string)

        # 4. Strip any leading or trailing whitespace that might have resulted
        #    from the removal (e.g., if the removed content was at the start/end).
        return cleaned_string.strip()

    ################################################################################################################### Fit History to Token Limit ####################################################################################################################
    @staticmethod
    def fit_to_token_limit(history_list, max_tokens):
        """
        Selects dictionaries from a list, starting from the most recent, until the accumulated 'token_count' exceeds max_tokens.

        Args:
            history_list (list): A list of dictionaries, where each dict has a 'token_count' key.
                                    Example: [{'role': 'user', 'content': 'hi', 'token_count': 5}, ...]
            max_tokens (int): The maximum allowed total token count.

        Returns:
            list: A new list containing the selected dictionaries, ordered from oldest to most recent.
                    Returns an empty list if no items fit the limit.
            int: The token count sum for all elements in the list
        """
        overall_token_counter = 0
        selected_history = []

        # Iterate through the list in reverse order (from most recent to oldest)
        for entry in reversed(history_list):
            current_entry_token_count = entry.get('token_count', 0) # Use .get() for safety

            # Check if adding this entry's token count would exceed the max_tokens limit
            if overall_token_counter + current_entry_token_count <= max_tokens:
                overall_token_counter += current_entry_token_count
                # Prepend the entry to the selected_history list
                # This ensures the final list is in chronological order (oldest to most recent)
                selected_history.insert(0, entry)
            else:
                # If adding this entry would exceed the limit, stop
                break

        # We do not want to start the history with a response from the assistant, so if that is the first element, remove it
        if selected_history and selected_history[0]['role'] == 'assistant':
            selected_history.pop(0)
        return selected_history, overall_token_counter


    ################################################################################################################### Token Count Discovery ####################################################################################################################
    @staticmethod
    def universal_token_count(llm, role, content, model_type="auto"):
        """
        This us a universal way to try and get the token count out of the model. The GGUF files _usually_ have he chat template properly embedded in the GGUF file. Usually. If so, you can simply use this to find the token count.

        :param llm: The LLM model.
        :param role: 'system', 'user', or 'assistant'
        :param content: The system prompt, the user request, or the LLM response.
        :param model_type: 'llama2', 'llama3', or 'command-r'
        :return: token count
        """
        try:
            # Try universal method first
            messages = [{"role": role, "content": content}]
            prompt = llm._format_chat_prompt(messages)
            tokens = llm.tokenize(prompt.encode())
            return len(tokens)
        except (AttributeError, Exception) as e:
            # Fall back to model-specific methods
            if model_type == "llama-3" or "llama-3" in llm.model_path.lower():
                return LlamaUtils.token_count_llama3(llm.tokenize, role, content)
            elif model_type == "llama2" or "llama-2" in llm.model_path.lower():
                return LlamaUtils.token_count_llama2(llm.tokenize, role, content)
            elif model_type == "command-r" or "command-r" in llm.model_path.lower():
                return LlamaUtils.token_count_command_r(llm.tokenize, role, content)
            elif model_type == "qwen" or "qwen" in llm.model_path.lower():
                return LlamaUtils.token_count_qwen(llm.tokenize, role, content)
            elif model_type == "gemma-3" or "gemma-3" in llm.model_path.lower():
                return LlamaUtils.token_count_gemma3(llm.tokenize, role, content)
            else:
                raise ValueError(f"Unknown model type and universal method failed: {e}")

    @staticmethod
    def token_count_gemma3(llm_tokenizer_func: Callable[[bytes], list[int]], role, content):
        """
            Formats a single message (role and content) according to Gemma 3's chat template segment
            and calculates its token count.

            This function calculates the tokens for:
            <start_of_turn>{role}\n{content}<end_of_turn>\n

            Notes:
            - Role 'assistant' is mapped to 'model' (official Gemma 3 convention)
            - Does NOT include <bos> (llama.cpp / tokenizer usually adds it automatically)
            - Does NOT include the next turn's <start_of_turn>model\n (that's for generation priming)
            - System prompts are typically formatted as user messages in Gemma 3

            Args:
                llm_tokenizer_func: The tokenizer function (e.g., llm.tokenize from llama_cpp.Llama).
                                    It should accept bytes and return a list/sequence of token IDs.
                role (str): The role of the message ('user', 'assistant', 'system').
                content (str): The text content of the message.

            Returns:
                int: The token count of the templated message segment.
        """

        if role not in ['system', 'user', 'assistant']:
            raise ValueError(f"Unsupported role: {role}. Expected 'system', 'user', or 'assistant'.")

        # Map 'assistant' → 'model' (Gemma 3 uses 'model' for assistant responses)
        # 'system' is usually treated as 'user' in practice
        template_role = 'model' if role == 'assistant' else role

        # Clean the content to avoid extra tokens from leading/trailing whitespace
        clean_content = content.strip()

        # Construct the message segment per Gemma 3 format
        templated_segment = f"<start_of_turn>{template_role}\n{clean_content}<end_of_turn>\n"

        # Tokenize the segment
        tokens = llm_tokenizer_func(templated_segment.encode("utf-8"))

        return len(tokens)

    @staticmethod
    def token_count_qwen(llm_tokenizer_func: Callable[[bytes], list[int]], role, content):
        """
        Formats a single message (role and content) according to Qwen's chat template segment and calculates its token count.

        This function calculates the tokens for:
        <|im_start|>{role}\n{content}<|im_end|>\n

        It does NOT include the initial system message or the final <|im_start|>assistant\n (which primes the model to respond),
        as those are part of the overall conversation prompt construction, not an individual message's self-contained segment.

        Args:
            llm_tokenizer_func: The tokenizer function (e.g., llm.tokenize from llama_cpp.Llama).
                                It should accept bytes and return a list/sequence of token IDs.
            role (str): The role of the message ('user', 'assistant', 'system').
            content (str): The text content of the message.

        Returns:
            int: The token count of the templated message segment.
        """

        if role not in ['system', 'user', 'assistant']:
            raise ValueError(f"Unsupported role: {role}. Expected 'system', 'user', or 'assistant'.")

        # Clean the content to avoid extra tokens from leading/trailing whitespace
        clean_content = content.strip()

        # Construct the message segment for Qwen (ChatML style)
        templated_segment = f"<|im_start|>{role}\n{clean_content}<|im_end|>\n"

        # Tokenize the segment
        tokens = llm_tokenizer_func(templated_segment.encode())

        return len(tokens)

    @staticmethod
    def token_count_llama3(llm_tokenizer_func: Callable[[bytes], list[int]], role, content):
        """
        Formats a single message (role and content) according to its Llama 3 Instruct chat template segment and calculates its token count.

        This function calculates the tokens for:
        <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>

        It does NOT include the <|begin_of_text|> token (which starts the whole conversation) or the final <|start_header_id|>assistant<|end_header_id|>\n\n (which primes the model to respond), as those are part of the overall conversation prompt construction, not an individual message's self-contained segment.

        Args:
            llm_tokenizer_func: The tokenizer function (e.g., llm.tokenize from llama_cpp.Llama or tokenizer.encode from HuggingFace transformers). It should accept a string and return a list/sequence of token IDs.
                                You can literally say 'llama2_tokenizer_func = llm.tokenize'
            role (str): The role of the message ('user', 'assistant', 'system').
            content (str): The text content of the message.

        Returns:
            int: The token count of the templated message segment.
        """
        
        if role not in ['system', 'user', 'assistant']:
            raise ValueError(f"Unsupported role: {role}. Expected 'system', 'user', or 'assistant'.")

        # Clean the content to avoid extra tokens from leading/trailing whitespace
        clean_content = content.strip()

        # Construct the message segment for Llama 3 - use to be encapsulated in parenthesis
        templated_segment = f"<|start_header_id|>{role}<|end_header_id|>\n\n{clean_content}<|eot_id|>"

        # Tokenize the segment - used to not have the .encode()
        tokens = llm_tokenizer_func(templated_segment.encode())
        
        return len(tokens)


    def token_count_llama2(llm_tokenizer_func, role, content, system_prompt=None):
        """
        Formats a single message (role and content) according to its Llama 2 Instruct
        chat template segment and calculates its token count.

        This function calculates the tokens for:
        - User message (first turn with system): <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]
        - User message (subsequent or no system): <s>[INST] {user_message} [/INST]
        - Assistant message: {model_answer}</s> (includes a leading space)

        It does NOT include the <s> token at the *start of every turn after the first* in a multi-turn conversation (as Llama 2 actually chains turns like `...</s><s>[INST]...`),
        but rather adds it if this segment is treated as a potential start of a prompt. For precise multi-turn counting, you'd build the full prompt string.

        Args:
            llm_tokenizer_func: The tokenizer function (e.g., llm.tokenize from llama_cpp.Llama or tokenizer.encode from HuggingFace transformers). It should accept a string and return a list/sequence of token IDs.
            role (str): The role of the message ('user', 'assistant'). 'system' roles are handled via the 'system_prompt' argument for a 'user' message.
            content (str): The text content of the message.
            system_prompt (str, optional): The system prompt content. This is only used when 'role' is 'user' and this is intended as the first turn of a conversation.

        Returns:
            int: The token count of the templated message segment.
        """
        templated_segment = ""
        clean_content = content.strip() # Remove extra whitespace

        if role == 'user':
            if system_prompt:
                # First user message with a system prompt
                clean_system_prompt = system_prompt.strip()
                templated_segment = f"<s>[INST] <<SYS>>\n{clean_system_prompt}\n<</SYS>>\n\n{clean_content} [/INST]"

            else:
                # Subsequent user messages or first without system prompt
                templated_segment = f"<s>[INST] {clean_content} [/INST]"
        elif role == 'assistant':
            # Assistant responses often start with a space and end with </s>
            templated_segment = f" {clean_content}</s>"
        elif role == 'system':
            raise ValueError(
                "For Llama 2, 'system' messages are typically part of the initial 'user' "
                "turn using the 'system_prompt' argument. A standalone 'system' role "
                "does not have a direct template segment for token counting."
            )
        else:
            raise ValueError(f"Unsupported role: {role}. Expected 'user' or 'assistant'.")

        # Tokenize the segment - used to not have the .encode()
        tokens = llm_tokenizer_func(templated_segment.encode())
        
        return len(tokens)

    def token_count_command_r(llm_tokenizer_func: Callable[[bytes], list[int]], role, content):
        """
        Like token_count_llama3 and token_count_llama2, but for Command R / Cohere chat template.

        :param role:
        :param content:
        :return:
        """

        # Map standard roles to Command R roles
        role_map = {
            'system': 'SYSTEM',
            'user': 'USER',
            'assistant': 'CHATBOT'
        }

        if role not in role_map:
            raise ValueError(f"Unsupported role: {role}. Expected 'system', 'user', or 'assistant'.")

        command_r_role = role_map[role]
        clean_content = content.strip()

        # Command R template format
        templated_segment = f"<|START_OF_TURN_TOKEN|><|{command_r_role}_TOKEN|>{clean_content}<|END_OF_TURN_TOKEN|>"

        tokens = llm_tokenizer_func(templated_segment.encode())

        return len(tokens)
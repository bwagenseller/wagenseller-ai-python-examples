import re
import argparse
import json
import os
from amadeo_utils.ai.llm.llama.subjective_constants import SubjectiveConstants
from amadeo_utils.colored_text import ColoredText
from typing import Callable, Optional


# CUDA numbers its devices 'fastest first' by default, and its idea of 'fastest' is driven by compute capability
# rather than raw performance. On a box holding, say, an RTX 3090 (sm_86) and an RTX 4070 Ti (sm_89), CUDA reports
# the 4070 Ti as device 0 even though 'nvidia-smi -L' lists the 3090 first - so '--gpu 0' would silently load the
# model onto the opposite card to the one the user was looking at. Pinning the order to the physical slot order makes
# the index mean what anyone reading '--gpu 0' will expect it to mean.
#
# This MUST be set before the CUDA runtime is initialised, which in practice means before 'llama_cpp' is imported -
# importing llama_cpp loads the llama.cpp shared library, which registers its GGML CUDA backend. Every module in this
# tree therefore imports llama_utils BEFORE it imports llama_cpp; do not let an import sorter 'tidy' that order.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


class LlamaUtils:

    ################################################################################################################### Constants ####################################################################################################################


    CONVO_NAME = "my_chat"

    MODEL_TYPE = "llama-3"
    CHAT_FORMAT = None

    BASE_SYSTEM_MESSAGE = "You are Ada, a helpful AI automatron."

    GPU_INDEX = 0  # Which physical CUDA device to load the models onto; 0 is the first GPU, and is the only index guaranteed to exist
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

    ################################################################################################################### GPU Selection ####################################################################################################################

    # llama.cpp's split modes, as raw integers from the ggml/llama.h enum 'llama_split_mode'. They are resolved from the
    # llama_cpp module when it exposes them (which it has for a long time) and fall back to these literals when it does
    # not, so that a slightly older or newer binding cannot break GPU selection outright. The values themselves are
    # stable ABI - changing them would break every existing gguf runner - so the fallback is safe.
    SPLIT_MODE_NONE = 0   # the whole model goes on one GPU: 'main_gpu'
    SPLIT_MODE_LAYER = 1  # layers are spread across all visible GPUs, proportional to their VRAM

    @staticmethod
    def get_cuda_device_count(log_func: Callable[[str], None] = print) -> int:
        """
        Counts the CUDA devices on this machine by asking 'nvidia-smi -L' and counting the GPUs it lists.

        nvidia-smi is used rather than a library call because this has to work in whatever environment llama.cpp is
        installed into, which may well have no torch and no pynvml. It also happens to be the exact list a user reads
        when they decide which index to pass to '--gpu', and it enumerates in PCI slot order - the same order this
        module pins CUDA to at import time - so the numbering the user sees and the numbering llama.cpp uses agree.

        :param log_func: The function that prints whatever we are targeting.
        :return: The number of CUDA devices, or 0 if that could not be determined (no nvidia-smi, no driver, etc). A 0
                 here means 'unknown', not necessarily 'no GPU' - callers should treat it as 'skip validation'.
        """

        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return 0
            # Every GPU gets one line of the form 'GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-...)'
            return len([line for line in result.stdout.splitlines() if line.strip().startswith('GPU ')])
        except (OSError, ValueError, subprocess.SubprocessError):
            # No nvidia-smi on PATH, no driver loaded, or it hung - either way we cannot validate, so say so.
            return 0


    @staticmethod
    def build_gpu_kwargs(gpu_index: int, split_gpus: bool, model_description: str, log_func: Callable[[str], None] = print) -> dict:
        """
        Works out the GPU-related keyword arguments to hand to the Llama constructor, and reports what it decided.

        There is a trap worth knowing about here: llama.cpp's DEFAULT split mode is 'layer', not 'none'. On a machine
        with more than one GPU, constructing a Llama without saying anything about GPUs does not put the model on GPU 0
        - it spreads the layers across every card it can see. So pinning a model to a single GPU is the case that needs
        explicit arguments ('split_mode' of none, plus 'main_gpu'), not the case that comes for free.

        :param gpu_index: The CUDA device index to use, numbered as 'nvidia-smi -L' lists them.
        :param split_gpus: If True, spread this model across every GPU instead of pinning it to 'gpu_index'.
        :param model_description: What is being loaded ('generative' / 'embedding'), for the log line only.
        :param log_func: The function that prints whatever we are targeting.
        :return: A dictionary of keyword arguments to splat into the Llama constructor.
        :raises ValueError: If gpu_index does not exist on this machine.
        """

        device_count = LlamaUtils.get_cuda_device_count(log_func)

        # Resolve the split mode constants from the binding where possible - see the note on SPLIT_MODE_NONE above.
        import llama_cpp
        split_none = getattr(llama_cpp, 'LLAMA_SPLIT_MODE_NONE', LlamaUtils.SPLIT_MODE_NONE)
        split_layer = getattr(llama_cpp, 'LLAMA_SPLIT_MODE_LAYER', LlamaUtils.SPLIT_MODE_LAYER)

        # A device count of 0 means nvidia-smi could not tell us anything - no nvidia-smi on PATH, no driver, or a
        # CPU-only box. We cannot VALIDATE the index in that state, but we still pass the request through rather than
        # quietly dropping it: silently ignoring an explicit '--gpu N' is worse than letting llama.cpp report the
        # problem itself, and on a genuinely CPU-only build these arguments are inert anyway.
        if device_count == 0:
            log_func(f"{ColoredText.YELLOW_TEXT}LlamaUtils.build_gpu_kwargs: Could not determine the CUDA device count (is nvidia-smi on PATH?), so GPU {gpu_index} could not be validated. Passing the request through as-is for the {model_description} model - if that index does not exist, llama.cpp will be the one to complain.{ColoredText.END_TEXT}")
            if split_gpus:
                return {'split_mode': split_layer, 'main_gpu': gpu_index, 'tensor_split': None}
            return {'split_mode': split_none, 'main_gpu': gpu_index, 'tensor_split': None}

        if gpu_index < 0 or gpu_index >= device_count:
            raise ValueError(
                f"Requested GPU index {gpu_index} does not exist; this machine has {device_count} CUDA device(s), "
                f"so valid indexes are 0 through {device_count - 1}."
            )

        if split_gpus and device_count > 1:
            # Spread the layers over every card. 'main_gpu' still matters in this mode: it is the card that holds the
            # small tensors that are not split, so it should be the one the user nominated.
            log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.build_gpu_kwargs: Spreading the {model_description} model across all {device_count} GPUs (primary GPU {gpu_index}).{ColoredText.END_TEXT}")
            return {'split_mode': split_layer, 'main_gpu': gpu_index, 'tensor_split': None}

        if split_gpus:
            log_func(f"{ColoredText.YELLOW_TEXT}LlamaUtils.build_gpu_kwargs: '--split-gpus' was requested but this machine has only {device_count} CUDA device; loading the {model_description} model onto GPU {gpu_index} instead.{ColoredText.END_TEXT}")

        # The single GPU case, which - per the note above - is the one that must be spelled out.
        log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.build_gpu_kwargs: Loading the {model_description} model onto GPU {gpu_index} of {device_count} available.{ColoredText.END_TEXT}")
        return {'split_mode': split_none, 'main_gpu': gpu_index, 'tensor_split': None}


    ################################################################################################################### Building The Argument Parser ####################################################################################################################

    """
    The four entry points in this tree - the role-play script, the knowledge base script, the role-play server, and the
    knowledge base server - all accept overlapping sets of arguments. Rather than each one spelling out its own copy of
    every 'add_argument' call (which is how these drifted apart in the first place), each argument group is defined
    exactly once below and the entry points compose the groups they need:

        role-play script         : system + role-play
        knowledge base script    : system + knowledge base
        role-play server         : system + role-play server + server
        knowledge base server    : system + knowledge base   + server

    'System' is the group shared by all four - it describes the machine and the models running on it. That is where
    '--gpu' lives, because which card to load onto is a property of the box, not of the story being told or the body of
    knowledge being answered from.
    """

    @staticmethod
    def add_system_arguments(parser: argparse.ArgumentParser, include_config_json: bool = True):
        """
        Adds the SYSTEM arguments - the models to load, which GPU to load them onto, how much of them to push onto that
        GPU, how big the context windows are, and how the vector database is tuned. These are the settings you would
        expect to keep stable across every conversation on a given box, and they are shared by all four entry points.

        :param parser: The parser to add the arguments to.
        :param include_config_json: Whether to add '--system-config-json'. True for the interactive scripts, which
                                    resolve their configuration from two independent JSON files; False for the servers,
                                    which take one monolithic '--json' instead (see add_monolithic_json_argument).
        """

        parser.add_argument("-bmd", "--base-model-dir", default=SubjectiveConstants.BASE_MODEL_DIR,help="The location of the base model that will generate text.")
        parser.add_argument("-bed", "--base-embedding-dir", default=SubjectiveConstants.BASE_EMBEDDING_DIR,help="The location of the embedding model.")
        parser.add_argument("-m", "--model", default=SubjectiveConstants.MODEL,help="The filename of your text generation model. Please, just the filename, not the directory.")
        parser.add_argument("-mt", "--model-type", default=LlamaUtils.MODEL_TYPE,help="The model type: ['llama-2', 'llama-3', 'alpaca', 'qwen', 'command-r', 'vicuna', 'oasst_llama', 'baichuan-2', 'baichuan', 'openbuddy', 'redpajama-incite', 'snoozy', 'phind', 'intel', 'open-orca', 'mistrallite', 'zephyr', 'pygmalion', 'chatml', 'mistral-instruct', 'chatglm3', 'openchat', 'saiga', 'gemma', 'functionary', 'functionary-v2', 'functionary-v1', 'chatml-function-calling'].")
        parser.add_argument("-cf", "--chat-format", default=LlamaUtils.CHAT_FORMAT,help="The chat format type; you should usually leave this None and let the system figure it out (with the exception of 'command-r' - in that case, use 'llama-3'). Values: ['llama-2', 'llama-3', 'alpaca', 'qwen', 'vicuna', 'oasst_llama', 'baichuan-2', 'baichuan', 'openbuddy', 'redpajama-incite', 'snoozy', 'phind', 'intel', 'open-orca', 'mistrallite', 'zephyr', 'pygmalion', 'chatml', 'mistral-instruct', 'chatglm3', 'openchat', 'saiga', 'gemma', 'functionary', 'functionary-v2', 'functionary-v1', 'chatml-function-calling'].")
        parser.add_argument("-em", "--embedding-model", default=SubjectiveConstants.EMBEDDING_MODEL,help="The filename of your embedding model. Please, just the filename, not the directory.")

        parser.add_argument("-g", "--gpu", type=int, default=LlamaUtils.GPU_INDEX,help="The index of the CUDA GPU to load the models onto, matching the order 'nvidia-smi -L' reports (0 for the first GPU, 1 for the second, and so on); the ordering is pinned to the physical slot order via CUDA_DEVICE_ORDER, so it does not depend on which card CUDA considers fastest. Defaults to the first GPU. Ignored if no CUDA device is available, and ignored if '--split-gpus' is used.")
        parser.add_argument("-sg", "--split-gpus", action='store_true',help="Spread the generative model across EVERY CUDA GPU on this machine rather than loading it onto the single card named by '--gpu'. This is how you run a model whose weights do not fit in the VRAM of any one card. The embedding model is small and is always loaded onto the '--gpu' card regardless. Has no effect on a single GPU machine.")
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

        parser.add_argument("-d", "--debug", action='store_true',help="Do you want to see some additional log lines while using the LLM?")

        if include_config_json:
            parser.add_argument("-scj", "--system-config-json", type=str, default="", help="If this points to a valid JSON file, the SYSTEM parameters (the models, the GPU selection, GPU layers, context sizes, and vector database tuning) are pulled from that file, and the defaults - and the matching arguments passed from the command line - are ignored. If the JSON load fails for whatever reason, though, the defaults WILL be engaged for the system parameters. This is the SAME system config file format the other interactive script uses, so one file can serve both. This is loaded independently of the other config file.")


    @staticmethod
    def add_monolithic_json_argument(parser: argparse.ArgumentParser):
        """
        Adds the single '--json' argument used by the two SERVERS.

        The servers deliberately keep one config file covering everything, rather than the two-file split the
        interactive scripts use. The split exists so that the machine settings can be separated from the settings of
        the one story being told - but a server is never telling one story: the client names its own player and picks
        its own system prompt out of a directory when it opens a session. There is therefore nothing story-shaped left
        in a server's config to split off, and one file per server is simpler to deploy.

        :param parser: The parser to add the argument to.
        """

        parser.add_argument("-j", "--json", type=str, default="", help="If this points to a valid JSON file, the ENTIRE parameter settings are pulled from that file, and the defaults - and other arguments passed from the command line - are ignored. If the JSON load fails for whatever reason, though, the defaults WILL be engaged.")


    @staticmethod
    def add_server_arguments(parser: argparse.ArgumentParser, default_host: str, default_port: int):
        """
        Adds the SERVER arguments - where the server binds. Only the two server entry points use these; the interactive
        scripts have no socket to bind.

        The defaults are passed in rather than read from a constant here because each server owns its own default port,
        and the two must not collide if they are run on the same box.

        :param parser: The parser to add the arguments to.
        :param default_host: The hostname/IP this particular server binds to when '--host' is not given.
        :param default_port: The port this particular server listens on when '--port' is not given.
        """

        parser.add_argument("-ho", "--host", default=default_host, help="The hostname/IP that the server will bind to.")
        parser.add_argument("-p", "--port", type=int, default=default_port, help="The port that the server will listen on for requests.")


    @staticmethod
    def add_role_play_arguments(parser: argparse.ArgumentParser):
        """
        Adds the ROLE-PLAY arguments for the INTERACTIVE script - the settings that describe the single story being
        told: who you are playing, where that one conversation is stored, and which system prompt sets the tone.

        The server equivalent is add_role_play_server_arguments, which is deliberately different - see that method.

        :param parser: The parser to add the arguments to.
        """

        parser.add_argument("-pn", "--player-name", default=SubjectiveConstants.BASE_PLAYER_NAME,help="Give your username - What did you say your name was in the prompt")
        parser.add_argument("-bcd", "--base-convo-dir", default=SubjectiveConstants.BASE_CONVO_DIR,help="The base directory where all conversation histories are stored.")
        parser.add_argument("-cn", "--convo-name", default=LlamaUtils.CONVO_NAME,help="What do you want this chat session to be named?")
        parser.add_argument("-obcd", "--override-base-convo-dir", action='store_true',help="Do you want to use the override base directory? This is a second directory that is immutable and is meant for temporary conversation storage")
        parser.add_argument("-spf", "--system-prompt-file", default=SubjectiveConstants.SYSTEM_PROMPT_FILE,help="A filename (full, absolute path to the file) that contains your system prompt; this is a text file. This is the initial message you send to the LLM to 'set the tone' of the entire conversation. This is critical! Be creative!")

        parser.add_argument("-en", "--encrypted", type=bool,default=False, help="Is the conversation history encrypted? If set to true, you will have to enter a passphrase.")

        parser.add_argument("-rpcj", "--role-play-config-json", type=str, default="", help="If this points to a valid JSON file, the ROLE-PLAY parameters (the player name, conversation directory, system prompt file, encryption, and debug) are pulled from that file, and the defaults - and the matching arguments passed from the command line - are ignored. If the JSON load fails for whatever reason, though, the defaults WILL be engaged for the role-play parameters. This is loaded independently of '--system-config-json'.")


    @staticmethod
    def add_role_play_server_arguments(parser: argparse.ArgumentParser):
        """
        Adds the ROLE-PLAY arguments for the SERVER - which are NOT the same as the interactive script's, because a
        server holds many conversations at once and cannot know any of them at startup:

          * The interactive script takes a single '--system-prompt-file'; the server takes a '--system-prompt-dir' and
            picks a prompt out of it per session, keyed by the 'system_prompt_id' the client sends.
          * The interactive script joins '--base-convo-dir' and '--convo-name' into one 'convo_dir' at parse time; the
            server keeps the base directory as-is and appends 'user_id/system_prompt_id' when each session is created.
          * '--player-name' is absent entirely - each client announces its own player name when it opens a session.

        :param parser: The parser to add the arguments to.
        """

        parser.add_argument("-bcd", "--base-convo-dir", default=SubjectiveConstants.BASE_CONVO_DIR,help="The base directory where all conversation histories are stored. Each session gets its own directory beneath this one, named for the user and the system prompt it is using.")
        parser.add_argument("-spf", "--system-prompt-dir", default=SubjectiveConstants.SYSTEM_PROMPT_DIR,help="A directory that contains system prompt files; this is just the directory, but the prompts themselves are in text files in this directory. A client picks one by sending its 'system_prompt_id', which is matched against the filename (minus the '.txt'). These files represent the initial message you send to the LLM to 'set the tone' of the entire conversation. This is critical! Be creative!")

        parser.add_argument("-en", "--encrypted", type=bool,default=False, help="Is the conversation history encrypted? If set to true, you will have to enter a passphrase when the server starts.")


    @staticmethod
    def add_knowledge_base_arguments(parser: argparse.ArgumentParser, include_config_json: bool = True):
        """
        Adds the KNOWLEDGE BASE arguments - the settings that describe the body of knowledge being answered from. These
        are shared verbatim by the interactive knowledge base script and the knowledge base server; unlike role-play, a
        knowledge base server answers every client from the same single body of knowledge, so there is nothing
        per-session to vary.

        :param parser: The parser to add the arguments to.
        :param include_config_json: Whether to add '--knowledge-base-config-json'. True for the interactive script;
                                    False for the server, which takes one monolithic '--json' instead.
        """

        parser.add_argument("-kbf", "--knowledge-base-file", type=str, default="", help="The JSON Lines (JSONL) file that contains your knowledge base. This must be in JSONL format - a list of dictionaries with fields 'id', 'question', and 'answer'.")
        parser.add_argument("-spf", "--system-prompt-file", default=SubjectiveConstants.SYSTEM_PROMPT_FILE,help="A filename (full, absolute path to the file) that contains your system prompt; this is a text file. This is the initial message you send to the LLM to 'set the tone' of the entire conversation. This is critical! Be creative!")

        if include_config_json:
            parser.add_argument("-kbcj", "--knowledge-base-config-json", type=str, default="", help="If this points to a valid JSON file, the KNOWLEDGE BASE parameters (the knowledge base file, the system prompt file, and debug) are pulled from that file, and the defaults - and the matching arguments passed from the command line - are ignored. If the JSON load fails for whatever reason, though, the defaults WILL be engaged for the knowledge base parameters. This is loaded independently of '--system-config-json'.")


    ################################################################################################################### Parsing Arguments From Command Line ####################################################################################################################

    @staticmethod
    def get_args_dict(log_func: Callable[[str], None] = print) -> dict:
        """
        Gets args dictionary for a traditional vector database, meant to save the conversation for later.

        The configuration is split across two concepts, each with its own optional JSON file:
          * SYSTEM ('--system-config-json') - the models, GPU layers, context sizes, and vector database tuning; the
            settings tied to the machine you are running on. See get_system_args_dict.
          * ROLE-PLAY ('--role-play-config-json') - the player name, conversation directory, system prompt file,
            encryption, and debug; the settings tied to the story you are telling. See get_role_play_args_dict.

        The two halves are resolved independently: if one JSON file is missing or malformed, only that half falls back
        to the command line arguments and defaults - the other half keeps whatever its own JSON gave it. The two halves
        are then merged into the single dictionary returned here, which is what RolePlay consumes.

        :param log_func: The function that prints whatever we are targeting. This will usually be 'print' or some form of 'logger.info'. 'logger'error' etc.
        :return: The merged dictionary of system and role-play settings, plus the loaded 'system_message'. Empty if argument parsing failed or '--help' was used.
        """

        parser = argparse.ArgumentParser(description='Run a LLM, as you see fit.')
        LlamaUtils.add_system_arguments(parser)
        LlamaUtils.add_role_play_arguments(parser)

        argDict = {}

        try:
            args = parser.parse_args()

            # The command line is parsed exactly once (argparse owns all of sys.argv), but the resulting namespace is
            # handed to two independent builders. Each builder decides on its own whether to source its values from its
            # JSON file or from the args / defaults, so a broken system config cannot poison the role-play settings
            # (and vice versa).
            argDict.update(LlamaUtils.get_system_args_dict(args, log_func))
            argDict.update(LlamaUtils.get_role_play_args_dict(args, log_func))

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
    def get_system_args_dict(args: argparse.Namespace, log_func: Callable[[str], None] = print) -> dict:
        """
        Builds the SYSTEM half of the args dictionary - the settings that describe the machine and the models running on
        it, rather than the story being told. These are the settings you would expect to keep stable across every
        role-play you run on a given box: which models to load, how much of them to push onto the GPU, how big the
        context windows are, and how the vector database is tuned.

        The values are sourced from '--system-config-json' if that file was supplied and loads cleanly; otherwise they
        fall back to the matching command line arguments (and their defaults). This decision is made independently of
        the role-play half - see get_role_play_args_dict.

        :param args: The already-parsed argparse namespace. Parsing happens once, in get_args_dict.
        :param log_func: The function that prints whatever we are targeting. This will usually be 'print' or some form of 'logger.info', 'logger.error', etc.
        :return: A dictionary of the system settings.
        """

        argDict = {}
        use_default_arg_config = True  # This is only flipped if we successfully load from a JSON file

        json_config_file = args.system_config_json

        if json_config_file and os.path.exists(json_config_file):
            try:
                config_dict = LlamaUtils.load_system_json_config(json_config_file)

                argDict['generating_model'] = os.path.join(config_dict['base_model_dir'], config_dict['model'])
                argDict['embedding_model'] = os.path.join(config_dict['base_embedding_dir'], config_dict['embedding_model'])

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
                argDict['chat_format'] = config_dict.get('chat_format', LlamaUtils.CHAT_FORMAT)

                # These three are optional so that a system config written before they existed still loads cleanly.
                # They fall back to the CLASS CONSTANTS, not to the matching command line arguments - which is the rule
                # every optional field in this half already follows (see 'model_type' and 'chat_format' above). Once a
                # system config loads it owns the whole system half: '--gpu' and friends are ignored, and the card is
                # changed by editing the file. Do not "helpfully" make these fall back to args instead - that would
                # leave two classes of optional field behaving differently and make the config harder to reason about.
                argDict['gpu'] = config_dict.get('gpu', LlamaUtils.GPU_INDEX)
                argDict['split_gpus'] = config_dict.get('split_gpus', False)
                argDict['repeat_penalty'] = config_dict.get('repeat_penalty', LlamaUtils.REPEAT_PENALITY)

                log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_system_args_dict: System config loaded from JSON file {json_config_file}; the generating model is '{argDict['generating_model']}' and the embedding model is '{argDict['embedding_model']}'.{ColoredText.END_TEXT}")
                use_default_arg_config = False

            except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
                log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_system_args_dict: Could not load system JSON config [{json_config_file}] - there are errors. Will attempt to load other defaults or args. Error: {e}.{ColoredText.END_TEXT}")

        elif json_config_file:
            log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_system_args_dict: Could not load system JSON config [{json_config_file}] - file does not exist. Loading from defaults or other parameters sent.{ColoredText.END_TEXT}")

        if use_default_arg_config:

            argDict['generating_model'] = os.path.join(args.base_model_dir, args.model)
            argDict['embedding_model'] = os.path.join(args.base_embedding_dir, args.embedding_model)

            argDict['generating_gpu_layers'] = args.gpu_layers
            argDict['embedding_gpu_layers'] = args.embedding_gpu_layers
            argDict['generating_max_context_tokens'] = args.max_context_tokens
            argDict['embedding_max_context_tokens'] = args.embedding_max_context_tokens
            argDict['max_response_tokens'] = args.max_response_tokens  # Maximum tokens the generative model is allowed to generate

            argDict['max_vector_database_pcnt'] = args.max_vector_database_pcnt
            argDict['buffer_context_pcnt'] = args.buffer_context_pcnt

            argDict['top_k'] = args.top_k
            argDict['min_vector_db_score'] = args.min_vector_db_score

            argDict['model_type'] = args.model_type
            argDict['chat_format'] = args.chat_format

            argDict['gpu'] = args.gpu
            argDict['split_gpus'] = args.split_gpus
            argDict['repeat_penalty'] = args.repeat_penalty

            log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_system_args_dict: System config loaded from args / defaults; the generating model is '{argDict['generating_model']}' and the embedding model is '{argDict['embedding_model']}'.{ColoredText.END_TEXT}")

        return argDict


    @staticmethod
    def get_role_play_args_dict(args: argparse.Namespace, log_func: Callable[[str], None] = print) -> dict:
        """
        Builds the ROLE-PLAY half of the args dictionary - the settings that describe the story being told rather than
        the machine telling it. These are the settings you would expect to swap every time you start a different
        conversation: who you are playing, where that conversation is stored, and which system prompt sets the tone.

        The values are sourced from '--role-play-config-json' if that file was supplied and loads cleanly; otherwise
        they fall back to the matching command line arguments (and their defaults). This decision is made independently
        of the system half - see get_system_args_dict.

        :param args: The already-parsed argparse namespace. Parsing happens once, in get_args_dict.
        :param log_func: The function that prints whatever we are targeting. This will usually be 'print' or some form of 'logger.info', 'logger.error', etc.
        :return: A dictionary of the role-play settings.
        """

        argDict = {}
        use_default_arg_config = True  # This is only flipped if we successfully load from a JSON file

        json_config_file = args.role_play_config_json

        if json_config_file and os.path.exists(json_config_file):
            try:
                config_dict = LlamaUtils.load_role_play_json_config(json_config_file)

                # The override directory is a second, immutable base directory meant for temporary conversation storage;
                # when it is requested, the configured base_convo_dir is ignored entirely.
                if config_dict['override_base_convo_dir']:
                    argDict['convo_dir'] = os.path.join(SubjectiveConstants.BASE_CONVO_DIR_OVERRIDE, config_dict['convo_name'])
                else:
                    argDict['convo_dir'] = os.path.join(config_dict['base_convo_dir'], config_dict['convo_name'])

                argDict['system_prompt_file'] = config_dict['system_prompt_file']

                argDict['player_name'] = config_dict.get('player_name', SubjectiveConstants.BASE_PLAYER_NAME)
                argDict['encrypted'] = config_dict.get('encrypted', False)
                argDict['debug'] = config_dict.get('debug', False)

                log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_role_play_args_dict: Role-play config loaded from JSON file {json_config_file}; prompt is from '{argDict['system_prompt_file']}' and convo directory is '{argDict['convo_dir']}'.{ColoredText.END_TEXT}")
                use_default_arg_config = False

            except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
                log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_role_play_args_dict: Could not load role-play JSON config [{json_config_file}] - there are errors. Will attempt to load other defaults or args. Error: {e}.{ColoredText.END_TEXT}")

        elif json_config_file:
            log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_role_play_args_dict: Could not load role-play JSON config [{json_config_file}] - file does not exist. Loading from defaults or other parameters sent.{ColoredText.END_TEXT}")

        if use_default_arg_config:

            if args.override_base_convo_dir:
                argDict['convo_dir'] = os.path.join(SubjectiveConstants.BASE_CONVO_DIR_OVERRIDE, args.convo_name)
            else:
                argDict['convo_dir'] = os.path.join(args.base_convo_dir, args.convo_name)

            argDict['system_prompt_file'] = args.system_prompt_file

            argDict['player_name'] = args.player_name
            argDict['encrypted'] = args.encrypted
            argDict['debug'] = args.debug

            log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_role_play_args_dict: Role-play config loaded from args / defaults; prompt is from '{argDict['system_prompt_file']}' and convo directory is '{argDict['convo_dir']}'.{ColoredText.END_TEXT}")

        return argDict


    @staticmethod
    def get_args_dict_knowledge_base(log_func: Callable[[str], None] = print) -> dict:
        """
        Gets args dictionary for a knowledge base - this is not meant to be a continuous chatbot, just a simple 'answer these questions' without caring too much about the conversation.

        As with get_args_dict, the configuration is split across two concepts, each with its own optional JSON file:
          * SYSTEM ('--system-config-json') - the models, GPU layers, context sizes, and vector database tuning; the
            settings tied to the machine you are running on. This is handled by the very same get_system_args_dict the
            role-play script uses, so one system JSON can drive both scripts on a given box.
          * KNOWLEDGE BASE ('--knowledge-base-config-json') - the knowledge base file, the system prompt file, and
            debug; the settings tied to the body of knowledge being answered from. See get_knowledge_base_args_dict.

        The two halves are resolved independently: if one JSON file is missing or malformed, only that half falls back
        to the command line arguments and defaults - the other half keeps whatever its own JSON gave it.

        :param log_func: The function that prints whatever we are targeting. This will usually be 'print' or some form of 'logger.info'. 'logger'error' etc.
        :return: The merged dictionary of system and knowledge base settings, plus the loaded 'system_message'. Empty if argument parsing failed or '--help' was used.
        """

        parser = argparse.ArgumentParser(description='Run a LLM, as you see fit.')
        LlamaUtils.add_system_arguments(parser)
        LlamaUtils.add_knowledge_base_arguments(parser)

        argDict = {}

        try:
            args = parser.parse_args()

            # As in get_args_dict, the command line is parsed exactly once and the namespace is handed to two
            # independent builders, so a broken system config cannot poison the knowledge base settings (or vice
            # versa). Note that get_system_args_dict is shared verbatim with the role-play script - the system half of
            # a knowledge base config is the same set of fields - which means a single system JSON can drive both.
            argDict.update(LlamaUtils.get_system_args_dict(args, log_func))
            argDict.update(LlamaUtils.get_knowledge_base_args_dict(args, log_func))

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
    def get_knowledge_base_args_dict(args: argparse.Namespace, log_func: Callable[[str], None] = print) -> dict:
        """
        Builds the KNOWLEDGE BASE half of the args dictionary - the settings that describe the body of knowledge being
        answered from rather than the machine answering. These are the settings you would expect to swap when you point
        the script at a different subject: which JSONL knowledge base to load, and which system prompt frames it.

        The values are sourced from '--knowledge-base-config-json' if that file was supplied and loads cleanly;
        otherwise they fall back to the matching command line arguments (and their defaults). This decision is made
        independently of the system half - see get_system_args_dict.

        This is the knowledge base counterpart to get_role_play_args_dict; the two are alternatives, and a given script
        pairs exactly one of them with get_system_args_dict.

        :param args: The already-parsed argparse namespace. Parsing happens once, in get_args_dict_knowledge_base.
        :param log_func: The function that prints whatever we are targeting. This will usually be 'print' or some form of 'logger.info', 'logger.error', etc.
        :return: A dictionary of the knowledge base settings.
        """

        argDict = {}
        use_default_arg_config = True  # This is only flipped if we successfully load from a JSON file

        json_config_file = args.knowledge_base_config_json

        if json_config_file and os.path.exists(json_config_file):
            try:
                config_dict = LlamaUtils.load_knowledge_base_json_config(json_config_file)

                argDict['knowledge_base_file'] = config_dict['knowledge_base_file']
                argDict['system_prompt_file'] = config_dict['system_prompt_file']

                argDict['debug'] = config_dict.get('debug', False)

                log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_knowledge_base_args_dict: Knowledge base config loaded from JSON file {json_config_file}; the knowledge base is '{argDict['knowledge_base_file']}' and the prompt is from '{argDict['system_prompt_file']}'.{ColoredText.END_TEXT}")
                use_default_arg_config = False

            except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
                log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_knowledge_base_args_dict: Could not load knowledge base JSON config [{json_config_file}] - there are errors. Will attempt to load other defaults or args. Error: {e}.{ColoredText.END_TEXT}")

        elif json_config_file:
            log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_knowledge_base_args_dict: Could not load knowledge base JSON config [{json_config_file}] - file does not exist. Loading from defaults or other parameters sent.{ColoredText.END_TEXT}")

        if use_default_arg_config:

            argDict['knowledge_base_file'] = args.knowledge_base_file
            argDict['system_prompt_file'] = args.system_prompt_file

            argDict['debug'] = args.debug

            log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_knowledge_base_args_dict: Knowledge base config loaded from args / defaults; the knowledge base is '{argDict['knowledge_base_file']}' and the prompt is from '{argDict['system_prompt_file']}'.{ColoredText.END_TEXT}")

        return argDict


    @staticmethod
    def get_args_dict_role_play_server(default_host: str, default_port: int, log_func: Callable[[str], None] = print) -> dict:
        """
        Gets the args dictionary for the ROLE-PLAY SERVER.

        Unlike the interactive scripts, this takes ONE config file ('--json') covering everything, not a system half and
        a story half. The interactive split exists to separate the machine from the single story being told; a server is
        never telling one story - each client names its own player and picks its own system prompt out of
        '--system-prompt-dir' when it opens a session - so there is nothing story-shaped left to split off. See
        add_monolithic_json_argument.

        The argument DEFINITIONS are still shared with every other entry point in this tree (add_system_arguments and
        friends); it is only the config file layout that differs.

        :param default_host: The hostname/IP the server binds to when '--host' is not given. Owned by the server class.
        :param default_port: The port the server listens on when '--port' is not given. Owned by the server class.
        :param log_func: The function that prints whatever we are targeting. This will usually be 'print' or some form of 'logger.info', 'logger.error', etc.
        :return: The dictionary of settings. Empty if argument parsing failed or '--help' was used.
        """

        parser = argparse.ArgumentParser(description='Run a LLM role-play server, as you see fit.')
        LlamaUtils.add_system_arguments(parser, include_config_json=False)
        LlamaUtils.add_role_play_server_arguments(parser)
        LlamaUtils.add_server_arguments(parser, default_host, default_port)
        LlamaUtils.add_monolithic_json_argument(parser)

        argDict = {}

        try:
            args = parser.parse_args()
            use_default_arg_config = True  # This is only flipped if we successfully load from a JSON file

            json_config_file = args.json

            if json_config_file and os.path.exists(json_config_file):
                try:
                    config_dict = LlamaUtils.load_role_play_server_json_config(json_config_file)

                    argDict['host'] = config_dict['host']
                    argDict['port'] = config_dict['port']

                    argDict['generating_model'] = os.path.join(config_dict['base_model_dir'], config_dict['model'])
                    argDict['embedding_model'] = os.path.join(config_dict['base_embedding_dir'], config_dict['embedding_model'])

                    # The base directory is kept whole; each session appends 'user_id/system_prompt_id' to it when it is
                    # created, so joining anything on at this point would be wrong.
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

                    # Optional so that a config written before GPU selection existed still loads. These fall back to
                    # the CLASS CONSTANTS, not to the command line: once this JSON loads it owns every setting, exactly
                    # as it did before GPU selection was added. See the same note in get_system_args_dict.
                    argDict['gpu'] = config_dict.get('gpu', LlamaUtils.GPU_INDEX)
                    argDict['split_gpus'] = config_dict.get('split_gpus', False)

                    log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_args_dict_role_play_server: Config loaded from JSON file {json_config_file}; prompt directory is '{argDict['system_prompt_dir']}' and base convo directory is '{argDict['base_convo_dir']}'.{ColoredText.END_TEXT}")
                    use_default_arg_config = False

                except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
                    log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict_role_play_server: Could not load JSON config [{json_config_file}] - there are errors. Will attempt to load other defaults or args. Error: {e}.{ColoredText.END_TEXT}")

            elif json_config_file:
                log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict_role_play_server: Could not load JSON config [{json_config_file}] - file does not exist. Loading from defaults or other parameters sent.{ColoredText.END_TEXT}")

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

                argDict['gpu'] = args.gpu
                argDict['split_gpus'] = args.split_gpus

                log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_args_dict_role_play_server: Config loaded from args / defaults; prompt directory is '{argDict['system_prompt_dir']}' and base convo directory is '{argDict['base_convo_dir']}'.{ColoredText.END_TEXT}")

            # Note that there is deliberately no 'system_message' here: a role-play server does not have one system
            # prompt, it has a directory of them, and it resolves the right one per session from the client's
            # 'system_prompt_id'. See RolePlayStream.create_session.

        except SystemExit as e:
            argDict = {}
            if e.code == 0:
                # --help was used, so print no error
                log_func(f"{ColoredText.BLUE_TEXT}Thank you!{ColoredText.END_TEXT}")
            else:
                log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict_role_play_server: Invalid arguments.{ColoredText.END_TEXT}")

        return argDict


    @staticmethod
    def get_args_dict_knowledge_base_server(default_host: str, default_port: int, log_func: Callable[[str], None] = print) -> dict:
        """
        Gets the args dictionary for the KNOWLEDGE BASE SERVER.

        As with the role-play server, this takes ONE config file ('--json') covering everything rather than the two-file
        split the interactive scripts use - see add_monolithic_json_argument. The argument DEFINITIONS are still shared
        with every other entry point in this tree.

        :param default_host: The hostname/IP the server binds to when '--host' is not given. Owned by the server class.
        :param default_port: The port the server listens on when '--port' is not given. Owned by the server class.
        :param log_func: The function that prints whatever we are targeting. This will usually be 'print' or some form of 'logger.info', 'logger.error', etc.
        :return: The dictionary of settings, plus the loaded 'system_message'. Empty if argument parsing failed or '--help' was used.
        """

        parser = argparse.ArgumentParser(description='Run a LLM knowledge base server, as you see fit.')
        LlamaUtils.add_system_arguments(parser, include_config_json=False)
        LlamaUtils.add_knowledge_base_arguments(parser, include_config_json=False)
        LlamaUtils.add_server_arguments(parser, default_host, default_port)
        LlamaUtils.add_monolithic_json_argument(parser)

        argDict = {}

        try:
            args = parser.parse_args()
            use_default_arg_config = True  # This is only flipped if we successfully load from a JSON file

            json_config_file = args.json

            if json_config_file and os.path.exists(json_config_file):
                try:
                    config_dict = LlamaUtils.load_knowledge_base_server_json_config(json_config_file)

                    argDict['host'] = config_dict['host']
                    argDict['port'] = config_dict['port']

                    argDict['generating_model'] = os.path.join(config_dict['base_model_dir'], config_dict['model'])
                    argDict['embedding_model'] = os.path.join(config_dict['base_embedding_dir'], config_dict['embedding_model'])

                    argDict['knowledge_base_file'] = config_dict['knowledge_base_file']
                    argDict['system_prompt_file'] = config_dict['system_prompt_file']

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

                    # Optional so that a config written before GPU selection existed still loads. These fall back to
                    # the CLASS CONSTANTS, not to the command line: once this JSON loads it owns every setting, exactly
                    # as it did before GPU selection was added. See the same note in get_system_args_dict.
                    argDict['gpu'] = config_dict.get('gpu', LlamaUtils.GPU_INDEX)
                    argDict['split_gpus'] = config_dict.get('split_gpus', False)

                    log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_args_dict_knowledge_base_server: Config loaded from JSON file {json_config_file}; system prompt file is '{argDict['system_prompt_file']}'.{ColoredText.END_TEXT}")
                    use_default_arg_config = False

                except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
                    log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict_knowledge_base_server: Could not load JSON config [{json_config_file}] - there are errors. Will attempt to load other defaults or args. Error: {e}.{ColoredText.END_TEXT}")

            elif json_config_file:
                log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict_knowledge_base_server: Could not load JSON config [{json_config_file}] - file does not exist. Loading from defaults or other parameters sent.{ColoredText.END_TEXT}")

            if use_default_arg_config:

                argDict['host'] = args.host
                argDict['port'] = args.port

                argDict['generating_model'] = os.path.join(args.base_model_dir, args.model)
                argDict['embedding_model'] = os.path.join(args.base_embedding_dir, args.embedding_model)

                argDict['knowledge_base_file'] = args.knowledge_base_file
                argDict['system_prompt_file'] = args.system_prompt_file

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
                argDict['model_type'] = args.model_type
                argDict['chat_format'] = args.chat_format

                argDict['gpu'] = args.gpu
                argDict['split_gpus'] = args.split_gpus

                log_func(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_args_dict_knowledge_base_server: Config loaded from args / defaults; system prompt file is '{argDict['system_prompt_file']}'.{ColoredText.END_TEXT}")

            argDict['system_message'] = LlamaUtils.get_system_message(argDict['system_prompt_file'])

        except SystemExit as e:
            argDict = {}
            if e.code == 0:
                # --help was used, so print no error
                log_func(f"{ColoredText.BLUE_TEXT}Thank you!{ColoredText.END_TEXT}")
            else:
                log_func(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict_knowledge_base_server: Invalid arguments.{ColoredText.END_TEXT}")

        return argDict


    @staticmethod
    def scrape_json_config(filepath: str, required_fields: dict, optional_fields: dict) -> dict:
        """
        Loads a JSON file and scrapes the given entries into a dictionary, validating that every required field is
        present and that every field it does find is of the expected type. This is the shared engine behind every
        config loader in the llama tree - load_system_json_config, load_role_play_json_config,
        load_knowledge_base_json_config, and the two stream classes' loaders - it holds no opinion about which fields
        belong to which config, it only enforces the field maps it is handed.

        Args:
            filepath (str): The path to the JSON file.
            required_fields (dict): A map of field name to expected type; a missing field here is an error.
            optional_fields (dict): A map of field name to expected type; a missing field here is simply skipped, and
                                    the caller is expected to supply the default. A type may also be a tuple of types
                                    (for example '(str, type(None))' for a field that is allowed to be null).

        Returns:
            dict: A dictionary containing the scraped configuration fields. Optional fields that were absent from the
                  JSON are absent from this dictionary as well.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file content is not valid JSON.
            KeyError: If any of the required fields are missing from the JSON.
            TypeError: If a field's value is not of the expected type.
        """

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

        def type_name(expected_type) -> str:
            """
            Renders a type - or a tuple of types - as a readable name for the error messages below.
            """
            if isinstance(expected_type, tuple):
                return " or ".join(t.__name__ for t in expected_type)
            return expected_type.__name__

        scraped_data = {}

        # Process required fields
        for field, expected_type in required_fields.items():
            if field not in data:
                raise KeyError(f"Error: Required field '{field}' missing from JSON in '{filepath}'.")

            value = data[field]
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Error: Field '{field}' in '{filepath}' has unexpected type "
                    f"'{type(value).__name__}', expected '{type_name(expected_type)}'."
                )
            scraped_data[field] = value

        # Process optional fields
        for field, expected_type in optional_fields.items():
            if field in data:  # Only process if present
                value = data[field]
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Error: Optional field '{field}' in '{filepath}' has unexpected type "
                        f"'{type(value).__name__}', expected '{type_name(expected_type)}'."
                    )
                scraped_data[field] = value

        return scraped_data


    @staticmethod
    def load_system_json_config(filepath: str) -> dict:
        """
        Loads a SYSTEM config JSON file - the settings that describe the machine and the models running on it - and
        scrapes its entries into a dictionary. This is the file passed via '--system-config-json'.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict: A dictionary containing the scraped configuration fields. All fields are required except
                  'model_type', 'chat_format', 'gpu', 'split_gpus', and 'repeat_penalty'. An example of a JSON doc:
            {
                "base_model_dir": "/home/kevin/ai/models/llama.cpp",
                "base_embedding_dir": "/home/kevin/ai/models/llama.cpp/embedding_models",
                "model": "llama-3-70b.Q4_K_M.gguf",
                "model_type": "llama-3",
                "chat_format": null,
                "embedding_model": "nomic-embed-text-v1.5.Q5_K_M.gguf",

                "gpu": 0,
                "split_gpus": false,
                "gpu_layers": 57,
                "embedding_gpu_layers": -1,
                "max_context_tokens": 4096,
                "embedding_max_context_tokens": 512,
                "max_response_tokens": 512,

                "repeat_penalty": 1.1,

                "max_vector_database_pcnt": 0.2,
                "buffer_context_pcnt": 0.05,

                "top_k": 4,
                "min_vector_db_score": 0.46
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

        # Optional fields with their types; if these are absent, the caller falls back to the class defaults.
        # 'gpu', 'split_gpus', and 'repeat_penalty' are optional rather than required so that a system config written
        # before they existed still loads cleanly - a missing 'gpu' simply means the first GPU.
        optional_fields = {
            'model_type': str,
            'chat_format': (str, type(None)),  # 'chat_format' is usually left null so llama.cpp can work it out itself
            'gpu': int,
            'split_gpus': bool,
            'repeat_penalty': float
        }

        return LlamaUtils.scrape_json_config(filepath, required_fields, optional_fields)


    @staticmethod
    def load_role_play_json_config(filepath: str) -> dict:
        """
        Loads a ROLE-PLAY config JSON file - the settings that describe the story being told rather than the machine
        telling it - and scrapes its entries into a dictionary. This is the file passed via '--role-play-config-json'.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict: A dictionary containing the scraped configuration fields. All fields are required except
                  'player_name', 'encrypted', and 'debug'. An example of a JSON doc:
            {
                "player_name": "Kevin",
                "base_convo_dir": "/home/kevin/ai/chat_history",
                "convo_name": "my_chat",
                "override_base_convo_dir": true,
                "system_prompt_file": "/home/kevin/ai/system_prompt.txt",

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
            'base_convo_dir': str,
            'convo_name': str,
            'override_base_convo_dir': bool,
            'system_prompt_file': str
        }

        # Optional fields with their types; if these are absent, the caller falls back to the class defaults
        optional_fields = {
            'player_name': str,
            'encrypted': bool,
            'debug': bool
        }

        return LlamaUtils.scrape_json_config(filepath, required_fields, optional_fields)


    @staticmethod
    def load_role_play_server_json_config(filepath: str) -> dict:
        """
        Loads the ROLE-PLAY SERVER config JSON file - ALL of the server's settings in one file. This is the file passed
        via '--json'.

        Unlike the interactive scripts, the server keeps one config rather than a system half and a story half: the
        client names its own player and picks its own system prompt out of 'system_prompt_dir' when it opens a session,
        so there is no per-story half to separate out. See add_monolithic_json_argument.

        Note 'system_prompt_dir' rather than the interactive script's 'system_prompt_file' - the server needs a whole
        directory of prompts to choose from, keyed by the 'system_prompt_id' each client sends.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict: A dictionary containing the scraped configuration fields. All fields are required except
                  'model_type', 'chat_format', 'encrypted', 'debug', 'gpu', and 'split_gpus'. An example of a JSON doc:
            {
                "host": "127.0.0.1",
                "port": 65440,

                "base_model_dir": "/home/kevin/ai/models/llama.cpp",
                "base_embedding_dir": "/home/kevin/ai/models/llama.cpp/embedding_models",
                "model": "llama-3-70b.Q4_K_M.gguf",
                "model_type": "llama-3",
                "chat_format": null,
                "embedding_model": "nomic-embed-text-v1.5.Q5_K_M.gguf",

                "base_convo_dir": "/home/kevin/ai/chat_history",
                "system_prompt_dir": "/home/kevin/ai/",

                "gpu": 0,
                "split_gpus": false,
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
            'system_prompt_dir': str,

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

        # Optional fields with their types; if these are absent, the caller falls back to the class defaults.
        # 'gpu' and 'split_gpus' are optional so that a server config written before GPU selection existed still loads.
        optional_fields = {
            'model_type': str,
            'chat_format': (str, type(None)),  # 'chat_format' is usually left null so llama.cpp can work it out itself
            'encrypted': bool,
            'debug': bool,
            'gpu': int,
            'split_gpus': bool
        }

        return LlamaUtils.scrape_json_config(filepath, required_fields, optional_fields)


    @staticmethod
    def load_knowledge_base_server_json_config(filepath: str) -> dict:
        """
        Loads the KNOWLEDGE BASE SERVER config JSON file - ALL of the server's settings in one file. This is the file
        passed via '--json'.

        As with the role-play server, this is one file rather than the interactive script's two - see
        add_monolithic_json_argument.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict: A dictionary containing the scraped configuration fields. All fields are required except
                  'model_type', 'chat_format', 'debug', 'gpu', and 'split_gpus'. An example of a JSON doc:
            {
                "host": "127.0.0.1",
                "port": 65450,

                "base_model_dir": "/home/kevin/ai/models/llama.cpp",
                "base_embedding_dir": "/home/kevin/ai/models/llama.cpp/embedding_models",
                "model": "llama-3-70b.Q4_K_M.gguf",
                "model_type": "llama-3",
                "chat_format": null,
                "embedding_model": "nomic-embed-text-v1.5.Q5_K_M.gguf",

                "knowledge_base_file": "/home/kevin/ai/knowledge_base.jsonl",
                "system_prompt_file": "/home/kevin/ai/system_prompt.txt",

                "gpu": 0,
                "split_gpus": false,
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

            'knowledge_base_file': str,
            'system_prompt_file': str,

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

        # Optional fields with their types; if these are absent, the caller falls back to the class defaults.
        # 'gpu' and 'split_gpus' are optional so that a server config written before GPU selection existed still loads.
        optional_fields = {
            'model_type': str,
            'chat_format': (str, type(None)),  # 'chat_format' is usually left null so llama.cpp can work it out itself
            'debug': bool,
            'gpu': int,
            'split_gpus': bool
        }

        return LlamaUtils.scrape_json_config(filepath, required_fields, optional_fields)


    @staticmethod
    def load_knowledge_base_json_config(filepath: str) -> dict:
        """
        Loads a KNOWLEDGE BASE config JSON file - the settings that describe the body of knowledge being answered from
        rather than the machine answering - and scrapes its entries into a dictionary. This is the file passed via
        '--knowledge-base-config-json'.

        The machine-side settings are NOT here; they live in the system config, which the knowledge base shares
        verbatim with the role-play script. See load_system_json_config.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict: A dictionary containing the scraped configuration fields. All fields are required except 'debug'.
                  An example of a JSON doc:
            {
                "knowledge_base_file": "/home/kevin/ai/knowledge_base.jsonl",
                "system_prompt_file": "/home/kevin/ai/system_prompt.txt",

                "debug": false
            }

        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file content is not valid JSON.
            KeyError: If any of the required fields are missing from the JSON.
            TypeError: If a field's value is not of the expected type.
        """

        required_fields = {
            'knowledge_base_file': str,
            'system_prompt_file': str
        }

        # Optional fields with their types; if these are absent, the caller falls back to the class defaults
        optional_fields = {
            'debug': bool
        }

        return LlamaUtils.scrape_json_config(filepath, required_fields, optional_fields)


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
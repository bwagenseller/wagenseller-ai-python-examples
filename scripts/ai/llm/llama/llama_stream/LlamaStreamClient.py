import os
import sys
import select
import socket

# Line editing for the '>>:' prompt. Importing readline is all it takes: input() then supports the arrow keys, Home/End,
# Ctrl-A/Ctrl-E and word jumps for fixing a typo mid-line, and Up/Down to recall this session's earlier inputs. Without
# it the arrow keys just insert escape codes such as '^[[D'. The history stays in memory only - nothing typed is written
# to disk. Optional, because some Python builds lack the module; the script works the same without it.
try:
    import readline  # noqa: F401 - imported for its side effect on input()
except ImportError:
    pass
import logging
from typing import Optional

from amadeo_utils.colored_text import ColoredText
from amadeo_utils.ai.llm.llama.subjective_constants import SubjectiveConstants
from amadeo_utils.client.amadeo_client import AmadeoClient
import argparse
import json

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class LlamaStreamClient:

    HOST = '127.0.0.1'
    PORT = 65440

    SPOKEN_RESPONSE = False
    USER_ID = 'Bob'
    MODE = 'role_play'
    # The three server families this one client talks to. Anything else is refused at start-up: an unrecognised mode
    # used to be sent as a knowledge-base session without a word, so a typo ("tool", "roleplay") opened the wrong kind
    # of session.
    VALID_MODES = ('role_play', 'knowledge_base', 'agent')
    CONTINUOUS_SAVE = False
    LOAD_PREVIOUS = True

    EXIT_PREFIX = '!exit'
    QUIT_PREFIX = '!quit'

    # The agent server's approval question is answered on a line starting '??', so it can never be mistaken for the
    # ordinary '>>' prompt. Only 'y' or 'yes' (any case) approves; anything else - including just Enter - is no.
    APPROVAL_PROMPT = '?? '
    APPROVAL_YES = ('y', 'yes')

    # How long to wait for a reply. A tool turn can run for the server's 'max_turn_seconds' (default 180) plus however
    # long the user takes over a '??' question, so agent mode waits longer than a plain chat reply ever needs.
    RESPONSE_TIMEOUT = 120
    TOOLS_RESPONSE_TIMEOUT = 600

    def __init__(self, argsDict: dict):
        self.argsDict = argsDict
        self.host = self.argsDict['host']
        self.port = self.argsDict['port']

        timeout = LlamaStreamClient.TOOLS_RESPONSE_TIMEOUT if self.argsDict.get('mode') == 'agent' else LlamaStreamClient.RESPONSE_TIMEOUT
        self.socket_client = AmadeoClient(self.host, self.port, additional_server_response_functionality = self.handle_server_response, persistent_request_timeout = timeout,
                                          interim_response_functionality = self.handle_interim_message)

    def handle_interim_message(self, message: dict) -> dict:
        """
        Answers a question the server asks part-way through a request - the agent server's request to approve a call.

        Shows the server's question (which names the tool and its exact arguments), then reads the answer on a '??'
        line. The server treats only 'y' / 'yes' as approval, but the answer is normalised here as well so what the
        user sees and what is sent always agree.

        Args:
            message: The interim message from the server.

        Returns:
            dict: the fields to send back to the server.
        """
        if message.get('type') != 'approval_request':
            return {'command': 'approval_response', 'answer': 'no'}
        timeout = float(message.get('timeout_seconds', 120))
        print(f"\n{ColoredText.YELLOW_TEXT}{message.get('message', 'Allow this tool call?')}{ColoredText.END_TEXT}")
        print(f"{ColoredText.BLUE_TEXT}(y/yes to allow; anything else, or no answer within {timeout:g} seconds, is no){ColoredText.END_TEXT}")
        # This client owns the deadline, not the server: if it waited past the server's, a late answer would arrive
        # as a stray request and every reply after it would be one behind. select() is only a readiness check, so the
        # line is still read by input() with its usual editing keys.
        print(LlamaStreamClient.APPROVAL_PROMPT, end='', flush=True)
        answer = ''
        try:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                answer = input().strip().lower()
            else:
                print()
        except EOFError:
            answer = ''
        approved = answer in LlamaStreamClient.APPROVAL_YES
        print(f"{ColoredText.BLUE_TEXT}{'Approved.' if approved else 'Not approved.'}{ColoredText.END_TEXT}")
        return {'command': 'approval_response', 'answer': 'yes' if approved else 'no'}

    def handle_server_response(self, response, raw_data):
        """
        Callback function to handle server responses from AmadeoClient
        """
        if response:

            if response.get("success"):
                if response.get('type') == 'llm_response':
                    text_response = response.get("response")
                    elapsed_time = response.get('elapsed_time')

                    print(f"\n{text_response}\n{ColoredText.BLUE_TEXT}({elapsed_time} Seconds){ColoredText.END_TEXT}")
                elif response.get('type') == 'system_message':
                    text_response = response.get("message")

                    print(f"\nSystem Message:\n{text_response}\n{ColoredText.BLUE_TEXT}")


            else:
                # Handle different error/status types
                if response.get('type') == 'error':
                    print(f"\n{ColoredText.RED_TEXT}Error from server: {response.get('message')} (lapsed time: {response.get('elapsed_time')} seconds).\n{ColoredText.END_TEXT}")

    def open_llm_session(self):
        """
        Asks the server for an LLM session on the current connection, with this client's settings - once at start-up,
        and again after a reconnect (see recover_connection).
        """
        if self.argsDict['mode'] == 'role_play':
            self.socket_client.send_persistent_request(
                command="create_llm_session",
                message="Request to LLM",
                binary_data=None,
                user_id=self.argsDict['user_id'],
                player_name=self.argsDict['player_name'],
                system_prompt_id=self.argsDict['system_prompt_id'],
                spoken_response=self.argsDict['spoken_response'],
                continuous_save=self.argsDict['continuous_save'],
                load_previous=self.argsDict['load_previous']
            )
        elif self.argsDict['mode'] == 'agent':
            # the agent server (tool-calling family, CS-21): like role play it takes a player name (for '@@NAME@@' in its
            # prompt) and a system_prompt_id, and saves and reloads
            self.socket_client.send_persistent_request(
                command="create_llm_session",
                message="Request to LLM",
                binary_data=None,
                user_id=self.argsDict['user_id'],
                player_name=self.argsDict['player_name'],
                system_prompt_id=self.argsDict['system_prompt_id'],
                spoken_response=self.argsDict['spoken_response'],
                continuous_save=self.argsDict['continuous_save'],
                load_previous=self.argsDict['load_previous']
            )
        else:
            # knowledge_base
            self.socket_client.send_persistent_request(
                command="create_llm_session",
                message="Request to LLM",
                binary_data=None,
                user_id=self.argsDict['user_id'],
                spoken_response=self.argsDict['spoken_response']
            )

    def recover_connection(self, user_input: str) -> bool:
        """
        Called when a request came back with no response. Says what happened, reconnects with a new session, and - only
        if the server had closed the connection, so the request never ran - sends the request again.

        The server closes a connection that sat idle past its client_idle_timeout_seconds (300 by default; the agent
        server's configs use 0 = never) and drops its session with it. Before this, the client said nothing and the
        next reply simply never came. A request that instead TIMED OUT here is not resent: the server may still be
        working on it, and running it twice could repeat a tool call.

        Args:
            user_input: The request that got no response.

        Returns:
            bool: True to carry on, False if the server cannot be reached (the client then exits).
        """
        error = getattr(self.socket_client, 'last_error', None)
        timed_out = isinstance(error, (socket.timeout, TimeoutError))
        if timed_out:
            print(f"\n{ColoredText.RED_TEXT}The server did not answer within {self.socket_client.persistent_request_timeout:g} seconds. Reconnecting; the request was not resent.{ColoredText.END_TEXT}")
        else:
            print(f"\n{ColoredText.RED_TEXT}Lost the connection to the server (it may have closed an idle session, or restarted). Reconnecting...{ColoredText.END_TEXT}")

        self.socket_client.close_connection()          # the old socket is dead; this also forgets the old session id
        if not self.socket_client.establish_persistent_connection():
            print(f"{ColoredText.RED_TEXT}Could not reconnect to the server. Exiting.{ColoredText.END_TEXT}")
            return False
        self.open_llm_session()
        restored = self.argsDict.get('continuous_save') and self.argsDict.get('load_previous') and self.argsDict['mode'] != 'knowledge_base'
        print(f"{ColoredText.BLUE_TEXT}Reconnected with a new session - "
              f"{'the saved conversation was reloaded.' if restored else 'the earlier conversation is not carried over.'}{ColoredText.END_TEXT}")

        if not timed_out:
            response, _ = self.socket_client.send_persistent_request(command="request", message="Request to LLM",
                                                                     binary_data=None, user_request=user_input)
            if response is None:
                print(f"{ColoredText.RED_TEXT}The request failed again after reconnecting ({self.socket_client.last_error}).{ColoredText.END_TEXT}")
        return True

    def graceful_shutdown(self):
        """Handles a clean shutdown of the client connection."""

        try:
            # Send end session command using the new client
            if hasattr(self.socket_client, 'is_persistent') and self.socket_client.is_persistent:
                self.socket_client.send_persistent_request("terminate_session", "Client shutting down")
                print(f"{ColoredText.BLUE_TEXT}Sent 'terminate_session' command to server.{ColoredText.END_TEXT}")

            # Close the connection
            self.socket_client.close_connection()

        except Exception as e:
            print(f"{ColoredText.RED_TEXT}Error during shutdown: {e}{ColoredText.END_TEXT}")

        print(f"{ColoredText.GREEN_TEXT}Connection closed. Exiting.{ColoredText.END_TEXT}")
        sys.exit(0)

    def run_client(self):

        try:
            print(f"{ColoredText.BLUE_TEXT}Attempting to connect to server on host: {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}{self.argsDict['host']}{ColoredText.END_TEXT}{ColoredText.BLUE_TEXT} port: {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}{self.argsDict['port']}{ColoredText.END_TEXT}")

            # Establish persistent connection
            if not self.socket_client.establish_persistent_connection():
                print(f"{ColoredText.RED_TEXT}Failed to establish connection. Exiting.{ColoredText.END_TEXT}")
                return

            logger.info(f"{ColoredText.BLUE_TEXT}Connected to server.{ColoredText.END_TEXT}")

            self.open_llm_session()

            while True:
                user_input = input("\n>>: ").strip()

                if user_input.lower().strip() in [LlamaStreamClient.EXIT_PREFIX, LlamaStreamClient.QUIT_PREFIX]:
                    print(f"{ColoredText.BLUE_TEXT}Exiting....{ColoredText.END_TEXT}")
                    break

                # Send using the persistent request method
                response, raw_data = self.socket_client.send_persistent_request(
                    command="request",
                    message="Request to LLM",
                    binary_data=None,
                    user_request=user_input
                )
                if response is None and not self.recover_connection(user_input):
                    break



        except Exception as e:
            print(f"{ColoredText.RED_TEXT}An unexpected error occurred: {e}{ColoredText.END_TEXT}")
        finally:
            pass

        self.graceful_shutdown()


    @staticmethod
    def get_args_dict() -> dict:
        """
        Gets args dictionary for a traditional vector database, meant to save the conversation for later.
        """
        ## NEW

        parser = argparse.ArgumentParser(description='Run a LLM, as you see fit.')
        parser.add_argument("-ho", "--host", default=LlamaStreamClient.HOST, help="The hostname/IP that the server will bind to.")
        parser.add_argument("-p", "--port", type=int, default=LlamaStreamClient.PORT, help="The port that the server will listen on for requests.")
        parser.add_argument("-mo", "--mode", type=str, default=LlamaStreamClient.MODE, choices=LlamaStreamClient.VALID_MODES, help="The mode of the chat: knowledge_base, role_play or agent (the tool-using agent server).")

        parser.add_argument("-pn", "--player-name", type=str, default=SubjectiveConstants.BASE_PLAYER_NAME,help=f"Give your username - How should the LLM address you? Leave blank if you do not want it addressing you directly via name.{ColoredText.RED_TEXT}NOT VALID{ColoredText.END_TEXT} for Knowledge Base instances.")
        parser.add_argument("-spf", "--system-prompt-id", type=str, default=SubjectiveConstants.SYSTEM_PROMPT_ID,help=f"The name or phrase that identifies the system prompt on the server that you wish to use.{ColoredText.RED_TEXT}NOT VALID{ColoredText.END_TEXT} for Knowledge Base instances.")

        parser.add_argument("-uid", "--user-id", type=str, default=LlamaStreamClient.USER_ID,help="A name or identification for this user. This is different from player_name- this is how the SYSTEM identifies you; think of it like an account.")
        parser.add_argument("-sr", "--spoken-response", type=bool, default=LlamaStreamClient.SPOKEN_RESPONSE,help="A name or identification for this user.")
        parser.add_argument("-cs", "--continuous-save", type=bool, default=LlamaStreamClient.CONTINUOUS_SAVE,help=f"True if you wish the conversation to be constantly saved so you can pick up the conversation later; False otherwise.{ColoredText.RED_TEXT}NOT VALID{ColoredText.END_TEXT} for Knowledge Base instances.")
        parser.add_argument("-lp", "--load-previous", type=bool, default=LlamaStreamClient.LOAD_PREVIOUS,help=f"True if you wish to load a previous conversation (if it exists) when you start (i.e. picking up where you previously left off); False otherwise.{ColoredText.RED_TEXT}NOT VALID{ColoredText.END_TEXT} for Knowledge Base instances.")

        parser.add_argument("-j", "--json", type=str, default="", help="If this points to a valid JSON file, the ENTIRE parameter settings are pulled from that file, and the defaults - and other arguments passed from the command line - are ignored. If the JSON load fails for whatever reason, though, the defaults WILL be engaged.")

        argDict = {}

        try:
            args = parser.parse_args()
            use_default_arg_config = True  # This is only flipped if we successfully load from a JSON file

            json_config_file = args.json

            if json_config_file and os.path.exists(json_config_file):
                try:
                    config_dict = LlamaStreamClient.load_json_config(json_config_file)

                    argDict['host'] = config_dict['host']
                    argDict['port'] = config_dict['port']

                    argDict['mode'] = config_dict['mode']
                    argDict['user_id'] = config_dict['user_id']
                    argDict['player_name'] = config_dict.get('player_name', SubjectiveConstants.BASE_PLAYER_NAME)
                    argDict['system_prompt_id'] = config_dict.get('system_prompt_id', SubjectiveConstants.SYSTEM_PROMPT_ID)
                    argDict['spoken_response'] = config_dict['spoken_response']
                    argDict['continuous_save'] = config_dict.get('continuous_save', LlamaStreamClient.CONTINUOUS_SAVE)
                    argDict['load_previous'] = config_dict.get('load_previous', LlamaStreamClient.LOAD_PREVIOUS)

                    print(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_args_dict: Config loaded from JSON file {json_config_file}; system_prompt_id is '{argDict['system_prompt_id']}', and continuous_save is '{argDict['continuous_save']}'.{ColoredText.END_TEXT}")
                    use_default_arg_config = False


                except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict: Could not load JSON config [{json_config_file}] - there are errors. Will attempt to load other defaults or args. Error: {e}.{ColoredText.END_TEXT}")


            elif json_config_file:
                print(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict: Could not load JSON config [{json_config_file}] - file does not exist. Loading from defaults or other parameters sent.{ColoredText.END_TEXT}")

            if use_default_arg_config:

                argDict['host'] = args.host
                argDict['port'] = args.port

                argDict['player_name'] = args.player_name
                argDict['system_prompt_id'] = args.system_prompt_id
                argDict['mode'] = args.mode
                argDict['user_id'] = args.user_id
                argDict['spoken_response'] = args.spoken_response
                argDict['continuous_save'] = args.continuous_save
                argDict['load_previous'] = args.load_previous

                print(f"{ColoredText.BLUE_TEXT}LlamaUtils.get_args_dict: Config loaded from args / defaults; system_prompt_id is '{argDict['system_prompt_id']}', and continuous_save is '{argDict['continuous_save']}'.{ColoredText.END_TEXT}")


        except SystemExit as e:
            argDict = {}
            if e.code == 0:
                # --help was used, so print no error
                print(f"{ColoredText.BLUE_TEXT}Thank you!{ColoredText.END_TEXT}")
            else:
                print(f"{ColoredText.RED_TEXT}LlamaUtils.get_args_dict: Invalid arguments.{ColoredText.END_TEXT}")

        if argDict and argDict.get('mode') not in LlamaStreamClient.VALID_MODES:
            print(f"{ColoredText.RED_TEXT}Unknown mode {argDict.get('mode')!r} - it must be one of: "
                  f"{', '.join(LlamaStreamClient.VALID_MODES)}.{ColoredText.END_TEXT}")
            argDict = {}

        return argDict


    @staticmethod
    def load_json_config(filepath: str) -> dict:
        """
        Loads a JSON file and scrapes specific entries into a dictionary.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict: A dictionary containing the scraped configuration fields. All fields are required. An example of a JSON doc (for Role Play):
            {
                "host": "127.0.0.1",
                "port": 65440,

                "mode": "role_play",

                "player_name": "Kevin",
                "user_id": "Kevin123",
                "system_prompt_id": "DungeonsAndDragons",
                "spoken_response": false,
                "continuous_save": false,
                "load_previous": true
            }

            An example of a JSON doc (for a Knowledge Base):
            {
                "host": "127.0.0.1",
                "port": 65440,

                "mode": "knowledge_base",

                "user_id": "Kevin123",
                "spoken_response": false
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
            'user_id': str,
            'mode': str,

            'spoken_response': bool,
        }

        # Add optional fields with their types
        optional_fields = {
            'player_name': str,
            'system_prompt_id': str,
            'continuous_save': bool,
            'load_previous': bool
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


if __name__ == "__main__":
    argsDict = LlamaStreamClient.get_args_dict()
    if not argsDict:
        sys.exit(2)                 # the reason (bad arguments, unknown mode) has already been printed
    client = LlamaStreamClient(argsDict)
    client.run_client()
import os
import sys
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
    CONTINUOUS_SAVE = False
    LOAD_PREVIOUS = True

    EXIT_PREFIX = '!exit'
    QUIT_PREFIX = '!quit'

    def __init__(self, argsDict: dict):
        self.argsDict = argsDict
        self.host = self.argsDict['host']
        self.port = self.argsDict['port']

        self.socket_client = AmadeoClient(self.host, self.port, additional_server_response_functionality = self.handle_server_response, persistent_request_timeout = 120)

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

            # Send using the persistent request method
            if self.argsDict['mode'] == 'role_play':
                response, raw_data = self.socket_client.send_persistent_request(
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
                response, raw_data = self.socket_client.send_persistent_request(
                    command="create_llm_session",
                    message="Request to LLM",
                    binary_data=None,
                    user_id=self.argsDict['user_id'],
                    spoken_response=self.argsDict['spoken_response']
                )


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
        parser.add_argument("-mo", "--mode", type=str, default=LlamaStreamClient.MODE, help="The mode of the chat: knowledge_base or role_play.")

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
    client = LlamaStreamClient(argsDict)
    client.run_client()
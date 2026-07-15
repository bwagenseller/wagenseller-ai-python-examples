from amadeo_utils.client.amadeo_client import AmadeoClient
from amadeo_utils.server.amadeo_server import AmadeoServer
import logging
import json
from amadeo_utils.colored_text import ColoredText
from typing import Dict, Any
import threading
import os
import argparse
from amadeo_utils.server.session_worker import SessionWorker

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

VALID_PIPELINES = ['reflection', 'translate_text', 'translate_voice', 'revoice', 'basic_conversational']

class ConversationalAiServer:

    HOST = 'localhost'
    PORT = 65400
    ASR_HOST = 'localhost'
    ASR_PORT = 65432
    TTS_HOST = 'localhost'
    TTS_PORT = 8888
    LLM_HOST = 'localhost'
    LLM_PORT = 65440

    def __init__(self, argsDict: dict):
        self.args_dict = argsDict
        self.session_workers = {}  # session_id -> SessionWorker
        self.workers_lock = threading.Lock()

        self.request_to_session_map = {}  # Simple dictionary that maps requestIDs to sessionIDs
        self.request_to_session_lock = threading.Lock()

        self.session_to_asr_client_map = {}  # Simple dictionary that maps sessionIDs to a tuple (asr clients, asr client locks); asr clients use a persistent connection, so we keep that connection for the whole session
        self.session_to_asr_client_lock = threading.Lock() # use this lock to interact with the asr client as well

        self.session_to_llm_client_map = {}  # Simple dictionary that maps sessionIDs to a tuple (llm clients, llm client locks); llm clients use a persistent connection, so we keep that connection for the whole session
        self.session_to_llm_client_lock = threading.Lock() # use this lock to interact with the llm client as well

        # Register global handlers for all sessions
        SessionWorker.register_global_handler('ping', self._handle_ping)
        SessionWorker.register_global_handler('status', self._handle_status)

        self.asr_host = argsDict['asr_host']
        self.asr_port = argsDict['asr_port']
        self.tts_host = argsDict['tts_host']
        self.tts_port = argsDict['tts_port']
        self.llm_host = argsDict['llm_host']
        self.llm_port = argsDict['llm_port']

        self.server = AmadeoServer(argsDict['host'], argsDict['port'],
                                 synchronous=False,
                                 additional_client_functionality=self.handle_client_request)


    def handle_client_request(self, request, client_binary_data):
        """
        The entrypoint of the code is here.

        Traditionally, this would accept the request, perform whatever needed to be done, and then return a dictionary and optional data to AmadeoServer - and AmadeoServer would take it from there
        This will no longer work, as we arent simply running it through a model (for example, ASR) and then returning a result - we are sending the request away for processing, THEN sending the results
        of that to ANOTHER service, THEN maybe sending that result to another service, THEN finally returning a result of some sort.

        Because of this, this method will now return 'basic_dictionary_that_will_not_be_used, None' to conform with what is expected, but a worker thread will actually handle the rest.

        This is achieved by setting the response host/port in the dictionary 'self.response_handlers' (with the sessionID as key), and then creating a 'job' and starting the queue process (with self.asr_queue).

        """

        session_id = request.get('sessionID') # comes from AmadeoServer

        # This is a new request - its entirely possible that a new request comes in while the old is being processed. Thus, the request_id is king for the lifespan, although we still need the session_id
        request_id = AmadeoServer.generate_session_id()
        request['requestID'] = request_id

        pipeline = request.get('pipeline')

        logger.info(f"{ColoredText.BLUE_TEXT}Handling client pipeline request '{pipeline}' for sessionID {session_id} - new requestID {request_id} created.{ColoredText.END_TEXT}")

        # Create session worker
        worker = self._create_worker(request_id, pipeline, request['_client_info'])

        # save the sessionID and requestID in the backpack
        worker.save_in_backpack('sessionID', session_id)
        worker.save_in_backpack('requestID', request_id)

        if pipeline == 'reflection' or pipeline == 'revoice':
            # outfit this job to interact with the ASR server

            if pipeline == 'revoice':
                # if this is a revoice, add a few more things
                voice = request.get('voice')
                worker.save_in_backpack('voice', voice)

            job = {
                'command': 'asr-send',
                'pipeline': pipeline,
                'sessionID': session_id,
                'requestID': request_id,
                'byte_data': client_binary_data,
                'request': request
            }
            worker.add_work(job)
        elif pipeline == 'basic_conversational':
            # Store user metadata for LLM first_request
            voice = request.get('voice')
            user_id = request.get('user_id')
            system_prompt_id = request.get('system_prompt_id', 'default')
            player_name = request.get('player_name', '')
            continuous_save = request.get('continuous_save', False)
            load_previous = request.get('load_previous', True)


            worker.save_in_backpack('voice', voice)
            worker.save_in_backpack('user_id', user_id)
            worker.save_in_backpack('system_prompt_id', system_prompt_id)
            worker.save_in_backpack('player_name', player_name)
            worker.save_in_backpack('continuous_save', continuous_save)
            worker.save_in_backpack('load_previous', load_previous)

            job = {
                'command': 'asr-send',
                'pipeline': pipeline,
                'sessionID': session_id,
                'requestID': request_id,
                'byte_data': client_binary_data,
                'request': request
            }
            worker.add_work(job)
        elif pipeline == 'terminate':
            self.remove_asr_client(session_id)
        else:
            logger.warning(f"{ColoredText.YELLOW_TEXT}Unknown command from sessionID {session_id}: {pipeline}{ColoredText.END_TEXT}")

        return {
            'success': True,
            'message': 'Pipeline queued',
            'sessionID': session_id
        }, None

    def _create_worker(self, request_id, pipeline, client_info):
        with self.workers_lock:
            if request_id not in self.session_workers:
                session_handlers = {
                    'asr-send': self._handle_asr_interaction,
                    'asr-receive': self.handle_asr_worker_drone,
                    'llm-send': self._handle_llm_interaction,
                    'llm-receive': self.handle_llm_worker_drone,
                    'tts-send': self._handle_tts_interaction,
                    'tts-receive': self.handle_tts_worker_drone
                    # Add other session-specific handlers here
                }

                logger.info(f"{ColoredText.BLUE_TEXT}Creating a worker for requestID {request_id}.{ColoredText.END_TEXT}")
                self.session_workers[request_id] = SessionWorker(
                    session_id=request_id,
                    client_socket=client_info['socket'],
                    address=client_info['address'],
                    parent_server=self,
                    pipeline=pipeline,
                    command_handlers=session_handlers,
                    max_workers=3
                )
            return self.session_workers[request_id]

    def _get_worker(self, request_id):
        with self.workers_lock:
            return self.session_workers[request_id]


    def _get_or_create_llm_client(self, session_id:str, request_id:str, user_id:str, player_name:str, system_prompt_id:str, continuous_save:bool = False, load_previous:bool = False):

        """
        Finds the llm_client by session ID and returns both the llm client and its lock in a tuple
        """
        with self.session_to_llm_client_lock:
            if session_id not in self.session_to_llm_client_map:
                logger.info(f"{ColoredText.BLUE_TEXT}Creating a LLM client for sessionID {session_id} to LLM host {self.llm_host} and LLM port {self.llm_port}.{ColoredText.END_TEXT}")
                llm_client = AmadeoClient( self.llm_host, self.llm_port, additional_server_response_functionality=self.handle_llm_server_response, session_id = session_id, request_id = request_id, persistent_request_timeout=120)


                llm_client_lock = threading.Lock() # use this lock to interact with the asr client as well

                if not llm_client.establish_persistent_connection():
                    logger.error(f"{ColoredText.RED_TEXT}Failed to establish LLM connection for worker with sessionID {session_id}!{ColoredText.END_TEXT}")
                else:

                    # Send using the persistent request method
                    # the first request to the LLM must establish some parameters
                    response, raw_data = llm_client.send_persistent_request(
                        command="create_llm_session",
                        message="Request to LLM",
                        binary_data=None,
                        user_id=user_id,
                        player_name=player_name,
                        system_prompt_id=system_prompt_id,
                        spoken_response=True,
                        continuous_save=continuous_save,
                        load_previous=load_previous
                    )

                self.session_to_llm_client_map[session_id] = (llm_client, llm_client_lock)
            llm_client, llm_client_lock = self.session_to_llm_client_map[session_id]
            return llm_client, llm_client_lock


    def _get_or_create_asr_client(self, session_id):
        """
        Finds the asr_client by session ID and returns both the asr client and its lock in a tuple
        """
        with self.session_to_asr_client_lock:
            if session_id not in self.session_to_asr_client_map:
                logger.info(f"{ColoredText.BLUE_TEXT}Creating an ASR client for sessionID {session_id} to ASR host {self.asr_host} and ASR port {self.asr_port}.{ColoredText.END_TEXT}")
                asr_client = AmadeoClient( self.asr_host, self.asr_port, additional_server_response_functionality=self.handle_asr_server_response, session_id = session_id)


                asr_client_lock = threading.Lock() # use this lock to interact with the asr client as well

                if not asr_client.establish_persistent_connection():
                    logger.error(f"{ColoredText.RED_TEXT}Failed to establish ASR connection for worker with sessionID {session_id}!{ColoredText.END_TEXT}")

                self.session_to_asr_client_map[session_id] = (asr_client, asr_client_lock)
            asr_client, asr_client_lock = self.session_to_asr_client_map[session_id]
            return asr_client, asr_client_lock


    def remove_session(self, session_id):
        """Called by SessionWorker during cleanup"""
        with self.session_to_asr_client_lock:
            # Yes, we know its resultID and not sessionID here - THE WORKERS HERE ARE DEFINED BY resultID
            logger.info(f"{ColoredText.BLUE_TEXT}Removed resultID {session_id} from workers threads.{ColoredText.END_TEXT}")
            self.session_workers.pop(session_id, None)

    def remove_asr_client(self, session_id):
        """Called by SessionWorker during cleanup"""
        with self.workers_lock:

            asr_client = self.session_to_asr_client_map.pop(session_id, None)
            if asr_client:
                asr_client.close_connection()
                logger.info(f"{ColoredText.BLUE_TEXT}Shut down and removed ASR client for session {session_id}.{ColoredText.END_TEXT}")

    def _handle_tts_interaction(self, worker, job):
        """
        job = {
            'command': 'tts-send',
            'pipeline': pipeline,
            'sessionID': session_id,
            'requestID': request_id,
            'voice': voice,
            'text': transcription
        }
        """
        session_id = job['sessionID']
        request_id = job['requestID']

        tts_client = AmadeoClient(self.tts_host, self.tts_port, additional_server_response_functionality = self.handle_tts_server_response)
        tts_client.send_transient_request('service_tts', '', text=job['text'], voice=job['voice'], requestID=request_id, sessionID = session_id)

        logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id} sent text to TTS.{ColoredText.END_TEXT}")
        # Response is handled automatically by handle_server_response callback

    def _handle_asr_interaction(self, worker, job):
        """
        Handles the asr action via a SessionWorker. Here, the SessionWorker will send the audio to the ASR server for processing

        Here is an example Job object for ASR, for reference
        job = {
            'command': 'asr',
            'sessionID': session_id,
            'requestID': request_id,
            'byte_data': client_binary_data,
            'request': request
        }

        """
        # This runs async in the worker's thread pool
        # Can call worker.save_results() to store intermediate results
        # Can access self.model, self.call_llm_service(), etc.

        session_id = job['sessionID']
        request_id = job['requestID']
        speech_segment_bytes = job['byte_data']

        # save the audio in the backpack
        worker.save_in_backpack('original_audio', speech_segment_bytes)


        asr_client, asr_client_lock = self._get_or_create_asr_client(session_id)
        with asr_client_lock:
            # we re-use self.session_to_asr_client_lock for the asr_client too

            asr_client.update_request_id(request_id)
            # Send using the new persistent request method with binary audio data
            # this already includes sessionID and requestID
            response, raw_data = asr_client.send_persistent_request(
                command="transcribe",
                message="Audio chunk for transcription",
                binary_data=speech_segment_bytes # Send as binary data after JSON
            )

        logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id} sent transcription to ASR.{ColoredText.END_TEXT}")
        # Response is handled automatically by handle_server_response callback

    def _handle_llm_interaction(self, worker, job):
        """
        job = {
            'command': 'llm-send',
            'pipeline': 'conversational',
            'sessionID': session_id,
            'requestID': request_id,
            'user_request': transcription,
            'user_id': user_id,
            'system_prompt_id': system_prompt_id,
            'player_name': player_name
        }
        """
        session_id = job['sessionID']
        request_id = job['requestID']
        llm_client, llm_client_lock = self._get_or_create_llm_client(session_id, request_id, job.get('user_id', 'Bob'), job.get('player_name', ''), job.get('system_prompt_id', 'default'), job.get('continuous_save', False), job.get('load_previous', False))
        with llm_client_lock:
            # we re-use self.session_to_llm_client_lock for the llm_client too

            llm_client.update_request_id(request_id)

            # Send using the persistent request method
            response, raw_data = llm_client.send_persistent_request(
                command="request",
                message="Request to LLM",
                binary_data=None,
                user_request=job['user_request']
            )

        logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id} sent transcription to LLM.{ColoredText.END_TEXT}")
        # Response is handled automatically by handle_server_response callback


    def handle_asr_server_response(self, response, raw_data):
        """
        Callback function to handle server responses from the ASR client. We quickly hand off to the worker, as its best to put the processing away from the main class running the server.

        The ASR server will pass back the sessionID, and if we include one, a requestID - and we did. Use this to find the worker and send it to the worker instead.
        """

        if response:
            session_id = response['sessionID']
            request_id = response['requestID']

            worker = self._get_worker(request_id)

            logger.info(f"{ColoredText.BLUE_TEXT}Got ASR server response for sessionID {session_id} amd requestID {request_id} - sending to worker.{ColoredText.END_TEXT}")

            # outfit this job to interact with the ASR server
            job = {
                'command': 'asr-receive',
                'pipeline': worker.get_pipeline,
                'sessionID': session_id,
                'requestID': request_id,
                'response': response
            }
            worker.add_work(job)
        else:
            logger.error(f"{ColoredText.RED_TEXT}ASR Server error: no response.{ColoredText.END_TEXT}")

    def handle_llm_server_response(self, response, raw_data):
        """Callback for LLM responses"""
        if response:
            session_id = response['sessionID']
            request_id = response['requestID']

            worker = self._get_worker(request_id)

            job = {
                'command': 'llm-receive',
                'pipeline': worker.get_pipeline(),
                'sessionID': session_id,
                'requestID': request_id,
                'response': response
            }
            worker.add_work(job)
        else:
            logger.error(f"{ColoredText.RED_TEXT}LLM Server error: no response.{ColoredText.END_TEXT}")

    def handle_tts_server_response(self, response, raw_data):
        """
        Callback function to handle server responses from the ASR client. We quickly hand off to the worker, as its best to put the processing away from the main class running the server.

        The ASR server will pass back the sessionID, and if we include one, a requestID - and we did. Use this to find the worker and send it to the worker instead.
        """

        if response:
            session_id = response['sessionID']
            request_id = response['requestID']

            worker = self._get_worker(request_id)

            logger.info(f"{ColoredText.BLUE_TEXT}Got TTS server response for sessionID {session_id} amd requestID {request_id} - sending to worker.{ColoredText.END_TEXT}")

            # outfit this job to interact with the ASR server
            job = {
                'command': 'tts-receive',
                'pipeline': worker.get_pipeline,
                'sessionID': session_id,
                'requestID': request_id,
                'response': response,
                'byte_data': raw_data
            }
            worker.add_work(job)
        else:
            logger.error(f"{ColoredText.RED_TEXT}TTS Server error: no response.{ColoredText.END_TEXT}")


    def handle_asr_worker_drone(self, worker, job):

        session_id = job['sessionID']
        request_id = job['requestID']
        response = job['response'] # response is, at least, not empty at this point - when it was packed it was verified to exist

        pipeline = worker.get_pipeline()

        logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id} processing ASR request for pipeline {pipeline}.{ColoredText.END_TEXT}")
        if response.get("success"):
            if response.get('type') == 'transcription':
                transcription = response.get("transcription")
                if transcription and transcription.strip():
                    # if the transcription is not blank or None, just pass
                    pass
                else:
                    logger.info(f"{ColoredText.BLUE_TEXT} TEXT IS BLANK for sessionID {session_id} amd requestID {request_id} for pipeline {pipeline}.{ColoredText.END_TEXT}")
                    transcription = "I didn't quite get that."
            elif response.get('type') == 'garbage_transcription':
                logger.info(f"{ColoredText.BLUE_TEXT} Garbage transcription for sessionID {session_id} amd requestID {request_id} for pipeline {pipeline} - ignoring.{ColoredText.END_TEXT}")

                to_client = {
                    'success': False,
                    'sessionID': session_id,
                    'requestID': request_id,
                    'file_size': 0,
                    'message': "Garbage transcription; ignoring."
                }
                worker.send_to_client(to_client, None)
                worker.shutdown()
                return
        else:
            logger.warning(f"{ColoredText.YELLOW_TEXT} ASR Response failed for sessionID {session_id} amd requestID {request_id} for pipeline {pipeline}.{ColoredText.END_TEXT}")
            transcription = "I didn't quite get that."


        if pipeline == 'reflection':

            client_binary_data = worker.get_from_backpack('original_audio')

            to_client = {
                'sessionID': session_id,
                'requestID': request_id,
                'success': True,
                'transcription': transcription
            }

            worker.send_to_client(to_client, client_binary_data)

            logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id} sending ASR transcription and audio back to client.{ColoredText.END_TEXT}")
            worker.shutdown()
        elif pipeline == 'revoice':
            voice = worker.get_from_backpack('voice')
            worker.save_in_backpack('transcription', transcription)

            job = {
                'command': 'tts-send',
                'pipeline': pipeline,
                'sessionID': session_id,
                'requestID': request_id,
                'voice': voice,
                'text': transcription
            }
            worker.add_work(job)
        elif pipeline == 'basic_conversational':
            worker.save_in_backpack('transcription', transcription)

            job = {
                'command': 'llm-send',
                'pipeline': pipeline,
                'sessionID': session_id,
                'requestID': request_id,
                'user_request': transcription,
                'user_id': worker.get_from_backpack('user_id'),
                'system_prompt_id': worker.get_from_backpack('system_prompt_id'),
                'player_name': worker.get_from_backpack('player_name'),
                'continuous_save': worker.get_from_backpack('continuous_save'),
                'load_previous': worker.get_from_backpack('load_previous')
            }
            worker.add_work(job)


    def handle_tts_worker_drone(self, worker, job):
        """
        job = {
            'command': 'tts-receive',
            'pipeline': worker.get_pipeline,
            'sessionID': session_id,
            'requestID': request_id,
            'response': response,
            'byte_data': raw_data
        }
        """
        session_id = job['sessionID']
        request_id = job['requestID']
        raw_data = job['byte_data']
        response = job['response'] # response is, at least, not empty at this point - when it was packed it was verified to exist
        transcription = worker.get_from_backpack('transcription')

        pipeline = worker.get_pipeline()

        logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id} processing TTS request for pipeline {pipeline}.{ColoredText.END_TEXT}")

        # if you want to handle other audio types or errors, do so here
        #if response.get("success"):
        #    if response.get('type') == 'audio':
        #    else:
        #        transcription = "I didn't quite get that."
        #else:
        #    transcription = "I didn't quite get that."


        if pipeline == 'revoice':

            to_client = {
                'sessionID': session_id,
                'requestID': request_id,
                'success': True,
                'transcription': transcription
            }

            worker.send_to_client(to_client, raw_data)

            logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id} for pipeline {pipeline} sending TTS transcription and audio back to client.{ColoredText.END_TEXT}")
            worker.shutdown()
        elif pipeline == 'basic_conversational':
            transcription = worker.get_from_backpack('transcription')
            llm_response = worker.get_from_backpack('llm_response')

            to_client = {
                'sessionID': session_id,
                'requestID': request_id,
                'success': True,
                'transcription': transcription,
                'llm_response': llm_response
            }

            worker.send_to_client(to_client, raw_data)
            logger.info(f"{ColoredText.BLUE_TEXT}Worker for sessionID {session_id} amd requestID {request_id} for pipeline {pipeline} sending TTS transcription and audio back to client.{ColoredText.END_TEXT}")
            worker.shutdown()


    def handle_llm_worker_drone(self, worker, job):
        """
        job = {
            'command': 'llm-receive',
            'pipeline': 'conversational',
            'sessionID': session_id,
            'requestID': request_id,
            'response': response
        }
        """
        session_id = job['sessionID']
        request_id = job['requestID']
        response = job['response']

        pipeline = worker.get_pipeline()

        if response.get("success"):
            llm_response = response.get("response", "I'm sorry, I didn't catch that.")
        else:
            llm_response = "I'm sorry, could you repeat that?"

        if pipeline == 'basic_conversational':
            voice = worker.get_from_backpack('voice')
            worker.save_in_backpack('llm_response', llm_response)

            job = {
                'command': 'tts-send',
                'pipeline': pipeline,
                'sessionID': session_id,
                'requestID': request_id,
                'voice': voice,
                'text': llm_response
            }
            worker.add_work(job)



    def _handle_ping(self, worker, job):
        """Global handler example"""
        logger.info(f"Ping from session {worker.session_id}")

    def _handle_status(self, worker, job):
        """Global handler example"""
        active_count = worker.get_active_command_count()
        logger.info(f"Session {worker.session_id} has {active_count} active commands")


    @staticmethod
    def load_json_config(filepath: str) -> dict:
        """
        Loads a JSON file and scrapes specific entries into a dictionary.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict: A dictionary containing the scraped configuration fields:


        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file content is not valid JSON.
            KeyError: If any of the required fields are missing from the JSON.
            TypeError: If a field's value is not of the expected type.
        """
        required_fields = {
            'host': str,
            'port': int,
        }

        # Add optional fields with their types
        optional_fields = {
            'asr_host': str,
            'asr_port': int,
            'llm_host': str,
            'llm_port': int,
            'tts_host': str,
            'tts_port': int
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
    def get_args_dict_server() -> dict:
        """
        Gets args dictionary for the conversational AI server.
        """

        # Set up command-line argument parsing
        parser = argparse.ArgumentParser(description='Conversational AI Suite - Multiple paths',formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('--host', default=ConversationalAiServer.HOST, help='Server host address (default: localhost)')
        parser.add_argument('--port', type=int, default=ConversationalAiServer.PORT, help=f"Server port number (default: {ConversationalAiServer.PORT})")

        parser.add_argument('--asr-host', default=ConversationalAiServer.ASR_HOST, help='The ASR server host address (default: localhost)')
        parser.add_argument('--asr-port', type=int, default=ConversationalAiServer.ASR_PORT, help=f"The ASR server port number (default: {ConversationalAiServer.ASR_PORT})")

        parser.add_argument('--tts-host', default=ConversationalAiServer.TTS_HOST, help='The TTS server host address (default: localhost)')
        parser.add_argument('--tts-port', type=int, default=ConversationalAiServer.TTS_PORT, help=f"The TTS server port number (default: {ConversationalAiServer.TTS_PORT})")

        parser.add_argument('--llm-host', default=ConversationalAiServer.LLM_HOST, help='The LLM server host address (default: localhost)')
        parser.add_argument('--llm-port', type=int, default=ConversationalAiServer.LLM_PORT, help=f"The LLM server port number (default: {ConversationalAiServer.LLM_PORT})")

        parser.add_argument("--json", type=str, default="", help="If this points to a valid JSON file, the ENTIRE parameter settings are pulled from that file, and the defaults - and other arguments passed from the command line - are ignored. If the JSON load fails for whatever reason, though, the defaults WILL be engaged. Just remember that if there is a dash in the arg name, its going to be an underscore in the JSON.")

        argDict = {}

        try:
            args = parser.parse_args()
            use_default_arg_config = True  # This is only flipped if we successfully load from a JSON file

            json_config_file = args.json

            if json_config_file and os.path.exists(json_config_file):
                try:
                    config_dict = ConversationalAiServer.load_json_config(json_config_file)


                    argDict['host'] = config_dict.get('host', ConversationalAiServer.HOST)
                    argDict['port'] = config_dict.get('port', ConversationalAiServer.PORT)

                    argDict['asr_host'] = config_dict.get('asr_host', ConversationalAiServer.ASR_HOST)
                    argDict['asr_port'] = config_dict.get('asr_port', ConversationalAiServer.ASR_HOST)

                    argDict['tts_host'] = config_dict.get('tts_host', ConversationalAiServer.TTS_HOST)
                    argDict['tts_port'] = config_dict.get('tts_port', ConversationalAiServer.TTS_HOST)

                    argDict['llm_host'] = config_dict.get('llm_host', ConversationalAiServer.LLM_HOST)
                    argDict['llm_port'] = config_dict.get('llm_port', ConversationalAiServer.LLM_HOST)

                    logger.info(f"Config loaded from JSON {json_config_file}.")

                    use_default_arg_config = False

                except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Could not load JSON config [{json_config_file}] - there are errors. Will attempt to load other defaults or args. Error: {e}.")

            elif json_config_file:
                logger.warning(f"Could not load JSON config [{json_config_file}] - file does not exist. Loading from defaults or other parameters sent.")

            if use_default_arg_config:
                argDict['host'] = args.host
                argDict['port'] = args.port

                argDict['asr_host'] = args.asr_host
                argDict['asr_port'] = args.asr_port

                argDict['tts_host'] = args.tts_host
                argDict['tts_port'] = args.tts_port

                argDict['llm_host'] = args.llm_host
                argDict['llm_port'] = args.llm_port

        except SystemExit as e:
            argDict = {}
            if e.code == 0:
                # --help was used, so print no error
                print(f"Thank you!")
            else:
                logger.error(f"Invalid arguments.")

        return argDict
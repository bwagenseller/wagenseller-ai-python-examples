import sounddevice as sd
import numpy as np
import signal
import sys
import threading
import webrtcvad
import argparse
import time
import wave
import io
import pygame


from amadeo_utils.colored_text import ColoredText
from amadeo_utils.client.amadeo_client import AmadeoClient
from amadeo_utils.media_utils.audio_devices import prefer_pulse_defaults
import logging
import os
import json
from amadeo_utils.ai.llm.llama.subjective_constants import SubjectiveConstants
from amadeo_utils.ai.combined.conversational_ai.conversational_ai import VALID_PIPELINES

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class ConversationalAiPipelineClient:

    SAMPLE_RATE = 16000 # in Hz. WhisperX EXCLUSIVELY uses 16k - anything else it will downsample. Might as well use it from the start.\, and hard-code it in.

    HOST = '127.0.0.1'
    PORT = 65400

    VAD_FRAME_DURATION_MS = 30
    VAD_AGGRESSIVENESS = 2
    SILENCE_DURATION_TO_END_BUFFER_MS = 800
    MIN_SPEECH_DURATION_MS = 300
    VOLUME_THRESHOLD = 0.03
    PIPELINE = 'basic_conversational'

    USER_ID = 'Bob'
    CONTINUOUS_SAVE = False
    LOAD_PREVIOUS = True

    # type hints
    socket_client: AmadeoClient


    """
    Constructor for WhisperXClient
    """
    def __init__(self, argsDict: dict):
        self.args_dict = argsDict
        self.VAD_FRAME_SIZE = int(ConversationalAiPipelineClient.SAMPLE_RATE * self.args_dict['vad_frame_duration'] / 1000)

        self.is_recording = True
        self.shutdown_lock = threading.Lock()

        self.pipeline = self.args_dict['pipeline']

        self.voice = self.args_dict['voice']

        self.player_name = self.args_dict['player_name']
        self.user_id = self.args_dict['user_id']
        self.system_prompt_id = self.args_dict['system_prompt_id']
        self.continuous_save = self.args_dict['continuous_save']
        self.load_previous = self.args_dict['load_previous']

        self.file_iterator = -1

        # Initialize pygame (for sound)
        pygame.mixer.init()
        logger.info("pygame initialized for audio playback.")

        # Initialize the new AmadeoClient
        self.socket_client = AmadeoClient( self.args_dict['host'], self.args_dict['port'], additional_server_response_functionality=self.handle_server_response)

    def handle_server_response(self, response, raw_data):
        """
        Callback function to handle server responses from AmadeoClient
        """

        file_size = 0
        if raw_data: file_size = len(raw_data)

        if response:
            if not response.get('success', False):
                # Handle different error/status types
                message = response.get("message", "Unknown error")
                if "busy" in message.lower():
                    logger.warning(f"{ColoredText.CYAN_TEXT}[Server busy, please wait for a moment.{ColoredText.END_TEXT}")
                elif "queued" in message.lower():
                    # Optionally show queued status
                    pass
                elif "garbage transcription" in message.lower():
                    logger.info(f"{ColoredText.BLUE_TEXT}Garbage transcription; please try again, as this was ignored (it was probably just a sound and not words).{ColoredText.END_TEXT}")
                else:
                    logger.error(f"{ColoredText.RED_TEXT}Server error: {message}{ColoredText.END_TEXT}")
            elif self.pipeline == 'reflection':
                transcription = response.get('transcription', '')
                sessionID = response.get('sessionID', '')
                requestID = response.get('requestID', '')


                # Save audio file with unique name
                self.file_iterator += 1
                output_file = f"audio_{self.file_iterator}.wav"

                #With 'reflection', we get back the SAME data we sent, which is in microphone format - its raw PCM and it does not have the proper .WAV headers expected
                #therefore, add those headers. Again, THIS DOES NOT HAVE TO BE DONE TO THE FILE RETURNED BY TTS
                self.create_wav_file(raw_data, output_file, ConversationalAiPipelineClient.SAMPLE_RATE)

                logger.info(f"sessionID: {sessionID} requestID: {requestID} audio file: {output_file} Transcription: {transcription}")

                # Play audio
                self.play_audio(output_file)
            elif self.pipeline == 'revoice':
                transcription = response.get('transcription', '')
                sessionID = response.get('sessionID', '')
                requestID = response.get('requestID', '')


                # Create in-memory audio buffer instead of file
                audio_buffer = io.BytesIO(raw_data)

                ## Save audio file with unique name
                #self.file_iterator += 1
                #output_file = f"audio_{self.file_iterator}.wav"

                #with open(output_file, 'wb') as f:
                #    f.write(raw_data)

                #logger.info(f"sessionID: {sessionID} requestID: {requestID} audio file: {output_file} Transcription: {transcription}")
                logger.info(f"sessionID: {sessionID} requestID: {requestID} Transcription: {transcription}")

                # Play audio
                self.play_audio(audio_buffer)
            elif self.pipeline == 'basic_conversational':
                transcription = response.get('transcription', '')
                llm_response = response.get('llm_response', '')
                sessionID = response.get('sessionID', '')
                requestID = response.get('requestID', '')


                # Create in-memory audio buffer instead of file
                audio_buffer = io.BytesIO(raw_data)

                ## Save audio file with unique name
                #self.file_iterator += 1
                #output_file = f"audio_{self.file_iterator}.wav"

                #with open(output_file, 'wb') as f:
                #    f.write(raw_data)

                logger.info(f"{ColoredText.YELLOW_TEXT}You:{ColoredText.END_TEXT} {transcription}")
                logger.info(f"{ColoredText.GREEN_TEXT}Response:{ColoredText.END_TEXT} {llm_response}")
                #logger.info(f"{ColoredText.BLUE_TEXT}sessionID: {sessionID} requestID: {requestID} audio file: {output_file}{ColoredText.END_TEXT}")
                logger.info(f"{ColoredText.BLUE_TEXT}sessionID: {sessionID} requestID: {requestID} {ColoredText.END_TEXT}")

                # Play audio
                self.play_audio(audio_buffer)


            else:
                logger.error(f"{ColoredText.RED_TEXT}Unknown pipeline received.{ColoredText.END_TEXT}")

    def create_wav_file(self, pcm_data, filename, sample_rate=16000, channels=1, sample_width=2):
        """Create a proper WAV file from raw PCM data"""
        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)  # 2 bytes for int16
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_data)
            logger.info(f"Created WAV file: {filename} ({len(pcm_data)} bytes PCM -> WAV)")
        except Exception as e:
            logger.error(f"Error creating WAV file: {e}")
            # Fallback: just write raw data
            with open(filename, 'wb') as f:
                f.write(pcm_data)

    def play_audio(self, audio_source):
        """
        Play audio from file path or BytesIO buffer using available library

        Args:
            audio_source: Either a file path (str/Path) or BytesIO buffer
        """
        try:
            # Check if it's a buffer or file path
            is_buffer = isinstance(audio_source, io.BytesIO)

            if is_buffer:
                logger.info("Playing audio from memory buffer")
                # pygame can load from file-like objects
                audio_source.seek(0)  # Reset buffer position to start
                pygame.mixer.music.load(audio_source)
            else:
                logger.info(f"Playing audio file: {audio_source}")
                pygame.mixer.music.load(str(audio_source))

            pygame.mixer.music.play()
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)


        except Exception as e:
            logger.error(f"Error playing audio: {e}")


    def graceful_shutdown(self):
        """Handles a clean shutdown of the client connection."""
        with self.shutdown_lock:
            if not self.is_recording:
                return

            logger.info(f"{ColoredText.YELLOW_TEXT}Ctrl+C detected. Shutting down gracefully...{ColoredText.END_TEXT}")
            self.is_recording = False

            try:
                # Send end session command using the new client
                if hasattr(self.socket_client, 'is_persistent') and self.socket_client.is_persistent:
                    self.socket_client.send_persistent_request("terminate_session", "Client shutting down")
                    logger.info(f"Sent 'terminate_session' command to server.")

                # Close the connection
                self.socket_client.close_connection()

            except Exception as e:
                logger.error(f"{ColoredText.RED_TEXT}Error during shutdown: {e}{ColoredText.END_TEXT}")

            logger.info(f"{ColoredText.GREEN_TEXT}Connection closed. Exiting.{ColoredText.END_TEXT}")
            sys.exit(0)

    def run_client(self):
        """
        The key insight of this algorithm is the three-state machine:
            Waiting: Monitoring for speech to start
            Accumulating: Building audio buffer during active speech
            Silence Tracking: Counting silent frames after speech to determine when to send

        The hybrid VAD + volume approach solves the original problem by ensuring both acoustic speech patterns AND sufficient volume are present before considering something "speech."

        :return:
        """
        signal.signal(signal.SIGINT, lambda s, f: self.graceful_shutdown())

        try:
            logger.info(f"{ColoredText.BLUE_TEXT}Attempting to connect to server on host: {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}{self.args_dict['host']}{ColoredText.END_TEXT}{ColoredText.BLUE_TEXT} port: {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}{self.args_dict['port']}{ColoredText.END_TEXT}")

            # Establish persistent connection
            if not self.socket_client.establish_persistent_connection():
                logger.error(f"{ColoredText.RED_TEXT}Failed to establish connection. Exiting.{ColoredText.END_TEXT}")
                return

            logger.info(f"{ColoredText.BLUE_TEXT}Starting microphone stream. Press Ctrl+C to exit.{ColoredText.END_TEXT}")

            # Initialize WebRTC Voice Activity Detection with configurable aggressiveness (0-3)
            # Higher values are more aggressive in filtering out non-speech
            vad = webrtcvad.Vad(self.args_dict['vad_aggressiveness'])

             # Buffer to accumulate audio frames that contain speech
            current_audio_buffer = bytearray()

            # Counter tracking consecutive silent frames after speech has been detected
            silent_frames_count = 0

            # Calculate how many consecutive silent frames constitute "end of speech"
            # Example: 800ms silence / 30ms per frame = ~27 frames
            silent_frames_threshold = int(self.args_dict['silence_duration'] / self.args_dict['vad_frame_duration'])

            # Flag indicating whether any speech has been detected in the current buffer
            # Prevents sending audio before the user has started speaking
            has_spoken = False


            # ============================================================================
            # MAIN AUDIO CAPTURE LOOP
            # ============================================================================

            # Route through PulseAudio where available: raw ALSA capture devices
            # reject SAMPLE_RATE (16 kHz) outright rather than resampling.
            prefer_pulse_defaults()

            # Open audio input stream with 16kHz sample rate (required by WhisperX)
            # int16 format provides 16-bit signed integer samples (-32768 to 32767)
            with sd.InputStream(samplerate=ConversationalAiPipelineClient.SAMPLE_RATE, channels=1, dtype='int16', blocksize=self.VAD_FRAME_SIZE) as stream:
                while self.is_recording:
                    # ========================================================================
                    # FRAME CAPTURE
                    # ========================================================================

                    # Read one frame of audio (30ms worth of samples at 16kHz = 480 samples)
                    audio_frame, _ = stream.read(self.VAD_FRAME_SIZE)

                    # Convert numpy array to raw bytes for VAD processing
                    int16_data = audio_frame.tobytes()

                    # ========================================================================
                    # HYBRID SPEECH DETECTION (VAD + VOLUME)
                    # ========================================================================

                    # Convert raw bytes back to numpy array for volume calculation
                    audio_array = np.frombuffer(int16_data, dtype=np.int16)

                    # Calculate RMS (Root Mean Square) volume as a measure of audio amplitude
                    # Steps:
                    # 1. Convert int16 to float32 for precision
                    # 2. Square all values (makes negatives positive, emphasizes louder samples)
                    # 3. Take the mean of squared values
                    # 4. Take square root to get RMS
                    # 5. Divide by 32768 (max int16 value) to normalize to 0-1 range
                    rms_volume = np.sqrt(np.mean(audio_array.astype(np.float32)**2)) / 32768.0

                    # Use WebRTC VAD to detect speech patterns (phonetic characteristics)
                    is_speech = vad.is_speech(int16_data, ConversationalAiPipelineClient.SAMPLE_RATE)

                    # Check if volume exceeds threshold (filters out quiet background noise)
                    is_loud_enough = rms_volume > ConversationalAiPipelineClient.VOLUME_THRESHOLD

                    # Require BOTH conditions: must sound like speech AND be loud enough
                    # This hybrid approach reduces false positives from:
                    # - VAD alone: can trigger on background noise with speech-like patterns
                    # - Volume alone: can trigger on non-speech loud sounds (door slam, cough)
                    is_actual_speech = is_speech and is_loud_enough

                    if is_actual_speech:
                        has_spoken = True

                        # Reset silence counter since we're actively receiving speech
                        silent_frames_count = 0

                        # Append this frame to our growing audio buffer
                        current_audio_buffer.extend(int16_data)
                        logger.debug(f"Speech detected (RMS: {rms_volume:.4f})")

                    elif has_spoken:
                         # We've previously detected speech, now tracking silence duration

                        # Increment counter for consecutive silent frames
                        silent_frames_count += 1

                        # Continue appending silent frames to buffer to avoid cutting off
                        # the end of speech abruptly (captures natural trailing off)
                        current_audio_buffer.extend(int16_data)

                        # Check if we've hit the silence threshold (e.g., 800ms of silence)
                        if silent_frames_count >= silent_frames_threshold:
                            # ================================================================
                            # PREPARE AUDIO CHUNK FOR TRANSMISSION
                            # ================================================================

                            # Remove the trailing silence frames from the buffer before sending
                            # Calculation: silent_frames_threshold * frame_size * 2_bytes_per_sample
                            # This gives us audio up to the START of the silence period
                            speech_segment_bytes = current_audio_buffer[:-silent_frames_threshold * self.VAD_FRAME_SIZE * 2]

                            # Calculate duration of the speech segment in milliseconds
                            # Formula: bytes / (sample_rate * bytes_per_sample) * 1000
                            # Example: 48000 bytes / (16000 Hz * 2 bytes) * 1000 = 1500ms
                            duration_ms = len(speech_segment_bytes) / (ConversationalAiPipelineClient.SAMPLE_RATE * 2) * 1000

                            if duration_ms >= ConversationalAiPipelineClient.MIN_SPEECH_DURATION_MS:
                                logger.info(f"{ColoredText.BLUE_TEXT}Silence detected after {duration_ms:.0f}ms of speech. Sending chunk...{ColoredText.END_TEXT}")

                                # Send using the persistent request method with binary audio data
                                logger.info(f"{ColoredText.BLUE_TEXT}Request sent to server - waiting on return...{ColoredText.END_TEXT}")
                                if self.pipeline == 'reflection':
                                    response, raw_data = self.socket_client.send_persistent_request(
                                        command=self.pipeline, # not really needed but is part of the structure, so - just repeat pipeline.
                                        message="Audio chunk",
                                        binary_data=speech_segment_bytes, # Send as binary data after JSON
                                        pipeline=self.pipeline # necessary
                                    )
                                elif self.pipeline == 'revoice':
                                    response, raw_data = self.socket_client.send_persistent_request(
                                        command=self.pipeline, # not really needed but is part of the structure, so - just repeat pipeline.
                                        message="Audio chunk",
                                        binary_data=speech_segment_bytes, # Send as binary data after JSON
                                        pipeline=self.pipeline, # necessary
                                        voice=self.voice
                                    )
                                elif self.pipeline == 'basic_conversational':
                                    response, raw_data = self.socket_client.send_persistent_request(
                                        command=self.pipeline, # not really needed but is part of the structure, so - just repeat pipeline.
                                        message="Audio chunk",
                                        binary_data=speech_segment_bytes, # Send as binary data after JSON
                                        pipeline=self.pipeline, # necessary
                                        voice=self.voice,
                                        user_id=self.user_id,
                                        system_prompt_id=self.system_prompt_id,
                                        player_name=self.player_name,
                                        continuous_save=self.continuous_save,
                                        load_previous=self.load_previous
                                    )
                                    # Response is handled automatically by handle_server_response callback

                                # Clear the buffer since we've successfully sent this chunk
                                current_audio_buffer = bytearray()
                                silent_frames_count = 0
                                has_spoken = False
                            else:
                                # The audio segment is shorter than minimum duration threshold
                                # This could be a brief utterance, mouth noise, or incomplete word
                                # Strategy: reset silence counter but KEEP the buffer
                                # If user continues speaking, we'll accumulate into same chunk
                                # If they don't, next silence period will try again

                                logger.info(f"Audio segment too short ({duration_ms:.0f}ms < {ConversationalAiPipelineClient.MIN_SPEECH_DURATION_MS}ms), waiting for more speech")
                                silent_frames_count = 0

                                # CRITICAL: Do NOT reset current_audio_buffer or has_spoken
                                # This allows accumulation if user pauses briefly then continues

                    else:
                        # No speech detected yet, and no prior speech in buffer
                        # Just monitor ambient noise levels for debugging

                        # Log volume periodically (every ~1 second) to help with threshold tuning
                        # int(time.time() * 10) % 10 == 0 creates a trigger every 10 time units (1 sec)
                        if int(time.time() * 10) % 10 == 0:  # Log every second
                            logger.debug(f"Waiting for speech... (RMS: {rms_volume:.4f})")

        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"{ColoredText.RED_TEXT}An unexpected error occurred: {e}{ColoredText.END_TEXT}")
        finally:
            if self.is_recording:
                # Send any remaining audio in the buffer
                if len(current_audio_buffer) > 0 and has_spoken:
                    logger.info(f"{ColoredText.BLUE_TEXT}Sending final audio chunk to server...{ColoredText.END_TEXT}")

                    self.socket_client.send_persistent_request(
                        command="transcribe",
                        message="Final audio chunk",
                        binary_data=current_audio_buffer  # Send as binary data
                    )

                self.graceful_shutdown()

    @staticmethod
    def load_json_config(filepath: str) -> dict:
        """
        Loads a JSON file and scrapes specific entries into a dictionary.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict: A dictionary containing the scraped configuration fields. host and port are required, all others are optional. An example of a JSON doc:
            {
                "host": "127.0.0.1",
                "port": 65400,

                'vad_frame_duration': 30,
                'vad_aggressiveness': 2,
                'silence_duration': 800,

                'pipeline': "basic_conversational",

                'voice': "default",

                'player_name': "Brent",
                'user_id': "Bob",
                'system_prompt_id': "default",
                'continuous_save': true,
                'load_previous': true
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
        }

        # Add optional fields with their types
        optional_fields = {
            'vad_frame_duration': int,
            'vad_aggressiveness': int,
            'silence_duration': int,
            'pipeline': str,
            'voice': str,
            'player_name': str,
            'user_id': str,
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


    """
    Gets args dictionary for a generic WhisperX streaming client  
    """
    @staticmethod
    def get_args_dict_streaming_client() -> dict:

        parser = argparse.ArgumentParser(description='Run a WhisperX client, as you see fit.')
        parser.add_argument("-ho", "--host", default=ConversationalAiPipelineClient.HOST, help="The hostname/IP that the server will bind to.")
        parser.add_argument("-p", "--port", type=int, default=ConversationalAiPipelineClient.PORT, help="The port that the server will listen on for requests.")
        parser.add_argument("-vfd", "--vad-frame-duration", type=int, default=ConversationalAiPipelineClient.VAD_FRAME_DURATION_MS, help="The VAD frame duration, in milliseconds.")
        parser.add_argument("-va", "--vad-aggressiveness", type=int, default=ConversationalAiPipelineClient.VAD_AGGRESSIVENESS, help="The VAD aggressiveness, from 1 to 3. 3 = block most non-human speech, 1 = be a bit more permissive.")
        parser.add_argument("-sd", "--silence-duration", type=int, default=ConversationalAiPipelineClient.SILENCE_DURATION_TO_END_BUFFER_MS, help="The number of milliseconds that must pass that will denote an end to speech (and the beginning of processing the speech segment).")
        parser.add_argument("-pi", "--pipeline", type=str, default=ConversationalAiPipelineClient.PIPELINE, help=f"The pipeline desired - you will send an audio clip to the server. What do you get back, and what do you do with it? The pipeline defines this. Can be: {VALID_PIPELINES}")
        parser.add_argument("-v", '--voice', type=str, default='default', help='Voice to use from the Text-To-Speech AI (revoice, basic_conversational pipelines only)')

        parser.add_argument("-pn", "--player-name", type=str, default=SubjectiveConstants.BASE_PLAYER_NAME,help="Give your username - How should the LLM address you? Leave blank if you do not want it addressing you directly via name (basic_conversational pipeline only).")
        parser.add_argument("-uid", "--user-id", type=str, default=ConversationalAiPipelineClient.USER_ID,help="A name or identification for this user. This is different from player-name - this is how the SYSTEM identifies you; think of it like an account (basic_conversational pipeline only).")
        parser.add_argument("-spf", "--system-prompt-id", type=str, default=SubjectiveConstants.SYSTEM_PROMPT_ID,help="The name or phrase that identifies the system prompt on the server that you wish to use (basic_conversational pipeline only).")
        parser.add_argument("-cs", "--continuous-save", type=bool, default=ConversationalAiPipelineClient.CONTINUOUS_SAVE,help="True if you wish the conversation to be constantly saved so you can pick up the conversation later; False otherwise (basic_conversational pipeline only).")
        parser.add_argument("-lp", "--load-previous", type=bool, default=ConversationalAiPipelineClient.LOAD_PREVIOUS,help="True if you wish to load a previous conversation (if it exists) when you start (i.e. picking up where you previously left off); False otherwise (basic_conversational pipeline only).")

        parser.add_argument("-j", "--json", type=str, default="", help="If this points to a valid JSON file, the ENTIRE parameter settings are pulled from that file, and the defaults - and other arguments passed from the command line - are ignored. If the JSON load fails for whatever reason, though, the defaults WILL be engaged.")

        argDict = {}

        try:
            args = parser.parse_args()
            use_default_arg_config = True  # This is only flipped if we successfully load from a JSON file

            json_config_file = args.json

            if json_config_file and os.path.exists(json_config_file):
                try:
                    config_dict = ConversationalAiPipelineClient.load_json_config(json_config_file)


                    argDict['host'] = config_dict.get('host', ConversationalAiPipelineClient.HOST)
                    argDict['port'] = config_dict.get('port', ConversationalAiPipelineClient.PORT)

                    argDict['vad_frame_duration'] = config_dict.get('vad_frame_duration', ConversationalAiPipelineClient.PORT)
                    argDict['vad_aggressiveness'] = config_dict.get('vad_aggressiveness', ConversationalAiPipelineClient.PORT)
                    argDict['silence_duration'] = config_dict.get('silence_duration', ConversationalAiPipelineClient.PORT)

                    argDict['pipeline'] = config_dict.get('pipeline', ConversationalAiPipelineClient.PORT)

                    argDict['voice'] = config_dict.get('voice', ConversationalAiPipelineClient.PORT)

                    argDict['player_name'] = config_dict.get('player_name', SubjectiveConstants.BASE_PLAYER_NAME)
                    argDict['system_prompt_id'] = config_dict.get('system_prompt_id', SubjectiveConstants.SYSTEM_PROMPT_ID)
                    argDict['user_id'] = config_dict.get('user_id', ConversationalAiPipelineClient.USER_ID)
                    argDict['continuous_save'] = config_dict.get('continuous_save', ConversationalAiPipelineClient.CONTINUOUS_SAVE)
                    argDict['load_previous'] = config_dict.get('load_previous', ConversationalAiPipelineClient.LOAD_PREVIOUS)

                    if argDict['pipeline'] not in VALID_PIPELINES:
                        logger.warning(f"{ColoredText.YELLOW_TEXT}Pipeline {argDict['pipeline']} not in list {VALID_PIPELINES} - setting to {VALID_PIPELINES[0]}.{ColoredText.END_TEXT}")
                        argDict['pipeline'] = VALID_PIPELINES[0]

                    logger.info(f"Config loaded from JSON {json_config_file}.")

                    use_default_arg_config = False

                except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Could not load JSON config [{json_config_file}] - there are errors. Will attempt to load other defaults or args. Error: {e}.")

            elif json_config_file:
                logger.warning(f"Could not load JSON config [{json_config_file}] - file does not exist. Loading from defaults or other parameters sent.")

            if use_default_arg_config:
                argDict['host'] = args.host
                argDict['port'] = args.port

                argDict['vad_frame_duration'] = args.vad_frame_duration
                argDict['vad_aggressiveness'] = args.vad_aggressiveness
                argDict['silence_duration'] = args.silence_duration
                argDict['pipeline'] = args.pipeline
                argDict['voice'] = args.voice

                argDict['player_name'] = args.player_name
                argDict['system_prompt_id'] = args.system_prompt_id
                argDict['user_id'] = args.user_id
                argDict['continuous_save'] = args.continuous_save
                argDict['load_previous'] = args.load_previous

                if argDict['pipeline'] not in VALID_PIPELINES:
                    logger.warning(f"{ColoredText.YELLOW_TEXT}Pipeline {argDict['pipeline']} not in list {VALID_PIPELINES} - setting to {VALID_PIPELINES[0]}.{ColoredText.END_TEXT}")
                    argDict['pipeline'] = VALID_PIPELINES[0]

                logger.debug(f"{ColoredText.BLUE_TEXT}ConversationalAiPipelineClient.get_args_dict_client: Config loaded; host: {argDict['host']} port: {argDict['port']}.{ColoredText.END_TEXT}")

        except SystemExit as e:
            argDict = {}
            if e.code == 0:
                # --help was used, so print no error
                print(f"{ColoredText.BLUE_TEXT}Thank you!{ColoredText.END_TEXT}")
            else:
                print(f"{ColoredText.RED_TEXT}ConversationalAiPipelineClient.get_args_dict_client: Invalid arguments.{ColoredText.END_TEXT}")

        return argDict



if __name__ == "__main__":
    argsDict = ConversationalAiPipelineClient.get_args_dict_streaming_client()
    client = ConversationalAiPipelineClient(argsDict)
    client.run_client()
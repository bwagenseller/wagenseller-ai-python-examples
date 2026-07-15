import sounddevice as sd
import signal
import sys
import threading
import webrtcvad
import argparse
from amadeo_utils.colored_text import ColoredText
from amadeo_utils.client.amadeo_client import AmadeoClient
from amadeo_utils.media_utils.audio_devices import prefer_pulse_defaults
import logging

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class WhisperXClient:

    SAMPLE_RATE = 16000 # in Hz. WhisperX EXCLUSIVELY uses 16k - anything else it will downsample. Might as well use it from the start.\, and hard-code it in.

    HOST = '127.0.0.1'
    PORT = 65432

    VAD_FRAME_DURATION_MS = 30
    VAD_AGGRESSIVENESS = 2
    SILENCE_DURATION_TO_END_BUFFER_MS = 800


    """
    Constructor for WhisperXClient - Updated to use AmadeoClient
    """
    def __init__(self, argsDict: dict):
        self.args_dict = argsDict
        self.VAD_FRAME_SIZE = int(WhisperXClient.SAMPLE_RATE * self.args_dict['vad_frame_duration'] / 1000)

        self.is_recording = True
        self.shutdown_lock = threading.Lock()

        # Initialize the new AmadeoClient
        self.socket_client = AmadeoClient( self.args_dict['host'], self.args_dict['port'], additional_server_response_functionality=self.handle_server_response)

    def handle_server_response(self, response, raw_data):
        """
        Callback function to handle server responses from AmadeoClient
        This replaces the old listen_for_transcriptions thread approach
        """
        if response:

            if response.get("success"):
                if response.get('type') == 'garbage_transcription':
                    logger.debug(f"{response.get('message')} detected_language: {response.get('detected_language')}  language_confidence: {response.get('language_confidence')} average_word_confidence: {response.get('average_word_confidence')} ")
                elif response.get('type') == 'transcription':
                    # Check if this is a transcription response
                    transcription = response.get("transcription")
                    detected_language = response.get("detected_language")
                    lang_conf = response.get("language_confidence", 0.0)
                    avg_word_conf = response.get("average_word_confidence", 0.0)

                    if transcription and transcription.strip():
                        logger.info(f"(language: {detected_language} ({lang_conf:.2f})) (word conf: {avg_word_conf:.2f}): {transcription}")

            else:
                # Handle different error/status types
                message = response.get("message", "Unknown error")
                if "busy" in message.lower():
                    logger.warning(f"{ColoredText.CYAN_TEXT}[Server busy, please wait for a moment.{ColoredText.END_TEXT}")
                elif "queued" in message.lower():
                    # Optionally show queued status
                    pass
                else:
                    logger.error(f"{ColoredText.RED_TEXT}Server error: {message}{ColoredText.END_TEXT}")

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
        signal.signal(signal.SIGINT, lambda s, f: self.graceful_shutdown())

        try:
            logger.info(f"{ColoredText.BLUE_TEXT}Attempting to connect to server on host: {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}{self.args_dict['host']}{ColoredText.END_TEXT}{ColoredText.BLUE_TEXT} port: {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}{self.args_dict['port']}{ColoredText.END_TEXT}")

            # Establish persistent connection
            if not self.socket_client.establish_persistent_connection():
                logger.error(f"{ColoredText.RED_TEXT}Failed to establish connection. Exiting.{ColoredText.END_TEXT}")
                return

            logger.info(f"{ColoredText.BLUE_TEXT}Starting microphone stream. Press Ctrl+C to exit.{ColoredText.END_TEXT}")

            # Route through PulseAudio where available: raw ALSA capture devices
            # reject SAMPLE_RATE (16 kHz) outright rather than resampling.
            prefer_pulse_defaults()

            # Initialize VAD and audio processing variables
            vad = webrtcvad.Vad(self.args_dict['vad_aggressiveness'])
            current_audio_buffer = bytearray()
            silent_frames_count = 0
            silent_frames_threshold = int(self.args_dict['silence_duration'] / self.args_dict['vad_frame_duration'])
            has_spoken = False

            with sd.InputStream(samplerate=WhisperXClient.SAMPLE_RATE, channels=1, dtype='int16', blocksize=self.VAD_FRAME_SIZE) as stream:
                while self.is_recording:
                    audio_frame, _ = stream.read(self.VAD_FRAME_SIZE)
                    int16_data = audio_frame.tobytes()

                    is_speech = vad.is_speech(int16_data, WhisperXClient.SAMPLE_RATE)

                    if is_speech:
                        has_spoken = True
                        silent_frames_count = 0
                        current_audio_buffer.extend(int16_data)

                    elif has_spoken:
                        silent_frames_count += 1
                        current_audio_buffer.extend(int16_data)

                        if silent_frames_count >= silent_frames_threshold:
                            speech_segment_bytes = current_audio_buffer[:-silent_frames_threshold * self.VAD_FRAME_SIZE * 2]

                            if len(speech_segment_bytes) > 0:
                                logger.debug(f"{ColoredText.BLUE_TEXT}End of speech detected. Sending audio chunk to server...{ColoredText.END_TEXT}")

                                # Send using the new persistent request method with binary audio data
                                response, raw_data = self.socket_client.send_persistent_request(
                                    command="transcribe",
                                    message="Audio chunk for transcription",
                                    binary_data=speech_segment_bytes # Send as binary data after JSON
                                )

                                # Response is handled automatically by handle_server_response callback

                            current_audio_buffer = bytearray()
                            silent_frames_count = 0
                            has_spoken = False

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

    """
    Gets args dictionary for a generic WhisperX streaming client  
    """
    @staticmethod
    def get_args_dict_streaming_client() -> dict:

        parser = argparse.ArgumentParser(description='Run a WhisperX client, as you see fit.')
        parser.add_argument("-ho", "--host", default=WhisperXClient.HOST,help="The hostname/IP that the server will bind to.")
        parser.add_argument("-p", "--port", type=int, default=WhisperXClient.PORT,help="The port that the server will listen on for requests.")
        parser.add_argument("-vfd", "--vad_frame_duration", type=int, default=WhisperXClient.VAD_FRAME_DURATION_MS,help="The VAD frame duration, in milliseconds.")
        parser.add_argument("-va", "--vad_aggressiveness", type=int, default=WhisperXClient.VAD_AGGRESSIVENESS,help="The VAD aggressiveness, from 1 to 3. 3 = block most non-human speech, 1 = be a bit more permissive.")
        parser.add_argument("-sd", "--silence_duration", type=int, default=WhisperXClient.SILENCE_DURATION_TO_END_BUFFER_MS,help="The number of milliseconds that must pass that will denote an end to speech (and the beginning of processing the speech segment).")

        argDict = {}

        try:
            args = parser.parse_args()

            argDict['host'] = args.host
            argDict['port'] = args.port

            argDict['vad_frame_duration'] = args.vad_frame_duration
            argDict['vad_aggressiveness'] = args.vad_aggressiveness
            argDict['silence_duration'] = args.silence_duration

            logger.debug(f"{ColoredText.BLUE_TEXT}WhisperXUtils.get_args_dict_client: Config loaded; host: {argDict['host']} port: {argDict['port']}.{ColoredText.END_TEXT}")

        except SystemExit as e:
            argDict = {}
            if e.code == 0:
                # --help was used, so print no error
                print(f"{ColoredText.BLUE_TEXT}Thank you!{ColoredText.END_TEXT}")
            else:
                print(f"{ColoredText.RED_TEXT}WhisperXUtils.get_args_dict_client: Invalid arguments.{ColoredText.END_TEXT}")

        return argDict

if __name__ == "__main__":
    argsDict = WhisperXClient.get_args_dict_streaming_client()
    client = WhisperXClient(argsDict)
    client.run_client()
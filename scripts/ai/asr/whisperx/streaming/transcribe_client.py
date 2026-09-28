import sounddevice as sd
import signal
import sys
import threading
import webrtcvad
import argparse
import numpy as np
from amadeo_utils.colored_text import ColoredText
from amadeo_utils.client.amadeo_client import AmadeoClient
from amadeo_utils.media_utils.audio_devices import prefer_pulse_defaults
import logging

# The two log layouts the client can use. The diagnostic one names the function and line that
# emitted the record along with the log level, which is useful when working on the client itself;
# the plain one keeps only the timestamp, so a transcription session reads as a transcript rather
# than a log.
# These are module level rather than class attributes because basicConfig below runs at import,
# before the class exists.
DIAGNOSTIC_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
PLAIN_LOG_FORMAT = '%(asctime)s - %(message)s'

# Configure logging to show timestamps and log levels. This runs before the command line has been
# parsed, so it starts out diagnostic; configure_logging() switches it once --diagnostics is known.
logging.basicConfig(level=logging.INFO, format=DIAGNOSTIC_LOG_FORMAT)
logger = logging.getLogger(__name__)

# How far back from the length limit a forced cut looks for a quiet moment (see quietest_cut)
FORCED_CUT_SEARCH_MS = 2000


def quietest_cut(buffer: bytes, frame_bytes: int, search_frames: int) -> int:
    """
    Where to cut a chunk of speech that has grown too long without a pause (a radio show, say): at the end of the
    quietest frame among the last search_frames. That is usually the small dip between two words, so the cut rarely
    lands in the middle of one.

    Args:
        buffer: the chunk so far - 16 bit mono PCM, a whole number of frames.
        frame_bytes: bytes per VAD frame.
        search_frames: how many of the most recent frames to consider.

    Returns:
        int: the byte offset to cut at - everything before it is sent, the rest is kept for the next chunk. Always at
        least one frame in, so something is sent, and at most the whole buffer.
    """
    frames = len(buffer) // frame_bytes
    if frames <= 1:
        return len(buffer)
    first = max(1, frames - search_frames)      # never frame 0: the cut must leave something to send
    samples = np.frombuffer(bytes(buffer[:frames * frame_bytes]), dtype=np.int16).astype(np.float32).reshape(frames, -1)
    energy = np.mean(samples[first:] ** 2, axis=1)
    quietest = first + int(np.argmin(energy))
    return (quietest + 1) * frame_bytes


class WhisperXClient:

    SAMPLE_RATE = 16000 # in Hz. WhisperX EXCLUSIVELY uses 16k - anything else it will downsample. Might as well use it from the start.\, and hard-code it in.

    HOST = '127.0.0.1'
    PORT = 65432

    VAD_FRAME_DURATION_MS = 30
    VAD_AGGRESSIVENESS = 2
    SILENCE_DURATION_TO_END_BUFFER_MS = 800
    # A chunk that never pauses is sent once it is this long (cut at its quietest moment); 0 = no limit, wait for a
    # pause however long it takes (right for a microphone, where people stop between sentences)
    MAX_CHUNK_SECONDS = 0.0

    # What is being listened to, for the start-up message (a subclass that listens to something else changes it)
    AUDIO_SOURCE_LABEL = "microphone"


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
                        if self.args_dict.get('diagnostics'):
                            logger.info(f"(language: {detected_language} ({lang_conf:.2f})) (word conf: {avg_word_conf:.2f}): {transcription}")
                        else:
                            logger.info(transcription)

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

    def audio_frames(self):
        """
        The audio to transcribe: yields one VAD frame at a time (VAD_FRAME_SIZE samples of 16 kHz mono int16, as
        bytes) while the client is recording. This one reads the default microphone; a subclass can listen to
        something else by overriding it (transcribe_output.py listens to the speakers).

        :return: a generator of bytes, one VAD frame each.
        """
        # Route through PulseAudio where available: raw ALSA capture devices
        # reject SAMPLE_RATE (16 kHz) outright rather than resampling.
        prefer_pulse_defaults()

        with sd.InputStream(samplerate=WhisperXClient.SAMPLE_RATE, channels=1, dtype='int16', blocksize=self.VAD_FRAME_SIZE) as stream:
            while self.is_recording:
                audio_frame, _ = stream.read(self.VAD_FRAME_SIZE)
                yield audio_frame.tobytes()

    def send_audio(self, segment: bytes, message: str):
        """
        Sends one chunk of speech to the server; the transcription comes back through handle_server_response.

        :param segment: 16 bit mono 16 kHz PCM.
        :param message: a note for the server's log.
        """
        self.socket_client.send_persistent_request(command="transcribe", message=message, binary_data=segment)

    def run_client(self):
        signal.signal(signal.SIGINT, lambda s, f: self.graceful_shutdown())

        # Set before anything can fail, so the 'finally' below can always look at them (a failed connection used to
        # end in a NameError there instead of a clean exit)
        current_audio_buffer = bytearray()
        has_spoken = False

        try:
            logger.info(f"{ColoredText.BLUE_TEXT}Attempting to connect to server on host: {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}{self.args_dict['host']}{ColoredText.END_TEXT}{ColoredText.BLUE_TEXT} port: {ColoredText.END_TEXT}{ColoredText.YELLOW_TEXT}{self.args_dict['port']}{ColoredText.END_TEXT}")

            # Establish persistent connection
            if not self.socket_client.establish_persistent_connection():
                logger.error(f"{ColoredText.RED_TEXT}Failed to establish connection. Exiting.{ColoredText.END_TEXT}")
                return

            logger.info(f"{ColoredText.BLUE_TEXT}Starting {self.AUDIO_SOURCE_LABEL} stream. Press Ctrl+C to exit.{ColoredText.END_TEXT}")

            # Initialize VAD and audio processing variables
            vad = webrtcvad.Vad(self.args_dict['vad_aggressiveness'])
            current_audio_buffer = bytearray()
            silent_frames_count = 0
            silent_frames_threshold = int(self.args_dict['silence_duration'] / self.args_dict['vad_frame_duration'])
            has_spoken = False

            # The length limit for audio that never pauses (0 = none), and how far back a forced cut looks for a gap
            frame_bytes = self.VAD_FRAME_SIZE * 2
            max_chunk_seconds = self.args_dict.get('max_chunk_seconds') or 0
            max_chunk_bytes = int(max_chunk_seconds * WhisperXClient.SAMPLE_RATE) * 2 if max_chunk_seconds > 0 else 0
            search_frames = max(1, int(FORCED_CUT_SEARCH_MS / self.args_dict['vad_frame_duration']))

            for int16_data in self.audio_frames():
                if not self.is_recording:
                    break

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
                            # The response is handled by the handle_server_response callback
                            self.send_audio(bytes(speech_segment_bytes), "Audio chunk for transcription")

                        current_audio_buffer = bytearray()
                        silent_frames_count = 0
                        has_spoken = False

                # Audio that never pauses long enough (a radio show) would otherwise grow into one endless chunk:
                # once it reaches the limit, send it up to its quietest recent moment and keep the rest going
                if max_chunk_bytes and has_spoken and len(current_audio_buffer) >= max_chunk_bytes:
                    cut = quietest_cut(current_audio_buffer, frame_bytes, search_frames)
                    logger.debug(f"{ColoredText.BLUE_TEXT}Chunk reached {max_chunk_seconds:g} s without a pause; sending {cut / 2 / WhisperXClient.SAMPLE_RATE:.1f} s of it.{ColoredText.END_TEXT}")
                    self.send_audio(bytes(current_audio_buffer[:cut]), "Audio chunk for transcription (length limit)")
                    current_audio_buffer = current_audio_buffer[cut:]
                    silent_frames_count = 0

        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"{ColoredText.RED_TEXT}An unexpected error occurred: {e}{ColoredText.END_TEXT}")
        finally:
            if self.is_recording:
                # Send any remaining audio in the buffer
                if len(current_audio_buffer) > 0 and has_spoken:
                    logger.info(f"{ColoredText.BLUE_TEXT}Sending final audio chunk to server...{ColoredText.END_TEXT}")

                    self.send_audio(bytes(current_audio_buffer), "Final audio chunk")

                self.graceful_shutdown()

    """
    Applies the chosen log layout to every handler the root logger already has
    """
    @staticmethod
    def configure_logging(diagnostics: bool):
        """
        Switch the log layout to match the --diagnostics setting.

        basicConfig has already installed a handler by the time the command line is parsed, and
        calling it a second time is a no-op once handlers exist, so the formatter is replaced on
        the existing handlers instead.

        Args:
            diagnostics: True to keep the log level, function name and line number in every record,
                         False for the plain timestamp-only layout.

        Returns:
            None
        """
        formatter = logging.Formatter(DIAGNOSTIC_LOG_FORMAT if diagnostics else PLAIN_LOG_FORMAT)

        for handler in logging.getLogger().handlers:
            handler.setFormatter(formatter)

        # Say so when the extra detail is being withheld, so that its absence looks like a setting
        # rather than something missing.
        if not diagnostics:
            logger.info(f"{ColoredText.BLUE_TEXT}Diagnostics off - run with --diagnostics to see the detected language and confidence scores.{ColoredText.END_TEXT}")

    @staticmethod
    def build_arg_parser(description: str = 'Run a WhisperX client, as you see fit.') -> argparse.ArgumentParser:
        """
        The command-line options every streaming WhisperX client shares (server, VAD, diagnostics). A client that
        needs more (transcribe_output.py's --output, say) adds its own to the parser this returns.

        Args:
            description: the --help description.

        Returns:
            argparse.ArgumentParser: the parser, with the shared options added.
        """
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("-ho", "--host", default=WhisperXClient.HOST,help="The hostname/IP that the server will bind to.")
        parser.add_argument("-p", "--port", type=int, default=WhisperXClient.PORT,help="The port that the server will listen on for requests.")
        parser.add_argument("-vfd", "--vad_frame_duration", type=int, default=WhisperXClient.VAD_FRAME_DURATION_MS,help="The VAD frame duration, in milliseconds.")
        parser.add_argument("-va", "--vad_aggressiveness", type=int, default=WhisperXClient.VAD_AGGRESSIVENESS,help="The VAD aggressiveness, from 1 to 3. 3 = block most non-human speech, 1 = be a bit more permissive.")
        parser.add_argument("-sd", "--silence_duration", type=int, default=WhisperXClient.SILENCE_DURATION_TO_END_BUFFER_MS,help="The number of milliseconds that must pass that will denote an end to speech (and the beginning of processing the speech segment).")
        parser.add_argument("-mcs", "--max_chunk_seconds", type=float, default=WhisperXClient.MAX_CHUNK_SECONDS, help="Send a chunk once it is this many seconds long even without a pause, cut at its quietest recent moment - for audio that never stops, like a radio show. 0 = no limit: wait for a pause (default here: %(default)s).")
        parser.add_argument("-d", "--diagnostics", action="store_true",help="Show diagnostic detail alongside each transcription: the function and line that logged it, plus the detected language and its confidence and the average word confidence. Off by default, which prints just the timestamp and the transcription itself.")
        return parser

    """
    Gets args dictionary for a generic WhisperX streaming client
    """
    @staticmethod
    def get_args_dict_streaming_client() -> dict:

        parser = WhisperXClient.build_arg_parser()

        argDict = {}

        try:
            args = parser.parse_args()

            argDict['host'] = args.host
            argDict['port'] = args.port

            argDict['vad_frame_duration'] = args.vad_frame_duration
            argDict['vad_aggressiveness'] = args.vad_aggressiveness
            argDict['silence_duration'] = args.silence_duration
            argDict['diagnostics'] = args.diagnostics
            argDict['max_chunk_seconds'] = args.max_chunk_seconds

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
    WhisperXClient.configure_logging(argsDict.get('diagnostics', False))
    client = WhisperXClient(argsDict)
    client.run_client()
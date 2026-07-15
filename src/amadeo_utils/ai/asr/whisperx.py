
import threading
import json
import gc
import torch
import whisperx
import numpy as np
import argparse
import logging
from typing import Dict, Any, Tuple
from whisperx.audio import N_SAMPLES, log_mel_spectrogram
from amadeo_utils.colored_text import ColoredText
from amadeo_utils.server.amadeo_server import AmadeoServer

"""
This is a basic implementation of WhisperX, an ASR (speech to text) library. Its a basic implementation. It was primarily built for responding from a server (handle_client_request acts as a callback function for a larger server script), but you can use it independently, too, 
with 'get_transcription'. 
"""

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)


def detect_language_with_probability(asr_model, audio: np.ndarray) -> Tuple[str, float]:
    """
    Detect the spoken language of an audio segment AND return the confidence score.

    whisperx's own FasterWhisperPipeline.detect_language() computes the language
    probability internally and then throws it away, returning only the language
    code. We want the confidence too, so this repeats the same steps and returns
    both.

    This previously existed as a method hand-added to whisperx/asr.py inside
    site-packages. That patch was invisible to this repo and did not survive
    rebuilding the conda env, so it is implemented here instead.

    :param asr_model: A loaded whisperx pipeline, i.e. the result of
                      whisperx.load_model().
    :param audio: Mono float32 audio at 16 kHz. Only the first 30 seconds
                  (N_SAMPLES) are used, as Whisper's language detection is
                  trained on 30s windows; shorter audio is zero-padded.
    :return: Tuple of (language_code, probability) - e.g. ('en', 0.98).
    """
    # asr_model.model is the faster-whisper WhisperModel wrapper; its own .model
    # attribute is the underlying CTranslate2 model, which is what actually
    # exposes detect_language().
    whisper_model = asr_model.model

    # Larger Whisper variants use 128 mel bins rather than the classic 80, so ask
    # the model rather than assuming.
    n_mels = whisper_model.feat_kwargs.get("feature_size")

    segment = log_mel_spectrogram(
        audio[:N_SAMPLES],
        n_mels=n_mels if n_mels is not None else 80,
        padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0],
    )

    encoder_output = whisper_model.encode(segment)
    results = whisper_model.model.detect_language(encoder_output)

    # results[0][0] is a (language_token, probability) pair for the most likely
    # language. The token is delimited like '<|en|>', so strip the two leading and
    # two trailing characters to recover the bare language code.
    language_token, language_probability = results[0][0]

    return language_token[2:-2], language_probability


class AmadeoWhisperX:

    MAX_POSITIVE_INT16_VALUE_AS_FLOAT = 32767.0

    HOST = '127.0.0.1'
    PORT = 65432
    WHISPER_MODEL_NAME = "large-v3"
    LANGUAGE_CODE = "en"
    COMBINED_CONFIDENCE_CUTOFF = 1.0

    def __init__(self, argsDict: dict):
        self.args_dict = argsDict
        # --- Configuration and Global Resources ---
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        self.server = AmadeoServer(argsDict['host'], argsDict['port'], additional_client_functionality = self.handle_client_request)

        # set this lock, which will 'lock' the GPU for its own purposes (really, it locks the methods that WhisperX uses to interact with the GPU)
        self.gpu_lock = threading.Lock()

        # WhisperX's VAD is pyannote, which disables TensorFloat-32 for
        # reproducibility on every CUDA inference and warns loudly each time it
        # finds TF32 still enabled (torch defaults cudnn.allow_tf32 to True).
        # Setting the same state up front leaves behaviour identical - pyannote
        # would force it anyway - while keeping the logs clean. Do not re-enable
        # TF32 to chase speed: pyannote re-disables it per inference, and the
        # actual Whisper transcription runs through CTranslate2, which ignores
        # these torch flags entirely.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # --- WhisperX Setup ---
        logger.info(f"{ColoredText.BLUE_TEXT}WhisperXServer: Loading WhisperX model '{self.args_dict['model']}' on {self.device} with {self.compute_type}...{ColoredText.END_TEXT}")
        self.asr_model = whisperx.load_model(self.args_dict['model'], device=self.device, compute_type=self.compute_type)
        logger.info(f"{ColoredText.BLUE_TEXT}WhisperXServer: ASR model loaded.{ColoredText.END_TEXT}")
        logger.info(f"{ColoredText.BLUE_TEXT}WhisperXServer: Loading WhisperX alignment model for language model '{self.args_dict['language_code']}' on {self.device}.. {ColoredText.END_TEXT}")

        self.align_model, self.metadata = whisperx.load_align_model(language_code=self.args_dict['language_code'], device=self.device)
        logger.info(f"{ColoredText.BLUE_TEXT}WhisperXServer: Alignment model loaded.{ColoredText.END_TEXT}")

    def get_transcription(self, data:bytes = None):
        """
        This simply acts as a wrapper for handle_client_request - this enables a more 'natural' method to use if you are not using the server and just using the model
        Args:
            request: A dictionary that will contain fields. It should ALWAYS contain 'command', which represents WHAT the user wants to do. That will determine one of several scenarios:

        Returns:
             The dictionary containing the transcription
        """
        sent_request = {
            'command': 'transcribe'
        }

        response, received_data = self.handle_client_request(sent_request, data)
        return response


    def handle_client_request(self, request: Dict[str, Any], data:bytes = None):
        """
        This method is designed specifically to handle a request from a server - this class can stay running alongside a server class, but the server class will call this method when it gets a request (the server class will handle stuff like sockets etc etc, but this will handle the SPECIFIC
        tasks related to WhisperX). This method (and other methods in other classes that implement this) expects a dictionary and data (bytes, which can represent all kinds of media files), although the data portion of that may not be used (depending on the case; in WhisperX's case, this is not used).
        This should return a dictionary (that will be turned into JSON) and byte data (if applicable, and in our case it is - its the PCM audio data in a WAV container).

        To see what is expected of the basics of what is expected fpr the server, see the main description for 'amadeo_server.AmadeoServer', although there are some additional ones specific to WhisperX:
        * 'command' - for now this is just 'transcribe' but there could be more later. This MUST be present if you want transcriptions!

        To see the base dictionary fields will be sent to the client. see the main description for 'amadeo_server.AmadeoServer'; here are ADDITIONAL fields that are sent:
        * transcription - the transcription
        * detected_language - The detected language
        * language_confidence - The confidence score for the language (0 - 1)
        * average_word_confidence - The average word confidence score (0 - 1)


        Args:
            request: A dictionary that will contain fields. It should ALWAYS contain 'command', which represents WHAT the user wants to do. That will determine one of several scenarios:
                    Scenario 1: generating transcriptions
                        'command' = 'transcribe'
            data: bytes - This will always be the audio from the client's microphone. Currently, its expected to be data that the Python library 'sounddevice' captures from a mic - a 16 bit signed integer, with values from -32,768 to 32,767 (in other words, raw PCM bytes)

        Returns:
            Tuple[dict, None] - The dictionary (that will be converted to JSON and sent to the client), None (Since this has to fit the format of what we may send to a client, that is (JSON, media_data) - and since this returns no media, its always None)
        """
        try:
            possible_commands = ('transcribe')
            command = request.get('command', '')
            address = request.get('client_address', 'NO_ADDRESS')
            port = request.get('client_port', 'NO_PORT')
            sessionID = request.get('sessionID', 'NO_SESSION_ID')
            if not command or command not in possible_commands:
                command = 'transcribe'
                logger.warning(f"Request came in with no command from {address}:{port} - setting command to {command}.")

            # Extract required fields from the request
            if command == 'transcribe':
            #    some_field = request.get('some_field', '').strip()

                ## Validate that text is provided and not empty
                #if not some_field:
                #    raise ValueError("No some_field provided in request")

                # Log the request for debugging/monitoring
                logger.info(f"Request from {address}:{port} - Transcription to be serviced.")

        except (json.JSONDecodeError, KeyError) as e:
            # Invalid JSON format or missing required fields
            raise ValueError(f"Invalid JSON request: {e}")

        response = {}
        if command == 'transcribe':

            logger.debug(f"{ColoredText.YELLOW_TEXT}[{sessionID}]{ColoredText.END_TEXT}{ColoredText.CYAN_TEXT} Processing job{ColoredText.END_TEXT}")

            with self.gpu_lock:
                try:

                    # Since the audio is in the format of what the Python library 'sounddevice' produces from a mic (see description above), we need to convert to what WhisperX is expecting,
                    # which is a 32 bit floating point audio (i.e. np.frombuffer(data, dtype=np.int16).astype(np.float32)) with values normalized between -1 and 1 (i.e. MAX_POSITIVE_INT16_VALUE_AS_FLOAT, which normalizes
                    # from int16 range (-32,768 to +32,767) to float32 range (-1.0 to +1.0))
                    # NOTE: WhisperX _EXCLUSIVELY_ uses a 16k Sample rate! If its higher, WhisperX will downsample, but....why waste the bandwidth if its just going to resample
                    audio_segment_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / AmadeoWhisperX.MAX_POSITIVE_INT16_VALUE_AS_FLOAT

                    audio_for_lang_detect = whisperx.audio.pad_or_trim(audio_segment_np)

                    detected_language, language_confidence = detect_language_with_probability(self.asr_model, audio_for_lang_detect)

                    result = self.asr_model.transcribe(audio_segment_np, batch_size=1, language=detected_language)

                    full_text = ""
                    average_word_confidence = 0.0

                    if result and "segments" in result and result["segments"]:
                        aligned_result = whisperx.align(result["segments"], self.align_model, self.metadata, audio_segment_np, device=self.device)

                        word_confidences = []
                        for segment in aligned_result["segments"]:
                            if "words" in segment:
                                for word_info in segment["words"]:
                                    if "word" in word_info and "score" in word_info:
                                        word_confidences.append(word_info["score"])
                                        full_text += f"{word_info['word']} "

                        if word_confidences:
                            average_word_confidence = sum(word_confidences) / len(word_confidences)

                        # The `full_text` from the loop above will have a trailing space.
                        full_text = full_text.strip()

                    if full_text == "":
                        msg = f"Blank transcription generated."
                        response = {
                            'success': True,
                            'type': 'garbage_transcription',
                            "message": msg,
                            'file_size': 0
                        }
                        logger.debug(msg)
                    elif (self.args_dict['combined_confidence_cutoff'] > (language_confidence + average_word_confidence)):
                        msg = f"Transcription blocked due to low confidence score."
                        response = {
                            'success': True,
                            'type': 'garbage_transcription',
                            "message": msg,
                            "transcription": full_text,
                            "detected_language": detected_language,
                            "language_confidence": language_confidence,
                            "average_word_confidence": average_word_confidence,
                            'file_size': 0
                        }
                        logger.debug(msg)
                    else:
                        response = {
                            'success': True,
                            'type': 'transcription',
                            "transcription": full_text,
                            "message": '',
                            "detected_language": detected_language,
                            "language_confidence": language_confidence,
                            "average_word_confidence": average_word_confidence,
                            'file_size': 0
                        }

                        logger.info(f"Transcription completed and sent. Lang: {detected_language} ({language_confidence:.2f}), Avg Conf: {average_word_confidence:.2f}")

                except Exception as e:
                    logger.warning(f"Error during transcription: {e}")
                    response = {
                        'success': False,
                        "message": str(e),
                        'file_size': 0
                    }

                finally:
                    gc.collect()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()


        else:
            msg = f"Unknown command sent."
            logger.info(msg)
            response = {
                'success': True,
                'type': 'error',
                "message": msg,
                'file_size': 0
            }

        return response, None


    ################################################################################################################### Parsing Arguments From Command Line ####################################################################################################################

    """
    Gets args dictionary for a generic WhisperX server  
    """
    @staticmethod
    def get_args_dict_server() -> dict:

        parser = argparse.ArgumentParser(description='Run a WhisperX server, as you see fit.')
        parser.add_argument("-ho", "--host", default=AmadeoWhisperX.HOST,help="The hostname/IP that the server will bind to.")
        parser.add_argument("-p", "--port", type=int, default=AmadeoWhisperX.PORT,help="The port that the server will listen on for requests.")
        parser.add_argument("-m", "--model", default=AmadeoWhisperX.WHISPER_MODEL_NAME,help="The Whisper model to use.")
        parser.add_argument("-l", "--language_code", default=AmadeoWhisperX.LANGUAGE_CODE,help="The default language code.")
        parser.add_argument("-ccc", "--combined_confidence_cutoff", type=float, default=AmadeoWhisperX.COMBINED_CONFIDENCE_CUTOFF,help="Each transcription has a confidence score (0-1) for the language and then an average confidence score for the words; if both of these numbers, summed, are less than this, the transcription will not be sent back to the client (as it is probably a false reading).")

        argDict = {}

        try:
            args = parser.parse_args()

            argDict['host'] = args.host
            argDict['port'] = args.port
            argDict['model'] = args.model
            argDict['language_code'] = args.language_code
            argDict['combined_confidence_cutoff'] = args.combined_confidence_cutoff

            logger.debug(f"{ColoredText.BLUE_TEXT}WhisperXUtils.get_args_dict_server: Config loaded; host: {argDict['host']} port: {argDict['port']} model: {argDict['model']}.{ColoredText.END_TEXT}")

        except SystemExit as e:
            argDict = {}
            if e.code == 0:
                # --help was used, so print no error
                print(f"{ColoredText.BLUE_TEXT}Thank you!{ColoredText.END_TEXT}")
            else:
                print(f"{ColoredText.RED_TEXT}WhisperXUtils.get_args_dict_server: Invalid arguments.{ColoredText.END_TEXT}")

        return argDict
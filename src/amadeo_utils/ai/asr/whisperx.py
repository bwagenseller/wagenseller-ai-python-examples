
import os

# CUDA numbers GPUs by 'FASTEST_FIRST' unless told otherwise, which ranks them by compute
# capability rather than by the slot they sit in - so CUDA's device 0 is not necessarily the
# device 0 that 'nvidia-smi' reports. On a mixed pair such as an RTX 3090 (sm_86) alongside an
# RTX 4070 Ti (sm_89), CUDA puts the newer-architecture 4070 Ti first even though nvidia-smi
# lists the 3090 first, so '--gpu 0' would silently pick the opposite card to the one the user
# was looking at. 'PCI_BUS_ID' orders by physical slot, which is what nvidia-smi shows and what
# anyone reading '--gpu 0' will expect.
#
# This has to be set before CUDA is initialised, hence before torch is imported rather than in
# the constructor. setdefault is used so that an explicit setting in the environment still wins.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import threading
import time
import wave
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
from amadeo_utils.ai.asr import speaker_id

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


def speech_span(aligned_result, total_seconds: float, pad_seconds: float = 0.15) -> Tuple[float, float, float]:
    """
    Where the speech is in a chunk, from WhisperX's aligned segments.

    A chunk from the conversational client ends with the silence that told it the speaker had stopped (most of a
    second), so its length says little about how much was said: a quick "yes" arrives as a one-second chunk. Speaker
    identification needs the SPEECH - both to judge whether there is enough of it, and to embed the voice rather
    than the room.

    :param aligned_result: whisperx.align()'s result (segments with 'start' / 'end' in seconds), or None.
    :param total_seconds: The chunk's length.
    :param pad_seconds: Kept either side of the speech, so the first and last sounds are not clipped.
    :return: (start, end, speech_seconds): the padded span to embed, and the seconds actually spoken (the segments'
             total). With no usable timings: (0, total_seconds, total_seconds).
    """
    spans = []
    for segment in (aligned_result or {}).get('segments', []):
        start, end = segment.get('start'), segment.get('end')
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
            spans.append((float(start), float(end)))
    if not spans:
        return 0.0, total_seconds, total_seconds
    start = max(0.0, min(s for s, _ in spans) - pad_seconds)
    end = min(total_seconds, max(e for _, e in spans) + pad_seconds)
    return start, end, sum(e - s for s, e in spans)


class AmadeoWhisperX:

    MAX_POSITIVE_INT16_VALUE_AS_FLOAT = 32767.0

    HOST = '127.0.0.1'
    PORT = 65432
    WHISPER_MODEL_NAME = "large-v3"
    LANGUAGE_CODE = "en"
    COMBINED_CONFIDENCE_CUTOFF = 1.0
    GPU_INDEX = 0

    # The keys of the server's --json config, with their types (see load_json_config)
    SERVER_CONFIG_FIELDS = {
        'host': str,
        'port': int,
        'model': str,
        'language_code': str,
        'gpu': int,
        'combined_confidence_cutoff': (int, float),
        'speaker_id': dict,
        'log_file': str,        # also log to this file (see amadeo_utils.logging_utils); missing = screen only
    }

    def __init__(self, argsDict: dict):
        self.args_dict = argsDict
        # --- Configuration and Global Resources ---

        # Which physical GPU to run on. Absent from the dictionary (i.e. when this class is used
        # outside of the server script) it falls back to the first GPU, which is the only index
        # guaranteed to exist on a CUDA machine.
        self.gpu_index = argsDict.get('gpu', AmadeoWhisperX.GPU_INDEX)

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()

            # Fail at startup rather than quietly transcribing on the wrong card - a silent
            # fallback would look identical to success while ignoring what was asked for.
            if self.gpu_index < 0 or self.gpu_index >= device_count:
                raise ValueError(f"Requested GPU index {self.gpu_index} does not exist; this machine has {device_count} CUDA device(s), so valid indexes are 0 through {device_count - 1}.")

            # This class runs behind a threaded server, and torch tracks the 'current' device per
            # thread rather than per process. Pinning it here only fixes the thread doing the
            # loading; every worker thread has to pin it again before it touches the GPU (see
            # _bind_thread_to_gpu), otherwise that thread would silently default back to GPU 0.
            torch.cuda.set_device(self.gpu_index)

            # Two different device spellings are needed downstream, because the two model stacks
            # do not agree on how a GPU is named:
            #  * torch_device ('cuda:N') is for the alignment model and whisperx.align, which are
            #    plain PyTorch and understand an indexed device string.
            #  * ct2_device ('cuda') plus a separate device_index is for whisperx.load_model,
            #    which sits on faster-whisper/CTranslate2. CTranslate2 accepts only the bare
            #    'cuda' here and rejects 'cuda:N', so the index travels as its own argument.
            self.device = f"cuda:{self.gpu_index}"
            self.ct2_device = "cuda"
            self.compute_type = "float16"

            logger.info(f"{ColoredText.BLUE_TEXT}WhisperXServer: Using GPU {self.gpu_index} ({torch.cuda.get_device_name(self.gpu_index)}) of {device_count} available.{ColoredText.END_TEXT}")
        else:
            # No CUDA at all - the GPU index is meaningless, so note that it is being ignored
            # instead of letting the caller assume it took effect.
            if self.gpu_index != AmadeoWhisperX.GPU_INDEX:
                logger.warning(f"A GPU index ({self.gpu_index}) was requested but no CUDA device is available; falling back to CPU and ignoring the index.")

            # Reset the index so the rest of the class does not have to keep asking whether it is
            # meaningful; on CPU it is simply unused.
            self.gpu_index = AmadeoWhisperX.GPU_INDEX
            self.device = "cpu"
            self.ct2_device = "cpu"
            self.compute_type = "int8"

        # Convenience flag - self.device is now 'cuda:N' rather than a bare 'cuda', so string
        # comparisons against "cuda" elsewhere would no longer hold.
        self.use_cuda = (self.ct2_device == "cuda")

        # NOTE: this class deliberately does NOT build its own AmadeoServer. The runnable script
        # that owns the host/port (scripts/ai/asr/whisperx/streaming/transcribe_server.py) builds
        # the server and hands it this class's handle_client_request as a callback, which is the
        # same arrangement the Kokoro, F5 and llama services use.

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
        # device/device_index are split here for the CTranslate2 reason described in __init__.
        self.asr_model = whisperx.load_model(self.args_dict['model'], device=self.ct2_device, device_index=self.gpu_index, compute_type=self.compute_type)
        logger.info(f"{ColoredText.BLUE_TEXT}WhisperXServer: ASR model loaded.{ColoredText.END_TEXT}")
        logger.info(f"{ColoredText.BLUE_TEXT}WhisperXServer: Loading WhisperX alignment model for language model '{self.args_dict['language_code']}' on {self.device}.. {ColoredText.END_TEXT}")

        self.align_model, self.metadata = whisperx.load_align_model(language_code=self.args_dict['language_code'], device=self.device)
        logger.info(f"{ColoredText.BLUE_TEXT}WhisperXServer: Alignment model loaded.{ColoredText.END_TEXT}")

        # --- Speaker identification (optional; see amadeo_utils.ai.asr.speaker_id) ---
        # Only set up when the server's config has a 'speaker_id' block. Without one, a request asking for voice
        # recognition gets speaker_status 'disabled' and the conversational server treats the voice as unknown.
        self.speaker_settings = None
        self.speaker_embedder = None
        self.speaker_index = speaker_id.SpeakerIndex([])
        self.speaker_signature = None           # the profiles directory's fingerprint when the index was built
        self.speaker_index_lock = threading.Lock()
        self.field_expiry_lock = threading.Lock()
        self.field_expired_at = 0.0             # when old field clips were last deleted (time.time())
        speaker_config = self.args_dict.get('speaker_id')
        if speaker_config:
            # A bad config stops the server at startup, rather than silently running without recognition
            self.speaker_settings = speaker_id.settings_from_dict(speaker_config, other_level_keys=AmadeoWhisperX.SERVER_CONFIG_FIELDS)
            from amadeo_utils.ai.asr.speaker_embedder import SpeakerEmbedder
            logger.info(f"{ColoredText.BLUE_TEXT}WhisperXServer: Loading speaker-embedding model '{self.speaker_settings.embedding_model}' on {self.device}...{ColoredText.END_TEXT}")
            self.speaker_embedder = SpeakerEmbedder(self.speaker_settings.embedding_model, self.device)
            index = self._current_speaker_index()
            logger.info(f"{ColoredText.BLUE_TEXT}WhisperXServer: Speaker identification on: {len(index)} person(s) enrolled in {self.speaker_settings.profiles_dir}.{ColoredText.END_TEXT}")
            settings = self.speaker_settings
            if settings.save_known_field_clips or settings.save_unknown_field_clips:
                kinds = ' and '.join(k for k, on in (('known', settings.save_known_field_clips), ('unrecognized', settings.save_unknown_field_clips)) if on)
                kept = f"kept {settings.field_retention_days:g} days" if settings.field_retention_days > 0 else "kept for ever"
                logger.info(f"{ColoredText.BLUE_TEXT}WhisperXServer: Saving {kinds} voices' speech to {settings.field_samples_dir} ({kept}), from clients that opt in.{ColoredText.END_TEXT}")
                self._expire_field_clips()

    def _current_speaker_index(self) -> speaker_id.SpeakerIndex:
        """
        The enrolled voices, reloaded whenever the profiles directory has changed (someone was enrolled or removed),
        so an enrollment takes effect without restarting the server. Checking costs one directory listing.

        :return: The SpeakerIndex to match against.
        """
        with self.speaker_index_lock:
            signature = speaker_id.profiles_signature(self.speaker_settings.profiles_dir)
            if signature != self.speaker_signature:
                profiles = speaker_id.load_profiles(self.speaker_settings.profiles_dir, self.speaker_settings.embedding_model)
                self.speaker_index = speaker_id.SpeakerIndex(profiles)
                self.speaker_signature = signature
                logger.info(f"{ColoredText.BLUE_TEXT}WhisperXServer: Voice profiles (re)loaded: {sorted(self.speaker_index.people)}.{ColoredText.END_TEXT}")
            return self.speaker_index

    def _expire_field_clips(self):
        """
        Deletes field clips older than the retention period. Runs at startup and then at most once an hour (from
        whichever request saves a clip), so a server left running for months keeps the directory trimmed without a
        separate job. Never raises: a clip that cannot be deleted is only logged.
        """
        now = time.time()
        with self.field_expiry_lock:
            if now - self.field_expired_at < 3600:
                return
            self.field_expired_at = now
        removed = 0
        for path in speaker_id.expired_field_clips(self.speaker_settings.field_samples_dir, self.speaker_settings.field_retention_days, now):
            try:
                os.remove(path)
                removed += 1
            except OSError as e:
                logger.warning(f"Could not delete expired field clip {path}: {e}")
        if removed:
            logger.info(f"Deleted {removed} field clip(s) older than {self.speaker_settings.field_retention_days:g} days.")

    def _save_field_clip(self, match: speaker_id.SpeakerMatch, location_id: str, speech_audio, client_known: bool = False,
                         client_unknown: bool = False) -> None:
        """
        Saves the judged speech as a field clip when both the config and the client allow this kind (see
        speaker_id.field_clip_kind),
        under <field_samples_dir>/<location>/<person or 'unrecognized'>/, as 16 bit mono 16 kHz WAV - the audio
        exactly as the live microphone delivered it, which is what enrollment wants. Clips are only ever reviewed
        and moved into the samples tree by hand, never enrolled automatically: a misidentified clip fed back into a
        profile would pull it toward the wrong person. Never raises: saving must not cost the transcription.

        :param match: The speaker decision.
        :param location_id: The request's location_id.
        :param speech_audio: The speech that was judged (float32, -1..1, 16 kHz), or None if it was not embedded.
        :param client_known: The request's save_known_field_clips (the client opting its microphone in).
        :param client_unknown: The request's save_unknown_field_clips.
        """
        kind = speaker_id.field_clip_kind(self.speaker_settings, match.status, client_known, client_unknown)
        if not kind or speech_audio is None or not len(speech_audio):
            return
        speaker_folder = (self.speaker_index.file_names.get(match.speaker, match.speaker) if kind == 'known'
                          else speaker_id.FIELD_UNRECOGNIZED_FOLDER)
        path = speaker_id.field_clip_path(self.speaker_settings.field_samples_dir, location_id, speaker_folder, time.time())
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pcm = (np.clip(speech_audio, -1.0, 1.0) * AmadeoWhisperX.MAX_POSITIVE_INT16_VALUE_AS_FLOAT).astype(np.int16)
            with wave.open(path, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm.tobytes())
            os.chmod(path, 0o660)      # biometric data: owner and group only
            logger.debug(f"Field clip saved: {path}")
        except Exception as e:
            logger.warning(f"Could not save field clip {path}: {e}")
        self._expire_field_clips()

    def _speaker_fields(self, voice_recognition: bool, location_id: str, seconds: float, embedding, embed_error: str,
                        speech_audio=None, client_field_clips: Tuple[bool, bool] = (False, False)) -> Dict[str, Any]:
        """
        Works out who spoke and returns the fields added to a transcription response.

        :param voice_recognition: Whether the request asked for recognition. If not, no fields are added.
        :param location_id: The client's location_id ('' for none); its enrolled samples are compared first.
        :param seconds: How long the speech is (see speech_span).
        :param embedding: The speech's embedding, or None if it was not computed.
        :param embed_error: Why the embedding failed ('' if it did not).
        :param speech_audio: The speech that was embedded, saved as a field clip if the config asks (see
                             _save_field_clip); None if there was none.
        :param client_field_clips: The request's (save_known_field_clips, save_unknown_field_clips).
        :return: {} or {'speaker', 'speaker_score', 'speaker_status', 'speaker_step'} - speaker is a display name,
                 speaker_id.UNRECOGNIZED_SPEAKER, or '' when the status leaves it to the caller (too_short, disabled,
                 error).
        """
        if not voice_recognition:
            return {}
        if self.speaker_embedder is None:
            match = speaker_id.SpeakerMatch(speaker_id.STATUS_DISABLED)
        elif embed_error:
            match = speaker_id.SpeakerMatch(speaker_id.STATUS_ERROR)
        else:
            match = self._current_speaker_index().identify(embedding, self.speaker_settings, location_id, seconds)
        scores = ', '.join(f"{name} {score:.2f}" for name, score in sorted(match.scores.items(), key=lambda kv: -kv[1]))
        logger.info(f"Speaker: {match.status} '{match.speaker}' (step '{match.step}', location '{location_id}', {seconds:.1f} s; {scores or 'no scores'}){' - ' + embed_error if embed_error else ''}")
        if self.speaker_settings is not None:
            self._save_field_clip(match, location_id, speech_audio, *client_field_clips)
        return {
            'speaker': match.speaker,
            'speaker_score': round(match.score, 4),
            'speaker_status': match.status,
            'speaker_step': match.step
        }

    def _bind_thread_to_gpu(self):
        """
        Pin the calling thread to the GPU this instance was configured for.

        torch stores the 'current' CUDA device per thread, and it always starts out as device 0.
        AmadeoServer hands every client connection to a fresh thread, so a thread that has not
        been pinned would send any torch work that was given a bare 'cuda' device - most notably
        the VAD stage inside the ASR pipeline - to GPU 0 regardless of what was asked for. On a
        single GPU machine that is invisible; on a multi GPU machine it silently splits the work
        across cards. Calling this at the top of each GPU bound request keeps every thread on the
        same card.

        This is a no-op when running on CPU.

        :return: None
        """
        if self.use_cuda:
            torch.cuda.set_device(self.gpu_index)

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
        * speaker, speaker_score, speaker_status, speaker_step - only when the request sets 'voice_recognition'; see
          _speaker_fields. The request's optional 'location_id' says which enrolled samples to compare first.

        A second command, 'speaker_embedding', returns the voice embedding of the audio instead of a transcription
        (used by scripts/ai/asr/speaker_id/enroll_voice.py to enroll a voice with exactly the server's model):
        * embedding - the vector; model - the embedding model's id; seconds - the audio's length.
        Its request may carry 'sample_rate' when the audio is not 16 kHz.


        Args:
            request: A dictionary that will contain fields. It should ALWAYS contain 'command', which represents WHAT the user wants to do. That will determine one of several scenarios:
                    Scenario 1: generating transcriptions
                        'command' = 'transcribe'
                    Scenario 2: a voice embedding, for enrollment
                        'command' = 'speaker_embedding'
            data: bytes - This will always be the audio from the client's microphone. Currently, its expected to be data that the Python library 'sounddevice' captures from a mic - a 16 bit signed integer, with values from -32,768 to 32,767 (in other words, raw PCM bytes)

        Returns:
            Tuple[dict, None] - The dictionary (that will be converted to JSON and sent to the client), None (Since this has to fit the format of what we may send to a client, that is (JSON, media_data) - and since this returns no media, its always None)
        """
        try:
            possible_commands = ('transcribe', 'speaker_embedding')
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
        if command == 'speaker_embedding':
            response = self._handle_speaker_embedding(request, data)
        elif command == 'transcribe':

            logger.debug(f"{ColoredText.YELLOW_TEXT}[{sessionID}]{ColoredText.END_TEXT}{ColoredText.CYAN_TEXT} Processing job{ColoredText.END_TEXT}")

            try:

                # ------------------------------------------------------------------ CPU only work
                # Nothing below touches the GPU, so it is deliberately done before the lock is
                # taken - decoding the client's audio while another client is mid transcription
                # costs that client nothing.
                #
                # Since the audio is in the format of what the Python library 'sounddevice' produces from a mic (see description above), we need to convert to what WhisperX is expecting,
                # which is a 32 bit floating point audio (i.e. np.frombuffer(data, dtype=np.int16).astype(np.float32)) with values normalized between -1 and 1 (i.e. MAX_POSITIVE_INT16_VALUE_AS_FLOAT, which normalizes
                # from int16 range (-32,768 to +32,767) to float32 range (-1.0 to +1.0))
                # NOTE: WhisperX _EXCLUSIVELY_ uses a 16k Sample rate! If its higher, WhisperX will downsample, but....why waste the bandwidth if its just going to resample
                audio_segment_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / AmadeoWhisperX.MAX_POSITIVE_INT16_VALUE_AS_FLOAT

                audio_for_lang_detect = whisperx.audio.pad_or_trim(audio_segment_np)

                # Voice recognition, if asked for: the speaker is worked out from the same audio, under the same GPU
                # lock, and returned with the transcription. Too little speech is not embedded at all.
                voice_recognition = bool(request.get('voice_recognition'))
                location_id = request.get('location_id') if isinstance(request.get('location_id'), str) else ''
                # Whether this client lets its microphone be recorded as field clips (the config must allow it too)
                client_field_clips = (request.get('save_known_field_clips') is True, request.get('save_unknown_field_clips') is True)
                # How much was actually said is only known once the words are aligned (see speech_span); until
                # then, the whole chunk
                seconds = len(audio_segment_np) / 16000.0
                embedding, embed_error, speech_audio = None, '', None

                # ------------------------------------------------------------------ GPU only work
                # The lock is held for the model calls and nothing else. The models are shared
                # across every client thread and are not safe to call concurrently, but a client
                # holding a session open is not a reason to keep the GPU to itself - so the lock
                # is scoped to this request's inference, not to the connection.
                #
                # Waiting here is a plain block rather than a 'GPU busy' rejection: a handful of
                # household clients doing one or two second jobs will not queue up long enough
                # for that to be worth the extra protocol.
                aligned_result = None

                with self.gpu_lock:
                    try:
                        # Every server thread has to claim the configured GPU for itself; see
                        # _bind_thread_to_gpu for why this is not just done once at startup.
                        self._bind_thread_to_gpu()

                        detected_language, language_confidence = detect_language_with_probability(self.asr_model, audio_for_lang_detect)

                        result = self.asr_model.transcribe(audio_segment_np, batch_size=1, language=detected_language)

                        if result and "segments" in result and result["segments"]:
                            aligned_result = whisperx.align(result["segments"], self.align_model, self.metadata, audio_segment_np, device=self.device)

                        if voice_recognition and self.speaker_embedder is not None:
                            # Only the speech is judged: its length against min_seconds, and its sound for the
                            # embedding (the trailing silence would only dilute the voice)
                            start, end, seconds = speech_span(aligned_result, seconds)
                            if seconds >= self.speaker_settings.min_seconds:
                                # A failed embedding must not cost the transcription: note it and carry on
                                try:
                                    speech_audio = audio_segment_np[int(start * 16000):int(end * 16000)]
                                    embedding = self.speaker_embedder.embed(speech_audio)
                                except Exception as e:
                                    embed_error = f"speaker embedding failed: {e}"
                                    logger.warning(embed_error)

                    finally:
                        # Release this request's VRAM before handing the lock on, so the next
                        # client's peak allocation does not have to sit alongside this one's
                        # leftovers. empty_cache() acts on the calling thread's current device,
                        # which _bind_thread_to_gpu has already set to the right card.
                        gc.collect()
                        if self.use_cuda:
                            torch.cuda.empty_cache()

                # ------------------------------------------------------------------ CPU only work
                # Scoring the words and assembling the response is pure Python, so it happens
                # after the lock has been released and the next client is already under way.
                full_text = ""
                average_word_confidence = 0.0

                if aligned_result is not None:
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

                speaker_fields = self._speaker_fields(voice_recognition, location_id, seconds, embedding, embed_error,
                                                      speech_audio, client_field_clips) if full_text else {}

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
                        'file_size': 0,
                        **speaker_fields
                    }

                    logger.info(f"Transcription completed and sent. Lang: {detected_language} ({language_confidence:.2f}), Avg Conf: {average_word_confidence:.2f}")

            except Exception as e:
                logger.warning(f"Error during transcription: {e}")
                response = {
                    'success': False,
                    "message": str(e),
                    'file_size': 0
                }


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


    def _handle_speaker_embedding(self, request: Dict[str, Any], data: bytes) -> Dict[str, Any]:
        """
        Handles the 'speaker_embedding' command: the voice embedding of a clip, for enrolling a voice.

        :param request: The request; an optional 'sample_rate' (default 16000) says the rate of the audio.
        :param data: The audio as raw 16 bit signed PCM, mono.
        :return: The response dictionary: 'embedding', 'model' and 'seconds' on success.
        """
        if self.speaker_embedder is None:
            return {'success': False, 'type': 'error', 'file_size': 0,
                    'message': "Speaker identification is not configured on this ASR server (no 'speaker_id' block in its --json config)."}
        sample_rate = request.get('sample_rate', 16000)
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 192000:
            return {'success': False, 'type': 'error', 'file_size': 0, 'message': f"Invalid sample_rate {sample_rate!r}."}
        if not data:
            return {'success': False, 'type': 'error', 'file_size': 0, 'message': "No audio sent."}
        try:
            from amadeo_utils.ai.asr.speaker_embedder import resample
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / AmadeoWhisperX.MAX_POSITIVE_INT16_VALUE_AS_FLOAT
            audio = resample(audio, sample_rate)
            with self.gpu_lock:
                try:
                    self._bind_thread_to_gpu()
                    embedding = self.speaker_embedder.embed(audio)
                finally:
                    if self.use_cuda:
                        torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Speaker embedding failed: {e}")
            return {'success': False, 'type': 'error', 'file_size': 0, 'message': f"Speaker embedding failed: {e}"}
        logger.info(f"Speaker embedding computed ({len(audio) / 16000.0:.1f} s of audio).")
        return {'success': True, 'type': 'speaker_embedding', 'message': '', 'file_size': 0,
                'embedding': embedding, 'model': self.speaker_embedder.model_name, 'seconds': len(audio) / 16000.0}


    ################################################################################################################### Parsing Arguments From Command Line ####################################################################################################################

    @staticmethod
    def load_json_config(filepath: str) -> dict:
        """
        Loads the WhisperX server's JSON config (--json). Every field is optional; the keys are the command line's
        long names with dashes turned into underscores, plus an optional 'speaker_id' object that turns on speaker
        identification. Example:

            {
                "host": "127.0.0.1",
                "port": 65432,
                "model": "large-v3",
                "language_code": "en",
                "gpu": 1,
                "combined_confidence_cutoff": 1.0,
                "speaker_id": {
                    "profiles_dir": "/path/to/voice-profiles",
                    "threshold": 0.50,
                    "margin": 0.05,
                    "min_seconds": 1.0,
                    "locations": {"office": {"threshold": 0.55}}
                },
                "log_file": "/path/to/logs/asr-server.log"
            }

        Only the speaker_id object's type is checked here; its contents are checked (speaker_id.settings_from_dict)
        when the server starts, so a bad speaker_id block stops the server rather than silently leaving recognition
        off. A key this server does not know is logged as a warning and ignored - with a hint when it is a speaker_id
        setting put at the top level; keys starting with '_' ("_comment") are notes and pass silently.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict: The recognised fields that are present in the file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file content is not valid JSON.
            TypeError: If the file is not a JSON object, or a field's value is not of the expected type.
        """
        optional_fields = AmadeoWhisperX.SERVER_CONFIG_FIELDS

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Error: The file '{filepath}' was not found.")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Error: '{filepath}' must hold a JSON object.")

        scraped_data = {}
        for field, expected_type in optional_fields.items():
            if field in data:
                value = data[field]
                # bool is a subclass of int in Python, so rule it out explicitly for the numeric fields
                if isinstance(value, bool) or not isinstance(value, expected_type):
                    raise TypeError(f"Error: Field '{field}' in '{filepath}' has unexpected type '{type(value).__name__}'.")
                scraped_data[field] = value

        # A typo, or a setting at the wrong level, would otherwise be ignored without a word
        for key in speaker_id.unknown_settings(data, optional_fields):
            hint = " - it belongs inside the 'speaker_id' block" if key in speaker_id.SPEAKER_ID_KEYS else ''
            logger.warning(f"Unknown setting '{key}' in {filepath} ignored{hint}.")

        return scraped_data

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
        parser.add_argument("-g", "--gpu", type=int, default=AmadeoWhisperX.GPU_INDEX,help="The index of the CUDA GPU to load the models onto, matching the order 'nvidia-smi -L' reports (0 for the first GPU, 1 for the second, and so on); the ordering is pinned to the physical slot order via CUDA_DEVICE_ORDER, so it does not depend on which card CUDA considers fastest. Defaults to the first GPU. Ignored if no CUDA device is available.")
        parser.add_argument("-ccc", "--combined_confidence_cutoff", type=float, default=AmadeoWhisperX.COMBINED_CONFIDENCE_CUTOFF,help="Each transcription has a confidence score (0-1) for the language and then an average confidence score for the words; if both of these numbers, summed, are less than this, the transcription will not be sent back to the client (as it is probably a false reading).")
        parser.add_argument("--json", type=str, default="", help="If this points to a valid JSON file, the ENTIRE parameter settings are pulled from that file, and the defaults - and other arguments passed from the command line - are ignored. If the JSON load fails for whatever reason, though, the defaults WILL be engaged. Just remember that if there is a dash in the arg name, its going to be an underscore in the JSON. Speaker identification can only be turned on here, with a 'speaker_id' object (see load_json_config): it points at the voice profiles, which are biometric data, so keep the file out of the repository. The Hugging Face token for the embedding model is read from HF_TOKEN.")

        argDict = {}

        try:
            args = parser.parse_args()
            use_default_arg_config = True  # This is only flipped if we successfully load from a JSON file

            json_config_file = args.json

            if json_config_file and os.path.exists(json_config_file):
                try:
                    config_dict = AmadeoWhisperX.load_json_config(json_config_file)

                    argDict['host'] = config_dict.get('host', AmadeoWhisperX.HOST)
                    argDict['port'] = config_dict.get('port', AmadeoWhisperX.PORT)
                    argDict['model'] = config_dict.get('model', AmadeoWhisperX.WHISPER_MODEL_NAME)
                    argDict['language_code'] = config_dict.get('language_code', AmadeoWhisperX.LANGUAGE_CODE)
                    argDict['gpu'] = config_dict.get('gpu', AmadeoWhisperX.GPU_INDEX)
                    argDict['combined_confidence_cutoff'] = config_dict.get('combined_confidence_cutoff', AmadeoWhisperX.COMBINED_CONFIDENCE_CUTOFF)
                    argDict['speaker_id'] = config_dict.get('speaker_id')
                    argDict['log_file'] = config_dict.get('log_file', '')

                    logger.info(f"Config loaded from JSON {json_config_file}.")

                    use_default_arg_config = False

                except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Could not load JSON config [{json_config_file}] - there are errors. Will attempt to load other defaults or args. Error: {e}.")

            elif json_config_file:
                logger.warning(f"Could not load JSON config [{json_config_file}] - file does not exist. Loading from defaults or other parameters sent.")

            if use_default_arg_config:
                argDict['host'] = args.host
                argDict['port'] = args.port
                argDict['model'] = args.model
                argDict['language_code'] = args.language_code
                argDict['gpu'] = args.gpu
                argDict['combined_confidence_cutoff'] = args.combined_confidence_cutoff
                argDict['speaker_id'] = None    # speaker identification is only configured through --json
                argDict['log_file'] = ''        # so is the log file: screen only

            logger.debug(f"{ColoredText.BLUE_TEXT}WhisperXUtils.get_args_dict_server: Config loaded; host: {argDict['host']} port: {argDict['port']} model: {argDict['model']} gpu: {argDict['gpu']} speaker ID: {'on' if argDict['speaker_id'] else 'off'}.{ColoredText.END_TEXT}")

        except SystemExit as e:
            argDict = {}
            if e.code == 0:
                # --help was used, so print no error
                print(f"{ColoredText.BLUE_TEXT}Thank you!{ColoredText.END_TEXT}")
            else:
                print(f"{ColoredText.RED_TEXT}WhisperXUtils.get_args_dict_server: Invalid arguments.{ColoredText.END_TEXT}")

        return argDict
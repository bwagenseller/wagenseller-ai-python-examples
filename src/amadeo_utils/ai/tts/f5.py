"""
This helps implement the f5-TTS - it has some basic setup features, some functions necessary for F5-TTS, handles a voice library (VERY useful for F5-TTS), converts the F5-TTS output to PCM (in a WAV container),
has a method that handles a server request (and returns a dictionary along with the raw PCM audio) and a few other things.

The voice library is located in `you/defined/path/voices.json`. It holds all of your voices, references their sample location, and stores their transcript. And example of this file:
```
{
  "default": {
    "file": "rick-sanchez.wav",
    "text": "Being nice is something stupid people do to hedge their bets. Now I haven't been exactly subtle about how little I trust marriage. I couldn't make it work and I can turn a"
  },
  "alexa": {
    "file": "alexa.wav",
    "text": "Memorial Day is a federal holiday in the United States for mourning the U S military personnel who died while serving in the United States armed forces. It is observed on the last Monday of May."
  }
}
```

Note the 'default' - you need a default. After that, the world is your oyster. Also, the text field must match the spoken words in the file EXACTLY. When you are calling this script, simply use the name (i.e. 'default', 'alexa', etc)

"""

import os

# CUDA numbers GPUs by 'FASTEST_FIRST' unless told otherwise, which ranks them by compute
# capability rather than by the slot they sit in - so CUDA's device 0 is not necessarily the
# device 0 that 'nvidia-smi' reports. On a mixed pair such as an RTX 3090 (sm_86) alongside an
# RTX 4070 Ti (sm_89), CUDA puts the newer-architecture 4070 Ti first even though nvidia-smi
# lists the 3090 first, so '--gpu 0' would silently pick the opposite card to the one the user
# was looking at. 'PCI_BUS_ID' orders by physical slot, which is what nvidia-smi shows and what
# anyone reading '--gpu 0' will expect.
#
# This has to be set before CUDA is initialised, hence before torch (or f5_tts, which imports
# torch itself) is imported rather than in the constructor. setdefault is used so that an
# explicit setting in the environment still wins.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
from pathlib import Path
from f5_tts.api import F5TTS
import soundfile as sf
import logging
import json
from typing import Dict, Tuple, Any
import io
import gc
import inspect
import threading
import argparse
import numpy as np # NumPy array library
from amadeo_utils.ai.tts.generic_utils import PhoneticReplacement, SplitDialogue

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class AmadeoF5:

    HOST = 'localhost'
    PORT = 8888
    VOICES = 'voice_samples'
    SPEED = .95 # .7 = 30% slower, .85 = 15% slower, 1 = normal, 1.15 = 15% faster, 1.3 = 30% faster
    NFE_STEP = 64 # NFE (Number of Function Evaluations) F5-TTS uses NFE=32 by default, but you can reduce this to NFE=7 with Sway Sampling for much faster generation with minimal quality loss
    CFG_STRENGTH = 2.0 # Classifier-free guidance. CFG=2 is commonly used for better quality control (usually left off by default)
    CROSS_FADE_DURATION = .15 # cross_fade_duration: blending between audio segments (in seconds)
    MODEL = 'F5TTS_v1_Base'
    MODEL_PATH = ''
    PHONETIC_REPLACEMENT_FILE = ''
    USE_DIFFERENT_SPEAKERS = False
    SAMPLE_RATE = 24000 # ONLY use this as a last resort - otherwise, use what is returned from F5-TTS
    NARRATOR_VOICE = ''
    VOICE_MAPPING_FILE = ''
    PAUSE_DURATION = .4
    SEGMENT_SPACER_DURATION = .3
    GPU_INDEX = 0


    def __init__(self, config_dict:Dict):

        self.config = config_dict

        self.model = self.config['model']
        self.model_path = self.config['model_path']

        self.cross_fade_duration = self.config['cross_fade_duration']

        self.pause_duration = self.config['pause_duration']
        self.segment_spacer_duration = self.config['segment_spacer_duration']

        self.voice_samples_dir = self.config['voices']

        # ---------------------------------------------------------------------------- GPU selection
        # Which physical GPU to run on. Absent from the dictionary (i.e. when this class is used
        # outside of the server script) it falls back to the first GPU, which is the only index
        # guaranteed to exist on a CUDA machine.
        self.gpu_index = self.config.get('gpu', AmadeoF5.GPU_INDEX)

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()

            # Fail at startup rather than quietly generating audio on the wrong card - a silent
            # fallback would look identical to success while ignoring what was asked for.
            if self.gpu_index < 0 or self.gpu_index >= device_count:
                raise ValueError(f"Requested GPU index {self.gpu_index} does not exist; this machine has {device_count} CUDA device(s), so valid indexes are 0 through {device_count - 1}.")

            # This class runs behind a threaded server, and torch tracks the 'current' device per
            # thread rather than per process. Pinning it here only fixes the thread doing the
            # loading; every worker thread has to pin it again before it touches the GPU (see
            # _bind_thread_to_gpu), otherwise that thread would silently default back to GPU 0.
            torch.cuda.set_device(self.gpu_index)

            self.device = f"cuda:{self.gpu_index}"

            logger.info(f"F5-TTS: Using GPU {self.gpu_index} ({torch.cuda.get_device_name(self.gpu_index)}) of {device_count} available.")
        else:
            # No CUDA at all - the GPU index is meaningless, so note that it is being ignored
            # instead of letting the caller assume it took effect.
            if self.gpu_index != AmadeoF5.GPU_INDEX:
                logger.warning(f"A GPU index ({self.gpu_index}) was requested but no CUDA device is available; falling back to CPU and ignoring the index.")

            # Reset the index so the rest of the class does not have to keep asking whether it is
            # meaningful; on CPU it is simply unused.
            self.gpu_index = AmadeoF5.GPU_INDEX
            self.device = "cpu"

        # Convenience flag - self.device is 'cuda:N' rather than a bare 'cuda', so string
        # comparisons against "cuda" elsewhere would not hold.
        self.use_cuda = self.device.startswith("cuda")

        # ---------------------------------------------------------------------------- Locks
        # AmadeoServer hands every client connection to its own thread, and its docstring is
        # explicit that anything sharing a resource such as a GPU has to do its own locking. The
        # F5-TTS model below is loaded once and shared by all of those threads, so:
        #
        #  * gpu_lock serialises the actual inference calls. It is taken per request rather than
        #    per connection - a client holding a session open is not a reason to keep the GPU to
        #    itself. Waiting is a plain block rather than a 'GPU busy' rejection: a handful of
        #    household clients will not queue up long enough for that to be worth the extra
        #    protocol.
        #  * dialogue_lock guards the one-time voice confirmation pass, which mutates the shared
        #    dialogue_helper. That is CPU work, so it gets its own lock rather than borrowing the
        #    GPU one - two threads arriving together would otherwise race on the same dictionary.
        self.gpu_lock = threading.Lock()
        self.dialogue_lock = threading.Lock()

        # Determine if the phonetic replacement file exists - if it does, set the phonetic_replacement for replacements
        self.phonetic_replacement = None
        if self.config.get('phonetic_replacement_file', ''):
            self.phonetic_replacement = PhoneticReplacement(self.config['phonetic_replacement_file'])

        self.dialogue_helper = SplitDialogue(self.config.get('use_different_speakers', False), self.config.get('narrator_voice', AmadeoF5.NARRATOR_VOICE), self.config.get('voice_mapping_file', AmadeoF5.VOICE_MAPPING_FILE))

        # Load voice configuration from voices.json file
        # This maps voice names to audio files and their transcriptions
        self.voice_config = self.load_voice_config()

        # Initialize F5-TTS model (requires f5-tts package to be installed)
        # IMPORTANT: Model is loaded ONCE here and reused for all requests
        # This provides significant performance benefits for successive calls
        try:

            # F5TTS takes the device as a constructor keyword and holds onto it for every later
            # infer() call, so it is the only place the card can be chosen. The keyword arrived
            # after F5-TTS' first releases, though, so it is only passed if the installed version
            # actually accepts it - on an older build the torch.cuda.set_device call above still
            # steers it, because F5-TTS falls back to a bare 'cuda' and torch resolves that to
            # whatever the calling thread's current device is.
            if 'device' in inspect.signature(F5TTS.__init__).parameters:
                device_kwargs = {'device': self.device}
            else:
                device_kwargs = {}
                logger.warning(f"This build of F5-TTS does not accept a 'device' argument; relying on the thread's current CUDA device ({self.device}) instead. Upgrade f5-tts if the model ends up on the wrong GPU.")

            if self.model_path:
                logger.info(f"Loading Custom F5-TTS model {self.model_path} on {self.device}... (this may take a moment)")
                self.tts_model = F5TTS(ckpt_file=self.model_path, **device_kwargs)  # Load the pre-trained F5-TTS model into memory
            elif self.model:
                logger.info(f"Loading F5-TTS model {self.model} on {self.device}... (this may take a moment)")
                self.tts_model = F5TTS(model=self.model, **device_kwargs)  # Load the pre-trained F5-TTS model into memory
            else:
                logger.info(f"Loading F5-TTS model on {self.device}... (this may take a moment)")
                self.tts_model = F5TTS(**device_kwargs)  # Load the pre-trained F5-TTS model into memory
            logger.info("F5-TTS model loaded successfully - ready for requests")
        except Exception as e:
            # Other errors loading the model (GPU issues, model files missing, etc.)
            logger.error(f"Error loading F5-TTS: {e}")
            raise


    def _bind_thread_to_gpu(self):
        """
        Pin the calling thread to the GPU this instance was configured for.

        torch stores the 'current' CUDA device per thread, and it always starts out as device 0.
        AmadeoServer hands every client connection to a fresh thread, so a thread that has not
        been pinned would send any torch work that was given a bare 'cuda' device to GPU 0
        regardless of what was asked for. On a single GPU machine that is invisible; on a multi
        GPU machine it silently splits the work across cards. Calling this at the top of each GPU
        bound request keeps every thread on the same card.

        This is a no-op when running on CPU.

        Returns:
            None
        """
        if self.use_cuda:
            torch.cuda.set_device(self.gpu_index)


    def _confirm_dialogue_voices(self):
        """
        Resolve every mapped speaker name - and the narrator - to a voice that actually exists in
        voices.json, once, on the first request that needs it.

        This is only meaningful when 'use_different_speakers' is set. It rewrites the shared
        dialogue_helper in place, which is why it is guarded: with a threaded server, two clients
        arriving at the same time on a cold server would otherwise walk the same dictionary while
        the other one was rewriting it. The flags are re-checked inside the lock so that the
        second thread through does not redo the work the first one just finished.

        This is pure configuration lookup - no GPU work - so it takes its own lock rather than the
        GPU lock, and callers should run it before claiming the GPU.

        Returns:
            None
        """
        with self.dialogue_lock:
            if not self.dialogue_helper.voice_mapping_list_confirmed:
                # go through each name in the mapped list to get what F5-TTS calls this voice
                for name in self.dialogue_helper.voice_mapping.keys():
                    # get the dictionary based on the passed name, and ensure its in there (if not, use the default) it usually will be the same, if this was set up properly, but on the off chance it wasn't....
                    one_character = self.dialogue_helper.voice_mapping[name]
                    one_character['tts_name'] = self.get_voice_info(one_character['tts_name'])[0] #get just the first item in the tuple returned (we do not care about the rest)
                self.dialogue_helper.voice_mapping_list_confirmed = True

            # settle the narrator - either confirm the voice OR set to default if the narrator voice is missing or is not in the voice listing
            if self.dialogue_helper.narrator and not self.dialogue_helper.narrator_confirmed:
                self.dialogue_helper.narrator = self.get_voice_info(self.dialogue_helper.narrator)[0]
                self.dialogue_helper.narrator_confirmed = True


    def load_voice_config(self):
        """
        Load voice configuration from voices.json file

        The voices.json file maps voice names to audio files and their exact transcriptions.
        F5-TTS requires the exact text that was spoken in each reference audio file
        for proper voice cloning to work.

        Format:
        {
          "voice-name": {
            "file": "audio-file.wav",
            "text": "Exact transcription of what is spoken in the audio",
            "speed": 0.92,
            "nfe_step": 64,
            "cfg_strength": 2.0
          }
        }

        *** Required Parameters in JSON ***
        * 'voice-name' is the shorthand name by which you will reference this voice.
        * 'file' is the location of the sample .wav file for this voice
        * 'text' is the EXACT text in the .wav file.

        *** Optional Parameters in JSON ***
        **The default is used if these are not set**
        * 'speed' is the speed of the generated text - if your sample voice speaks quickly you WILL have to slow it down a bit.  .7 = 30% slower, .85 = 15% slower, 1 = normal, 1.15 = 15% faster, 1.3 = 30% faster
        * 'nfe_step' is Number of Function Evaluations. F5-TTS uses NFE=32 by default, but you can reduce this to NFE=7 with Sway Sampling for much faster generation with minimal quality loss
        * 'cfg_strength' is Classifier-free guidance. CFG=2 is commonly used for better quality control (usually left off by default)

        Returns:
            dict: Voice configuration mapping voice names to file/text pairs
        Raises:
            Exception: If there is a problem getting voices.json or it does not exist
        """
        config_file = Path(self.voice_samples_dir) / "voices.json"

        # Try to load existing configuration file
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                    # ensure all keys are lower case
                    config = {k.lower(): v for k, v in config.items()}
                logger.info(f"Loaded voice configuration with {len(config)} voices")
                return config
            except Exception as e:
                logger.critical(f"Error loading voice config: {e}")
        else:
            logger.critical(f"No voices.json file for F5-TTS - will not be able to generate voices...")


    def get_voice_info(self, voice_name: str) -> Tuple[str, str, str, float, int, float]:
        """
        Get the audio file path and reference text for a given voice name

        This function looks up a voice in the configuration and returns the path
        to its audio file and the exact transcription text. If the requested voice
        doesn't exist, it falls back to the "default" voice.

        Args:
            voice_name: Name of the voice to look up (e.g., "rick-sanchez-1")

        Returns:
            Tuple: (voice_name, file_path, reference_text, speed, nfe_step, cfg_strength)

        Raises:
            FileNotFoundError: If voice not found in config or audio file missing on disk
        """

        # ensure key is lower case
        voice_name = voice_name.lower()

        # Check if requested voice exists in configuration
        if voice_name not in self.voice_config.keys():
            # Try to fall back to default voice
            if "default" in self.voice_config:
                logger.warning(f"Voice '{voice_name}' not found, using default voice")
                voice_name = "default"
            else:
                # No default voice either - configuration problem
                raise FileNotFoundError(f"Voice '{voice_name}' not found in configuration and no default voice available")

        # Get voice configuration entry - at this point we are guaranteed that the voice exists
        voice_info = self.voice_config[voice_name]
        voice_file = voice_info["file"]  # Audio file name (e.g., "rick-sanchez-1.wav")
        voice_text = voice_info["text"]  # Exact transcription of the audio

        voice_speed = voice_info.get("speed", AmadeoF5.SPEED)  # Exact speed of the audio
        voice_nfe_step = voice_info.get("nfe_step", AmadeoF5.NFE_STEP)  # Exact nfe step of the audio
        voice_cfg_strength = voice_info.get("cfg_strength", AmadeoF5.CFG_STRENGTH)  # Exact cfg strength of the audio

        # Build full path to the audio file
        voice_path = Path(self.voice_samples_dir) / voice_file

        # Check if the actual audio file exists on disk
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file '{voice_file}' not found in {self.voice_samples_dir}")

        return voice_name, str(voice_path), voice_text, voice_speed, voice_nfe_step, voice_cfg_strength


    def single_pass(self, text:str, voice_name: str) -> Tuple[np.ndarray, int]:
        """
        This makes a 'single pass' at F5-TTS and returns the raw tensor data. this is done like this because there is a chance the speech is broken up by different speakers and the narrator, so this must be called multiple times to properly
        build the audio.

        Args:
            text: The text to be turned to speech.
            voice_name: The voice name, as understood by Kokoro

        Returns:
            Tuple (np.ndarray, sample_rate): Raw audio tensor data plus the sample rate
        """
        # Get the reference audio file and its transcription
        voice_name, voice_path, ref_text, speed, nfe_step, cfg_strength = self.get_voice_info(voice_name)

        logger.info(f"Generating speech using voice '{voice_name}'")

        # Call F5-TTS to generate speech using the pre-loaded model
        # ref_file: reference audio file for voice cloning
        # ref_text: exact transcription of the reference audio
        # gen_text: new text to generate speech for
        # cross_fade_duration: blending between audio segments (in seconds)
        # speed: playback speed multiplier (1.0 = normal speed)
        result = self.tts_model.infer(
            ref_file=voice_path,
            ref_text=ref_text,
            gen_text=text,
            speed = speed,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            cross_fade_duration=self.cross_fade_duration
        )

        # Handle return from F5-TTS API. Expected format: (audio_tensor, sample_rate, extra_data)
        if isinstance(result, tuple) and len(result) >= 3:
            tensor, sample_rate, _ = result
        else:
            # Fallback for unexpected formats
            raise ValueError(f"Unexpected return format from F5-TTS: {type(result)}, length: {len(result) if isinstance(result, tuple) else 'N/A'}")

        return tensor, sample_rate

    def generate_speech(self, text, voice_name):
        """
        Generate speech audio using F5-TTS

        This is the core function that takes text and a voice name, uses the pre-loaded
        F5-TTS model to generate speech, and returns a path to the generated WAV file.

        The F5-TTS model is already loaded in memory (from __init__), so this function
        is fast on successive calls.

        Args:
            text: Text to convert to speech
            voice_name: Name of voice to use (must exist in voices.json)

        Returns:
            raw_data: PCM data in a WAV container; from here, we can directly send this over a network connection OR simply save to a file

        Raises:
            Exception: If speech generation fails for any reason
        """

        # ------------------------------------------------------------------ CPU only work
        # Nothing here touches the GPU, so it is deliberately done before the lock is taken -
        # rewriting one client's text while another client is mid generation costs that client
        # nothing.

        # Verify all speakers first, if it has not been done
        if self.dialogue_helper.use_different_speakers:
            self._confirm_dialogue_voices()

        # if the phonetic replacement object exists, replace the words in the text with what they sound like phonetically
        if self.phonetic_replacement:
            text = self.phonetic_replacement.phonetic_replacement(text)

        #initialize - hopefully this is overwritten, but if not, this is a good guess
        sample_rate = AmadeoF5.SAMPLE_RATE

        try:

            # ------------------------------------------------------------------ GPU only work
            # The lock is held for the F5-TTS calls and nothing else. The model is shared across
            # every client thread and is not safe to call concurrently.
            #
            # For multi-speaker text the lock covers the whole segment loop rather than each
            # single_pass: the segments of one line of dialogue belong together, and interleaving
            # two clients' segments would just thrash the GPU for no gain in fairness.
            with self.gpu_lock:
                try:
                    # Every server thread has to claim the configured GPU for itself; see
                    # _bind_thread_to_gpu for why this is not just done once at startup.
                    self._bind_thread_to_gpu()

                    if not self.dialogue_helper.use_different_speakers:
                        # if we are not using multiple speakers, just run a single pass through F5-TTS
                        final_tensor, sample_rate = self.single_pass(text, voice_name)

                        # Handle numpy array vs torch tensor
                        # Different F5-TTS versions return different data types, but the version as of August 2025 is indeed a NumPy array
                        if isinstance(final_tensor, np.ndarray):
                            final_tensor = torch.from_numpy(final_tensor).float()
                    else:
                        # if we ARE using multiple speakers (the narrator and the mapped speakers
                        # were already resolved by _confirm_dialogue_voices, above the lock)

                        audio_segments = []
                        parts = self.dialogue_helper.split_narrator_dialogue_with_names(text)
                        for content, speaker_type, character_name in parts:
                            if speaker_type == 'pause':
                                logger.info("Inserting pause...")
                                # if the '**' was passed, this could be a forced pause - so add that
                                pause_samples = int(sample_rate * self.pause_duration)
                                segment_tensor = torch.zeros(pause_samples)
                            else:
                                if speaker_type == 'narrator':
                                    this_voice = self.dialogue_helper.narrator
                                elif character_name is not None:
                                    temp_name = character_name.lower().strip()
                                    if temp_name in self.dialogue_helper.voice_mapping.keys():
                                        one_character = self.dialogue_helper.voice_mapping[temp_name]
                                    else:
                                        logger.warning(f"Warning - {temp_name} not in names list, using 'default' instead.")
                                        temp_name = 'default'
                                        one_character = self.dialogue_helper.voice_mapping[temp_name]
                                    if one_character and one_character.get('tts_name', ''):
                                        this_voice = one_character['tts_name']
                                    else:
                                        this_voice = voice_name
                                else:
                                    this_voice = voice_name

                                # run this subsection through F5-TTS
                                segment_tensor, sample_rate = self.single_pass(content, this_voice)  # Returns tensor

                                # Convert to torch tensor immediately:
                                if isinstance(segment_tensor, np.ndarray):
                                    segment_tensor = torch.from_numpy(segment_tensor).float()

                            audio_segments.append(segment_tensor)

                        if audio_segments:
                            # Add pauses between segments (all as tensors)
                            pause_samples = int(sample_rate * self.segment_spacer_duration)
                            pause = torch.zeros(pause_samples)

                            # Concatenate all tensors
                            final_audio = []
                            for i, segment in enumerate(audio_segments):
                                final_audio.append(segment)
                                if i < len(audio_segments) - 1:
                                    final_audio.append(pause)

                            final_tensor = torch.cat(final_audio)
                        else:
                            # Nothing to assemble - raise rather than fall through, as everything
                            # below this point works on a tensor that would never have been built.
                            raise ValueError("No audio segments returned for multiple speakers - nothing to turn into audio.")
                    # END - multiple speakers

                finally:
                    # Release this request's VRAM before handing the lock on, so the next client's
                    # peak allocation does not have to sit alongside this one's leftovers.
                    # empty_cache() acts on the calling thread's current device, which
                    # _bind_thread_to_gpu has already set to the right card.
                    gc.collect()
                    if self.use_cuda:
                        torch.cuda.empty_cache()

            # ------------------------------------------------------------------ CPU only work
            # Reshaping the tensor and encoding the WAV is pure CPU work, so it happens after the
            # lock has been released and the next client is already under way.

            # Ensure audio tensor has correct dimensions for saving
            # torchaudio.save expects (channels, samples) format
            if final_tensor.dim() == 1:
                # 1D tensor (samples only) -> add channel dimension
                # Shape changes from (samples,) to (1, samples)
                final_tensor = final_tensor.unsqueeze(0)
                logger.debug("Added channel dimension to 1D audio tensor")
            elif final_tensor.dim() == 3:
                # 3D tensor (batch, channels, samples) -> remove batch dimension
                # Shape changes from (1, channels, samples) to (channels, samples)
                final_tensor = final_tensor.squeeze(0)
                logger.debug("Removed batch dimension from 3D audio tensor")

            # Convert to PCM data in a WAV container (from here, we can directly send this over a network connection OR simply save to a file)

            raw_data = self.convert_to_wav(final_tensor, sample_rate)

            logger.info(f"Successfully generated {final_tensor.shape[1]/sample_rate:.1f}s of audio at {sample_rate}Hz")
            return raw_data

        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            # Re-raise the exception so the caller can handle it appropriately
            raise

    def convert_to_wav(self, local_data, local_sample_rate):
        """
        Converts the data directly from the F5 output to PCM data in WAV container format without saving to disk

        Args:
            local_data: (either class 'numpy.ndarray'> or 'torch.Tensor') The output data from F5-TTS infer
            local_sample_rate: (Class: int) the sample rate of the local_data.

        Returns:
            data: (class: bytes) PCM data in WAV container format (that can either be saved to a file or sent over a network connection)

        Raises:

        """
        #
        # creates an in-memory file-like object that acts just like a real file, but stores data in RAM instead of on disk

        # Handle PyTorch tensors - Claude told me we would need to convert from a PyTorch tensor, and this IS a PyTorch tensor - but - this is apparently not needed.
        #if torch.is_tensor(local_data):
        #    local_data = local_data.detach().cpu().numpy()

        # Ensure proper shape (soundfile expects 1D for mono)
        # This IS needed - iwe we do not do this, we get an error: <_io.BytesIO object at 0x7852400e4bd0>: Format not recognised.
        if local_data.ndim == 2 and local_data.shape[0] == 1:
            local_data = local_data.squeeze(0)

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, local_data, local_sample_rate, format='WAV')
        return wav_buffer.getvalue()


    def handle_client_request(self, request: Dict[str, Any], data:bytes = None):
        """
        This method is designed specifically to handle a request from a server - this class can stay running alongside a server class, but the server class will call this method when it gets a request (the server class will handle stuff like sockets etc etc, but this will handle the SPECIFIC
        tasks related to F5-TTS). This method (and other methods in other classes that implement this) expects a dictionary and data (bytes, which can represent all kinds of media files), although the data portion of that may not be used (depending on the case; in F5-TTS' case, this is not used).
        This should return a dictionary (that will be turned into JSON) and byte data (if applicable, and in our case it is - its the PCM audio data in a WAV container).

        To see what is expected of the basics of what is expected fpr the server, see the main description for 'amadeo_server.AmadeoServer', although there are some additional ones specific to F5-TTS:
        * 'command' - either 'service_tts' or 'show_voices', depending on what is to be done
        * if 'command' is 'service_tts', we also need 'voice' (the voice you want to use in the audio) as well as 'text' (the text you want the audio to read).

        To see what dictionary fields will be sent to the client. see the main description for 'amadeo_server.AmadeoServer'

        Args:
            request: A dictionary that will contain fields. It should ALWAYS contain 'command', which represents WHAT the user wants to do. That will determine one of several scenarios:
                    Scenario 1: generating speech from text
                        'command' = 'service_tts'
                        'voice' (representing an available 'voice' supported in this F5-TTS instance)
                        'text' (the text the user wishes to turn to speech)
                    Scenario 2: Listing available voices
                        'command' = 'show_voices'
            data: bytes - This will always be None (as the client purely sends text and not bytes for a media file)
        """
        try:
            possible_commands = ('service_tts', 'show_voices')
            command = request.get('command', '')
            address = request.get('client_address', 'NO_ADDRESS')
            port = request.get('client_port', 'NO_PORT')
            if not command or command not in possible_commands:
                command = 'service_tts'
                logger.warning(f"Request came in with no command from {address}:{port} - setting command to {command}.")

            # Extract required fields from the request
            if command == 'service_tts':
                text = request.get('text', '').strip()      # Text to speak
                voice = request.get('voice', 'default').strip()  # Voice to use

                # Validate that text is provided and not empty
                if not text:
                    raise ValueError("No text provided in request")

                # Log the request for debugging/monitoring
                logger.info(f"Request from {address}:{port} - Voice: '{voice}' to be serviced.")

        except (json.JSONDecodeError, KeyError) as e:
            # Invalid JSON format or missing required fields
            raise ValueError(f"Invalid JSON request: {e}")

        if command == 'service_tts':
            try:
                # Use F5-TTS to generate audio file (this uses the pre-loaded model)
                # This is where the actual AI text-to-speech happens
                raw_data = self.generate_speech(text, voice)

                return_dict = { 'success': True, 'type': 'audio', 'message': '', 'file_size': len(raw_data)}
                logger.info(f"Successfully processed request from {address}:{port}")

                return return_dict, raw_data

            except Exception as e:
                # Speech generation failed - send error response to client
                error_msg = f"Speech generation failed: {str(e)}"
                return_dict = { 'success': False, 'type': 'error', 'message': error_msg, 'file_size': 0}
                logger.error(f"Request from {address}:{port} failed: {error_msg}")

                return return_dict, None
        elif command == 'show_voices':
            logger.info(f"Servicing 'show_voices' request...")
            local_keys = ''
            for key in self.voice_config:
                local_keys += f"\n{key}"
            return { 'success': True, 'type': 'voices', 'message': local_keys, 'file_size': 0}, None
        else:
            return_dict = { 'success': False, 'type': 'error', 'message': f"No 'command' provided for F5-TTS service; available commands ({possible_commands}).", 'file_size': 0}
            return return_dict, None


    @staticmethod
    def load_json_config(filepath: str) -> dict:
        """
        Loads a JSON file and scrapes specific entries into a dictionary.

        Args:
            filepath (str): The path to the JSON file.

        Returns:
            dict: A dictionary containing the scraped configuration fields:
                  'voices' (str)

        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file content is not valid JSON.
            KeyError: If any of the required fields are missing from the JSON.
            TypeError: If a field's value is not of the expected type.
        """
        required_fields = {
            'voices': str
        }

        # Add optional fields with their types
        optional_fields = {
            'host': str,
            'port': int,
            'model': str,
            'model_path': str,
            'cross_fade_duration': float,
            'phonetic_replacement_file': str,
            'use_different_speakers': bool,
            'narrator_voice': str,
            'voice_mapping_file': str,
            'pause_duration': float,
            'segment_spacer_duration': float,
            'gpu': int
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
    def get_args_dict() -> dict:
        """
        Gets args dictionary for a traditional vector database, meant to save the conversation for later.
        """

        # Set up command-line argument parsing
        parser = argparse.ArgumentParser(description='F5-TTS Server - Voice cloning text-to-speech server',formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('--host', default=AmadeoF5.HOST, help='Server host address (default: localhost)')
        parser.add_argument('--port', type=int, default=AmadeoF5.PORT, help='Server port number (default: 8888)')

        parser.add_argument('--voices', default=AmadeoF5.VOICES, help='Voice samples directory (default: voice_samples)')
        parser.add_argument("--phonetic-replacement-file", default=AmadeoF5.PHONETIC_REPLACEMENT_FILE,help="The path/filename of the JSON file you will use for phonetic replacements (i.e. 'mr.' to 'mister'). Not required.")

        parser.add_argument("--cross-fade-duration", type=float, default=AmadeoF5.CROSS_FADE_DURATION,help="Smoothing between audio segments (in seconds). 0.0 = No blending (potential audio pops/clicks); 0.15 = Default smooth transitions; 0.3+ = Longer blends for very seamless audio. Only matters for longer text that gets split into multiple segments.")

        parser.add_argument("--pause-duration", type=float, default=AmadeoF5.PAUSE_DURATION,help="If a pause is indicated in the text (via the string: **), a pause of this length (in seconds, fractional) is inserted. This is only valid if 'use-different-speakers' is set. Note: if you segment a very short sentence with this (where one spoken part is 1-3 words), the model may truncate the speech.")
        parser.add_argument("--segment-spacer-duration", type=float, default=AmadeoF5.SEGMENT_SPACER_DURATION,help="If 'use-different-speakers' is set, this will add a segment of silence between speakers that equals this length in seconds (fractional).")

        parser.add_argument("--narrator-voice", default=AmadeoF5.NARRATOR_VOICE,help="The voice for the narrator; leave blank if you do not want to use this. The narrator voice MUST be in the file identified by the 'voices' parameter.")
        parser.add_argument("--use-different-speakers", type=bool, default=AmadeoF5.USE_DIFFERENT_SPEAKERS,help="Select this if you wish to use different speakers if they are identified with colons in the text (the narrator reads things between asterisks)")
        parser.add_argument("--voice-mapping-file", default=AmadeoF5.VOICE_MAPPING_FILE,help="This is a JSON file that maps a name that can appear in a script-like fashion to a voice in 'voices' for example, It can map 'am_adam' to the voice if it sees 'Adam:' in the text.")

        parser.add_argument("--model", default=AmadeoF5.MODEL,help="Select a pre-defined model. These are the options: F5TTS_v1_Base - The current default model; F5-TTS - Main model, good balance of speed/quality; E2-TTS - Alternative architecture, might have different characteristics.")
        parser.add_argument("--model_path", default=AmadeoF5.MODEL_PATH,help="The path to a custom model, if you find one on huggingface....")
        parser.add_argument("--gpu", type=int, default=AmadeoF5.GPU_INDEX,help="The index of the CUDA GPU to load the model onto, matching the order 'nvidia-smi -L' reports (0 for the first GPU, 1 for the second, and so on); the ordering is pinned to the physical slot order via CUDA_DEVICE_ORDER, so it does not depend on which card CUDA considers fastest. Defaults to the first GPU. Ignored if no CUDA device is available.")
        parser.add_argument("--json", type=str, default="", help="If this points to a valid JSON file, the ENTIRE parameter settings are pulled from that file, and the defaults - and other arguments passed from the command line - are ignored. If the JSON load fails for whatever reason, though, the defaults WILL be engaged. Just remember that if there is a dash in the arg name, its going to be an underscore in the JSON.")

        argDict = {}

        try:
            args = parser.parse_args()
            use_default_arg_config = True  # This is only flipped if we successfully load from a JSON file

            json_config_file = args.json

            if json_config_file and os.path.exists(json_config_file):
                try:
                    config_dict = AmadeoF5.load_json_config(json_config_file)

                    argDict['host'] = config_dict.get('host', AmadeoF5.HOST)
                    argDict['port'] = config_dict.get('port', AmadeoF5.PORT)

                    argDict['voices'] = config_dict['voices']
                    argDict['phonetic_replacement_file'] = config_dict.get('phonetic_replacement_file', AmadeoF5.PHONETIC_REPLACEMENT_FILE)

                    argDict['cross_fade_duration'] = config_dict.get('cross_fade_duration', AmadeoF5.CROSS_FADE_DURATION)

                    argDict['pause_duration'] = config_dict.get('pause_duration', AmadeoF5.PAUSE_DURATION)
                    argDict['segment_spacer_duration'] = config_dict.get('segment_spacer_duration', AmadeoF5.SEGMENT_SPACER_DURATION)

                    argDict['model'] = config_dict.get('model', AmadeoF5.MODEL)
                    argDict['model_path'] = config_dict.get('model_path', AmadeoF5.MODEL_PATH)
                    argDict['gpu'] = config_dict.get('gpu', AmadeoF5.GPU_INDEX)

                    argDict['voice_mapping_file'] = config_dict.get('voice_mapping_file', AmadeoF5.VOICE_MAPPING_FILE)
                    argDict['narrator_voice'] = config_dict.get('narrator_voice', AmadeoF5.NARRATOR_VOICE)
                    argDict['use_different_speakers'] = config_dict.get('use_different_speakers', AmadeoF5.USE_DIFFERENT_SPEAKERS)

                    logger.info(f"Config loaded from JSON {json_config_file}.")

                    use_default_arg_config = False

                except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Could not load JSON config [{json_config_file}] - there are errors. Will attempt to load other defaults or args. Error: {e}.")

            elif json_config_file:
                logger.warning(f"Could not load JSON config [{json_config_file}] - file does not exist. Loading from defaults or other parameters sent.")

            if use_default_arg_config:

                argDict['host'] = args.host
                argDict['port'] = args.port

                argDict['voices'] = args.voices
                argDict['phonetic_replacement_file'] = args.phonetic_replacement_file

                argDict['cross_fade_duration'] = args.cross_fade_duration

                argDict['pause_duration'] = args.pause_duration
                argDict['segment_spacer_duration'] = args.segment_spacer_duration

                argDict['model'] = args.model
                argDict['model_path'] = args.model_path
                argDict['gpu'] = args.gpu
                argDict['voice_mapping_file'] = args.voice_mapping_file
                argDict['use_different_speakers'] = args.use_different_speakers
                argDict['narrator_voice'] = args.narrator_voice

                logger.info(f"Config loaded from args / defaults.")

            if not os.path.exists(argDict['voices']):
                logger.error(f"{argDict['voices']} does not exist - subsequently, F5-TTS will not be able to load voices, so this is dead in the water.")

            if argDict['model_path'] and not os.path.exists(argDict['model_path']):
                logger.warning(f"{argDict['model_path']} does not exist - reverting to model {argDict['model']}.")
                argDict['model_path'] = ''

        except SystemExit as e:
            argDict = {}
            if e.code == 0:
                # --help was used, so print no error
                print(f"Thank you!")
            else:
                logger.error(f"Invalid arguments.")

        return argDict
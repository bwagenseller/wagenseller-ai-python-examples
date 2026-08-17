import warnings
from typing import Tuple

"""
suppress warnings:
UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1
  warnings.warn(
FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
"""

warnings.filterwarnings("ignore", message=".*dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*is deprecated.*")

"""
This helps implement Kokoro TTS - it has some basic setup features, some functions necessary for Kokoro, handles a voice library (VERY useful for juggling built-in and custom voices in Kokoro), converts the Kokoro output to PCM (in a WAV container),
has a method that handles a server request (and returns a dictionary along with the raw PCM audio) and a few other things.

The voice library is located in `/your/defined/path/some_voice_file_you_name.json`. It holds all of your voices. 
```
{
 "heart": {
   "kokoro_name": "af_heart",
   "language": "American English",
   "traits": "woman, mid voice"
 },
  },
 "adam": {
   "kokoro_name": "am_adam",
   "language": "American English",
   "traits": "man, black?, mid voice"
 },
 "marsha": {
   "file": "/media/kokoro-tts/homegrown/marsha.pt", 
   "language": "English, Mandarin Chinese",
   "traits": "Woman, mix of xiaoyi and bella"
 },
 "sasha": {
   "file": "/media/kokoro-tts/homegrown/sasha.pt", 
   "language": "English",
   "traits": "Woman, mix of lily and bella"
 }
}
```
Some notes on the fields:
* the key name - can be anything you want 
* kokoro_name - this is the name _as used by Kokoro officially_. This is stuff like af_heart, am_adam, bf_lily, etc. If this is missing or an empty string, it is assumed this entry is a custom voice with an associated .pt file.
* file - If this exists and kokoro_name is missing or the empty string, it is assumed that this is a custom voice; file indicates the path and filename to your custom .pt file.
* language - Describes the language. This can be anything you want.  
* traits - a description of the voice. 


There is also a phonetic replacement file, where you can replace 'natalia' with 'nahtahlia' etc etc in the text - good for words that the TTS just cant get right phonetically and need that extra help. This is the 'phonetic-replacement-file' argument, and the JSON file is 
structured as so: 
{
  "phonetic_replacements": {
    "Natalia": "Nahtahlia",
    "Maria": "Mahria",
    "Joaquin": "Waah-keen",
    "Sean": "Shawn",
    "Wednesday": "Wensdee"
  }
}

'phonetic_replacements' is important, but everything else is arbitrary; the basic format is "word or phrase to be replaced": "will be replaced with"

You can also have different simultaneous voices, too you need to set 'use_different_speakers' to true. Once you do this you have two options:
* Setting a narrator: If you make a sidebar in your sentences encased in asterisks like this- *Takes a pause for a dramatic effect* - anything between the *'s will be spoken by the narrator voice. You can set this with 'narrator_voice', and it should match a key in the voice library.
* Different speakers: this library keeps a voice mapping for character voices - if you have a JSON file structured like this, and you store the location of that JSON file to 'voice_mapping_file':
    {
     "Lily": {
         "tts_name": "bf_lily"
     },
     "George": {
         "tts_name": "george"
     }
    }

Note that the 'tts_name' is however its defined as a key in the voice library - it will be converted to what Kokoro actually uses after it loads. 
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
# This has to be set before CUDA is initialised, hence before torch (or kokoro, which imports
# torch itself) is imported rather than in the constructor. setdefault is used so that an
# explicit setting in the environment still wins.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
from pathlib import Path
from kokoro import KPipeline
import soundfile as sf
import logging
import json
from typing import Dict, Any
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

class AmadeoKokoro:

    HOST = 'localhost'
    PORT = 8888
    VOICES = 'voice_samples'
    REPO_ID = 'hexgrad/Kokoro-82M'
    MODEL_PATH = ''
    LANGUAGE_CODE = 'a'
    DEFAULT_VOICE = 'am_santa'
    SAMPLE_RATE = 24000  # Use common default sample rate for Kokoro
    PHONETIC_REPLACEMENT_FILE = ''
    USE_DIFFERENT_SPEAKERS = False
    NARRATOR_VOICE = ''
    VOICE_MAPPING_FILE = ''
    PAUSE_DURATION = .4
    SEGMENT_SPACER_DURATION = .3
    GPU_INDEX = 0

    def __init__(self, config_dict:Dict):

        self.config = config_dict

        self.repo_id = self.config['repo_id']
        self.model_path = self.config['model_path']

        self.pause_duration = self.config['pause_duration']
        self.segment_spacer_duration = self.config['segment_spacer_duration']

        self.voice_samples_dir = self.config['voices']

        # ---------------------------------------------------------------------------- GPU selection
        # Which physical GPU to run on. Absent from the dictionary (i.e. when this class is used
        # outside of the server script) it falls back to the first GPU, which is the only index
        # guaranteed to exist on a CUDA machine.
        self.gpu_index = self.config.get('gpu', AmadeoKokoro.GPU_INDEX)

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

            logger.info(f"Kokoro: Using GPU {self.gpu_index} ({torch.cuda.get_device_name(self.gpu_index)}) of {device_count} available.")
        else:
            # No CUDA at all - the GPU index is meaningless, so note that it is being ignored
            # instead of letting the caller assume it took effect.
            if self.gpu_index != AmadeoKokoro.GPU_INDEX:
                logger.warning(f"A GPU index ({self.gpu_index}) was requested but no CUDA device is available; falling back to CPU and ignoring the index.")

            # Reset the index so the rest of the class does not have to keep asking whether it is
            # meaningful; on CPU it is simply unused.
            self.gpu_index = AmadeoKokoro.GPU_INDEX
            self.device = "cpu"

        # Convenience flag - self.device is 'cuda:N' rather than a bare 'cuda', so string
        # comparisons against "cuda" elsewhere would not hold.
        self.use_cuda = self.device.startswith("cuda")

        # ---------------------------------------------------------------------------- Locks
        # AmadeoServer hands every client connection to its own thread, and its docstring is
        # explicit that anything sharing a resource such as a GPU has to do its own locking. The
        # Kokoro pipeline below is loaded once and shared by all of those threads, so:
        #
        #  * gpu_lock serialises the actual inference calls. It is taken per request rather than
        #    per connection - a client holding a session open is not a reason to keep the GPU to
        #    itself. Waiting is a plain block rather than a 'GPU busy' rejection: a handful of
        #    household clients will not queue up long enough for that to be worth the extra
        #    protocol.
        #  * voice_bank_lock guards the pipeline's own voice dictionary (tts_model.voices), which
        #    get_voice_info writes custom .pt voices into on first use. Two clients asking for the
        #    same uncached custom voice at once would otherwise both torch.load it and race to
        #    insert it.
        #  * dialogue_lock guards the one-time voice confirmation pass, which mutates the shared
        #    dialogue_helper. That is configuration lookup, so it gets its own lock rather than
        #    borrowing the GPU one - two threads arriving together would otherwise race on the
        #    same dictionary.
        #
        # Lock ordering, to keep this deadlock free: dialogue_lock -> voice_bank_lock. Nothing
        # ever takes them the other way round, and get_voice_info (which takes voice_bank_lock)
        # never reaches back for dialogue_lock.
        self.gpu_lock = threading.Lock()
        self.voice_bank_lock = threading.Lock()
        self.dialogue_lock = threading.Lock()

        # Determine if the phonetic replacement file exists - if it does, set the phonetic_replacement for replacements
        self.phonetic_replacement = None
        if self.config.get('phonetic_replacement_file', ''):
            self.phonetic_replacement = PhoneticReplacement(self.config['phonetic_replacement_file'])

        self.dialogue_helper = SplitDialogue(self.config.get('use_different_speakers', False), self.config.get('narrator_voice', AmadeoKokoro.NARRATOR_VOICE), self.config.get('voice_mapping_file', AmadeoKokoro.VOICE_MAPPING_FILE))


        # Load voice configuration from voices.json file
        # This maps voice names to audio files and their transcriptions
        self.voice_config = self.load_voice_config()

        # Initialize the Kokoro model
        # IMPORTANT: Model is loaded ONCE here and reused for all requests
        # This provides significant performance benefits for successive calls
        try:
            # KPipeline gained a 'device' keyword after its first releases, so it is only passed if
            # the installed version actually accepts it. On a build without it, the
            # torch.cuda.set_device call above still steers the load, because Kokoro falls back to
            # a bare 'cuda' and torch resolves that to the calling thread's current device.
            if 'device' in inspect.signature(KPipeline.__init__).parameters:
                device_kwargs = {'device': self.device}
            else:
                device_kwargs = {}
                logger.warning(f"This build of Kokoro does not accept a 'device' argument; relying on the thread's current CUDA device ({self.device}) instead. Upgrade kokoro if the model ends up on the wrong GPU.")

            logger.info(f"Loading Kokoro model {self.config['repo_id']} on {self.device}... (this may take a moment)")

            try:
                self.tts_model = KPipeline(lang_code=self.config['language_code'], repo_id=self.config['repo_id'], **device_kwargs)
            except (ValueError, AssertionError, RuntimeError) as e:
                # Some Kokoro builds validate the device string against a short list of bare names
                # ('cpu', 'cuda', 'mps') and reject an indexed one such as 'cuda:1'. That is a
                # rejection of the spelling, not of the card, so fall back to letting Kokoro pick
                # and then move the loaded model onto the requested GPU by hand.
                if not device_kwargs:
                    raise

                logger.warning(f"Kokoro rejected the device '{self.device}' ({e}); loading without it and moving the model onto that GPU afterwards.")
                self.tts_model = KPipeline(lang_code=self.config['language_code'], repo_id=self.config['repo_id'])

                # getattr rather than a direct attribute read - a build old enough to reject the
                # keyword may not expose '.model' either, and losing the original error to an
                # AttributeError here would be actively misleading.
                loaded_model = getattr(self.tts_model, 'model', None)
                if loaded_model is not None:
                    self.tts_model.model = loaded_model.to(self.device)

            # Confirm where the weights actually landed rather than assuming the keyword took
            # effect - on a multi GPU box the difference between 'asked for card 1' and 'running on
            # card 1' is otherwise invisible until someone watches nvidia-smi. Anything unexpected
            # here is worth a warning but not worth refusing to start over.
            try:
                loaded_device = next(self.tts_model.model.parameters()).device

                if self.use_cuda and (loaded_device.type != 'cuda' or loaded_device.index != self.gpu_index):
                    logger.warning(f"Kokoro was asked for {self.device} but its weights are on {loaded_device}. Generation will still work, but it is not using the card that was requested.")
                else:
                    logger.info(f"Kokoro model weights confirmed on {loaded_device}.")
            except (AttributeError, StopIteration):
                # A build that does not expose '.model', or exposes one with no parameters - not
                # fatal, it just means this check cannot be made.
                logger.debug("Could not inspect the Kokoro model's device; skipping the placement check.")

            logger.info("Kokoro model loaded successfully - ready for requests")

        except Exception as e:
            # Other errors loading the model (GPU issues, model files missing, etc.)
            logger.error(f"Error loading Kokoro: {e}")
            raise


    def load_voice_config(self):
        """
        Load voice configuration from voices.json file

        This is loaded from the 'voice library' mentioned in the description of this class - its the 'voices' parameter. Whatever JSON is in that file gets loaded here.

        Returns:
            dict: Voice configuration mapping voice names to their name according to Kokoro plus a few other fields
        Raises:
            Exception: If there is a problem getting the json file or it does not exist
        """
        config_file = Path(self.voice_samples_dir)

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
                logger.warning(f"Error loading voice config: {e}. The Kokoro simple server can still be used, but you will have to enter the default names that Kokoro gives the voices - and you will not be able to sue custom voices.")
        else:
            logger.warning(f"No JSON file given to explain the voices. The Kokoro simple server can still be used, but you will have to enter the default names that Kokoro gives the voices - and you will not be able to sue custom voices.")


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
        Resolve every mapped speaker name - and the narrator - to a voice Kokoro actually knows,
        once, on the first request that needs it.

        This is only meaningful when 'use_different_speakers' is set. It rewrites the shared
        dialogue_helper in place, which is why it is guarded: with a threaded server, two clients
        arriving at the same time on a cold server would otherwise walk the same dictionary while
        the other one was rewriting it. The flags are re-checked inside the lock so that the
        second thread through does not redo the work the first one just finished.

        This is configuration lookup rather than inference, so it takes its own lock rather than
        the GPU lock, and callers should run it before claiming the GPU. It can still reach the
        GPU indirectly - resolving a custom voice makes get_voice_info load that voice's .pt onto
        the card - but that is a small tensor copy guarded by voice_bank_lock, not a model call,
        so it is safe to run while another thread is mid generation.

        Returns:
            None
        """
        with self.dialogue_lock:
            if not self.dialogue_helper.voice_mapping_list_confirmed:
                # go through each name in the mapped list to get what kokoro calls this voice
                for name in self.dialogue_helper.voice_mapping.keys():
                    # get the dictionary based on the passed name, and ensure its in there (if not, use the default) it usually will be the same, if this was set up properly, but on the off chance it wasn't....
                    one_character = self.dialogue_helper.voice_mapping[name]
                    one_character['tts_name'] = self.get_voice_info(one_character['tts_name'])
                self.dialogue_helper.voice_mapping_list_confirmed = True

            # settle the narrator - either confirm the voice OR set to default if the narrator voice is missing or is not in the voice listing
            if self.dialogue_helper.narrator and not self.dialogue_helper.narrator_confirmed:
                self.dialogue_helper.narrator = self.get_voice_info(self.dialogue_helper.narrator)
                self.dialogue_helper.narrator_confirmed = True


    def get_voice_info(self, voice_name: str) -> str:
        """
        Lookup the name in the config file - if its an official Kokoro name, the official name will be returned in a tuple ('official Kokoro Name', ''); if this is a custom voice file, the custom name and the path to the .pt file is determined,
        then it is loaded to the Kokoro's model; if the name is a base Kokoro voice, or if the custom model was already loaded, this passes back the actual name of the voice as used by Kokoro.

        If this is a custom file but the file doesnt exist, it falls back to a default Kokoro voice.

        Args:
            voice_name: Name of the voice to look up (e.g., "adam", which is an alias for "am_adam")

        Returns:
            typle str: voice_name, as understood by Kokoro

        Raises:

        """

        # ensure key is lower case
        voice_name = voice_name.lower()

        voice_file = ''
        # Check if requested voice exists in configuration
        if not self.voice_config or (len(self.voice_config) == 0):
            logger.warning(f"Voice config file did not load properly - will try using '{voice_name}' directly as a name in Kokoro. No promises it will work.")
        elif voice_name not in self.voice_config:
            logger.warning(f"Voice '{voice_name}' not found in the JSON config - will try using it directly as a name in Kokoro. No promises it will work.")
        else:
            #voice does exist in config - now see if its custom or not

            single_voice = self.voice_config[voice_name]

            if single_voice.get('kokoro_name', '').strip():
                # if the name exists and is not the empty string, this is an alias for a Kokoro name - so get the official name
                voice_name = single_voice.get('kokoro_name', '').strip()

            elif single_voice.get('file', '').strip():
                # If the 'file' field exists, this is not a built-in Kokoro voice - this is a custom voice in a .pt file. Load it!
                voice_file_path = Path(single_voice.get('file', '').strip())

                # Check if the actual audio file exists on disk
                if voice_file_path.exists():
                    # convert to a string
                    voice_file = str(voice_file_path)
                else:
                    logger.warning(f"Voice '{voice_name}' is a custom voice, but its .pt file {voice_file} is corrupt or missing - you are getting the default voice instead {AmadeoKokoro.DEFAULT_VOICE}.")
                    voice_name = AmadeoKokoro.DEFAULT_VOICE
                    voice_file = ''

        if voice_file:
            # The voice bank (tts_model.voices) is shared by every client thread, so the
            # check-then-insert below has to be atomic - without the lock two clients asking for
            # the same uncached custom voice at the same time would both see it missing, both
            # torch.load it, and both write it.
            with self.voice_bank_lock:
                if voice_name not in self.tts_model.voices:
                    logger.info(f"Retrieving speech using custom voice '{voice_name}' from file '{voice_file}' - saving to model's voice bank.")

                    # map_location matters on a multi GPU box: a .pt saved from a machine that was
                    # using cuda:0 would otherwise be restored straight onto cuda:0 no matter which
                    # card was asked for, which either wastes memory on the wrong GPU or trips a
                    # cross-device error when Kokoro pairs it with the model's weights.
                    voice_data = torch.load(voice_file, map_location=self.device)

                    self.tts_model.voices[voice_name] = voice_data
                else:
                    logger.info(f"Retrieving speech using custom voice '{voice_name}' - retrieving from voice bank, as it was previously cached.")
        else:
            logger.info(f"Retrieving speech using built-in voice '{voice_name}'.")

        return voice_name


    def generate_speech(self, text, voice_name):
        """
        Generate speech audio using Kokoro

        This is the core function that takes text and a voice name, and then uses the pre-loaded
        Kokoro model to generate speech.

        The Kokoro model is already loaded in memory (from __init__), so this function is fast on successive calls.

        Args:
            text: Text to convert to speech
            voice_name: Name of voice to use

        Returns:
            raw_data: PCM data in a WAV container; from here, we can directly send this over a network connection OR simply save to a file

        Raises:
            Exception: If speech generation fails for any reason

        """

        # ------------------------------------------------------------------ CPU only work
        # Nothing here runs the model, so it is deliberately done before the lock is taken -
        # rewriting one client's text while another client is mid generation costs that client
        # nothing.

        # Verify all speakers first, if it has not been done
        if self.dialogue_helper.use_different_speakers:
            self._confirm_dialogue_voices()

        # if the phonetic replacement object exists, replace the words in the text with what they sound like phonetically
        if self.phonetic_replacement:
            text = self.phonetic_replacement.phonetic_replacement(text)

        try:
            # Resolve the requested voice to the name Kokoro knows it by, loading it into the
            # shared voice bank first if it is a custom .pt. This is left outside the GPU lock
            # because it is a dictionary lookup plus at most one small tensor load, and it has a
            # lock of its own.
            voice_name = self.get_voice_info(voice_name)

            raw_data = None

            # ------------------------------------------------------------------ GPU only work
            # The lock is held for the Kokoro calls and nothing else. The pipeline is shared
            # across every client thread and is not safe to call concurrently.
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
                        # if we are not using multiple speakers
                        final_tensor = self.single_pass(text, voice_name)

                        # Convert to torch tensor immediately:
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
                                pause_samples = int(AmadeoKokoro.SAMPLE_RATE * self.pause_duration)
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
                                    # either the narrator spoke in the middle of a line - in which ase, we are using the previous name - or there is no name, so use the master name given
                                    this_voice = voice_name

                                # Get raw audio tensor instead of WAV bytes
                                segment_tensor = self.single_pass(content, this_voice)  # Returns tensor

                                # Convert to torch tensor immediately:
                                if isinstance(segment_tensor, np.ndarray):
                                    segment_tensor = torch.from_numpy(segment_tensor).float()

                            audio_segments.append(segment_tensor)

                        if audio_segments:
                            # Add pauses between segments (all as tensors)
                            pause_duration = self.segment_spacer_duration
                            pause_samples = int(AmadeoKokoro.SAMPLE_RATE * pause_duration)
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

            logger.info(f"Successfully generated {final_tensor.shape[1]/AmadeoKokoro.SAMPLE_RATE:.1f}s of audio at {AmadeoKokoro.SAMPLE_RATE}Hz")

            # Convert to PCM data in a WAV container (from here, we can directly send this over a network connection OR simply save to a file)
            raw_data = self.convert_to_wav(final_tensor, AmadeoKokoro.SAMPLE_RATE)

            return raw_data

        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            # Re-raise the exception so the caller can handle it appropriately
            raise


    def single_pass(self, text:str, voice_name: str) -> np.ndarray:
        """
        This makes a 'single pass' at Kokoro and returns the raw tensor data. this is done like this because there is a chance the speech is broken up by different speakers and the narrator, so this must be called multiple times to properly
        build the audio.

        Args:
            text: The text to be turned to speech.
            voice_name: The voice name, as understood by Kokoro

        This runs the shared Kokoro pipeline, so callers must already hold gpu_lock and must have
        pinned their thread with _bind_thread_to_gpu - it does neither itself, because a
        multi-speaker request calls it once per segment and should hold the lock across the lot.

        Returns:
            np.ndarray: Raw audio tensor data

        Raises:
            Exception: If Kokoro fails, or if it produced no audio at all
        """
        all_audio = []

        try:
            # Call Kokoro to generate speech using the pre-loaded model
            generator = self.tts_model(text, voice=voice_name)
            for i, (gs, ps, audio) in enumerate(generator):
                # Kokoro hands back a torch tensor, and on some builds that tensor is still on the
                # GPU. np.concatenate cannot touch CUDA memory, so each chunk is brought back to
                # the host here - this also keeps everything downstream (torch.cat with the CPU
                # silence tensors, then soundfile) on one device.
                if torch.is_tensor(audio):
                    audio = audio.detach().cpu().numpy()

                all_audio.append(audio)
        except Exception as e:
            # Re-raise rather than falling through to the concatenate below: swallowing this used
            # to turn a real Kokoro error into a confusing 'need at least one array to
            # concatenate', which hid what actually went wrong.
            logger.error(f"ERROR - {e}")
            raise

        if not all_audio:
            raise RuntimeError(f"Kokoro returned no audio for voice '{voice_name}' - the text may have been empty or unpronounceable after phonetic replacement.")

        return np.concatenate(all_audio)


    def convert_to_wav(self, local_data, local_sample_rate):
        """
        Converts the data directly from the Kokoro output to PCM data in WAV container format without saving to disk

        Args:
            local_data: (either class 'numpy.ndarray'> or 'torch.Tensor') The output data from Kokoro
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
        tasks related to Kokoro). This method (and other methods in other classes that implement this) expects a dictionary and data (bytes, which can represent all kinds of media files), although the data portion of that may not be used (depending on the case; in Kokoro's case, this is not used).
        This should return a dictionary (that will be turned into JSON) and byte data (if applicable, and in our case it is - its the PCM audio data in a WAV container).

        To see what is expected of the basics of what is expected fpr the server, see the main description for 'amadeo_server.AmadeoServer', although there are some additional ones specific to Kokoro:
        * 'command' - either 'service_tts' or 'show_voices', depending on what is to be done
        * if 'command' is 'service_tts', we also need 'voice' (the voice you want to use in the audio) as well as 'text' (the text you want the audio to read).

        To see what dictionary fields will be sent to the client. see the main description for 'amadeo_server.AmadeoServer'

        Args:
            request: A dictionary that will contain fields. It should ALWAYS contain 'command', which represents WHAT the user wants to do. That will determine one of several scenarios:
                    Scenario 1: generating speech from text
                        'command' = 'service_tts'
                        'voice' (representing an available 'voice' supported in this Kokoro instance)
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
                # Use Kokoro to generate audio file (this uses the pre-loaded model)
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
            return_info = ''
            for key in self.voice_config.keys():
                return_info += f"\n{key}"
                if self.voice_config[key].get('language',''): return_info += f" - Language: {self.voice_config[key]['language']}"
                if self.voice_config[key].get('traits',''): return_info += f" - Traits: {self.voice_config[key]['traits']}"
            return { 'success': True, 'type': 'voices', 'message': return_info, 'file_size': 0}, None
        else:
            return_dict = { 'success': False, 'type': 'error', 'message': f"No 'command' provided for Kokoro service; available commands ({possible_commands}).", 'file_size': 0}
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
            'repo_id': str,
            'model_path': str,
            'language_code': str,
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
        parser = argparse.ArgumentParser(description='Kokoro Server - Voice cloning text-to-speech server',formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('--host', default=AmadeoKokoro.HOST, help='Server host address (default: localhost)')
        parser.add_argument('--port', type=int, default=AmadeoKokoro.PORT, help='Server port number (default: 8888)')
        parser.add_argument('--voices', default=AmadeoKokoro.VOICES, help='The file that houses the identities of the various voices for this model.')
        parser.add_argument('--language-code', default=AmadeoKokoro.LANGUAGE_CODE, help='The language code; a = American English, British English = b, Spanish = e, French = f, Italian = i, Brazilian Portuguese = p, Hindi = h.')
        parser.add_argument("--repo-id", default=AmadeoKokoro.REPO_ID,help="The repo ID - the vast majority of people use 'hexgrad/Kokoro-82M'.")
        parser.add_argument("--phonetic-replacement-file", default=AmadeoKokoro.PHONETIC_REPLACEMENT_FILE,help="The path/filename of the JSON file you will use for phonetic replacements (i.e. 'mr.' to 'mister'). Not required.")

        parser.add_argument("--pause-duration", type=float, default=AmadeoKokoro.PAUSE_DURATION,help="If a pause is indicated in the text (via the string: **), a pause of this length (in seconds, fractional) is inserted. This is only valid if 'use-different-speakers' is set. Note: if you segment a very short sentence with this (where one spoken part is 1-3 words), the model may truncate the speech.")
        parser.add_argument("--segment-spacer-duration", type=float, default=AmadeoKokoro.SEGMENT_SPACER_DURATION,help="If 'use-different-speakers' is set, this will add a segment of silence between speakers that equals this length in seconds (fractional).")

        parser.add_argument("--narrator-voice", default=AmadeoKokoro.NARRATOR_VOICE,help="The voice for the narrator; leave blank if you do not want to use this. The narrator voice MUST be in the file identified by the 'voices' parameter.")
        parser.add_argument("--use-different-speakers", type=bool, default=AmadeoKokoro.USE_DIFFERENT_SPEAKERS,help="Select this if you wish to use different speakers if they are identified with colons in the text (the narrator reads things between asterisks)")
        parser.add_argument("--voice-mapping-file", default=AmadeoKokoro.VOICE_MAPPING_FILE,help="This is a JSON file that maps a name that can appear in a script-like fashion to a voice in 'voices' for example, It can map 'am_adam' to the voice if it sees 'Adam:' in the text.")

        parser.add_argument("--model_path", default=AmadeoKokoro.MODEL_PATH,help="The path to a custom model, if you find one on huggingface....")
        parser.add_argument("--gpu", type=int, default=AmadeoKokoro.GPU_INDEX,help="The index of the CUDA GPU to load the model onto, matching the order 'nvidia-smi -L' reports (0 for the first GPU, 1 for the second, and so on); the ordering is pinned to the physical slot order via CUDA_DEVICE_ORDER, so it does not depend on which card CUDA considers fastest. Defaults to the first GPU. Ignored if no CUDA device is available.")
        parser.add_argument("--json", type=str, default="", help="If this points to a valid JSON file, the ENTIRE parameter settings are pulled from that file, and the defaults - and other arguments passed from the command line - are ignored. If the JSON load fails for whatever reason, though, the defaults WILL be engaged. Just remember that if there is a dash in the arg name, its going to be an underscore in the JSON.")

        argDict = {}

        try:
            args = parser.parse_args()
            use_default_arg_config = True  # This is only flipped if we successfully load from a JSON file

            json_config_file = args.json

            if json_config_file and os.path.exists(json_config_file):
                try:
                    config_dict = AmadeoKokoro.load_json_config(json_config_file)

                    argDict['voices'] = config_dict['voices']
                    argDict['host'] = config_dict.get('host', AmadeoKokoro.HOST)
                    argDict['port'] = config_dict.get('port', AmadeoKokoro.PORT)
                    argDict['language_code'] = config_dict.get('language_code', AmadeoKokoro.LANGUAGE_CODE)
                    argDict['repo_id'] = config_dict.get('repo_id', AmadeoKokoro.REPO_ID)
                    argDict['phonetic_replacement_file'] = config_dict.get('phonetic_replacement_file', AmadeoKokoro.PHONETIC_REPLACEMENT_FILE)
                    argDict['voice_mapping_file'] = config_dict.get('voice_mapping_file', AmadeoKokoro.VOICE_MAPPING_FILE)
                    argDict['narrator_voice'] = config_dict.get('narrator_voice', AmadeoKokoro.NARRATOR_VOICE)
                    argDict['use_different_speakers'] = config_dict.get('use_different_speakers', AmadeoKokoro.USE_DIFFERENT_SPEAKERS)
                    argDict['model_path'] = config_dict.get('model_path', AmadeoKokoro.MODEL_PATH)
                    argDict['pause_duration'] = config_dict.get('pause_duration', AmadeoKokoro.PAUSE_DURATION)
                    argDict['segment_spacer_duration'] = config_dict.get('segment_spacer_duration', AmadeoKokoro.SEGMENT_SPACER_DURATION)
                    argDict['gpu'] = config_dict.get('gpu', AmadeoKokoro.GPU_INDEX)

                    logger.info(f"Config loaded from JSON {json_config_file}.")

                    use_default_arg_config = False

                except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Could not load JSON config [{json_config_file}] - there are errors. Will attempt to load other defaults or args. Error: {e}.")

            elif json_config_file:
                logger.warning(f"Could not load JSON config [{json_config_file}] - file does not exist. Loading from defaults or other parameters sent.")

            if use_default_arg_config:

                argDict['voices'] = args.voices
                argDict['host'] = args.host
                argDict['port'] = args.port
                argDict['language_code'] = args.language_code
                argDict['repo_id'] = args.repo_id
                argDict['phonetic_replacement_file'] = args.phonetic_replacement_file
                argDict['voice_mapping_file'] = args.voice_mapping_file

                argDict['use_different_speakers'] = args.use_different_speakers
                argDict['narrator_voice'] = args.narrator_voice

                argDict['pause_duration'] = args.pause_duration
                argDict['segment_spacer_duration'] = args.segment_spacer_duration

                argDict['model_path'] = args.model_path
                argDict['gpu'] = args.gpu

                logger.info(f"Config loaded from args / defaults.")

            if not os.path.exists(argDict['voices']):
                logger.error(f"{argDict['voices']} does not exist - subsequently, Kokoro will not be able to load voices, so this is dead in the water.")

            if argDict['model_path'] and not os.path.exists(argDict['model_path']):
                # Note: this reports the repo, not a 'model' key - Kokoro identifies its model by
                # repo_id, and there is no separate model name to fall back to.
                logger.warning(f"{argDict['model_path']} does not exist - reverting to the model from repo {argDict['repo_id']}.")
                argDict['model_path'] = ''

        except SystemExit as e:
            argDict = {}
            if e.code == 0:
                # --help was used, so print no error
                print(f"Thank you!")
            else:
                logger.error(f"Invalid arguments.")

        return argDict
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

import torch
from pathlib import Path
from f5_tts.api import F5TTS
import soundfile as sf
import logging
import json
from typing import Dict, Tuple, Any
import io
import os
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


    def __init__(self, config_dict:Dict):

        self.config = config_dict

        self.model = self.config['model']
        self.model_path = self.config['model_path']

        self.cross_fade_duration = self.config['cross_fade_duration']

        self.pause_duration = self.config['pause_duration']
        self.segment_spacer_duration = self.config['segment_spacer_duration']

        self.voice_samples_dir = self.config['voices']

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

            if self.model_path:
                logger.info(f"Loading Custom F5-TTS model {self.model_path}... (this may take a moment)")
                self.tts_model = F5TTS(ckpt_file=self.model_path)  # Load the pre-trained F5-TTS model into memory
            elif self.model:
                logger.info(f"Loading F5-TTS model {self.model}... (this may take a moment)")
                self.tts_model = F5TTS(model=self.model)  # Load the pre-trained F5-TTS model into memory
            else:
                logger.info("Loading F5-TTS model... (this may take a moment)")
                self.tts_model = F5TTS()  # Load the pre-trained F5-TTS model into memory
            logger.info("F5-TTS model loaded successfully - ready for requests")
        except Exception as e:
            # Other errors loading the model (GPU issues, model files missing, etc.)
            logger.error(f"Error loading F5-TTS: {e}")
            raise


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

        # Verify all speakers first, if it has not been done
        if self.dialogue_helper.use_different_speakers and not self.dialogue_helper.voice_mapping_list_confirmed:
            # go through each name in the mapped list to get what kokoro calls this voice
            for name in self.dialogue_helper.voice_mapping.keys():
                # get the dictionary based on the passed name, and ensure its in there (if not, use the default) it usually will be the same, if this was set up properly, but on the off chance it wasn't....
                one_character = self.dialogue_helper.voice_mapping[name]
                one_character['tts_name'] = self.get_voice_info(one_character['tts_name'])[0] #get just the first item in the tuple returned (we do not care about the rest)
            self.dialogue_helper.voice_mapping_list_confirmed = True

        # if the phonetic replacement object exists, replace the words in the text with what they sound like phonetically
        if self.phonetic_replacement:
            text = self.phonetic_replacement.phonetic_replacement(text)

        #initialize - hopefully this is overwritten, but if not, this is a good guess
        sample_rate = AmadeoF5.SAMPLE_RATE

        try:

            raw_data = None
            if not self.dialogue_helper.use_different_speakers:
                # if we are not using multiple speakers, just run a single pass through F5-TTS
                final_tensor, sample_rate = self.single_pass(text, voice_name)

                # Handle numpy array vs torch tensor
                # Different F5-TTS versions return different data types, but the version as of August 2025 is indeed a NumPy array
                if isinstance(final_tensor, np.ndarray):
                    final_tensor = torch.from_numpy(final_tensor).float()
            else:
                # if we ARE using multiple speakers

                # settle the narrator - either confirm the voice OR set to default if the narrator voice is missing or is not in the voice listing
                if self.dialogue_helper.use_different_speakers and self.dialogue_helper.narrator and not self.dialogue_helper.narrator_confirmed:
                    # if we want to use different speakers, there is an alleged narrator, and the narrator has not been confirmed
                    self.dialogue_helper.narrator = self.get_voice_info(self.dialogue_helper.narrator)[0]
                    self.dialogue_helper.narrator_confirmed = True

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
                    logger.error("No audio segments returned for multiple speakers...")
            # END - multiple speakers


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
            'segment_spacer_duration': float
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
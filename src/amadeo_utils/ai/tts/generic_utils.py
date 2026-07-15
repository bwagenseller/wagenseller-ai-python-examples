from typing import Dict, List, Tuple, Optional
import json
import logging
import re

# Configure logging to show timestamps and log levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

class PhoneticReplacement:

    def __init__(self, json_file: str):

        self.replacements = self.get_phonetic_replacement_dict(json_file)

    def get_phonetic_replacement_dict(self, json_file: str) -> Dict:
        """
        Load name replacements from JSON file

        Args:
            json_file: The full path and name of the JSON file that has the data. The file should be in the format:
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
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"Phonetic replacement file {json_file} loaded into dictionary.")
            return data.get("phonetic_replacements", {})
        except FileNotFoundError:
            logger.error(f"{json_file} not found, no phonetic replacements will be applied.")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error reading {json_file}, no phonetic replacements will be applied.")
            return {}

    def phonetic_replacement(self, text: str) -> str:
        """
        Replace text if it matches what is in our replacements dictionary

        Args:
            text: The text that is to be searched for replacements.

        Returns:
            str: The text, with replacements
        """
        for original, fixed in self.replacements.items():
            # Check if original contains only word characters (letters, numbers, underscore)
            if re.match(r'^\w+$', original):
                # Use word boundaries for whole words only
                pattern = rf'\b{re.escape(original)}\b'
                text = re.sub(pattern, fixed, text, flags=re.IGNORECASE)
            else:
                # For punctuation or mixed content, do simple replacement
                # Case-sensitive for punctuation (! should not match I)
                if original.isalpha():
                    # If it's letters but has non-word chars, still do case-insensitive
                    text = re.sub(re.escape(original), fixed, text, flags=re.IGNORECASE)
                else:
                    # For punctuation, do exact match (case-sensitive)
                    text = text.replace(original, fixed)

        return text

############################################################################################### SplitDialogue ###############################################################################################

class SplitDialogue:
    """
    This class is helpful if the text that is being fed to this is from a LLM and you want different voices for different characters - or a narrator. For example, split_narrator_dialogue_with_names will return:

        Example 1: "Ya know, I am not so sure I like that idea *she fidgets nervously* But I will think about it."

        Output (Tuples returned):
            dialogue        Ya know, I am not so sure I like that idea      None
            narrator        she fidgets nervously                           None
            dialogue        But I will think about it.                      None

        Example 2: "Lily: Ya know, I am not so sure I like that idea *she fidgets nervously* But I will think about it.
        George: I wouldn't worry about it, Lily."

        Output (Tuples returned):
            dialogue        Ya know, I am not so sure I like that idea      Lily
            narrator        she fidgets nervously                           None
            dialogue        But I will think about it.                      Lily
            dialogue        I wouldn't worry about it, Lily.                George


    Note that for output 2, 'George:' is on a newline - this is critical.
    If there is no name with a colon, split_narrator_dialogue_with_names will properly name who said what. If there is no colon, only one speaker is assumed. Words between asterisks are considered to be spoken by
    a narrator.

    In addition, this class keeps a voice mapping for character voices - if you have a JSON file structured like this:
    {
     "Default": {
         "tts_name": "bf_lily"
     },
     "Lily": {
         "tts_name": "bf_lily"
     },
     "George": {
         "tts_name": "george"
     }
    }

    Note that the 'tts_name' is however its defined in your mapping; if you are using this library's 'Kokoro' class, it will be the key of the 'voices' JSON for Kokoro (my examples here use what I have in my 'voices' JSON).
    Also note the 'Default' - this is critical, as if a name does not exist, it will bomb out - UNLESS the Default is defined.

    """
    def __init__(self, use_different_speakers: bool, narrator_voice: str, voice_mapping_file: str):

        self.narrator = narrator_voice
        self.use_different_speakers = use_different_speakers
        self.narrator_confirmed = False
        self.voice_mapping_list_confirmed = False

        if voice_mapping_file:
            self.voice_mapping = self.load_speakers_config(voice_mapping_file)
        else:
            self.voice_mapping = {}

    @staticmethod
    def split_narrator_dialogue_with_names(text: str) -> List[Tuple[str, str, Optional[str]]]:
        """
        Split text into narrator and dialogue parts, detecting character names (if stated like so: 'Mary:'). If there is a double asterisk (i.e. **), it will indicate a forced pause; if its
        text within asterisks (i.e. *She turned to face him*), it would be counted as a narrator.

        Example 1: "Ya know, I am not so sure I like that idea *she fidgets nervously* But I will think about it."
        Output (Tuples returned):
            dialogue        Ya know, I am not so sure I like that idea      None
            narrator        she fidgets nervously                           None
            dialogue        But I will think about it.                      None

        Example 2: "Mary: Ya know, I am not so sure I like that idea *she fidgets nervously* But I will think about it.
                        Bob: I wouldn't worry about it, Mary."
        Output (Tuples returned):
            dialogue        Ya know, I am not so sure I like that idea      Mary
            narrator        she fidgets nervously                           None
            dialogue        But I will think about it.                      Mary
            dialogue        I wouldn't worry about it, Mary.                Bob

        Example 3: "Mary: Ya know, ** I am not so sure I like that idea *she fidgets nervously* But I will think about it.
                        Bob: I wouldn't worry about it, Mary."
        Output (Tuples returned):
            dialogue        Ya know,                                        Mary
            pause                                                           None
            dialogue        I am not so sure I like that idea               Mary
            narrator        she fidgets nervously                           None
            dialogue        But I will think about it.                      Mary
            dialogue        I wouldn't worry about it, Mary.                Bob


        Args:
            text: Input text with narrator parts in *asterisks* and optional "Name:" prefixes

        Returns:
            List of tuples: (text_content, speaker_type, character_name)
            - speaker_type: 'dialogue', 'narrator', 'pause'
            - character_name: None for narrator, character name for dialogue (or None if no name)
        """
        parts = []

        # Split text by lines first to handle "Name:" prefixes
        lines = text.split('\n')
        current_character = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line starts with "Name:"
            name_match = re.match(r'^([^:]+):\s*(.*)$', line)
            if name_match:
                current_character = name_match.group(1).strip()
                remaining_text = name_match.group(2).strip()
            else:
                remaining_text = line
                # Keep using the last character name if no new one specified

            # Now split the remaining text by asterisks for narrator parts
            segments = re.split(r'(\*[^*]*\*)', remaining_text)

            for segment in segments:
                segment = segment.strip()

                if not segment:
                    continue

                if segment.startswith('*') and segment.endswith('*'):
                    if segment == '**':
                        # Double asterisk indicates a pause
                        parts.append(('', 'pause', None))
                    else:
                        # Narrator text - remove asterisks
                        narrator_text = segment[1:-1].strip()
                        if narrator_text:
                            parts.append((narrator_text, 'narrator', None))

                else:
                    # Dialogue text
                    if segment:
                        parts.append((segment, 'dialogue', current_character))

        return parts

    def load_speakers_config(self, filename: str) -> Dict:
        """Load speakers configuration from JSON file into a dictionary"""

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                speakers_config = json.load(f)

            # Convert all keys to lowercase
            speakers_config = {k.lower(): v for k, v in speakers_config.items()}

            logger.info(f"Successfully loaded speaker mapping config file {filename}.")
            return speakers_config

        except FileNotFoundError:
            logger.error(f"Speaker mapping config file not found: {filename}.")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in speaker mapping config file: {filename}: {e}.")
            return {}
        except Exception as e:
            logger.error(f"Error reading speaker mapping config file: {filename}: {e}.")
            return {}


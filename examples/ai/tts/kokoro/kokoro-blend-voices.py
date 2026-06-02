import warnings

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
While you cannot directly clone a voice, you _can_ mix existing voices to get novel voices. The trick is you have to load the voices _first_ - which gives you the tensors. Curiously, the default voices are 
_not_ initially loaded - they are loaded when you try to use them. Subsequent uses looks to see if the voice was loaded before - and if it was, it just uses that. Kokoro forces you to make a 'pipeline' 
object; its this object that you load voices into. 

This file loads the default voices, then gives you a chance to mix the default voices in novel ways. The basic premise is you pick a voice and assign it a weight (0-1, but all weights, when summed, should give you 1).
For example, this would give you a blend of xiaoyi (20%), xiaoxiao (20%), tebukuro  (30%), and jessica (30%), creating two files (voice_file.pt and voice_file.wav):
python kokoro-blend-voices.py --file-name "voice_file" --voice-recipe "('zf_xiaoyi', .2), ('zf_xiaoxiao', .2), ('jf_tebukuro', .3), ('af_jessica', .3)"

Or, if you wanted to simply use an interactive menu (saving the output to ~/Downloads):
python kokoro-blend-voices.py --interactive --output-dir "~/Downloads"

This script produces two files:  
* A .pt file - this file can be loaded to Kokoro later for voice generation.
* A .wav file - this is just a sample of the voice. 
"""
import torch
import numpy as np
from kokoro import KPipeline
import soundfile as sf
import argparse
import ast
import os
from pathlib import Path

REPO_ID = 'hexgrad/Kokoro-82M'
LANGUAGE_CODE = 'a'
FILE_NAME = ''
OUTPUT_DIR = os.path.expanduser('~/Downloads')
SAMPLE_RATE = 24000

def load_kokoro_voices(lang_code:str, repo_id:str):
    """
    Load the default Kokoro voices - which also loads their tensors.
    Args:
         lang_code: the language code as defined by Kokoro. a = American English, British English = b, Spanish = e, French = f, Italian = i, Brazilian Portuguese = p, Hindi = h
         repo_id: The repo ID where the voices exist in HuggingFace. The default is hexgrad/Kokoro-82M
    Returns:
        A Tuple - (KPipeline (the Kokoro pipeline), a dictionary of the tensors associated with the working voices)
    """
    print("Loading Kokoro voices...")
    
    # Initialize pipeline
    pipeline = KPipeline(lang_code=lang_code, repo_id=repo_id)
    
    # Common Kokoro voice names - these are the 'default' names in Kokoro (i.e. the ones that are in the default repo)
    common_voices = ['af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah',
        'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa', 'bf_alice', 
    'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis', 'ef_dora', 'em_alex', 'em_santa', 
    'ff_siwis', 'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi', 'if_sara', 'im_nicola', 'jf_alpha', 'jf_gongitsune', 'jf_nezumi', 
    'jf_tebukuro', 'jm_kumo', 'pf_dora', 'pm_alex', 'pm_santa', 'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 
    'zm_yunxi', 'zm_yunxia', 'zm_yunyang']

    print(f"Loading {len(common_voices)} common voice names...")

    voice_data = {}
    for voice_name in common_voices:
        #actually load the voice into Kokoro
        voice_tensor = pipeline.load_voice(voice_name)
        if voice_tensor is not None:
            #save the tensor information
            voice_data[voice_name] = voice_tensor

    return pipeline, voice_data

def replace_prefix(voice: str) -> str:
    """
    Replaces the prefix with the literal meaning, i.e. the contents of the 'codes' dictionary.

    Args:
        voice: the voice to have the prefix replaced
    Returns:
        str: the literal meaning of the voice code + the original name, if it ws in the correct format; otherwise, it will simply return the given voice
    """
    codes = {
        'af': 'American female',
        'am': 'American male',
        'bf': 'British female',
        'bm': 'British male',
        'ef': 'Spanish female',
        'em': 'Spanish male',
        'ff': 'French female',
        'fm': 'French male',
        'hf': 'Hindi female',
        'hm': 'Hindi male',
        'if': 'Italian female',
        'im': 'Italian male',
        'jf': 'Japanese female',
        'jm': 'Japanese male',
        'pf': 'Portuguese female',
        'pm': 'Portuguese male',
        'zf': 'Mandarin female',
        'zm': 'Mandarin male'
    }
    voice_parts = voice.split('_')
    returned_voice=''
    if len(voice_parts) == 1:
        returned_voice=voice_parts[0]
    elif len(voice_parts) == 2:
        returned_voice=codes[voice_parts[0]] + " " + voice_parts[1]
    else:
        returned_voice=voice

    return returned_voice


def create_voice_blend(pipeline, voice_data: dict, voices: list, weights: list, output_name:str = ""):
    """
    Create a voice blend, based on voices previously loaded into Kokoro.

    Args:
        pipeline: The Kokoro pipeline.
        voice_data: A dictionary, with the key being the voices, and the value being the tensor data for that voice
        voices: A list, containing the voices you wish to blend. The voice MUST be present in voice_data; the length of this list MUST equal that of the length of 'weights'.
        weights: A list, containing the weights of the voices. The length MUST equal the length of 'voices', and the weights should, summed, equal 1 (you can do something else as its normalized, but best to keep the sum to 1).
        output_name: The name of the output file. Do not include an extension - this will be used for both a .wav and a .pt file.
    Returns:
         The name of the .pt file generated.
    """
    print(f"Creating voice blend: {voices} with weights {weights}")
    
    if len(voices) != len(weights):
        print("Number of voices must match number of weights")
        return None
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Load voice tensors
    voice_tensors = []
    for voice_name in voices:
        if voice_name not in voice_data:
            print(f"Voice {voice_name} not available")
            return None
        voice_tensors.append(voice_data[voice_name])
    
    # Check all shapes match
    reference_shape = voice_tensors[0].shape
    for i, tensor in enumerate(voice_tensors):
        if tensor.shape != reference_shape:
            print(f"Shape mismatch: {voices[i]} has shape {tensor.shape}, expected {reference_shape}")
            return None
    
    # Create weighted blend - THIS IS THE PART THAT ACTUALLY DOES THE BLENDING
    blended_tensor = torch.zeros_like(voice_tensors[0])
    for tensor, weight in zip(voice_tensors, weights):
        blended_tensor += tensor * weight
    
    
    ## Normalize to keep in reasonable range
    #reference_std = voice_tensors[0].std()
    #if blended.std() > reference_std * 1.8:
    #    blended = (blended - blended.mean()) / blended.std()
    #    blended = blended * reference_std + voice_tensors[0].mean()
    #    print("   Applied normalization")
    
    # Save blend
    filename = f"{output_name}.pt"
    torch.save(blended_tensor, filename)
    print(f"PyTorch Pickle File: {filename}")
    
    # Test the blend
    try:
        test_voice_name = f"{output_name}"
        pipeline.voices[test_voice_name] = blended_tensor

        test_text = f"This is a voice blend of {', '.join(replace_prefix(voice) for voice in voices)}. I hope you like it."
        generator = pipeline(test_text, voice=test_voice_name)
        
        audio_chunks = []
        for _, _, audio in generator:
            if audio is not None:
                audio_chunks.append(audio)
            break
        
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            test_filename = f"{output_name}.wav"
            sf.write(test_filename, full_audio, SAMPLE_RATE)
            print(f"Test audio: {test_filename}")
            
            del pipeline.voices[test_voice_name]
            return filename
        else:
            print("No test audio generated")
            return None
            
    except Exception as e:
        print(f"Test failed: {e}")
        return None

def parse_voice_recipe(recipe_str):
    """
    Parse voice recipe tuples from string and return two lists: voices and weights
    This is used to parse the arguments sent to this script.
    """
    if not recipe_str.strip():
        return [], []

    try:
        # Wrap in list brackets if not already present
        if not recipe_str.strip().startswith('['):
            recipe_str = f"[{recipe_str}]"

        # Use ast.literal_eval for safe evaluation
        tuples = ast.literal_eval(recipe_str)

        # Validate format
        if not isinstance(tuples, list):
            raise ValueError("Recipe must be a list of tuples")

        voices = []
        weights = []

        for item in tuples:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("Each item must be a tuple with 2 elements")
            if not isinstance(item[0], str) or not isinstance(item[1], (int, float)):
                raise ValueError("Tuple format must be (str, float)")

            voices.append(item[0])
            weights.append(float(item[1]))  # Ensure it's a float

        return voices, weights

    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid voice recipe format: {e}")

def get_args_dict() -> dict:
    """
    Gets args dictionary useful for the Kokoro blend voices script.
    """

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Kokoro Voice Blending Tool - Voice blending of existing voices.',formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--language-code', default=LANGUAGE_CODE, help='The language code; a = American English, British English = b, Spanish = e, French = f, Italian = i, Brazilian Portuguese = p, Hindi = h.')
    parser.add_argument("--repo-id", default=REPO_ID,help="The repo ID - the vast majority of people use 'hexgrad/Kokoro-82M'.")
    parser.add_argument("--interactive", action='store_true', help="Select this if you wish to use different speakers if they are identified with colons in the text (the narrator reads things between asterisks). MUST use either --interactive or set voice-recipe!")
    parser.add_argument("--voice-recipe", type=parse_voice_recipe, default=([], []), help='Tuples that represent (voice, weight): "("bf_alice", 0.7), ("af_jessica", 0.3)" as an example. Tuple is (str, float) MUST use either voice-recipe or --interactive!')
    parser.add_argument('--file-name', default=FILE_NAME, help='The base of the output filenames (.pt and .wav). Do not include an extension.')
    parser.add_argument("--output-dir", default=OUTPUT_DIR,help="The output path of the interactive results; only useful in interactive mode.")

    argDict = {}

    try:
        args = parser.parse_args()

        argDict['language_code'] = args.language_code
        argDict['repo_id'] = args.repo_id
        argDict['interactive'] = args.interactive
        argDict['file_name'] = args.file_name
        if os.path.exists(args.output_dir):
            argDict['output_dir'] = args.output_dir
        else:
            print(f"Path '{args.output_dir}' does not exist - using '{OUTPUT_DIR}'")
            argDict['output_dir'] = OUTPUT_DIR

        argDict['voices'], argDict['weights'] = args.voice_recipe

        if not argDict['interactive'] and (len(argDict['voices']) == 0 or len(argDict['voices']) != len(argDict['weights'])):
            print(f"You have attempted to pre-populate weights and voices, but there were either no entries or the lengths did not equal each other; forcing interactive mode")
            argDict['interactive'] = True

        print(f"Arguments parsed.")


    except SystemExit as e:
        argDict = {}
        if e.code == 0:
            # --help was used, so print no error
            print(f"Thank you!")
        else:
            print(f"Invalid arguments.")

    return argDict

def main():

    args_dict = get_args_dict()

    # Load voices
    pipeline, voice_data = load_kokoro_voices(args_dict['language_code'], args_dict['repo_id'])
    working_voices = voice_data.keys()
    
    if not working_voices:
        print("No working voices found!")
        return

    if args_dict['interactive']:
        #if this is interactive, print some additional info so it can be used when making decisions
        print(f"Voice Analysis:")
        for voice_name in working_voices:
            tensor = voice_data[voice_name]
            print(f"   {voice_name}: {tensor.shape}, range: [{tensor.min():.3f}, {tensor.max():.3f}]")

        # Create blends
        if len(working_voices) >= 2: print(f"\nCreating blends...")

    # Create blends
    if len(working_voices) >= 2:

        keep_going = True
        voices = []
        weights = []

        #if there are voices in args_dict['voices'], make sure they are all valid - if not, bounce it
        for voice in args_dict['voices']:
            if voice not in working_voices:
                print(f"voice '{voice}' is not in the list of working voices - defaulting to interactive mode.")
                args_dict['interactive'] = True


        if args_dict['interactive']:
            # cycle through, getting an arbitrary number of voice/weight pairs until the user uses !blend (to blend the current pairs) or !quit
            while keep_going:
                one_voice = input(f"Enter voice #{len(voices) + 1} (or !quit to quit, or !blend to blend current voices entered): ").strip()
                if one_voice.lower() == '!quit':
                    break
                elif one_voice.lower() == '!blend':
                    # use whatever voices we have entered so far
                    if len(voices) == 0:
                        print(f"No voices entered to blend - quitting.")
                        break
                    else:
                        keep_going = False
                elif not one_voice or one_voice not in working_voices:
                    print(f"Voice {one_voice} not in working voice list.")
                    continue
                else:
                    voices.append(one_voice)

                    # Get weight with proper validation
                    while True:
                        one_weight = input(f"Enter weight for voice '{one_voice}' (preferably a float from 0 to 1): ").strip()
                        if one_weight.lower() == '!quit':
                            keep_going = False
                            break
                        try:
                            weight_float = float(one_weight)
                            weights.append(weight_float)
                            break
                        except ValueError:
                            print("Must be a valid number. Please try again.")

                    # If user quit during weight entry, break out of main loop too
                    if not keep_going:
                        break
        else:
            # we are guaranteed that these lists are equal and are valid
            voices = args_dict['voices']
            weights = args_dict['weights']

        if len(voices) > 0 and (len(voices) == len(weights)):
            #create the filename
            if args_dict['file_name']:
                filename_no_ext = args_dict['file_name']
            else:
                filename_no_ext = "_".join(f"{voice}_{weight}" for voice, weight in zip(voices, weights))

            # if this is interactive, we also have to append the path to all files
            if args_dict['interactive']:
                filename_no_ext = Path(args_dict['output_dir']) / filename_no_ext

            # Actually blend
            created_file = create_voice_blend(
                pipeline, voice_data,
                voices=voices,
                weights=weights,
                output_name=filename_no_ext
            )

        elif len(voices) > 0 and (len(voices) != len(weights)):
            print(f"Error - the number of voices and weights do not match.")
    else:
        print("Only 1 voice available?")


if __name__ == "__main__":
    main()
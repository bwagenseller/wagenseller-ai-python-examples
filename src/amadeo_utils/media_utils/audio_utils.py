import librosa
import soundfile as sf
import numpy as np
import ast
import warnings
import noisereduce as nr
import os
from pathlib import Path

from amadeo_utils.colored_text import ColoredText

"""
This SimpleAudioManipulation class has a bunch of helper methods, all of which are stand-alone:

    get_sample_rate - simply gets the sample rate of the file
    
    extract_full_audio - Simply extracts audio from a source file, returning a numpy array representing the audio data; from here, you can save to a file via write_audio_data_to_file
    or pass it to another function for further processing.
    
    extract_audio_sample - Simply extracts audio from a source file, returning a numpy array representing the audio data; you select a start time (in seconds, can be fractional) and a duration 
    of the clip. from here, you can save to a file via write_audio_data_to_file or pass it to another function for further processing.
    
    extract_multiple_samples - Much like extract_audio_sample, but this extracts multiple time segments from an audio file and joins them together with silence (this can do a single 
    segment, too). This returns a numpy array representing the audio data; from here, you can save to a file via write_audio_data_to_file or pass it to another function for further processing. 
        
    merge_audio_files - Merge two audio files, adding silence between them. This returns a numpy array representing the audio data
    
    parse_time_segments - Parse string of tuples into list of (start_time, duration) tuples. this is necessary when we want to splice together various portions of a single 
    audio file (we get this string of tuple from the command line).
    
    simple_audio_noise_reduction - Simple noise reduction using the noisereduce library
    
    noise_reduction_dir  - Simple noise reduction using the noisereduce library; cleans entire directory.  
    
"""


"""
This line:
  audio, sr = librosa.load(input_file, sr=target_sample_rate, mono=True)
  
Causes:
FutureWarning: librosa.core.audio.__audioread_load
	Deprecated as of librosa version 0.10.0.
	It will be removed in librosa version 1.0.
  y, sr_native = __audioread_load(path, offset, duration, dtype)

"""
# Suppress all FutureWarnings from librosa
warnings.filterwarnings("ignore", message=".*__audioread_load.*", category=FutureWarning)

"""
This line:
  audio, sr = librosa.load(input_file, sr=target_sample_rate, mono=True)
  
Causes:
UserWarning: PySoundFile failed. Trying audioread instead.
"""
warnings.filterwarnings("ignore", message=".*Trying audioread instead.*", category=UserWarning)



class SimpleAudioManipulation:
    """
    Simply returns the original, unaltered sample rate of the file
    """
    @staticmethod
    def get_sample_rate(input_file):
        # Load WITHOUT specifying sample rate (see what the file actually contains)
        audio_original, original_sample_rate = librosa.load(input_file, sr=None, mono=True)

        # audio_original is a numpy array of numbers containing the actual audio - but we dont use that, here
        return original_sample_rate

    """
    Simply extracts the entire audio from a source file, returning a numpy array of all the samples; from here, you can save to a file via write_audio_data_to_file
    or pass it to another function for further processing.
    
    input_file - the file we are extracting data from
    target_sample_rate - The sample rate you want the audio to end up as. Popular choices: 24000 (F5-TTS), 22050 (most other TTS libraries),
        44100 (CD quality), 16000 (lower quality), 8000 (old telephone quality)
        
    returns: data (Numpy array), actual sample rate of data
    """
    @staticmethod
    def extract_full_audio(input_file, target_sample_rate=24000):
        try:
            # We want the final result to have a sample rate of target_sample_rate, but there is no guarantee on the sample rate of the input file.
            # Therefore, Librosa takes in the target sample rate (sr) and will either upsample (i.e. add more samples) or downsample (i.e. remove samples)
            # to hit the target_sample_rate mark that said, sometimes its impossible to hit, so librosa returns sr to tell you the final sample rate
            audio, sr = librosa.load(input_file, sr=target_sample_rate, mono=True)

            # at this point, audio is a numpy array of numbers that represents the audio, with sr samples per second represented
        except Exception as e:
            print(f"{ColoredText.RED_TEXT}SimpleAudioManipulation.extract_full_audio: Error loading audio: {e}{ColoredText.END_TEXT}")
            return None

        # Normalize
        # Normalization scales all audio to use the full range of (-1.0 to +1.0)
        # Basically, normalization is Finding the loudest part and scales everything proportionally, ensuring consistent volume between different
        # source files
        # This is critical for AI models that expect consistent input levels
        audio = librosa.util.normalize(audio)

        # Apply fade in/out to the entire audio to prevent pops/clicks at the very beginning and end
        # 10ms fade: long enough to work, short enough to be inaudible
        # np.linspace(0, 1, fade_samples) creates a smooth volume curve from 0% to 100%
        fade_samples = int(0.01 * sr)  # 10 millisecond fade
        if len(audio) > 2 * fade_samples:
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)        # Fade in
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)       # Fade out

        return audio, sr


    """
    Simply extracts audio from a source file, returning a numpy array representing the audio data; you select a start time (in seconds, can be fractional) and a duration 
    of the clip. from here, you can save to a file via write_audio_data_to_file or pass it to another function for further processing.
    
    input_file - the file we are extracting data from
    start_time - The location of the video, in seconds, where to start the extraction.
    duration - The duration of the result; in other words, we will start at second start_time in the video and extract
        'duration' seconds after that.
    target_sample_rate - The sample rate you want the audio to end up as. Popular choices: 24000 (F5-TTS), 22050 (most other TTS libraries),
        44100 (CD quality), 16000 (lower quality), 8000 (old telephone quality)
        
    returns: data (Numpy array), actual sample rate of data
    """
    @staticmethod
    def extract_audio_sample(input_file, start_time=45, duration=15, target_sample_rate = 24000):

        try:
            # We want the final result to have a sample rate of target_sample_rate, but there is no guarantee on the sample rate of the input file.
            # Therefore, Librosa takes in the target sample rate (sr) and will either upsample (i.e. add more samples) or downsample (i.e. remove samples)
            # to hit the target_sample_rate mark that said, sometimes its impossible to hit, so librosa returns sr to tell you the final sample rate
            audio, sr = librosa.load(input_file, sr=target_sample_rate, mono=True)

            # at this point, audio is a numpy array of numbers that represents the audio, with sr samples per second represented
        except Exception as e:
            print(f"{ColoredText.RED_TEXT}SimpleAudioManipulation.extract_sample_to_audio: Error loading audio: {e}{ColoredText.END_TEXT}")
            return None

        # Normalize
        # Normalization scales all audio to use the full range of (-1.0 to +1.0)
        # Basically, normalization is Finding the loudest part and scales everything proportionally, ensuring consistent volume between different
        # source files
        # This is critical for AI models that expect consistent input levels
        audio = librosa.util.normalize(audio)

        # Extract the specific segment
        # Converting time to samples: seconds × samples_per_second = sample_index
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)

        if end_sample > len(audio):
            end_sample = len(audio)
            actual_duration = (end_sample - start_sample) / sr
            print(f"{ColoredText.BLUE_TEXT}SimpleAudioManipulation.extract_sample_to_audio: Warning: Requested segment goes beyond file length (may be OK if you are simply shotgunning this). Adjusted duration: {actual_duration:.2f}s{ColoredText.END_TEXT}")

        # chunk is literally a numpy array
        # len(chunk)/sr will tell us the curation in seconds
        chunk = audio[start_sample:end_sample]

        #  Fade in/out:
        # Prevents audio "pops" and "clicks" when audio starts/stops abruptly
        # 10ms fade: long enough to work, short enough to be inaudible
        # np.linspace(0, 1, fade_samples) creates a smooth volume curve from 0% to 100%
        fade_samples = int(0.01 * sr)  # 10 millisecond fade
        if len(chunk) > 2 * fade_samples:
            chunk[:fade_samples] *= np.linspace(0, 1, fade_samples)        # Fade in
            chunk[-fade_samples:] *= np.linspace(1, 0, fade_samples)       # Fade out

        return chunk, sr


    """
    Much like extract_audio_sample, but this extracts multiple time segments from an audio file and joins them together with silence (this can do a single segment, 
    too). This returns a numpy array of the samples given; from here, you can save to a file via write_audio_data_to_file or pass it to another function for further processing.
    
    Args:
        input_file: Path to input audio/video file
        time_segments: List of tuples [(start1, duration1), (start2, duration2), ...]
        target_sample_rate: Target sample rate for output
        silence_duration: Seconds of silence between segments
        
    Returns:
        Tuple of (audio data (Numpy Array), sample_rate)
    """
    @staticmethod
    def extract_multiple_samples(input_file, time_segments, target_sample_rate=24000, silence_duration=0.5):

        # Load the audio file once (more efficient than loading multiple times)
        try:
            audio, sr = librosa.load(input_file, sr=target_sample_rate, mono=True)
        except Exception as e:
            return None, None

        # Normalize the entire audio once
        audio = librosa.util.normalize(audio)

        # Create silence segment (500ms by default)
        silence_samples = int(silence_duration * sr)
        silence = np.zeros(silence_samples)

        # Extract all segments
        segments = []
        total_extracted_duration = 0

        for i, (start_time, duration) in enumerate(time_segments):

            # Calculate sample indices
            start_sample = int(start_time * sr)
            end_sample = int((start_time + duration) * sr)

            # Check bounds
            if start_sample >= len(audio):
                print(f"{ColoredText.BLUE_TEXT}SimpleAudioManipulation.extract_multiple_samples: Segment {i+1}: Start time {start_time}s beyond audio length.{ColoredText.END_TEXT}")
                continue

            if end_sample > len(audio):
                end_sample = len(audio)
                actual_duration = (end_sample - start_sample) / sr
                print(f"{ColoredText.BLUE_TEXT}SimpleAudioManipulation.extract_multiple_samples: Segment {i+1}: Truncated to {actual_duration:.2f}s.{ColoredText.END_TEXT}")

            # Extract the chunk
            chunk = audio[start_sample:end_sample]

            if len(chunk) == 0:
                print(f"{ColoredText.BLUE_TEXT}SimpleAudioManipulation.extract_multiple_samples: Segment {i+1}: Empty chunk, skipping.{ColoredText.END_TEXT}")
                continue

            # Apply fade in/out to each segment
            fade_samples = int(0.01 * sr)  # 10 millisecond fade
            if len(chunk) > 2 * fade_samples:
                chunk[:fade_samples] *= np.linspace(0, 1, fade_samples)
                chunk[-fade_samples:] *= np.linspace(1, 0, fade_samples)

            segments.append(chunk)
            total_extracted_duration += len(chunk) / sr

        if not segments:
            print(f"{ColoredText.RED_TEXT}SimpleAudioManipulation.extract_multiple_samples: No valid segments extracted.{ColoredText.END_TEXT}")


        # Combine all segments with silence between them
        combined_parts = []

        for i, segment in enumerate(segments):
            combined_parts.append(segment)

            # Add silence between segments (but not after the last one)
            if i < len(segments) - 1:
                combined_parts.append(silence)

        # Concatenate all parts
        combined_audio = np.concatenate(combined_parts)

        return combined_audio, sr

    """
    Merge two audio files, adding silence between them. This returns a numpy array representing the audio data
    
    Args:
        file1: Path to first audio file
        file2: Path to second audio file  
        silence_duration: Seconds of silence between files (default: 0.5)
        target_sample_rate: Target sample rate (default: 24000)
        
    Returns:
        Tuple of (audio data (Numpy Array), sample_rate)
    """
    def merge_audio_files(file1, file2, silence_duration=0.5, target_sample_rate=24000):
        try:
            # First file
            audio1, sr1 = librosa.load(file1, sr=target_sample_rate, mono=True)
            audio1 = librosa.util.normalize(audio1)

            # second file
            audio2, sr2 = librosa.load(file2, sr=target_sample_rate, mono=True)
            audio2 = librosa.util.normalize(audio2)

            # if the sample rates do not match, we need to force them to match
            if sr1 != sr2:
                if sr1 == target_sample_rate:
                    print(f"{ColoredText.BLUE_TEXT}SimpleAudioManipulation.merge_audio_files: Sample rate of file2 is off - resampling to force it to {target_sample_rate}.{ColoredText.END_TEXT}")
                    # Resample audio2 to match audio1
                    audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr1)
                    final_sr = sr1
                else:
                    print(f"{ColoredText.BLUE_TEXT}SimpleAudioManipulation.merge_audio_files: Sample rate of both files are off - resampling both to force it to {target_sample_rate}.{ColoredText.END_TEXT}")
                    # Resample both to target
                    audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=target_sample_rate)
                    audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=target_sample_rate)
                    final_sr = target_sample_rate

            # Create silence
            silence_samples = int(silence_duration * target_sample_rate)
            silence = np.zeros(silence_samples)

            # Combine: audio1 + silence + audio2
            combined = np.concatenate([audio1, silence, audio2])

            return combined, target_sample_rate

        except Exception as e:
            print(f"{ColoredText.RED_TEXT}SimpleAudioManipulation.merge_audio_files: Error merging files: {e}.{ColoredText.END_TEXT}")
            return None

    """
    Simply writes an audio data chunk (Nupy array) to a file.
    Basically, a wrapper for soundfile.write 
    """
    @staticmethod
    def write_audio_data_to_file(output_file, data, sample_rate):
        """
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_f5_exact_{start_time}s_{duration}s.wav"
        """

        sf.write(output_file, data, sample_rate)

        return str(output_file)

    """
    Parse string of tuples into list of (start_time, duration) tuples. this is necessary when we want to splice together various portions of a single 
    audio file (we get this string of tuple from the command line).
    
    Examples:
        "(5,15), (25,10), (40,35)" -> [(5, 15), (25, 10), (40, 35)]
        "(10, 5)" -> [(10, 5)]
    """
    @staticmethod
    def parse_time_segments(segments_string):
        try:
            # Clean the string and make it a proper Python list
            cleaned = segments_string.strip()
            if not cleaned.startswith('['):
                cleaned = '[' + cleaned + ']'

            # Parse as Python literal
            segments = ast.literal_eval(cleaned)

            # Validate that all items are tuples with 2 elements
            validated_segments = []
            for i, segment in enumerate(segments):
                if not isinstance(segment, tuple) or len(segment) != 2:
                    raise ValueError(f"Segment {i+1} must be a tuple with 2 values (start_time, duration)")

                start_time, duration = segment
                if not isinstance(start_time, (int, float)) or not isinstance(duration, (int, float)):
                    raise ValueError(f"Segment {i+1} times must be numbers")

                if start_time < 0 or duration <= 0:
                    raise ValueError(f"Segment {i+1}: start_time must be ≥0, duration must be >0")

                validated_segments.append((float(start_time), float(duration)))

            return validated_segments

        except Exception as e:
            raise ValueError(f"Invalid time segments format: {e}")

################################################################################ Noise Reduction Classes ################################################################################

    @staticmethod
    def simple_audio_noise_reduction(input_file, output_file=None):
        """
        Simple noise reduction using noisereduce library

        Args:
            input_file: Path to input audio file
            output_file: Path for output (optional)
        """

        print(f"Processing: {input_file}")

        # Generate output filename if not provided
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.parent / f"{input_path.stem}_cleaned.wav"

        try:
            # Load audio file
            audio, sample_rate = librosa.load(input_file, sr=None)  # Keep original sample rate
            print(f"  Loaded audio: {len(audio)/sample_rate:.1f}s at {sample_rate}Hz")

            # Reduce noise - try different parameter combinations
            print("  Reducing noise...")
            try:
                # Try newer API first
                cleaned_audio = nr.reduce_noise(y=audio, sr=sample_rate)
            except TypeError:
                try:
                    # Try older API
                    cleaned_audio = nr.reduce_noise(audio_clip=audio, noise_clip=audio)
                except TypeError:
                    # Try minimal parameters
                    cleaned_audio = nr.reduce_noise(audio, sample_rate)
            except Exception as e:
                print(f"    All reduce_noise attempts failed: {e}")
                return None

            # Save cleaned audio
            sf.write(output_file, cleaned_audio, sample_rate)
            print(f"  Saved to: {output_file}")

            return output_file

        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            return None

    @staticmethod
    def noise_reduction_dir(input_dir, output_dir=None):
        """
        Simple noise reduction using the noisereduce library; cleans all .wav / .mp3 files in a directory
        Different strength levels to try:

        Args:
            input_file: Path to input audio file
            output_file: Path for output (optional)

        Returns:
            A list of all files cleaned
        """

        input_path = Path(input_dir)

        if output_dir is None:
            output_path = input_path / "cleaned"
        else:
            output_path = Path(output_dir)

        # Create output directory
        output_path.mkdir(exist_ok=True)

        # Find all wav files
        wav_files = list(input_path.glob("*.wav"))
        wav_files.extend(input_path.glob("*.WAV"))
        wav_files.extend(input_path.glob("*.mp3"))  # Also handle mp3

        if not wav_files:
            print(f"No audio files found in {input_dir}")
            return []

        print(f"Found {len(wav_files)} files to clean")

        cleaned_files = []
        for i, audio_file in enumerate(wav_files, 1):
            print(f"\n[{i}/{len(wav_files)}]")

            output_file = output_path / f"{audio_file.stem}_cleaned.wav"
            result = SimpleAudioManipulation.simple_audio_noise_reduction(audio_file, output_file)

            if result:
                cleaned_files.append(result)

        print(f"\nCompleted! Cleaned {len(cleaned_files)} files")
        print(f"Output directory: {output_path}")

        return cleaned_files
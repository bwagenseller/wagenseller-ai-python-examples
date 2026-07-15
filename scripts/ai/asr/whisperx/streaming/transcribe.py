import sounddevice as sd # Library for recording and playing audio from devices like microphones.
import numpy as np       # Numerical computing library; essential for handling audio data as arrays efficiently.
import whisperx          # The enhanced Whisper ASR library, optimized for speed and precise timestamps.
import queue             # Module for implementing multi-producer, multi-consumer queues; crucial for safe
                         # data exchange between different threads without race conditions.
import threading         # Module for managing separate threads of execution, allowing concurrent tasks
                         # (e.g., recording audio, processing VAD, transcribing) to run simultaneously.
import time              # Module for time-related functions; used for introducing small delays and
                         # for calculating time durations if needed.
import os                # Module for interacting with the operating system; not heavily used here,
                         # but generally useful for file paths, environment variables, etc.
import gc                # Garbage Collector interface; useful for explicitly prompting Python to
                         # reclaim unused memory, especially important in long-running applications
                         # that handle large data (like audio) or models.
import torch             # PyTorch deep learning library; used for model operations (loading, inference)
                         # and for checking GPU (CUDA) availability.
import webrtcvad         # WebRTC Voice Activity Detector; a highly efficient library for distinguishing
                         # speech from non-speech (silence, noise) in audio.
from datetime import datetime

from amadeo_utils.ai.asr.whisperx import detect_language_with_probability

"""
I got this from Google Gemini.

This simulates streaming by chunking the audio into segments; it utilizes VAD (Voice Activity Detector) to determine silence, and if there is an 800 millisecond chunk of time that is silence (configurable), the segment is determined to have ended and is 
sent to the whisperx model for processing / to transcribe it. 

This is an improvement off of the 'streaming.py' file.
"""

# ---------------------------------------------------------------------
# Place these lines IMMEDIATELY after importing torch
# This enables TF32 for your NVIDIA RTX 5090 for performance
# By default, TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues and lower accuracy.
# However, for more modern Nvidia GPUs, especially for the Ampere, Ada Lovelace, and Blackwell architectures - you will probably see a performance boost
# If, however, there are accuracy issues, do not enable these

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# ---------------------------------------------------------------------

# --- Configuration ---
# These variables configure how the real-time transcription process will behave.
# Adjust them based on your hardware, desired accuracy, and latency requirements.

# Specifies the Whisper ASR model to use. Smaller models like "base.en" or "small.en"
# are generally preferred for real-time streaming as they offer lower latency and faster inference,
# even if slightly less accurate than larger models.
#WHISPER_MODEL_NAME = "base.en"
WHISPER_MODEL_NAME = "large-v3"

# Determines the computational device for WhisperX models.
# `torch.cuda.is_available()` is the standard PyTorch way to check if a CUDA-enabled GPU
# is detected and properly configured. If true, "cuda" is used; otherwise, "cpu".
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sets the computation precision for the models.
# "float16" (half-precision floating point) is faster and uses less VRAM on GPUs,
# typically with minimal impact on ASR accuracy during inference.
# "int8" (8-bit integer quantization) is highly efficient for CPU inference,
# offering speed benefits on systems without a strong GPU.
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# The language code of the speech you expect to transcribe. This is crucial for:
# 1. Loading the correct language-specific alignment model.
# 2. Guiding the Whisper ASR model, especially multilingual ones.
LANGUAGE_CODE = "en"

# --- Audio Recording Parameters (for sounddevice and VAD) ---
# These settings define how audio is captured from the microphone and processed by VAD.

# The sample rate of the audio (samples per second). Whisper models and WebRTC VAD
# are optimized for 16kHz audio, so it's the recommended rate.
SAMPLE_RATE = 16000 # Hz

# The duration of each tiny audio frame (chunk) that WebRTC VAD will analyze.
# WebRTC VAD requires this to be strictly 10, 20, or 30 milliseconds.
# A smaller value allows for more precise silence detection but slightly more overhead.
VAD_FRAME_DURATION_MS = 30 # milliseconds

# The actual number of audio frames (samples) that `sounddevice` will pass to its
# callback function at each invocation. This is derived from the VAD frame duration.
BLOCK_SIZE_SOUNDDEVICE = int(SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)

# --- VAD (Voice Activity Detection) Parameters ---

# Aggressiveness mode for WebRTC VAD (integer from 0 to 3).
# Higher values (e.g., 3) mean the VAD is more aggressive in filtering out
# non-speech, potentially missing some quiet speech. Lower values (e.g., 0)
# are less aggressive, allowing more background noise to be considered speech.
# A value of 1 or 2 is often a good balance for general use.
VAD_AGGRESSIVENESS = 1

# The duration of consecutive silence, in milliseconds, that will trigger
# the end of the current speech segment. Once this much silence is detected,
# the accumulated audio is sent for transcription.
SILENCE_DURATION_TO_END_BUFFER_MS = 800 # milliseconds

# --- Global State & Queues ---
# These queues facilitate communication and data transfer between the different threads
# in a thread-safe manner.

# Queue for raw, small audio frames (e.g., 30ms) coming directly from the microphone callback.
# These frames are consumed by the VAD processing thread.
raw_audio_frames_queue = queue.Queue()

# Queue for full speech segments that have been identified by the VAD. These segments
# are ready to be transcribed by the WhisperX transcription thread.
transcription_queue = queue.Queue()

# A threading.Event object used for signaling. When `stop_event.set()` is called,
# it signals all threads listening to this event (via `stop_event.is_set()`) to
# gracefully terminate their operations. Essential for clean shutdown.
stop_event = threading.Event()

# --- WhisperX Model Loading ---
# Models are loaded only once at the beginning of the script to minimize overhead
# and avoid repeated loading during continuous real-time processing.

# WhisperX's VAD is pyannote, which disables TensorFloat-32 for reproducibility on
# every CUDA inference and warns loudly each time it finds TF32 still enabled
# (torch defaults cudnn.allow_tf32 to True). Setting the same state up front leaves
# behaviour identical - pyannote would force it anyway - while keeping the output
# clean. Do not re-enable TF32 to chase speed: pyannote re-disables it per
# inference, and the actual Whisper transcription runs through CTranslate2, which
# ignores these torch flags entirely.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print(f"Loading WhisperX model '{WHISPER_MODEL_NAME}' on {DEVICE} with {COMPUTE_TYPE}...")
# `whisperx.load_model()` loads the main ASR model (e.g., "base.en").
# It handles downloading the model if not cached and converting it to the
# optimized CTranslate2 format for faster inference.
asr_model = whisperx.load_model(WHISPER_MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
print("ASR model loaded.")

print(f"Loading WhisperX alignment model for '{LANGUAGE_CODE}' on {DEVICE}...")
# `whisperx.load_align_model()` loads the specialized model used for forced alignment.
# This model takes the ASR-transcribed text and the audio, and then pinpoints
# the exact start and end times for each individual word.
# It returns the alignment model itself and `metadata` which is also needed by `align()`.
align_model, metadata = whisperx.load_align_model(language_code=LANGUAGE_CODE, device=DEVICE)
print("Alignment model loaded.")

# Initialize WebRTC VAD. The `Vad` class constructor takes the aggressiveness mode.
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
print(f"WebRTC VAD initialized with aggressiveness mode {VAD_AGGRESSIVENESS}.")

# --- Audio Callback Function ---
# This function is executed by `sounddevice` in a separate, internal audio thread.
# It's called automatically whenever a new block (chunk) of audio data is received from the microphone.
def audio_callback(indata, frames, time_info, status):
    """
    Callback function for `sounddevice.InputStream`.
    It receives raw audio data from the microphone and places it into `raw_audio_frames_queue`.

    Args:
        indata (np.ndarray): NumPy array containing the audio data for the current block.
                             Typically float32, with shape (frames, channels).
        frames (int): The number of frames (samples) in `indata`.
        time_info (dict): Dictionary containing stream timing information.
        status (sd.CallbackFlags): An object indicating any stream warnings or errors.
    """
    if status:
        # Print any warnings or errors reported by `sounddevice` (e.g., input overflow).
        print(f"Audio status: {status}", flush=True)

    # WebRTC VAD requires audio in 16-bit integer PCM format (Pulse-Code Modulation).
    # `sounddevice` typically provides `float32`. We convert it here:
    # 1. Multiply by 32767.0 (max value for int16) to scale `float32` (-1.0 to 1.0) to `int16` range.
    # 2. Cast to `np.int16`.
    # 3. `.tobytes()` converts the NumPy array of int16s into a raw byte string, which VAD expects.
    #    This also implicitly flattens the array.
    int16_data = (indata * 32767.0).astype(np.int16).tobytes()
    
    # Put the processed audio frame into the `raw_audio_frames_queue`. This transfers
    # the data from the `sounddevice` thread to the `vad_processing_thread` safely.
    raw_audio_frames_queue.put(int16_data)

# --- VAD Processing Thread Function ---
# This is a custom thread that continuously pulls small audio frames from `raw_audio_frames_queue`,
# applies WebRTC VAD, and intelligently buffers them into larger speech segments based on silence.
def vad_processing_thread():
    """
    Thread function to perform VAD on incoming audio frames and assemble complete speech segments.
    It identifies speech boundaries based on detected silence and puts full segments into `transcription_queue`.
    """
    print("\nStarting VAD processing thread...")

    # A `bytearray` is used to efficiently accumulate raw 16-bit PCM audio frames.
    current_audio_buffer = bytearray()

    # Counter for consecutive silent frames. Used to detect an end-of-speech segment.
    silent_frames_count = 0

    # Calculate the number of consecutive silent frames required to trigger a segment end.
    # This calculation MUST be done here, outside the `while` loop, so `silent_frames_threshold`
    # is defined when first accessed in the `if` condition within the loop.
    silent_frames_threshold = int(SILENCE_DURATION_TO_END_BUFFER_MS / VAD_FRAME_DURATION_MS)

    # The main loop of the VAD thread. It continues until `stop_event` is set.
    while not stop_event.is_set():
        try:
            # Attempt to get a small audio frame from the queue.
            # `timeout=0.1` makes the `get()` call non-blocking for a short period.
            # This allows the thread to periodically check `stop_event.is_set()`
            # even if no audio is coming in, enabling graceful shutdown.
            frame_bytes = raw_audio_frames_queue.get(timeout=0.1)
            
            # Use `vad.is_speech()` to classify the current frame as speech or non-speech.
            # It expects 16-bit PCM `bytes` and the sample rate.
            is_speech = vad.is_speech(frame_bytes, SAMPLE_RATE)

            if is_speech:
                # If speech is detected, reset the silence counter (as speech broke any silence)
                silent_frames_count = 0
                # Extend the current buffer with the new speech frame.
                current_audio_buffer.extend(frame_bytes)
            else:
                # If silence/non-speech is detected, increment the silence counter.
                silent_frames_count += 1
                # Still add the silent frame to the buffer. We'll trim leading/trailing silence later.
                current_audio_buffer.extend(frame_bytes)

                # Check if the accumulated consecutive silence exceeds the defined threshold.
                if silent_frames_count >= silent_frames_threshold:
                    # If silence threshold is met AND there's audio in the buffer (meaning speech occurred before silence).
                    if len(current_audio_buffer) > 0:
                        # Convert the `bytearray` buffer back to a NumPy array of `int16`.
                        temp_np_array_int16 = np.frombuffer(current_audio_buffer, dtype=np.int16)
                        
                        # --- Trim Trailing Silence ---
                        # Calculate the number of samples corresponding to the `silence_frames_threshold`.
                        num_samples_to_trim = silent_frames_threshold * BLOCK_SIZE_SOUNDDEVICE
                        
                        # Determine the effective end of the speech by removing the trailing silence.
                        # `max(0, ...)` ensures we don't end up with a negative index if the buffer is tiny.
                        effective_speech_samples = max(0, len(temp_np_array_int16) - num_samples_to_trim)
                        
                        # Convert the trimmed `int16` NumPy array to `float32` and scale it
                        # back to the -1.0 to 1.0 range, as required by WhisperX.
                        speech_segment_np_float32 = (temp_np_array_int16[:effective_speech_samples].astype(np.float32) / 32767.0)
                        
                        # Only put the segment into the `transcription_queue` if it contains actual audio
                        # after trimming (i.e., not an empty array).
                        if speech_segment_np_float32.shape[0] > 0:
                            transcription_queue.put(speech_segment_np_float32)
                        
                        # Reset the buffer and silence counter for the next speech segment.
                        current_audio_buffer = bytearray()
                        silent_frames_count = 0
            
            # Immediately check if the global stop event has been set, and exit the loop if it has.
            if stop_event.is_set():
                break

        except queue.Empty:
            # If `raw_audio_frames_queue.get(timeout=0.1)` times out, this exception is raised.
            # It simply means no new audio frame arrived within the timeout.
            # We then check the `stop_event` and `continue` the loop to try again.
            if stop_event.is_set():
                break
            continue # Loop back to try getting a frame again
        except Exception as e:
            # Catch any other unexpected errors in the VAD processing thread.
            print(f"Error in VAD processing thread: {e}", flush=True)
            break # Exit the thread on unexpected error

    print("VAD processing thread stopping.")
    # After the VAD thread has stopped (its loop has exited), put a `None` sentinel
    # into the `transcription_queue` to signal the `transcribe_audio_thread` to stop.
    transcription_queue.put(None)

# --- Transcription Thread Function ---
# This thread continuously pulls full speech segments (identified by VAD) from `transcription_queue`
# and processes them using WhisperX's ASR and alignment capabilities.
def transcribe_audio_thread():
    """
    Thread function to transcribe full speech segments received from the VAD thread.
    It uses WhisperX for ASR and word-level alignment, then prints the results.
    """
    print("Starting transcription thread...")

    while True: # Loop indefinitely to process incoming speech segments
        try:

            # Get a complete speech segment from the `transcription_queue`.
            # `get()` blocks until a segment (or the `None` sentinel) is available.
            audio_segment = transcription_queue.get()
            
            # Check for the `None` sentinel value; if received, it's time to stop the thread.
            if audio_segment is None:
                print("Transcription thread stopping.")
                break # Exit the loop, terminating the thread


            # Use the .detect_language() method of the asr_model object
            # This returns a tuple: (language_code, {language_code: probability_score})
            # where the second element is a dictionary of all language probabilities.
            # We're interested in the probability of the *detected* language.
            
            # Pad or trim the audio for language detection if it's not already 30s
            # Whisper's language detection is optimized for 30s chunks.
            # Convert to appropriate format for detect_language
            audio_for_language_detection = whisperx.audio.pad_or_trim(audio_segment.astype(np.float32))

            # whisperx's own asr_model.detect_language() computes the language
            # probability internally but returns ONLY the language code, discarding
            # the confidence we want. detect_language_with_probability() repeats the
            # same steps and returns both.
            #
            # This used to be a method hand-patched into whisperx/asr.py inside
            # site-packages. That patch was invisible to this repo and vanished the
            # moment the conda env was rebuilt, so it now lives in amadeo_utils.
            detected_language, language_probability = detect_language_with_probability(asr_model, audio_for_language_detection)


            # Now perform the transcription, explicitly passing the detected language.
            #    This suppresses the "Warning: audio is shorter than 30s..." message
            #    and guides the model for better transcription of the detected language.
            result = asr_model.transcribe(audio_segment, batch_size=1, language=detected_language)

            # The 'result' has a few pieces:
            # * 'segments' - a list of dictionaries, with each dictionary containing 'text' (the entire spoken segment), 'start' (the start of the entire segment in the string), and 'end' (the end of the entire segment in the string)
            # * 'language' - the language code (i.e. 'en')
            # * Note this does NOT seem to have confidence scores or individual word start/end times - that seems to come from 'whisperx.align'

            # Process and print the transcription results.
            # First, check if the ASR model actually detected any speech segments.
            # `result` might be empty or not contain "segments" if no speech was confidently detected.
            if result and "segments" in result and result["segments"]:
                # If segments are found, proceed with forced alignment to get precise word timestamps.
                # All necessary parameters (`segments`, `align_model`, `metadata`, `audio_segment`, `device`)
                # are passed to `whisperx.align()`.
                word_segments_result = whisperx.align(result["segments"], align_model, metadata, audio_segment, device=DEVICE)
                # word_segments_result is built of
                # * segments  - like above, still containing 'start/end/text but now has a 'words' list of dictionaries, one for every word spoken
                #   * 'words' dictionary - for every entry:
                #     * 'word' - the word
                #     * 'start' - the start time of the word, relative to the segment beginning
                #     * 'end' - the end time of the word, relative to the segment beginning
                #     * 'score' - the confidence score behind the word
                # * 'word_segments' dictionary - seems to be the same as the words dictionary, just...outside of the 'segments' dictionary

                
                # After alignment, check if `word_segments_result` actually contains
                # precise word-level segments. It might be empty if alignment failed or
                # if the original ASR segments were too short/noisy for confident alignment.
                if word_segments_result and "word_segments" in word_segments_result and word_segments_result["word_segments"]:
                    full_text = ""
                    # Iterate directly over the word segments list.
                    # Each 'word_info' will directly be a dictionary like {'word': 'going', 'start': ..., 'end': ...}
                    for word_info in word_segments_result["word_segments"]:
                        if "word" in word_info: # Check if the 'word' key exists in the current word_info dictionary
                            full_text += f"{word_info['word']} "

                            #There are other things you can use too - start, end, (confidence) score. The start and end are relative to the segment
                            #full_text += f"{word_info['word']} ({word_info['start']}/{word_info['end']}) ({word_info['score']}) "
                        else:
                            # This warning is a safety check for a malformed word_info, less likely now.
                            print(f"Warning: 'word' key missing in word_info: {word_info}", flush=True)

                                
                    # If `full_text` is not empty after stripping whitespace, print the transcription.
                    if full_text.strip():
                        print(f"{datetime.now().isoformat(timespec='seconds')}: {full_text.strip()}", flush=True)
                else:
                    # This path is executed if ASR found speech, but alignment couldn't produce words.
                    # Suppressed for cleaner output.
                    pass # print("... (alignment found no words in detected speech) ...", flush=True)
            else:
                # This path is executed if WhisperX's ASR model itself detected no speech
                # in the current segment passed from the VAD thread.
                # Suppressed for cleaner output.
                pass # print("... (no discernible speech in segment by ASR) ...", flush=True)

            # --- Memory Management ---
            # Explicitly delete the audio segment and trigger garbage collection
            # to free up memory after processing, especially important for GPU VRAM.
            del audio_segment
            gc.collect()
            if DEVICE == "cuda":
                # Clear PyTorch's internal GPU memory cache to make VRAM immediately available.
                torch.cuda.empty_cache()

        except Exception as e:
            # General error handling for the transcription thread.
            print(f"Error during transcription: {e}", flush=True)
            # Depending on the error, you might want to `break` here to stop the thread
            # or `continue` to try processing the next segment. For now, it continues.

# --- Main Execution Block ---
# This is the bullseye of the script when it's run, orchestrating the threads and microphone input.
if __name__ == "__main__":

    # Create and start the VAD processing thread.
    # It will begin pulling raw audio frames from `raw_audio_frames_queue`.
    vad_thread = threading.Thread(target=vad_processing_thread)
    vad_thread.start()

    # Create and start the transcription thread.
    # It will begin pulling processed speech segments from `transcription_queue`.
    transcription_thread = threading.Thread(target=transcribe_audio_thread)
    transcription_thread.start()
    
    try:
        # Initialize and start the microphone input stream using `sounddevice`.
        # `samplerate`: The audio sample rate.
        # `channels`: 1 for mono audio, suitable for ASR.
        # `callback`: Specifies `audio_callback` to be called with incoming audio data.
        # `blocksize`: The size of audio chunks passed to `audio_callback`.
        #              This is now set to a small value (e.g., 30ms frames) for VAD.
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=BLOCK_SIZE_SOUNDDEVICE
        )
        
        print(f"Recording audio from microphone at {SAMPLE_RATE} Hz in {VAD_FRAME_DURATION_MS} ms frames...")
        
        # The `with stream:` statement ensures the audio stream is properly
        # opened and automatically closed when the `with` block is exited.
        with stream:
            # The main thread simply stays alive here, allowing the `sounddevice`
            # audio callback, VAD thread, and transcription thread to run in the background.
            # It waits for a `KeyboardInterrupt` (Ctrl+C) to initiate shutdown.
            while True:
                time.sleep(0.1) # Small delay to prevent the main thread from busy-waiting and consuming excessive CPU.
                
    except KeyboardInterrupt:
        # This block is executed when the user presses Ctrl+C.
        print("\nStopping application via Ctrl+C...")
    except Exception as e:
        # Catch any other unexpected exceptions that might occur in the main thread.
        print(f"An error occurred in the main process: {e}")
    finally:
        # --- Graceful Shutdown Sequence ---
        # This `finally` block ensures that cleanup operations are performed
        # whether an exception occurred or the script was stopped gracefully.

        # 1. Signal all running threads to stop.
        # `stop_event.set()` changes the event's internal flag, which threads check.
        stop_event.set() 
        
        # 2. Wait for the VAD processing thread to finish its current work and exit its loop.
        # `join()` blocks the main thread until the target thread terminates.
        vad_thread.join()
        
        # 3. Wait for the transcription thread to finish processing any remaining items
        # in its queue and then exit its loop (triggered by the `None` sentinel from VAD thread).
        transcription_thread.join()
        
        # 4. Explicitly delete the loaded WhisperX models. This is important for
        # releasing GPU VRAM and system memory occupied by the models.
        del asr_model
        del align_model
        
        # 5. Trigger Python's garbage collector one last time to reclaim any remaining
        # unused memory from Python objects.
        gc.collect()
        
        # 6. If running on CUDA, clear PyTorch's internal GPU memory cache. This ensures
        # that all GPU memory is released back to the system.
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            
        print("Application stopped.")
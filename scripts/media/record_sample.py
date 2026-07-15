import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import time
import sys
import termios
import tty
import select

from amadeo_utils.media_utils.audio_devices import prefer_pulse_defaults

"""
##########Recording Settings##########

Sample Rate:
    24000 Hz is used by F5-TTS 
    22050 Hz is used by Tortoise
    44100 Hz is the best in general. 
    If the sample rate is higher - or lower - the TTS ingesting this will have to resample
Channels: Mono (1 channel) - stereo will be converted anyway
Bit Depth: 16-bit minimum, 24-bit preferred for quality
Format: WAV (uncompressed)

##########Number and Length##########

For F5-TTS:
    ONE Sample, 10-15 seconds (although 8-30 would work)  
For Tortoise:
    Count: 6-10 audio files minimum, 15-20 for better results
    Length: 6-10 seconds each (Tortoise works best with this range)
    Total: About 2-3 minutes of total audio


Recording tips:

For Tortoise:
    Don't sound like you're "reading" - speak the Harvard sentences like you're explaining them to a friend
    Include some sentences with questions/statements that have natural inflection
    Vary your pacing slightly between recordings
    If you mess up, just re-record that sample - don't try to edit
    **15-25 high-quality, varied samples is probably optimal. Beyond that, you're mostly just increasing processing time without major quality gains.**
    The key is sounding like you across different types of content, not like a formal narrator.

##########Recording Quality Tips##########

Quiet environment - minimal background noise
Consistent distance from microphone (~6-12 inches)
Natural speaking pace - not too slow or fast
Varied content - different sentences, emotions, phonemes
Clear articulation - avoid mumbling
Consistent volume levels across all files

Different emotions (neutral, happy, serious)
Various phonetic sounds (cover different consonants/vowels)
Different sentence lengths
Natural pauses and inflections
"""


# Audio recording parameters optimized for F5-TTS and Tortoise TTS
RATE = 24000  # 24kHz for F5-TTS (22050 for Tortoise, but 24k works for both)
CHANNELS = 1  # Mono - TTS models expect single channel audio
DTYPE = 'int16'  # 16-bit integer - good balance of quality/file size for TTS
MAX_RECORDING_TIME = 60*60*3  # Maximum recording time: 3 hours
GAIN = 1.0  # Audio gain multiplier (0.1 to 2.0)

# Global variables for recording control
stop_recording = False
start_time = None

def wait_for_stop_key():
    """
    Block until the user presses the spacebar in this terminal, or until recording
    stops for another reason (e.g. the audio callback hitting MAX_RECORDING_TIME).

    Reads this process's own stdin rather than grabbing a global hotkey. Wayland
    does not permit applications to grab global key events - by design, since that
    is what a keylogger does - so an X11-based listener silently receives nothing
    on a Wayland session. Reading our own terminal works identically under
    Wayland, X11 and over SSH.

    The trade-off is that this terminal must have focus, which is a non-issue for
    a tool the user is sitting in front of and speaking into.

    :return: None. Sets the global stop_recording flag once a stop is requested.
    """
    global stop_recording

    # Without a TTY (piped/redirected stdin) there is no terminal to reconfigure,
    # so fall back to waiting for a newline instead of a single keypress.
    if not sys.stdin.isatty():
        sys.stdin.readline()
        stop_recording = True
        return

    file_descriptor = sys.stdin.fileno()
    original_settings = termios.tcgetattr(file_descriptor)

    try:
        # cbreak delivers keys as they are typed without waiting for Enter, while
        # (unlike raw mode) leaving Ctrl+C able to interrupt.
        tty.setcbreak(file_descriptor)

        while not stop_recording:
            # Poll rather than block on read() so that stop_recording being set
            # elsewhere - such as the max-duration guard in the audio callback -
            # still ends this loop promptly.
            if select.select([sys.stdin], [], [], 0.1)[0]:
                if sys.stdin.read(1) == ' ':
                    stop_recording = True
    finally:
        # Always restore the terminal, otherwise a raised exception would leave
        # the user's shell in cbreak mode with no echo.
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, original_settings)

def normalize_audio_int16(audio_data, target_level=-12):
    """
    Normalize int16 audio to a target dB level.

    Process:
    1. Convert int16 (-32768 to 32767) to float32 (-1.0 to 1.0)
    2. Calculate RMS and current dB level
    3. Apply gain to reach target dB level
    4. Convert back to int16 format
    """
    # Convert int16 to float32 for calculations (range -32768 to 32767 -> -1.0 to 1.0)
    audio_float = audio_data.astype(np.float32) / 32768.0

    # Calculate RMS (Root Mean Square) - measure of average audio level
    rms = np.sqrt(np.mean(audio_float**2))

    if rms > 0:
        # Calculate current dB level (reference is 1.0 for full scale)
        current_db = 20 * np.log10(rms)

        # Calculate gain needed to reach target level
        gain_db = target_level - current_db
        gain_linear = 10**(gain_db/20)

        # Apply gain and clip to prevent distortion
        audio_normalized = audio_float * gain_linear
        audio_normalized = np.clip(audio_normalized, -1.0, 1.0)

        # Convert back to int16 (range -1.0 to 1.0 -> -32767 to 32767)
        return (audio_normalized * 32767).astype(np.int16)

    return audio_data

def normalize_audio_float32(audio_data, target_level=-12):
    """
    Normalize float32 audio to a target dB level.
    Audio is already in [-1.0, 1.0] range, so no conversion needed.
    """
    # Calculate RMS of float32 data
    rms = np.sqrt(np.mean(audio_data**2))

    if rms > 0:
        # For float32, full scale is 1.0, so reference is 1.0
        current_db = 20 * np.log10(rms)

        # Calculate and apply gain
        gain_db = target_level - current_db
        gain_linear = 10**(gain_db/20)

        # Apply gain and clip to [-1.0, 1.0]
        audio_normalized = audio_data * gain_linear
        audio_normalized = np.clip(audio_normalized, -1.0, 1.0)

        return audio_normalized

    return audio_data

def check_audio_levels():
    """
    Quick microphone test to check input levels before recording.
    Shows a real-time level meter for 3 seconds.
    """
    print("Mic test - speak normally for 3 seconds...")
    test_duration = 3

    def callback(indata, frames, time, status):
        # FIXED: Normalize int16 values to 0-1 range for proper level display
        if DTYPE == 'int16':
            # Convert int16 (-32768 to 32767) to normalized range (0-1)
            normalized_data = np.abs(indata.astype(np.float32)) / 32768.0
            level = np.max(normalized_data)  # Use peak level for responsiveness
        else:  # float32
            # Data is already in [-1, 1] range, just get absolute max
            level = np.max(np.abs(indata))

        # Create visual level bar (50 characters wide)
        bar_length = max(0, min(50, int(level * 50)))
        bar = '█' * bar_length + '░' * (50 - bar_length)
        sys.stdout.write(f'\r[{bar}] {level:.3f}')
        sys.stdout.flush()

    # Stream audio for level checking
    with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype=DTYPE, callback=callback):
        sd.sleep(int(test_duration * 1000))  # Sleep duration in milliseconds

    print("\n")

def record_audio():
    """
    Main recording function.

    Process:
    1. Check audio levels first
    2. Wait for user confirmation
    3. Record audio with real-time level display
    4. Process and normalize if needed
    5. Save to WAV file
    """
    global stop_recording, start_time
    stop_recording = False

    # Route through PulseAudio where available: RATE (24 kHz) is not natively
    # supported by most raw ALSA capture devices, which reject it outright.
    prefer_pulse_defaults()

    # Quick level check before recording
    check_audio_levels()

    input("Press Enter when ready to start recording (then spacebar to stop)...")

    # Countdown before recording starts
    print("Get ready! Recording will start in...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("🔴 RECORDING! Press SPACEBAR to stop.")

    # Initialize recording variables
    start_time = time.time()
    frames = []  # Store audio chunks
    peak_level = 0.0  # Track maximum level reached

    def callback(indata, frame_count, time_info, status):
        """
        Audio callback function - called for each audio buffer.
        Processes audio in real-time and updates display.
        """
        nonlocal peak_level
        global stop_recording

        elapsed_time = time.time() - start_time

        # FIXED: Properly normalize peak level calculation for display
        if DTYPE == 'int16':
            # Convert int16 to normalized range for consistent display
            normalized_data = np.abs(indata.astype(np.float32)) / 32768.0
            current_peak = np.max(normalized_data)
        else:  # float32
            # Data already normalized, just get absolute max
            current_peak = np.max(np.abs(indata))

        # Track overall peak level for the entire recording
        peak_level = max(peak_level, current_peak)

        # Stop if maximum time reached
        if elapsed_time >= MAX_RECORDING_TIME:
            stop_recording = True

        # FIXED: Create level bar using normalized values (0-1 range)
        level_bar_length = max(0, min(20, int(current_peak * 20)))
        level_bar = '█' * level_bar_length
        sys.stdout.write(f"\rTime: {elapsed_time:.1f}s [{level_bar:<20}] Peak: {peak_level:.3f}")
        sys.stdout.flush()

        # Store this audio chunk
        frames.append(indata.copy())

        # Signal to stop the audio stream
        if stop_recording:
            raise sd.CallbackStop

    # Start the audio stream, then block here until the user presses spacebar or
    # the callback's max-duration guard sets stop_recording.
    try:
        with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype=DTYPE, callback=callback):
            wait_for_stop_key()
    except sd.CallbackStop:
        pass  # Normal termination

    print("\n\n⏹️  Recording stopped!")

    # Check if any audio was recorded
    if not frames:
        print("No audio recorded!")
        return

    # Combine all audio chunks into single array
    audio_data = np.concatenate(frames, axis=0).flatten()

    # Calculate recording statistics
    duration = len(audio_data) / RATE

    # Calculate RMS for the raw data (for display purposes)
    if DTYPE == 'int16':
        raw_rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
    else:  # float32
        raw_rms = np.sqrt(np.mean(audio_data**2))

    print(f"Duration: {duration:.2f}s")
    print(f"Raw RMS Level: {raw_rms:.4f}")
    print(f"Peak Level: {peak_level:.3f}")

    # Calculate normalized RMS for consistent threshold comparison
    normalized_rms = 1.0  # Default to skip normalization
    if DTYPE == 'int16':
        # Convert int16 to float range for RMS calculation
        normalized_rms = np.sqrt(np.mean((audio_data.astype(np.float32) / 32768.0)**2))
    elif DTYPE == 'float32':
        # Data already normalized
        normalized_rms = np.sqrt(np.mean(audio_data**2))
    else:
        print(f"Warning: Unsupported DTYPE {DTYPE}, skipping normalization")

    print(f"Normalized RMS: {normalized_rms:.4f}")

    # Apply normalization if audio is too quiet
    if normalized_rms < 0.05:  # 5% of full scale
        print("Audio seems quiet, normalizing...")
        if DTYPE == 'int16':
            audio_data = normalize_audio_int16(audio_data)
        elif DTYPE == 'float32':
            audio_data = normalize_audio_float32(audio_data)

    # Determine bit depth for display
    local_bits = ''
    if DTYPE == 'int16':
        local_bits = '16-'
    elif DTYPE == 'float32':
        local_bits = '32-'
    else:
        local_bits = 'unknown-'

    # Save the recording
    filename = "audio_recording.wav"
    wavfile.write(filename, RATE, audio_data)
    print(f"✅ Audio saved to {filename}")
    print(f"Format: {RATE}Hz, {CHANNELS} channel, {local_bits}bit")

if __name__ == "__main__":
    print("=== TTS Voice Recording Tool ===")
    record_audio()
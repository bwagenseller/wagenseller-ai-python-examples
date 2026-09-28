"""
Transcribes whatever is playing through the speakers, live, and saves it to a WAV file as well.

It is transcribe_client.py (the microphone client behind 'whisper.client') listening to the speaker output instead
of the microphone: the system loopback ("what you hear") is captured through PulseAudio / PipeWire, split into
chunks of speech by the same voice activity detection, sent to the WhisperX ASR server, and each transcription is
printed as it comes back. Everything that plays - speech, music, silence - is also written to a WAV file as it
arrives, so stopping with Ctrl+C keeps all of it.

Examples:
    # transcribe the speakers; saves ./speakers-YYYYmmdd-HHMMSS.wav
    python transcribe_output.py --host asr.host --port 65432

    # a radio show (no pauses): captions every 20 s by default; every 10 s here
    python transcribe_output.py --host asr.host --port 65432 --max_chunk_seconds 10

    # name the file, and show the language and confidence of each transcription
    python transcribe_output.py --host asr.host --port 65432 -o meeting.wav --diagnostics

    # listen to a particular output device instead of the default one (see 'pactl list short sources')
    python transcribe_output.py --speaker-source alsa_output.usb-Jabra_SPEAK_710-00.analog-stereo.monitor

The WAV is 16 kHz, mono, 16 bit - what the ASR server hears, not a hi-fi copy of the output.

Needs the 'media' conda environment (sounddevice, webrtcvad, numpy, soundfile) and PulseAudio's command-line tools:
sudo apt install pulseaudio-utils (for pactl / parec). Linux only.
"""

import os
import time
import wave

from transcribe_client import WhisperXClient, logger
from amadeo_utils.colored_text import ColoredText
from amadeo_utils.media_utils.audio_capture import PulseAudioCapture


class SpeakerOutputClient(WhisperXClient):
    """
    A WhisperXClient that listens to the speaker output instead of the microphone, and saves what it hears.

    Only the audio source changes - audio_frames() - so voice activity detection, sending to the server and printing
    the transcriptions are exactly the microphone client's.
    """

    AUDIO_SOURCE_LABEL = "speaker output"
    # Broadcast audio (a radio show) may never pause: send what has built up every 20 s, cut at a quiet moment -
    # steady captions, and under Whisper's 30 s window. Meetings still send at their natural pauses, well before this.
    MAX_CHUNK_SECONDS = 20.0

    def __init__(self, argsDict: dict):
        """
        :param argsDict: the microphone client's settings (see WhisperXClient.build_arg_parser), plus 'output' (the
                         WAV path) and 'speaker_source' (a PulseAudio monitor source, or None for the default output).
        """
        super().__init__(argsDict)
        self.output_path = argsDict['output']
        self.wav = None             # opened on the first chunk, so a source that cannot be found leaves no empty file
        self.saved_frames = 0

    def audio_frames(self):
        """
        Yields the speaker output one VAD frame at a time (16 kHz mono int16, as bytes), writing each frame to the WAV
        file first. parec does the conversion from whatever the output device runs at.

        :return: a generator of bytes, one VAD frame each.
        """
        capture = PulseAudioCapture(capture_speakers=True, speaker_source=self.args_dict.get('speaker_source'),
                                    sample_rate=WhisperXClient.SAMPLE_RATE, channels=1, verbose=False)
        frame_bytes = self.VAD_FRAME_SIZE * 2
        for chunk in capture.stream_raw(chunk_frames=self.VAD_FRAME_SIZE):
            if not self.is_recording:
                break
            if self.wav is None:
                self.open_wav()
            self.wav.writeframes(chunk)
            self.saved_frames += len(chunk) // 2
            # WebRTC VAD only takes whole frames; a short chunk only comes at the very end, when parec stops
            if len(chunk) == frame_bytes:
                yield chunk
        self.close_wav()

    def open_wav(self):
        """Opens the WAV file (16 kHz mono 16 bit) and says where it is going."""
        self.wav = wave.open(self.output_path, 'wb')
        self.wav.setnchannels(1)
        self.wav.setsampwidth(2)
        self.wav.setframerate(WhisperXClient.SAMPLE_RATE)
        logger.info(f"{ColoredText.BLUE_TEXT}Saving the speaker output to {self.output_path}{ColoredText.END_TEXT}")

    def close_wav(self):
        """Closes the WAV file, if one was opened, and reports its length. Safe to call more than once."""
        if self.wav is not None:
            self.wav.close()
            self.wav = None
            seconds = self.saved_frames / WhisperXClient.SAMPLE_RATE
            logger.info(f"{ColoredText.GREEN_TEXT}Saved {self.output_path} ({seconds:.1f} s).{ColoredText.END_TEXT}")

    def graceful_shutdown(self):
        """Closes the WAV file before the microphone client's shutdown (which ends the session and exits)."""
        self.close_wav()
        super().graceful_shutdown()

    @staticmethod
    def get_args_dict() -> dict:
        """
        Parses the command line: the microphone client's options plus --output and --speaker-source.

        Returns:
            dict: the settings; empty if the arguments were invalid or --help was used.
        """
        parser = WhisperXClient.build_arg_parser('Transcribe what is playing through the speakers, live, and save it to a WAV file.')
        parser.add_argument("-o", "--output", default=None,
                            help="The WAV file to save the speaker output to (default: speakers-YYYYmmdd-HHMMSS.wav in the current directory).")
        parser.set_defaults(max_chunk_seconds=SpeakerOutputClient.MAX_CHUNK_SECONDS)
        parser.add_argument("-ss", "--speaker-source", default=None,
                            help="A PulseAudio monitor source to listen to instead of the default output's (see 'pactl list short sources').")
        try:
            args = parser.parse_args()
        except SystemExit as e:
            if e.code == 0:
                print(f"{ColoredText.BLUE_TEXT}Thank you!{ColoredText.END_TEXT}")   # --help
            else:
                print(f"{ColoredText.RED_TEXT}transcribe_output: invalid arguments.{ColoredText.END_TEXT}")
            return {}

        output = args.output or os.path.join(os.getcwd(), time.strftime('speakers-%Y%m%d-%H%M%S.wav'))
        return {
            'host': args.host,
            'port': args.port,
            'vad_frame_duration': args.vad_frame_duration,
            'vad_aggressiveness': args.vad_aggressiveness,
            'silence_duration': args.silence_duration,
            'diagnostics': args.diagnostics,
            'max_chunk_seconds': args.max_chunk_seconds,
            'output': os.path.abspath(os.path.expanduser(output)),
            'speaker_source': args.speaker_source,
        }


if __name__ == "__main__":
    argsDict = SpeakerOutputClient.get_args_dict()
    if argsDict:
        WhisperXClient.configure_logging(argsDict['diagnostics'])
        SpeakerOutputClient(argsDict).run_client()

import argparse

from amadeo_utils.media_utils.audio_capture import PulseAudioCapture, DEFAULT_VAD_AGGRESSIVENESS, DEFAULT_VOLUME_THRESHOLD

"""
Records live audio to a WAV file: the speaker output (system loopback / "what you hear"),
the microphone, or both at once. Works with PulseAudio and PipeWire (via pipewire-pulse)
on Ubuntu.

This is a thin wrapper around amadeo_utils.media_utils.audio_capture.

System dependency: sudo apt install pulseaudio-utils   (for pactl/parec)
Python dependency: pip install -e ".[media]"           (from the repo root)

Source selection:
    (none)                      speakers only (the default)
    -m/--mic                    ALSO record the microphone (mixed in, or use --split)
    -m --no-speakers            microphone only
    --split                     with --mic: mic -> left channel, speakers -> right channel
                                (handy if you want to separate them later in post)

Stop conditions (whichever fires first wins; Ctrl+C always works):
    (none)                      record until Ctrl+C
    -d/--duration SECONDS       stop after a fixed time
    -ss/--silence-stop MS       stop after MS milliseconds of silence; by default this uses
                                speech detection (WebRTC VAD + volume), so use
                                --silence-mode any for music or other non-speech audio

Examples:
    python3 capture_audio.py                             # speakers until Ctrl+C
    python3 capture_audio.py -o out.wav -d 30            # speakers, 30 seconds
    python3 capture_audio.py -m                          # mic + speakers, mixed
    python3 capture_audio.py -m --split                  # mic -> L, speakers -> R
    python3 capture_audio.py -m --no-speakers            # mic only
    python3 capture_audio.py -ss 2000                    # stop after 2s of speech-silence
    python3 capture_audio.py -ss 3000 --silence-mode any # music: 3s of true quiet
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record speaker output (system loopback) and/or the mic to a WAV file.")

    source_group = parser.add_argument_group("source selection")
    source_group.add_argument("-m", "--mic", action="store_true", help="Also record the microphone (speakers are recorded unless --no-speakers).")
    source_group.add_argument("--no-speakers", action="store_true", help="Do not record the speaker loopback (use with --mic for mic-only capture).")
    source_group.add_argument("--split", action="store_true", help="With --mic: put mic on the left channel and speakers on the right instead of mixing.")
    source_group.add_argument("--mic-source", default=None, help="Explicit mic source name (default: the default input source).")
    source_group.add_argument("--speaker-source", default=None, help="Explicit monitor source name (default: the default sink's monitor).")

    parser.add_argument("-o", "--output", default=None, help="Output WAV path. Defaults to capture-YYYYmmdd-HHMMSS.wav")

    stop_group = parser.add_argument_group("stop conditions (Ctrl+C always works)")
    stop_group.add_argument("-d", "--duration", type=float, default=None, help="Stop after this many seconds.")
    stop_group.add_argument("-ss", "--silence-stop", type=int, default=None, help="Stop after this many milliseconds of continuous silence.")
    stop_group.add_argument("-sm", "--silence-mode", choices=["speech", "any"], default="speech", help="Silence detector: 'speech' = WebRTC VAD + volume (spoken audio), 'any' = volume only (music etc).")
    stop_group.add_argument("-va", "--vad-aggressiveness", type=int, default=DEFAULT_VAD_AGGRESSIVENESS, choices=[0, 1, 2, 3], help="WebRTC VAD aggressiveness; 3 blocks the most non-speech.")
    stop_group.add_argument("-vt", "--volume-threshold", type=float, default=DEFAULT_VOLUME_THRESHOLD, help="Normalized RMS (0-1) below which a chunk counts as quiet.")
    stop_group.add_argument("-w", "--wait-for-sound", type=int, default=None, help="With --silence-stop: give up if no sound at all arrives within this many milliseconds.")

    args = parser.parse_args()

    if args.no_speakers and not args.mic:
        parser.error("--no-speakers requires --mic (otherwise there is nothing to record).")
    if args.split and not args.mic:
        parser.error("--split only makes sense with --mic (it separates mic from speakers).")

    PulseAudioCapture(capture_speakers=not args.no_speakers, capture_mic=args.mic,
                      split_channels=args.split, mic_source=args.mic_source,
                      speaker_source=args.speaker_source).record(
        outfile=args.output,
        duration=args.duration,
        stop_after_silence_ms=args.silence_stop,
        silence_mode=args.silence_mode,
        vad_aggressiveness=args.vad_aggressiveness,
        volume_threshold=args.volume_threshold,
        max_wait_for_sound_ms=args.wait_for_sound,
    )


if __name__ == "__main__":
    main()

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import soundfile as sf

"""
This module captures live audio from PulseAudio / PipeWire (via pipewire-pulse) sources:
the system speakers (loopback / "what you hear"), the microphone, or both at once.
It was built for Ubuntu but should work on any Linux box running PulseAudio or PipeWire.

The PulseAudioCapture class is the core; three module-level convenience functions wrap it:

    capture_speakers - Records everything sent to the default output sink (speakers) to a WAV file.

    capture_mic - Records the default microphone to a WAV file.

    capture_mic_and_speakers - Records BOTH the microphone and the speaker loopback to a single
    WAV file, either mixed together or split (mic -> left channel, speakers -> right channel,
    handy if you want to separate them later in post).

All capture is done by spawning one `parec` process per source and reading raw s16le PCM from
its stdout. Both processes produce audio in real time at the same rate, so blocking reads of
equal size keep multiple sources aligned without any explicit clock synchronization.

(An earlier standalone version of the speaker capture used sounddevice/PortAudio with the
PULSE_SOURCE environment variable hack; that approach only works for a single source and the
env var must be set before PortAudio initializes, so everything was unified on parec instead.)

Stop conditions (any combination; recording stops on whichever fires first):

    Ctrl+C            - always available when running interactively.
    duration          - stop after a fixed number of seconds.
    silence           - stop after N milliseconds of silence. Two detector modes:
                          'speech' - hybrid WebRTC VAD + RMS volume (same approach as the
                                     conversational AI client): a chunk only counts as activity
                                     if it BOTH looks like speech AND is loud enough. Use this
                                     when the audio of interest is people talking.
                          'any'    - RMS volume only. Use this for non-speech audio (music,
                                     videos, game audio) where VAD would wrongly report silence.
    stop_event        - a threading.Event; set it from another thread to stop programmatically.
                        This is the hook other library code should use.

System dependencies (NOT pip packages):

    pactl / parec     - from the 'pulseaudio-utils' package: sudo apt install pulseaudio-utils

Python dependencies (covered by the [media] extra of amadeo-utils):

    numpy, soundfile, and (only if silence_mode='speech') webrtcvad-wheels
"""

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 2
CHUNK_MS = 30                  # 30ms chunks: one of the frame sizes WebRTC VAD accepts
BYTES_PER_SAMPLE = 2           # s16le

# Silence detection defaults (mirrors the conversational AI client)
DEFAULT_VAD_AGGRESSIVENESS = 2     # 0-3; higher = more aggressive at rejecting non-speech
DEFAULT_VOLUME_THRESHOLD = 0.03    # normalized RMS (0-1); below this a chunk is "quiet"


@dataclass
class CaptureResult:
    """What a recording session produced and why it ended."""
    outfile: str
    duration_sec: float            # seconds of audio actually written
    frames_written: int            # total PCM frames written to the WAV
    stop_reason: str               # 'ctrl-c' | 'duration' | 'silence' | 'stop_event' | 'stream-ended'


class PulseAudioCapture:
    """
    Captures one or two PulseAudio sources to a WAV file.

    Configure WHAT to record in the constructor; configure WHEN to stop in record().

    Examples:

        # Record the speakers until Ctrl+C
        PulseAudioCapture(capture_speakers=True).record()

        # Record mic + speakers, split channels, for at most 60 seconds
        PulseAudioCapture(capture_speakers=True, capture_mic=True,
                          split_channels=True).record(duration=60)

        # Record the speakers, stopping after 2s of silence (e.g. capturing TTS output)
        PulseAudioCapture(capture_speakers=True).record(stop_after_silence_ms=2000)

        # Programmatic stop from another thread
        stop = threading.Event()
        PulseAudioCapture(capture_speakers=True).record(stop_event=stop)
    """

    def __init__(self, capture_speakers: bool = True, capture_mic: bool = False,
                 split_channels: bool = False, speaker_source: str | None = None,
                 mic_source: str | None = None, sample_rate: int = DEFAULT_SAMPLE_RATE,
                 channels: int = DEFAULT_CHANNELS, verbose: bool = True):
        """
        :param capture_speakers: record the speaker loopback (default output sink's monitor)
        :param capture_mic: record the microphone (default input source)
        :param split_channels: only meaningful when capturing BOTH mic and speakers.
                               False: mix the two streams together (clipped to [-1, 1]).
                               True: downmix each to mono, mic -> left, speakers -> right.
        :param speaker_source: explicit monitor source name; None = resolve the default sink's monitor
        :param mic_source: explicit mic source name; None = resolve the default source
        :param sample_rate: capture sample rate in Hz
        :param channels: channels per captured stream (the WAV is always written with this count)
        :param verbose: print progress messages to stdout
        """
        if not capture_speakers and not capture_mic:
            raise ValueError("Nothing to capture: enable capture_speakers and/or capture_mic.")

        self.capture_speakers = capture_speakers
        self.capture_mic = capture_mic
        self.split_channels = split_channels and capture_speakers and capture_mic
        self.speaker_source = speaker_source
        self.mic_source = mic_source
        self.sample_rate = sample_rate
        self.channels = channels
        self.verbose = verbose

        self.chunk_frames = int(sample_rate * CHUNK_MS / 1000)
        self.chunk_bytes = self.chunk_frames * BYTES_PER_SAMPLE * channels

    # ------------------------------------------------------------------
    # PulseAudio source resolution
    # ------------------------------------------------------------------

    @staticmethod
    def pactl(*args: str) -> str:
        """Run a pactl command and return its stripped stdout."""
        return subprocess.run(
            ["pactl", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    @staticmethod
    def list_sources() -> set[str]:
        """Return the names of all available PulseAudio sources (mics AND monitors)."""
        return {line.split("\t")[1]
                for line in PulseAudioCapture.pactl("list", "short", "sources").splitlines()
                if "\t" in line}

    @staticmethod
    def resolve_default_monitor() -> str:
        """Return the monitor source name for the default output sink (the 'what you hear' tap)."""
        sink = PulseAudioCapture.pactl("get-default-sink")
        if not sink:
            raise RuntimeError("Could not determine default audio sink.")
        return f"{sink}.monitor"

    @staticmethod
    def resolve_default_mic() -> str:
        """Return the default input source name, refusing monitors (which would double-record)."""
        mic = PulseAudioCapture.pactl("get-default-source")
        if not mic:
            raise RuntimeError("Could not determine default audio source.")
        if mic.endswith(".monitor"):
            raise RuntimeError(
                f"Default source '{mic}' is a monitor, not a mic.\n"
                "Set a real input device as default (pactl set-default-source ...).")
        return mic

    def _resolve_sources(self) -> list[str]:
        """Build the ordered list of sources to capture: [mic][, monitor]."""
        sources = []
        if self.capture_mic:
            sources.append(self.mic_source or self.resolve_default_mic())
        if self.capture_speakers:
            sources.append(self.speaker_source or self.resolve_default_monitor())

        available = self.list_sources()
        for src in sources:
            if src not in available:
                raise RuntimeError(f"Source '{src}' not found.\nAvailable:\n"
                                   + "\n".join(sorted(available)))
        return sources

    # ------------------------------------------------------------------
    # parec stream handling
    # ------------------------------------------------------------------

    def _spawn_parec(self, source: str) -> subprocess.Popen:
        """Start a parec process emitting raw s16le PCM at our rate/channels."""
        return subprocess.Popen(
            ["parec",
             f"--device={source}",
             "--format=s16le",
             f"--rate={self.sample_rate}",
             f"--channels={self.channels}",
             "--latency-msec=50",
             "--raw"],
            stdout=subprocess.PIPE,
        )

    def _read_chunk(self, proc: subprocess.Popen) -> np.ndarray | None:
        """Read exactly one chunk; returns float32 array in [-1, 1] or None on EOF."""
        data = proc.stdout.read(self.chunk_bytes)
        if not data:
            return None
        if len(data) < self.chunk_bytes:                 # pad final short read
            data += b"\x00" * (self.chunk_bytes - len(data))
        pcm = np.frombuffer(data, dtype=np.int16).reshape(-1, self.channels)
        return pcm.astype(np.float32) / 32768.0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, outfile: str | None = None, duration: float | None = None,
               stop_after_silence_ms: int | None = None, silence_mode: str = "speech",
               vad_aggressiveness: int = DEFAULT_VAD_AGGRESSIVENESS,
               volume_threshold: float = DEFAULT_VOLUME_THRESHOLD,
               require_sound_first: bool = True, max_wait_for_sound_ms: int | None = None,
               stop_event: threading.Event | None = None) -> CaptureResult:
        """
        Record until a stop condition fires; returns a CaptureResult.

        :param outfile: output WAV path; None = capture-YYYYmmdd-HHMMSS.wav in the cwd
        :param duration: stop after this many seconds (None = no time limit)
        :param stop_after_silence_ms: stop after this many milliseconds of continuous silence
                                      (None = silence detection disabled)
        :param silence_mode: 'speech' (hybrid WebRTC VAD + RMS - for spoken audio) or
                             'any' (RMS only - for music / arbitrary audio)
        :param vad_aggressiveness: WebRTC VAD aggressiveness 0-3 (silence_mode='speech' only)
        :param volume_threshold: normalized RMS (0-1) below which a chunk counts as quiet
        :param require_sound_first: if True, the silence countdown only starts AFTER sound has
                                    been detected at least once (so the recording does not stop
                                    before anything has played/been said)
        :param max_wait_for_sound_ms: with require_sound_first, give up if no sound at all
                                      arrives within this window (None = wait forever)
        :param stop_event: a threading.Event another thread can set to end the recording
        :return: CaptureResult with the output path, duration, and why the recording stopped
        """
        if silence_mode not in ("speech", "any"):
            raise ValueError("silence_mode must be 'speech' or 'any'")

        # Lazy import: webrtcvad is only required when speech-based silence detection is used
        vad = None
        if stop_after_silence_ms is not None and silence_mode == "speech":
            import webrtcvad
            vad = webrtcvad.Vad(vad_aggressiveness)

        if outfile is None:
            outfile = f"capture-{datetime.now():%Y%m%d-%H%M%S}.wav"

        sources = self._resolve_sources()
        procs = [self._spawn_parec(src) for src in sources]

        if self.verbose:
            for label, src in zip(self._source_labels(), sources):
                print(f"{label:<16} {src}")
            print(f"{'Output file:':<16} {outfile}")
            if self.capture_mic and self.capture_speakers:
                print(f"{'Mode:':<16} {'split channels' if self.split_channels else 'mixed'}")
            print("Recording..." + ("" if duration else " press Ctrl+C to stop."))

        # Silence-tracking state
        silent_chunks = 0
        silent_chunks_threshold = (int(stop_after_silence_ms / CHUNK_MS)
                                   if stop_after_silence_ms is not None else None)
        heard_sound = False
        max_wait_chunks = (int(max_wait_for_sound_ms / CHUNK_MS)
                           if max_wait_for_sound_ms is not None else None)
        chunks_read = 0

        stop_reason = "stream-ended"
        frames_written = 0
        start = time.monotonic()

        try:
            with sf.SoundFile(outfile, mode="w", samplerate=self.sample_rate,
                              channels=self.channels, subtype="PCM_16") as wav:
                while True:
                    chunks = [self._read_chunk(p) for p in procs]
                    if any(c is None for c in chunks):
                        print("\nA capture stream ended unexpectedly.", file=sys.stderr)
                        break

                    frame = self._combine(chunks)
                    wav.write(frame)
                    frames_written += len(frame)
                    chunks_read += 1

                    if stop_event is not None and stop_event.is_set():
                        stop_reason = "stop_event"
                        break

                    if duration and (time.monotonic() - start) >= duration:
                        stop_reason = "duration"
                        break

                    if silent_chunks_threshold is not None:
                        if self._chunk_has_sound(frame, vad, volume_threshold):
                            heard_sound = True
                            silent_chunks = 0
                        elif heard_sound or not require_sound_first:
                            silent_chunks += 1
                            if silent_chunks >= silent_chunks_threshold:
                                stop_reason = "silence"
                                break
                        elif max_wait_chunks is not None and chunks_read >= max_wait_chunks:
                            # Never heard anything and we are done waiting
                            stop_reason = "silence"
                            break
        except KeyboardInterrupt:
            stop_reason = "ctrl-c"
            if self.verbose:
                print("\nStopped.")
        finally:
            for proc in procs:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()

        duration_sec = frames_written / self.sample_rate
        if self.verbose:
            print(f"Saved: {outfile} ({duration_sec:.1f}s, stopped by: {stop_reason})")

        return CaptureResult(outfile=outfile, duration_sec=duration_sec,
                             frames_written=frames_written, stop_reason=stop_reason)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _source_labels(self) -> list[str]:
        labels = []
        if self.capture_mic:
            labels.append("Mic source:")
        if self.capture_speakers:
            labels.append("Speaker monitor:")
        return labels

    def _combine(self, chunks: list[np.ndarray]) -> np.ndarray:
        """Combine per-source chunks into the frame that gets written to the WAV."""
        if len(chunks) == 1:
            return chunks[0]
        mic, spk = chunks                                # order fixed by _resolve_sources
        if self.split_channels:
            # Downmix each to mono, then mic -> L, speakers -> R
            return np.column_stack((mic.mean(axis=1), spk.mean(axis=1)))
        return np.clip(mic + spk, -1.0, 1.0)

    def _chunk_has_sound(self, frame: np.ndarray, vad, volume_threshold: float) -> bool:
        """
        Decide whether a (frames, channels) float32 chunk contains activity.

        RMS volume is always checked; when a VAD is supplied the chunk must ALSO look like
        speech (the hybrid approach: VAD alone can trigger on speech-like background noise,
        volume alone on door slams / coughs).
        """
        mono = frame.mean(axis=1)
        rms = float(np.sqrt(np.mean(mono ** 2)))
        if rms <= volume_threshold:
            return False
        if vad is None:
            return True
        mono_int16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
        return vad.is_speech(mono_int16.tobytes(), self.sample_rate)


# ----------------------------------------------------------------------
# Convenience wrappers
# ----------------------------------------------------------------------

def capture_speakers(outfile: str | None = None, **kwargs) -> CaptureResult:
    """Record the speaker loopback. kwargs are passed to PulseAudioCapture.record()."""
    return PulseAudioCapture(capture_speakers=True, capture_mic=False).record(outfile, **kwargs)


def capture_mic(outfile: str | None = None, **kwargs) -> CaptureResult:
    """Record the default microphone. kwargs are passed to PulseAudioCapture.record()."""
    return PulseAudioCapture(capture_speakers=False, capture_mic=True).record(outfile, **kwargs)


def capture_mic_and_speakers(outfile: str | None = None, split_channels: bool = False,
                             **kwargs) -> CaptureResult:
    """Record mic + speakers to one WAV. kwargs are passed to PulseAudioCapture.record()."""
    return PulseAudioCapture(capture_speakers=True, capture_mic=True,
                             split_channels=split_channels).record(outfile, **kwargs)

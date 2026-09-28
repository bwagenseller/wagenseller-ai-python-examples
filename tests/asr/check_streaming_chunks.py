#!/usr/bin/env python3
"""The streaming clients' length limit (--max_chunk_seconds): where a chunk that never pauses is cut.

Why
---
The live transcription clients (transcribe_client.py for the microphone, transcribe_output.py for the speakers) send
a chunk of speech to the ASR server when the speaker pauses. A radio show never pauses, so a chunk is also sent once it
reaches --max_chunk_seconds - cut at the quietest recent moment, so the cut rarely splits a word. This pins that choice.

What it proves (quietest_cut, with made-up audio - no server, no microphone)
---------------------------------------------------------------------------
1. The cut lands at the end of the quietest frame among the most recent ones.
2. Only the last search_frames frames are considered - an older, quieter frame is ignored.
3. The cut never leaves nothing to send (never before the first frame), and never goes past the buffer.
4. Ties go to the earliest quiet frame in the window; a one-frame buffer is sent whole.
5. The option exists on both clients: off (0) by default for the microphone, 20 s for the speakers.

Usage:  python check_streaming_chunks.py      (the media env: sounddevice, webrtcvad and numpy, as the clients import)
"""
import os
import sys

import numpy as np

# testlib finds the library under test (tests/README.md); the clients are scripts, found by their folder.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testlib import settings  # noqa: E402
settings.use_src()
sys.path.insert(0, os.path.join(os.path.dirname(settings.SRC), 'scripts', 'ai', 'asr', 'whisperx', 'streaming'))

import logging  # noqa: E402
logging.disable(logging.WARNING)
from transcribe_client import quietest_cut, WhisperXClient  # noqa: E402
from transcribe_output import SpeakerOutputClient  # noqa: E402

failures = []
FRAME = 480                 # samples in a 30 ms frame at 16 kHz
FRAME_BYTES = FRAME * 2


def check(label, got, expected):
    """Records one comparison and prints PASS / FAIL."""
    if got == expected:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}: got {got!r}, expected {expected!r}")
        failures.append(label)


def audio(levels):
    """A buffer of frames, each a steady tone at the given amplitude (0 = silence)."""
    t = np.arange(FRAME)
    return b''.join((np.sin(t / 5.0) * level).astype(np.int16).tobytes() for level in levels)


loud = 8000
buffer = audio([loud] * 10 + [300] + [loud] * 5)          # a quiet dip at frame 10 of 16
check("the cut is at the end of the quietest recent frame", quietest_cut(buffer, FRAME_BYTES, 16), 11 * FRAME_BYTES)

buffer = audio([0] + [loud] * 10 + [2000] + [loud] * 5)   # silence at frame 0, a dip at frame 11
check("only the search window counts: an older, quieter frame is ignored", quietest_cut(buffer, FRAME_BYTES, 8), 12 * FRAME_BYTES)
check("never before the first frame (something is always sent)", quietest_cut(buffer, FRAME_BYTES, 100) >= FRAME_BYTES, True)

buffer = audio([loud] * 6)
check("all equally loud: the earliest frame in the window", quietest_cut(buffer, FRAME_BYTES, 3), 4 * FRAME_BYTES)
check("the quietest frame last: the whole buffer is sent", quietest_cut(audio([loud] * 5 + [0]), FRAME_BYTES, 3), 6 * FRAME_BYTES)
check("a one-frame buffer is sent whole", quietest_cut(audio([loud]), FRAME_BYTES, 3), FRAME_BYTES)

check("microphone client: no length limit by default", WhisperXClient.MAX_CHUNK_SECONDS, 0.0)
check("speaker client: 20 s by default", SpeakerOutputClient.MAX_CHUNK_SECONDS, 20.0)

print(f"\n{len(failures)} failure(s)")
sys.exit(len(failures))

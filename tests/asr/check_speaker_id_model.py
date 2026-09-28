#!/usr/bin/env python3
"""CS-23: speaker identification end to end, through the WhisperX ASR server's request handler, with a real model.

Why
---
check_speaker_id.py pins the matching rules with fake vectors. This proves the real pieces fit: the ASR server's
'speaker_embedding' command (what enroll_voice.py uses), the profiles it produces, and a 'transcribe' request with
voice_recognition on returning the right speaker next to the transcription - and a stranger coming back unknown.

The "people" are Kokoro's stock voices (make_synthetic_voices.py), never a real person's.

What it proves (AmadeoWhisperX on the CPU, Whisper 'tiny', pyannote's wespeaker embedding model)
------------------------------------------------------------------------------------------------
1. 'speaker_embedding' returns an embedding and the model's id; the same clip sent at 24 kHz (resampled by the
   server) and at 16 kHz embeds to nearly the same vector.
2. Three voices enrolled from 4 clips each (location 'studio'); their other clips are each identified, by the
   location step, with the transcription still there.
3. The held-back fourth voice is an unrecognized voice.
4. From a location nobody enrolled at, the enrolled voices are still identified (the 'everyone' step).
5. Speech under min_seconds is too_short (no speaker), however much trailing silence the chunk has; without voice_recognition no speaker fields are returned.
6. A profile added while the server runs is picked up without a restart.
7. Field clips: with save_unknown_field_clips on, exactly the unrecognized voices' speech is saved, as 16 kHz mono
   16 bit WAVs under <field dir>/<location>/unrecognized/ (too-short speech and recognition-off requests are not);
   known voices are saved under their profile name only once save_known_field_clips is on in the config AND the
   request (the client opting its microphone in) - a client that doesn't opt in is never recorded; a location_id that
   tries to climb out of the directory stays inside it; clips past the retention period are deleted.
8. The server's --json loader warns about a key it doesn't know - with a hint when it is a speaker_id setting put
   at the top level - and '_' notes pass silently.

Needs: the 'stt' env (local setting stt_python), the rendered voices (speaker_id_voices_dir), the Whisper 'tiny'
and alignment models, and the embedding model (downloaded on first use; HF_TOKEN if it is gated for you).

Usage:  python check_speaker_id_model.py      (stt env; runs on the CPU)
"""
import json
import os
import stat
import sys
import tempfile
import time
import wave

# The CPU is plenty for Whisper 'tiny' and the embedding model, and keeps this off a GPU another service holds
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

# testlib finds the library under test (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testlib import settings  # noqa: E402
settings.use_src()

import logging  # noqa: E402
import numpy as np  # noqa: E402

from amadeo_utils.ai.asr import speaker_id as sid  # noqa: E402
from amadeo_utils.ai.asr.speaker_embedder import resample  # noqa: E402
from amadeo_utils.ai.asr.whisperx import AmadeoWhisperX  # noqa: E402

logging.disable(logging.WARNING)

VOICES_DIR = settings.get('speaker_id_voices_dir')
ENROLLED = ['af_heart', 'am_adam', 'bf_emma']
STRANGER = 'am_michael'
ENROLL_CLIPS = 4            # clips 00-03 enroll; the rest are identified

failures = []


def check(label, got, expected):
    """Records one comparison and prints PASS / FAIL."""
    if got == expected:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}: got {got!r}, expected {expected!r}")
        failures.append(label)


def clips(voice):
    """The voice's WAV files, in order."""
    folder = os.path.join(VOICES_DIR, voice)
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.wav')]


def read_pcm(path):
    """(16 bit mono PCM bytes, sample rate) of a WAV."""
    with wave.open(path, 'rb') as w:
        return w.readframes(w.getnframes()), w.getframerate()


def pcm_16k(path, seconds=None):
    """A WAV as the conversational client sends it: 16 kHz, 16 bit mono PCM; optionally only its first seconds."""
    pcm, rate = read_pcm(path)
    audio = resample(np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0, rate)
    if seconds is not None:
        audio = audio[:int(seconds * 16000)]
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


def display(voice):
    """The display name enroll gives a voice (the file-safe name in title case)."""
    return sid.VoiceProfile(name=voice).display_name


with tempfile.TemporaryDirectory() as workdir:
    profiles_dir = os.path.join(workdir, 'profiles')
    field_dir = os.path.join(workdir, 'field')
    asr = AmadeoWhisperX({'model': 'tiny', 'language_code': 'en', 'gpu': 0, 'combined_confidence_cutoff': 0.0,
                          'speaker_id': {'profiles_dir': profiles_dir, 'field_samples_dir': field_dir,
                                         'save_unknown_field_clips': True}})

    def embed(pcm, rate):
        """The server's embedding of a clip."""
        response, _ = asr.handle_client_request({'command': 'speaker_embedding', 'sample_rate': rate}, pcm)
        return response

    def transcribe(pcm, **fields):
        """A transcribe request, as the conversational server sends it - from a client that lets its microphone be
        recorded as field clips (the server's config decides what it actually keeps), unless told otherwise."""
        fields = dict({'save_known_field_clips': True, 'save_unknown_field_clips': True}, **fields)
        response, _ = asr.handle_client_request(dict({'command': 'transcribe'}, **fields), pcm)
        return response

    # 1. embeddings
    first = clips(ENROLLED[0])[0]
    r24 = embed(*read_pcm(first))
    check("speaker_embedding: success, the model's id, a vector",
          (r24['success'], r24['model'], len(r24['embedding']) > 100), (True, sid.DEFAULT_EMBEDDING_MODEL, True))
    r16 = embed(pcm_16k(first), 16000)
    check("24 kHz resampled by the server ~ the same clip sent at 16 kHz",
          sid.cosine(r24['embedding'], r16['embedding']) > 0.95, True)

    # 2. enroll three voices (enroll_voice.py's path: embeddings from the server, one profile per person)
    for voice in ENROLLED:
        profile = sid.VoiceProfile(name=voice, model=r24['model'])
        profile.add('studio', [embed(*read_pcm(path))['embedding'] for path in clips(voice)[:ENROLL_CLIPS]])
        sid.save_profile(profiles_dir, profile)

    for voice in ENROLLED:
        results = [transcribe(pcm_16k(path), voice_recognition=True, location_id='studio') for path in clips(voice)[ENROLL_CLIPS:]]
        check(f"{voice}: every held-out clip identified at its location",
              [(r['speaker'], r['speaker_status'], r['speaker_step']) for r in results],
              [(display(voice), 'identified', 'location')] * len(results))
        check(f"{voice}: the transcription is still returned", all(r.get('transcription') for r in results), True)
        scores = [r['speaker_score'] for r in results]
        print(f"      scores {', '.join(f'{s:.2f}' for s in scores)}")

    # 3. the stranger
    results = [transcribe(pcm_16k(path), voice_recognition=True, location_id='studio') for path in clips(STRANGER)]
    check(f"{STRANGER} (never enrolled): an unrecognized voice every time",
          [(r['speaker'], r['speaker_status']) for r in results], [(sid.UNRECOGNIZED_SPEAKER, 'unknown')] * len(results))
    best = [r['speaker_score'] for r in results]
    print(f"      best scores {', '.join(f'{s:.2f}' for s in best)}")
    # ...and once more from a client that does NOT let its microphone be recorded: section 7 checks it was not saved
    transcribe(pcm_16k(clips(STRANGER)[0]), voice_recognition=True, location_id='studio', save_unknown_field_clips=False)

    # 4. another location
    r = transcribe(pcm_16k(clips(ENROLLED[1])[-1]), voice_recognition=True, location_id='garage')
    check("from a location nobody enrolled at: identified by the 'everyone' step",
          (r['speaker'], r['speaker_step']), (display(ENROLLED[1]), 'everyone'))

    # 5. too short / off
    # 0.8 s: under min_seconds (1.0) but long enough for Whisper to transcribe (0.5 s is usually blank to it, and a
    # blank transcription carries no speaker fields at all)
    r = transcribe(pcm_16k(clips(ENROLLED[0])[0], seconds=0.8), voice_recognition=True, location_id='studio')
    check("0.8 s of speech: transcribed, but too_short to judge (no speaker)",
          (r.get('type'), r.get('speaker_status'), r.get('speaker')), ('transcription', 'too_short', ''))
    # The client's chunks end with the silence that told it the speaker had stopped: the speech is what counts
    silence = b'\x00\x00' * int(1.2 * 16000)
    r = transcribe(pcm_16k(clips(ENROLLED[0])[0], seconds=0.8) + silence, voice_recognition=True, location_id='studio')
    check("0.8 s of speech + 1.2 s of trailing silence: still too_short (the silence does not count)",
          r.get('speaker_status'), 'too_short')
    r = transcribe(pcm_16k(clips(ENROLLED[0])[-1]) + silence, voice_recognition=True, location_id='studio')
    check("a full clip + trailing silence: still identified", r.get('speaker'), display(ENROLLED[0]))
    r = transcribe(pcm_16k(clips(ENROLLED[0])[-1]))
    check("voice_recognition off: no speaker fields", [k for k in r if k.startswith('speaker')], [])

    # 6. enroll the stranger while the server runs
    profile = sid.VoiceProfile(name=STRANGER, model=r24['model'])
    profile.add('studio', [embed(*read_pcm(path))['embedding'] for path in clips(STRANGER)[:ENROLL_CLIPS]])
    sid.save_profile(profiles_dir, profile)
    r = transcribe(pcm_16k(clips(STRANGER)[-1]), voice_recognition=True, location_id='studio')
    check("a profile added while running is picked up", r['speaker'], display(STRANGER))

    # 7. field clips
    def field_files(*parts):
        """The .wav files saved under <field dir>/<parts...>."""
        folder = os.path.join(field_dir, *parts)
        return sorted(os.path.join(folder, f) for f in os.listdir(folder)) if os.path.isdir(folder) else []

    unknown_clips = field_files('studio', sid.FIELD_UNRECOGNIZED_FOLDER)
    check("field clips: each unrecognized chunk saved once, under <location>/unrecognized - except the one from a "
          "client that didn't opt in", len(unknown_clips), len(clips(STRANGER)))
    check("field clips: nothing saved for known voices (off), too-short speech or recognition off",
          sorted(os.listdir(field_dir)), ['studio'])
    check("field clips: ...and nothing else at that location", os.listdir(os.path.join(field_dir, 'studio')),
          [sid.FIELD_UNRECOGNIZED_FOLDER])
    with wave.open(unknown_clips[0], 'rb') as w:
        check("a field clip is a 16 kHz mono 16 bit WAV of the speech",
              (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes() > 16000), (16000, 1, 2, True))
    check("a field clip is readable by owner and group only", stat.S_IMODE(os.stat(unknown_clips[0]).st_mode) & 0o007, 0)

    asr.speaker_settings.save_known_field_clips = True
    transcribe(pcm_16k(clips(ENROLLED[0])[-1]), voice_recognition=True, location_id='studio', save_known_field_clips=False)
    check("known voices: the server allows it but this client doesn't - not saved", len(field_files('studio', ENROLLED[0])), 0)
    transcribe(pcm_16k(clips(ENROLLED[0])[-1]), voice_recognition=True, location_id='studio')
    check("with save_known_field_clips on, a known voice is saved under its profile name",
          len(field_files('studio', ENROLLED[0])), 1)
    transcribe(pcm_16k(clips(ENROLLED[0])[-1]), voice_recognition=True, location_id='../../escape')
    check("a location_id cannot climb out of the field clips directory",
          (os.path.exists(os.path.join(workdir, 'escape')), len(field_files('escape', ENROLLED[0]))), (False, 1))

    old = unknown_clips[0]
    os.utime(old, (time.time() - 91 * 86400,) * 2)
    asr.field_expired_at = 0.0          # the hourly throttle: pretend the last expiry run was long ago
    asr._expire_field_clips()
    check("clips older than the retention period (90 days) are deleted, newer ones kept",
          (os.path.exists(old), len(field_files('studio', sid.FIELD_UNRECOGNIZED_FOLDER))), (False, len(clips(STRANGER)) - 1))

# 8. unknown settings in the server's --json config
class Captured(logging.Handler):
    """Collects the warning messages logged while it is attached."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


with tempfile.TemporaryDirectory() as workdir:
    path = os.path.join(workdir, 'server.json')
    with open(path, 'w') as f:
        json.dump({'_comment': 'notes', 'port': 1, 'prot': 2, 'profiles_dir': '/p', 'speaker_id': {'profiles_dir': '/p'}}, f)
    captured = Captured()
    whisperx_logger = logging.getLogger('amadeo_utils.ai.asr.whisperx')
    whisperx_logger.addHandler(captured)
    logging.disable(logging.NOTSET)     # this check reads the warnings the rest of the script keeps quiet
    try:
        loaded = AmadeoWhisperX.load_json_config(path)
    finally:
        logging.disable(logging.WARNING)
        whisperx_logger.removeHandler(captured)
    check("server config: an unknown key is warned about, and the rest still loads",
          (loaded['port'], any("'prot'" in w and 'ignored.' in w for w in captured.messages)), (1, True))
    check("...a speaker_id setting at the top level gets a hint to move it",
          any("'profiles_dir'" in w and "belongs inside the 'speaker_id' block" in w for w in captured.messages), True)
    check("...and '_' notes pass silently", any("_comment" in w for w in captured.messages), False)

print(f"\n{len(failures)} failure(s)")
sys.exit(len(failures))

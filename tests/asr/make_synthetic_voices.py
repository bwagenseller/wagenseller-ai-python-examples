#!/usr/bin/env python3
"""CS-23 test helper: renders made-up "people" for the speaker-ID model check, from Kokoro's STOCK voices.

The speaker-ID tests must never use a real person's voice (voices are biometric data, and this folder is public),
so the test people are Kokoro's built-in voices - never a clone of anyone. Run once per machine, in the 'kokoro'
conda environment; point the 'speaker_id_voices_dir' local setting at the output directory.

Output: <out>/<voice>/<nn>.wav - 16 bit mono WAVs at Kokoro's 24 kHz, one per sentence, SENTENCES_PER_VOICE each.

Usage:  python make_synthetic_voices.py OUT_DIR      (kokoro env; downloads the stock voices on first use)
"""
import os
import sys
import wave

import numpy as np
from kokoro import KPipeline

# Two women, two men, American and British - check_speaker_id_model.py enrolls some and holds one back as a stranger
VOICES = ['af_heart', 'am_adam', 'bf_emma', 'am_michael']
SAMPLE_RATE = 24000
SENTENCES = [
    "The quick brown fox jumps over the lazy dog, and then it naps in the warm afternoon sun.",
    "Could you remind me to pick up milk, eggs and bread on the way home tomorrow?",
    "I think the weather this weekend is supposed to be cold and rainy, so let's plan something indoors.",
    "My favourite movie has a surprising ending that nobody in the theatre saw coming.",
    "Please turn the living room lights down a little and play some quiet music.",
    "We should repaint the fence before winter, maybe a darker shade of green this time.",
    "Seven hundred and forty two people signed up for the charity run last year.",
    "What would you cook for dinner if you only had rice, onions, garlic and a can of beans?",
]
SENTENCES_PER_VOICE = len(SENTENCES)


def main() -> int:
    """Renders every sentence in every voice. Returns the exit code."""
    if len(sys.argv) != 2:
        print(__doc__)
        return 64
    out_dir = sys.argv[1]
    american, british = KPipeline(lang_code='a'), KPipeline(lang_code='b')
    for voice in VOICES:
        pipeline = british if voice.startswith('b') else american
        os.makedirs(os.path.join(out_dir, voice), exist_ok=True)
        for number, sentence in enumerate(SENTENCES):
            audio = np.concatenate([np.asarray(chunk.audio, dtype=np.float32) for chunk in pipeline(sentence, voice=voice)])
            path = os.path.join(out_dir, voice, f"{number:02d}.wav")
            with wave.open(path, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
            print(f"{path}  {len(audio) / SAMPLE_RATE:.1f} s")
    return 0


if __name__ == '__main__':
    sys.exit(main())

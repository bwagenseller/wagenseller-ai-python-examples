# Wagenseller AI — Python Examples

A homegrown Python toolkit built around **local, self-hosted AI** — speech-to-text,
local LLMs, and text-to-speech — plus the client/server, media, and crypto plumbing
needed to wire them into real applications.

Everything here runs on your own hardware (no cloud APIs): WhisperX for ASR,
`llama.cpp` for LLM inference, and F5/Kokoro for TTS, glued together with a
threaded socket server.

```
amadeo_utils/   ← the reusable library (installable package)
scripts/        ← runnable scripts that drive the library
```

## Highlight: a full voice-conversation pipeline

The flagship example (`scripts/ai/combos/conversational_ai/`) is an end-to-end,
real-time **voice → voice** loop:

```
🎤 mic ─▶ VAD (speech detection) ─▶ socket/JSON protocol ─▶ ┌─────────────────────┐
                                                            │  ASR  (WhisperX)    │
                                                            │   ▼                 │
                                                            │  LLM  (llama.cpp)   │
                                                            │   ▼                 │
                                                            │  TTS  (F5 / Kokoro) │
                                                            └─────────┬───────────┘
🔊 speaker ◀──────────────── audio response ◀───────────────────────┘
```

Notable engineering details:

- **Custom socket protocol** — every message is length-prefixed with a 4-byte header,
  so the server always knows exactly how many bytes to read (no partial-message bugs).
- **Concurrency with a single GPU** — client connections are handled on their own
  threads, but GPU-bound work (transcription, generation) is funneled through a
  **lock**, so the model is never hit by two requests at once; requests take turns
  instead of crashing it.
- **Client-side Voice Activity Detection** (`webrtcvad`) decides when you've stopped
  talking and a chunk is ready to send — no push-to-talk needed.
- **Session management** — each client gets a UUID session; resources are cleaned up
  on disconnect.

### Several agents, and knowing who is talking

On top of the basic loop, the pipeline handles a whole household:

- **Multiple agents with wake words** — each agent has its own name, system prompt, voice and
  memory ("Hey Rose…", "Frasier, what do you think?"). A wake word only counts near the start of
  what was said, so talking *about* an agent doesn't wake it. Once awake, follow-ups within a
  short window need no wake word.
- **Switching agents mid-conversation** — the agent you switch to gets a short note of what it
  missed ("Kevin said to Rose: … Rose replied: …"). When several agents are named and punctuation
  can't tell who was addressed, the LLM is asked to decide.
- **Shared memory per household** — clients in different rooms that share a `user_id` share
  one conversation with each agent, so every spot hears what was said at the others.
- **Voice recognition** — the ASR server also works out *who* spoke from their voice (speaker
  identification with pyannote's embedding model, in the same pass as the transcription). Agents
  see `[Kevin, to Rose]: …` instead of assuming it's always the same person. Voice profiles are
  kept per microphone location, since a mic and its room change how a voice sounds.
- **Per-agent voice gates** — an agent can refuse voices it doesn't know
  (`allow_unknown_speakers`) or answer only certain people (`allowed_speakers`), e.g. keeping a
  grown-up character away from the kids. (Voice matching is a convenience, not security.)
- **Field clips** — optionally, the ASR server keeps the speech it judged, filed by location and
  speaker, as raw enrollment material from the real microphones. Recording is opt-in per spot.

> ### Why I built it: "talk to Santa"
> The original motivation was letting my kids have a live, spoken back-and-forth
> conversation with Santa Claus — speak into the mic, hear Santa answer in a custom
> cloned voice. That turned into this general-purpose, swappable ASR→LLM→TTS pipeline.

## Install

The library is `pip`-installable. An **editable install** lets the `scripts/`
import `amadeo_utils` from anywhere with no `PYTHONPATH` juggling:

```bash
# core library only (client/server framework + file encryption)
pip install -e .

# one extra per component — each in its OWN environment (their pins conflict):
pip install -e ".[asr]"         # WhisperX speech-to-text
pip install -e ".[llm]"         # llama.cpp chat / streaming / vector-DB
pip install -e ".[tts-f5]"      # F5-TTS  (Python 3.12 only)
pip install -e ".[tts-kokoro]"  # Kokoro TTS
pip install -e ".[client]"      # mic + playback client / conversational-ai orchestrator
pip install -e ".[media]"       # audio extraction / recording / noise reduction
pip install -e ".[infinite-campus]"  # Playwright (Infinite Campus tool)
```

> **Each extra mirrors a dedicated environment and is meant to be installed alone.**
> Their `numpy`/`torch` pins intentionally differ (e.g. `[tts-f5]` needs numpy 1.x while
> `[asr]` needs numpy 2.3) and will not co-resolve in a single environment. The heavy
> extras (`asr`, `llm`, `tts-f5`, `tts-kokoro`) pull in CUDA builds of `torch` /
> `llama-cpp-python` — install them on a machine matched to your GPU. For a specific CUDA
> build, add the PyTorch index, e.g. `--index-url https://download.pytorch.org/whl/cu128`.
> Tested on Python 3.12, except `[llm]` and `[infinite-campus]` (Python 3.13); `[tts-f5]` is 3.12-only.

**System dependency — `ffmpeg`.** The `[asr]` (WhisperX) and `[media]` components shell out to
`ffmpeg` for audio decoding/extraction, and it is **not** a pip package. Install it via your OS
package manager before using those extras, e.g. `sudo apt install ffmpeg` (Debian/Ubuntu),
`brew install ffmpeg` (macOS), or `conda install -c conda-forge ffmpeg`. The Playwright
(`[infinite-campus]`) extra also needs its browser binaries: `playwright install` after `pip install`.

**System dependency — `pulseaudio-utils`.** The live audio-capture component of `[media]`
(`media_utils/audio_capture.py` and the `scripts/media/capture_*.py` scripts) shells out to
`pactl`/`parec`, so it is Linux-only (PulseAudio, or PipeWire via `pipewire-pulse`):
`sudo apt install pulseaudio-utils` (Debian/Ubuntu).

## Running the voice pipeline

Five processes, each in its own environment (see Install), each started with a JSON config
(`--json <file>`). Start them in this order:

| # | Process | Script | Extra / env |
| --- | --- | --- | --- |
| 1 | LLM server (role-play, knowledge-base or tool-calling agent) | `scripts/ai/llm/llama/llama_stream/role_play_server.py` (or `knowledge_base_server.py`, `amadeo_agent_server.py`) | `[llm]` |
| 2 | TTS server | `scripts/ai/tts/kokoro/kokoro-simple-server.py` (or `scripts/ai/tts/f5-tts/f5-simple-server.py`) | `[tts-kokoro]` / `[tts-f5]` |
| 3 | ASR server | `scripts/ai/asr/whisperx/streaming/transcribe_server.py` | `[asr]` |
| 4 | Conversational server — routes each request ASR → LLM → TTS | `scripts/ai/combos/conversational_ai/conversational-ai-server.py` | `[client]` |
| 5 | Client — mic, voice activity detection, playback; one per room | `scripts/ai/combos/conversational_ai/conversational-ai-client.py` | `[client]` |

```bash
python scripts/ai/asr/whisperx/streaming/transcribe_server.py --json /path/to/asr-server.json
python scripts/ai/combos/conversational_ai/conversational-ai-server.py --json /path/to/conversational-server.json
python scripts/ai/combos/conversational_ai/conversational-ai-client.py --json /path/to/office-client.json
```

Where the settings are documented:

- **LLM servers:** `scripts/ai/llm/llama/llama_stream/example_*_server_config.json`.
- **ASR server:** `scripts/ai/asr/whisperx/streaming/example_transcribe_server_config.json`. Its
  optional `speaker_id` block turns on voice recognition.
- **Conversational server:** the `host`/`port` it listens on, plus the `asr_*`, `tts_*` and
  `llm_*` host/port of the three servers above.
- **Client:** agents, wake words, voice recognition, gates and field clips all live in the
  client's JSON. The full example is in `load_json_config`'s docstring in
  `conversational-ai-client.py`. A trimmed one:

```json
{
    "host": "127.0.0.1", "port": 65400,
    "pipeline": "basic_conversational",
    "player_name": "Kevin", "user_id": "household",
    "voice_recognition": true, "location_id": "kitchen",
    "agent_defaults": {"voice": "heart", "continuous_save": true, "load_previous": true},
    "agents": [
        {"name": "rose", "wake_words": ["rose", "hey rose"], "system_prompt_id": "assistant-rose"},
        {"name": "rick", "wake_words": ["rick"], "system_prompt_id": "assistant-rick",
         "allowed_speakers": ["Kevin", "Sam"]}
    ]
}
```

**Voice recognition** needs people enrolled first: record a few clips of each person at each
spot, then enroll them. `scripts/ai/asr/speaker_id/README.md` walks through it: recording,
enrolling, the client keys, reading the results, and tuning. The embedding model comes from
Hugging Face, so the ASR server needs your token in `HF_TOKEN`.

**Logs:** every server logs to the screen. Add `"log_file": "/path/to/server.log"` to a
server's config to also write a log file (plain text, one file a day, 90 days kept).

## The library — `src/amadeo_utils/`

| Module | What it does |
| --- | --- |
| `ai/asr/` | WhisperX speech-to-text wrapper, plus speaker identification (who is speaking) |
| `ai/llm/llama/` | `llama.cpp` chat, streaming, role-play & knowledge-base sessions, LoRA fine-tuning helpers |
| `ai/llm/vector_database/` | local vector DB for long-term conversational memory |
| `ai/tts/` | F5-TTS and Kokoro text-to-speech, with a custom voice library |
| `ai/combined/` | the conversational pipeline that orchestrates ASR + LLM + TTS: agents, wake words, handoff notes, speaker gates |
| `client/`, `server/` | the threaded socket framework (length-prefixed JSON protocol) |
| `media_utils/` | audio manipulation helpers + live capture of speakers/mic (with timer, VAD-silence, and programmatic stop conditions) |
| `misc_utils/` | `FileEncryption` — authenticated file encryption (Argon2id + Fernet/AES) |
| `colored_text.py` | terminal color helper |
| `logging_utils.py` | optional log files for the servers (screen + daily file, 90 days kept) |

## The scripts — `scripts/`

- **`ai/combos/conversational_ai/`** — the full voice pipeline (server + client). ⭐ start here
- **`ai/asr/whisperx/streaming/`** — streaming transcription: the ASR server, a live microphone client, and `transcribe_output.py`, which transcribes whatever is playing through the speakers and saves it to a WAV
- **`ai/asr/speaker_id/`** — voice enrollment for speaker identification (`enroll_voice.py`), with its own README
- **`ai/llm/llama/llama_stream/`** — streaming LLM server + client (role-play & knowledge-base modes)
- **`ai/llm/llama/llama_local_vector_db/`** — role-play chat with vector-DB long-term memory
- **`ai/llm/llama/local_knowledge_base/`** — retrieval-augmented Q&A over a local knowledge base
- **`ai/tts/`** — F5 and Kokoro TTS servers, a voice-blending demo, and a simple client
- **`media/`** — audio extraction, recording, and noise reduction utilities, plus
  live speaker-loopback / mic capture (`capture_audio.py`)
- **`tools/infinite_campus/`** — Playwright-driven SSO scraper that emails a school-grades
  report (credentials are read from environment variables; see `.env.example`)
- **`utils/`** — a CLI wrapper around the file-encryption module

## Configuration notes

- **LLM prompts & chat history are external.** The role-play / chat examples read system
  prompts and store conversation history in user-supplied paths — none of that content
  ships in this repo. Defaults live in `subjective_constants.py` as placeholders; drop a
  (gitignored) `subjective_constants_local.py` next to it to override them on your machine.
- **Secrets are never hardcoded.** The Infinite Campus tool reads everything from
  environment variables — copy `.env.example` to `.env` and fill in your own values.
- **Voice data stays out of the repo.** Voice profiles, recorded clips and field clips are
  biometric data: keep them with your private configs. The tests use synthetic stock voices.

## Tests

Regression tests live in `tests/` — one folder per area, each with a `suite.sh`;
`tests/run_tests.sh` runs them all. See `tests/README.md` for the settings file and naming
rules.

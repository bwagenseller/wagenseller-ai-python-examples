# amadeo-utils

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
  job **queue** guarded by a **lock**, so the model is never hit by two requests at
  once. Clients are told the server is "busy" rather than crashing it.
- **Client-side Voice Activity Detection** (`webrtcvad`) decides when you've stopped
  talking and a chunk is ready to send — no push-to-talk needed.
- **Session management** — each client gets a UUID session; resources are cleaned up
  on disconnect.

> ### Why I built it: "talk to Santa"
> The original motivation was letting my kids have a live, spoken back-and-forth
> conversation with Santa Claus — speak into the mic, hear Santa answer in a custom
> cloned voice. That turned into this general-purpose, swappable ASR→LLM→TTS pipeline.

## Install

The library is `pip`-installable. An **editable install** lets the `scripts/`
import `amadeo_utils` from anywhere with no `PYTHONPATH` juggling:

```bash
# core library only
pip install -e .

# with the heavy local-AI stack (torch, whisperx, llama-cpp-python, f5-tts, kokoro, ...)
pip install -e ".[ml]"

# browser-automation extra (for the Infinite Campus tool)
pip install -e ".[automation]"
```

> The `[ml]` extras are large and hardware-specific (CUDA builds of `torch`,
> `llama-cpp-python`, etc.). Install them in an environment matched to your GPU.

## The library — `src/amadeo_utils/`

| Module | What it does |
| --- | --- |
| `ai/asr/` | WhisperX speech-to-text wrapper |
| `ai/llm/llama/` | `llama.cpp` chat, streaming, role-play & knowledge-base sessions, LoRA fine-tuning helpers |
| `ai/llm/vector_database/` | local vector DB for long-term conversational memory |
| `ai/tts/` | F5-TTS and Kokoro text-to-speech, with a custom voice library |
| `ai/combined/` | the conversational pipeline that orchestrates ASR + LLM + TTS |
| `client/`, `server/` | the threaded socket framework (length-prefixed JSON protocol) |
| `media_utils/` | audio manipulation helpers |
| `misc_utils/` | `FileEncryption` — authenticated file encryption (Argon2id + Fernet/AES) |
| `colored_text.py` | terminal color helper |

## The examples — `scripts/`

- **`ai/combos/conversational_ai/`** — the full voice pipeline (server + client). ⭐ start here
- **`ai/asr/whisperx/streaming/`** — streaming transcription client/server
- **`ai/llm/llama/llama_stream/`** — streaming LLM server + client (role-play & knowledge-base modes)
- **`ai/llm/llama/llama_local_vector_db/`** — role-play chat with vector-DB long-term memory
- **`ai/llm/llama/local_knowledge_base/`** — retrieval-augmented Q&A over a local knowledge base
- **`ai/tts/`** — F5 and Kokoro TTS servers, a voice-blending demo, and a simple client
- **`media/`** — audio extraction, recording, and noise reduction utilities
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

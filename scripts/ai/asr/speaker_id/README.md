# speaker_id: who is speaking

With voice recognition on, the conversational AI pipeline works out **who** said each thing from their voice,
instead of assuming it is always the client's `player_name`. The agents then see `[Kevin, to Rose]: ...` or
`[Unrecognized voice, to Rose]: ...`, handoff notes and saved history name the right person, and an agent can be
limited to voices it knows, or to certain people.

This is speaker **identification**: matching a voice against people who have been enrolled. It is not
diarization (splitting one recording between several unnamed speakers).

| Piece | Where |
|---|---|
| Matching rules, profiles, config | `src/amadeo_utils/ai/asr/speaker_id.py` |
| The voice model (pyannote wespeaker ResNet34) | `src/amadeo_utils/ai/asr/speaker_embedder.py` |
| Runs inside the ASR server, on the same audio it transcribes | `src/amadeo_utils/ai/asr/whisperx.py`, served by `scripts/ai/asr/whisperx/streaming/transcribe_server.py` |
| Uses the result: speaker tag, refusals | `src/amadeo_utils/ai/combined/conversational_ai/speakers.py` |
| Enrollment (this folder) | `enroll_voice.py` |

> **Voice profiles, and the clips they are made from, are biometric data.** Keep them with your private configs,
> never in this repository - not even as test fixtures. The tests use synthetic Kokoro stock voices instead.

---

## How it works

1. The client sends a chunk of speech, with `voice_recognition` and its `location_id`.
2. The ASR server transcribes it and, in the same request, turns the speech into a **voiceprint** (an embedding: a
   list of numbers; similar voices give similar lists).
3. It compares the voiceprint with every enrolled person (cosine similarity: 1 = same voice, about 0 = unrelated),
   in two steps:
   1. **This location first** - only the samples enrolled at the request's `location_id`.
   2. **Otherwise everyone** - every sample of every person, at every location.

   In each step, the best person wins if their score is at least the **threshold** AND beats the runner-up (a
   different person) by at least the **margin**. If neither step produces a winner, the speaker is
   `Unrecognized voice`.
4. The conversational server uses that name everywhere: the speaker tag, handoff notes, saved history, the reply
   to the client.

Details worth knowing:

* **Too little speech is not judged.** Under `min_seconds` (default 1 s) of speech, nothing is compared. In the
  middle of a conversation the turn keeps the previous speaker, so a short "yes" is judged as whoever spoke
  before it. At the start of a conversation it is `Unrecognized voice`.
* **Speech length comes from the transcription**, so silence at the end of a chunk does not count.
* **It fails closed.** If the ASR server cannot judge (no `speaker_id` config, a model error, an older server),
  the speaker is `Unrecognized voice`, never the client's `player_name`.
* **Each location has its own average.** A person's samples are grouped by location, and each location's clips
  are averaged into one voiceprint. Mic, room and gain all shift a voiceprint, so samples from different spots
  are kept apart rather than blurred together.
* **Profiles reload by themselves.** The server notices when the profiles folder changes, so enrolling someone
  needs no restart.
* **Voice matching is weak security.** A recording, the TV or a similar voice can get through. It is good for
  keeping an agent away from guests or children, not for anything that matters.

---

## Setting it up

### 1. Configure the ASR server

Start `transcribe_server.py` with a `--json` config (see `../whisperx/streaming/example_transcribe_server_config.json`).
Speaker identification is on when the config has a `speaker_id` block:

```json
{
    "host": "127.0.0.1",
    "port": 65432,
    "model": "large-v3",
    "gpu": 0,
    "speaker_id": {
        "profiles_dir": "/path/to/private/configs/voice-profiles",
        "samples_dir": "/path/to/private/configs/voice-samples",
        "threshold": 0.50,
        "margin": 0.05,
        "min_seconds": 1.0,
        "locations": {
            "kitchen": {"threshold": 0.55, "margin": 0.08}
        }
    }
}
```

| Key | Meaning |
|---|---|
| `profiles_dir` | **Required.** Where the voice profiles (`<name>.json`) live. |
| `samples_dir` | Where the recorded clips live, as `<person>/<location_id>/*.wav`. Only `enroll_voice.py --all` reads it. |
| `threshold` | Minimum score to count as a match (default 0.50). |
| `margin` | How far the best person must beat the runner-up (default 0.05). |
| `min_seconds` | Less speech than this is not judged (default 1.0). |
| `embedding_model` | The voice model (default `pyannote/wespeaker-voxceleb-resnet34-LM`). Profiles made with another model are ignored. |
| `locations` | Per-`location_id` overrides of `threshold` and `margin` only. They apply to both matching steps of a request from that location. |
| `field_samples_dir` | Where field clips are saved (see [Field clips](#field-clips)). Needed if either switch below is on. |
| `save_known_field_clips` | Allow saving the speech of identified speakers (default `false`). The master switch: each client must opt in too. |
| `save_unknown_field_clips` | The same, for unrecognized voices (default `false`). |
| `field_retention_days` | Delete field clips older than this (default 90; 0 keeps them for ever). |

Other points:

* The model is downloaded from Hugging Face on first start. Put your token in the `HF_TOKEN` environment variable
  of the server's process. A non-interactive SSH session or a tmux session may not see it, so pass it explicitly.
* A malformed `speaker_id` block stops the server at startup, rather than leaving recognition quietly off.
* A setting the server doesn't recognise (a typo, or one at the wrong level, like `log_file` inside
  `speaker_id`) is logged as a warning at startup, with a hint when it belongs at the other level, and ignored.
  Keys starting with `_` (`_comment`) are notes and pass silently.
* **The client can never change the threshold, margin or minimum length.** If it could, any client could lower
  the threshold and walk past a gated agent.

### 2. Record clips

Record each person **at each location, through that location's own microphone**, the same way the client hears
them. `--record-only` just records and saves WAVs: no server and no profiles folder are needed, so it can run on
any machine with the mic (a Raspberry Pi, a desktop):

```bash
# media conda env (numpy + sounddevice); read the sentence shown, 8 s per clip
python enroll_voice.py --record 6 --record-only --save-dir <samples_dir>/kevin/kitchen
```

* Aim for **4-8 clips of 5-10 s** per person per location, each a different sentence. With children, anything
  they say for 8 s works.
* Talk the way you would talk to an agent, at the usual distance from the mic.
* The clips are recorded exactly as the conversational client records: through PulseAudio / PipeWire when it is
  there (the desktop's chosen input device, resampled to 16 kHz), mono, 16 bit, **unprocessed**.
* Running it again into the same folder adds clips; it never overwrites.
* The folder names matter. The **person** folder becomes the profile name (letters, digits, `_ . -`). The
  **location** folder must equal the `location_id` of the clients at that spot.

> **Use raw recordings, not cleaned-up ones.** Clips that were denoised, noise-gated or prepared as voice-cloning
> references teach the profile a voice the live mic never delivers. In one real case, cleaned clips scored about
> 0.87 against each other but only 0.35-0.50 against the same person speaking live, under the threshold, so
> every turn came back unrecognized. Raw clips from the same mic scored 0.65-0.82 live. A quick check: a real
> room's quietest moments sit around -50 to -70 dBFS; a cleaned clip is near -90.

### 3. Enroll

Run this where the profiles folder is (usually on the ASR server's machine), with the ASR server running:

```bash
python enroll_voice.py --json /path/to/asr-server.json --all
```

For each `<person>/<location_id>/` folder under `samples_dir`, this:

1. sends every WAV to the ASR server, which turns it into a voiceprint (so enrollment always uses exactly the
   server's model and audio handling);
2. writes the voiceprints into `<profiles_dir>/<person>.json`, **replacing** that person's samples for that
   location (other locations are left alone), so running it again after adding clips gives the same result;
3. prints how each new clip scores against everyone already enrolled. A clip that scores closer to someone
   else than to its own person is worth re-recording.

| Command | Does |
|---|---|
| `enroll_voice.py --json CFG --all` | Enroll everyone at every location found |
| `enroll_voice.py --json CFG --all --prune` | ...and remove whatever is no longer in the samples tree, so the profiles match it exactly |
| `enroll_voice.py --json CFG --all --name kevin` | ...only Kevin |
| `enroll_voice.py --json CFG --all --location-id kitchen` | ...only the kitchen |
| `enroll_voice.py --json CFG --list` | Who is enrolled, where, and with how many clips |
| `enroll_voice.py --json CFG --name kevin --location-id kitchen --record 6 --save-dir DIR` | Record, keep the WAVs, and enroll in one go (on a machine that can reach both the server and the profiles folder) |
| `enroll_voice.py --profiles-dir DIR --name kevin --location-id kitchen --wav a.wav b.wav` | Enroll specific files without a server config; add `--replace` to replace instead of adding |

`--json` reads the server's host, port, `profiles_dir` and `samples_dir` from the server's own config, so
enrollment always writes where the server reads. `--profiles-dir`, `--samples-dir`, `--asr-host` and `--asr-port`
override it. The server's `host` must be reachable from where you run this. A server bound to `127.0.0.1` can
only be reached from its own machine.

**Removing samples:** plain `--all` only touches the locations it finds, so deleting a location's folder does
**not** remove its voiceprints. Delete the folder, then run `--all --prune`:

* a location with no folder any more is dropped from that person's profile;
* a person with no folders left has their profile deleted;
* every removal is printed;
* `--name` / `--location-id` narrow what may be removed, the same way they narrow what is enrolled
  (`--all --prune --name kevin` after deleting Kevin's whole folder deletes just Kevin's profile).

Safety rules: nothing is pruned when any clip fails to enroll, or when the samples tree is empty or missing (a
wrong path or an unmounted share would otherwise delete every profile). Note that `--prune` also removes
anything enrolled from clips that were never saved into the tree (`--record` without `--save-dir`, or `--wav`
files kept elsewhere).

### 4. Turn it on in a client

In the conversational client's JSON (`basic_conversational` pipeline):

```json
{
    "voice_recognition": true,
    "location_id": "kitchen",
    "agents": [
        {"name": "rose", "wake_words": ["rose"], "system_prompt_id": "assistant-rose"},
        {"name": "crane", "wake_words": ["frasier"], "system_prompt_id": "assistant-frasier",
         "allow_unknown_speakers": false},
        {"name": "rick", "wake_words": ["rick"], "system_prompt_id": "assistant-rick",
         "allowed_speakers": ["Kevin", "Sam"]}
    ]
}
```

| Key | Where | Meaning |
|---|---|---|
| `voice_recognition` | top level (default `false`; `--voice-recognition` on the command line) | Identify the speaker from their voice. Off, the speaker is `player_name`, as before. |
| `location_id` | top level (`--location-id`) | Which spot this client is. Its samples are compared first. It must match the folder name the clips were recorded under. |
| `allow_unknown_speakers` | per agent, or `agent_defaults` (default `true`) | `false`: the agent refuses an `Unrecognized voice` (`unknown_speaker`). An unrecognized voice that is **continuing a conversation with that same agent** still goes through. |
| `save_known_field_clips` / `save_unknown_field_clips` | top level (default `false`; `--save-known-field-clips` / `--save-unknown-field-clips`) | Let the ASR server keep this microphone's speech as field clips (known / unrecognized voices). Only works if the ASR server's config allows it too. See [Field clips](#field-clips). |
| `allowed_speakers` | per agent, or `agent_defaults` (default empty = anyone) | The only people this agent answers, by the names the agents hear (the profiles' display names, case ignored). Anyone else is refused (`speaker_not_allowed`), and an unrecognized voice too (`unknown_speaker`). This is **strict**: it applies even in the middle of a conversation with that agent. |

Both gates need `voice_recognition` on. With it off they do nothing, and the client warns about
`allowed_speakers` at startup. `location_id` is separate from `user_id`: `user_id` decides whose conversation
history continues, and `location_id` only decides which samples are compared first.

With recognition on, a system prompt's `@@NAME@@` is filled with a neutral phrase (the conversational server's
`household_name`, default "the members of the household"), because the session is shared by whoever is talking.
Who is actually talking comes from the speaker tag on each turn.

---

## Reading the results

The ASR server logs one line per chunk:

```
Speaker: identified 'Kevin' (step 'location', location 'kitchen', 3.2 s; Kevin 0.81, Sam 0.38, Kim 0.22)
Speaker: unknown 'Unrecognized voice' (step '', location 'kitchen', 2.1 s; Kim 0.72, Sam 0.70, Kevin 0.31)
Speaker: too_short '' (step '', location 'kitchen', 0.6 s; no scores)
```

* **Status:** `identified`, `unknown`, `too_short`, `no_profiles` (nobody enrolled), `disabled` (no `speaker_id`
  config), or `error`.
* **Step:** `location` or `everyone` - which step decided. Empty if nobody matched.
* **Scores:** from the last step run, best first. For an `unknown`, they show why: in the second line Kim beat
  Sam by only 0.02, less than the margin.

The conversational server logs the speaker it settled on and how. The reply to the client carries `speaker` and
`speaker_source`:

| `speaker_source` | Meaning |
|---|---|
| `voice` | The ASR server judged the voice (identified or unrecognized) |
| `last_speaker` | Too little speech mid-conversation, so the previous turn's speaker |
| `fallback` | Nothing to go on, so `Unrecognized voice` |
| `request` | Recognition off, so the client's `player_name` |

## Field clips

The ASR server can keep the speech it judges, which gives you raw clips from the real microphones (the best
enrollment material there is) and a way to catch voices nobody has enrolled yet. Every chunk a client sends goes
through the ASR server, including chunks that turn out not to be for any agent, so clips are saved whether or not
an agent was addressed.

```
<field_samples_dir>/<location_id>/<person>/20260928-140509-250000.wav      identified (save_known_field_clips)
<field_samples_dir>/<location_id>/unrecognized/20260928-140522-031000.wav  not recognized (save_unknown_field_clips)
```

**Both sides must say yes**, for each kind separately:

| ASR server config | Client JSON | Saved? |
|---|---|---|
| off | anything | No - the server's switch turns all recording off at once |
| on | off, or not set | No |
| on | on | Yes |

So the server owns the disk, the directory and the retention, and each spot decides whether its own microphone
may be recorded: set `save_known_field_clips` / `save_unknown_field_clips` in the clients where recording is
fine (an office, say), and leave them out where it isn't (a bedroom). The client side defaults to off, so a new
client, or a copied config, records nothing until you say so. A client can only narrow what the server allows,
never widen it.

* Only judged speech is saved, and only the speech itself (no trailing silence): 16 kHz mono 16 bit WAV,
  readable by owner and group only. Too-short speech, requests with recognition off, and chunks the server
  could not judge are not saved.
* `<person>` is the profile name, so the folders line up with the samples tree. No `location_id` gives
  `no-location/`. Folder names are cleaned, so a location can never point outside the directory.
* Clips older than `field_retention_days` (default 90) are deleted, at startup and then at most once an hour.

**Field clips are never enrolled automatically.** A misidentified clip fed back into a profile would pull it
toward the wrong person, and the next mistake would be more likely. Review them yourself: listen, then move the
good ones into `<samples_dir>/<person>/<location_id>/` and run `enroll_voice.py --json CFG --all --prune`. An
`unrecognized` clip of someone who should be enrolled is how a new person (or a known person on a new mic) gets
added.

> **Privacy:** with field clips on, the house gets a rolling recording of whatever is said near a microphone -
> not only speech meant for the agents - including guests who don't know. Keep the directory with the private
> configs, never in a repository, and keep the retention period short.

## Tuning

| You see | Try |
|---|---|
| The right person on top, far ahead of everyone, but under the threshold | The enrollment clips don't match the live audio. Re-record them raw at that spot (see step 2). Don't just lower the threshold. |
| Two similar voices (siblings, say) often `unknown`, each just ahead of the other | More clips of both; then a slightly smaller `margin` for that location. |
| Someone identified as the wrong person | A larger `margin` (or `threshold`) for that location, and re-record the clips that score close to the other person. |
| Short replies often `too_short` | Expected under `min_seconds`. Mid-conversation, the previous speaker is kept. |

Change the ASR server's config and restart the server to apply new settings. Profiles themselves reload without
a restart.

## Tests

`tests/asr/` (see `tests/README.md`): `check_speaker_id.py` pins the rules with made-up vectors (any Python 3).
`check_speaker_id_model.py` runs the real model end to end through the ASR server's request handler against
synthetic Kokoro stock voices (`stt` env; render the voices once with `make_synthetic_voices.py` in the `kokoro`
env). The client and server gates are in `tests/conversational_ai/`.

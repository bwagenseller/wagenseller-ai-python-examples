"""
Enrolls a person's voice for speaker identification (CS-23): builds or extends <name>.json in the voice profiles
directory, which the WhisperX ASR server matches every chunk of speech against when a client turns voice recognition
on (see amadeo_utils.ai.asr.speaker_id).

The embeddings are made BY THE ASR SERVER (its 'speaker_embedding' command), so enrollment always uses exactly the
model and audio handling that recognition will use, and this script needs no model or GPU of its own - it runs in the
'media' conda environment (numpy, plus sounddevice for recording), e.g. on the Raspberry Pi at the location being
enrolled. The ASR server must be running with a 'speaker_id' block in its --json config.

Enroll each person AT EACH LOCATION, through that location's own microphone: a mic, its room and its gain all shift a
voice's embedding, so the profile keeps the embeddings of each location apart (location_id) and the server compares
the samples from the client's own location first. Aim for 4-8 clips of 5-10 seconds each, of different sentences.

The easiest way to run it is against the ASR server's own --json config (--json): the server's host and port and
the 'speaker_id' block's profiles_dir (and samples_dir, for --all) come from there, so enrollment always writes
where the server reads. --profiles-dir / --samples-dir / --asr-host / --asr-port override the file, or replace it.

Examples:
    # enroll everyone from the samples tree <samples_dir>/<person>/<location_id>/*.wav (each location found
    # there replaces that person's earlier samples for it; other locations are left alone)
    python enroll_voice.py --json /path/to/asr-server.json --all

    # ...and also remove what is no longer in the tree (a deleted location folder, or a person's whole folder), so
    # the profiles match the tree exactly
    python enroll_voice.py --json /path/to/asr-server.json --all --prune

    # ...only one person, or only one location
    python enroll_voice.py --json /path/to/asr-server.json --all --name kevin
    python enroll_voice.py --json /path/to/asr-server.json --all --location-id kitchen

    # record 6 clips of 8 seconds through this machine's microphone, for the 'office' location
    python enroll_voice.py --json /path/to/asr-server.json --name kevin --location-id office --record 6

    # enroll from existing recordings (any sample rate; 16 bit PCM WAV), replacing the kitchen's old samples
    python enroll_voice.py --profiles-dir /path/to/voice-profiles --name kevin --location-id kitchen --wav a.wav b.wav --replace

    # record raw clips at a spot and only SAVE them (no ASR server, no profile) - e.g. on a Pi, or a machine that
    # cannot reach the profiles directory; copy the folder into the samples tree and enroll there with --all
    python enroll_voice.py --record 6 --record-only --save-dir ~/voice-samples/kevin/kitchen

    # who is enrolled, where
    python enroll_voice.py --json /path/to/asr-server.json --list

Recording captures exactly what the conversational client sends (16 kHz mono 16 bit, the default input device, no
processing), so enroll with clips recorded this way rather than cleaned-up ones: a denoised or noise-gated clip
teaches the profile a voice the live microphone never delivers, and live scores fall below the threshold.

Voice profiles, and the clips they are made from, are biometric data: keep them with the private configs, never in
the repository.
"""

import argparse
import json
import logging
import os
import sys
import time
import wave
from typing import List, Tuple

import numpy as np

from amadeo_utils.client.amadeo_client import AmadeoClient
from amadeo_utils.ai.asr import speaker_id

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ASR_HOST = 'localhost'
ASR_PORT = 65432
LOCATION_ID = 'default'
RECORD_SECONDS = 8.0
RECORD_SAMPLE_RATE = 16000      # what the conversational client records at, and what the ASR server works in
RECORD_BLOCK = 480              # frames per read: 30 ms at 16 kHz, the client's VAD frame size

# Things to read aloud while recording: varied sounds, a few seconds each
PROMPTS = [
    "The quick brown fox jumps over the lazy dog, and then it naps in the warm afternoon sun.",
    "Could you remind me to pick up milk, eggs and bread on the way home tomorrow?",
    "I think the weather this weekend is supposed to be cold and rainy, so let's plan something indoors.",
    "My favourite movie has a surprising ending that nobody in the theatre saw coming.",
    "Please turn the living room lights down a little and play some quiet music.",
    "We should repaint the fence before winter, maybe a darker shade of green this time.",
    "Seven hundred and forty two people signed up for the charity run last year.",
    "What would you cook for dinner if you only had rice, onions, garlic and a can of beans?",
]


def read_wav(path: str) -> Tuple[bytes, int]:
    """
    Reads a WAV file as the raw 16 bit mono PCM the ASR server takes.

    Args:
        path: a 16 bit PCM WAV (mono or stereo; stereo is averaged to mono).

    Returns:
        Tuple[bytes, int]: (the PCM bytes, the sample rate).

    Raises:
        ValueError: the file is not 16 bit PCM.
    """
    with wave.open(path, 'rb') as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16 bit PCM WAV files are supported (this one is {8 * w.getsampwidth()} bit)")
        channels, rate = w.getnchannels(), w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples.tobytes(), rate


def record_clips(count: int, seconds: float) -> List[Tuple[bytes, int]]:
    """
    Records clips through the default microphone, prompting a sentence to read for each.

    Args:
        count: how many clips.
        seconds: how long each one is.

    Returns:
        List[Tuple[bytes, int]]: (16 bit mono PCM, sample rate) per clip.
    """
    import sounddevice as sd    # only needed to record
    from amadeo_utils.media_utils.audio_devices import prefer_pulse_defaults

    # Record exactly as the conversational client does: through PulseAudio / PipeWire, which follows the desktop's
    # chosen microphone and resamples to 16 kHz. Raw ALSA devices refuse 16 kHz outright on most hardware
    # ('Invalid sample rate'). On a host with no PulseAudio host API the defaults are left alone.
    prefer_pulse_defaults()

    clips = []
    for index in range(count):
        prompt = PROMPTS[index % len(PROMPTS)]
        input(f"\nClip {index + 1} of {count}. Press Enter, then read aloud (or say anything) for {seconds:g} s:\n    \"{prompt}\"\n")
        # A blocking InputStream read, the way the conversational client captures audio. sd.rec() / sd.wait() rely on
        # PortAudio's stream-finished callback, which never arrived through the PulseAudio host API on the desktop
        # (the recording hung for ever); plain reads have no such dependency.
        wanted = int(seconds * RECORD_SAMPLE_RATE)
        chunks, captured = [], 0
        started = time.time()
        with sd.InputStream(samplerate=RECORD_SAMPLE_RATE, channels=1, dtype='int16', blocksize=RECORD_BLOCK) as stream:
            while captured < wanted:
                data, overflowed = stream.read(min(RECORD_BLOCK, wanted - captured))
                if overflowed:
                    logger.warning("Input overflow while recording - a few samples were dropped.")
                chunks.append(data.reshape(-1))
                captured += len(data)
        print(f"    ...recorded {time.time() - started:.1f} s.")
        clips.append((np.concatenate(chunks).astype(np.int16).tobytes(), RECORD_SAMPLE_RATE))
    return clips


def save_clips(save_dir: str, clips: List[Tuple[bytes, int]]) -> List[str]:
    """
    Saves recorded clips as 16 bit mono WAVs, named by the time of the run (rec-<YYYYMMDD-HHMMSS>-<nn>.wav). An
    existing file is never overwritten (the number is moved on past it), so a second session into the same folder
    always adds clips.

    Args:
        save_dir: the folder (created if missing), e.g. <samples_dir>/<person>/<location_id>.
        clips: (16 bit mono PCM, sample rate) per clip.

    Returns:
        List[str]: the paths written.
    """
    os.makedirs(save_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    paths = []
    number = 0
    for pcm, rate in clips:
        number += 1
        path = os.path.join(save_dir, f"rec-{stamp}-{number:02d}.wav")
        while os.path.exists(path):
            number += 1
            path = os.path.join(save_dir, f"rec-{stamp}-{number:02d}.wav")
        with wave.open(path, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm)
        paths.append(path)
        logger.info(f"Saved {path}.")
    return paths


def embed_clips(host: str, port: int, clips: List[Tuple[bytes, int]], labels: List[str]) -> Tuple[List[List[float]], str]:
    """
    Asks the ASR server for each clip's embedding.

    Args:
        host, port: the WhisperX ASR server (its --json config must have a 'speaker_id' block).
        clips: (PCM bytes, sample rate) per clip.
        labels: a name per clip, for messages.

    Returns:
        Tuple[List[List[float]], str]: (the embeddings, the server's embedding model id).

    Raises:
        RuntimeError: the server could not be reached, or refused a clip.
    """
    embeddings, model = [], ''
    # The replies are read from send_transient_request's return value; the callback is a no-op only so AmadeoClient
    # does not warn that it has none
    client = AmadeoClient(host, port, additional_server_response_functionality=lambda response, raw_data: None)
    for (pcm, rate), label in zip(clips, labels):
        response, _ = client.send_transient_request('speaker_embedding', 'Voice enrollment', binary_data=pcm, sample_rate=rate)
        if not response:
            raise RuntimeError(f"The ASR server at {host}:{port} isn't reachable - is it running (with a 'speaker_id' "
                               f"block in its --json config), and reachable from this machine? Nothing was enrolled for {label}.")
        if not response.get('success'):
            raise RuntimeError(f"The ASR server refused {label}: {response.get('message')}")
        seconds = response.get('seconds', 0.0)
        if seconds < 3.0:
            logger.warning(f"{label} is only {seconds:.1f} s long; longer clips (5-10 s) make a better profile.")
        embeddings.append(response['embedding'])
        model = response.get('model', '')
        logger.info(f"Embedded {label} ({seconds:.1f} s).")
    return embeddings, model


def report_similarity(profiles_dir: str, name: str, display: str, embeddings: List[List[float]], model: str):
    """
    Prints how each new clip scores against everyone already enrolled - a clip that scores close to someone ELSE
    (or low against this person's own earlier samples) is worth re-recording.

    Args:
        profiles_dir: the profiles directory.
        name: the person being enrolled (their own earlier samples are included, under their display name).
        display: their display name.
        embeddings: the new clips' embeddings.
        model: the embedding model.
    """
    index = speaker_id.SpeakerIndex(speaker_id.load_profiles(profiles_dir, model))
    if not len(index):
        return
    print("\nHow the new clips score against the people already enrolled (1 = same voice, ~0 = unrelated):")
    for number, embedding in enumerate(embeddings, 1):
        unit = speaker_id.normalize(embedding)
        scores = {person: max(speaker_id.cosine(unit, c) for _, c in entries) for person, entries in index.people.items()}
        ranked = ', '.join(f"{person} {score:.2f}" for person, score in sorted(scores.items(), key=lambda kv: -kv[1]))
        best = max(scores, key=scores.get)
        flag = '' if best == display or display not in scores else f"   <- closer to {best} than to {display}"
        print(f"    clip {number}: {ranked}{flag}")


def list_profiles(profiles_dir: str):
    """
    Prints who is enrolled, and how many clips at each location.

    Args:
        profiles_dir: the profiles directory.
    """
    profiles = speaker_id.load_profiles(profiles_dir)
    if not profiles:
        print(f"Nobody is enrolled in {profiles_dir}.")
    for profile in profiles:
        places = ', '.join(f"{loc} ({len(v)} clips)" for loc, v in sorted(profile.locations.items()))
        print(f"{profile.name} ('{profile.display_name}', {profile.model}): {places or 'no samples'}")


def save_enrollment(profiles_dir: str, name: str, display_name: str, location_id: str,
                    embeddings: List[List[float]], model: str, replace: bool) -> bool:
    """
    Adds one person's new embeddings for one location to their profile (other locations are left alone), or starts
    a new profile, after showing how the new clips score against everyone already enrolled.

    Args:
        profiles_dir: the profiles directory.
        name: the person's file-safe name.
        display_name: what the agents call them; '' keeps the profile's (or the default).
        location_id: where the clips were recorded.
        embeddings: the clips' embeddings.
        model: the embedding model the server used.
        replace: replace this location's earlier samples instead of adding to them.

    Returns:
        bool: False if the profile was made with a different embedding model and could not be updated.
    """
    path = speaker_id.profile_path(profiles_dir, name)
    if os.path.exists(path):
        profile = speaker_id.load_profile(path)
        if profile.model != model:
            # Embeddings from different models cannot be compared; only a profile that holds nothing but this
            # location, which is being replaced, can switch model
            if not replace or len(profile.locations) > 1 or location_id not in profile.locations:
                logger.error(f"{path} was made with '{profile.model}' but the server now uses '{model}': the embeddings "
                             f"cannot be mixed. Move the old profile aside and enroll again at every location.")
                return False
            profile.model = model
    else:
        profile = speaker_id.VoiceProfile(name=name, model=model)
    if display_name:
        profile.display_name = display_name

    report_similarity(profiles_dir, name, profile.display_name, embeddings, model)

    profile.add(location_id, embeddings, replace=replace)
    written = speaker_id.save_profile(profiles_dir, profile)
    print(f"\nSaved {written}: '{profile.display_name}' now has "
          + ', '.join(f"{len(v)} clip(s) at {loc}" for loc, v in sorted(profile.locations.items())) + '.')
    return True


def prune_profiles(profiles_dir: str, found, only_name: str, only_location: str):
    """
    Removes what is enrolled but no longer in the samples tree (see speaker_id.prune_plan): a location without its
    folder is dropped from the person's profile, and a profile left with no locations is deleted. Every removal is
    printed.

    Args:
        profiles_dir: the profiles directory.
        found: the WHOLE samples tree (speaker_id.find_enrollment_samples).
        only_name, only_location: narrow what may be removed, as for enrolling.
    """
    profiles = {p.name: p for p in speaker_id.load_profiles(profiles_dir)}
    drop_locations, delete_profiles = speaker_id.prune_plan(list(profiles.values()), found, only_name, only_location)
    if not drop_locations and not delete_profiles:
        print("\nPrune: every enrolled location is still in the samples tree; nothing removed.")
        return
    for name, locations in drop_locations.items():
        profile = profiles[name]
        for location in locations:
            del profile.locations[location]
        speaker_id.save_profile(profiles_dir, profile)
        print(f"\nPrune: removed {', '.join(locations)} from '{profile.display_name}' (no longer in the samples tree).")
    for name in delete_profiles:
        os.remove(speaker_id.profile_path(profiles_dir, name))
        print(f"\nPrune: deleted {name}.json - none of its locations are in the samples tree any more.")


def enroll_all(samples_dir: str, profiles_dir: str, host: str, port: int, only_name: str, only_location: str,
               prune: bool = False) -> int:
    """
    Enrolls everyone found in the samples tree (see speaker_id.find_enrollment_samples). Each person and location
    found REPLACES that location's earlier samples, so the tree is the record of what is enrolled there and running
    this again after adding or removing clips gives the same result. Locations not in the tree are left alone -
    unless prune is set, which then removes them (prune_profiles), so the profiles match the tree exactly.

    Pruning is skipped when anything failed to enroll, and never happens against an empty or missing samples tree
    (a wrong path or an unmounted share would otherwise delete every profile).

    Args:
        samples_dir: the samples directory.
        profiles_dir: the profiles directory.
        host, port: the ASR server.
        only_name: enroll (and prune) only this person ('' for everyone).
        only_location: enroll (and prune) only this location_id ('' for every one).
        prune: also remove enrolled locations that are no longer in the tree.

    Returns:
        int: the exit code - 1 if nothing was found, or any person could not be enrolled.
    """
    tree = speaker_id.find_enrollment_samples(samples_dir)
    if not tree:
        logger.error(f"Nothing to enroll in {samples_dir} (expected <person>/<location_id>/*.wav)"
                     + ("; not pruning against an empty samples tree." if prune else "."))
        return 1

    found = tree
    if only_name:
        found = {only_name: found[only_name]} if only_name in found else {}
    if only_location:
        found = {person: {only_location: places[only_location]} for person, places in found.items() if only_location in places}
    if not found and not prune:
        # With prune, an empty selection is fine: it is how a removed person or location is cleared out
        logger.error(f"Nothing to enroll in {samples_dir}"
                     + (f" for {only_name}" if only_name else '') + (f" at {only_location}" if only_location else '')
                     + " (expected <person>/<location_id>/*.wav).")
        return 1

    failed = []
    for person, places in found.items():
        for location_id, wavs in places.items():
            print(f"\n=== {person} at {location_id}: {len(wavs)} clip(s)")
            clips, labels = [], []
            for path in wavs:
                try:
                    clips.append(read_wav(path))
                    labels.append(os.path.relpath(path, samples_dir))
                except (OSError, ValueError, EOFError, wave.Error) as e:
                    # One bad clip (not a WAV, not 16 bit, truncated) should not stop the rest of the tree
                    logger.error(f"Skipping {path}: {e}")
            if not clips:
                failed.append(f"{person}/{location_id}")
                continue
            embeddings, model = embed_clips(host, port, clips, labels)
            if not save_enrollment(profiles_dir, person, '', location_id, embeddings, model, replace=True):
                failed.append(f"{person}/{location_id}")

    if failed:
        logger.error(f"Not enrolled: {', '.join(failed)}." + (" Not pruning until everything enrolls." if prune else ''))
        return 1
    if prune:
        prune_profiles(profiles_dir, tree, only_name, only_location)
    return 0


def main() -> int:
    """
    Parses the command line and enrolls (or lists).

    Returns:
        int: the exit code.
    """
    parser = argparse.ArgumentParser(description="Enroll a voice for speaker identification.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    # The defaults of the settings the server config can supply are None, so an explicit value can be told apart
    # from one to take from --json (or, failing that, from the built-in default)
    parser.add_argument('--json', default='', help="The ASR server's --json config: host, port, and the 'speaker_id' block's profiles_dir and samples_dir are read from it.")
    parser.add_argument('--profiles-dir', default=None, help="The voice profiles directory (default: the server config's speaker_id.profiles_dir).")
    parser.add_argument('--list', action='store_true', help="List who is enrolled, then stop.")
    parser.add_argument('--all', action='store_true', help="Enroll everyone in the samples directory, <samples_dir>/<person>/<location_id>/*.wav; each location found replaces that person's earlier samples for it. --name / --location-id narrow it down.")
    parser.add_argument('--prune', action='store_true', help="With --all: also remove enrolled locations that are no longer in the samples tree, and delete a profile left with none, so the profiles match the tree exactly (--name / --location-id narrow it). Anything enrolled from clips that were never saved into the tree is removed too.")
    parser.add_argument('--samples-dir', default=None, help="The samples directory for --all (default: the server config's speaker_id.samples_dir).")
    parser.add_argument('--name', help="The person's file-safe name (letters, digits, '_', '.', '-'); the profile is <name>.json.")
    parser.add_argument('--display-name', default='', help="What the agents call them (default: the name in title case, or what the profile already says).")
    parser.add_argument('--location-id', default=None, help=f"Where these clips were recorded - the client's location_id (default: '{LOCATION_ID}'; with --all, every location found).")
    parser.add_argument('--wav', nargs='*', default=[], help="16 bit PCM WAV files of this person (any sample rate).")
    parser.add_argument('--record', type=int, default=0, help="Record this many clips through the default microphone.")
    parser.add_argument('--seconds', type=float, default=RECORD_SECONDS, help=f"Length of each recorded clip (default: {RECORD_SECONDS:g}).")
    parser.add_argument('--save-dir', default='', help="Also save the recorded clips as WAVs in this folder - make it <samples_dir>/<person>/<location_id> so --all can enroll them later.")
    parser.add_argument('--record-only', action='store_true', help="Only record and save (needs --record and --save-dir): no ASR server, no profile. For recording at a spot that cannot reach the server or the profiles directory.")
    parser.add_argument('--replace', action='store_true', help="Replace this location's earlier samples instead of adding to them (--all always replaces).")
    parser.add_argument('--asr-host', default=None, help=f"The WhisperX ASR server (default: the server config's host, else {ASR_HOST}).")
    parser.add_argument('--asr-port', type=int, default=None, help=f"Its port (default: the server config's port, else {ASR_PORT}).")
    args = parser.parse_args()

    # Recording only: nothing below (server, profiles) is needed
    if args.record_only:
        if args.record <= 0 or not args.save_dir:
            parser.error("--record-only needs --record N and --save-dir.")
        if args.wav or args.all or args.list:
            parser.error("--record-only cannot be combined with --wav, --all or --list.")
        paths = save_clips(os.path.expanduser(args.save_dir), record_clips(args.record, args.seconds))
        print(f"\nSaved {len(paths)} clip(s) in {os.path.expanduser(args.save_dir)}. Copy the folder into the samples "
              f"tree (<samples_dir>/<person>/<location_id>) and enroll with --all.")
        return 0

    # Settings from the server's own config, if given; explicit arguments win
    config_host, config_port, config_settings = '', 0, None
    if args.json:
        try:
            with open(os.path.expanduser(args.json), 'r', encoding='utf-8') as f:
                config_host, config_port, config_settings = speaker_id.enrollment_target(json.load(f))
        except (OSError, ValueError, TypeError) as e:
            parser.error(f"cannot use the ASR server config {args.json}: {e}")
    host = args.asr_host or config_host or ASR_HOST
    port = args.asr_port or config_port or ASR_PORT
    profiles_dir = args.profiles_dir or (config_settings.profiles_dir if config_settings else '')
    if not profiles_dir:
        parser.error("give --profiles-dir, or --json with the ASR server's config.")
    profiles_dir = os.path.expanduser(profiles_dir)

    if args.list:
        list_profiles(profiles_dir)
        return 0
    if args.prune and not args.all:
        parser.error("--prune only works with --all (it makes the profiles match the samples tree).")
    if args.name and not speaker_id.is_safe_name(args.name):
        parser.error("--name may only hold letters, digits, '_', '.' and '-'.")

    if args.all:
        if args.wav or args.record > 0 or args.display_name:
            parser.error("--all takes its clips from the samples directory: it cannot be combined with --wav, --record or --display-name.")
        samples_dir = args.samples_dir or (config_settings.samples_dir if config_settings else '')
        if not samples_dir:
            parser.error("--all needs --samples-dir, or --json with a server config whose speaker_id block has samples_dir.")
        try:
            return enroll_all(os.path.expanduser(samples_dir), profiles_dir, host, port, args.name or '', args.location_id or '',
                              prune=args.prune)
        except RuntimeError as e:
            # The ASR server is down or refused a clip: say so plainly (a traceback here only hides the message).
            # Whatever was enrolled before this point is saved; nothing is pruned.
            logger.error(str(e))
            return 1

    if not args.name:
        parser.error("--name is required (or use --all).")
    if not args.wav and args.record <= 0:
        parser.error("give --wav files and/or --record N (or use --all).")

    clips, labels = [], []
    for path in args.wav:
        clips.append(read_wav(path))
        labels.append(os.path.basename(path))
    if args.record > 0:
        recorded = record_clips(args.record, args.seconds)
        if args.save_dir:
            save_clips(os.path.expanduser(args.save_dir), recorded)
        clips += recorded
        labels += [f"recording {i + 1}" for i in range(len(recorded))]

    try:
        embeddings, model = embed_clips(host, port, clips, labels)
    except RuntimeError as e:
        logger.error(str(e))    # server down, or it refused a clip; nothing was saved
        return 1
    saved = save_enrollment(profiles_dir, args.name, args.display_name, args.location_id or LOCATION_ID,
                            embeddings, model, replace=args.replace)
    return 0 if saved else 1


if __name__ == '__main__':
    sys.exit(main())

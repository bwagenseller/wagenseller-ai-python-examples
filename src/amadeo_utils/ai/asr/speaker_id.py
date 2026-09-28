"""
Speaker identification: who, of the people enrolled, is speaking - or nobody we know.

This is identification, not diarization. Each chunk of speech is one utterance (usually one person); the ASR server
turns it into a voice embedding (a vector, see speaker_embedder.py) and this module matches it against voice profiles
enrolled beforehand (see scripts/ai/asr/speaker_id/enroll_voice.py). Nothing here needs a model, torch or numpy - it
is plain Python, so the rules can be tested anywhere and the conversational server could use them too.

Profiles
--------
One JSON file per person in the profiles directory, named after the person:

    kevin.json
    {
        "name": "kevin",
        "display_name": "Kevin",
        "model": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "locations": {
            "office":  [[0.01, -0.12, ...], [...], ...],
            "kitchen": [[...], ...]
        }
    }

Embeddings are kept per location (the client's location_id: the room / device the audio was recorded on) rather than
averaged into one vector. A microphone, its room and its gain all shift an embedding; averaging clips from different
mics blurs them into a profile that matches no mic well. Within one location the clips come from the same mic, so
they ARE averaged (into a centroid), which the CS-23 bake-off showed scores better than matching clip by clip.

Profiles are biometric data: they never go in the repository, not even as test fixtures.

Matching
--------
    1. This location first: only the centroids enrolled under the request's location_id take part. If the best
       person clears the threshold AND beats the runner-up (a different person) by the margin, they are the speaker.
    2. Otherwise everyone: every centroid of every person, same rules.
    3. Otherwise the speaker is unknown (UNRECOGNIZED_SPEAKER).

A person with no samples at this location is simply absent from step 1 and found in step 2; a missing or new
location_id goes straight to step 2. A person's score is the best of their centroids that take part.

Speech shorter than min_seconds is not judged at all ('too_short'): an embedding of "yes" is not worth trusting. What
to do then (keep the last speaker mid-conversation, or treat it as unknown) is the caller's decision.

Settings
--------
The threshold, margin, minimum length and profiles directory come from the 'speaker_id' block of the ASR server's
--json config (see settings_from_dict, and AmadeoWhisperX.load_json_config for the whole file), with optional per-location overrides of threshold and margin - different mics will want
slightly different values. They are never taken from a client: a client that could lower the threshold could walk
past an agent's allow_unknown_speakers gate.
"""

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The name an unknown voice is given, in speaker tags, handoff notes and the reply to the client
UNRECOGNIZED_SPEAKER = 'Unrecognized voice'

# The embedding model the profiles are made with, unless the config says otherwise. It is what pyannote's
# speaker-diarization-community-1 pipeline (shipped with WhisperX) uses internally, so it needs no new packages.
DEFAULT_EMBEDDING_MODEL = 'pyannote/wespeaker-voxceleb-resnet34-LM'

# Defaults from the CS-23 bake-off (2026-09-27; four household voices, one mic, wespeaker ResNet34): with these, no
# clip of 1.5 s or more was ever given to the wrong person, and at 2 s about 92% were identified and 8% left unknown.
DEFAULT_THRESHOLD = 0.50
DEFAULT_MARGIN = 0.05
DEFAULT_MIN_SECONDS = 1.0

# Field clips (see field_clip_path): how long they are kept, and the folders for clips with no speaker / location
DEFAULT_FIELD_RETENTION_DAYS = 90
FIELD_UNRECOGNIZED_FOLDER = 'unrecognized'
FIELD_NO_LOCATION_FOLDER = 'no-location'

# Every key the 'speaker_id' block may hold (see settings_from_dict); anything else is warned about and ignored
SPEAKER_ID_KEYS = frozenset({
    'profiles_dir', 'samples_dir', 'embedding_model', 'threshold', 'margin', 'min_seconds', 'locations',
    'field_samples_dir', 'save_known_field_clips', 'save_unknown_field_clips', 'field_retention_days',
})

# Where a speaker comes from; also what the ASR server reports as 'speaker_status'
STATUS_IDENTIFIED = 'identified'    # a person matched
STATUS_UNKNOWN = 'unknown'          # judged, and nobody matched well enough
STATUS_TOO_SHORT = 'too_short'      # too little speech to judge
STATUS_NO_PROFILES = 'no_profiles'  # nobody is enrolled (so every voice is unknown)
STATUS_DISABLED = 'disabled'        # the ASR server has no speaker-ID config
STATUS_ERROR = 'error'              # the embedding failed

# A profile's file name (and 'name') must be a plain, safe name - it becomes a path
_SAFE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')


# ------------------------------------------------------------------------------------------------------------ vectors

def normalize(vector: List[float]) -> List[float]:
    """
    Scales a vector to unit length, so a dot product between two of them is their cosine similarity.

    Args:
        vector: the vector.

    Returns:
        List[float]: the unit vector.

    Raises:
        ValueError: the vector is empty or all zeros (it has no direction to compare).
    """
    length = math.sqrt(sum(x * x for x in vector))
    if not vector or length == 0.0:
        raise ValueError("cannot normalize an empty or all-zero vector")
    return [x / length for x in vector]


def cosine(a: List[float], b: List[float]) -> float:
    """
    Cosine similarity of two vectors: 1 = same direction, 0 = unrelated, -1 = opposite.

    Args:
        a, b: vectors of the same length (not necessarily unit length).

    Returns:
        float: the cosine similarity.

    Raises:
        ValueError: the lengths differ, or either vector is all zeros.
    """
    if len(a) != len(b):
        raise ValueError(f"vector lengths differ ({len(a)} vs {len(b)})")
    dot = sum(x * y for x, y in zip(a, b))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norms == 0.0:
        raise ValueError("cannot compare an all-zero vector")
    return dot / norms


def centroid(vectors: List[List[float]]) -> List[float]:
    """
    The direction a set of embeddings shares: the mean of their unit vectors, itself scaled to unit length.

    Args:
        vectors: one or more vectors of the same length.

    Returns:
        List[float]: the unit-length centroid.
    """
    units = [normalize(v) for v in vectors]
    mean = [sum(column) / len(units) for column in zip(*units)]
    return normalize(mean)


# ------------------------------------------------------------------------------------------------------------ settings

@dataclass
class SpeakerIdSettings:
    """
    The rules for accepting a match (see the module docstring), with optional per-location overrides.

    Attributes:
        profiles_dir: the directory of <name>.json voice profiles.
        embedding_model: the embedding model; profiles made with another model are ignored.
        threshold: the best person's score must be at least this.
        margin: ...and beat the runner-up (another person) by at least this.
        min_seconds: shorter speech is not judged (STATUS_TOO_SHORT).
        locations: location_id -> {'threshold': x, 'margin': y}, either key optional.
        samples_dir: enrollment only - the <person>/<location_id>/*.wav tree enroll_voice.py --all reads (see
            find_enrollment_samples); '' for none. The ASR server itself never reads it.
        field_samples_dir: where the ASR server saves the speech it judges ("field clips", see field_clip_path),
            for reviewing and enrolling later; '' for none.
        save_known_field_clips: allow saving clips of identified speakers - the master switch; a client must also
            ask (see field_clip_kind).
        save_unknown_field_clips: the same, for unrecognized voices.
        field_retention_days: field clips older than this are deleted; 0 keeps them for ever.
    """
    profiles_dir: str = ''
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    threshold: float = DEFAULT_THRESHOLD
    margin: float = DEFAULT_MARGIN
    min_seconds: float = DEFAULT_MIN_SECONDS
    locations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    samples_dir: str = ''
    field_samples_dir: str = ''
    save_known_field_clips: bool = False
    save_unknown_field_clips: bool = False
    field_retention_days: float = DEFAULT_FIELD_RETENTION_DAYS

    def rules_for(self, location_id: Optional[str]) -> Dict[str, float]:
        """
        The threshold and margin to use for audio from a location.

        Args:
            location_id: the client's location_id ('' or None for none).

        Returns:
            Dict[str, float]: {'threshold': ..., 'margin': ...} - the location's overrides where it has them.
        """
        override = self.locations.get(location_id or '', {})
        return {'threshold': override.get('threshold', self.threshold), 'margin': override.get('margin', self.margin)}


def _number(value: Any, key: str, low: float, high: float) -> float:
    """
    Checks one numeric setting.

    Args:
        value: the value from the config.
        key: its name, for the error message.
        low, high: the allowed range (inclusive).

    Returns:
        float: the value.

    Raises:
        TypeError: it is not a number (booleans are not numbers here).
        ValueError: it is out of range.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"speaker-ID setting '{key}' must be a number, not {value!r}")
    if not low <= value <= high:
        raise ValueError(f"speaker-ID setting '{key}' must be between {low} and {high}, not {value}")
    return float(value)


def unknown_settings(data: Dict[str, Any], known) -> List[str]:
    """
    The keys of a config object that are not recognised - usually a typo, or a setting put at the wrong level (e.g.
    'log_file' inside the 'speaker_id' block). They would otherwise be ignored without a word. Keys starting with '_'
    ('_comment') are notes, never settings, and are not reported.

    Args:
        data: the config object.
        known: the keys it may hold.

    Returns:
        List[str]: the unknown keys, sorted.
    """
    return sorted(key for key in data if key not in known and not str(key).startswith('_'))


def settings_from_dict(data: Dict[str, Any], other_level_keys=frozenset()) -> SpeakerIdSettings:
    """
    Builds and checks settings from the 'speaker_id' block of the ASR server's --json config. Private (it points at
    biometric data), so the config lives with the other private configs, never in the repository. Example:

        {
            "profiles_dir": "/path/to/voice-profiles",
            "samples_dir": "/path/to/voice-samples",
            "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
            "threshold": 0.50,
            "margin": 0.05,
            "min_seconds": 1.0,
            "locations": {"office": {"threshold": 0.55}},
            "field_samples_dir": "/path/to/field-samples",
            "save_known_field_clips": false,
            "save_unknown_field_clips": true,
            "field_retention_days": 90
        }

    Only profiles_dir is required. samples_dir is for enrollment only (enroll_voice.py --all). Saving field clips
    needs field_samples_dir. A key it does not know is logged as a warning and ignored (see unknown_settings).

    Args:
        data: the parsed JSON object.
        other_level_keys: the keys of the config level this block sits in (the ASR server's top-level settings); an
            unknown key that is one of them gets a hint that it belongs there instead.

    Returns:
        SpeakerIdSettings: the checked settings.

    Raises:
        TypeError / ValueError: a setting is missing, of the wrong type or out of range.
    """
    if not isinstance(data, dict):
        raise TypeError("the speaker-ID config must be a JSON object")
    profiles_dir = data.get('profiles_dir')
    if not isinstance(profiles_dir, str) or not profiles_dir:
        raise ValueError("the speaker-ID config needs 'profiles_dir' (the directory of voice profiles)")
    model = data.get('embedding_model', DEFAULT_EMBEDDING_MODEL)
    if not isinstance(model, str) or not model:
        raise TypeError("speaker-ID setting 'embedding_model' must be a non-empty string")
    samples_dir = data.get('samples_dir', '')
    if not isinstance(samples_dir, str):
        raise TypeError("speaker-ID setting 'samples_dir' must be a string (a directory)")

    settings = SpeakerIdSettings(
        profiles_dir=os.path.expanduser(profiles_dir),
        embedding_model=model,
        threshold=_number(data.get('threshold', DEFAULT_THRESHOLD), 'threshold', -1.0, 1.0),
        margin=_number(data.get('margin', DEFAULT_MARGIN), 'margin', 0.0, 2.0),
        min_seconds=_number(data.get('min_seconds', DEFAULT_MIN_SECONDS), 'min_seconds', 0.0, 60.0),
        samples_dir=os.path.expanduser(samples_dir) if samples_dir else '',
        field_retention_days=_number(data.get('field_retention_days', DEFAULT_FIELD_RETENTION_DAYS),
                                     'field_retention_days', 0.0, 36500.0),
    )

    field_dir = data.get('field_samples_dir', '')
    if not isinstance(field_dir, str):
        raise TypeError("speaker-ID setting 'field_samples_dir' must be a string (a directory)")
    settings.field_samples_dir = os.path.expanduser(field_dir) if field_dir else ''
    for key in ('save_known_field_clips', 'save_unknown_field_clips'):
        value = data.get(key, False)
        if not isinstance(value, bool):
            raise TypeError(f"speaker-ID setting '{key}' must be true or false")
        setattr(settings, key, value)
    if (settings.save_known_field_clips or settings.save_unknown_field_clips) and not settings.field_samples_dir:
        raise ValueError("saving field clips needs 'field_samples_dir' (where to put them)")

    locations = data.get('locations', {})
    if not isinstance(locations, dict):
        raise TypeError("speaker-ID setting 'locations' must be an object of location_id -> overrides")
    for location_id, override in locations.items():
        if not isinstance(override, dict):
            raise TypeError(f"speaker-ID location '{location_id}' must be an object")
        unknown = set(override) - {'threshold', 'margin'}
        if unknown:
            raise ValueError(f"speaker-ID location '{location_id}' may only override threshold / margin, not {sorted(unknown)}")
        checked = {}
        if 'threshold' in override:
            checked['threshold'] = _number(override['threshold'], f"locations.{location_id}.threshold", -1.0, 1.0)
        if 'margin' in override:
            checked['margin'] = _number(override['margin'], f"locations.{location_id}.margin", 0.0, 2.0)
        settings.locations[location_id] = checked

    for key in unknown_settings(data, SPEAKER_ID_KEYS):
        hint = " - it belongs at the top level of the config, not inside speaker_id" if key in other_level_keys else ''
        logger.warning(f"Unknown speaker_id setting '{key}' ignored{hint}.")
    return settings


def enrollment_target(server_config: Dict[str, Any]) -> Tuple[str, int, SpeakerIdSettings]:
    """
    What enrollment needs from the ASR server's --json config: where to reach the server, and its speaker-ID
    settings (profiles_dir, samples_dir). Reading the server's own file keeps enrollment and recognition from
    drifting apart. Kept here, free of the server's heavy imports, so the enrollment script (media env) can use it.

    Args:
        server_config: the parsed --json config of the ASR server (transcribe_server.py).

    Returns:
        Tuple[str, int, SpeakerIdSettings]: (host, port, settings). The host is the address the server binds; one
        that only makes sense for binding ('0.0.0.0', '::') becomes 'localhost'. A missing host / port comes back
        as '' / 0, for the caller to fill with its own defaults.

    Raises:
        TypeError / ValueError: the config is not an object, host / port have the wrong type, or it has no valid
            'speaker_id' block.
    """
    if not isinstance(server_config, dict):
        raise TypeError("the ASR server config must be a JSON object")
    host = server_config.get('host', '')
    port = server_config.get('port', 0)
    if not isinstance(host, str):
        raise TypeError("the ASR server config's 'host' must be a string")
    if isinstance(port, bool) or not isinstance(port, int):
        raise TypeError("the ASR server config's 'port' must be an integer")
    if 'speaker_id' not in server_config:
        raise ValueError("the ASR server config has no 'speaker_id' block, so the server does not do speaker identification")
    if host in ('0.0.0.0', '::'):
        host = 'localhost'
    return host, port, settings_from_dict(server_config['speaker_id'])


def find_enrollment_samples(samples_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Finds the WAVs to enroll under a samples directory laid out as <samples_dir>/<person>/<location_id>/*.wav: one
    folder per person (its name is the profile name, so it must be safe - see is_safe_name), one folder per location
    inside it (the client's location_id, where those clips were recorded). Anything else is skipped with a warning:
    loose files, a person folder with an unsafe name, a location folder with no WAVs.

    Args:
        samples_dir: the samples directory.

    Returns:
        Dict[str, Dict[str, List[str]]]: person -> location_id -> the WAV paths, sorted. Empty if the directory does
        not exist.
    """
    found = {}
    if not os.path.isdir(samples_dir):
        logger.warning(f"Voice samples directory {samples_dir} does not exist.")
        return found
    for person in sorted(os.listdir(samples_dir)):
        person_dir = os.path.join(samples_dir, person)
        if not os.path.isdir(person_dir):
            logger.warning(f"Skipping {person_dir}: expected a folder per person.")
            continue
        if not is_safe_name(person):
            logger.warning(f"Skipping {person_dir}: '{person}' is not a safe profile name (letters, digits, '_', '.', '-').")
            continue
        for location_id in sorted(os.listdir(person_dir)):
            location_dir = os.path.join(person_dir, location_id)
            if not os.path.isdir(location_dir):
                logger.warning(f"Skipping {location_dir}: expected a folder per location inside {person_dir}.")
                continue
            wavs = sorted(os.path.join(location_dir, f) for f in os.listdir(location_dir)
                          if f.lower().endswith('.wav') and os.path.isfile(os.path.join(location_dir, f)))
            if not wavs:
                logger.warning(f"Skipping {location_dir}: no WAV files.")
                continue
            found.setdefault(person, {})[location_id] = wavs
    return found


def prune_plan(profiles: List['VoiceProfile'], found: Dict[str, Dict[str, List[str]]], only_name: str = '',
               only_location: str = '') -> Tuple[Dict[str, List[str]], List[str]]:
    """
    What enroll_voice.py --all --prune removes so the profiles match the samples tree: every enrolled location that
    no longer has a folder <person>/<location_id> in the tree. A profile left with no locations at all is deleted
    rather than kept empty. --name / --location-id narrow what may be removed, exactly as they narrow what is
    enrolled.

    Args:
        profiles: the enrolled profiles (every one in the profiles directory).
        found: the WHOLE samples tree, from find_enrollment_samples (not narrowed by name or location).
        only_name: only prune this person ('' for everyone).
        only_location: only prune this location_id ('' for every location).

    Returns:
        Tuple[Dict[str, List[str]], List[str]]: (profile name -> the locations to drop from it, sorted; the profile
        names to delete outright, sorted).
    """
    drop_locations, delete_profiles = {}, []
    for profile in profiles:
        if only_name and profile.name != only_name:
            continue
        in_tree = set(found.get(profile.name, {}))
        drop = sorted(location for location in profile.locations
                      if location not in in_tree and (not only_location or location == only_location))
        if not drop:
            continue
        if len(drop) == len(profile.locations):
            delete_profiles.append(profile.name)
        else:
            drop_locations[profile.name] = drop
    return drop_locations, sorted(delete_profiles)


# ------------------------------------------------------------------------------------------------------ field clips

def field_folder_name(name: Any, fallback: str) -> str:
    """
    Makes a location_id or speaker name safe as one folder name. A location_id arrives from the network, so
    anything that could climb out of the field clips directory ('..', '/') must not survive.

    Args:
        name: the location_id or name.
        fallback: used when nothing usable is left (e.g. no location_id).

    Returns:
        str: letters, digits, '_', '.', '-' only; never starting with '.'; at most 64 characters.
    """
    if not isinstance(name, str):
        return fallback
    cleaned = re.sub(r'[^A-Za-z0-9_.-]', '_', name.strip()).lstrip('._-')[:64].replace('..', '_')
    return cleaned if is_safe_name(cleaned) else fallback


def field_clip_kind(settings: SpeakerIdSettings, status: str, client_known: bool = False,
                    client_unknown: bool = False) -> str:
    """
    Whether a judged chunk of speech is saved as a field clip. BOTH sides must agree: the ASR server's config
    (save_*_field_clips - the master switch, which owns the disk and the retention) and the request (the client
    whose microphone it is, which opts in per kind; a client that says nothing is never recorded). So the server
    can turn all recording off at once, and each spot decides for itself whether it may be recorded - a client can
    only narrow what the server allows, never widen it.

    Args:
        settings: the speaker-ID settings.
        status: the match status (STATUS_ values). Only judged speech is saved: identified (known) or unknown /
            no_profiles (unrecognized). Too short, disabled and error were never judged.
        client_known: the request's save_known_field_clips.
        client_unknown: the request's save_unknown_field_clips.

    Returns:
        str: 'known', 'unknown', or '' if it is not saved.
    """
    if not settings.field_samples_dir:
        return ''
    if status == STATUS_IDENTIFIED and settings.save_known_field_clips and client_known is True:
        return 'known'
    if status in (STATUS_UNKNOWN, STATUS_NO_PROFILES) and settings.save_unknown_field_clips and client_unknown is True:
        return 'unknown'
    return ''


def field_clip_path(field_dir: str, location_id: str, speaker_folder: str, when: float) -> str:
    """
    Where a field clip goes: <field_dir>/<location>/<speaker>/<YYYYMMDD-HHMMSS-ffffff>.wav - the same
    <person>/<location> split as the samples tree (inverted, so one spot's clips sit together), so reviewing a clip
    means moving it into <samples_dir>/<person>/<location_id>/.

    Args:
        field_dir: the field clips directory.
        location_id: the request's location_id ('' goes under FIELD_NO_LOCATION_FOLDER).
        speaker_folder: the profile name of the identified speaker, or FIELD_UNRECOGNIZED_FOLDER.
        when: the time of the clip (time.time()).

    Returns:
        str: the path (its folders are not created here).
    """
    stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(when)) + f"-{int((when % 1) * 1_000_000):06d}"
    return os.path.join(field_dir, field_folder_name(location_id, FIELD_NO_LOCATION_FOLDER),
                        field_folder_name(speaker_folder, FIELD_UNRECOGNIZED_FOLDER), f"{stamp}.wav")


def expired_field_clips(field_dir: str, retention_days: float, now: float) -> List[str]:
    """
    The field clips older than the retention period, by file modification time.

    Args:
        field_dir: the field clips directory.
        retention_days: how long clips are kept; 0 keeps them for ever (nothing expires).
        now: the current time (time.time()).

    Returns:
        List[str]: the .wav files to delete. Empty if the directory does not exist.
    """
    if retention_days <= 0 or not os.path.isdir(field_dir):
        return []
    cutoff = now - retention_days * 86400
    expired = []
    for folder, _, files in os.walk(field_dir):
        for file_name in files:
            path = os.path.join(folder, file_name)
            try:
                if file_name.lower().endswith('.wav') and os.path.getmtime(path) < cutoff:
                    expired.append(path)
            except OSError:
                continue    # removed meanwhile
    return sorted(expired)


# ------------------------------------------------------------------------------------------------------------ profiles

def is_safe_name(name: Any) -> bool:
    """
    Whether a person's name can be used as a profile file name (letters, digits, '_', '.', '-'; no path parts).

    Args:
        name: the name.

    Returns:
        bool: True if it is safe.
    """
    return isinstance(name, str) and bool(_SAFE_NAME.match(name)) and '..' not in name


@dataclass
class VoiceProfile:
    """
    One enrolled person.

    Attributes:
        name: the file-safe name ('kevin' for kevin.json).
        display_name: what the person is called in speaker tags ('Kevin'); defaults to name in title case.
        model: the embedding model the embeddings were made with.
        locations: location_id -> the raw embeddings enrolled there (kept so they can be re-averaged or extended).
    """
    name: str
    display_name: str = ''
    model: str = DEFAULT_EMBEDDING_MODEL
    locations: Dict[str, List[List[float]]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name.replace('_', ' ').title()

    def centroids(self) -> Dict[str, List[float]]:
        """
        One unit-length centroid per location that has embeddings.

        Returns:
            Dict[str, List[float]]: location_id -> centroid.
        """
        return {location: centroid(vectors) for location, vectors in self.locations.items() if vectors}

    def add(self, location_id: str, embeddings: List[List[float]], replace: bool = False):
        """
        Enrolls embeddings at a location, leaving the other locations alone.

        Args:
            location_id: the location (room / device) they were recorded at.
            embeddings: the new embeddings.
            replace: True to drop the location's old embeddings first; False to add to them.
        """
        kept = [] if replace else self.locations.get(location_id, [])
        self.locations[location_id] = kept + [list(map(float, e)) for e in embeddings]

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns:
            Dict[str, Any]: the profile in its JSON layout (see the module docstring).
        """
        return {'name': self.name, 'display_name': self.display_name, 'model': self.model, 'locations': self.locations}


def profile_from_dict(data: Dict[str, Any], name: str = '') -> VoiceProfile:
    """
    Builds and checks a profile from its JSON layout.

    Args:
        data: the parsed JSON object.
        name: the name to use when the data has none (the file name, without .json).

    Returns:
        VoiceProfile: the profile.

    Raises:
        TypeError / ValueError: the layout is wrong, or the embeddings are not all the same length.
    """
    if not isinstance(data, dict):
        raise TypeError("a voice profile must be a JSON object")
    name = data.get('name') or name
    if not is_safe_name(name):
        raise ValueError(f"voice profile name {name!r} is not a safe file name")
    locations = data.get('locations', {})
    if not isinstance(locations, dict):
        raise TypeError(f"voice profile '{name}': 'locations' must be an object")
    length = None
    for location_id, vectors in locations.items():
        if not isinstance(vectors, list) or not all(isinstance(v, list) and v for v in vectors):
            raise TypeError(f"voice profile '{name}': location '{location_id}' must be a list of embeddings")
        for vector in vectors:
            if length is None:
                length = len(vector)
            if len(vector) != length or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vector):
                raise ValueError(f"voice profile '{name}': every embedding must be {length} numbers")
    return VoiceProfile(name=name, display_name=data.get('display_name') or '',
                        model=data.get('model') or DEFAULT_EMBEDDING_MODEL,
                        locations={k: [list(map(float, v)) for v in vs] for k, vs in locations.items()})


def profile_path(profiles_dir: str, name: str) -> str:
    """
    Where a person's profile is stored.

    Args:
        profiles_dir: the profiles directory.
        name: the person's file-safe name.

    Returns:
        str: <profiles_dir>/<name>.json

    Raises:
        ValueError: the name is not safe as a file name.
    """
    if not is_safe_name(name):
        raise ValueError(f"{name!r} is not a safe profile name (letters, digits, '_', '.', '-')")
    return os.path.join(profiles_dir, f"{name}.json")


def load_profile(path: str) -> VoiceProfile:
    """
    Reads one profile file.

    Args:
        path: the <name>.json file.

    Returns:
        VoiceProfile: the profile.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return profile_from_dict(json.load(f), os.path.splitext(os.path.basename(path))[0])


def save_profile(profiles_dir: str, profile: VoiceProfile) -> str:
    """
    Writes a profile, replacing any earlier file for the person. The file is written beside the old one and then
    renamed over it, so a reader never sees half a profile. It is made readable by its owner (and group) only.

    Args:
        profiles_dir: the profiles directory (created if missing).
        profile: the profile.

    Returns:
        str: the path written.
    """
    path = profile_path(profiles_dir, profile.name)
    os.makedirs(profiles_dir, exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as f:
        json.dump(profile.to_dict(), f)
    os.chmod(temporary, 0o660)
    os.replace(temporary, path)
    return path


def load_profiles(profiles_dir: str, model: str = '') -> List[VoiceProfile]:
    """
    Reads every profile in a directory. A file that cannot be read, or was made with a different embedding model
    (its embeddings are not comparable), is skipped with a warning rather than stopping the others.

    Args:
        profiles_dir: the profiles directory.
        model: the embedding model in use; '' accepts any.

    Returns:
        List[VoiceProfile]: the usable profiles, in file-name order. Empty if the directory does not exist.
    """
    profiles = []
    if not os.path.isdir(profiles_dir):
        logger.warning(f"Voice profile directory {profiles_dir} does not exist; nobody is enrolled.")
        return profiles
    for file_name in sorted(os.listdir(profiles_dir)):
        if not file_name.endswith('.json'):
            continue
        path = os.path.join(profiles_dir, file_name)
        try:
            profile = load_profile(path)
        except (OSError, ValueError, TypeError) as e:
            logger.warning(f"Skipping voice profile {path}: {e}")
            continue
        if model and profile.model != model:
            logger.warning(f"Skipping voice profile {path}: made with '{profile.model}', but the server uses '{model}'.")
            continue
        profiles.append(profile)
    return profiles


def profiles_signature(profiles_dir: str) -> tuple:
    """
    A cheap fingerprint of the profiles directory (each profile's name, size and modification time), so a server can
    notice a new enrollment and reload without being restarted.

    Args:
        profiles_dir: the profiles directory.

    Returns:
        tuple: changes whenever a profile is added, removed or rewritten.
    """
    try:
        entries = sorted(e for e in os.listdir(profiles_dir) if e.endswith('.json'))
    except OSError:
        return ()
    signature = []
    for entry in entries:
        try:
            stat = os.stat(os.path.join(profiles_dir, entry))
            signature.append((entry, stat.st_size, stat.st_mtime_ns))
        except OSError:
            continue
    return tuple(signature)


# ------------------------------------------------------------------------------------------------------------ matching

@dataclass
class SpeakerMatch:
    """
    The outcome of identify().

    Attributes:
        status: STATUS_IDENTIFIED, STATUS_UNKNOWN, STATUS_TOO_SHORT or STATUS_NO_PROFILES.
        speaker: the person's display name if identified; UNRECOGNIZED_SPEAKER if unknown or nobody is enrolled;
            '' if too short (the caller decides).
        score: the best person's score in the step that decided (0.0 if nothing was scored).
        runner_up: the second-best person's score in that step (None if there was no second person).
        step: 'location' or 'everyone' - which step decided; '' if none did.
        scores: person display name -> score in the last step run, for logging.
    """
    status: str
    speaker: str = ''
    score: float = 0.0
    runner_up: Optional[float] = None
    step: str = ''
    scores: Dict[str, float] = field(default_factory=dict)


def _score(embedding: List[float], candidates: Dict[str, List[List[float]]]) -> Dict[str, float]:
    """
    Each person's score: their best cosine similarity over the centroids taking part.

    Args:
        embedding: the unit-length embedding of the speech.
        candidates: display name -> that person's centroids taking part in this step.

    Returns:
        Dict[str, float]: display name -> score, for the people with at least one centroid.
    """
    return {person: max(sum(x * y for x, y in zip(embedding, c)) for c in centroids)
            for person, centroids in candidates.items() if centroids}


def _decide(scores: Dict[str, float], threshold: float, margin: float):
    """
    Applies the threshold and margin to one step's scores.

    Args:
        scores: display name -> score.
        threshold, margin: see SpeakerIdSettings.

    Returns:
        (winner or None, best score, runner-up score or None)
    """
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked:
        return None, 0.0, None
    best_person, best = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else None
    if best < threshold or (runner_up is not None and best - runner_up < margin):
        return None, best, runner_up
    return best_person, best, runner_up


class SpeakerIndex:
    """
    The enrolled profiles, ready to match against: each person's centroid per location, computed once.
    """

    def __init__(self, profiles: List[VoiceProfile]):
        """
        Args:
            profiles: the enrolled people (see load_profiles). Two profiles with the same display name are treated as
                one person (their centroids are pooled).
        """
        # display name -> [(location_id, centroid), ...]. A list rather than a dict keyed by location, so two
        # profiles with the same display name (two files for one person) pool their centroids instead of clashing.
        self.people: Dict[str, List[Tuple[str, List[float]]]] = {}
        # display name -> the (first) profile's file-safe name, for filing field clips under <person> folders that
        # match the samples tree
        self.file_names: Dict[str, str] = {}
        self.dimension = None
        for profile in profiles:
            self.file_names.setdefault(profile.display_name, profile.name)
            for location_id, vector in profile.centroids().items():
                if self.dimension is None:
                    self.dimension = len(vector)
                if len(vector) != self.dimension:
                    logger.warning(f"Voice profile '{profile.name}' at '{location_id}': embedding length {len(vector)} "
                                   f"does not match {self.dimension}; ignored.")
                    continue
                self.people.setdefault(profile.display_name, []).append((location_id, vector))

    def __len__(self) -> int:
        return len(self.people)

    def identify(self, embedding: Optional[List[float]], settings: SpeakerIdSettings, location_id: str = '',
                 seconds: Optional[float] = None) -> SpeakerMatch:
        """
        Works out who is speaking (see the module docstring for the rules).

        Args:
            embedding: the speech's embedding (any length scale); may be None when too short to embed.
            settings: the threshold / margin / minimum length, with per-location overrides.
            location_id: where the speech was recorded ('' for unknown).
            seconds: how long the speech is; None skips the length check.

        Returns:
            SpeakerMatch: the outcome.
        """
        if seconds is not None and seconds < settings.min_seconds:
            return SpeakerMatch(STATUS_TOO_SHORT)
        if not self.people:
            return SpeakerMatch(STATUS_NO_PROFILES, UNRECOGNIZED_SPEAKER)
        if embedding is None or len(embedding) != self.dimension:
            raise ValueError(f"embedding length {None if embedding is None else len(embedding)} does not match the "
                             f"profiles' {self.dimension}")

        unit = normalize(embedding)
        rules = settings.rules_for(location_id)

        # Step 1: only the centroids enrolled at this location
        if location_id:
            local = {person: [c for loc, c in entries if loc == location_id] for person, entries in self.people.items()}
            scores = _score(unit, local)
            if scores:
                winner, best, runner_up = _decide(scores, rules['threshold'], rules['margin'])
                if winner:
                    return SpeakerMatch(STATUS_IDENTIFIED, winner, best, runner_up, 'location', scores)

        # Step 2: everyone, every location
        scores = _score(unit, {person: [c for _, c in entries] for person, entries in self.people.items()})
        winner, best, runner_up = _decide(scores, rules['threshold'], rules['margin'])
        if winner:
            return SpeakerMatch(STATUS_IDENTIFIED, winner, best, runner_up, 'everyone', scores)
        return SpeakerMatch(STATUS_UNKNOWN, UNRECOGNIZED_SPEAKER, best, runner_up, '', scores)

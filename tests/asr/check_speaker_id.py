#!/usr/bin/env python3
"""CS-23: speaker identification rules (amadeo_utils.ai.asr.speaker_id), with fake embeddings.

Why
---
The ASR server decides who is speaking by matching a voice embedding against enrolled profiles. Whether an agent
that only answers known voices lets someone through depends entirely on these rules, so they are pinned here
without any model: threshold and margin, the too-short rule, the two-step location order, per-location overrides,
and the profile / config files.

What it proves (plain Python - no model, no numpy)
--------------------------------------------------
1. Vectors: cosine, normalize and centroid behave; an all-zero vector is refused.
2. Threshold: a best score under the threshold is unknown. Margin: a best score too close to a DIFFERENT person's is
   unknown; two centroids of the same person never compete with each other.
3. Too short: under min_seconds nothing is judged (status too_short, no speaker); nobody enrolled -> no_profiles.
4. Location order: this location's samples decide first; if they cannot, everyone's are tried; a person with no
   samples here is found in step 2; a step-1 winner stands even when step 2 would favour someone else; an unknown or
   missing location goes straight to step 2.
5. Per-location overrides of threshold / margin apply only to their location.
6. Config: defaults, overrides, and bad values rejected. Profiles: save / load round trip (owner+group only), a
   corrupt file or another model's profile skipped, unsafe names refused, add() keeps other locations, the
   directory signature changes when a profile is rewritten.
7. Enrollment: the ASR server's --json config gives enrollment its host / port / profiles_dir / samples_dir (a bind-all
   host becomes localhost; no speaker_id block is refused); the samples tree <person>/<location_id>/*.wav is found,
   and loose files, unsafe person names and WAV-less locations are skipped. Prune: an enrolled location with no folder
   is dropped, a profile left with none is deleted, and --name / --location-id narrow what may go.
8. Field clips: the settings (off by default, 90-day retention, strict booleans, a directory needed to save),
   which statuses are saved (the server's switch AND the client's opt-in), the <location>/<speaker>/<time>.wav layout, folder names that cannot escape the
   directory, which clips have expired, and the display name -> profile name map used to file them.
9. Unknown settings: a key the speaker_id block doesn't know is warned about (with a hint when it belongs at the
   top level, like log_file), known keys and '_' notes pass silently, and the settings still load.

Usage:  python check_speaker_id.py      (any Python 3 with the repo's src on the path)
"""
import json
import os
import stat
import sys
import tempfile
import time

# testlib finds the library under test (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testlib import settings  # noqa: E402
settings.use_src()

from amadeo_utils.ai.asr import speaker_id as sid  # noqa: E402

failures = []


def check(label, got, expected):
    """Records one comparison and prints PASS / FAIL."""
    if got == expected:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}: got {got!r}, expected {expected!r}")
        failures.append(label)


def raises(label, exception, function, *args):
    """Records whether calling function(*args) raises the given exception."""
    try:
        function(*args)
    except exception:
        print(f"PASS  {label}")
        return
    except Exception as e:     # the wrong exception is a failure too
        print(f"FAIL  {label}: raised {type(e).__name__}: {e}")
        failures.append(label)
        return
    print(f"FAIL  {label}: nothing raised")
    failures.append(label)


def close(a, b, tolerance=1e-9):
    """Whether two floats (or lists of floats) are equal within a tolerance."""
    if isinstance(a, list):
        return len(a) == len(b) and all(abs(x - y) <= tolerance for x, y in zip(a, b))
    return abs(a - b) <= tolerance


# Made-up people with made-up 3-D "voices". Rows are unit-ish directions; the numbers are chosen so the scores below
# are easy to reason about.
KEVIN = [1.0, 0.0, 0.0]
SAM = [0.0, 1.0, 0.0]
KIM = [0.0, 0.0, 1.0]


def profile(name, **locations):
    """A VoiceProfile with the given location -> embeddings."""
    return sid.VoiceProfile(name=name, locations={k: v for k, v in locations.items()})


SETTINGS = sid.SpeakerIdSettings(profiles_dir='unused', threshold=0.5, margin=0.1, min_seconds=1.0)

# 1. vectors
check("cosine of identical directions is 1", close(sid.cosine([2, 0], [5, 0]), 1.0), True)
check("cosine of orthogonal vectors is 0", close(sid.cosine([1, 0], [0, 3]), 0.0), True)
check("normalize gives unit length", close(sid.normalize([3.0, 4.0]), [0.6, 0.8]), True)
check("centroid averages directions, not magnitudes", close(sid.centroid([[10, 0], [0, 1]]), [2 ** -0.5, 2 ** -0.5]), True)
raises("an all-zero vector cannot be normalized", ValueError, sid.normalize, [0.0, 0.0])
raises("vectors of different lengths cannot be compared", ValueError, sid.cosine, [1, 0], [1, 0, 0])

# 2. threshold and margin
index = sid.SpeakerIndex([profile('kevin', den=[KEVIN]), profile('sam', den=[SAM])])
m = index.identify([1.0, 0.05, 0.0], SETTINGS, 'den', 2.0)
check("a clear match is identified", (m.status, m.speaker, m.step), (sid.STATUS_IDENTIFIED, 'Kevin', 'location'))
m = index.identify([0.3, 0.1, 0.95], SETTINGS, 'den', 2.0)
check("below the threshold: unknown", (m.status, m.speaker), (sid.STATUS_UNKNOWN, sid.UNRECOGNIZED_SPEAKER))
m = index.identify([1.0, 0.9, 0.0], SETTINGS, 'den', 2.0)
check("above the threshold but within the margin of another person: unknown", (m.status, m.speaker),
      (sid.STATUS_UNKNOWN, sid.UNRECOGNIZED_SPEAKER))
index2 = sid.SpeakerIndex([profile('kevin', den=[KEVIN], office=[[0.98, 0.2, 0.0]]), profile('sam', den=[SAM])])
m = index2.identify([1.0, 0.1, 0.0], SETTINGS, '', 2.0)
check("two centroids of the same person never compete on the margin", (m.status, m.speaker),
      (sid.STATUS_IDENTIFIED, 'Kevin'))
lone = sid.SpeakerIndex([profile('kevin', den=[KEVIN])])
check("one person enrolled: no runner-up, the threshold alone decides",
      lone.identify([1.0, 0.2, 0.0], SETTINGS, '', 2.0).speaker, 'Kevin')

# 3. too short / nobody enrolled
m = index.identify([1.0, 0.0, 0.0], SETTINGS, 'den', 0.6)
check("too short: not judged, no speaker", (m.status, m.speaker), (sid.STATUS_TOO_SHORT, ''))
check("too short is decided before anything is compared", index.identify(None, SETTINGS, 'den', 0.2).status, sid.STATUS_TOO_SHORT)
m = sid.SpeakerIndex([]).identify([1.0, 0.0, 0.0], SETTINGS, 'den', 2.0)
check("nobody enrolled: no_profiles, an unrecognized voice", (m.status, m.speaker), (sid.STATUS_NO_PROFILES, sid.UNRECOGNIZED_SPEAKER))
raises("an embedding of the wrong length is refused", ValueError, index.identify, [1.0, 0.0], SETTINGS, 'den', 2.0)

# 4. location order
# In the kitchen, Kim's kitchen samples look like 'KIM'; Kevin's kitchen samples look like a tilted Kevin. Sam has no
# kitchen samples at all.
house = sid.SpeakerIndex([
    profile('kevin', kitchen=[[0.8, 0.0, 0.6]], office=[KEVIN]),
    profile('kim', kitchen=[KIM], office=[[0.0, 0.6, 0.8]]),
    profile('sam', office=[SAM]),
])
m = house.identify([0.8, 0.0, 0.6], SETTINGS, 'kitchen', 3.0)
check("step 1: this location's samples decide", (m.speaker, m.step), ('Kevin', 'location'))
m = house.identify(SAM, SETTINGS, 'kitchen', 3.0)
check("someone with no samples here is found in step 2", (m.speaker, m.step), ('Sam', 'everyone'))
check("step 1 scores only people enrolled here", sorted(house.identify([0.8, 0.0, 0.6], SETTINGS, 'kitchen', 3.0).scores), ['Kevin', 'Kim'])
# [0.6, 0.0, 0.8]: kitchen scores Kim 0.8 vs Kevin 0.96 -> Kevin wins at the kitchen by 0.16 (> margin). Over every
# location Kevin's office sample would give Kevin only 0.6, so step 1 is what makes it Kevin - and it stands.
m = house.identify([0.6, 0.0, 0.8], SETTINGS, 'kitchen', 3.0)
check("a step-1 winner stands", (m.speaker, m.step), ('Kevin', 'location'))
m = house.identify([0.7, 0.0, 0.0], SETTINGS, 'garage', 3.0)
check("an unknown location goes straight to step 2", (m.speaker, m.step), ('Kevin', 'everyone'))
m = house.identify(SAM, SETTINGS, '', 3.0)
check("no location goes straight to step 2", (m.speaker, m.step), ('Sam', 'everyone'))
m = house.identify([-1.0, 0.0, -0.3], SETTINGS, 'kitchen', 3.0)
check("neither step matches: unknown", (m.status, m.speaker, m.step), (sid.STATUS_UNKNOWN, sid.UNRECOGNIZED_SPEAKER, ''))

# 5. per-location overrides
strict = sid.SpeakerIdSettings(profiles_dir='unused', threshold=0.5, margin=0.1, min_seconds=1.0,
                               locations={'den': {'threshold': 0.99}})
check("an override applies at its own location", index.identify([1.0, 0.3, 0.0], strict, 'den', 2.0).status, sid.STATUS_UNKNOWN)
check("...and nowhere else", index.identify([1.0, 0.3, 0.0], strict, 'porch', 2.0).speaker, 'Kevin')
check("rules_for falls back to the defaults", strict.rules_for('porch'), {'threshold': 0.5, 'margin': 0.1})

# 6. config
s = sid.settings_from_dict({'profiles_dir': '~/p'})
check("config defaults", (s.threshold, s.margin, s.min_seconds, s.embedding_model),
      (sid.DEFAULT_THRESHOLD, sid.DEFAULT_MARGIN, sid.DEFAULT_MIN_SECONDS, sid.DEFAULT_EMBEDDING_MODEL))
check("profiles_dir expands ~", s.profiles_dir, os.path.expanduser('~/p'))
s = sid.settings_from_dict({'profiles_dir': 'p', 'threshold': 0.6, 'margin': 0.02, 'min_seconds': 1.5,
                            'locations': {'office': {'threshold': 0.7}}})
check("config values and overrides are read", (s.threshold, s.margin, s.min_seconds, s.rules_for('office')),
      (0.6, 0.02, 1.5, {'threshold': 0.7, 'margin': 0.02}))
raises("config without profiles_dir is refused", ValueError, sid.settings_from_dict, {'threshold': 0.5})
raises("a threshold that is not a number is refused", TypeError, sid.settings_from_dict, {'profiles_dir': 'p', 'threshold': 'high'})
raises("a boolean is not a number", TypeError, sid.settings_from_dict, {'profiles_dir': 'p', 'margin': True})
raises("an out-of-range margin is refused", ValueError, sid.settings_from_dict, {'profiles_dir': 'p', 'margin': -0.1})
raises("a location may only override threshold / margin", ValueError, sid.settings_from_dict,
       {'profiles_dir': 'p', 'locations': {'office': {'min_seconds': 0.1}}})

# 6b. profiles on disk
with tempfile.TemporaryDirectory() as folder:
    p = profile('kevin', den=[KEVIN, [0.9, 0.1, 0.0]])
    path = sid.save_profile(folder, p)
    check("a profile is saved as <name>.json", os.path.basename(path), 'kevin.json')
    check("the profile file is not world-readable", stat.S_IMODE(os.stat(path).st_mode) & 0o007, 0)
    loaded = sid.load_profile(path)
    check("save / load round trip", (loaded.name, loaded.display_name, loaded.locations), ('kevin', 'Kevin', p.locations))
    before = sid.profiles_signature(folder)
    loaded.add('office', [SAM])
    loaded.add('den', [KIM], replace=True)
    check("add() extends one location and replace only resets that one", loaded.locations, {'den': [KIM], 'office': [SAM]})
    time.sleep(0.01)
    sid.save_profile(folder, loaded)
    check("rewriting a profile changes the directory signature", sid.profiles_signature(folder) != before, True)

    with open(os.path.join(folder, 'broken.json'), 'w') as f:
        f.write('{not json')
    with open(os.path.join(folder, 'other.json'), 'w') as f:
        json.dump({'name': 'other', 'model': 'some/other-model', 'locations': {'den': [SAM]}}, f)
    with open(os.path.join(folder, 'notes.txt'), 'w') as f:
        f.write('ignored')
    check("unreadable files and other models' profiles are skipped", [x.name for x in sid.load_profiles(folder, sid.DEFAULT_EMBEDDING_MODEL)], ['kevin'])
    check("with no model given, any model's profile loads", sorted(x.name for x in sid.load_profiles(folder)), ['kevin', 'other'])
    check("a missing directory means nobody is enrolled", sid.load_profiles(os.path.join(folder, 'nope')), [])

raises("an unsafe profile name is refused", ValueError, sid.profile_path, '/tmp', '../escape')
check("safe names", [sid.is_safe_name(n) for n in ('kevin', 'mary-jo', 'j.r', '', '.hidden', 'a/b', 'a..b')],
      [True, True, True, False, False, False, False])
raises("embeddings of different lengths in one profile are refused", ValueError, sid.profile_from_dict,
       {'name': 'x', 'locations': {'den': [[1, 0], [1, 0, 0]]}})
check("display_name defaults to the name in title case", sid.VoiceProfile(name='mary_jo').display_name, 'Mary Jo')

# 7. enrollment: the server config and the samples tree
s = sid.settings_from_dict({'profiles_dir': 'p', 'samples_dir': '~/samples'})
check("samples_dir is read and expands ~", s.samples_dir, os.path.expanduser('~/samples'))
check("samples_dir defaults to none", sid.settings_from_dict({'profiles_dir': 'p'}).samples_dir, '')
raises("a samples_dir that is not a string is refused", TypeError, sid.settings_from_dict, {'profiles_dir': 'p', 'samples_dir': 5})

host, port, s = sid.enrollment_target({'host': 'asr.example', 'port': 1234, 'gpu': 1,
                                       'speaker_id': {'profiles_dir': '/v/profiles', 'samples_dir': '/v/samples'}})
check("enrollment reads host, port and the speaker_id block from the server config",
      (host, port, s.profiles_dir, s.samples_dir), ('asr.example', 1234, '/v/profiles', '/v/samples'))
check("a bind-all host is reached as localhost", sid.enrollment_target({'host': '0.0.0.0', 'speaker_id': {'profiles_dir': 'p'}})[0], 'localhost')
check("missing host / port are left to the caller", sid.enrollment_target({'speaker_id': {'profiles_dir': 'p'}})[:2], ('', 0))
raises("a server config without speaker_id is refused", ValueError, sid.enrollment_target, {'host': 'h', 'port': 1})
raises("a server config with a non-integer port is refused", TypeError, sid.enrollment_target, {'port': '1', 'speaker_id': {'profiles_dir': 'p'}})

with tempfile.TemporaryDirectory() as folder:
    def touch(*parts):
        """Creates an empty file (and its folders) under the samples folder."""
        path = os.path.join(folder, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'w').close()
        return path
    touch('kevin', 'den', 'b.wav')
    touch('kevin', 'den', 'a.WAV')
    touch('kevin', 'den', 'notes.txt')
    touch('kevin', 'porch', 'c.wav')
    touch('kevin', 'empty', 'readme.txt')     # a location with no WAVs
    touch('kevin', 'loose.wav')               # a clip outside any location folder
    touch('kim', 'den', 'd.wav')
    touch('.hidden', 'den', 'e.wav')         # not a safe profile name
    touch('stray.wav')                       # a file where a person folder should be
    found = sid.find_enrollment_samples(folder)
    check("the samples tree is found as person -> location -> sorted WAVs",
          {person: {loc: [os.path.basename(w) for w in wavs] for loc, wavs in places.items()} for person, places in found.items()},
          {'kevin': {'den': ['a.WAV', 'b.wav'], 'porch': ['c.wav']}, 'kim': {'den': ['d.wav']}})
    check("a missing samples directory finds nothing", sid.find_enrollment_samples(os.path.join(folder, 'nope')), {})

# prune: the profiles follow the samples tree
tree = {'kevin': {'den': ['a.wav']}, 'kim': {'den': ['b.wav'], 'porch': ['c.wav']}}
enrolled = [profile('kevin', den=[KEVIN], porch=[KEVIN]), profile('kim', den=[KIM], porch=[KIM]), profile('sam', den=[SAM])]
check("prune drops locations without a folder and deletes profiles left with none",
      sid.prune_plan(enrolled, tree), ({'kevin': ['porch']}, ['sam']))
check("prune with --name only touches that person", sid.prune_plan(enrolled, tree, only_name='sam'), ({}, ['sam']))
check("prune with --location-id only touches that location", sid.prune_plan(enrolled, tree, only_location='den'), ({}, ['sam']))
check("prune with nothing out of date removes nothing", sid.prune_plan(enrolled[1:2], tree), ({}, []))

# 8. field clips
s = sid.settings_from_dict({'profiles_dir': 'p'})
check("field clips are off by default, kept 90 days",
      (s.field_samples_dir, s.save_known_field_clips, s.save_unknown_field_clips, s.field_retention_days), ('', False, False, 90))
s = sid.settings_from_dict({'profiles_dir': 'p', 'field_samples_dir': '~/f', 'save_unknown_field_clips': True, 'field_retention_days': 30})
check("field clip settings are read", (s.field_samples_dir, s.save_known_field_clips, s.save_unknown_field_clips, s.field_retention_days),
      (os.path.expanduser('~/f'), False, True, 30))
raises("saving field clips needs a directory", ValueError, sid.settings_from_dict, {'profiles_dir': 'p', 'save_known_field_clips': True})
raises("save_*_field_clips must be a real boolean", TypeError, sid.settings_from_dict,
       {'profiles_dir': 'p', 'field_samples_dir': 'f', 'save_unknown_field_clips': 'yes'})
raises("a negative retention is refused", ValueError, sid.settings_from_dict, {'profiles_dir': 'p', 'field_retention_days': -1})

both = sid.settings_from_dict({'profiles_dir': 'p', 'field_samples_dir': 'f', 'save_known_field_clips': True, 'save_unknown_field_clips': True})
check("identified -> known; unknown and no_profiles -> unknown; never-judged statuses are not saved",
      [sid.field_clip_kind(both, st, True, True) for st in (sid.STATUS_IDENTIFIED, sid.STATUS_UNKNOWN, sid.STATUS_NO_PROFILES,
                                                  sid.STATUS_TOO_SHORT, sid.STATUS_DISABLED, sid.STATUS_ERROR)],
      ['known', 'unknown', 'unknown', '', '', ''])
only_unknown = sid.settings_from_dict({'profiles_dir': 'p', 'field_samples_dir': 'f', 'save_unknown_field_clips': True})
check("each kind has its own switch", [sid.field_clip_kind(only_unknown, st, True, True) for st in (sid.STATUS_IDENTIFIED, sid.STATUS_UNKNOWN)], ['', 'unknown'])
check("the client must opt in too: a client that says nothing is never recorded",
      [sid.field_clip_kind(both, sid.STATUS_IDENTIFIED), sid.field_clip_kind(both, sid.STATUS_UNKNOWN)], ['', ''])
check("...each kind separately",
      [sid.field_clip_kind(both, sid.STATUS_IDENTIFIED, True, False), sid.field_clip_kind(both, sid.STATUS_UNKNOWN, True, False)], ['known', ''])
check("...and only with a real true", sid.field_clip_kind(both, sid.STATUS_UNKNOWN, 'yes', 'yes'), '')
check("a client can't turn on what the server has off",
      sid.field_clip_kind(sid.settings_from_dict({'profiles_dir': 'p', 'field_samples_dir': 'f'}), sid.STATUS_UNKNOWN, True, True), '')

check("folder names: a plain name is kept; spaces become '_'; nothing may climb out",
      [sid.field_folder_name(n, 'x') for n in ('office-mic-2', 'Unrecognized voice', '../../etc', '..', '', None, '/abs/path')],
      ['office-mic-2', 'Unrecognized_voice', 'etc', 'x', 'x', 'x', 'abs_path'])
when = time.mktime((2026, 9, 28, 14, 5, 9, 0, 0, -1)) + 0.25
check("field clips are filed as <location>/<speaker>/<YYYYMMDD-HHMMSS-ffffff>.wav",
      os.path.relpath(sid.field_clip_path('/f', 'kitchen', 'kevin', when), '/f'), os.path.join('kitchen', 'kevin', '20260928-140509-250000.wav'))
check("no location / no speaker get their own folders",
      os.path.relpath(os.path.dirname(sid.field_clip_path('/f', '', '', when)), '/f'),
      os.path.join(sid.FIELD_NO_LOCATION_FOLDER, sid.FIELD_UNRECOGNIZED_FOLDER))

with tempfile.TemporaryDirectory() as folder:
    now = time.time()
    for name, age_days in (('old.wav', 91), ('new.wav', 5), ('old.txt', 200)):
        path = os.path.join(folder, 'kitchen', 'kevin', name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'w').close()
        os.utime(path, (now - age_days * 86400,) * 2)
    check("clips past the retention period have expired (WAVs only)",
          [os.path.basename(x) for x in sid.expired_field_clips(folder, 90, now)], ['old.wav'])
    check("retention 0 keeps everything", sid.expired_field_clips(folder, 0, now), [])
    check("a missing directory has nothing to expire", sid.expired_field_clips(os.path.join(folder, 'nope'), 90, now), [])

index = sid.SpeakerIndex([sid.VoiceProfile(name='mary_jo', locations={'den': [KEVIN]})])
check("the index maps a display name back to the profile's file name", index.file_names, {'Mary Jo': 'mary_jo'})

# 9. unknown settings
import logging  # noqa: E402


class Captured(logging.Handler):
    """Collects the warning messages logged while it is attached."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def warnings_from(function, *args, **kwargs):
    """Calls function and returns (its result, the warnings speaker_id logged meanwhile)."""
    captured = Captured()
    sid.logger.addHandler(captured)
    try:
        return function(*args, **kwargs), captured.messages
    finally:
        sid.logger.removeHandler(captured)


check("unknown_settings lists unknown keys, sorted, skipping '_' notes",
      sid.unknown_settings({'b': 1, 'a': 1, '_comment': 'x', 'ok': 1}, {'ok'}), ['a', 'b'])
s, warned = warnings_from(sid.settings_from_dict, {'profiles_dir': 'p', '_comment': 'notes', 'threshold': 0.6})
check("known keys and '_' notes: no warning", (warned, s.threshold), ([], 0.6))
s, warned = warnings_from(sid.settings_from_dict, {'profiles_dir': 'p', 'treshold': 0.6, 'log_file': '/l.log'},
                          other_level_keys={'host', 'log_file'})
check("an unknown key is warned about and ignored; the settings still load",
      (s.threshold, any("'treshold' ignored." in w for w in warned)), (sid.DEFAULT_THRESHOLD, True))
check("...with a hint when it belongs at the top level (log_file inside speaker_id)",
      any("'log_file' ignored - it belongs at the top level" in w for w in warned), True)

print(f"\n{len(failures)} failure(s)")
sys.exit(len(failures))

"""
Where the tests find things on this machine: the library under test, models, conda environments and tool configs.

Nothing machine-specific is written in the tests themselves. Paths come from ``tests/local_settings.json``
(gitignored; copy ``tests/local_settings.example.json`` and fill it in), or from the file named by the
``AMADEO_TEST_SETTINGS`` environment variable. A test that needs a setting which is missing fails with a message
naming the key, rather than guessing a path.

Every test script starts the same way::

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # up to tests/
    from testlib import settings       # noqa: E402
    settings.use_src()                  # the library under test goes first on sys.path

The number of ``..`` is the script's depth below ``tests/``.

Also usable from shell scripts: ``python3 settings.py --get llama_python`` prints one setting (paths expanded),
``python3 settings.py --path`` prints the library path.
"""
import json
import os
import sys

# tests/testlib/settings.py -> tests/ -> the repository root.
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TESTS_DIR)

# The library under test. AMADEO_TEST_SRC points a test at another copy of the library - for example the committed
# sources exported to a scratch folder (git archive HEAD src), to take a baseline from them or to prove a mutation
# is caught.
SRC = os.path.abspath(os.environ.get("AMADEO_TEST_SRC") or os.path.join(REPO_ROOT, "src"))

EXAMPLE_FILE = os.path.join(TESTS_DIR, "local_settings.example.json")
_cache = None


class MissingSetting(KeyError):
    """A setting the test needs is not in the local settings file."""


def settings_file() -> str:
    """The local settings file in use: AMADEO_TEST_SETTINGS, else tests/local_settings.json."""
    return os.environ.get("AMADEO_TEST_SETTINGS") or os.path.join(TESTS_DIR, "local_settings.json")


def load() -> dict:
    """
    Reads the local settings once. A missing file is an empty dict, so tests that need no local paths still run;
    get() then names whatever is missing.
    """
    global _cache
    if _cache is None:
        path = settings_file()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                _cache = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
        else:
            _cache = {}
    return _cache


def get(key: str, default=None, required: bool = True):
    """
    One setting, with '~' and environment variables expanded in string values.

    Args:
        key: The setting's name, as in local_settings.example.json.
        default: Returned when the setting is absent and not required.
        required: Raise MissingSetting (naming the file and the key) when the setting is absent.
    """
    value = load().get(key)
    if value in (None, ""):
        if required:
            raise MissingSetting(f"'{key}' is not set in {settings_file()} (see {EXAMPLE_FILE})")
        return default
    if isinstance(value, str):
        value = os.path.expandvars(os.path.expanduser(value))
    return value


def use_src():
    """Puts the library under test first on sys.path (once)."""
    if SRC not in sys.path:
        sys.path.insert(0, SRC)


def use_area(*parts: str):
    """
    Makes another test area's helper modules importable, e.g. use_area("llm", "llama", "streams") for the
    stream harness that the agent tests build on.
    """
    path = os.path.join(TESTS_DIR, *parts)
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def model_path(name: str) -> str:
    """
    A model file by its name inside the 'model_dir' setting. Baselines record models by this name only, so they
    carry no machine's directory layout.
    """
    return os.path.join(get("model_dir"), name)


def resolve_model(arg: str) -> str:
    """
    A --model argument as a file path: an absolute or existing path is used as given, anything else is a name
    inside 'model_dir' (the form baselines record).
    """
    if os.path.isabs(arg) or os.path.exists(arg):
        return arg
    return model_path(arg)


def model_name(path: str) -> str:
    """The inverse of model_path: a model path as baselines record it (its name inside model_dir)."""
    base = load().get("model_dir")
    if base:
        base = os.path.expanduser(base).rstrip("/") + "/"
        if path.startswith(base):
            return path[len(base):]
    return os.path.basename(path)


def scrub_model_dir(text: str) -> str:
    """Replaces this machine's model_dir in text (an error message quoting a model's path) with '<model_dir>/'."""
    base = load().get("model_dir")
    if base:
        text = text.replace(os.path.expanduser(base).rstrip("/") + "/", "<model_dir>/")
    return text


def write_capture(path: str, data) -> None:
    """
    Writes a golden capture: sorted, indented JSON with a trailing newline, and model_dir scrubbed, so a baseline
    is the same on every machine and carries none of their directory layouts.
    """
    text = json.dumps(data, indent=2, sort_keys=True, default=str)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(scrub_model_dir(text) + "\n")


if __name__ == "__main__":
    # Shell helper: --get KEY prints a setting (exit 1 with the reason if missing); --path prints SRC.
    if len(sys.argv) == 3 and sys.argv[1] == "--get":
        try:
            print(get(sys.argv[2]))
        except MissingSetting as e:
            print(e, file=sys.stderr)
            sys.exit(1)
    elif len(sys.argv) == 2 and sys.argv[1] == "--path":
        print(SRC)
    else:
        print("usage: settings.py --get KEY | --path", file=sys.stderr)
        sys.exit(64)

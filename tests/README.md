# Tests

Regression tests for the Amadeo library (`src/amadeo_utils`) and the scripts built on it. They are plain Python and
shell scripts, not a pytest suite: most of them need real model files, conda environments or live services, and
each one prints `PASS` / `FAIL` / `SKIP` lines and exits with its number of failures.

## Running them

```bash
cp tests/local_settings.example.json tests/local_settings.json   # once per machine, then fill in the paths
tests/run_tests.sh                    # every suite
tests/run_tests.sh --quick            # skip the slow checks (the ~4-minute model baselines)
tests/run_tests.sh --offline          # skip everything that touches the internet
tests/run_tests.sh llm/llama/agent    # only the suites under one folder
```

While another process holds the GPU, prefix with `CUDA_VISIBLE_DEVICES=`: every model-loading check here runs on
the CPU, and would otherwise fail trying to allocate GPU memory, which is not a code failure.

`AMADEO_TEST_SRC=/path/to/another/src` runs the tests against another copy of the library. Use it to take a
baseline from the committed code (`git archive HEAD src | tar -x -C /tmp/head`), or to check that a deliberately
broken copy is caught.

### Local settings

Nothing machine-specific lives in the tests. `tests/local_settings.json` (gitignored) holds this machine's paths.
Tests read it through `testlib/settings.py`, and a test that needs a missing setting fails naming the key:

| Key | Used for |
|---|---|
| `model_dir` | The folder of GGUF models. Baselines name models by their file name inside it. |
| `embedding_model` | The embedding GGUF that the stream families load. |
| `llama_python` | The python of the conda env with `llama_cpp` (the stream and agent suites). |
| `agent_tools_python` | The python of the env that runs the agent's tool scripts. |
| `bash_python` | The python of the env that runs `get_grades` (live and audit scripts only). |
| `media_python` | The python of the env with sounddevice / pygame / webrtcvad (the conversational AI client check). |
| `ai_tools_config_dir` | The tools' own config files (never in the repo): the `ai_tools` suite and the live scripts. |

A suite whose python is not set is skipped. A baseline whose model is not in `model_dir` is skipped, not failed.

## Layout

Folders mirror what they test. Each folder with a `suite.sh` is one suite, and `run_tests.sh` finds suites itself.

```
tests/
  run_tests.sh                  runs every suite.sh below it
  local_settings.example.json   template for local_settings.json (gitignored)
  testlib/
    settings.py                 paths from local_settings.json; model name <-> path; write_capture()
    suite_lib.sh                run / compare / skip / finish, sourced by every suite.sh
  llm/llama/streams/            role-play + knowledge-base stream families and StreamBase
    baselines/
  llm/llama/agent/              the tool-calling agent (ToolStream): tool loop, parsers, filters
    baselines/
  client_server/                AmadeoServer / AmadeoClient (the shared socket layer)
  ai_tools/                     the agent's tool scripts (scripts/ai/ai-tools/)
  conversational_ai/            the conversational AI pipeline: wake words, agents, ASR gate, client config
  security/                     network / file-write audits (run by hand; no suite)
```

## File names say what kind of test a file is

| Prefix | What it is | In a suite? |
|---|---|---|
| `check_*.py` | Asserts behaviour directly. Fast, usually no model. | Yes |
| `golden_*.py` | Captures what the code produces (a rendered prompt, a config, a scenario outcome) as JSON. The suite diffs it against `baselines/`. | Yes |
| `live_*.py` | Holds a real conversation on the GPU and/or the internet. The output is sampled, so a person reads it. | No |
| `probe_*.py` | Measures something (VRAM per context size, what a chat template renders). A record, not pass/fail. | No |
| `audit_*.py` | Traces what a real run touches (sockets, files) with `strace`. | No |
| anything else | A helper module (fixtures, scenario tables, a stand-in tool). | - |

Baselines are named `<what>_baseline[_<model>].json`, next to the harness that makes them, in `baselines/`.

## Adding tests

* **For code that already has a folder:** add a `check_*.py` (or `golden_*.py` plus its baseline) there, and one
  `run` / `compare` line to that folder's `suite.sh`.
* **For something unrelated** (a new area, e.g. text-to-speech): make a folder that mirrors it (`tests/tts/`), put
  the tests in it, and give it a `suite.sh`. Copy one of the existing ones; they are short. `run_tests.sh` picks it
  up with no other change, and `tests/run_tests.sh tts` runs it on its own. Areas share nothing but `testlib`, so a
  new area cannot break an old one.
* **Every test script starts the same way**, so it finds `testlib` and the library under test:

  ```python
  import os, sys
  sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # up to tests/
  from testlib import settings  # noqa: E402
  settings.use_src()
  ```

  Use one `".."` per folder level below `tests/`.
* **Keep it public-safe:** this folder is in the public repository. No paths from any machine (use
  `settings.get(...)`), no names or personal data in fixtures (make them up), no credentials. `write_capture()`
  scrubs `model_dir` out of golden captures for you.
* **A new baseline must come from code you trust**: it becomes the definition of "correct" that every later run is
  held to. Capture it from the committed code (see `AMADEO_TEST_SRC` above), check that it passes, then check that a
  deliberately broken copy fails it.

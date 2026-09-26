# ai-tools: tools for the tool-calling LLM server

Each directory here is one **tool**: a small Python script the tool server
(`scripts/ai/llm/llama/llama_stream/amadeo_agent_server.py`, the agent server) can offer to a local LLM. The model
decides when to call a tool and with which arguments; the server runs the script and hands the
answer back to the model.

| Tool | What it does | Flags |
|---|---|---|
| `get_weather/` | National Weather Service forecast (daily or hourly, with rainfall amounts) | outbound |
| `web_search/` | Searches the web through a self-hosted SearXNG instance | outbound, untrusted_output |
| `fetch_url/` | Fetches one public web page and returns its readable text | outbound, untrusted_output |
| `get_grades/` | Grades and assignments from Infinite Campus (wraps `scripts/tools/infinite_campus/`) | private |

Two tools are built into the server rather than living here: `get_datetime` and `calculator`
(`src/amadeo_utils/ai/llm/tools/builtin_tools.py`). `delegate` and `recall` are part of the
server's loop itself.

This page explains how to add a tool. Read the security section before writing one: **whatever a
tool can do, a prompt injection can try to make it do.**

---

## How a tool script works

A tool is an ordinary script that ends with one call:

```python
if __name__ == "__main__":
    tool_script_main(DEFINITION, handle)
```

`tool_script_main` (in `src/amadeo_utils/ai/llm/tools/script_tools.py`) gives every tool the same
command-line contract:

| Invocation | What happens |
|---|---|
| `my_tool.py --describe` | Prints `DEFINITION` as JSON. The server runs this **once at startup** to learn the tool's name, description, argument schema, security flags and timeout. |
| `my_tool.py --config FILE` | Runs **one call**. The arguments arrive as JSON on **stdin** (never on the command line, where `ps` would show them to every user). Prints one JSON answer on **stdout**. |

The answer is `{"ok": true, "result": ..., "summary": "..."}` or `{"ok": false, "error": "..."}`.
You never write it yourself - return a value from your handler, or raise `ToolError`.

What the server guarantees around each call:

* **A minimal environment.** The script gets only `PATH`, `HOME`, `LANG`/`LC_ALL`, `TZ` and a
  `PYTHONPATH` pointing at `src/`. Nothing else from the server's environment - no tokens, no keys.
  If your tool needs a setting, put it in its config file.
* **A timeout.** `timeout_s` from your `DEFINITION`. A script that overruns is **killed**, not
  abandoned, and the call counts as failed.
* **Argument checks.** Required arguments must be present and unknown ones are refused, before
  your script runs. **Types and values are yours to validate** (see below).
* **A size cap.** Results longer than the server's per-result cap (shared between parallel calls,
  so it shrinks as the context window fills) are cut and marked as shortened.
* **Honest failures.** A `ToolError` message is shown to the model and the user. A crash is
  reported as a failure *without* its traceback, which could contain argument values.

---

## Adding a tool, step by step

### 1. Write the script

Create `scripts/ai/ai-tools/<name>/<name>.py`. A complete minimal tool:

```python
#!/usr/bin/env python3
"""
word_count - counts the words in a piece of text (example tool).

Security flags: none - it reads only its argument and touches nothing else.

Usage:
    word_count.py --describe
    echo '{"text": "one two three"}' | word_count.py --config word_count.json
"""
from amadeo_utils.ai.llm.tools.script_tools import ToolAnswer, ToolError, tool_script_main

MAX_CHARS = 20_000

DEFINITION = {
    "name": "word_count",
    "description": "Counts the words in a piece of text.",          # the model reads this - be specific
    "parameters": {                                                  # JSON schema for the arguments
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The text to count."}},
        "required": ["text"],
    },
    "flags": {},                                                     # see step 2
    "timeout_s": 10,
}


def handle(arguments, config):
    """One call: validate, work, return. 'config' is this tool's JSON config file, as a dict."""
    text = arguments.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ToolError("text must be a non-empty string")           # shown to the model and the user
    if len(text) > MAX_CHARS:
        raise ToolError(f"text is longer than {MAX_CHARS} characters")
    words = len(text.split())
    # ToolAnswer = (full result for this turn, one short line kept in chat history for later turns)
    return ToolAnswer({"words": words}, f"{words} words")


if __name__ == "__main__":
    tool_script_main(DEFINITION, handle)
```

Things worth copying from the existing tools:

* **Validate every argument yourself** - type, range, format. Clamp numbers to sane limits
  (`get_weather` caps `hours` at 48 whatever the model asks). Pick from fixed lists where you can
  (`get_grades` accepts only configured terms).
* **Keep stdout clean.** The server reads the *last line* of stdout as the answer. Anything else
  your code (or a library) prints must go to stderr - `get_grades` wraps the script it reuses in
  `contextlib.redirect_stdout(sys.stderr)`.
* **Write your own summary** with `ToolAnswer` when the generic one would be poor. The summary
  line stays in the conversation history and the vector database; the full result is seen only on
  the turn it was fetched. Keep it short (it is cut at 160 characters) and put the most important
  part first.
* **Degrade instead of failing** when part of the work is optional - `get_weather` still returns
  the forecast if the rainfall request times out, with a note saying so.

### 2. Choose the security flags

Flags go in `DEFINITION["flags"]`. They are how the server knows what a tool can do; **every rule
it enforces is derived from them, never from a tool's name.** An unknown flag name stops the
server at startup, so a typo cannot leave a tool unflagged.

| Flag | Set it when... | What the server then does |
|---|---|---|
| `outbound` | anything the **model chose** leaves the machine (a URL, a query, even coordinates) | Refuses a call whose arguments look like a credential (passwords, API keys, tokens, `user:pass@` URLs). In `direct` mode, disables the tool once private data is in the session. |
| `private` | the **result** holds private data (grades, a local file's contents) | Returning it marks the session private: in `direct` mode, outbound tools are then off for the rest of the session. |
| `untrusted_output` | the **result** is text a third party wrote (web pages, search snippets) - it may carry prompt injection | In the default `auto` mode, the tool is reachable **only through `delegate`**: a separate worker reads the untrusted text, and only its final answer is shown to the user - it never enters the main model's context. |
| `internal_commands` | the tool acts on the local machine: `"none"` (default), `"read"` or `"write"` | Disabled once untrusted text has reached the model (`direct` mode). A `"read"` tool can surface private data, so flag it `private` too. |
| `needs_approval` | a person should confirm each call | The user is asked (y/n) with the exact tool name and arguments before it runs; no answer means no. **`"write"` tools always need approval, whatever the flags say.** |

The existing tools as worked examples:

* `get_weather` - **outbound**: the coordinates it is given go to api.weather.gov. The result is
  public data written by the NWS, so it is not `untrusted_output`.
* `web_search`, `fetch_url` - **outbound + untrusted_output**: the model picks the query or URL,
  and anyone can write the page.
* `get_grades` - **private**, *not* outbound: the model only picks a term and a number of days,
  and the request always goes to the one configured site.

### 3. Follow the security rules

* **Never a shell.** No `shell=True`, no `os.system`, no running a string the model wrote. If a
  tool must run a program, use `subprocess.run([...])` with a fixed argument list and only
  validated values in it.
* **Never a bare command or path.** Anything touching the local machine is a named, narrow
  operation with typed arguments. Resolve paths with `os.path.realpath` and confine them to
  allowed directories.
* **Treat what you fetch as hostile.** `fetch_url` refuses non-http(s) URLs, `user:pass@` URLs,
  and any host that resolves to a private or local address (the router, the NAS, this machine),
  checks every redirect, and connects to the address it checked (so DNS rebinding cannot switch
  it). Copy that pattern for anything that fetches.
* **Credentials never go through the model.** A tool that logs in reads its credentials from a
  file named in its config (mode `600`, or `660` with an explicitly trusted group - see
  `get_grades`). The model only ever sees the tool's arguments and result.
* **Nothing private in logs.** Don't log arguments or results. The server's own log records tool
  names and outcomes only.
* **Private results summarise with counts, not content** (`get_grades`: "9 courses; 40
  assignments ... 5 missing").
* **Summaries of untrusted results quote nothing from them** (`fetch_url`: the URL and a size -
  not even the page title, which the page's author wrote).

### 4. Give it a config file - outside the repo

`--config FILE` is the tool's own JSON settings: API endpoints, home locations, paths to secrets,
limits. It is loaded fresh on every call, so edits need no restart. **Keep these files outside
the repository** - they hold local addresses and paths, and this repo is public. Document the
keys in the script's docstring instead (each existing tool shows its config there).

### 5. Choose the Python environment

The tool server itself runs in the `llama` conda env. Tools run as separate processes, each with
whatever interpreter its config entry names:

* `agent-tools` - the general tool env (`requests`, `pgeocode`, `trafilatura`, ...). Pinned in
  `requirements.txt` in this directory; add new dependencies there, pinned, and re-verify it
  installs into a fresh env.
* A dedicated env when a tool needs something heavy - `get_grades` runs in the `bash` env, which
  has Playwright and Chromium.

`script_tools.py` imports only the standard library, so any env can run a tool as long as the
server's `src/` is on its path (the server sets `PYTHONPATH` for you).

### 6. Register it with the server

In the tool server's JSON config, add the script and allow the tool:

```json
"tools_allowed": ["get_datetime", "calculator", "word_count"],
"script_tools": [
    {"script": "/path/to/Python/scripts/ai/ai-tools/word_count/word_count.py",
     "python": "/path/to/miniforge3/envs/agent-tools/bin/python",
     "config": "/path/to/configs/ai-tools/word_count.json"}
]
```

A tool in `script_tools` but not in `tools_allowed` is loaded but never offered. A name in
`tools_allowed` that is not registered stops the server - on purpose. A session can narrow the
list; it can never widen it. See `example_amadeo_agent_server_config.json` next to `amadeo_agent_server.py` for
every server setting.

### 7. Test it

* **By hand**, exactly as the server runs it:

  ```bash
  python my_tool.py --describe
  echo '{"text": "one two three"}' | PYTHONPATH=/path/to/Python/src python my_tool.py --config my_tool.json
  ```

* **Through the real contract**, in a check script: `load_script_tool(python, script, config)`
  plus `registry.execute(tool, arguments, max_chars)` runs your tool with the server's minimal
  environment and timeout. `agent_projects/CS-21/tests/check_ai_tools.py` does this for the
  existing tools - add your cases there (refusals as well as successes).
* **Without the network or credentials**, run the real code against a stand-in:
  `check_get_grades.py` executes the real Infinite Campus parser with a fake browser and a fixture
  login, so everything but the login itself is tested offline.
* **One command** runs every check: `agent_projects/CS-21/tests/verify_cs21.sh` (`--quick` skips
  the slow part, `--offline` skips the internet).
* **With a model:** `live_script_tools_smoke.py` holds a real conversation on the GPU; add a turn
  set that should make the model reach for your tool, and read what it did.

---

## Checklist

- [ ] `DEFINITION`: clear description, JSON schema, flags, `timeout_s`
- [ ] every argument validated and clamped; `ToolError` messages written for the model to act on
- [ ] nothing but the answer on stdout
- [ ] flags match what the tool really does (outbound? private? untrusted? local?)
- [ ] no shell, no model-written commands or paths
- [ ] config and any secrets outside the repo; secrets file `600`
- [ ] a `ToolAnswer` summary that is short, useful in later turns, and leaks nothing it shouldn't
- [ ] dependencies pinned in `requirements.txt` (or a dedicated env, documented)
- [ ] registered in `script_tools` and `tools_allowed`
- [ ] checks added, and `verify_cs21.sh` passes

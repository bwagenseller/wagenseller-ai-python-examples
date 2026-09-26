"""
Tool definitions, their security flags, and the policy that decides which tools a model may use.

What this owns
--------------
A tool is a JSON-schema definition plus a Python function, with five security flags. Every
rule about which tools are reachable - the server's allow-list, a session narrowing it, the
routing of untrusted-output tools through 'delegate', and the two taint rules - is computed
here from those flags, never from tool names, and never by asking the model.

Nothing in this module is specific to llama.cpp or to any model: it deals in OpenAI-style
function definitions, which is what every chat template in use renders.

The flags (CS-21, AGENTS.md decision 1)
---------------------------------------
  outbound          - arguments the model chooses leave the machine (a search query, a URL).
  private           - the result contains private data (grades).
  untrusted_output  - the result is third-party text that may carry a prompt injection.
  internal_commands - 'none' | 'read' | 'write': acts on the local machine.
  needs_approval    - the user must approve each call. Code-enforced floor: any 'write' tool
                      needs approval whatever this says.

The taint rules
---------------
Once a PRIVATE result is in a loop's context, no OUTBOUND tool may run (private data plus an
outbound channel is how injected text exfiltrates). Once an UNTRUSTED result is in a loop's
context, no INTERNAL_COMMANDS tool may run (untrusted text plus a local command is how injected
text does damage). With the quarantine (see quarantine.py) untrusted text never reaches the
main model in 'auto' or 'delegate' mode, so in practice these only bite in 'direct' mode and
inside a worker.
"""
import json
import re
import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


INTERNAL_NONE = "none"
INTERNAL_READ = "read"
INTERNAL_WRITE = "write"
_INTERNAL_LEVELS = (INTERNAL_NONE, INTERNAL_READ, INTERNAL_WRITE)

# tool_mode values. 'auto' is the default.
MODE_AUTO = "auto"          # untrusted-output tools only via delegate; everything else direct
MODE_DIRECT = "direct"      # everything direct, no delegate; the session is tainted by what it reads
MODE_DELEGATE = "delegate"  # the main model gets only delegate and recall
TOOL_MODES = (MODE_AUTO, MODE_DIRECT, MODE_DELEGATE)

# The two tools the loop implements itself rather than through a registered function.
DELEGATE = "delegate"
RECALL = "recall"


@dataclass
class Tool:
    """
    One tool: what the model is shown, what runs, and how dangerous it is.

    Attributes:
        name (str): The name the model calls it by.
        description (str): Shown to the model.
        parameters (dict): JSON schema of the arguments ('type': 'object', 'properties', 'required').
        function (Callable): Called with the arguments as keywords; returns a string, or anything
            JSON-serialisable, which is sent back to the model.
        outbound, private, untrusted_output (bool), internal_commands (str), needs_approval (bool):
            The security flags described in the module docstring.
        timeout_s (float): How long the function may run before the call counts as failed.
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Optional[Callable[..., Any]]
    outbound: bool = False
    private: bool = False
    untrusted_output: bool = False
    internal_commands: str = INTERNAL_NONE
    needs_approval: bool = False
    timeout_s: float = 10.0

    def __post_init__(self):
        if self.internal_commands not in _INTERNAL_LEVELS:
            raise ValueError(f"tool {self.name}: internal_commands must be one of {_INTERNAL_LEVELS}")

    @property
    def requires_approval(self) -> bool:
        """True if the user must approve each call. A 'write' tool always does - no flag can turn that off."""
        return self.needs_approval or self.internal_commands == INTERNAL_WRITE

    def definition(self) -> Dict[str, Any]:
        """The OpenAI-style function definition the chat template renders into the prompt."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


class ToolRegistry:
    """
    Every tool the process knows about. The server config's allow-list picks from these; nothing
    outside the registry can ever be offered to a model.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        """Adds a tool. Raises ValueError on a duplicate name, rather than silently replacing one."""
        if tool.name in self._tools:
            raise ValueError(f"a tool named {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def tool(self, name: str, description: str, parameters: Dict[str, Any], **flags: Any):
        """
        Decorator form of register():

            @registry.tool("get_datetime", "Returns the current date and time.", {...})
            def get_datetime(timezone="UTC"): ...
        """
        def wrap(function: Callable[..., Any]) -> Callable[..., Any]:
            self.register(Tool(name, description, parameters, function, **flags))
            return function
        return wrap

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools)


@dataclass
class Taint:
    """
    What has entered one loop's context. Tracked per loop: the main loop and each worker have
    their own. The main loop's copy lives in the session, so it lasts the rest of the session.
    """
    private: bool = False
    untrusted: bool = False

    def absorb(self, tool: Tool):
        """Updates the taint after 'tool' has returned a result into this loop's context."""
        self.private = self.private or tool.private
        self.untrusted = self.untrusted or tool.untrusted_output


@dataclass
class ToolPolicy:
    """
    The decision of which tools one loop may be offered and run.

    Attributes:
        registry: All known tools.
        allowed (list[str]): The allow-list - the server config's, intersected with the session's
            narrowing. Names not in the registry are ignored.
        mode (str): One of TOOL_MODES.
        is_worker (bool): True for a delegate's worker loop.
    """
    registry: ToolRegistry
    allowed: List[str]
    mode: str = MODE_AUTO
    is_worker: bool = False
    _allowed_tools: List[Tool] = field(init=False, default_factory=list)

    def __post_init__(self):
        if self.mode not in TOOL_MODES:
            raise ValueError(f"tool_mode must be one of {TOOL_MODES}, not {self.mode!r}")
        self._allowed_tools = [t for t in (self.registry.get(n) for n in self.allowed) if t is not None]

    def _base_tools(self) -> List[Tool]:
        """The tools this loop could ever be offered, before taint is applied."""
        if self.is_worker:
            # A worker never sees private data (so nothing it reads can exfiltrate any) and never
            # touches the local machine. It has no delegate or recall of its own: one level deep.
            return [t for t in self._allowed_tools if not t.private and t.internal_commands == INTERNAL_NONE]
        if self.mode == MODE_DIRECT:
            return list(self._allowed_tools)
        if self.mode == MODE_DELEGATE:
            return []
        return [t for t in self._allowed_tools if not t.untrusted_output]      # MODE_AUTO

    def delegation_offered(self) -> bool:
        """Whether this loop is offered 'delegate' and 'recall'."""
        if self.is_worker or self.mode == MODE_DIRECT:
            return False
        if self.mode == MODE_DELEGATE:
            return True
        return any(t.untrusted_output for t in self._allowed_tools)             # MODE_AUTO

    def worker_policy(self) -> "ToolPolicy":
        """The policy for a worker started by this loop's 'delegate'."""
        return ToolPolicy(self.registry, self.allowed, self.mode, is_worker=True)

    def taint_applies(self) -> bool:
        """
        Whether the taint rules restrict this loop. Only where untrusted text can actually reach the
        model reading it: a worker, or a main loop in 'direct' mode. In 'auto' and 'delegate' modes
        the quarantine keeps untrusted text out of the main model entirely, so nothing in its context
        can hijack it, and locking tools there would only break the always-on use case.
        """
        return self.is_worker or self.mode == MODE_DIRECT

    def offered(self, taint: Taint) -> List[Tool]:
        """
        The tools to offer this round, taint applied. Delegate and recall are not included - they
        are loop-implemented; see delegation_offered().
        """
        tools = self._base_tools()
        if not self.taint_applies():
            return tools
        if taint.private:
            tools = [t for t in tools if not t.outbound]
        if taint.untrusted:
            tools = [t for t in tools if t.internal_commands == INTERNAL_NONE]
        return tools

    def offered_names(self, taint: Taint) -> List[str]:
        """Sorted names of everything offered this round, delegate and recall included."""
        names = [t.name for t in self.offered(taint)]
        if self.delegation_offered():
            # Still offered after private data has entered the conversation. The remaining risk - the
            # main model honestly copying private data into a task string - is handled by the loop
            # asking the user to approve such a delegation (see ToolStream), not by removing it.
            names += [DELEGATE, RECALL]
        return sorted(names)

    def refusal(self, name: str, taint: Taint) -> Optional[str]:
        """
        Re-checks a call immediately before it runs. A model can emit a call to a tool it was never
        offered, so being absent from the prompt is not enough.

        Returns:
            str | None: Why the call is refused - worded for the model and the user - or None if it may run.
        """
        if name in self.offered_names(taint):
            return None
        tool = self.registry.get(name)
        if tool is None or tool not in self._base_tools():
            return f"There is no tool called '{name}' available to you."
        if taint.private and tool.outbound:
            return f"'{name}' is disabled: private data is in this conversation and '{name}' sends data off this machine."
        return f"'{name}' is disabled: this conversation has read untrusted content, and '{name}' acts on this machine."


# --------------------------------------------------------------------------------------------------
# Running a tool
# --------------------------------------------------------------------------------------------------

@dataclass
class Summarized:
    """
    A tool's result together with its own one-line summary for the chat history.

    A tool function may return this instead of a bare result. The full result goes to the model this turn; only the
    summary is kept for later turns (decision 3: "a short result"). The tool knows what matters in its own output -
    the weather tool keeps the forecast, not the first 160 characters of its JSON. Tools whose output is untrusted
    should summarise facts ABOUT the result (a URL, a count), never text the result contains.
    """
    result: Any
    summary: str


@dataclass
class ToolOutcome:
    """
    The result of running one tool call.

    Attributes:
        ok (bool): True if the function returned normally.
        content (str): What goes back to the model: the result, or an error message.
        truncated (bool): True if the result was cut to the per-result cap.
        summary (str | None): The tool's own one-line summary for history, if it gave one.
    """
    ok: bool
    content: str
    truncated: bool = False
    summary: Optional[str] = None


SUMMARY_CHARS = 160


def summarize_content(content: str, limit: int = SUMMARY_CHARS) -> str:
    """
    A generic one-line summary of a result, for tools that do not write their own.

    A JSON object becomes 'key: value; key: value' from its top-level scalars (nested values become 'N items' / '...');
    anything else is its first line. Either way it is cut at a word boundary, with an ellipsis, rather than mid-token -
    the old behaviour of cutting a JSON result at 160 characters left half an object in the history.
    """
    text = content.strip()
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict):
        parts = []
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                value = f"{len(value)} items"
            elif isinstance(value, dict):
                value = "..."
            parts.append(f"{key}: {value}")
        text = "; ".join(parts)
    else:
        text = (text.splitlines() or [""])[0]
    return shorten(text, limit)


def shorten(text: str, limit: int = SUMMARY_CHARS) -> str:
    """Cuts text to 'limit' characters at a word boundary, marking the cut with an ellipsis."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return cut + "…"


# One shared pool. A tool that overruns its timeout cannot be killed from Python - its thread runs
# on until the function returns - so the pool is bounded rather than unbounded, and tools that can
# genuinely hang should run a subprocess with its own timeout instead (the grades tool does).
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")


def validate_arguments(tool: Tool, arguments: Dict[str, Any]) -> Optional[str]:
    """
    Checks the arguments against the tool's schema: required ones present, no unknown ones.

    Types were already coerced by the parser; a value of the wrong type is left for the function
    itself to reject, with a message the model can act on.

    Returns:
        str | None: An error message for the model, or None if the arguments are acceptable.
    """
    properties = tool.parameters.get("properties") or {}
    missing = [k for k in tool.parameters.get("required") or [] if k not in arguments]
    unknown = [k for k in arguments if k not in properties]
    if missing:
        return f"missing required argument(s) for {tool.name}: {', '.join(missing)}"
    if unknown:
        return f"unknown argument(s) for {tool.name}: {', '.join(unknown)}"
    return None


def execute(tool: Tool, arguments: Dict[str, Any], max_result_chars: int) -> ToolOutcome:
    """
    Runs one validated call with the tool's timeout, and caps what comes back.

    Every failure - bad arguments, an exception, a timeout - becomes a ToolOutcome with ok=False
    rather than an exception, so the loop can report it to the model AND the user (decision 4).
    Exception text is included: tools are ours, and their messages are written to be shown.

    Args:
        tool: The tool to run.
        arguments: Its arguments, already parsed and coerced.
        max_result_chars: The per-result cap. A longer result is cut and marked, so one fetched page
            cannot fill the context window.

    Returns:
        ToolOutcome
    """
    problem = validate_arguments(tool, arguments)
    if problem:
        return ToolOutcome(False, problem)
    future = _EXECUTOR.submit(tool.function, **arguments)
    try:
        result = future.result(timeout=tool.timeout_s)
    except concurrent.futures.TimeoutError:
        return ToolOutcome(False, f"{tool.name} did not finish within {tool.timeout_s:g} seconds.")
    except Exception as e:                      # a tool's own failure is data, not a crash
        return ToolOutcome(False, f"{tool.name} failed: {e}")
    summary = None
    if isinstance(result, Summarized):
        result, summary = result.result, shorten(str(result.summary))
    content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    if len(content) > max_result_chars:
        return ToolOutcome(True, content[:max_result_chars] + "\n[result truncated to fit the context window]", True, summary)
    return ToolOutcome(True, content, False, summary)


# --------------------------------------------------------------------------------------------------
# Secrets in outbound arguments
# --------------------------------------------------------------------------------------------------
#
# The owner's rule (2026-09-23): private data MAY go into a web search - except passwords and keys. The
# real protection is structural: credentials never enter the model's context (the grades tool keeps
# its login inside its own subprocess). This is the backstop for anything that slips through: it
# refuses an outbound call whose arguments LOOK like they carry a credential. It deliberately
# matches credential SHAPES - "password: x", "api_key=x", known key formats, credentials inside a
# URL - rather than the bare word, so "how do I reset my password" is still a normal search. It
# cannot recognise an arbitrary password written as plain prose; nothing can.

_SECRET_PATTERNS = [
    ("a password or key assignment",
     re.compile(r"(?i)\b(pass(?:word|wd|phrase)?|pwd|api[ _-]?key|secret(?:[ _-]?key)?|access[ _-]?token|auth[ _-]?token)"
                r"\s*[:=]\s*\S{3,}")),
    ("a bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("an AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("a GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})\b")),
    ("an API key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("a Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("a JSON web token", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("credentials in a URL", re.compile(r"://[^/\s:@]+:[^/\s@]+@")),
]


def _strings(value: Any):
    """Every string inside an argument value, however nested."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def find_secret(arguments: Dict[str, Any]) -> Optional[str]:
    """
    Checks outbound arguments for anything shaped like a credential.

    Returns:
        str | None: what it looks like ("an API key", ...) - never the matched text itself, which must not be
            echoed into logs, notices or the model's context - or None if nothing matched.
    """
    for text in _strings(arguments):
        for label, pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                return label
    return None


def definitions(tools: Iterable[Tool]) -> List[Dict[str, Any]]:
    """OpenAI-style definitions for a list of tools, in the order given."""
    return [t.definition() for t in tools]

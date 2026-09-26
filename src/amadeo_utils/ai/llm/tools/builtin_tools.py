"""
The built-in tools that need nothing from outside the process: the date and time, and arithmetic.

Both carry no security flags: they send nothing anywhere, read nothing private, return nothing
a third party wrote, and touch nothing on the machine. The model picks the arguments, so both
treat those arguments as hostile anyway - see the calculator's evaluator in particular.
"""
import ast
import math
import operator
from datetime import datetime
from typing import Any, Dict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from amadeo_utils.ai.llm.tools.registry import Tool, ToolRegistry


# --------------------------------------------------------------------------------------------------
# get_datetime
# --------------------------------------------------------------------------------------------------

# Everyday names a model or a person uses for a zone, mapped to the DST-aware IANA zone they mean. Matched after
# upper-casing and dropping the words TIME / STANDARD / DAYLIGHT, so "Eastern Standard Time" and "eastern" both work.
# This table exists because Python accepts some of these names literally - 'EST' is a FIXED UTC-5 zone with no
# daylight saving, so in summer it answered an hour off (seen live, 2026-09-26) - and rejects the rest ('EDT', 'ET').
_ZONE_ALIASES = {
    "America/New_York": ("EST", "EDT", "ET", "EASTERN", "US EASTERN", "US/EASTERN"),
    "America/Chicago": ("CST", "CDT", "CT", "CENTRAL", "US CENTRAL", "US/CENTRAL"),
    "America/Denver": ("MST", "MDT", "MT", "MOUNTAIN", "US MOUNTAIN", "US/MOUNTAIN"),
    "America/Phoenix": ("ARIZONA",),
    "America/Los_Angeles": ("PST", "PDT", "PT", "PACIFIC", "US PACIFIC", "US/PACIFIC"),
    "America/Anchorage": ("AKST", "AKDT", "AKT", "ALASKA"),
    "Pacific/Honolulu": ("HST", "HAWAII"),
    "UTC": ("UTC", "GMT", "Z", "ZULU", "UNIVERSAL", "COORDINATED UNIVERSAL"),
}
_ALIAS_TO_ZONE = {alias: zone for zone, aliases in _ZONE_ALIASES.items() for alias in aliases}
_FILLER_WORDS = {"TIME", "STANDARD", "DAYLIGHT", "ZONE"}


def local_timezone() -> str:
    """
    The machine's own IANA zone ('America/New_York'), from TZ or /etc/localtime; 'UTC' if it cannot be told.
    """
    import os
    name = os.environ.get("TZ", "").lstrip(":")
    if name:
        return name
    try:
        target = os.path.realpath("/etc/localtime")
        if "/zoneinfo/" in target:
            return target.split("/zoneinfo/", 1)[1]
    except OSError:
        pass
    return "UTC"


def resolve_timezone(name, default: str = "UTC"):
    """
    Turns what a model asked for into a real, DST-aware zone.

    In order: nothing -> 'default'; an everyday name (Eastern, EST/EDT, Pacific, GMT, ... - see _ZONE_ALIASES);
    an IANA name (Europe/London); a city, matched against the last part of every IANA name ('new york' ->
    America/New_York, 'Tokyo' -> Asia/Tokyo).

    Returns:
        tuple[ZoneInfo, str, str | None]: the zone, its IANA name, and a note saying how an everyday name was read
            (or None when the name was already exact).

    Raises:
        ValueError: if nothing matches.
    """
    if name is None or (isinstance(name, str) and not name.strip()):
        return ZoneInfo(default), default, None
    if not isinstance(name, str):
        raise ValueError("timezone must be a string such as 'America/New_York', 'Eastern' or 'London'")
    text = name.strip()
    key = " ".join(w for w in text.upper().replace("_", " ").split() if w not in _FILLER_WORDS)
    if key in _ALIAS_TO_ZONE:
        zone = _ALIAS_TO_ZONE[key]
        note = None if zone == text else f"'{text}' was read as {zone}, which follows daylight saving"
        return ZoneInfo(zone), zone, note
    try:
        return ZoneInfo(text), text, None
    except (ZoneInfoNotFoundError, ValueError):
        pass
    city = text.replace(" ", "_").lower()
    for zone in sorted(available_timezones()):
        if "/" in zone and zone.rsplit("/", 1)[1].lower() == city:
            return ZoneInfo(zone), zone, f"'{text}' was read as {zone}"
    raise ValueError(f"unknown timezone {text!r}; use a name such as 'America/New_York', 'Eastern', 'UTC' or a city "
                     f"such as 'London'")


def get_datetime(timezone: str = None, default_timezone: str = "UTC") -> str:
    """
    The current date and time in a zone, written so it cannot be misread: weekday, date, 12-hour clock with the
    zone's abbreviation, the IANA name and UTC offset, and the ISO 8601 form.

    Args:
        timezone (str): What the model asked for - see resolve_timezone. None or '' means 'default_timezone'.
        default_timezone (str): The server's local zone (set when the tool is built - see make_get_datetime_tool).

    Returns:
        str: e.g. "Saturday, 2026-09-26, 7:36:21 AM EDT (America/New_York, UTC-04:00); ISO 2026-09-26T07:36:21-04:00".

    Raises:
        ValueError: if the zone cannot be resolved.
    """
    zone, zone_name, note = resolve_timezone(timezone, default_timezone)
    now = datetime.now(zone)
    offset = now.strftime("%z")
    offset = f"UTC{offset[:3]}:{offset[3:]}" if offset else "UTC"
    hour = now.hour % 12 or 12
    text = (f"{now.strftime('%A')}, {now.strftime('%Y-%m-%d')}, {hour}:{now.strftime('%M:%S')} "
            f"{'AM' if now.hour < 12 else 'PM'} {now.tzname()} ({zone_name}, {offset}); "
            f"ISO {now.isoformat(timespec='seconds')}")
    return f"{text}. Note: {note}." if note else text


def make_get_datetime_tool(default_timezone: str = None) -> Tool:
    """
    The get_datetime tool for a server whose local zone is 'default_timezone' (None: this machine's own zone).

    The description tells the model the default and - because models convert between zones badly - to call the tool
    again for every zone it is asked about rather than doing the arithmetic itself.

    Raises:
        ValueError: if 'default_timezone' is not a zone resolve_timezone can find (a config mistake stops the server).
    """
    requested = default_timezone or local_timezone()
    _, default_name, _ = resolve_timezone(requested)

    def function(timezone: str = None) -> str:
        return get_datetime(timezone, default_name)

    return Tool(
        name="get_datetime",
        description=f"Returns the current date, time and weekday. With no timezone it gives local time "
                    f"({default_name}). Call it again for EACH timezone the user asks about - never convert "
                    f"between timezones yourself.",
        parameters={"type": "object",
                    "properties": {"timezone": {"type": "string",
                                                "description": "Optional. An IANA name (America/New_York), an everyday "
                                                               "name (Eastern, Pacific, UTC, GMT) or a city (London). "
                                                               f"Default: {default_name}."}},
                    "required": []},
        function=function,
        timeout_s=2.0,
    )


GET_DATETIME = make_get_datetime_tool("UTC")        # a fixed default, for code that registers the tool directly


# --------------------------------------------------------------------------------------------------
# calculator
# --------------------------------------------------------------------------------------------------

# Everything the evaluator will accept. Anything else in the expression - names, attribute access,
# calls to anything not listed, comprehensions, strings - is rejected before evaluation, so there
# is no route from the model's input to arbitrary Python.
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin,
              "acos": math.acos, "atan": math.atan, "log": math.log, "log10": math.log10, "log2": math.log2,
              "exp": math.exp, "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
              "radians": math.radians, "degrees": math.degrees}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

# Guards against expressions that are legal but would hang or exhaust memory: 9**9**9 is three
# characters of input and billions of digits of output.
_MAX_EXPRESSION_CHARS = 200
_MAX_EXPONENT = 1000
_MAX_MAGNITUDE = 1e300


def _evaluate(node: ast.AST) -> Any:
    """Evaluates one node of an already-parsed arithmetic expression, refusing anything not whitelisted."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent {right} is too large (limit {_MAX_EXPONENT})")
        result = _BINARY[type(node.op)](left, right)
        if isinstance(result, (int, float)) and abs(result) > _MAX_MAGNITUDE:
            raise ValueError("result is too large")
        return result
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS
            and not node.keywords):
        return _FUNCTIONS[node.func.id](*(_evaluate(a) for a in node.args))
    raise ValueError(f"unsupported element in expression: {type(node).__name__}")


def calculator(expression: str) -> str:
    """
    Evaluates an arithmetic expression safely.

    Supports + - * / // % **, parentheses, the constants pi, e and tau, and the functions in
    _FUNCTIONS. Parsed with Python's own parser but evaluated by a whitelist walker, never by
    eval().

    Args:
        expression (str): e.g. '2 * (3 + 4) ** 2' or 'sqrt(2) * pi'.

    Returns:
        str: The result.

    Raises:
        ValueError: for anything that is not a supported arithmetic expression, or is too large.
    """
    if len(expression) > _MAX_EXPRESSION_CHARS:
        raise ValueError(f"expression is longer than {_MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError:
        raise ValueError(f"not a valid arithmetic expression: {expression!r}")
    try:
        result = _evaluate(tree)
    except ZeroDivisionError:
        raise ValueError("division by zero")
    except OverflowError:
        raise ValueError("result is too large")
    return repr(result) if isinstance(result, float) else str(result)


CALCULATOR = Tool(
    name="calculator",
    description="Evaluates an arithmetic expression exactly. Supports + - * / // % ** (or ^), parentheses, "
                "pi, e, and sqrt, sin, cos, tan, log, log10, log2, exp, abs, round, floor, ceil.",
    parameters={"type": "object",
                "properties": {"expression": {"type": "string", "description": "e.g. 2 * (3 + 4) ** 2"}},
                "required": ["expression"]},
    function=calculator,
    timeout_s=2.0,
)


def register_builtins(registry: ToolRegistry, default_timezone: str = None) -> ToolRegistry:
    """
    Adds every built-in tool to a registry and returns it.

    Args:
        registry: The registry to add to.
        default_timezone: The zone get_datetime uses when the model names none; None means this machine's own zone.
    """
    for tool in (make_get_datetime_tool(default_timezone), CALCULATOR):
        registry.register(tool)
    return registry

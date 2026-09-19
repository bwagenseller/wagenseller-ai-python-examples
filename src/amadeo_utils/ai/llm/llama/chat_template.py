"""
Chat-template handling for llama-cpp-python.

Why this module exists
----------------------
'llama_utils.universal_token_count()' was written against 'Llama._format_chat_prompt()',
a method that does not exist and never has - it is absent from llama-cpp-python 0.3.9 and
from 0.3.35 alike. Every call to it therefore raised AttributeError and fell through to a
chain of hand-written, per-model token counters that had to be taught each new model's
delimiters by name. Models whose filename matched no branch ('gemma-4...', 'Muse-Glimmer...')
simply crashed on startup.

The fix is to do properly what that call was reaching for: render the conversation through
the chat template that is already embedded in the GGUF. llama.cpp stores it under the
'tokenizer.chat_template' metadata key, and llama-cpp-python exposes it on the loaded model
as 'llm.metadata'. Rendering it gives an exact prompt for any architecture - current or
future - with no per-model branching.

Thinking / reasoning
--------------------
llama-cpp-python 0.3.35 has no reasoning support whatsoever: no 'enable_thinking' parameter,
no 'chat_template_kwargs', no 'reasoning_content' field, and no '<think>' parsing. Those are
features of the separate 'llama-server' binary, not of the Python bindings. 'create_chat_completion()'
has a fixed signature with no '**kwargs', so there is no supported route for passing a template
variable through it.

What IS possible is to build our own formatter, bake the template variables into it, and hand
it to Llama(chat_handler=...). That is what ChatTemplateFormatter below does. Because the
variables are read at render time from a mutable attribute rather than frozen at construction,
a single installed handler can be toggled between reasoning and non-reasoning mode per turn.

Each model family spells the control differently, so the mapping lives in _SCHEMES:
  - Qwen 3.x   ('qwen35moe')    'enable_thinking' (bool). When false the template emits a
                                pre-closed '<think>\n\n</think>\n\n'; when true it emits a bare
                                opening '<think>\n' INTO THE PROMPT, so the model's own output
                                begins inside the block and contains no opening tag.
  - Gemma 4    ('gemma4')       'enable_thinking' (bool), with reasoning delimited by the
                                '<|channel>' / '<channel|>' pair.
  - Muse Glimmer ('muse-glimmer') 'reasoning_strength' (str), defaulting to 'high'. There is no
                                hard off switch - the template renders the value as a plain
                                instruction line in the system block - so 'low' is as quiet as
                                it gets. Reasoning arrives on a Harmony-style 'to=self' channel.

Import ordering
---------------
This module deliberately does NOT import llama_cpp at module scope. Importing llama_cpp loads
the llama.cpp shared library, which pins CUDA device ordering, and llama_utils must win that
race (see the comment at the top of llama_utils.py). Every llama_cpp import here is local to
the function that needs it, by which point the caller necessarily holds a loaded model anyway.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------------------------------
# Reasoning schemes
# --------------------------------------------------------------------------------------------------

# Delimiter families. A scheme covers how a model marks reasoning in its OUTPUT and which
# template variable turns that reasoning off, which are separate concerns from the architecture
# name itself - several architectures can share one scheme.
SCHEME_QWEN = "qwen"            # <think> ... </think>
SCHEME_GEMMA_CHANNEL = "gemma"  # <|channel> ... <channel|>
SCHEME_HARMONY = "harmony"      # <|start|>assistant to=self<|message|> ... <|eom|>
SCHEME_NONE = "none"            # model has no reasoning mode

# Maps the GGUF's 'general.architecture' string to its reasoning scheme, the template kwargs
# that enable and disable reasoning, and the stop strings that end an assistant turn.
#
# A note on stop strings: they must NOT include a token that merely ends the reasoning segment,
# only ones that end the whole turn. Harmony's '<|eom|>' closes the 'to=self' message and
# '<|start|>' opens the answer that follows it - stopping on either would truncate the reply to
# nothing but its own deliberation.
_SCHEMES: Dict[str, Dict[str, Any]] = {
    "qwen35moe": {
        "scheme": SCHEME_QWEN,
        "think_on": {"enable_thinking": True},
        "think_off": {"enable_thinking": False},
        "stops": ["<|im_end|>", "<|endoftext|>"],
    },
    "qwen3moe": {
        "scheme": SCHEME_QWEN,
        "think_on": {"enable_thinking": True},
        "think_off": {"enable_thinking": False},
        "stops": ["<|im_end|>", "<|endoftext|>"],
    },
    "gemma4": {
        "scheme": SCHEME_GEMMA_CHANNEL,
        "think_on": {"enable_thinking": True},
        "think_off": {"enable_thinking": False},
        "stops": ["<turn|>", "<|turn>"],
    },
    "muse-glimmer": {
        "scheme": SCHEME_HARMONY,
        "think_on": {"reasoning_strength": "high"},
        "think_off": {"reasoning_strength": "low"},
        "stops": ["<|eot|>", "<|return|>"],
    },
}

# Applied when the architecture is unrecognised. Reasoning-free models render identically with
# or without these kwargs, so the safe default is to send none and strip the most common
# delimiter pair defensively.
_DEFAULT_SCHEME: Dict[str, Any] = {
    "scheme": SCHEME_QWEN,
    "think_on": {},
    "think_off": {},
    "stops": [],
}


# Cache for the one-off probe below; None means 'not yet determined'.
_TEMPLATE_KWARGS_SUPPORTED: Optional[bool] = None


def template_kwargs_supported() -> bool:
    """
    Reports whether the installed llama-cpp-python forwards template variables to the renderer.

    This is a real version floor, and a silent one. llama-cpp-python 0.3.9's
    'Jinja2ChatFormatter.__call__' accepts '**kwargs' and then does not pass them to
    'render()' - so 'enable_thinking=False' is accepted, ignored, and the model reasons anyway
    with nothing at all to indicate why. 0.3.35 forwards them correctly.

    Rather than compare version strings, this renders a one-line probe template and checks
    whether the variable actually arrived.

    Returns:
        bool: True if template variables reach the template.
    """
    global _TEMPLATE_KWARGS_SUPPORTED
    if _TEMPLATE_KWARGS_SUPPORTED is None:
        try:
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter
            probe = Jinja2ChatFormatter(template="{{ probe_value }}", eos_token="", bos_token="")
            _TEMPLATE_KWARGS_SUPPORTED = (
                probe(messages=[{"role": "user", "content": ""}], probe_value="ok").prompt == "ok")
        except Exception:
            _TEMPLATE_KWARGS_SUPPORTED = False
    return _TEMPLATE_KWARGS_SUPPORTED


def model_architecture(llm) -> str:
    """
    Reads the architecture name the GGUF declares for itself.

    This is the same value as 'general.architecture' from the gguf reader - 'qwen35moe',
    'gemma4', 'muse-glimmer' and so on. It is far more reliable than matching substrings
    against the model's filename, which breaks the moment a file is renamed or a new
    version number lands ('gemma-4' does not contain 'gemma-3').

    Args:
        llm: A loaded llama_cpp.Llama instance.

    Returns:
        str: The lowercased architecture name, or '' if the metadata is unavailable.
    """
    metadata = getattr(llm, "metadata", None) or {}
    return str(metadata.get("general.architecture", "")).strip().lower()


def scheme_for(llm) -> Dict[str, Any]:
    """
    Looks up the reasoning scheme for a loaded model.

    Args:
        llm: A loaded llama_cpp.Llama instance.

    Returns:
        dict: The matching entry from _SCHEMES, or _DEFAULT_SCHEME if the architecture
              is not one we know about.
    """
    return _SCHEMES.get(model_architecture(llm), _DEFAULT_SCHEME)


def stop_tokens(llm) -> List[str]:
    """
    Returns the architecture-appropriate stop strings for an assistant turn.

    The caller is expected to merge these with whatever conversational stops it already
    uses (a player name, 'User:', and so on) rather than replacing them.

    Args:
        llm: A loaded llama_cpp.Llama instance.

    Returns:
        list[str]: Stop strings; empty if the architecture is unknown.
    """
    return list(scheme_for(llm)["stops"])


# --------------------------------------------------------------------------------------------------
# Stripping reasoning out of a generated response
# --------------------------------------------------------------------------------------------------

# The header opening a Harmony message, after its '<|start|>': an optional role, an optional recipient, then
# '<|message|>'. Anything before the first '<|message|>' in a segment that does not fit this is not a header.
_HARMONY_HEADER = re.compile(r"^\s*(?:assistant)?\s*(?:to=(?P<to>[\w.\-]+))?\s*<\|message\|>", re.DOTALL)
_HARMONY_CONTROL = re.compile(r"<\|(?:start|message|end|eom|eot|return|channel)\|>")


def _harmony_messages(text: str):
    """
    Splits Harmony-format output (Muse Glimmer) into deliberation and reply, by message recipient.

    The output is a run of messages, each opened by '<|start|>' and a header naming its recipient. The first has no
    '<|start|>' of its own, because the prompt already ended with '<|start|>assistant'. Messages to 'self' - and to
    tools - are deliberation; messages to 'user', or with no recipient, are the reply.

    Why classify by recipient rather than delete what looks like reasoning: an earlier version deleted 'to=self'
    spans and kept everything else, which let anything the model did not format as expected straight through.
    Captured from the real model under a '!short' prompt, four times out of four:

        ...Proceed.<|start|>enough? Let's give.<|start|>assistant to=user<|message|>Autumn, the year...

    - a message opened and then filled with more deliberation, no header at all. The fragment reached the screen and
    the chat history. Classifying every segment makes such fragments deliberation by construction.

    Args:
        text (str): Raw generated text.

    Returns:
        tuple[list[str], list[str]]: (deliberation parts, reply parts), control tokens removed. The reply parts are
            those addressed to 'user' if there are any, otherwise the unaddressed ones.
    """
    deliberation, to_user, unaddressed = [], [], []
    for i, segment in enumerate(text.split("<|start|>")):
        header = _HARMONY_HEADER.match(segment)
        if header is None:
            if i == 0:
                # The first segment may be headerless prose from a model that skipped the protocol and just
                # answered. A header still arriving (' to=se...') is neither yet.
                if segment.strip() and "<|message|>" not in segment and not segment.lstrip().startswith("to="):
                    unaddressed.append(segment)
            else:
                # A headerless segment after '<|start|>' is a malformed fragment: deliberation, never a reply.
                deliberation.append(_HARMONY_CONTROL.sub("", segment))
            continue
        # A header can arrive stacked on itself - ' to=user<|message|>to=user<|message|>...' - which Muse Glimmer
        # produced once its conversation history already held raw headers it could imitate (CS-18). Consume every
        # further header at the start of the body; the first one names the recipient.
        body = segment[header.end():]
        stacked = _HARMONY_HEADER.match(body)
        while stacked is not None and stacked.end() > 0:
            body = body[stacked.end():]
            stacked = _HARMONY_HEADER.match(body)
        body = _HARMONY_CONTROL.sub("", body)
        recipient = header.group("to")
        if recipient == "user":
            to_user.append(body)
        elif recipient is None:
            unaddressed.append(body)
        else:
            deliberation.append(body)
    return deliberation, (to_user if to_user else unaddressed)


def _harmony_answer(text: str) -> str:
    """
    The reply from Harmony-format output; see _harmony_messages().

    Args:
        text (str): Raw generated text.

    Returns:
        str: The reply; '' if no user-facing message was produced (for instance, the budget ran out mid-reasoning).
    """
    _, reply = _harmony_messages(text)
    return "\n\n".join(part.strip() for part in reply if part.strip()).strip()


# The label Gemma 4 writes at the start of its thought channel: '<|channel>thought\n...'.
_GEMMA_THOUGHT_LABEL = "thought"


def split_reasoning(text: str, llm=None, scheme: Optional[str] = None,
                    thinking: bool = True) -> Tuple[str, Optional[str]]:
    """
    Splits a response into its deliberation and its reply, both as readable text with the protocol markers removed.

    Used for displaying a '!reason' turn: the deliberation is shown, but without the raw '<|channel>', '</think>' or
    ' to=user<|message|>' markers that the models use to delimit it. Works on partial text as it streams in.

    Args:
        text (str): The raw generated text, possibly incomplete.
        llm: A loaded llama_cpp.Llama instance, used to infer the scheme.
        scheme (str, optional): One of the SCHEME_* constants, to override detection.
        thinking (bool): Whether reasoning was enabled for this generation. Only matters for Qwen, whose prompt opens
            the think block itself, so text with no closing tag is either all deliberation (on) or all reply (off).

    Returns:
        tuple[str, str | None]: (deliberation, reply). The reply is None while none of it has started.
    """
    if scheme is None:
        scheme = scheme_for(llm)["scheme"] if llm is not None else SCHEME_QWEN

    if scheme == SCHEME_HARMONY:
        deliberation, reply = _harmony_messages(text)
        reasoning = "\n".join(part.strip() for part in deliberation if part.strip())
        return reasoning, ("\n\n".join(part.strip() for part in reply if part.strip()) if reply else None)

    if scheme == SCHEME_GEMMA_CHANNEL:
        # Same parsing as the model's own 'strip_thinking' macro, but keeping both halves.
        reasoning_bits, reply_bits, reply_started = [], [], False
        parts = text.split("<channel|>")
        for i, part in enumerate(parts):
            if i > 0:
                reply_started = True     # a thought channel has closed; whatever follows is reply
            if "<|channel>" in part:
                before, thought = part.split("<|channel>", 1)
                reply_bits.append(before)
                label = thought.lstrip()
                if _GEMMA_THOUGHT_LABEL.startswith(label):
                    thought = ""         # the 'thought' label has not fully arrived yet
                elif label.startswith(_GEMMA_THOUGHT_LABEL) and label[len(_GEMMA_THOUGHT_LABEL):][:1].isspace():
                    thought = label[len(_GEMMA_THOUGHT_LABEL):]
                reasoning_bits.append(thought)
            else:
                reply_bits.append(part)
        if "<|channel>" not in text:
            reply_started = True         # no thought channel at all: it is all reply
        reply = "".join(reply_bits)
        return "\n".join(b.strip() for b in reasoning_bits if b.strip()), (reply if reply_started or reply.strip() else None)

    # SCHEME_QWEN and the unknown-architecture default.
    if "</think>" in text:
        reasoning, reply = text.rsplit("</think>", 1)
        return reasoning.replace("<think>", "").strip(), reply
    if "<think>" in text:
        before, reasoning = text.split("<think>", 1)
        return reasoning.strip(), (before if before.strip() else None)
    return (text.strip(), None) if thinking else ("", text)


def strip_reasoning(text: str, llm=None, scheme: Optional[str] = None,
                    thinking: Optional[bool] = None) -> str:
    """
    Removes a model's reasoning/deliberation from its response, leaving only the answer.

    This is needed even when reasoning has been suppressed at the template, as a backstop for
    models that emit a stray block anyway, and it is needed in full when the caller has
    deliberately turned reasoning back on but wants to display or store the answer alone.

    The Qwen case is the subtle one. Its template writes the OPENING '<think>' tag into the
    prompt, not the output, so the generated text looks like:

        some deliberation...</think>

        the actual answer

    - there is no opening tag to match. A naive '<think>.*?</think>' regex finds nothing and
    silently leaves the whole mess in place. Splitting on the closing tag is what actually works.

    Truncation is the other trap. With reasoning enabled, a model that runs out of token budget
    mid-thought never emits its closing marker at all, and without knowing reasoning was on there
    is no way to tell that from an ordinary reply - so the deliberation would be kept and saved as
    though it were the answer. Hence the 'thinking' argument: when reasoning was on and the
    closing marker never arrived, everything generated was deliberation, and the result is ''.

    Args:
        text (str): The raw generated response.
        llm: A loaded llama_cpp.Llama instance, used to infer the scheme and, if 'thinking' is
             not given, whether reasoning was switched on for this generation.
        scheme (str, optional): One of the SCHEME_* constants, to override detection.
        thinking (bool, optional): Whether reasoning was enabled for the generation that produced
             'text'. Pass it explicitly wherever the model is shared between threads - reading it
             back from the model afterwards can pick up another caller's setting.

    Returns:
        str: The response with any reasoning removed, stripped of surrounding whitespace.
    """
    if not text:
        return text

    if scheme is None:
        scheme = scheme_for(llm)["scheme"] if llm is not None else SCHEME_QWEN

    if llm is not None:
        if thinking is None:
            thinking = thinking_enabled(llm)
        # Only a model that genuinely has a reasoning mode can have been mid-thought. Without this,
        # a '!reason' turn on an ordinary model would have every reply discarded as 'truncated'.
        if not scheme_for(llm)["think_on"]:
            thinking = False
    thinking = bool(thinking)

    if scheme == SCHEME_GEMMA_CHANNEL:
        # A faithful port of the model's own 'strip_thinking' jinja macro: split on the closing
        # marker, and from any fragment containing an opening marker keep only what preceded it.
        pieces = []
        for part in text.split("<channel|>"):
            pieces.append(part.split("<|channel>")[0] if "<|channel>" in part else part)
        return "".join(pieces).strip()

    if scheme == SCHEME_HARMONY:
        return _harmony_answer(text)

    # SCHEME_QWEN and the unknown-architecture default.
    if "</think>" in text:
        # Everything up to and including the final closing tag is deliberation.
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        # Defensive: the model opened a block of its own. Remove it, and if it never closed,
        # everything from the tag onward was deliberation cut off by the token budget.
        text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL)
    elif thinking:
        # Reasoning was on, so the prompt opened '<think>' and generation began inside it. No
        # closing tag means the budget ran out mid-thought: none of this is an answer.
        return ""
    return text.strip()


# --------------------------------------------------------------------------------------------------
# The formatter
# --------------------------------------------------------------------------------------------------

class ChatTemplateFormatter:
    """
    Renders conversations through a GGUF's own embedded chat template, with template variables
    that can be changed between calls.

    llama-cpp-python builds one 'Jinja2ChatFormatter' per model at load time and freezes its
    behaviour; there is no way to vary a template variable per request through the public API.
    This wrapper keeps the same rendering machinery but reads its extra variables from
    'self.template_kwargs' at render time, so flipping reasoning on or off is a matter of
    reassigning an attribute rather than reloading the model.

    Attributes:
        template_kwargs (dict): Extra variables passed to the jinja render on every call.
    """

    def __init__(self, llm, add_generation_prompt: bool = True,
                 template_kwargs: Optional[Dict[str, Any]] = None):
        """
        Builds a formatter from a loaded model's embedded chat template.

        Args:
            llm: A loaded llama_cpp.Llama instance.
            add_generation_prompt (bool): Whether to append the assistant-turn opener. True for
                                          generation; False when measuring token counts, where
                                          the trailing opener is noise.
            template_kwargs (dict, optional): Initial template variables.

        Raises:
            ValueError: If the GGUF carries no chat template, which leaves nothing to render.
        """
        # Deferred deliberately - see the 'Import ordering' note in the module docstring.
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter

        metadata = getattr(llm, "metadata", None) or {}
        template = metadata.get("tokenizer.chat_template")
        if not template:
            raise ValueError(
                f"The GGUF at [{getattr(llm, 'model_path', '?')}] has no embedded chat template "
                f"('tokenizer.chat_template'), so its prompt format cannot be determined. Pass an "
                f"explicit 'chat_format' to Llama() for this model."
            )

        # Mirror how llama-cpp-python resolves the BOS/EOS strings: ask the model for the text of
        # the token ids it reports, treating -1 (the 'unset' sentinel) as an empty string.
        bos_id = llm.token_bos()
        eos_id = llm.token_eos()
        bos_token = llm._model.token_get_text(bos_id) if bos_id != -1 else ""
        eos_token = llm._model.token_get_text(eos_id) if eos_id != -1 else ""

        self.template_kwargs: Dict[str, Any] = dict(template_kwargs or {})
        self._eos_token = eos_token
        self._formatter = Jinja2ChatFormatter(
            template=template,
            eos_token=eos_token,
            bos_token=bos_token,
            add_generation_prompt=add_generation_prompt,
            stop_token_ids=[eos_id] if eos_id != -1 else None,
        )

    def __call__(self, *, messages: List[Dict[str, Any]], **kwargs: Any):
        """
        Renders a conversation.

        Args:
            messages (list[dict]): The conversation, in OpenAI role/content form.
            **kwargs: Passed through to the underlying formatter. Anything set here wins over
                      'self.template_kwargs', so a caller can still override per call.

        Returns:
            ChatFormatterResponse: Carrying the rendered '.prompt' and its stop criteria.
        """
        merged = dict(self.template_kwargs)
        merged.update(kwargs)
        return self._formatter(messages=messages, **merged)

    def render(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        """
        Renders a conversation and returns the prompt text alone.

        Args:
            messages (list[dict]): The conversation, in OpenAI role/content form.
            **kwargs: Extra template variables for this call only.

        Returns:
            str: The rendered prompt.
        """
        return self(messages=messages, **kwargs).prompt

    def to_chat_handler(self) -> Callable:
        """
        Wraps this formatter as a chat-completion handler suitable for Llama(chat_handler=...).

        Returns:
            Callable: The handler, which llama-cpp-python will call in place of its own.
        """
        from llama_cpp.llama_chat_format import chat_formatter_to_chat_completion_handler
        return chat_formatter_to_chat_completion_handler(self)


# --------------------------------------------------------------------------------------------------
# Installing a handler on a model
# --------------------------------------------------------------------------------------------------

# Attribute names used to cache our objects on the Llama instance itself. Stashing them there
# keeps the lifetime tied to the model - no module-level registry to leak or to go stale when a
# model is reloaded.
_HANDLER_FORMATTER_ATTR = "_amadeo_chat_formatter"
_COUNT_FORMATTER_ATTR = "_amadeo_count_formatter"
_COUNT_BASELINE_ATTR = "_amadeo_count_baselines"


def install_chat_handler(llm, thinking: bool = False):
    """
    Installs a toggleable chat handler on a loaded model.

    After this call 'create_chat_completion()' renders through the GGUF's own template with our
    template variables applied, and 'set_thinking()' can change those variables at any point
    without touching the model.

    Args:
        llm: A loaded llama_cpp.Llama instance.
        thinking (bool): Whether reasoning starts enabled. Defaults to False, which is what a
                         conversational pipeline wants - deliberation is latency and context
                         spent on text that is then discarded.

    Returns:
        ChatTemplateFormatter: The installed formatter, for callers that want to poke at it.

    Raises:
        ValueError: If the GGUF carries no embedded chat template, or if the installed
                    llama-cpp-python is too old to forward template variables - in which case
                    reasoning could not be suppressed and the caller deserves to be told so
                    rather than silently getting a model that reasons anyway.
    """
    if scheme_for(llm)["think_on"] and not template_kwargs_supported():
        raise ValueError(
            "This llama-cpp-python does not forward chat-template variables to the template "
            "(0.3.9 accepts them and drops them on the floor; 0.3.35 does not), so reasoning "
            "cannot be suppressed for this model. Upgrade llama-cpp-python."
        )

    formatter = ChatTemplateFormatter(llm, add_generation_prompt=True)
    scheme = scheme_for(llm)
    formatter.template_kwargs = dict(scheme["think_on"] if thinking else scheme["think_off"])

    setattr(llm, _HANDLER_FORMATTER_ATTR, formatter)
    llm.chat_handler = formatter.to_chat_handler()
    return formatter


def set_thinking(llm, thinking: bool) -> bool:
    """
    Turns reasoning on or off for subsequent generations.

    Args:
        llm: A loaded llama_cpp.Llama instance that has been through install_chat_handler().
        thinking (bool): True to let the model deliberate, False to suppress it.

    Returns:
        bool: True if the setting was applied, False if no handler was installed (in which case
              the model's reasoning behaviour is whatever its template defaults to).
    """
    formatter = getattr(llm, _HANDLER_FORMATTER_ATTR, None)
    if formatter is None:
        return False

    scheme = scheme_for(llm)
    formatter.template_kwargs = dict(scheme["think_on"] if thinking else scheme["think_off"])
    return True


def supports_thinking(llm) -> bool:
    """
    Reports whether this model has a reasoning mode we know how to control.

    Useful for deciding whether to advertise a reasoning toggle in a help listing.

    Args:
        llm: A loaded llama_cpp.Llama instance.

    Returns:
        bool: True if the architecture maps to a known set of reasoning template variables.
    """
    return bool(scheme_for(llm)["think_on"])


def thinking_enabled(llm) -> bool:
    """
    Reports whether reasoning is currently switched on for this model's installed handler.

    Args:
        llm: A loaded llama_cpp.Llama instance.

    Returns:
        bool: True if a handler is installed and currently carries the model's 'reasoning on'
              template variables; False otherwise, including when no handler is installed.
    """
    formatter = getattr(llm, _HANDLER_FORMATTER_ATTR, None)
    think_on = scheme_for(llm)["think_on"]
    return bool(think_on) and formatter is not None and formatter.template_kwargs == think_on


def split_stops(llm, stops: List[str]):
    """
    Divides stop strings into those llama.cpp should enforce and those that must only apply to the answer.

    role_play.py and RolePlayStream.py stop generation on conversational markers - 'User:', 'Assistant:' and the
    player's name followed by a colon - to stop a model speaking for the user. Handed to llama.cpp, those fire on the
    RAW output, reasoning included, and a model's reasoning notes are exactly where such markers appear: measured on
    the RTX 5090, 'User:' turned up inside Gemma 4's or Qwen 3.6's deliberation in 5 of 6 reasoning runs, and never in
    an answer. Each hit cut the deliberation short before any answer existed, and the turn came back empty.

    For a model with a reasoning mode the conversational stops are therefore withheld from llama.cpp and applied to
    the answer after the reasoning is stripped (see 'truncate_at_stops' and ReasoningStreamFilter). Control-token
    stops ('<|im_end|>', '[INST]' and the architecture's own end-of-turn markers) stay with llama.cpp: they never occur
    inside reasoning, and they are what properly ends a turn. A model with no reasoning mode keeps every stop in
    llama.cpp, exactly as before.

    Args:
        llm: A loaded llama_cpp.Llama instance.
        stops (list[str]): Every stop string the caller wants.

    Returns:
        tuple[list[str], list[str]]: (stops for llama.cpp, stops to apply to the stripped answer).
    """
    if not supports_thinking(llm):
        return list(stops), []
    conversational = [s for s in stops if s.rstrip().endswith(":") and not s.startswith(("<", "["))]
    return [s for s in stops if s not in conversational], conversational


def truncate_at_stops(text: str, stops: List[str]) -> str:
    """
    Cuts text at the earliest occurrence of any stop string, as llama.cpp would have.

    Args:
        text (str): The (already reasoning-stripped) answer.
        stops (list[str]): The answer-only stops from split_stops().

    Returns:
        str: The text up to, not including, the first stop found; unchanged if none occurs.
    """
    positions = [text.find(s) for s in stops if s and s in text]
    return text[:min(positions)].rstrip() if positions else text


class ReasoningStreamFilter:
    """
    Hides a model's deliberation from a live token stream, releasing only the answer.

    'strip_reasoning()' cleans a finished response, but a streaming caller has already printed
    each chunk by then. That matters for models whose reasoning cannot be switched off: Muse
    Glimmer's only control is a 'Reasoning strength' instruction line, and even at 'low' it
    deliberates on its 'to=self' channel before every reply - so a streamed session would show
    that deliberation live on every turn, suppression notwithstanding.

    The approach is deliberately generic rather than a per-scheme state machine: re-clean the
    whole buffer on every chunk with 'strip_reasoning()' and release whatever the cleaned text has
    gained since last time. Two details keep that honest:
      - A trailing fragment that could be the start of a control tag ('<|sta') is held back
        until it resolves, so a half-arrived marker is never printed and then 'unprinted'.
      - If cleaning ever rewrites text that has already been released, the filter stops
        releasing rather than print something garbled. It cannot take back what is on screen.

    When reasoning was deliberately enabled the caller asked to see it, so the filter shows it - but as
    readable text, via split_reasoning(): the deliberation with its protocol markers ('<|channel>',
    '</think>', ' to=user<|message|>') removed, then a blank line, then the reply. The same hold-back and
    never-rewrite rules apply as in suppressed mode.

    Answer stops are the filter's second job. Once the answer (never the reasoning) contains one,
    'finished' becomes True and nothing past the stop is released; the caller should then stop
    consuming the stream, which ends generation exactly as a llama.cpp stop string would have.
    """

    # Longer than any control token these schemes use; a held-back tail beyond this is prose.
    _MAX_TAG_LENGTH = 16

    def __init__(self, llm, thinking: Optional[bool] = None, answer_stops: Optional[List[str]] = None,
                 answer_separator: str = "\n\n"):
        """
        Args:
            llm: The loaded llama_cpp.Llama instance generating the stream.
            thinking (bool, optional): Whether reasoning was enabled for this generation. Pass it
                explicitly where the model is shared between threads; otherwise it is read from
                the installed handler.
            answer_stops (list[str], optional): Stops to enforce on the answer only - the second
                value returned by split_stops().
            answer_separator (str): What separates the deliberation from the reply on a '!reason' turn.
        """
        self._llm = llm
        self._thinking = thinking_enabled(llm) if thinking is None else bool(thinking)
        if not supports_thinking(llm):
            self._thinking = False       # '!reason' on an ordinary model is an ordinary turn
        self._separator = answer_separator
        self._answer_stops = [s for s in (answer_stops or []) if s]
        self.finished = False
        self._raw = ""
        self._released = ""
        self._halted = False

    def feed(self, chunk: str) -> str:
        """
        Accepts the next chunk of the stream.

        Args:
            chunk (str): Newly generated text.

        Returns:
            str: The text that is now safe to display; possibly ''.
        """
        if self.finished:
            return ""
        self._raw += chunk
        stable = self._stable_prefix()
        if self._thinking:
            return self._release(self._reasoning_view(stable))

        cleaned = strip_reasoning(stable, self._llm, thinking=False)
        if self._answer_stops:
            truncated = truncate_at_stops(cleaned, self._answer_stops)
            if truncated != cleaned:
                self.finished = True
                return self._release(truncated)
            cleaned = self._hold_stop_prefix(cleaned)
        return self._release(cleaned)

    def flush(self) -> str:
        """
        Ends the stream and releases anything still held back.

        Returns:
            str: The remaining displayable text; possibly ''.
        """
        if self.finished:
            return ""   # everything up to the stop was released in feed()
        if self._thinking:
            return self._release(self._reasoning_view(self._raw, final=True))
        cleaned = strip_reasoning(self._raw, self._llm, thinking=False)
        return self._release(truncate_at_stops(cleaned, self._answer_stops))

    def _stable_prefix(self) -> str:
        """
        The raw text minus any tail still too incomplete to classify.

        A half-arrived control tag ('<|sta') is held back, and so is a Harmony message header still in progress:
        until its '<|message|>' lands it does not look like a header - ' to=self' on its own reads as prose. That
        means everything from the last '<|start|>' with no '<|message|>' after it, and, at the very beginning of the
        stream (where the prompt has already supplied '<|start|>assistant'), anything that is or could become ' to=...'.

        Returns:
            str: The part of the raw text that is safe to interpret.
        """
        stable = self._raw
        cut = stable.rfind("<")
        if cut != -1:
            tail = stable[cut:]
            if ">" not in tail and len(tail) < self._MAX_TAG_LENGTH:
                stable = stable[:cut]
        if scheme_for(self._llm)["scheme"] == SCHEME_HARMONY:
            start = stable.rfind("<|start|>")
            if start != -1 and "<|message|>" not in stable[start:]:
                stable = stable[:start]
            head = stable.lstrip()
            if "<|message|>" not in stable and ("to=".startswith(head) or head.startswith("to=")):
                stable = ""
            # The same at the start of a message body, where a header can arrive stacked on the one just completed
            # (' to=user<|message|>to=us...'; see _harmony_messages). Held until it either completes or turns out to
            # be prose.
            body_at = stable.rfind("<|message|>")
            if body_at != -1:
                body_at += len("<|message|>")
                body = stable[body_at:].lstrip()
                if body and ("to=".startswith(body) or (body.startswith("to=") and "<|message|>" not in body)):
                    stable = stable[:body_at]
        return stable

    def _hold_stop_prefix(self, text: str) -> str:
        """Holds back a tail that could be the start of an answer stop ('Bre' of 'Brent:')."""
        for stop in self._answer_stops:
            for k in range(len(stop) - 1, 0, -1):
                if text.endswith(stop[:k]):
                    return text[:-k]
        return text

    def _reasoning_view(self, text: str, final: bool = False) -> str:
        """
        Renders a '!reason' turn for display: deliberation, separator, reply - markers removed.

        The reply is still watched for the conversational stops that split_stops() withheld from llama.cpp.

        Args:
            text (str): Raw text to render.
            final (bool): True at the end of the stream, when nothing more will arrive to complete a stop.

        Returns:
            str: The display text.
        """
        reasoning, reply = split_reasoning(text, self._llm, thinking=True)
        view = reasoning
        if reply is not None:
            reply = reply.lstrip()
            if self._answer_stops:
                truncated = truncate_at_stops(reply, self._answer_stops)
                if truncated != reply:
                    self.finished = True
                    reply = truncated
                elif not final:
                    reply = self._hold_stop_prefix(reply)
            if reply.strip():
                view = (reasoning + self._separator if reasoning else "") + reply
        return view

    def _release(self, cleaned: str) -> str:
        """Returns the part of 'cleaned' not yet released, or '' if cleaning rewrote history."""
        if self._halted or not cleaned.startswith(self._released):
            self._halted = True
            return ""
        delta = cleaned[len(self._released):]
        self._released = cleaned
        return delta


# --------------------------------------------------------------------------------------------------
# Token counting
# --------------------------------------------------------------------------------------------------

# A short, legal conversation used as the baseline when measuring a single message's cost.
# The content is arbitrary; only the roles matter, since the baseline is subtracted away.
_PROBE_SYSTEM = {"role": "system", "content": "s"}
_PROBE_USER = {"role": "user", "content": "x"}
_PROBE_ASSISTANT = {"role": "assistant", "content": "y"}


def _count_formatter(llm) -> ChatTemplateFormatter:
    """
    Returns (building and caching on first use) the formatter used for token counting.

    This is a separate instance from the generation handler because it must render WITHOUT the
    trailing generation prompt, and because it must not carry reasoning kwargs that would shift
    counts around as the user toggles reasoning mid-conversation.

    Args:
        llm: A loaded llama_cpp.Llama instance.

    Returns:
        ChatTemplateFormatter: The cached counting formatter.
    """
    formatter = getattr(llm, _COUNT_FORMATTER_ATTR, None)
    if formatter is None:
        formatter = ChatTemplateFormatter(llm, add_generation_prompt=False)
        setattr(llm, _COUNT_FORMATTER_ATTR, formatter)
    return formatter


def _tokenize(llm, text: str) -> int:
    """
    Tokenises rendered prompt text the way the model will actually see it.

    Both non-default arguments matter, and the original code passed neither:
      - special=True, or the template's control tokens ('<|im_start|>', '<|start|>') are counted
        as their constituent letters. That inflates every count, and the inflation grows with
        the number of turns - exactly the direction that causes a context overflow.
      - add_bos=False, because the rendered template has already emitted its own BOS token.
        Leaving the default on counts it twice.

    Args:
        llm: A loaded llama_cpp.Llama instance.
        text (str): Rendered prompt text.

    Returns:
        int: The token count.
    """
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))


def _render_or_none(formatter: ChatTemplateFormatter, messages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Renders a message list, returning None instead of raising if the template rejects it.

    Templates disagree sharply about what conversation shapes are legal. Mistral-derived ones
    raise 'conversation roles must alternate user/assistant/...' on two consecutive messages of
    the same role; Qwen's raises 'No user query found in messages' on a system-only conversation;
    some reject a system role outright. Probing is therefore trial and error, and a rejection is
    an ordinary answer rather than an error.

    Args:
        formatter (ChatTemplateFormatter): The formatter to render with.
        messages (list[dict]): The conversation to try.

    Returns:
        str | None: The rendered prompt, or None if the template refused this shape.
    """
    try:
        return formatter.render(messages)
    except Exception:
        return None


# Candidate (baseline, suffix-role) probes per role, most preferred first. Every candidate keeps
# user and assistant strictly alternating after an optional leading system message, because that
# is the narrowest rule any of these templates imposes and it is easier to satisfy it everywhere
# than to detect who imposes it.
_PROBE_CANDIDATES: Dict[str, List[List[Dict[str, Any]]]] = {
    "user": [
        [_PROBE_SYSTEM, _PROBE_USER, _PROBE_ASSISTANT],
        [_PROBE_USER, _PROBE_ASSISTANT],
    ],
    "assistant": [
        [_PROBE_SYSTEM, _PROBE_USER],
        [_PROBE_USER],
    ],
}


def count_message_tokens(llm, role: str, content: str) -> int:
    """
    Counts the tokens a single message adds to a conversation.

    Rendering one message on its own would not answer this question: most templates emit a
    preamble (a BOS token, a default system block, a tool-recipient list) that belongs to the
    conversation rather than to any message in it, and Muse Glimmer's preamble alone runs to
    several hundred tokens. Measuring the difference between a baseline conversation and that
    same conversation plus the message cancels the preamble out and leaves the true incremental
    cost, delimiters included.

    A system message is the exception and is measured in absolute terms, because the caller uses
    that figure as the conversation's fixed floor and adds it exactly once, so it has to include
    the surrounding preamble rather than just the message body. It cannot be measured as a delta
    against a conversation with NO system message either: Muse Glimmer substitutes a long default
    persona when none is supplied, so providing one makes the prompt SHORTER and the delta comes
    out negative.

    Accuracy note: appending a message can slightly change how EARLIER messages render - Qwen
    keeps the '<think>' block on assistant turns after the last user query and drops it from
    those before it, so appending a user message rewrites the assistant turn above it by a few
    tokens. That is a small fixed bias, well inside the caller's context buffer, and it is the
    price of keeping to the strict alternation that other templates demand.

    This function does not raise. A token count that explodes is what broke three models in the
    first place, so an unrenderable template degrades to counting the bare content instead.

    Args:
        llm: A loaded llama_cpp.Llama instance.
        role (str): 'system', 'user' or 'assistant'.
        content (str): The message text.

    Returns:
        int: The number of tokens this message contributes.
    """
    formatter = _count_formatter(llm)

    if role == "system":
        # Rendered with one user message alongside it (system-only is rejected by some templates),
        # then that user message's own cost is subtracted back off.
        whole = _render_or_none(formatter, [{"role": "system", "content": content}, _PROBE_USER])
        if whole is not None:
            return max(0, _tokenize(llm, whole) - count_message_tokens(llm, "user",
                                                                       _PROBE_USER["content"]))
        # Some templates refuse a system role entirely; fall back to rendering it as a user turn.
        whole = _render_or_none(formatter, [{"role": "user", "content": content}])
        if whole is not None:
            return _tokenize(llm, whole)
        return _tokenize(llm, content)

    baselines = getattr(llm, _COUNT_BASELINE_ATTR, None)
    if baselines is None:
        baselines = {}
        setattr(llm, _COUNT_BASELINE_ATTR, baselines)

    probe_role = role if role in _PROBE_CANDIDATES else "user"
    for baseline_messages in _PROBE_CANDIDATES[probe_role]:
        full_messages = baseline_messages + [{"role": probe_role, "content": content}]
        full = _render_or_none(formatter, full_messages)
        if full is None:
            continue
        if probe_role not in baselines:
            baseline = _render_or_none(formatter, baseline_messages)
            if baseline is None:
                continue
            baselines[probe_role] = _tokenize(llm, baseline)
        return max(0, _tokenize(llm, full) - baselines[probe_role])

    # Nothing rendered - count the content alone rather than failing the conversation.
    return _tokenize(llm, content)


def format_chat_prompt(llm, messages: List[Dict[str, Any]],
                       add_generation_prompt: bool = True, **template_kwargs: Any) -> str:
    """
    Renders a conversation to the exact prompt string the model expects.

    This is the function 'llama_utils.universal_token_count()' was originally reaching for when
    it called the non-existent 'Llama._format_chat_prompt()'.

    Args:
        llm: A loaded llama_cpp.Llama instance.
        messages (list[dict]): The conversation, in OpenAI role/content form.
        add_generation_prompt (bool): Whether to append the assistant-turn opener.
        **template_kwargs: Extra template variables, e.g. enable_thinking=False.

    Returns:
        str: The rendered prompt.
    """
    if add_generation_prompt:
        formatter = getattr(llm, _HANDLER_FORMATTER_ATTR, None) or ChatTemplateFormatter(
            llm, add_generation_prompt=True)
    else:
        formatter = _count_formatter(llm)
    return formatter.render(messages, **template_kwargs)

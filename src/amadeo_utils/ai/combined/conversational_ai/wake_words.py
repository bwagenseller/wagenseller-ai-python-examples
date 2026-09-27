"""
Wake words and agents for the conversational AI pipeline.

A client can list several agents, each with its own system prompt, voice and wake words. The server transcribes
every chunk of speech it is sent, then uses this module to decide which agent (if any) the speech was meant for,
so the audio is only transcribed once.

This module holds the parts that are pure logic, so they can be tested without a microphone or a GPU:
* build_agents() - turns a client config (agent_defaults + agents, or the older single-agent keys) into a
  validated list of agent dictionaries. Used by the client.
* select_agent() - given a transcript and the agent list, picks the agent that should answer, or says that it is
  ambiguous and which agents are in play (the server then asks the LLM - see routing.py). Used by the server.
"""

import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

# How far into a transcript a wake word may start (in words) and still count. "Okay, hey Rose, ..." should wake
# Rose, but "I was telling my sister about Rose..." should not.
WAKE_WORD_MAX_POSITION = 10

# The keys an agent may carry, with their expected types. 'wake_words' is a list of strings.
AGENT_KEYS = {
    'name': str,
    'display_name': str,    # how other agents' handoff notes refer to it (see handoff.py); defaults to name.title()
    'wake_words': list,
    'system_prompt_id': str,
    'voice': str,
    'continuous_save': bool,
    'load_previous': bool,
}

# Apostrophes are deleted, so "Rose's" becomes "roses" (a possessive is talk ABOUT Rose, and must not match "rose").
_APOSTROPHES = re.compile(r"['’]")
# Any other punctuation becomes a space, so "Dr. Crane," becomes "dr crane" and "hey-rose" becomes "hey rose".
_PUNCTUATION = re.compile(r"[^\w\s]")

# Words that can open a sentence before a name. A greeting marks the name as spoken TO someone by itself
# ("Hey Rick what do you think"); a filler does not ("And Rick said..."), so the name still needs a pause after it
# ("And Rick, what do you think?").
GREETINGS = {'hey', 'hi', 'hello', 'yo'}
FILLERS = {'ok', 'okay', 'oh', 'alright', 'so', 'well', 'and', 'now'}
# Titles whose full stop is not the end of a sentence ("Dr. Crane")
_ABBREVIATIONS = {'dr', 'mr', 'mrs', 'ms', 'prof', 'st', 'jr', 'sr'}
_SENTENCE_END = '.!?'
_PAUSE = ',;:.!?-–—…'


def normalise(text: str) -> List[str]:
    """
    Lowercases text, strips punctuation and splits it into words.

    Args:
        text: a transcript or a wake word.

    Returns:
        List[str]: the words, in order. Empty if there were none.
    """
    return _PUNCTUATION.sub(' ', _APOSTROPHES.sub('', text.lower())).split()


class _Word(NamedTuple):
    """One normalised word of a transcript, with the punctuation around it that the vocative rules read."""
    text: str
    sentence_start: bool    # first word of the transcript, or the word before it ended a sentence (. ! ?)
    pause_before: bool      # sentence_start, or the word before it ended in any pause (, ; : and so on)
    pause_after: bool       # this word ends in punctuation, or is the last word
    sentence_end: bool      # this word ends a sentence (. ! ?), or is the last word


def _tokenise(transcript: str) -> List[_Word]:
    """
    Splits a transcript into normalised words, keeping what the punctuation said about each one.

    The ASR punctuates its transcripts, and that punctuation is the best cheap evidence of who is being spoken to:
    "Rick, what do you think?" versus "Rick said something odd". A title's full stop ("Dr.") is not a sentence end.

    Args:
        transcript: what the ASR heard.

    Returns:
        List[_Word]: the words, in order.
    """
    words: List[_Word] = []
    raw_tokens = transcript.split()
    previous_ending = _SENTENCE_END[0]      # the start of the transcript counts as the start of a sentence
    for token in raw_tokens:
        pieces = normalise(token)
        if not pieces:
            # pure punctuation, e.g. a free-standing dash: it is still a pause
            if token.strip():
                previous_ending = token.strip()[-1]
            continue
        # the punctuation that ends this token, looking past closing quotes and brackets ('Rick,"' ends in ',')
        bare = token.rstrip('"\')]}\u201d\u2019')
        trailing = bare[-1] if bare and not bare[-1].isalnum() else ''
        # "Dr." is a title, not a sentence end
        if trailing == '.' and pieces[-1] in _ABBREVIATIONS:
            trailing = ''
        for index, piece in enumerate(pieces):
            first, last = index == 0, index == len(pieces) - 1
            words.append(_Word(
                text=piece,
                # (an empty ending must be tested for explicitly: '' is "in" every string)
                sentence_start=first and bool(previous_ending) and previous_ending in _SENTENCE_END,
                pause_before=first and bool(previous_ending) and previous_ending in _PAUSE,
                pause_after=last and bool(trailing) and trailing in _PAUSE,
                sentence_end=last and bool(trailing) and trailing in _SENTENCE_END))
        previous_ending = trailing

    if words:
        # the last word is followed by the end of what was said
        words[-1] = words[-1]._replace(pause_after=True, sentence_end=True)
    return words


class Mention(NamedTuple):
    """One wake word found in a transcript."""
    agent: Dict[str, Any]
    wake_word: str          # as written in the config
    start: int              # index of its first word
    end: int                # index of its last word
    vocative: bool          # said TO the agent (see _is_vocative())


def _is_vocative(words: List[_Word], start: int, end: int) -> bool:
    """
    Whether the name at words[start..end] is said TO someone, judged from punctuation alone.

    Only two shapes count, because they are the ones that are hard to misread:
    * leading: at the start of a sentence (optionally after a filler such as "okay" or "so"), followed by a pause -
      "Rick, what do you think?", "Okay, Rose. ...". After a greeting ("hey", "hi"...) no pause is needed:
      "Hey Rick what do you think".
    * trailing: after a pause, and ending the sentence - "What do you think, Rick?"
    A name in the middle of a sentence ("so, Rick, what...", "my friend, Rick, said...") is not judged - an appositive
    looks the same - so it is left to the LLM.

    Args:
        words: the tokenised transcript.
        start: index of the name's first word.
        end: index of the name's last word.

    Returns:
        bool: True if it is clearly said to someone.
    """
    first = words[start]
    before = words[start - 1] if start > 0 else None
    if first.sentence_start:
        lead = True
        greeted = first.text in GREETINGS and end > start       # the wake word itself is "hey rose"
    elif before and before.sentence_start and (before.text in GREETINGS or before.text in FILLERS):
        lead = True
        greeted = before.text in GREETINGS
    else:
        lead, greeted = False, False

    if lead and (greeted or words[end].pause_after):
        return True
    if first.pause_before and not first.sentence_start and words[end].sentence_end:
        return True
    return False


def _alias_table(agents: List[Dict[str, Any]], key_sets: Tuple[str, ...]) -> List[Tuple[List[str], Dict[str, Any], str]]:
    """
    Every (words, agent, original text) an agent answers to, longest first (the tie-break at one position).

    Args:
        agents: the agent list.
        key_sets: which keys to read: 'wake_words' (a list), 'name' and/or 'display_name' (strings).

    Returns:
        the alias table.
    """
    table = []
    for agent in agents:
        for key in key_sets:
            value = agent.get(key)
            for alias in (value if isinstance(value, list) else [value]):
                if isinstance(alias, str) and normalise(alias):
                    table.append((normalise(alias), agent, alias))
    table.sort(key=lambda entry: len(entry[0]), reverse=True)
    return table


def find_mentions(transcript: str, agents: List[Dict[str, Any]], key_sets: Tuple[str, ...] = ('wake_words',)) -> List[Mention]:
    """
    Finds every wake word in a transcript, left to right.

    Matching is on whole words, after normalise(). At each position the longest wake word wins, and a match is
    skipped over, so "hey rose" is one mention, not two.

    Args:
        transcript: what the ASR heard.
        agents: the agent list (see build_agents()).
        key_sets: what counts as a mention (see _alias_table()); the routing reply also accepts names.

    Returns:
        List[Mention]: in the order they were said.
    """
    words = _tokenise(transcript)
    texts = [w.text for w in words]
    table = _alias_table(agents, key_sets)
    mentions = []
    position = 0
    while position < len(texts):
        for alias_words, agent, alias in table:
            if texts[position:position + len(alias_words)] == alias_words:
                end = position + len(alias_words) - 1
                mentions.append(Mention(agent, alias, position, end, _is_vocative(words, position, end)))
                position = end + 1
                break
        else:
            position += 1
    return mentions


def find_wake_word(transcript: str, agents: List[Dict[str, Any]], max_position: int = WAKE_WORD_MAX_POSITION) -> Optional[Tuple[Dict[str, Any], str]]:
    """
    Finds the first wake word said in a transcript, if it starts within the first max_position words.

    The FIRST wake word said wins: "Rick, do you think Dr. Crane is right?" gives Rick, even though "dr crane" is the
    longer wake word. Only when two wake words start at the same word does the longer one win. This is the fallback
    when several agents are named and neither the rules nor the LLM can tell which was addressed.

    Args:
        transcript: what the ASR heard.
        agents: the agent list (see build_agents()).
        max_position: the wake word must start at word index < max_position. 0 or less means anywhere.

    Returns:
        (agent, wake_word) for the match, or None if no wake word was said in time.
    """
    for mention in find_mentions(transcript, agents):
        if max_position <= 0 or mention.start < max_position:
            return mention.agent, mention.wake_word
        break
    return None


def _distinct_agents(agents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The agents in order, each once (by name)."""
    seen, result = set(), []
    for agent in agents:
        if agent.get('name') not in seen:
            seen.add(agent.get('name'))
            result.append(agent)
    return result


class Selection(NamedTuple):
    """What select_agent() decided."""
    agent: Optional[Dict[str, Any]]     # who answers (for 'ambiguous': the fallback - the first agent named)
    reason: str                         # 'wake_word', 'vocative', 'ambiguous', 'continuation', 'always_on', 'not_addressed'
    candidates: List[Dict[str, Any]]    # for 'ambiguous': the agents in play, for the LLM to choose from


def select_agent(transcript: str, agents: List[Dict[str, Any]], continuation: bool = False, active_agent: str = '',
                 max_position: int = WAKE_WORD_MAX_POSITION) -> Selection:
    """
    Decides which agent should answer a transcript.

    When a wake word starts within the first max_position words, the agents "in play" are every agent named anywhere
    in the transcript, plus - mid-conversation - the agent being talked to. Then:
    1. Only one agent in play: it answers ('wake_word'). Naming the agent you are already talking to never switches.
    2. Several, but exactly one NAMED agent is clearly spoken to (see _is_vocative()): it answers ('vocative').
    3. Otherwise 'ambiguous': the server asks the LLM to choose among the candidates, falling back to the first
       agent named.
    With no wake word in time:
    4. A continuation (the client says the user spoke inside the conversation window) goes to the active agent.
    5. An agent with no wake words is always listening, so it gets anything left (the first such agent).
    6. Otherwise nobody was addressed.

    A blank transcript never wakes anyone and is never a continuation - only an always-listening agent gets it (the
    old behaviour, where the LLM is told "I didn't quite get that.").

    Args:
        transcript: what the ASR heard.
        agents: the agent list (see build_agents()).
        continuation: True if the client says this speech started inside the conversation window.
        active_agent: the name of the agent that the conversation is with (used only when continuation is True).
        max_position: see find_wake_word().

    Returns:
        Selection: the agent (or None), the reason, and the candidates when ambiguous.
    """
    active = None
    if continuation and active_agent:
        active = next((a for a in agents if a.get('name') == active_agent), None)

    if transcript and transcript.strip():
        mentions = find_mentions(transcript, agents)
        if mentions and (max_position <= 0 or mentions[0].start < max_position):
            named = _distinct_agents([m.agent for m in mentions])
            in_play = _distinct_agents(named + ([active] if active else []))
            if len(in_play) == 1:
                return Selection(in_play[0], 'wake_word', [])

            spoken_to = _distinct_agents([m.agent for m in mentions if m.vocative])
            if len(spoken_to) == 1:
                return Selection(spoken_to[0], 'vocative', [])

            return Selection(named[0], 'ambiguous', in_play)

        if active:
            return Selection(active, 'continuation', [])

    for agent in agents:
        if not agent.get('wake_words'):
            return Selection(agent, 'always_on', [])

    return Selection(None, 'not_addressed', [])


def build_agents(config: Dict[str, Any], fallback: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Builds the agent list from a client config.

    New style: 'agents' is a list; each agent starts from 'agent_defaults' and overrides it with its own keys. Old
    style: no 'agents' key, so one agent named 'default' is built from the top-level keys, with no wake words
    (always listening, the behaviour before wake words existed).

    Missing keys come from fallback, which must hold every key in AGENT_KEYS except 'name' and 'wake_words'.

    Args:
        config: the client config dictionary (the parsed JSON file, or the command-line arguments).
        fallback: the values used when neither the agent nor agent_defaults sets a key.

    Returns:
        List[Dict[str, Any]]: the agents, each with every key in AGENT_KEYS set.

    Raises:
        TypeError: a key has the wrong type.
        ValueError: an agent has no name, two agents share a name, or two agents share a wake word.
    """
    if 'agents' in config:
        defaults = config.get('agent_defaults', {})
        raw_agents = config['agents']
        if not isinstance(defaults, dict):
            raise TypeError("'agent_defaults' must be an object.")
        if not isinstance(raw_agents, list) or not raw_agents:
            raise ValueError("'agents' must be a non-empty list.")
    else:
        # Old style: the agent settings are top-level keys.
        defaults = {}
        raw_agents = [{k: config[k] for k in AGENT_KEYS if k in config}]
        raw_agents[0].setdefault('name', 'default')

    agents = []
    seen_names = set()
    seen_wake_words = {}    # normalised wake word -> agent name
    for index, raw in enumerate(raw_agents):
        if not isinstance(raw, dict):
            raise TypeError(f"Agent #{index + 1} must be an object.")

        # fallback, then agent_defaults, then the agent's own keys
        agent = {k: v for k, v in fallback.items() if k in AGENT_KEYS}
        agent['wake_words'] = []
        # name and display_name belong to one agent, so agent_defaults may not set them
        agent.update({k: v for k, v in defaults.items() if k in AGENT_KEYS and k not in ('name', 'display_name')})
        agent.update({k: v for k, v in raw.items() if k in AGENT_KEYS})

        name = agent.get('name')
        if not name or not isinstance(name, str):
            raise ValueError(f"Agent #{index + 1} has no 'name'.")
        agent.setdefault('display_name', name.title())
        if name in seen_names:
            raise ValueError(f"Two agents are named '{name}'.")
        seen_names.add(name)

        for key, expected_type in AGENT_KEYS.items():
            if not isinstance(agent.get(key), expected_type):
                raise TypeError(f"Agent '{name}': '{key}' must be of type {expected_type.__name__}.")

        for wake_word in agent['wake_words']:
            if not isinstance(wake_word, str) or not normalise(wake_word):
                raise ValueError(f"Agent '{name}': every wake word must be a non-empty string (got {wake_word!r}).")
            key = ' '.join(normalise(wake_word))
            if key in seen_wake_words and seen_wake_words[key] != name:
                raise ValueError(f"Wake word '{wake_word}' is used by both '{seen_wake_words[key]}' and '{name}'.")
            seen_wake_words[key] = name

        agents.append(agent)

    return agents

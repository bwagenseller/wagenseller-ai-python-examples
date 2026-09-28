"""
Who said what, for the conversational AI pipeline: speaker tags and handoff notes.

Speaker tags. Several agents - and, one day, several people - share one conversation, but each agent's LLM only ever
sees "user" and "assistant". Unless it is told who is talking, a model fills the gap from whatever names it has read
(Santa once called the user "Rose", after the agent the user had just been talking to). So every request is tagged
with who is speaking and, when the client has several agents, to whom:

    [Kevin, to Santa]: what do you think about that?

The tag is saved in the agent's chat history with the words, so the history keeps saying who spoke each line. The
speaker is the client's player_name, or - with voice recognition on - the voice the ASR server recognized (see
speakers.py); nothing here depends on which.

Handoff notes. Each agent has its own LLM session and chat history, so when the user talks to Rose and then says
"Hey Frasier, what do you think about that?", Frasier has never heard what "that" was. A handoff note fixes this
without sharing histories: the turns the answering agent missed go in front of its request, naming everyone:

    (Since you last spoke: Kevin said to Rose: "Should I repaint the deck?" Rose replied: "Yes, before the frost.")
    [Kevin, to Frasier]: Hey Frasier, what do you think about that?

The note is saved in the agent's chat history with the rest of the request, so the agent can bring up what the
others said on later turns too - characters do, and it makes the conversation feel shared. That is also why it names
every speaker: an unnamed "I" in a saved note is re-read on every later turn, and a model once took "I" to be the
agent the user had been talking to.

The client keeps the turns of the current conversation (it owns the conversation window) and sends them with each
request; the server writes the tag with speaker_tag() and the note with build_handoff_note().
"""

from typing import Any, Dict, List

# At most this many of the missed turns go into a note (the most recent ones)
HANDOFF_MAX_TURNS = 3
# Each quoted utterance or reply is cut to about this many characters, so one long answer can't swamp the request
HANDOFF_MAX_CHARS = 600
# Text between a pair of these is sent to the model for one turn but never saved. It must match
# HIDDEN_INSTRUCTION_DELIMITER in the LLM streams (RolePlayStream, ToolStream, KnowledgeBaseStream); it is repeated
# here rather than imported so this module does not pull in llama_cpp. Nothing here uses it to hide text: it is
# taken out of names, quotes and transcripts, so nothing said can hide part of a request by accident.
HIDDEN_DELIMITER = '##'
# Who is speaking, when neither the turn nor the request says
UNKNOWN_SPEAKER = 'The user'


def shorten(text: str, max_chars: int) -> str:
    """
    Cuts text to at most max_chars characters, at a word boundary where possible, marking the cut with '...'.

    Args:
        text: the text to shorten.
        max_chars: the limit; 0 or less means no limit.

    Returns:
        str: the text, whitespace-trimmed, shortened if it was too long.
    """
    text = ' '.join(text.split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(' ', 1)[0] or text[:max_chars]
    return cut.rstrip(' ,;:') + '...'


def missed_turns(recent_turns: List[Dict[str, Any]], agent_name: str) -> List[Dict[str, Any]]:
    """
    The turns an agent has not heard: every turn after the last one it answered itself.

    Args:
        recent_turns: the conversation's turns, oldest first, each {'agent': name, 'user': text, 'reply': text} and
            optionally 'speaker': who said 'user'. Anything malformed (not a dict, or missing text) is skipped - this
            arrives over the network.
        agent_name: the agent about to answer.

    Returns:
        List[Dict[str, Any]]: the missed turns, oldest first. Empty if the agent answered the most recent turn.
    """
    turns = [t for t in (recent_turns or [])
             if isinstance(t, dict) and isinstance(t.get('agent'), str)
             and isinstance(t.get('user'), str) and isinstance(t.get('reply'), str)]
    for index in range(len(turns) - 1, -1, -1):
        if turns[index]['agent'] == agent_name:
            return turns[index + 1:]
    return turns


def clean_name(name: Any) -> str:
    """
    A speaker name made safe to write into a tag or note: whitespace collapsed, and brackets / the hidden-instruction
    delimiter removed so a name cannot break the tag or open a hidden block.

    Args:
        name: the name, as it arrived over the network (anything that is not a string counts as no name).

    Returns:
        str: the cleaned name; '' if there was none.
    """
    if not isinstance(name, str):
        return ''
    name = name.replace(HIDDEN_DELIMITER, '').replace('[', '').replace(']', '')
    return ' '.join(name.split())


def speaker_tag(speaker: str, addressee: str = '') -> str:
    """
    The label put in front of what was said, telling the agent who is talking and to whom.

    Args:
        speaker: who is talking (see clean_name). With no speaker there is no tag - an old client that sends no
            player_name gets exactly what it used to.
        addressee: the agent's display name. Leave it '' when the client has only one agent: "to" whom is then
            obvious, and an always-listening agent's internal name ('default') means nothing to the model.

    Returns:
        str: e.g. '[Kevin, to Santa]: ' or '[Kevin]: ', or '' with no speaker.
    """
    speaker = clean_name(speaker)
    if not speaker:
        return ''
    addressee = clean_name(addressee)
    return f'[{speaker}, to {addressee}]: ' if addressee else f'[{speaker}]: '


def build_handoff_note(recent_turns: List[Dict[str, Any]], agent_name: str, display_names: Dict[str, str],
                       max_turns: int = HANDOFF_MAX_TURNS, max_chars: int = HANDOFF_MAX_CHARS,
                       default_speaker: str = '') -> str:
    """
    Writes the note that tells an agent what was said to other agents since it last spoke. The note names the
    speaker of every turn, and is saved in the agent's history with the request.

    Args:
        recent_turns: see missed_turns().
        agent_name: the agent about to answer.
        display_names: agent name -> the name to use in the note (e.g. 'crane' -> 'Frasier'). An agent that is not
            listed is called by its name in title case.
        max_turns: at most this many missed turns (the most recent). 0 or less turns notes off.
        max_chars: see shorten().
        default_speaker: who said a turn that does not name its speaker (the request's speaker); UNKNOWN_SPEAKER
            if this is blank too.

    Returns:
        str: the note, ending in a newline, or '' if the agent missed nothing (or notes are off).
    """
    if max_turns <= 0:
        return ''
    turns = missed_turns(recent_turns, agent_name)[-max_turns:]
    if not turns:
        return ''

    def quote(text: str) -> str:
        """Shortens a quotation and takes out the delimiter, which would otherwise hide part of the note."""
        return shorten(text.replace(HIDDEN_DELIMITER, ' '), max_chars)

    fallback = clean_name(default_speaker) or UNKNOWN_SPEAKER
    sentences = []
    for turn in turns:
        who = clean_name(turn.get('speaker')) or fallback
        agent = clean_name(display_names.get(turn['agent'])) or turn['agent'].title()
        sentences.append(f'{who} said to {agent}: "{quote(turn["user"])}" '
                         f'{agent} replied: "{quote(turn["reply"])}"')
    return f"(Since you last spoke: {' Then '.join(sentences)})\n"

"""
Who is speaking, for the conversational AI pipeline, once voice recognition is on (CS-23).

The ASR server works out the speaker from the same audio it transcribes (see amadeo_utils.ai.asr.speaker_id) and
reports it as 'speaker' plus a 'speaker_status'. This module turns that into the one name the rest of the pipeline
uses - the speaker tag, the handoff notes, the saved history and the reply to the client all follow from it (see
handoff.py) - and decides whether an agent that only answers known voices (allow_unknown_speakers false), or only
certain people (allowed_speakers), may answer.

With voice recognition off, nothing changes: the request's speaker (the client's player_name) is used, as before.
"""

from typing import Any, Dict, List, Optional, Tuple

from amadeo_utils.ai.asr.speaker_id import (UNRECOGNIZED_SPEAKER, STATUS_IDENTIFIED, STATUS_UNKNOWN, STATUS_TOO_SHORT,
                                            STATUS_NO_PROFILES)

# What an LLM session's '@@NAME@@' is filled with when voice recognition is on: the session is shared by everyone who
# talks to the agent, so the prompt must not claim the user is one person. Who is talking comes from the speaker tag
# on each turn instead. The server config can change it ('household_name').
HOUSEHOLD_NAME = 'the members of the household'

# How the speaker was decided (logged, and returned to the client as 'speaker_source')
SOURCE_REQUEST = 'request'              # voice recognition off: the client's own speaker / player_name
SOURCE_VOICE = 'voice'                  # the ASR server matched (or failed to match) the voice
SOURCE_LAST_SPEAKER = 'last_speaker'    # too little speech to judge, mid-conversation: whoever spoke last
SOURCE_FALLBACK = 'fallback'            # nothing to go on: an unrecognized voice


def last_speaker(recent_turns: Optional[List[Dict[str, Any]]]) -> str:
    """
    Who spoke the most recent turn of the conversation.

    Args:
        recent_turns: the conversation's turns, oldest first (see handoff.missed_turns); arrives over the network, so
            anything malformed is ignored.

    Returns:
        str: that turn's speaker, or '' if there is none.
    """
    for turn in reversed(recent_turns or []):
        if isinstance(turn, dict) and isinstance(turn.get('speaker'), str) and turn['speaker'].strip():
            return turn['speaker']
    return ''


def resolve_speaker(voice_recognition: bool, asr_response: Dict[str, Any], request_speaker: str, continuation: bool,
                    recent_turns: Optional[List[Dict[str, Any]]]) -> Tuple[str, str]:
    """
    Decides who is speaking this turn.

    * Voice recognition off: the request's speaker.
    * The ASR server judged the voice (identified, unknown, or nobody enrolled): its answer.
    * Too little speech to judge (or nothing was heard): mid-conversation, whoever spoke last; otherwise an
      unrecognized voice.
    * The ASR server could not judge (recognition not configured there, the embedding failed, or an older server
      that reports nothing): an unrecognized voice. This fails closed on purpose - falling back to the client's
      player_name would let any voice past an agent that only answers known voices.

    Args:
        voice_recognition: whether the client asked for recognition.
        asr_response: the ASR server's reply ('speaker', 'speaker_status').
        request_speaker: the request's speaker (the client's player_name).
        continuation: whether this turn continues a conversation under way.
        recent_turns: the conversation's turns so far (see last_speaker).

    Returns:
        Tuple[str, str]: (the speaker, how it was decided - one of the SOURCE_ values).
    """
    if not voice_recognition:
        return request_speaker, SOURCE_REQUEST

    status = asr_response.get('speaker_status')
    speaker = asr_response.get('speaker') if isinstance(asr_response.get('speaker'), str) else ''
    if status == STATUS_IDENTIFIED and speaker:
        return speaker, SOURCE_VOICE
    if status in (STATUS_UNKNOWN, STATUS_NO_PROFILES):
        return UNRECOGNIZED_SPEAKER, SOURCE_VOICE

    heard = asr_response.get('transcription')
    if status == STATUS_TOO_SHORT or not (isinstance(heard, str) and heard.strip()):
        previous = last_speaker(recent_turns) if continuation else ''
        if previous:
            return previous, SOURCE_LAST_SPEAKER
    return UNRECOGNIZED_SPEAKER, SOURCE_FALLBACK


def refuses_unlisted_speaker(voice_recognition: bool, speaker: str, agent: Dict[str, Any]) -> bool:
    """
    Whether an agent must not answer this turn because the speaker is not on its allowed_speakers list.

    Strict on purpose: unlike the unrecognized-voice rule (refuses_unknown_speaker), carrying on a conversation does
    not let anyone else in - someone who chimes in mid-conversation with a restricted agent is refused. An
    unrecognized voice is never on the list. Names are matched ignoring case and surrounding spaces. Note that a
    turn too short to judge, mid-conversation, keeps the previous turn's speaker (resolve_speaker), so a very short
    reply is judged as whoever spoke before it.

    With voice recognition off nothing is refused: the speaker is then only the client's own player_name, which says
    nothing about who is in the room.

    Args:
        voice_recognition: whether recognition is on.
        speaker: the speaker from resolve_speaker().
        agent: the agent chosen to answer; 'allowed_speakers' missing or empty means no restriction. It arrives over
            the network: anything but a list of names refuses everyone (fails closed).

    Returns:
        bool: True if the request must be refused.
    """
    if not voice_recognition:
        return False
    allowed = agent.get('allowed_speakers')
    if allowed is None or allowed == []:
        return False
    if not isinstance(allowed, list):
        return True
    names = {name.strip().casefold() for name in allowed if isinstance(name, str) and name.strip()}
    return speaker == UNRECOGNIZED_SPEAKER or speaker.strip().casefold() not in names


def refuses_unknown_speaker(voice_recognition: bool, speaker: str, agent: Dict[str, Any], continuation: bool,
                            active_agent: str) -> bool:
    """
    Whether an agent must not answer this turn because the voice is not recognized.

    An agent with allow_unknown_speakers false only answers known voices - except to carry on a conversation it is
    already having: an unrecognized voice continuing a conversation with that SAME agent goes through (tagged as an
    unrecognized voice). Continuing a conversation with one agent does not open another: an unrecognized voice that
    wakes a different, gated agent is refused.

    Args:
        voice_recognition: whether recognition is on (with it off, nothing is refused).
        speaker: the speaker from resolve_speaker().
        agent: the agent chosen to answer; 'allow_unknown_speakers' defaults to True.
        continuation: whether this turn continues a conversation under way.
        active_agent: the agent that conversation is with.

    Returns:
        bool: True if the request must be refused.
    """
    if not voice_recognition or speaker != UNRECOGNIZED_SPEAKER:
        return False
    if agent.get('allow_unknown_speakers', True) is not False:
        return False
    return not (continuation and active_agent and active_agent == agent.get('name'))

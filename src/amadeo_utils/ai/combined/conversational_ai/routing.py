"""
Asking the LLM which agent was addressed, for the conversational AI pipeline.

wake_words.select_agent() settles most sentences with rules. When several agents are in play and the rules cannot
tell which one is being spoken TO ("Rick and Frasier, what do you both think?", or "Rick said something odd" in the
middle of a conversation with Frasier), the server asks the LLM, using the LLM server's stateless 'one_shot'
command: this module's system prompt and the transcript go in, a name comes out, and nothing is remembered.

This module holds the pure parts - writing the question and reading the answer - so they can be tested without a
model. The server does the call itself (ConversationalAiServer._route_with_llm()).
"""

from typing import Any, Dict, List, Optional, Tuple

from amadeo_utils.ai.combined.conversational_ai.wake_words import find_mentions

# The reply should be one name; a few spare tokens let a model add a full stop or a title without being cut off
ROUTING_MAX_TOKENS = 12


def display_name(agent: Dict[str, Any]) -> str:
    """The name an agent is called by in prompts: its display_name, or its name in title case."""
    return agent.get('display_name') or agent.get('name', '').title()


def build_routing_prompt(transcript: str, candidates: List[Dict[str, Any]], last_speaker: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """
    Writes the routing question.

    Args:
        transcript: what the user said.
        candidates: the agents in play (at least two), in the order they were named.
        last_speaker: the agent that answered the previous turn of this conversation, if any - a strong hint, since a
            sentence that names nobody in particular is usually still aimed at them.

    Returns:
        (system_prompt, user_request) for a 'one_shot' request.
    """
    names = [display_name(agent) for agent in candidates]
    system_prompt = (
        "You decide who a spoken sentence is addressed to. The user is talking with several assistants: "
        f"{', '.join(names)}. The user may mention one assistant while speaking to another - choose the one being "
        "spoken TO, not the one being talked about."
    )
    if last_speaker is not None:
        system_prompt += f" {display_name(last_speaker)} spoke last, and the user may be replying to them."
    system_prompt += f" Answer with exactly one name from this list and nothing else: {', '.join(names)}."

    user_request = f'The user said: "{transcript.strip()}"\nWho is the user speaking to?'
    return system_prompt, user_request


def parse_routing_reply(reply: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Reads the LLM's answer.

    The answer counts only if it names exactly one of the candidates (by display name, name or wake word). Anything
    else - no name, a name that is not a candidate, two names ("Rick or Frasier") - is rejected, and the server falls
    back to the first agent named.

    Args:
        reply: the LLM's answer.
        candidates: the agents it was asked to choose from.

    Returns:
        The chosen agent, or None if the reply was unusable.
    """
    if not reply or not reply.strip():
        return None
    # Only the first line is the answer. Chattier models add an explanation after it ("Rose\n\nThe user is asking
    # Rose what Rick would say"), which mentions other candidates and would otherwise make a clear answer look like two.
    answer = next(line for line in reply.splitlines() if line.strip())
    mentions = find_mentions(answer, candidates, key_sets=('display_name', 'name', 'wake_words'))
    chosen = {m.agent.get('name'): m.agent for m in mentions}
    if len(chosen) != 1:
        return None
    return next(iter(chosen.values()))

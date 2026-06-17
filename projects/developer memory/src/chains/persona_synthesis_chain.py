"""Chain for developer persona synthesis — Spec §5, §7 (persona_synthesizer node).

Produces a structured PersonaProfile from aggregated TendencyData using
with_structured_output(PersonaProfile). Temperature 0.3 per §5 — enough creativity
to produce a narrative style_summary while keeping structured fields grounded.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from src.models.persona import PersonaProfile

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a developer persona analyst. Your job is to synthesize a coherent developer "
    "profile from aggregated code tendency data. The profile should capture the developer's "
    "characteristic style, recurring design decisions, and technology preferences.\n\n"
    "Write in second person ('This developer...'). Be specific and grounded in the data. "
    "Do not invent patterns not evidenced in the provided tendency data."
)

_HUMAN = (
    "Tendency data aggregated from {doc_count} indexed documents:\n\n"
    "Semantic type distribution: {semantic_type_distribution}\n"
    "Tech stack frequency: {tech_stack_frequency}\n\n"
    "Dominant patterns (representative intent summaries):\n{dominant_patterns}\n\n"
    "Synthesize a PersonaProfile for this developer. You MUST populate all four fields:\n"
    "- style_summary: 2-4 sentence narrative\n"
    "- dominant_patterns: list of 3-6 concise phrases (3-8 words each) distilled from "
    "the dominant patterns above — e.g. 'uses factory pattern for LLM instantiation'. "
    "This list MUST NOT be empty.\n"
    "- tech_preferences: ranked list of technologies from tech_stack_frequency\n"
    "- coaching_notes: observations grounded only in the data above"
)

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


def build_persona_synthesis_chain(llm: ChatOllama) -> Runnable:
    """Build the persona synthesis chain — called once at graph construction time.

    The returned chain accepts a dict with TendencyData fields and returns a
    PersonaProfile with style_summary, dominant_patterns, tech_preferences, coaching_notes.

    Args:
        llm: ChatOllama instance constructed at server startup. Temperature 0.3 per §5.

    Returns:
        A LangChain Runnable: dict → PersonaProfile.
    """
    # with_structured_output enforces PersonaProfile JSON schema at inference time.
    # External data (tendency stats, patterns) go into the human turn.
    structured_llm = llm.with_structured_output(PersonaProfile)
    return _PROMPT | structured_llm

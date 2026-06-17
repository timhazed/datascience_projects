"""Chain for diff coaching analysis — Spec §5, §7 (coaching_analyzer node).

Produces a structured AnalysisResult from a diff + optional persona context.
Temperature 0.0 per §5 — deterministic scoring for consistent coaching feedback.
deviation_score, rationale, and patterns_matched are the structured output fields.
"""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from src.models.analysis import AnalysisResult

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a senior code reviewer and developer coach. Your job is to analyse a code diff "
    "and assess how well it aligns with the developer's established patterns and style.\n\n"
    "Score the deviation on a scale from 0.0 (perfectly aligned) to 1.0 (complete departure). "
    "A score above 0.65 triggers a coaching alert. Be precise and evidence-based.\n\n"
    "If no persona context is provided, use general software engineering best practices "
    "as the baseline."
)

_HUMAN = (
    "Developer persona context:\n{persona_context}\n\n"
    "Code diff to analyse:\n```diff\n{diff_text}\n```\n\n"
    "Analyse the diff and produce: deviation_score, rationale, patterns_matched."
)

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


def build_coaching_chain(llm: ChatOllama) -> Runnable:
    """Build the coaching analysis chain — called once at graph construction time.

    The returned chain accepts {"diff_text": str, "persona_context": str} and returns
    an AnalysisResult with deviation_score, rationale, and patterns_matched.

    Args:
        llm: ChatOllama instance constructed at server startup. Temperature 0.0 per §5.

    Returns:
        A LangChain Runnable: dict → AnalysisResult.
    """
    # with_structured_output enforces AnalysisResult JSON schema.
    # diff_text and persona_context go into the human turn — external data rule.
    structured_llm = llm.with_structured_output(AnalysisResult)
    return _PROMPT | structured_llm

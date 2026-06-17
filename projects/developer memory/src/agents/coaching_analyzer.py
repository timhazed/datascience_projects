"""coaching_analyzer LangGraph node — Spec §2.4, §7.

Generates an AnalysisResult from a diff + optional PersonaProfile using Gemma 4.
Operates with or without persona context — deviation_guard reads the score from the result.
LLM errors are non-fatal — pipeline returns analysis_result=None with error surfaced.
"""

import logging
from collections.abc import Callable

from langchain_ollama import ChatOllama

from src.chains.coaching_chain import build_coaching_chain
from src.models.analysis import AnalysisResult
from src.models.persona import PersonaProfile
from src.utils.retry import invoke_with_retry

logger = logging.getLogger(__name__)


def make_coaching_analyzer_node(llm: ChatOllama) -> Callable[[dict], dict]:
    """Return a coaching_analyzer node with an injected ChatOllama.

    Args:
        llm: ChatOllama instance constructed at server startup. Temperature 0.0 per §5.

    Returns:
        LangGraph node function that reads state["diff_text"] and state["persona_context"]
        and writes state["analysis_result"].
    """
    chain = build_coaching_chain(llm)

    def coaching_analyzer(state: dict) -> dict:
        """Analyse the diff against the developer's persona and produce an AnalysisResult.

        Formats persona_context as a string for the human turn — external data rule.
        When persona_context is None, passes a baseline description so the LLM can
        apply general software engineering best practices as the reference.
        """
        diff_text: str = state.get("diff_text", "")
        persona_context: PersonaProfile | None = state.get("persona_context")
        trace: list[str] = list(state.get("trace", []))

        # Format persona context for the human turn — never the system prompt.
        persona_text = _format_persona(persona_context)

        try:
            result: AnalysisResult = invoke_with_retry(
                chain,
                {"diff_text": diff_text, "persona_context": persona_text},
            )
        except Exception as exc:
            logger.error(
                "coaching_analyzer failed [%s]: %s", type(exc).__name__, str(exc)[:200]
            )
            return {
                "analysis_result": None,
                "error": "Analysis failed. Please retry.",
                "trace": trace + ["coaching_analyzer: error"],
            }

        logger.debug(
            "coaching_analyzer: deviation_score=%.2f, %d patterns matched",
            result.deviation_score,
            len(result.patterns_matched),
        )
        return {
            "analysis_result": result,
            "trace": trace + [f"coaching_analyzer: score={result.deviation_score:.2f}"],
        }

    return coaching_analyzer


def _format_persona(persona: PersonaProfile | None) -> str:
    """Format a PersonaProfile as a text block for the human turn of the prompt.

    Returns a generic baseline description when persona is None, allowing the LLM
    to apply general best practices rather than failing or leaving the field blank.
    """
    if persona is None:
        return "(No developer persona available — apply general software engineering best practices.)"

    lines = [
        f"Style: {persona.style_summary}",
        f"Dominant patterns: {', '.join(persona.dominant_patterns) or 'none'}",
        f"Tech preferences: {', '.join(persona.tech_preferences) or 'none'}",
    ]
    if persona.coaching_notes:
        lines.append(f"Prior coaching notes: {'; '.join(persona.coaching_notes)}")

    return "\n".join(lines)

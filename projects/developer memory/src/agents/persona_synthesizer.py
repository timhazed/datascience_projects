"""persona_synthesizer LangGraph node — Spec §2.3, §7.

Generates a structured PersonaProfile from TendencyData using Gemma 4.
Runs after tendency_scanner produces valid tendency_data.
LLM errors are non-fatal — pipeline returns persona_profile=None with error surfaced.
"""

import logging
from collections.abc import Callable

from langchain_ollama import ChatOllama

from src.chains.persona_synthesis_chain import build_persona_synthesis_chain
from src.models.persona import PersonaProfile, TendencyData
from src.utils.retry import invoke_with_retry

logger = logging.getLogger(__name__)


def make_persona_synthesizer_node(llm: ChatOllama) -> Callable[[dict], dict]:
    """Return a persona_synthesizer node with an injected ChatOllama.

    Args:
        llm: ChatOllama instance constructed at server startup. Temperature 0.3 per §5.

    Returns:
        LangGraph node function that reads state["tendency_data"] and writes
        state["persona_profile"].
    """
    chain = build_persona_synthesis_chain(llm)

    def persona_synthesizer(state: dict) -> dict:
        """Generate a PersonaProfile from TendencyData via Gemma 4.

        Converts TendencyData fields into prompt variables for the human turn.
        External data (tendency stats) goes into the human turn — never system prompt.
        """
        tendency_data: TendencyData | None = state.get("tendency_data")
        trace: list[str] = list(state.get("trace", []))

        if tendency_data is None:
            return {
                "persona_profile": None,
                "error": "No tendency data available for persona synthesis.",
                "trace": trace + ["persona_synthesizer: skipped (no tendency_data)"],
            }

        # Format dominant_patterns as a bulleted list for the human turn
        patterns_text = "\n".join(f"- {p}" for p in tendency_data.dominant_patterns) or "(none)"

        try:
            profile: PersonaProfile = invoke_with_retry(
                chain,
                {
                    "doc_count": tendency_data.doc_count,
                    "semantic_type_distribution": str(tendency_data.semantic_type_distribution),
                    "tech_stack_frequency": str(tendency_data.tech_stack_frequency),
                    "dominant_patterns": patterns_text,
                },
            )
        except Exception as exc:
            logger.error(
                "persona_synthesizer failed [%s]: %s", type(exc).__name__, str(exc)[:200]
            )
            return {
                "persona_profile": None,
                "error": "Persona synthesis failed. Please retry.",
                "trace": trace + ["persona_synthesizer: error"],
            }

        logger.debug("persona_synthesizer: profile synthesized from %d docs", tendency_data.doc_count)
        return {
            "persona_profile": profile,
            "trace": trace + ["persona_synthesizer: ok"],
        }

    return persona_synthesizer

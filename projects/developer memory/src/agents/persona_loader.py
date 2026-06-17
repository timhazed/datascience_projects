"""persona_loader LangGraph node — Spec §2.4, §7.

Loads a lightweight PersonaProfile from ChromaDB indexed data for use as context
in coaching_analyzer. This is a best-effort query — the pipeline continues without
persona context if insufficient data is indexed (diff_validator already ran; there
is always a diff to analyse).

Unlike persona_synthesizer (which calls Gemma 4 to generate a narrative), persona_loader
constructs a PersonaProfile directly from ChromaDB metadata aggregation — no LLM call.
The profile used here is a lightweight signal for coaching alignment, not a publishable
persona document.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from src.db.chroma_client import ChromaLibrarianClient
from src.ingest.embeddings import apply_temporal_weight
from src.models.persona import PersonaProfile
from src.utils.chroma_aggregation import aggregate_chroma_docs

logger = logging.getLogger(__name__)

_MIN_DOCS = 5
_TOP_N = 5


def make_persona_loader_node(chroma: ChromaLibrarianClient) -> Callable[[dict], dict]:
    """Return a persona_loader node with an injected ChromaLibrarianClient.

    Args:
        chroma: ChromaLibrarianClient instance constructed at server startup.

    Returns:
        LangGraph node function that writes state["persona_context"] (PersonaProfile | None).
        Returns None if fewer than _MIN_DOCS documents are indexed — coaching_analyzer
        handles this by using general best-practice baselines.
    """

    def persona_loader(state: dict) -> dict:
        """Query ChromaDB and assemble a lightweight PersonaProfile for coaching context.

        Applies temporal weighting to identify the most recent dominant patterns and
        tech preferences. persona_context=None is a valid return — coaching continues.
        """
        trace: list[str] = list(state.get("trace", []))

        try:
            results = chroma.query(text="developer patterns code style", n_results=50)
        except Exception as exc:
            logger.warning(
                "persona_loader: ChromaDB query failed [%s]: %s — proceeding without persona",
                type(exc).__name__,
                str(exc)[:200],
            )
            return {
                "persona_context": None,
                "trace": trace + ["persona_loader: skipped (ChromaDB error)"],
            }

        if len(results) < _MIN_DOCS:
            logger.info("persona_loader: %d docs — insufficient for persona context", len(results))
            return {
                "persona_context": None,
                "trace": trace + [f"persona_loader: insufficient data ({len(results)} docs)"],
            }

        # Apply temporal weighting — recent docs weighted 3×
        weighted = apply_temporal_weight(results, now=datetime.now(UTC))

        profile = _build_profile(weighted)
        logger.debug("persona_loader: built profile from %d docs", len(results))
        return {
            "persona_context": profile,
            "trace": trace + [f"persona_loader: loaded profile from {len(results)} docs"],
        }

    return persona_loader


def _build_profile(weighted_docs: list[dict]) -> PersonaProfile:
    """Assemble a lightweight PersonaProfile from weighted ChromaDB docs.

    Delegates counter extraction to aggregate_chroma_docs (src/utils/chroma_aggregation.py).
    No LLM call — this is a fast, metadata-only profile for coaching context, not a
    publishable persona document.
    """
    semantic_counter, tech_counter, patterns = aggregate_chroma_docs(weighted_docs)

    top_tech = [t for t, _ in tech_counter.most_common(_TOP_N)]
    top_types = [t for t, _ in semantic_counter.most_common(3)]
    patterns.sort(key=lambda t: t[0], reverse=True)
    dominant = [p for _, p in patterns[:_TOP_N]]

    doc_count = len(weighted_docs)
    style_summary = (
        f"Based on {doc_count} indexed documents. "
        f"Primary semantic types: {', '.join(top_types) or 'unknown'}. "
        f"Top technologies: {', '.join(top_tech) or 'unknown'}."
    )

    return PersonaProfile(
        style_summary=style_summary,
        dominant_patterns=dominant,
        tech_preferences=top_tech,
        coaching_notes=[],
    )

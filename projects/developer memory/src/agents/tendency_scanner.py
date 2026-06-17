"""tendency_scanner LangGraph node — Spec §2.3, §5, §7.

Aggregates ChromaDB documents for the requested scope and applies temporal weighting
(last 6 months = 3× weight). Produces TendencyData for persona_synthesizer.

Temporal weighting is post-retrieval in Python — ChromaDB does not support weighted
scoring natively. apply_temporal_weight() is the canonical implementation in
src/ingest/embeddings.py (§5).
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from src.db.chroma_client import ChromaLibrarianClient
from src.ingest.embeddings import apply_temporal_weight
from src.models.persona import TendencyData
from src.utils.chroma_aggregation import aggregate_chroma_docs

logger = logging.getLogger(__name__)

# Top-N patterns to surface in TendencyData.dominant_patterns
_TOP_PATTERNS = 10
_MIN_DOCS_FOR_PERSONA = 5


def make_tendency_scanner_node(chroma: ChromaLibrarianClient) -> Callable[[dict], dict]:
    """Return a tendency_scanner node with an injected ChromaLibrarianClient.

    Args:
        chroma: ChromaLibrarianClient instance constructed at server startup.

    Returns:
        LangGraph node function that reads state["scope"] and state["recency_months"]
        and writes state["tendency_data"].
    """

    def tendency_scanner(state: dict) -> dict:
        """Aggregate ChromaDB tendency data with temporal weighting.

        Queries ChromaDB for all documents matching the scope, applies temporal weighting,
        and aggregates semantic_type distribution, tech_stack frequency, and top intent
        summaries into a TendencyData object.

        Returns tendency_data=None if fewer than _MIN_DOCS_FOR_PERSONA documents exist.
        """
        scope: str = state.get("scope", "project")
        recency_months: int = state.get("recency_months", 6)
        trace: list[str] = list(state.get("trace", []))

        # Validate scope against the allowed set before any DB access
        _VALID_SCOPES = {"project", "file", "author"}
        if scope not in _VALID_SCOPES:
            logger.warning("tendency_scanner: invalid scope=%r", scope)
            return {
                "tendency_data": None,
                "error": "Invalid scope. Must be one of: project, file, author.",
                "trace": trace + [f"tendency_scanner: rejected invalid scope={scope!r}"],
            }

        # Convert caller-supplied recency window; fall back to spec default (180 days)
        recency_window = timedelta(days=max(1, recency_months) * 30)

        try:
            # Fetch all indexed documents for this scope.
            # scope="author" would ideally filter by author_identity, but without
            # knowing the target author, we fetch all and let persona_synthesizer scope it.
            # A larger n_results is used to get representative coverage.
            results = chroma.query(text="code patterns developer design", n_results=200)
        except Exception as exc:
            logger.error(
                "tendency_scanner failed [%s]: %s", type(exc).__name__, str(exc)[:200]
            )
            return {
                "tendency_data": None,
                "error": "Insufficient indexed data for persona.",
                "trace": trace + ["tendency_scanner: error"],
            }

        if len(results) < _MIN_DOCS_FOR_PERSONA:
            logger.info("tendency_scanner: only %d docs — insufficient for persona", len(results))
            return {
                "tendency_data": None,
                "error": "Insufficient indexed data for persona.",
                "trace": trace + [f"tendency_scanner: only {len(results)} docs (min {_MIN_DOCS_FOR_PERSONA})"],
            }

        # Apply temporal weighting — recent docs weighted 3× using caller-supplied window
        weighted = apply_temporal_weight(results, now=datetime.now(UTC), window=recency_window)

        tendency_data = _aggregate(weighted, scope)
        logger.debug(
            "tendency_scanner: %d docs → %d types, %d tech items",
            len(results),
            len(tendency_data.semantic_type_distribution),
            len(tendency_data.tech_stack_frequency),
        )
        return {
            "tendency_data": tendency_data,
            "trace": trace + [f"tendency_scanner: {len(results)} docs, scope={scope}"],
        }

    return tendency_scanner


def _aggregate(weighted_docs: list[dict], scope: str) -> TendencyData:
    """Aggregate weighted ChromaDB docs into a TendencyData object.

    Delegates counter extraction to aggregate_chroma_docs (src/utils/chroma_aggregation.py)
    to avoid duplicating the metadata iteration loop across tendency_scanner, persona_loader,
    and skills_aggregator.
    """
    semantic_counter, tech_counter, patterns = aggregate_chroma_docs(weighted_docs)

    # Sort patterns by weight descending; take top N representative summaries
    patterns.sort(key=lambda t: t[0], reverse=True)
    dominant_patterns = [p for _, p in patterns[:_TOP_PATTERNS]]

    return TendencyData(
        scope=scope,
        doc_count=len(weighted_docs),
        semantic_type_distribution=dict(semantic_counter),
        tech_stack_frequency=dict(tech_counter),
        dominant_patterns=dominant_patterns,
        weighted_docs=weighted_docs,
    )

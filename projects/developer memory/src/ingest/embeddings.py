"""Embedding model singleton and temporal weighting utility.

Spec §5 — OllamaEmbeddings is constructed once via lru_cache and injected into
ChromaLibrarianClient at server startup. apply_temporal_weight() is a pure utility
called by tendency_scanner after ChromaDB retrieval — ChromaDB does not support
weighted scoring natively so weighting happens post-retrieval in Python.

Embedding model: mxbai-embed-large (1024 dims)
  - Outperforms nomic-embed-text on technical entity retrieval.
  - Pull once: ollama pull mxbai-embed-large
  - ⚠ CHUNK SIZE CONSTRAINT: 512-token context window.
    At ~3 chars/token for dense Python/markdown, max safe chunk size is 1500 chars.
    nomic-embed-text (2048-token window) safely handles 3000-char chunks.
    If switching embedding models, adjust CHUNK_SIZE in multimodal_parser accordingly.
    Exceeding the model's context window returns HTTP 400 from Ollama /api/embed.
"""

import logging
import os
from datetime import UTC, datetime, timedelta
from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

# Embedding model — read from OLLAMA_EMBED_MODEL env var, defaults to nomic-embed-text.
# nomic-embed-text is already pulled and used consistently across all experiments.
# To switch to mxbai-embed-large (1024 dims, 512-token window): set OLLAMA_EMBED_MODEL=mxbai-embed-large
# and run: ollama pull mxbai-embed-large
EMBEDDING_MODEL: str = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Maximum safe chunk size in characters for nomic-embed-text (2048-token context window).
# At ~2 chars/token for dense code, 2048 tokens ≈ 1500 chars (conservative).
# If switching to mxbai-embed-large (512-token window), lower this to 1500.
# Exceeding the model's context window returns HTTP 400 from Ollama /api/embed.
# All parser files (python_parser, markdown_parser, pdf_parser, multimodal_parser)
# import this constant so that a model switch requires only one edit here.
CHUNK_SIZE: int = 1500

# Maximum safe chunk size per embedding model's context window (chars).
# nomic-embed-text: 2048-token window → 1500 chars safe (conservative, ~2 chars/token for code).
# mxbai-embed-large: 512-token window → 1500 chars safe (same limit, tighter budget).
# Both models share the same CHUNK_SIZE here. If a future model has a shorter window,
# lower this constant — all parsers import it so one edit propagates everywhere.
_MAX_SAFE_CHUNK_SIZE_BY_MODEL: dict[str, int] = {
    "nomic-embed-text": 3000,   # 2048-token window; 1500 is conservative, 3000 is the hard cap
    "mxbai-embed-large": 1500,  # 512-token window; 1500 is the hard cap
}

# Warn at startup if CHUNK_SIZE exceeds the selected model's safe limit.
# This prevents silent HTTP 400 errors from Ollama /api/embed at ingest time.
_model_hard_cap = _MAX_SAFE_CHUNK_SIZE_BY_MODEL.get(EMBEDDING_MODEL)
if _model_hard_cap is not None and _model_hard_cap < CHUNK_SIZE:
    logger.warning(
        "CHUNK_SIZE=%d exceeds the safe limit (%d) for embedding model %r. "
        "Ollama /api/embed will return HTTP 400 on oversized chunks. "
        "Lower CHUNK_SIZE or switch to a model with a larger context window.",
        CHUNK_SIZE,
        _model_hard_cap,
        EMBEDDING_MODEL,
    )

# Temporal weighting constants used by tendency_scanner (§5, §2.3)
RECENCY_WINDOW: timedelta = timedelta(days=180)
RECENCY_WEIGHT: float = 3.0


@lru_cache(maxsize=1)  # maxsize=1: single global instance — no arg variants unlike get_llm
def get_embeddings() -> OllamaEmbeddings:
    """Return a cached OllamaEmbeddings instance — constructed once at server startup.

    Reads OLLAMA_HOST from the environment (default: http://localhost:11434).
    In Mode A hybrid deployment, containers set OLLAMA_HOST=http://host.docker.internal:11434
    to reach the native macOS Ollama process.

    Returns:
        OllamaEmbeddings using mxbai-embed-large. The same object on every call.
    """
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=base_url)


def apply_temporal_weight(
    results: list[dict],
    now: datetime,
    window: timedelta | None = None,
) -> list[dict]:
    """Apply recency weighting to ChromaDB result dicts post-retrieval.

    Documents indexed within `window` of `now` receive RECENCY_WEIGHT (3.0).
    Older documents receive weight 1.0. The input list is never mutated — a new list
    of new dicts (spread from the originals) is returned.

    Malformed result dicts (missing metadata or indexed_at) are assigned weight 1.0
    and a warning is logged — they are never dropped, which would silently shrink the
    tendency aggregation input.

    Timezone handling: if `now` is timezone-aware and the parsed `indexed_at` is naive,
    the timestamp is assumed to be UTC and converted to an aware datetime before comparison.
    All ingest timestamps are stored as UTC ISO 8601 strings — mixing is a schema drift
    indicator, not a normal operating condition.

    Args:
        results: ChromaDB result dicts, each expected to have
            result["metadata"]["indexed_at"] as an ISO 8601 timestamp string.
        now: Reference datetime for age calculation. Use datetime.now(UTC) for consistency.
        window: Recency window — documents within this timedelta of `now` receive 3× weight.
            Defaults to RECENCY_WINDOW (180 days) when None.

    Returns:
        A new list of dicts, each a copy of the original with a "weight" key added.
        Empty input returns an empty list.
    """
    # Use the caller-supplied window or fall back to the spec default (180 days)
    effective_window = window if window is not None else RECENCY_WINDOW
    weighted = []
    for r in results:
        try:
            indexed_at = datetime.fromisoformat(r["metadata"]["indexed_at"])
        except (KeyError, ValueError) as exc:
            # Schema drift or partial write — degrade gracefully rather than crashing
            # tendency_scanner. Weight 1.0 is conservative (no boosting for unknown age).
            logger.warning(
                "apply_temporal_weight: malformed result dict [%s]: %s — assigning weight 1.0",
                type(exc).__name__,
                exc,
            )
            weighted.append({**r, "weight": 1.0})
            continue

        # Normalize timezone: if now is aware and indexed_at is naive, assume UTC.
        # Mixing aware and naive datetimes raises TypeError; this prevents that crash.
        if now.tzinfo is not None and indexed_at.tzinfo is None:
            indexed_at = indexed_at.replace(tzinfo=UTC)

        age = now - indexed_at
        # Docs within 180 days are weighted 3× to prevent stale tendencies from older
        # projects diluting the current developer persona (spec §5, tendency_scanner)
        weight = RECENCY_WEIGHT if age <= effective_window else 1.0
        weighted.append({**r, "weight": weight})

    return weighted

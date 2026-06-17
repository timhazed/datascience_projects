"""Tests for src/ingest/embeddings.py.

Phase 2 exit gate — OllamaEmbeddings is never instantiated live. All tests that
exercise get_embeddings() patch langchain_ollama.OllamaEmbeddings.

apply_temporal_weight() is a pure function with no external dependencies — tested
without mocking.

Key cases:
  - get_embeddings() returns singleton
  - apply_temporal_weight(): within-window docs get 3.0 weight
  - apply_temporal_weight(): older docs get 1.0 weight
  - apply_temporal_weight(): empty input returns empty list
  - apply_temporal_weight(): boundary (exactly 180 days) receives 3.0 weight
  - apply_temporal_weight(): malformed result (missing metadata/indexed_at) → weight 1.0, no crash
  - apply_temporal_weight(): timezone-naive indexed_at with timezone-aware now → normalized to UTC
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.embeddings import (
    RECENCY_WEIGHT,
    RECENCY_WINDOW,
    apply_temporal_weight,
    get_embeddings,
)


@pytest.fixture(autouse=True)
def clear_embeddings_cache() -> None:
    """Reset the lru_cache before every test to prevent cross-test singleton leakage."""
    get_embeddings.cache_clear()
    yield
    get_embeddings.cache_clear()


def _make_result(indexed_at: datetime) -> dict:
    """Build a minimal ChromaDB result dict with the given indexed_at timestamp."""
    return {
        "id": "abc123",
        "metadata": {"indexed_at": indexed_at.isoformat()},
        "document": "some content",
    }


class TestGetEmbeddings:
    @patch("src.ingest.embeddings.OllamaEmbeddings")
    def test_returns_singleton(self, mock_cls: MagicMock) -> None:
        """Two calls to get_embeddings() must return the same object — no double construction."""
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        first = get_embeddings()
        second = get_embeddings()

        assert first is second
        mock_cls.assert_called_once()

    @patch("src.ingest.embeddings.OllamaEmbeddings")
    def test_constructed_with_correct_model(self, mock_cls: MagicMock, monkeypatch) -> None:
        """OllamaEmbeddings must be initialised with EMBEDDING_MODEL and base_url from env."""
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
        monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        from src.ingest.embeddings import EMBEDDING_MODEL

        mock_cls.return_value = MagicMock()
        get_embeddings()

        mock_cls.assert_called_once_with(model=EMBEDDING_MODEL, base_url="http://localhost:11434")
        assert EMBEDDING_MODEL == "nomic-embed-text"


class TestApplyTemporalWeight:
    def test_empty_list_returns_empty(self) -> None:
        """Empty input must return an empty list — not raise."""
        now = datetime.now(UTC)
        result = apply_temporal_weight([], now)
        assert result == []

    def test_recent_doc_gets_high_weight(self) -> None:
        """A document indexed 1 day ago is within the 180-day window → RECENCY_WEIGHT (3.0)."""
        now = datetime.now(UTC)
        recent = _make_result(indexed_at=now - timedelta(days=1))

        weighted = apply_temporal_weight([recent], now)

        assert len(weighted) == 1
        assert weighted[0]["weight"] == RECENCY_WEIGHT

    def test_old_doc_gets_unit_weight(self) -> None:
        """A document indexed 181 days ago is outside the window → weight 1.0."""
        now = datetime.now(UTC)
        old = _make_result(indexed_at=now - timedelta(days=181))

        weighted = apply_temporal_weight([old], now)

        assert weighted[0]["weight"] == 1.0

    def test_boundary_exactly_180_days_gets_high_weight(self) -> None:
        """Exactly 180 days old is <= RECENCY_WINDOW — should receive RECENCY_WEIGHT.

        Spec §5: age <= RECENCY_WINDOW → RECENCY_WEIGHT. Boundary is inclusive.
        """
        now = datetime.now(UTC)
        boundary = _make_result(indexed_at=now - RECENCY_WINDOW)

        weighted = apply_temporal_weight([boundary], now)

        assert weighted[0]["weight"] == RECENCY_WEIGHT

    def test_mixed_results_weighted_correctly(self) -> None:
        """A mix of recent and old docs are each weighted independently."""
        now = datetime.now(UTC)
        recent = _make_result(indexed_at=now - timedelta(days=10))
        old = _make_result(indexed_at=now - timedelta(days=365))

        weighted = apply_temporal_weight([recent, old], now)

        assert len(weighted) == 2
        weights = [w["weight"] for w in weighted]
        assert weights == [RECENCY_WEIGHT, 1.0]

    def test_original_dicts_not_mutated(self) -> None:
        """apply_temporal_weight must not mutate the input dicts — returns new dicts."""
        now = datetime.now(UTC)
        doc = _make_result(indexed_at=now - timedelta(days=1))
        original_keys = set(doc.keys())

        apply_temporal_weight([doc], now)

        assert set(doc.keys()) == original_keys
        assert "weight" not in doc

    def test_preserves_existing_fields(self) -> None:
        """The spread {**r, 'weight': ...} must preserve all original fields."""
        now = datetime.now(UTC)
        doc = _make_result(indexed_at=now - timedelta(days=5))

        weighted = apply_temporal_weight([doc], now)

        assert weighted[0]["id"] == doc["id"]
        assert weighted[0]["document"] == doc["document"]
        assert weighted[0]["metadata"] == doc["metadata"]

    def test_recency_constants(self) -> None:
        """Verify RECENCY_WINDOW and RECENCY_WEIGHT match spec values."""
        assert timedelta(days=180) == RECENCY_WINDOW
        assert RECENCY_WEIGHT == 3.0

    # --- Error handling: malformed result dicts ---

    def test_missing_metadata_key_assigns_unit_weight(self) -> None:
        """A result dict with no 'metadata' key gets weight 1.0 — does not raise."""
        now = datetime.now(UTC)
        malformed = {"id": "xyz", "document": "content"}  # no 'metadata' key

        weighted = apply_temporal_weight([malformed], now)

        assert len(weighted) == 1
        assert weighted[0]["weight"] == 1.0
        assert weighted[0]["id"] == "xyz"  # original fields preserved

    def test_missing_indexed_at_key_assigns_unit_weight(self) -> None:
        """A result with metadata dict but no 'indexed_at' key gets weight 1.0."""
        now = datetime.now(UTC)
        malformed = {"id": "xyz", "metadata": {"other_key": "val"}, "document": "c"}

        weighted = apply_temporal_weight([malformed], now)

        assert weighted[0]["weight"] == 1.0

    def test_invalid_indexed_at_value_assigns_unit_weight(self) -> None:
        """A non-ISO-8601 indexed_at string gets weight 1.0 — fromisoformat raises ValueError."""
        now = datetime.now(UTC)
        malformed = {"id": "xyz", "metadata": {"indexed_at": "not-a-date"}, "document": "c"}

        weighted = apply_temporal_weight([malformed], now)

        assert weighted[0]["weight"] == 1.0

    def test_malformed_result_in_mixed_list(self) -> None:
        """A malformed doc in a list with valid docs does not interrupt processing."""
        now = datetime.now(UTC)
        good = _make_result(indexed_at=now - timedelta(days=10))
        bad = {"id": "bad", "document": "no metadata"}

        weighted = apply_temporal_weight([good, bad], now)

        assert len(weighted) == 2
        assert weighted[0]["weight"] == RECENCY_WEIGHT
        assert weighted[1]["weight"] == 1.0

    # --- Timezone normalization ---

    def test_naive_indexed_at_with_aware_now_normalized_to_utc(self) -> None:
        """A timezone-naive indexed_at is normalized to UTC when now is timezone-aware.

        This prevents TypeError from subtracting naive from aware datetime.
        All ingest timestamps are UTC — naive timestamps indicate schema drift, not a
        different timezone.
        """
        now = datetime.now(UTC)
        # Build a naive timestamp (no tzinfo) representing 1 day ago
        naive_indexed_at = (now - timedelta(days=1)).replace(tzinfo=None)
        result = {
            "id": "x",
            "metadata": {"indexed_at": naive_indexed_at.isoformat()},
            "document": "c",
        }

        # Should not raise TypeError; naive timestamp normalized to UTC → treated as recent
        weighted = apply_temporal_weight([result], now)

        assert weighted[0]["weight"] == RECENCY_WEIGHT

    def test_both_aware_datetimes_work_correctly(self) -> None:
        """Both now and indexed_at timezone-aware — standard path, no normalization needed."""
        now = datetime.now(UTC)
        aware_indexed_at = now - timedelta(days=200)  # old, UTC aware
        result = _make_result(indexed_at=aware_indexed_at)

        weighted = apply_temporal_weight([result], now)

        assert weighted[0]["weight"] == 1.0

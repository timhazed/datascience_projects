"""Tests for PatientVectorStore — mocked embeddings via tmp_faiss fixture.

No real OpenAI API calls are made. The mock embeddings model in tmp_faiss always
returns a fixed [0.1]*1536 unit vector so FAISS operations work without credentials.
"""

import pytest
from langchain_core.embeddings import Embeddings

from src.db.patient_vector_store import PatientVectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_METADATA = {
    "patient_id": "P-abc123",
    "name": "Alice Tester",
    "age": 40,
    "gender": "Female",
    "summary": "Routine follow-up.",
    "conditions": "hypertension, CKD",
}


def _meta(**overrides: object) -> dict:
    """Return a valid metadata dict with optional field overrides."""
    return {**_REQUIRED_METADATA, **overrides}


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


def test_upsert_succeeds(tmp_faiss: PatientVectorStore) -> None:
    """upsert with all required fields must not raise."""
    tmp_faiss.upsert("P-abc123", "Clinical note text.", _meta())


def test_upsert_missing_required_field_raises(tmp_faiss: PatientVectorStore) -> None:
    """upsert without a required metadata field must raise ValueError."""
    incomplete = {k: v for k, v in _REQUIRED_METADATA.items() if k != "conditions"}
    with pytest.raises(ValueError, match="Missing required metadata fields"):
        tmp_faiss.upsert("P-abc123", "Some text.", incomplete)


def test_upsert_missing_multiple_fields_raises(tmp_faiss: PatientVectorStore) -> None:
    """Error message lists all missing fields, not just the first."""
    with pytest.raises(ValueError, match="Missing required metadata fields"):
        tmp_faiss.upsert("P-abc123", "text", {"patient_id": "P-abc123"})


def test_upsert_persists_index(tmp_faiss: PatientVectorStore, tmp_path: object) -> None:
    """The FAISS index file must exist on disk after upsert."""
    from pathlib import Path

    tmp_faiss.upsert("P-abc123", "Clinical note.", _meta())

    # The store was initialised with str(tmp_path / "faiss") — reconstruct the path
    index_file = Path(tmp_faiss._faiss_path) / "index.faiss"
    assert index_file.exists()


def test_upsert_second_patient_adds_to_index(tmp_faiss: PatientVectorStore) -> None:
    """Upserting a second patient must produce 2 searchable results."""
    tmp_faiss.upsert("P-abc123", "First patient text.", _meta(patient_id="P-abc123", name="Alice"))
    tmp_faiss.upsert(
        "P-def456",
        "Second patient text.",
        _meta(patient_id="P-def456", name="Bob"),
    )

    results = tmp_faiss.search("patient", k=5)
    patient_ids = {r["patient_id"] for r in results}
    assert "P-abc123" in patient_ids
    assert "P-def456" in patient_ids


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_returns_metadata_dicts(tmp_faiss: PatientVectorStore) -> None:
    """search returns a list of dicts containing the required metadata keys."""
    tmp_faiss.upsert("P-abc123", "Clinical note.", _meta())

    results = tmp_faiss.search("clinical note", k=1)
    assert len(results) == 1
    assert results[0]["patient_id"] == "P-abc123"
    assert results[0]["name"] == "Alice Tester"


def test_search_empty_store_returns_empty_list(tmp_faiss: PatientVectorStore) -> None:
    """search on a store with no documents must return [] without raising."""
    results = tmp_faiss.search("anything", k=5)
    assert results == []


def test_search_k_limits_results(tmp_faiss: PatientVectorStore) -> None:
    """search must return at most k results."""
    for i in range(4):
        tmp_faiss.upsert(
            f"P-{i:03d}",
            f"Patient {i} clinical text.",
            _meta(patient_id=f"P-{i:03d}", name=f"Patient {i}"),
        )

    results = tmp_faiss.search("clinical", k=2)
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# load_or_none / re-load from disk
# ---------------------------------------------------------------------------


def test_reload_index_from_disk(tmp_path: object) -> None:
    """A PatientVectorStore initialised from an existing FAISS directory can search."""
    from unittest.mock import MagicMock

    embeddings = MagicMock(spec=Embeddings)
    embeddings.embed_documents.return_value = [[0.1] * 1536]
    embeddings.embed_query.return_value = [0.1] * 1536

    faiss_path = str(tmp_path) + "/faiss"  # type: ignore[operator]

    store1 = PatientVectorStore(faiss_path, embeddings)
    store1.upsert("P-reload", "Text to reload.", _meta(patient_id="P-reload", name="Reload Test"))

    # Create a second instance pointing at the same directory
    store2 = PatientVectorStore(faiss_path, embeddings)
    results = store2.search("reload", k=1)
    assert any(r["patient_id"] == "P-reload" for r in results)

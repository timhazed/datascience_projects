"""PatientVectorStore — FAISS-backed vector store for patient notes and summaries.

Wraps langchain_community.vectorstores.FAISS with a focused interface for
per-patient upsert and semantic search. The FAISS index is persisted to disk
after every upsert so the store survives restarts.

Required metadata fields for every upsert:
    patient_id, name, age, gender, summary, conditions
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_REQUIRED_METADATA = frozenset({"patient_id", "name", "age", "gender", "summary", "conditions"})


class PatientVectorStore:
    """FAISS vector store for patient clinical text (notes + summary).

    One document per patient — multiple encounters are concatenated before
    embedding. The index is persisted to faiss_path/ on every upsert.
    """

    def __init__(self, faiss_path: str, embeddings_model: Any) -> None:
        """Load an existing FAISS index or initialise an empty one.

        Args:
            faiss_path: Directory path for FAISS index files (index.faiss, index.pkl).
            embeddings_model: LangChain-compatible embeddings instance (e.g. OpenAIEmbeddings).
        """
        self._faiss_path = faiss_path
        self._embeddings = embeddings_model
        self._index: FAISS | None = self._load_or_none()

    def _load_or_none(self) -> FAISS | None:
        """Load index from disk if present, otherwise return None."""
        index_file = Path(self._faiss_path) / "index.faiss"
        if index_file.exists():
            return FAISS.load_local(
                self._faiss_path,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
        return None

    def add(self, patient_id: str, text: str, metadata: dict) -> None:
        """Embed text and add the patient's vector to the in-memory index.

        Does **not** persist to disk. Call persist() once after a bulk load to
        avoid N disk writes. For single-patient registration use upsert() instead.

        Args:
            patient_id: Patient slug — used for logging and stored in metadata.
            text: Full clinical text to embed (notes + summary concatenated).
            metadata: Dict containing all 6 required fields: patient_id, name,
                age, gender, summary, conditions.

        Raises:
            ValueError: If any required metadata field is missing.
        """
        missing = _REQUIRED_METADATA - set(metadata)
        if missing:
            raise ValueError(f"Missing required metadata fields: {missing}")

        doc = Document(page_content=text, metadata=metadata)

        if self._index is None:
            self._index = FAISS.from_documents([doc], self._embeddings)
        else:
            self._index.add_documents([doc])

        logger.debug("Added vector for patient %s (not yet persisted)", patient_id)

    def upsert(self, patient_id: str, text: str, metadata: dict) -> None:
        """Embed text and add the patient's vector, then immediately persist to disk.

        Use for single-patient registration where immediate durability is needed.
        For bulk loads (seeding) use add() + persist() to avoid N disk writes.

        Args:
            patient_id: Patient slug — used for logging and stored in metadata.
            text: Full clinical text to embed (notes + summary concatenated).
            metadata: Dict containing all 6 required fields: patient_id, name,
                age, gender, summary, conditions.

        Raises:
            ValueError: If any required metadata field is missing.
        """
        self.add(patient_id, text, metadata)
        self._persist()
        logger.debug("Upserted vector for patient %s", patient_id)

    def persist(self) -> None:
        """Persist the current in-memory index to disk.

        Call once after a bulk add() sequence. No-op if the index is empty.
        """
        if self._index is not None:
            self._persist()

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Similarity search over the patient index.

        Args:
            query: Natural-language query string.
            k: Number of nearest neighbours to return.

        Returns:
            List of metadata dicts from the k most similar documents.
            Returns an empty list if the index is empty.
        """
        if self._index is None:
            return []
        docs = self._index.similarity_search(query, k=k)
        return [doc.metadata for doc in docs]

    def _persist(self) -> None:
        """Save the current index to disk."""
        Path(self._faiss_path).mkdir(parents=True, exist_ok=True)
        self._index.save_local(self._faiss_path)  # type: ignore[union-attr]

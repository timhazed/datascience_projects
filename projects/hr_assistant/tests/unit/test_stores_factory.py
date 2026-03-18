import pytest
from unittest.mock import patch

from src.stores.stores_factory import get_vector_store
from src.stores.faiss_vector_store import FAISSVectorStore
from src.stores.chroma_vector_store import ChromaVectorStore


class TestGetVectorStore:
    """Tests for the get_vector_store factory function."""

    def test_faiss_store_type(self, mock_embeddings):
        """FAISS store type returns FAISSVectorStore."""
        store = get_vector_store("faiss", "/data/faiss", mock_embeddings)
        assert isinstance(store, FAISSVectorStore)
        assert store.persist_directory == "/data/faiss"
        assert store.embeddings is mock_embeddings

    def test_chroma_store_type(self, mock_embeddings):
        """Chroma store type returns ChromaVectorStore."""
        with patch("src.stores.chroma_vector_store.Chroma"):
            store = get_vector_store("chroma", "/data/chroma", mock_embeddings)
            assert isinstance(store, ChromaVectorStore)
            assert store.persist_directory == "/data/chroma"

    def test_invalid_store_type_raises_error(self, mock_embeddings):
        """Invalid store type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_vector_store("invalid_store", "/data/invalid", mock_embeddings)
        assert "Invalid store type: invalid_store" in str(exc_info.value)

    def test_case_sensitive_store_type(self, mock_embeddings):
        """Store type is case-sensitive."""
        with pytest.raises(ValueError):
            get_vector_store("FAISS", "/data/faiss", mock_embeddings)

    def test_embeddings_passed_correctly(self, mock_embeddings):
        """Embeddings are passed to the store."""
        store = get_vector_store("faiss", "/data/faiss", mock_embeddings)
        assert store.embeddings is mock_embeddings

    def test_empty_store_type_raises_error(self, mock_embeddings):
        """Empty store type raises ValueError."""
        with pytest.raises(ValueError):
            get_vector_store("", "/data/faiss", mock_embeddings)

    def test_persist_directory_passed_correctly(self, mock_embeddings):
        """Persist directory is passed correctly to store."""
        custom_path = "/custom/path/to/store"
        store = get_vector_store("faiss", custom_path, mock_embeddings)
        assert store.persist_directory == custom_path

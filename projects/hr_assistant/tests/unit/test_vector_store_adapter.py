import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.stores.vector_store_adapter import VectorStoreAdapter


class ConcreteVectorStoreAdapter(VectorStoreAdapter):
    """Concrete implementation for testing abstract base class."""

    def db_exists(self) -> bool:
        return True

    def collection_exists(self, collection_name: str) -> bool:
        return True

    def get_store(self, collection_name, documents):
        return MagicMock()


class TestVectorStoreAdapter:
    """Tests for VectorStoreAdapter abstract base class."""

    @pytest.fixture
    def adapter(self, mock_embeddings):
        """Create concrete adapter for testing."""
        return ConcreteVectorStoreAdapter("/data/test", mock_embeddings)

    def test_init_sets_attributes(self, adapter, mock_embeddings):
        """__init__ sets persist_directory and embeddings."""
        assert adapter.persist_directory == "/data/test"
        assert adapter.embeddings is mock_embeddings

    def test_make_chunk_id_with_full_metadata(self, adapter):
        """_make_chunk_id uses metadata when available."""
        doc = Document(
            page_content="test content",
            metadata={
                "document_name": "policy.pdf",
                "page_number": 1,
                "chunk_number": 3
            }
        )
        chunk_id = adapter._make_chunk_id(doc)
        assert chunk_id == "policy.pdf_1_3"

    def test_make_chunk_id_without_metadata_uses_hash(self, adapter):
        """_make_chunk_id falls back to content hash."""
        doc = Document(
            page_content="test content without metadata",
            metadata={}
        )
        with patch("src.stores.vector_store_adapter.get_chunk_id", return_value="hash123"):
            chunk_id = adapter._make_chunk_id(doc)
            assert chunk_id == "hash123"

    def test_make_chunk_id_partial_metadata_uses_hash(self, adapter):
        """_make_chunk_id uses hash when metadata is incomplete."""
        doc = Document(
            page_content="partial metadata content",
            metadata={
                "document_name": "policy.pdf",
            }
        )
        with patch("src.stores.vector_store_adapter.get_chunk_id", return_value="hash456"):
            chunk_id = adapter._make_chunk_id(doc)
            assert chunk_id == "hash456"

    def test_make_chunk_id_none_values_use_hash(self, adapter):
        """_make_chunk_id uses hash when metadata values are None."""
        doc = Document(
            page_content="content",
            metadata={
                "document_name": "policy.pdf",
                "page_number": None,
                "chunk_number": 1
            }
        )
        with patch("src.stores.vector_store_adapter.get_chunk_id", return_value="hash789"):
            chunk_id = adapter._make_chunk_id(doc)
            assert chunk_id == "hash789"

    def test_abstract_methods_must_be_implemented(self, mock_embeddings):
        """Abstract methods cannot be called on base class."""
        with pytest.raises(TypeError):
            VectorStoreAdapter("/data", mock_embeddings)

    def test_make_chunk_id_different_metadata_different_ids(self, adapter):
        """Different metadata produces different IDs."""
        doc1 = Document(
            page_content="content",
            metadata={"document_name": "doc1.pdf", "page_number": 1, "chunk_number": 1}
        )
        doc2 = Document(
            page_content="content",
            metadata={"document_name": "doc2.pdf", "page_number": 1, "chunk_number": 1}
        )
        id1 = adapter._make_chunk_id(doc1)
        id2 = adapter._make_chunk_id(doc2)
        assert id1 != id2

    def test_make_chunk_id_same_metadata_same_id(self, adapter):
        """Same metadata produces same ID regardless of content."""
        doc1 = Document(
            page_content="content A",
            metadata={"document_name": "doc.pdf", "page_number": 1, "chunk_number": 1}
        )
        doc2 = Document(
            page_content="content B",
            metadata={"document_name": "doc.pdf", "page_number": 1, "chunk_number": 1}
        )
        id1 = adapter._make_chunk_id(doc1)
        id2 = adapter._make_chunk_id(doc2)
        assert id1 == id2

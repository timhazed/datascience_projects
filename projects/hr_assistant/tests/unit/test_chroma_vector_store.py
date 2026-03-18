import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.stores.chroma_vector_store import ChromaVectorStore


class TestChromaVectorStore:
    """Tests for ChromaVectorStore implementation."""

    @pytest.fixture
    def chroma_store(self, mock_embeddings):
        """Create ChromaVectorStore instance."""
        with patch("src.stores.chroma_vector_store.Chroma"):
            return ChromaVectorStore("/data/chroma", mock_embeddings)

    # ========== db_exists tests ==========

    def test_db_exists_directory_not_exists(self, chroma_store):
        """db_exists returns False when directory doesn't exist."""
        with patch("os.path.exists", return_value=False):
            assert chroma_store.db_exists() is False

    def test_db_exists_directory_empty(self, chroma_store):
        """db_exists returns False when directory is empty."""
        with patch("os.path.exists", return_value=True):
            with patch("os.listdir", return_value=[]):
                assert chroma_store.db_exists() is False

    def test_db_exists_directory_with_files(self, chroma_store):
        """db_exists returns True when directory has files."""
        with patch("os.path.exists", return_value=True):
            with patch("os.listdir", return_value=["chroma.sqlite3"]):
                assert chroma_store.db_exists() is True

    # ========== collection_exists tests ==========

    def test_collection_exists_returns_true(self, chroma_store):
        """collection_exists returns True when collection is in list."""
        mock_collection = MagicMock()
        mock_collection.name = "hr_policies"

        with patch.object(chroma_store, "db_exists", return_value=True):
            with patch("src.stores.chroma_vector_store.chromadb.PersistentClient") as mock_client:
                mock_client.return_value.list_collections.return_value = [mock_collection]
                assert chroma_store.collection_exists("hr_policies") is True

    def test_collection_exists_returns_false(self, chroma_store):
        """collection_exists returns False when collection not in list."""
        with patch.object(chroma_store, "db_exists", return_value=True):
            with patch("src.stores.chroma_vector_store.chromadb.PersistentClient") as mock_client:
                mock_client.return_value.list_collections.return_value = []
                assert chroma_store.collection_exists("nonexistent") is False

    def test_collection_exists_db_not_exists(self, chroma_store):
        """collection_exists returns False when db doesn't exist."""
        with patch.object(chroma_store, "db_exists", return_value=False):
            assert chroma_store.collection_exists("any_collection") is False

    # ========== get_store tests ==========

    def test_get_store_creates_client_and_collection(self, chroma_store, sample_documents):
        """get_store creates PersistentClient and gets/creates collection."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("src.stores.chroma_vector_store.chromadb.PersistentClient", return_value=mock_client):
            with patch("src.stores.chroma_vector_store.Chroma") as mock_chroma:
                mock_store = MagicMock()
                mock_chroma.return_value = mock_store

                chroma_store.get_store("test_collection", sample_documents)

                mock_client.get_or_create_collection.assert_called_once_with(name="test_collection")

    def test_get_store_filters_existing_documents(self, chroma_store):
        """get_store doesn't add documents that already exist."""
        doc = Document(page_content="test", metadata={"id": "existing_id"})

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["existing_id"]}
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("src.stores.chroma_vector_store.chromadb.PersistentClient", return_value=mock_client):
            with patch("src.stores.chroma_vector_store.Chroma") as mock_chroma:
                mock_store = MagicMock()
                mock_chroma.return_value = mock_store

                chroma_store.get_store("test", [doc])

                mock_store.add_documents.assert_not_called()

    def test_get_store_adds_new_documents(self, chroma_store, sample_documents):
        """get_store adds documents that don't exist."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("src.stores.chroma_vector_store.chromadb.PersistentClient", return_value=mock_client):
            with patch("src.stores.chroma_vector_store.Chroma") as mock_chroma:
                mock_store = MagicMock()
                mock_chroma.return_value = mock_store

                chroma_store.get_store("test", sample_documents)

                mock_store.add_documents.assert_called_once()

    def test_get_store_generates_ids_for_documents(self, chroma_store, document_without_metadata):
        """get_store generates IDs for documents without them."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("src.stores.chroma_vector_store.chromadb.PersistentClient", return_value=mock_client):
            with patch("src.stores.chroma_vector_store.Chroma") as mock_chroma:
                mock_store = MagicMock()
                mock_chroma.return_value = mock_store

                with patch.object(chroma_store, "_make_chunk_id", return_value="generated_id"):
                    chroma_store.get_store("test", [document_without_metadata])

    def test_get_store_returns_chroma_store(self, chroma_store, sample_documents):
        """get_store returns the Chroma store instance."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("src.stores.chroma_vector_store.chromadb.PersistentClient", return_value=mock_client):
            with patch("src.stores.chroma_vector_store.Chroma") as mock_chroma:
                mock_store = MagicMock()
                mock_chroma.return_value = mock_store

                result = chroma_store.get_store("test", sample_documents)

                assert result is mock_store

    def test_get_store_passes_correct_params_to_chroma(self, chroma_store, sample_documents):
        """get_store passes correct parameters to Chroma constructor."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("src.stores.chroma_vector_store.chromadb.PersistentClient", return_value=mock_client):
            with patch("src.stores.chroma_vector_store.Chroma") as mock_chroma:
                mock_store = MagicMock()
                mock_chroma.return_value = mock_store

                chroma_store.get_store("test_collection", sample_documents)

                call_kwargs = mock_chroma.call_args[1]
                assert call_kwargs["client"] is mock_client
                assert call_kwargs["collection_name"] == "test_collection"

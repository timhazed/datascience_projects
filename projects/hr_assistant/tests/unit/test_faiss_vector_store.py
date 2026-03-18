import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.stores.faiss_vector_store import FAISSVectorStore


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore implementation."""

    @pytest.fixture
    def faiss_store(self, mock_embeddings):
        """Create FAISSVectorStore instance."""
        return FAISSVectorStore("/data/faiss", mock_embeddings)

    # ========== db_exists tests ==========

    def test_db_exists_directory_not_exists(self, faiss_store):
        """db_exists returns False when directory doesn't exist."""
        with patch("os.path.exists", return_value=False):
            assert faiss_store.db_exists() is False

    def test_db_exists_directory_exists_with_index(self, faiss_store):
        """db_exists returns True when directory and index files exist."""
        with patch("os.path.exists", return_value=True):
            assert faiss_store.db_exists() is True

    def test_db_exists_directory_exists_no_faiss_file(self, faiss_store):
        """db_exists returns False when .faiss file is missing."""
        def exists_side_effect(path):
            if path == "/data/faiss":
                return True
            if "index.faiss" in path:
                return False
            return True
        with patch("os.path.exists", side_effect=exists_side_effect):
            assert faiss_store.db_exists() is False

    # ========== collection_exists tests ==========

    def test_collection_exists_both_files_present(self, faiss_store):
        """collection_exists returns True when .faiss and .pkl exist."""
        with patch("os.path.exists", return_value=True):
            assert faiss_store.collection_exists("index") is True

    def test_collection_exists_faiss_missing(self, faiss_store):
        """collection_exists returns False when .faiss is missing."""
        def exists_side_effect(path):
            return "index.pkl" in path
        with patch("os.path.exists", side_effect=exists_side_effect):
            assert faiss_store.collection_exists("index") is False

    def test_collection_exists_pkl_missing(self, faiss_store):
        """collection_exists returns False when .pkl is missing."""
        def exists_side_effect(path):
            return "index.faiss" in path
        with patch("os.path.exists", side_effect=exists_side_effect):
            assert faiss_store.collection_exists("index") is False

    def test_collection_exists_checks_correct_paths(self, faiss_store):
        """collection_exists checks correct file paths."""
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            faiss_store.collection_exists("my_collection")

            calls = [str(c) for c in mock_exists.call_args_list]
            assert any("my_collection.faiss" in str(c) for c in calls)
            assert any("my_collection.pkl" in str(c) for c in calls)

    # ========== get_store tests ==========

    def test_get_store_loads_existing_collection(self, faiss_store, sample_documents):
        """get_store loads existing index and syncs documents."""
        mock_store = MagicMock()
        mock_store.docstore._dict = {
            "doc1": Document(page_content="test", metadata={"id": "existing_id"})
        }

        with patch.object(faiss_store, "collection_exists", return_value=True):
            with patch("src.stores.faiss_vector_store.FAISS.load_local", return_value=mock_store):
                result = faiss_store.get_store("test_collection", sample_documents)
                assert result is mock_store

    def test_get_store_creates_new_collection(self, faiss_store, sample_documents):
        """get_store creates new index when collection doesn't exist."""
        mock_store = MagicMock()

        with patch.object(faiss_store, "collection_exists", return_value=False):
            with patch("src.stores.faiss_vector_store.FAISS.from_documents", return_value=mock_store):
                result = faiss_store.get_store("new_collection", sample_documents)

                assert result is mock_store
                mock_store.save_local.assert_called_once()

    def test_get_store_adds_new_documents(self, faiss_store, sample_documents):
        """get_store adds only new documents to existing index."""
        mock_store = MagicMock()
        mock_store.docstore._dict = {}

        with patch.object(faiss_store, "collection_exists", return_value=True):
            with patch("src.stores.faiss_vector_store.FAISS.load_local", return_value=mock_store):
                faiss_store.get_store("test_collection", sample_documents)

                mock_store.add_documents.assert_called_once()
                mock_store.save_local.assert_called_once()

    def test_get_store_skips_existing_documents(self, faiss_store):
        """get_store doesn't add documents that already exist."""
        new_doc = Document(
            page_content="existing content",
            metadata={"id": "existing_chunk_id"}
        )

        mock_store = MagicMock()
        mock_store.docstore._dict = {
            "internal_id": Document(page_content="x", metadata={"id": "existing_chunk_id"})
        }

        with patch.object(faiss_store, "collection_exists", return_value=True):
            with patch("src.stores.faiss_vector_store.FAISS.load_local", return_value=mock_store):
                faiss_store.get_store("test", [new_doc])

                mock_store.add_documents.assert_not_called()

    def test_get_store_assigns_ids_to_documents(self, faiss_store, document_without_metadata):
        """get_store assigns IDs to documents without them."""
        mock_store = MagicMock()

        with patch.object(faiss_store, "collection_exists", return_value=False):
            with patch("src.stores.faiss_vector_store.FAISS.from_documents", return_value=mock_store):
                faiss_store.get_store("test", [document_without_metadata])

                assert "id" in document_without_metadata.metadata

    def test_get_store_uses_dangerous_deserialization(self, faiss_store, sample_documents):
        """get_store enables allow_dangerous_deserialization for pickle."""
        mock_store = MagicMock()
        mock_store.docstore._dict = {}

        with patch.object(faiss_store, "collection_exists", return_value=True):
            with patch("src.stores.faiss_vector_store.FAISS.load_local", return_value=mock_store) as mock_load:
                faiss_store.get_store("test", sample_documents)

                mock_load.assert_called_once()
                call_kwargs = mock_load.call_args[1]
                assert call_kwargs["allow_dangerous_deserialization"] is True

    def test_get_store_passes_correct_parameters_to_load_local(self, faiss_store, sample_documents):
        """get_store passes correct parameters to FAISS.load_local."""
        mock_store = MagicMock()
        mock_store.docstore._dict = {}

        with patch.object(faiss_store, "collection_exists", return_value=True):
            with patch("src.stores.faiss_vector_store.FAISS.load_local", return_value=mock_store) as mock_load:
                faiss_store.get_store("my_index", sample_documents)

                mock_load.assert_called_once()
                call_args = mock_load.call_args
                assert call_args[0][0] == "/data/faiss"
                assert call_args[1]["index_name"] == "my_index"

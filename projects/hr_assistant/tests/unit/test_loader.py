import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.loader.loader import Loader


class TestLoaderInit:
    """Tests for Loader initialization."""

    def test_init_with_valid_directory(self):
        """Loader initializes with valid directory."""
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["file1.pdf", "file2.txt"]):
                loader = Loader(file_path="/data/documents")
                assert loader.file_path == "/data/documents"

    def test_init_with_nonexistent_directory(self):
        """Loader raises ValueError for nonexistent directory."""
        with patch("os.path.isdir", return_value=False):
            with patch("os.listdir", return_value=[]):
                with pytest.raises(ValueError) as exc_info:
                    Loader(file_path="/nonexistent/path")
                assert "is not valid" in str(exc_info.value)

    def test_init_with_empty_directory(self):
        """Loader raises ValueError for empty directory."""
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=[]):
                with pytest.raises(ValueError) as exc_info:
                    Loader(file_path="/empty/path")
                assert "is not valid" in str(exc_info.value)

    def test_init_sets_text_splitter(self):
        """Loader initializes RecursiveCharacterTextSplitter."""
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["file.pdf"]):
                loader = Loader(file_path="/data", chunk_size=500, chunk_overlap=100)
                assert loader.text_splitter._chunk_size == 500
                assert loader.text_splitter._chunk_overlap == 100

    def test_init_default_chunk_parameters(self):
        """Loader uses default chunk parameters."""
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["file.pdf"]):
                loader = Loader(file_path="/data")
                assert loader.text_splitter._chunk_size == 1000
                assert loader.text_splitter._chunk_overlap == 200

    def test_init_creates_loaders_list(self):
        """Loader creates DirectoryLoaders for pdf and txt."""
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["file.pdf"]):
                loader = Loader(file_path="/data")
                assert len(loader.loaders_list) == 2


class TestLoaderValidateDirectoryPath:
    """Tests for _validate_directory_path method."""

    def test_validate_existing_nonempty_directory(self):
        """Returns True for existing non-empty directory."""
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["file.pdf"]):
                loader = Loader.__new__(Loader)
                result = loader._validate_directory_path("/data")
                assert result is True

    def test_validate_nonexistent_directory(self):
        """Returns False for nonexistent directory."""
        with patch("os.path.isdir", return_value=False):
            with patch("os.listdir", return_value=[]):
                loader = Loader.__new__(Loader)
                result = loader._validate_directory_path("/nonexistent")
                assert result is False

    def test_validate_empty_directory(self):
        """Returns False for empty directory."""
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=[]):
                loader = Loader.__new__(Loader)
                result = loader._validate_directory_path("/empty")
                assert result is False


class TestLoaderCreateLoader:
    """Tests for _create_loader method."""

    def test_create_loader_pdf(self):
        """Creates PyPDFLoader for .pdf files."""
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["file.pdf"]):
                with patch("src.loader.loader.PyPDFLoader") as mock_pdf_loader:
                    loader = Loader(file_path="/data")
                    loader._create_loader("/path/to/document.pdf")
                    mock_pdf_loader.assert_called_with("/path/to/document.pdf")

    def test_create_loader_txt(self):
        """Creates TextLoader for .txt files."""
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["file.txt"]):
                with patch("src.loader.loader.TextLoader") as mock_text_loader:
                    loader = Loader(file_path="/data")
                    loader._create_loader("/path/to/document.txt")
                    mock_text_loader.assert_called_with("/path/to/document.txt")

    def test_create_loader_unknown_defaults_to_text(self):
        """Defaults to TextLoader for unknown extensions."""
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["file.md"]):
                with patch("src.loader.loader.TextLoader") as mock_text_loader:
                    loader = Loader(file_path="/data")
                    loader._create_loader("/path/to/document.md")
                    mock_text_loader.assert_called_with("/path/to/document.md")

    def test_create_loader_uppercase_extension(self):
        """Handles uppercase extensions correctly."""
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["file.PDF"]):
                with patch("src.loader.loader.PyPDFLoader") as mock_pdf_loader:
                    loader = Loader(file_path="/data")
                    loader._create_loader("/path/to/document.PDF")
                    mock_pdf_loader.assert_called_once()


class TestLoaderLoad:
    """Tests for load method."""

    def test_load_returns_chunks_with_metadata(self):
        """load returns chunks with proper metadata."""
        raw_doc = Document(
            page_content="This is test content for HR policy.",
            metadata={"source": "/data/hr_policy.pdf", "page": 0}
        )

        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["hr_policy.pdf"]):
                with patch("src.loader.loader.DirectoryLoader") as mock_dir_loader:
                    mock_instance = MagicMock()
                    mock_instance.load.return_value = [raw_doc]
                    mock_dir_loader.return_value = mock_instance

                    loader = Loader(file_path="/data")
                    chunks = loader.load()

                    assert len(chunks) > 0
                    assert "document_name" in chunks[0].metadata
                    assert "chunk_number" in chunks[0].metadata
                    assert "id" in chunks[0].metadata

    def test_load_groups_by_source(self):
        """load groups documents by source before splitting."""
        docs = [
            Document(page_content="Page 1 content", metadata={"source": "/data/file1.pdf"}),
            Document(page_content="Page 2 content", metadata={"source": "/data/file1.pdf"}),
            Document(page_content="Different file", metadata={"source": "/data/file2.txt"}),
        ]

        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["file1.pdf", "file2.txt"]):
                with patch("src.loader.loader.DirectoryLoader") as mock_dir_loader:
                    mock_instance = MagicMock()
                    mock_instance.load.return_value = docs
                    mock_dir_loader.return_value = mock_instance

                    loader = Loader(file_path="/data")
                    chunks = loader.load()

                    doc_names = set(c.metadata["document_name"] for c in chunks)
                    assert "file1.pdf" in doc_names
                    assert "file2.txt" in doc_names

    def test_load_joins_pages_with_newlines(self):
        """load joins multi-page PDFs with double newlines."""
        docs = [
            Document(page_content="First page", metadata={"source": "/data/doc.pdf"}),
            Document(page_content="Second page", metadata={"source": "/data/doc.pdf"}),
        ]

        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["doc.pdf"]):
                with patch("src.loader.loader.DirectoryLoader") as mock_dir_loader:
                    mock_instance = MagicMock()
                    mock_instance.load.return_value = docs
                    mock_dir_loader.return_value = mock_instance

                    loader = Loader(file_path="/data", chunk_size=10000)
                    chunks = loader.load()

                    full_content = chunks[0].page_content
                    assert "First page" in full_content
                    assert "Second page" in full_content

    def test_load_assigns_sequential_chunk_numbers(self):
        """load assigns sequential chunk numbers starting from 1."""
        long_content = "x" * 2500
        doc = Document(page_content=long_content, metadata={"source": "/data/long.txt"})

        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["long.txt"]):
                with patch("src.loader.loader.DirectoryLoader") as mock_dir_loader:
                    mock_instance = MagicMock()
                    mock_instance.load.return_value = [doc]
                    mock_dir_loader.return_value = mock_instance

                    loader = Loader(file_path="/data", chunk_size=1000, chunk_overlap=0)
                    chunks = loader.load()

                    chunk_numbers = [c.metadata["chunk_number"] for c in chunks]
                    assert chunk_numbers == list(range(1, len(chunks) + 1))

    def test_load_generates_unique_ids(self):
        """load generates unique IDs for each chunk."""
        docs = [
            Document(page_content="Unique content A", metadata={"source": "/data/a.txt"}),
            Document(page_content="Unique content B", metadata={"source": "/data/b.txt"}),
        ]

        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["a.txt", "b.txt"]):
                with patch("src.loader.loader.DirectoryLoader") as mock_dir_loader:
                    mock_instance = MagicMock()
                    mock_instance.load.return_value = docs
                    mock_dir_loader.return_value = mock_instance

                    loader = Loader(file_path="/data")
                    chunks = loader.load()

                    ids = [c.metadata["id"] for c in chunks]
                    assert len(ids) == len(set(ids))

    def test_load_handles_empty_loaders(self):
        """load handles case when loaders return no documents."""
        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["file.pdf"]):
                with patch("src.loader.loader.DirectoryLoader") as mock_dir_loader:
                    mock_instance = MagicMock()
                    mock_instance.load.return_value = []
                    mock_dir_loader.return_value = mock_instance

                    loader = Loader(file_path="/data")
                    chunks = loader.load()

                    assert chunks == []

    def test_load_extracts_document_name_from_source(self):
        """load extracts document_name from source path."""
        doc = Document(
            page_content="Content",
            metadata={"source": "/path/to/my_document.pdf"}
        )

        with patch("os.path.isdir", return_value=True):
            with patch("os.listdir", return_value=["my_document.pdf"]):
                with patch("src.loader.loader.DirectoryLoader") as mock_dir_loader:
                    mock_instance = MagicMock()
                    mock_instance.load.return_value = [doc]
                    mock_dir_loader.return_value = mock_instance

                    loader = Loader(file_path="/data")
                    chunks = loader.load()

                    assert chunks[0].metadata["document_name"] == "my_document.pdf"

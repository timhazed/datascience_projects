import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


# ============================================================================
# DOCUMENT FIXTURES
# ============================================================================

@pytest.fixture
def sample_document():
    """Single document with full metadata."""
    return Document(
        page_content="Section 4.2: Senior associates get 25 days of PTO.",
        metadata={
            "source": "/data/hr_policy.pdf",
            "document_name": "hr_policy.pdf",
            "page_number": 1,
            "chunk_number": 1,
            "id": "test_id_123"
        }
    )


@pytest.fixture
def sample_documents():
    """List of documents for testing batch operations."""
    return [
        Document(
            page_content="Section 4.2: Senior associates get 25 days of PTO.",
            metadata={
                "source": "/data/hr_policy.pdf",
                "document_name": "hr_policy.pdf",
                "page_number": 1,
                "chunk_number": 1,
            }
        ),
        Document(
            page_content="Vacation days must be requested 2 weeks in advance.",
            metadata={
                "source": "/data/hr_policy.pdf",
                "document_name": "hr_policy.pdf",
                "page_number": 1,
                "chunk_number": 2,
            }
        ),
        Document(
            page_content="Remote work policy allows up to 3 days per week.",
            metadata={
                "source": "/data/remote_policy.txt",
                "document_name": "remote_policy.txt",
                "page_number": 1,
                "chunk_number": 1,
            }
        ),
    ]


@pytest.fixture
def document_without_metadata():
    """Document without metadata for hash-based ID generation."""
    return Document(
        page_content="This is content without proper metadata.",
        metadata={}
    )


# ============================================================================
# EMBEDDINGS FIXTURES
# ============================================================================

@pytest.fixture
def mock_embeddings():
    """Mock OpenAIEmbeddings that returns deterministic vectors."""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 10
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock


# ============================================================================
# LLM FIXTURES
# ============================================================================

@pytest.fixture
def mock_llm():
    """Mock LLM that returns predictable responses."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="SAFE")
    return mock


@pytest.fixture
def mock_llm_unsafe():
    """Mock LLM that flags content as unsafe."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="UNSAFE")
    return mock


# ============================================================================
# VECTOR STORE FIXTURES
# ============================================================================

@pytest.fixture
def mock_faiss_store():
    """Mock FAISS vector store with docstore."""
    mock_store = MagicMock()
    mock_store.docstore._dict = {
        "doc1": Document(page_content="test", metadata={"id": "existing_id_1"}),
        "doc2": Document(page_content="test2", metadata={"id": "existing_id_2"}),
    }
    mock_store.as_retriever.return_value = MagicMock()
    return mock_store


@pytest.fixture
def mock_chroma_client():
    """Mock Chroma PersistentClient."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": ["existing_id_1", "existing_id_2"]}
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.list_collections.return_value = [MagicMock(name="test_collection")]
    return mock_client


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for HRAssistant tests."""
    mock = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(
            page_content="PTO policy content here",
            metadata={"document_name": "hr_policy.pdf", "chunk_number": 1}
        )
    ]
    mock.as_retriever.return_value = mock_retriever
    return mock


# ============================================================================
# SETTINGS/CONFIG FIXTURES
# ============================================================================

@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        "provider": {
            "openai": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.2,
                "max_tokens": 512
            }
        },
        "store": {
            "faiss": {
                "persist_directory": "data/faiss"
            }
        }
    }


@pytest.fixture
def sample_config_groq():
    """Sample configuration with Groq provider."""
    return {
        "provider": {
            "groq": {
                "model": "llama-3.3-70b-versatile",
                "temperature": 0.2,
                "max_tokens": 512
            }
        },
        "store": {
            "chroma": {
                "persist_directory": "data/chroma"
            }
        }
    }


@pytest.fixture
def invalid_config_multiple_providers():
    """Invalid config with multiple providers."""
    return {
        "provider": {
            "openai": {"model": "gpt-3.5-turbo"},
            "groq": {"model": "llama-3.3-70b-versatile"}
        },
        "store": {
            "faiss": {"persist_directory": "data/faiss"}
        }
    }


# ============================================================================
# FILE SYSTEM FIXTURES
# ============================================================================

@pytest.fixture
def mock_directory_exists():
    """Mock os.path functions for existing directory."""
    with patch("os.path.isdir", return_value=True):
        with patch("os.listdir", return_value=["file1.pdf", "file2.txt"]):
            yield


@pytest.fixture
def mock_directory_not_exists():
    """Mock os.path functions for non-existent directory."""
    with patch("os.path.isdir", return_value=False):
        yield


@pytest.fixture
def mock_empty_directory():
    """Mock os.path functions for empty directory."""
    with patch("os.path.isdir", return_value=True):
        with patch("os.listdir", return_value=[]):
            yield


# ============================================================================
# ENVIRONMENT FIXTURES
# ============================================================================

@pytest.fixture
def mock_env_vars():
    """Mock environment variables."""
    with patch.dict("os.environ", {
        "OPENAI_API_KEY": "test-api-key-123",
        "GROQ_API_KEY": "test-groq-key-456"
    }):
        yield


@pytest.fixture
def mock_load_dotenv():
    """Mock load_dotenv to prevent actual .env loading."""
    with patch("dotenv.load_dotenv"):
        yield

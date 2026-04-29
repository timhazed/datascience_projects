"""Shared pytest fixtures.

"""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.embeddings import Embeddings

from src.config.settings import Settings
from src.db.appointment_db import AppointmentDB
from src.db.patient_db import PatientDB
from src.db.patient_vector_store import PatientVectorStore

# ---------------------------------------------------------------------------
# Canonical test config — matches config.yaml structure; safe for all tests
# ---------------------------------------------------------------------------

_TEST_CONFIG_YAML = textwrap.dedent("""\
    provider:
      groq:
        model: "llama-3.3-70b-versatile"
        temperature: 0.1
        temperature_summarizer: 0.3
        temperature_search: 0.3
        max_tokens_planner: 500
        max_tokens_history: 400
        max_tokens_search: 650
        max_tokens_appointment: 150
        max_tokens_summarizer: 700
        max_tokens_guard: 25
    search:
      provider: "serper"
    embeddings:
      provider: "openai"
      model: "text-embedding-3-small"
    db:
      sqlite_path: "data/healthcare.db"
      faiss_path: "data/faiss"
""")


@pytest.fixture(scope="session")
def mock_settings(tmp_path_factory) -> Settings:
    """Settings instance loaded from a minimal in-memory test config.

    Scoped to the session — constructed once and reused across all tests.
    Tests that need custom values should construct Settings(**_valid_data())
    directly rather than modifying this fixture.

    Returns:
        Settings instance with groq provider, serper search, and default paths.
    """
    cfg_path: Path = tmp_path_factory.mktemp("config") / "config.yaml"
    cfg_path.write_text(_TEST_CONFIG_YAML)
    from src.config.settings import load_settings
    return load_settings(cfg_path)


@pytest.fixture
def in_memory_db(tmp_path) -> str:
    """SQLite database with all tables seeded, isolated per test.

    Returns:
        Path string to the temp SQLite file.
    """
    db_path = str(tmp_path / "test.db")
    AppointmentDB().init_db(db_path)
    PatientDB().init_db(db_path)
    return db_path


@pytest.fixture
def tmp_faiss(tmp_path) -> PatientVectorStore:
    """PatientVectorStore backed by a mocked embeddings model, isolated per test.

    The mock embeddings always return a unit vector of length 1536 so FAISS
    operations work without an OpenAI API key.

    Returns:
        PatientVectorStore instance with a fresh empty index.
    """
    embeddings = MagicMock(spec=Embeddings)
    embeddings.embed_documents.return_value = [[0.1] * 1536]
    embeddings.embed_query.return_value = [0.1] * 1536
    return PatientVectorStore(str(tmp_path / "faiss"), embeddings)

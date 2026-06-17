"""Shared pytest fixtures for the Developer Memory test suite.

Phase 1: minimal fixtures — later phases extend this with ephemeral ChromaDB,
mock git Repo, and fake LLM instances per the mock strategy in spec §10.

Env isolation note:
    src/server.py imports at module level trigger ChromaLibrarianClient() which calls
    chromadb.HttpClient() if CHROMA_HOST is set — this establishes a live connection at
    *collection time*, before any test function or monkeypatch fixture runs.

    To prevent this, we install a session-wide patch on chromadb.HttpClient and
    chromadb.PersistentClient at conftest *module level* (i.e. before pytest collects
    any test file).  The patch returns a MagicMock that satisfies get_or_create_collection
    without making any network call.

    Tests that need a real EphemeralClient (test_chroma_client.py) pass an explicit
    client= argument to ChromaLibrarianClient() and are therefore unaffected by this patch.
"""

from unittest.mock import MagicMock, patch

import chromadb
import psutil
import pytest

# ── Session-wide chromadb connection guard ────────────────────────────────────
# Applied at module-import time so it is in place before test_server.py is collected.
# Both HttpClient and PersistentClient are patched because the active client depends on
# whether CHROMA_HOST is set in the developer's shell — we must guard both paths.

_mock_chroma_client = MagicMock(spec=chromadb.ClientAPI)
_mock_collection = MagicMock()
_mock_collection.count.return_value = 0
_mock_chroma_client.get_or_create_collection.return_value = _mock_collection

_http_patch = patch("chromadb.HttpClient", return_value=_mock_chroma_client)
_persistent_patch = patch("chromadb.PersistentClient", return_value=_mock_chroma_client)

# Start patches immediately — they remain active for the entire session.
# Tests that need real chromadb inject chromadb.EphemeralClient() explicitly.
_http_patch.start()
_persistent_patch.start()

# ── Session-wide psutil memory guard ─────────────────────────────────────────
# src/server.py calls _check_memory() at module level (line ~355), which reads
# psutil.virtual_memory().available.  On a host with < 8 GB free this raises
# RuntimeError and aborts pytest collection before any test function runs.
#
# We patch psutil.virtual_memory at conftest module-import time — before
# test_server.py is collected — so the guard always sees 32 GB available.
# Individual _check_memory() tests in test_server.py override this patch locally
# via unittest.mock.patch to exercise each memory threshold independently.

_mock_vmem = MagicMock(spec=psutil.virtual_memory())
_mock_vmem.available = 32 * 1024**3  # 32 GB — safely above all thresholds

_psutil_patch = patch("psutil.virtual_memory", return_value=_mock_vmem)
_psutil_patch.start()


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_repo_url() -> str:
    """A valid GitHub repo URL used as a baseline across model and request tests."""
    return "https://github.com/user/repo"


@pytest.fixture
def sample_diff_text() -> str:
    """A minimal but valid unified diff for use in diff model tests."""
    return "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1,3 +1,4 @@\n x = 1\n+y = 2\n"

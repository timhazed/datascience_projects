"""Tests for make_skills_aggregator_node (src/agents/skills_aggregator.py).

Uses real ChromaDB EphemeralClient with fake embeddings — no live Ollama calls.
This tests the full chroma.query(where=...) path including _unpack_query_results
deserialization. MagicMock would bypass that path and miss JSON parsing bugs.

Test cases (14 total, §8 test plan):
  - python chunks ranked before markdown
  - identifier_index filters import lines
  - identifier_index capped per file
  - module_groups grouped by two-level key
  - file_manifest deduplicated
  - has_source_code=True when .py chunk has real identifiers
  - has_source_code=False when .py chunk has import-only identifiers
  - has_source_code=False when only .md Logic chunks present
  - Config summaries appear in pattern_summaries
  - notebook_files separated from file_manifest
  - repo_url propagated from chunk metadata
  - overflow guard reduces caps when rendered size exceeds limit
  - empty collection returns None with error
  - ChromaDB error returns None with error
"""

import hashlib
import uuid
from unittest.mock import patch

import chromadb

from src.agents.skills_aggregator import (
    _MAX_IDENTIFIERS_PER_FILE,
    make_skills_aggregator_node,
)
from src.db.chroma_client import ChromaLibrarianClient
from src.models.chunk import SummarizedChunk


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic 4-dim embeddings from SHA-256 — avoids live Ollama calls."""
    result = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        result.append([float(digest[i] - 128) / 128 for i in range(4)])
    return result


def _make_chroma() -> ChromaLibrarianClient:
    """Fresh EphemeralClient with unique collection name — test isolation."""
    return ChromaLibrarianClient(
        client=chromadb.EphemeralClient(),
        embedding_fn=_fake_embed,
        collection_name=f"test_{uuid.uuid4().hex[:12]}",
    )


def _chunk(
    i: int,
    *,
    path: str | None = None,
    tech: list[str] | None = None,
    stype: str = "Logic",
    key_identifiers: list[str] | None = None,
    intent_summary: str | None = None,
) -> SummarizedChunk:
    """Build a SummarizedChunk with controllable fields for seeding ChromaDB."""
    return SummarizedChunk(
        content=f"code_content_{i}_{uuid.uuid4().hex[:8]}",
        path=path or f"src/f{i}.py",
        intent_summary=intent_summary or f"Implements feature {i}.",
        tech_stack=tech or ["FastAPI", "Pydantic"],
        semantic_type=stype,
        key_identifiers=key_identifiers or [],
        author_identity="alice",
    )


_META = {
    "repo_url": "https://github.com/u/r",
    "branch": "main",
    "commit_sha": "abc123",
    "indexed_at": "2025-01-01",
}


def _state() -> dict:
    return {"trace": []}


class TestSkillsAggregatorNode:
    def test_python_chunks_ranked_before_md(self) -> None:
        """Python .py Logic chunks appear before .md chunks in logic_chunks.

        Seeds 2 .md chunks and 2 .py chunks with equal identifier counts.
        After ranking, source_type='python' must appear first.
        """
        chroma = _make_chroma()
        chroma.upsert_chunk(
            _chunk(0, path="README.md", stype="Logic", key_identifiers=["SomeClass"]), _META
        )
        chroma.upsert_chunk(
            _chunk(1, path="PROJECTS.md", stype="Logic", key_identifiers=["AnotherClass"]), _META
        )
        chroma.upsert_chunk(
            _chunk(2, path="src/model.py", stype="Logic", key_identifiers=["MyModel"]), _META
        )
        chroma.upsert_chunk(
            _chunk(3, path="src/trainer.py", stype="Logic", key_identifiers=["Trainer"]), _META
        )

        node = make_skills_aggregator_node(chroma)
        result = node(_state())

        assert result["skills_data"] is not None
        chunks = result["skills_data"].logic_chunks
        assert len(chunks) > 0
        assert chunks[0]["source_type"] == "python"

    def test_identifier_index_filters_import_lines(self) -> None:
        """Import lines in key_identifiers are excluded from identifier_index."""
        chroma = _make_chroma()
        chroma.upsert_chunk(
            _chunk(
                0,
                path="src/model.py",
                stype="Logic",
                key_identifiers=["import os", "MyClass", "my_func"],
            ),
            _META,
        )

        node = make_skills_aggregator_node(chroma)
        result = node(_state())

        data = result["skills_data"]
        assert data is not None
        ids_for_file = data.identifier_index.get("src/model.py", [])
        assert "import os" not in ids_for_file
        assert "MyClass" in ids_for_file or "my_func" in ids_for_file

    def test_identifier_index_capped_per_file(self) -> None:
        """identifier_index is capped at _MAX_IDENTIFIERS_PER_FILE per file."""
        chroma = _make_chroma()
        many_ids = [f"Func{i}" for i in range(20)]
        chroma.upsert_chunk(
            _chunk(0, path="src/big_module.py", stype="Logic", key_identifiers=many_ids),
            _META,
        )

        node = make_skills_aggregator_node(chroma)
        result = node(_state())

        data = result["skills_data"]
        assert data is not None
        ids = data.identifier_index.get("src/big_module.py", [])
        assert len(ids) <= _MAX_IDENTIFIERS_PER_FILE

    def test_module_groups_by_top_dir(self) -> None:
        """module_groups keys use two path components (e.g. 'src/agents')."""
        chroma = _make_chroma()
        chroma.upsert_chunk(
            _chunk(0, path="src/agents/foo.py", stype="Logic", key_identifiers=["FooAgent"]),
            _META,
        )
        chroma.upsert_chunk(
            _chunk(1, path="src/agents/bar.py", stype="Logic", key_identifiers=["BarAgent"]),
            _META,
        )

        node = make_skills_aggregator_node(chroma)
        result = node(_state())

        data = result["skills_data"]
        assert data is not None
        assert "src/agents" in data.module_groups
        paths_in_group = data.module_groups["src/agents"]
        assert "src/agents/foo.py" in paths_in_group
        assert "src/agents/bar.py" in paths_in_group

    def test_module_groups_two_component_path(self) -> None:
        """Files at two-component paths (src/model.py) group under 'src', not 'src/model.py'.

        Verifies that _build_module_groups strips the filename before computing the key,
        so the key is always a directory, never a file path.
        """
        chroma = _make_chroma()
        chroma.upsert_chunk(
            _chunk(0, path="src/model.py", stype="Logic", key_identifiers=["MyModel"]),
            _META,
        )

        node = make_skills_aggregator_node(chroma)
        result = node(_state())

        data = result["skills_data"]
        assert data is not None
        # 'src/model.py' must never appear as a key — that's a file path, not a directory
        assert "src/model.py" not in data.module_groups
        # the parent directory 'src' must be the key
        assert "src" in data.module_groups
        assert "src/model.py" in data.module_groups["src"]

    def test_file_manifest_deduplicated(self) -> None:
        """Multiple chunks from the same file appear only once in file_manifest."""
        chroma = _make_chroma()
        # Upsert 3 chunks from the same file (different content → different content_hash)
        for i in range(3):
            chroma.upsert_chunk(
                _chunk(
                    i,
                    path="src/model.py",
                    stype="Logic",
                    key_identifiers=[f"Class{i}"],
                    intent_summary=f"Chunk {i} from model.py",
                ),
                _META,
            )

        node = make_skills_aggregator_node(chroma)
        result = node(_state())

        data = result["skills_data"]
        assert data is not None
        assert data.file_manifest.count("src/model.py") == 1

    def test_has_source_code_true(self) -> None:
        """has_source_code=True when .py Logic chunk has a real identifier."""
        chroma = _make_chroma()
        chroma.upsert_chunk(
            _chunk(0, path="src/model.py", stype="Logic", key_identifiers=["MyClass"]), _META
        )

        node = make_skills_aggregator_node(chroma)
        result = node(_state())

        assert result["skills_data"] is not None
        assert result["skills_data"].has_source_code is True

    def test_has_source_code_false_import_only(self) -> None:
        """has_source_code=False when .py Logic chunk has only import-line identifiers."""
        chroma = _make_chroma()
        chroma.upsert_chunk(
            _chunk(0, path="src/model.py", stype="Logic", key_identifiers=["import os"]),
            _META,
        )

        node = make_skills_aggregator_node(chroma)
        result = node(_state())

        assert result["skills_data"] is not None
        assert result["skills_data"].has_source_code is False

    def test_has_source_code_false_md_only(self) -> None:
        """has_source_code=False when only .md Logic chunks are indexed."""
        chroma = _make_chroma()
        chroma.upsert_chunk(
            _chunk(0, path="README.md", stype="Logic", key_identifiers=["SomePattern"]),
            _META,
        )

        node = make_skills_aggregator_node(chroma)
        result = node(_state())

        assert result["skills_data"] is not None
        assert result["skills_data"].has_source_code is False

    def test_config_summaries_in_pattern_summaries(self) -> None:
        """Config chunk intent_summary appears in pattern_summaries."""
        chroma = _make_chroma()
        chroma.upsert_chunk(
            _chunk(0, stype="Logic", intent_summary="Implements the trainer loop."), _META
        )
        chroma.upsert_chunk(
            _chunk(1, path="config/settings.py", stype="Config", intent_summary="uses dotenv"),
            _META,
        )

        node = make_skills_aggregator_node(chroma)
        result = node(_state())

        assert result["skills_data"] is not None
        assert "uses dotenv" in result["skills_data"].pattern_summaries

    def test_notebook_files_separated(self) -> None:
        """Jupyter notebook paths appear in notebook_files."""
        chroma = _make_chroma()
        chroma.upsert_chunk(
            _chunk(
                0,
                path="notebooks/analysis.ipynb",
                stype="Logic",
                key_identifiers=["run_analysis"],
            ),
            _META,
        )

        node = make_skills_aggregator_node(chroma)
        result = node(_state())

        assert result["skills_data"] is not None
        assert "notebooks/analysis.ipynb" in result["skills_data"].notebook_files

    def test_repo_url_propagated(self) -> None:
        """repo_url from chunk metadata is propagated to SkillsData."""
        chroma = _make_chroma()
        meta_with_url = {**_META, "repo_url": "https://github.com/x/y"}
        chroma.upsert_chunk(
            _chunk(0, path="src/model.py", stype="Logic", key_identifiers=["MyModel"]),
            meta_with_url,
        )

        node = make_skills_aggregator_node(chroma)
        result = node(_state())

        assert result["skills_data"] is not None
        assert result["skills_data"].repo_url == "https://github.com/x/y"

    def test_overflow_guard_reduces_caps(self) -> None:
        """Overflow guard reduces logic_chunks to ≤10 and identifier_index to ≤50 files
        when the rendered size exceeds _CONTEXT_CHAR_LIMIT.

        Seeds enough distinct files to build a large identifier_index, then patches
        _CONTEXT_CHAR_LIMIT to 0 to force the guard to trigger regardless of actual size.
        """
        chroma = _make_chroma()
        # Seed 25 .py files with real identifiers — builds a non-trivial identifier_index
        for i in range(25):
            chroma.upsert_chunk(
                _chunk(
                    i,
                    path=f"src/module_{i}.py",
                    stype="Logic",
                    key_identifiers=[f"ClassA{i}", f"ClassB{i}", f"func_{i}"],
                ),
                _META,
            )

        # IMPORTANT: node must be created AND invoked inside the patch context.
        # _apply_overflow_guard reads _CONTEXT_CHAR_LIMIT from the module global at
        # call time (not captured by value at factory creation time). Moving either
        # call outside the `with` block would silently stop exercising the guard.
        with patch(
            "src.agents.skills_aggregator._CONTEXT_CHAR_LIMIT",
            0,
        ):
            node = make_skills_aggregator_node(chroma)
            result = node(_state())

        assert result["skills_data"] is not None
        data = result["skills_data"]
        assert len(data.logic_chunks) <= 10
        assert len(data.identifier_index) <= 50

    def test_empty_collection_returns_none(self) -> None:
        """Empty ChromaDB collection → skills_data=None, error set."""
        node = make_skills_aggregator_node(_make_chroma())
        result = node(_state())

        assert result["skills_data"] is None
        assert result.get("error")
        assert len(result["trace"]) >= 1

    def test_chroma_error_handled(self) -> None:
        """ChromaDB RuntimeError → skills_data=None, error set, no crash."""
        chroma = _make_chroma()
        node = make_skills_aggregator_node(chroma)

        # Patch query on the consuming instance — not the class definition
        with patch.object(chroma, "query", side_effect=RuntimeError("ChromaDB unavailable")):
            result = node(_state())

        assert result["skills_data"] is None
        assert result.get("error")
        assert len(result["trace"]) >= 1

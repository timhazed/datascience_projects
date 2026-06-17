"""Tests for make_semantic_searcher_node (src/agents/semantic_searcher.py).

Uses real ChromaDB EphemeralClient with fake embeddings for integration coverage.
No live Ollama calls.

Cases:
  - Happy path: returns raw_results list from ChromaDB
  - Empty collection → raw_results=[]
  - Explicit tech_filter applied → only matching docs returned
  - ChromaDB error → raw_results=[], error set, no crash
  - Trace entry added on success and failure
  - Dynamic tech inference from corpus labels
  - Synonym override inference (pyautogen → autogen)
  - Project filter inferred → results scoped to that project's file_path prefix
  - Tech filter suppressed when project filter is active
  - Corpus TTL refresh picks up newly indexed tech and projects
  - lc/lg abbreviations do not false-positive
  - get_distinct_tech_stack and get_distinct_project_prefixes return correct data
"""

import hashlib
import uuid
from unittest.mock import patch

import chromadb

from src.agents.semantic_searcher import (
    _infer_project_filter,
    _infer_tech_filter,
    make_semantic_searcher_node,
)
from src.db.chroma_client import ChromaLibrarianClient
from src.models.chunk import SummarizedChunk

_META = {
    "repo_url": "https://github.com/u/r",
    "branch": "main",
    "commit_sha": "abc123",
    "indexed_at": "2025-01-01",
}


def _fake_embed(texts: list[str]) -> list[list[float]]:
    result = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        result.append([float(digest[i] - 128) / 128 for i in range(4)])
    return result


def _make_chroma() -> ChromaLibrarianClient:
    return ChromaLibrarianClient(
        client=chromadb.EphemeralClient(),
        embedding_fn=_fake_embed,
        collection_name=f"test_{uuid.uuid4().hex[:12]}",
    )


def _chunk(
    content: str = "x = 1",
    path: str = "src/x.py",
    tech: list[str] | None = None,
) -> SummarizedChunk:
    return SummarizedChunk(
        content=content,
        path=path,
        intent_summary="Does something.",
        tech_stack=tech or ["Python"],
        semantic_type="Logic",
        author_identity="alice",
    )


def _state(query: str = "how are routes structured", tech_filter: list[str] | None = None) -> dict:
    return {"query": query, "tech_filter": tech_filter, "trace": []}


# ---------------------------------------------------------------------------
# Node integration tests
# ---------------------------------------------------------------------------


class TestSemanticSearcherNode:
    def test_returns_results_for_populated_collection(self) -> None:
        """Querying a populated collection returns at least one result."""
        chroma = _make_chroma()
        chroma.upsert_chunk(_chunk(content="app = FastAPI()", path="src/main.py", tech=["FastAPI"]), _META)
        node = make_semantic_searcher_node(chroma)

        result = node(_state("FastAPI route registration"))

        assert "raw_results" in result
        assert len(result["raw_results"]) >= 1

    def test_empty_collection_returns_empty_list(self) -> None:
        """Empty collection → raw_results=[] with no error."""
        node = make_semantic_searcher_node(_make_chroma())
        result = node(_state("anything"))

        assert result["raw_results"] == []
        assert "error" not in result or result.get("error") is None

    def test_explicit_tech_filter_narrows_results(self) -> None:
        """Explicit tech_filter excludes docs not matching the requested tech."""
        chroma = _make_chroma()
        chroma.upsert_chunk(_chunk(content="FastAPI route", path="src/api.py", tech=["FastAPI"]), _META)
        chroma.upsert_chunk(_chunk(content="Flask route", path="src/flask.py", tech=["Flask"]), _META)
        node = make_semantic_searcher_node(chroma)

        result = node(_state("route", tech_filter=["FastAPI"]))

        for r in result["raw_results"]:
            tech = r.get("metadata", {}).get("tech_stack", [])
            assert "FastAPI" in tech, f"Expected FastAPI in tech_stack, got {tech}"

    def test_chroma_error_returns_empty_results_and_error(self) -> None:
        """ChromaDB error → raw_results=[], error set, no crash."""
        chroma = _make_chroma()
        node = make_semantic_searcher_node(chroma)
        with patch.object(chroma, "query", side_effect=RuntimeError("ChromaDB unavailable")):
            result = node(_state("query"))

        assert result["raw_results"] == []
        assert result.get("error")

    def test_trace_added_on_success(self) -> None:
        """Trace gains an entry on successful search."""
        node = make_semantic_searcher_node(_make_chroma())
        result = node(_state("query"))

        assert len(result["trace"]) >= 1

    def test_trace_added_on_error(self) -> None:
        """Trace gains an error entry when ChromaDB fails."""
        chroma = _make_chroma()
        node = make_semantic_searcher_node(chroma)
        with patch.object(chroma, "query", side_effect=RuntimeError("ChromaDB unavailable")):
            result = node(_state("query"))

        assert any("error" in t for t in result["trace"])

    def test_dynamic_tech_inference_from_corpus(self) -> None:
        """Tech label present in corpus and query triggers inferred tech_filter."""
        chroma = _make_chroma()
        chroma.upsert_chunk(_chunk(content="chroma.query()", path="src/db.py", tech=["ChromaDB"]), _META)
        chroma.upsert_chunk(_chunk(content="fastapi app", path="src/app.py", tech=["FastAPI"]), _META)
        node = make_semantic_searcher_node(chroma)

        result = node(_state("ChromaDB collection setup"))

        assert any("inferred tech_filter" in t for t in result["trace"])
        for r in result["raw_results"]:
            tech = r.get("metadata", {}).get("tech_stack", [])
            assert "ChromaDB" in tech

    def test_no_tech_inference_when_label_not_in_corpus(self) -> None:
        """Query mentioning a tech not in the corpus produces no inferred filter."""
        chroma = _make_chroma()
        chroma.upsert_chunk(_chunk(content="some code", path="src/x.py", tech=["Python"]), _META)
        node = make_semantic_searcher_node(chroma)

        result = node(_state("Redis cache setup"))

        assert not any("inferred tech_filter" in t for t in result["trace"])

    def test_project_filter_scopes_results_to_project(self) -> None:
        """Query naming a project returns only chunks from that project's file_path prefix."""
        chroma = _make_chroma()
        chroma.upsert_chunk(
            _chunk(content="SELECT * FROM users", path="projects/MyApp/db.py", tech=["Python"]), _META
        )
        chroma.upsert_chunk(
            _chunk(content="SELECT * FROM orders", path="projects/OtherApp/db.py", tech=["Python"]), _META
        )
        node = make_semantic_searcher_node(chroma)

        result = node(_state("What database queries are in MyApp"))

        assert any("inferred project_filter" in t for t in result["trace"])
        for r in result["raw_results"]:
            fp = r.get("metadata", {}).get("file_path", "")
            assert fp.startswith("projects/MyApp/"), f"Expected MyApp path, got {fp}"

    def test_project_filter_suppresses_tech_filter(self) -> None:
        """When a project filter is inferred, tech filter inference is skipped."""
        chroma = _make_chroma()
        chroma.upsert_chunk(
            _chunk(content="lc chain", path="projects/MyApp/chain.py", tech=["LangChain"]), _META
        )
        node = make_semantic_searcher_node(chroma)

        result = node(_state("LangChain usage in MyApp"))

        assert any("inferred project_filter" in t for t in result["trace"])
        assert not any("inferred tech_filter" in t for t in result["trace"])

    def test_synonym_override_maps_alias_to_corpus_label(self) -> None:
        """'pyautogen' in query maps to 'autogen' corpus label via synonym override."""
        chroma = _make_chroma()
        chroma.upsert_chunk(_chunk(content="autogen agent", path="src/ag.py", tech=["autogen"]), _META)
        node = make_semantic_searcher_node(chroma)

        result = node(_state("pyautogen agent setup"))

        assert any("inferred tech_filter" in t for t in result["trace"])
        for r in result["raw_results"]:
            tech = r.get("metadata", {}).get("tech_stack", [])
            assert "autogen" in tech

    def test_corpus_refresh_picks_up_newly_indexed_tech(self) -> None:
        """Tech label ingested after node construction is matched after TTL expires."""
        import src.agents.semantic_searcher as searcher_mod

        chroma = _make_chroma()
        node = make_semantic_searcher_node(chroma)

        chroma.upsert_chunk(_chunk(content="redis.set()", path="src/cache.py", tech=["Redis"]), _META)

        result_before = node(_state("Redis cache setup"))
        assert not any("inferred tech_filter" in t for t in result_before["trace"])

        with patch.object(searcher_mod, "_CORPUS_TTL_SECONDS", -1):
            node_refreshed = make_semantic_searcher_node(chroma)

        result_after = node_refreshed(_state("Redis cache setup"))
        assert any("inferred tech_filter" in t for t in result_after["trace"])

    def test_corpus_refresh_picks_up_newly_indexed_project(self) -> None:
        """Project ingested after node construction is matched after TTL expires."""
        import src.agents.semantic_searcher as searcher_mod

        chroma = _make_chroma()
        node = make_semantic_searcher_node(chroma)

        chroma.upsert_chunk(
            _chunk(content="new project code", path="projects/NewProject/main.py", tech=["Python"]), _META
        )

        result_before = node(_state("code in NewProject"))
        assert not any("inferred project_filter" in t for t in result_before["trace"])

        with patch.object(searcher_mod, "_CORPUS_TTL_SECONDS", -1):
            node_refreshed = make_semantic_searcher_node(chroma)

        result_after = node_refreshed(_state("code in NewProject"))
        assert any("inferred project_filter" in t for t in result_after["trace"])

    def test_lc_abbreviation_does_not_trigger_langchain_filter(self) -> None:
        """Short abbreviation 'lc' must not false-positive as LangChain."""
        chroma = _make_chroma()
        chroma.upsert_chunk(_chunk(content="chain = lc.Chain()", path="src/x.py", tech=["LangChain"]), _META)
        node = make_semantic_searcher_node(chroma)

        result = node(_state("what does lc do in this circuit"))

        assert not any("inferred tech_filter" in t for t in result["trace"])

    def test_lg_abbreviation_does_not_trigger_langgraph_filter(self) -> None:
        """Short abbreviation 'lg' must not false-positive as LangGraph."""
        chroma = _make_chroma()
        chroma.upsert_chunk(_chunk(content="g = lg.Graph()", path="src/x.py", tech=["LangGraph"]), _META)
        node = make_semantic_searcher_node(chroma)

        result = node(_state("lg monitor setup"))

        assert not any("inferred tech_filter" in t for t in result["trace"])


# ---------------------------------------------------------------------------
# ChromaLibrarianClient.get_distinct_tech_stack tests
# ---------------------------------------------------------------------------


class TestGetDistinctTechStack:
    def test_returns_empty_for_empty_collection(self) -> None:
        """Empty collection → empty list."""
        chroma = _make_chroma()
        assert chroma.get_distinct_tech_stack() == []

    def test_returns_all_distinct_labels(self) -> None:
        """All distinct tech labels across documents are returned."""
        chroma = _make_chroma()
        chroma.upsert_chunk(_chunk(content="a", path="src/a.py", tech=["FastAPI", "Pydantic"]), _META)
        chroma.upsert_chunk(_chunk(content="b", path="src/b.py", tech=["LangChain"]), _META)
        chroma.upsert_chunk(_chunk(content="c", path="src/c.py", tech=["FastAPI"]), _META)

        labels = chroma.get_distinct_tech_stack()

        assert sorted(labels) == sorted(["FastAPI", "Pydantic", "LangChain"])

    def test_returns_sorted_list(self) -> None:
        """Result is sorted alphabetically."""
        chroma = _make_chroma()
        chroma.upsert_chunk(_chunk(content="z", path="src/z.py", tech=["Zebra", "Alpha"]), _META)

        labels = chroma.get_distinct_tech_stack()

        assert labels == sorted(labels)

    def test_chroma_error_returns_empty_list(self) -> None:
        """ChromaDB error on get() → empty list, no crash."""
        chroma = _make_chroma()
        chroma.upsert_chunk(_chunk(content="x", path="src/x.py", tech=["Python"]), _META)
        with patch.object(chroma._collection, "get", side_effect=RuntimeError("db error")):
            result = chroma.get_distinct_tech_stack()

        assert result == []


# ---------------------------------------------------------------------------
# ChromaLibrarianClient.get_distinct_project_prefixes tests
# ---------------------------------------------------------------------------


class TestGetDistinctProjectPrefixes:
    def test_returns_empty_for_empty_collection(self) -> None:
        """Empty collection → empty dict."""
        chroma = _make_chroma()
        assert chroma.get_distinct_project_prefixes() == {}

    def test_extracts_project_names_from_file_paths(self) -> None:
        """Project names are extracted from projects/<name>/... paths."""
        chroma = _make_chroma()
        chroma.upsert_chunk(_chunk(content="a", path="projects/Alpha/src/a.py"), _META)
        chroma.upsert_chunk(_chunk(content="b", path="projects/Beta/src/b.py"), _META)
        chroma.upsert_chunk(_chunk(content="c", path="projects/Alpha/src/c.py"), _META)

        prefixes = chroma.get_distinct_project_prefixes()

        assert "Alpha" in prefixes
        assert "Beta" in prefixes
        assert prefixes["Alpha"] == "projects/Alpha/"
        assert prefixes["Beta"] == "projects/Beta/"

    def test_ignores_paths_without_project_structure(self) -> None:
        """Flat paths (no subdirectory) are not extracted as project names."""
        chroma = _make_chroma()
        chroma.upsert_chunk(_chunk(content="x", path="README.md"), _META)

        prefixes = chroma.get_distinct_project_prefixes()

        assert prefixes == {}

    def test_works_with_non_projects_root(self) -> None:
        """Paths not starting with 'projects/' still yield a project name entry."""
        chroma = _make_chroma()
        chroma.upsert_chunk(_chunk(content="x", path="repos/MyRepo/src/main.py"), _META)

        prefixes = chroma.get_distinct_project_prefixes()

        assert "MyRepo" in prefixes
        assert prefixes["MyRepo"] == "repos/MyRepo/"

    def test_chroma_error_returns_empty_dict(self) -> None:
        """ChromaDB error → empty dict, no crash."""
        chroma = _make_chroma()
        chroma.upsert_chunk(_chunk(content="x", path="projects/X/f.py"), _META)
        with patch.object(chroma._collection, "get", side_effect=RuntimeError("db error")):
            result = chroma.get_distinct_project_prefixes()

        assert result == {}


# ---------------------------------------------------------------------------
# _infer_project_filter unit tests (pure function)
# ---------------------------------------------------------------------------


class TestInferProjectFilter:
    def test_returns_none_for_empty_corpus(self) -> None:
        """No corpus projects → no inference possible."""
        assert _infer_project_filter("MyApp database", {}) is None

    def test_exact_name_match(self) -> None:
        """Query containing the exact project name returns that project."""
        result = _infer_project_filter(
            "What database is used in Chatbot_DataPipeline",
            {"Chatbot_DataPipeline": "projects/Chatbot_DataPipeline/", "Chatbot": "projects/Chatbot/"},
        )
        assert result is not None
        name, prefix = result
        assert name == "Chatbot_DataPipeline"
        assert prefix == "projects/Chatbot_DataPipeline/"

    def test_longer_name_matches_before_shorter_substring(self) -> None:
        """Chatbot_DataPipeline must match before Chatbot when both are in corpus."""
        result = _infer_project_filter(
            "show me Chatbot_DataPipeline routes",
            {"Chatbot": "projects/Chatbot/", "Chatbot_DataPipeline": "projects/Chatbot_DataPipeline/"},
        )
        assert result is not None
        assert result[0] == "Chatbot_DataPipeline"

    def test_case_insensitive_match(self) -> None:
        """Match is case-insensitive."""
        result = _infer_project_filter(
            "code in geoportugal",
            {"Geoportugal": "projects/Geoportugal/"},
        )
        assert result is not None
        assert result[0] == "Geoportugal"

    def test_no_project_in_query_returns_none(self) -> None:
        """Query with no project name returns None."""
        result = _infer_project_filter(
            "how do I set up FastAPI routes",
            {"Geoportugal": "projects/Geoportugal/", "MyApp": "projects/MyApp/"},
        )
        assert result is None


# ---------------------------------------------------------------------------
# _infer_tech_filter unit tests (pure function)
# ---------------------------------------------------------------------------


class TestInferTechFilter:
    def test_returns_none_for_empty_corpus(self) -> None:
        """No corpus labels → no inference possible."""
        assert _infer_tech_filter("LangChain route", []) is None

    def test_direct_label_match(self) -> None:
        """Query containing a corpus label returns that label."""
        result = _infer_tech_filter("LangChain pipeline setup", ["LangChain", "FastAPI"])
        assert result == ["LangChain"]

    def test_case_insensitive_match(self) -> None:
        """Match is case-insensitive against the corpus label."""
        result = _infer_tech_filter("langchain pipeline", ["LangChain"])
        assert result == ["LangChain"]

    def test_synonym_override_used_when_label_in_corpus(self) -> None:
        """pyautogen alias maps to autogen when autogen is in the corpus."""
        result = _infer_tech_filter("pyautogen agent example", ["autogen", "LangChain"])
        assert result == ["autogen"]

    def test_synonym_override_skipped_when_label_not_in_corpus(self) -> None:
        """pyautogen alias returns None when autogen is not in the corpus."""
        result = _infer_tech_filter("pyautogen agent example", ["LangChain"])
        assert result is None

    def test_no_match_returns_none(self) -> None:
        """No matching label → None."""
        result = _infer_tech_filter("redis cache setup", ["LangChain", "FastAPI"])
        assert result is None

    def test_first_match_only_returned(self) -> None:
        """When multiple corpus labels match, only the longest match is returned.

        Labels are sorted longest-first so more specific names win over shorter ones.
        LangChain (9 chars) is longer than FastAPI (7 chars), so LangChain wins.
        """
        result = _infer_tech_filter("fastapi langchain", ["FastAPI", "LangChain"])
        assert result == ["LangChain"]


# ---------------------------------------------------------------------------
# Tech-filter reranking: impl files before overview docs
# ---------------------------------------------------------------------------


class TestTechFilterReranking:
    def test_impl_files_ranked_before_overview_docs(self) -> None:
        """When tech_filter active and no project scope, projects/ files precede root docs."""
        chroma = _make_chroma()
        # Root-level overview doc mentioning Qdrant in summary context
        chroma.upsert_chunk(
            _chunk(content="Qdrant vector database overview and comparison", path="PROJECTS.md", tech=["Qdrant"]),
            _META,
        )
        # Actual implementation file
        chroma.upsert_chunk(
            _chunk(content="client = QdrantClient(url=...)", path="projects/Chatbot_DataPipeline/insert_db/qdrant.py", tech=["Qdrant"]),
            _META,
        )
        node = make_semantic_searcher_node(chroma)

        result = node(_state("Where is Qdrant vector database used?"))

        raw = result["raw_results"]
        assert len(raw) >= 1
        # Implementation file must appear before any root-level doc
        paths = [r.get("metadata", {}).get("file_path", "") for r in raw]
        impl_indices = [i for i, p in enumerate(paths) if p.startswith("projects/")]
        overview_indices = [i for i, p in enumerate(paths) if not p.startswith("projects/")]
        # All implementation files must appear before all overview files
        if impl_indices and overview_indices:
            assert max(impl_indices) < min(overview_indices), (
                f"Expected all impl files before overview docs, got paths={paths}"
            )

    def test_reranking_not_applied_when_project_filter_active(self) -> None:
        """Project filter active → reranking is skipped (project filter already scopes results)."""
        chroma = _make_chroma()
        chroma.upsert_chunk(
            _chunk(content="Qdrant client usage", path="projects/MyApp/qdrant.py", tech=["Qdrant"]),
            _META,
        )
        chroma.upsert_chunk(
            _chunk(content="overview doc", path="projects/MyApp/OVERVIEW.md", tech=["Qdrant"]),
            _META,
        )
        node = make_semantic_searcher_node(chroma)

        # Query names a project → project filter active, reranking skipped
        result = node(_state("Qdrant usage in MyApp"))

        assert any("inferred project_filter" in t for t in result["trace"])
        # Results should only be from MyApp (project filter applied)
        for r in result["raw_results"]:
            assert r["metadata"]["file_path"].startswith("projects/MyApp/")

    def test_overview_docs_still_included_as_fallback(self) -> None:
        """When no impl files match, overview docs are still returned (not discarded)."""
        chroma = _make_chroma()
        chroma.upsert_chunk(
            _chunk(content="Qdrant is used across projects for vector similarity", path="README.md", tech=["Qdrant"]),
            _META,
        )
        node = make_semantic_searcher_node(chroma)

        result = node(_state("Where is Qdrant vector database used?"))

        # Should still get the README result — it's not dropped, just deprioritized
        assert len(result["raw_results"]) >= 1
        assert result["raw_results"][0]["metadata"]["file_path"] == "README.md"

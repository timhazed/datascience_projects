"""Integration tests for build_diff_graph (src/graphs/diff_graph.py).

Uses real ChromaDB EphemeralClient + RunnableLambda LLM mock — no live Ollama calls.

Cases:
  - Invalid diff (empty) → routes to END, no LLM call
  - Valid diff, no persona data → coaching_analyzer uses baseline, produces result
  - Valid diff, persona loaded → full pipeline, deviation_guard runs
  - High deviation score → coaching_alert populated
  - Low deviation score → no coaching_alert
  - Graph compiles without error
"""

import hashlib
import uuid
from unittest.mock import MagicMock, patch

import chromadb
from langchain_core.runnables import RunnableLambda

from src.db.chroma_client import ChromaLibrarianClient
from src.graphs.diff_graph import build_diff_graph
from src.models.analysis import AnalysisResult


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


_META = {
    "repo_url": "https://github.com/u/r",
    "branch": "main",
    "commit_sha": "abc",
    "indexed_at": "2025-01-01",
}

_VALID_DIFF = (
    "--- a/src/auth.py\n"
    "+++ b/src/auth.py\n"
    "@@ -1,3 +1,4 @@\n"
    " def login():\n"
    "+    print('debug')\n"
    "     return True\n"
)


def _make_low_deviation_llm() -> MagicMock:
    result = AnalysisResult(
        deviation_score=0.2,
        rationale="Well aligned with established patterns.",
        patterns_matched=["typed returns", "dependency injection"],
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = RunnableLambda(lambda _: result)
    return llm


def _make_high_deviation_llm() -> MagicMock:
    result = AnalysisResult(
        deviation_score=0.9,
        rationale="Bare print statements contradict established logging patterns.",
        patterns_matched=[],
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = RunnableLambda(lambda _: result)
    return llm


def _state(diff: str = _VALID_DIFF) -> dict:
    return {"diff_text": diff, "trace": []}


class TestDiffGraphCompilation:
    def test_builds_without_error(self) -> None:
        """build_diff_graph returns a compiled graph without raising."""
        chroma = _make_chroma()
        llm = _make_low_deviation_llm()
        graph = build_diff_graph(llm=llm, chroma=chroma)
        assert graph is not None


class TestDiffGraphValidatorPath:
    def test_empty_diff_routes_to_end(self) -> None:
        """Empty diff string → diff_safe=False, pipeline ends, no LLM call."""
        chroma = _make_chroma()
        llm = _make_low_deviation_llm()
        graph = build_diff_graph(llm=llm, chroma=chroma)

        result = graph.invoke(_state(diff=""))

        assert result.get("diff_safe") is False
        assert result.get("analysis_result") is None
        assert result.get("error")

    def test_invalid_format_rejected(self) -> None:
        """Diff without --- / +++ headers → diff_safe=False."""
        chroma = _make_chroma()
        llm = _make_low_deviation_llm()
        graph = build_diff_graph(llm=llm, chroma=chroma)

        result = graph.invoke(_state(diff="just some text without headers"))

        assert result.get("diff_safe") is False

    def test_oversized_diff_rejected(self) -> None:
        """Diff exceeding 100 KB → diff_safe=False."""
        chroma = _make_chroma()
        llm = _make_low_deviation_llm()
        graph = build_diff_graph(llm=llm, chroma=chroma)

        big_diff = "--- a/x\n+++ b/x\n" + "x" * 100_001
        result = graph.invoke(_state(diff=big_diff))

        assert result.get("diff_safe") is False


class TestDiffGraphHappyPath:
    def test_valid_diff_no_persona_produces_result(self) -> None:
        """Valid diff, empty ChromaDB → coaching_analyzer uses baseline, analysis_result set."""
        chroma = _make_chroma()  # no seeded data → persona_loader skips
        llm = _make_low_deviation_llm()
        graph = build_diff_graph(llm=llm, chroma=chroma)

        result = graph.invoke(_state())

        assert result.get("diff_safe") is True
        assert result.get("analysis_result") is not None
        assert result["analysis_result"].deviation_score == 0.2

    def test_low_deviation_no_coaching_alert(self) -> None:
        """Deviation score 0.2 → deviation_guard does not populate coaching_alert."""
        chroma = _make_chroma()
        llm = _make_low_deviation_llm()
        graph = build_diff_graph(llm=llm, chroma=chroma)

        result = graph.invoke(_state())

        assert result.get("coaching_alert") is None

    def test_high_deviation_populates_coaching_alert(self) -> None:
        """Deviation score 0.9 → deviation_guard populates coaching_alert."""
        chroma = _make_chroma()
        llm = _make_high_deviation_llm()
        graph = build_diff_graph(llm=llm, chroma=chroma)

        result = graph.invoke(_state())

        assert result.get("coaching_alert") is not None
        assert result["coaching_alert"].severity in ("warning", "critical")

    def test_trace_records_all_nodes(self) -> None:
        """Trace has entries from all 4 nodes in the happy path."""
        chroma = _make_chroma()
        llm = _make_low_deviation_llm()
        graph = build_diff_graph(llm=llm, chroma=chroma)

        result = graph.invoke(_state())

        trace_str = " ".join(result["trace"])
        assert "diff_validator" in trace_str
        assert "persona_loader" in trace_str
        assert "coaching_analyzer" in trace_str
        assert "deviation_guard" in trace_str


class TestDiffGraphErrorPath:
    def test_chroma_error_persona_loader_continues(self) -> None:
        """ChromaDB error in persona_loader → persona_context=None, pipeline continues."""
        chroma = _make_chroma()
        llm = _make_low_deviation_llm()
        graph = build_diff_graph(llm=llm, chroma=chroma)

        with patch.object(chroma, "query", side_effect=RuntimeError("ChromaDB down")):
            result = graph.invoke(_state())

        # Pipeline should complete with a result despite the persona_loader error
        assert result.get("diff_safe") is True
        assert result.get("analysis_result") is not None
        assert result.get("persona_context") is None

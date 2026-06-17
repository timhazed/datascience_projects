"""Tests for make_chunk_dispatcher_node (src/agents/chunk_dispatcher.py).

Phase 4 exit gate — no LangGraph graph needed; node tested in isolation.

LangGraph 1.x contract: nodes must return dict.  Send fan-out is performed by the
conditional edge routing function route_after_dispatch() in build_sync_graph().
These tests verify that:
  - The node always returns a dict (never list[Send])
  - The trace entry is correct for both the chunks-present and no-chunks paths
  - The route_after_dispatch routing function (tested in test_sync_graph.py) handles Send
"""

from src.agents.chunk_dispatcher import make_chunk_dispatcher_node
from src.models.chunk import ParsedChunk
from src.models.trace_event import TraceEventType


def _make_chunks(n: int) -> list[ParsedChunk]:
    return [ParsedChunk(path=f"src/file_{i}.py", content=f"content {i}") for i in range(n)]


def _state(chunks: list[ParsedChunk]) -> dict:
    return {
        "parsed_chunks": chunks,
        "sanitized_files": [],
        "quarantined_files": [],
        "summarized_chunks": [],
        "upsert_results": [],
        "error": None,
        "trace": [],
    }


class TestChunkDispatcherNode:
    def test_five_chunks_returns_dict_with_trace(self) -> None:
        """5 ParsedChunks → node returns dict (Send fan-out is the edge's responsibility)."""
        node = make_chunk_dispatcher_node()
        result = node(_state(_make_chunks(5)))

        assert isinstance(result, dict)
        assert "trace" in result

    def test_five_chunks_trace_mentions_count(self) -> None:
        """5 chunks → trace entry contains the chunk count."""
        node = make_chunk_dispatcher_node()
        result = node(_state(_make_chunks(5)))

        assert any("5" in entry for entry in result["trace"])

    def test_zero_chunks_returns_dict_not_send_list(self) -> None:
        """Empty parsed_chunks → returns dict; no crash; no Send objects."""
        node = make_chunk_dispatcher_node()
        result = node(_state([]))

        assert isinstance(result, dict)
        assert "trace" in result

    def test_zero_chunks_trace_message_present(self) -> None:
        """Empty parsed_chunks → trace entry describes the no-chunks routing decision."""
        node = make_chunk_dispatcher_node()
        result = node(_state([]))

        assert isinstance(result, dict)
        assert any("no chunks" in entry for entry in result["trace"])

    def test_single_chunk_returns_dict(self) -> None:
        """1 ParsedChunk → node returns dict (not list)."""
        node = make_chunk_dispatcher_node()
        result = node(_state(_make_chunks(1)))

        assert isinstance(result, dict)


class TestChunkDispatcherTraceEvents:
    """Assert trace_events are emitted on every code path."""

    def test_no_chunks_emits_skip_event(self) -> None:
        """Empty parsed_chunks → TraceEventType.SKIP with count=0."""
        node = make_chunk_dispatcher_node()
        result = node(_state([]))
        events = result["trace_events"]
        assert len(events) == 1
        assert events[0].node == "chunk_dispatcher"
        assert events[0].event == TraceEventType.SKIP
        assert events[0].count == 0

    def test_chunks_present_emits_dispatched_event(self) -> None:
        """Non-empty parsed_chunks → TraceEventType.DISPATCHED with count=N."""
        node = make_chunk_dispatcher_node()
        result = node(_state(_make_chunks(3)))
        events = result["trace_events"]
        assert len(events) == 1
        assert events[0].event == TraceEventType.DISPATCHED
        assert events[0].count == 3

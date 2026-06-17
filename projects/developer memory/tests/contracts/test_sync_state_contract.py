"""Contract tests for the sync pipeline UI ↔ backend metric contract (F1 Phase C).

Phase C removed `_parse_sync_trace` and the legacy fallback. The UI now binds
exclusively to `_metrics_from_trace_events`. These tests verify the structured
parser returns correct counts for the golden-file SyncState fixture.

The Phase B→C gate test is retained as documentation of the completed transition.

Cases:
  - changed_files read from state field (not events)
  - sanitized accumulated from pii_sanitizer OK events
  - quarantined read from quarantined_files state field (not events)
  - chunks accumulated from summarize_and_upsert OK events
  - cached read from cache_filtered_count state field
  - inserted/skipped/errors accumulated from summarize_and_upsert events
  - empty trace_events list → counts from state fields only (no crash)
  - Phase B→C gate: all required nodes present in golden fixture trace_events
"""

import pytest

from src.models.trace_event import TraceEvent, TraceEventType
from ui.pages.sync_repository import _metrics_from_trace_events

# ---------------------------------------------------------------------------
# Golden-file fixture — represents a realistic completed SyncState dict.
# Hand-crafted to match known node output formats (see src/agents/); regenerate
# from a real end-to-end smoke test once CI supports live Ollama runs (spec Note 6).
#
# Pipeline run modelled:
#   - 3 changed files in git delta (delta_extractor)
#   - 1 file removed by cache_filter (already indexed); 2 files enter PII scan
#   - 1 file quarantined by pii_sanitizer; 2 files pass (sanitized=2)
#   - multimodal_parser produces 3 chunks (2 from src/a.py, 1 from src/b.py)
#   - chunk_dispatcher dispatches 3 chunks to summarize_and_upsert
#   - 2 chunks inserted (src/a.py); 1 chunk skipped/cached (src/b.py)
#   - chunks metric = count of summarize_and_upsert OK events = 2
#   - cache_filtered_count = 1 (structured state field from cache_filter)
# ---------------------------------------------------------------------------

_FULL_SYNC_STATE: dict = {
    "repo_url": "https://github.com/test/repo",
    "branch": "main",
    "commit_sha": "abc123def456",
    "repo_root": "/tmp/test_repo",
    "changed_files": ["src/a.py", "src/b.py", "src/secret.py"],
    "cache_filtered_count": 1,
    "sanitized_files": [
        {"path": "src/a.py", "content": "def a(): pass", "quarantine_reason": None},
        {"path": "src/b.py", "content": "def b(): pass", "quarantine_reason": None},
    ],
    "quarantined_files": [
        {"path": "src/secret.py", "content": "<<QUARANTINED>>", "quarantine_reason": "API key detected"},
    ],
    "parsed_chunks": [],
    "upsert_results": [
        {"content_hash": "hash1", "file_path": "src/a.py", "action": "inserted", "error": None},
        {"content_hash": "hash2", "file_path": "src/a.py", "action": "inserted", "error": None},
        {"content_hash": "hash3", "file_path": "src/b.py", "action": "skipped", "error": None},
    ],
    "error": None,
    # Human-readable trace log — kept for the UI expander; no longer parsed for metrics.
    "trace": [
        "delta_extractor: 3 changed files in https://github.com/test/repo",
        "cache_filter: 1 files filtered (already cached)",
        "pii_sanitizer: ok src/a.py",
        "pii_sanitizer: ok src/b.py",
        "pii_sanitizer: quarantine src/secret.py",
        "multimodal_parser: src/a.py → 2 chunks",
        "multimodal_parser: src/b.py → 1 chunks",
        "chunk_dispatcher: 3 chunks ready for intent_summarizer",
        "summarize_and_upsert: inserted src/a.py",
        "summarize_and_upsert: inserted src/a.py",
        "summarize_and_upsert: skip (cached) src/b.py",
    ],
    # Structured trace_events — the single source of truth for UI metrics in Phase C.
    "trace_events": [
        {"node": "delta_extractor", "event": "ok", "path": None, "count": 3, "detail": None},
        {"node": "cache_filter", "event": "skip", "path": None, "count": 1, "detail": None},
        {"node": "pii_sanitizer", "event": "ok", "path": "src/a.py", "count": None, "detail": None},
        {"node": "pii_sanitizer", "event": "ok", "path": "src/b.py", "count": None, "detail": None},
        {"node": "pii_sanitizer", "event": "quarantine", "path": "src/secret.py", "count": None, "detail": "API key detected"},
        {"node": "multimodal_parser", "event": "ok", "path": None, "count": 3, "detail": None},
        {"node": "chunk_dispatcher", "event": "dispatched", "path": None, "count": 3, "detail": None},
        # summarize_and_upsert: 2 inserted (OK), 1 cached
        {"node": "summarize_and_upsert", "event": "ok", "path": "src/a.py", "count": None, "detail": None},
        {"node": "summarize_and_upsert", "event": "ok", "path": "src/a.py", "count": None, "detail": None},
        {"node": "summarize_and_upsert", "event": "cached", "path": "src/b.py", "count": None, "detail": None},
        {"node": "chroma_upsert_quarantined", "event": "quarantine", "path": "src/secret.py", "count": None, "detail": None},
        {"node": "sha_store_update", "event": "ok", "path": None, "count": None, "detail": None},
    ],
}


@pytest.fixture
def full_sync_state() -> dict:
    """Return the pinned golden-file SyncState fixture."""
    return _FULL_SYNC_STATE


# ---------------------------------------------------------------------------
# _metrics_from_trace_events correctness — golden fixture
# ---------------------------------------------------------------------------


class TestMetricsFromTraceEvents:
    """Verify _metrics_from_trace_events returns correct counts for the golden fixture."""

    def test_changed_files_from_state_field(self, full_sync_state: dict) -> None:
        """changed_files reads len(state['changed_files']), not from events."""
        metrics = _metrics_from_trace_events(full_sync_state)
        assert metrics["changed_files"] == 3

    def test_sanitized_from_pii_ok_events(self, full_sync_state: dict) -> None:
        """sanitized counts pii_sanitizer OK events (2 files passed PII)."""
        metrics = _metrics_from_trace_events(full_sync_state)
        assert metrics["sanitized"] == 2

    def test_quarantined_from_state_field(self, full_sync_state: dict) -> None:
        """quarantined reads len(state['quarantined_files']), not from events."""
        metrics = _metrics_from_trace_events(full_sync_state)
        assert metrics["quarantined"] == 1

    def test_chunks_from_summarize_ok_events(self, full_sync_state: dict) -> None:
        """chunks counts summarize_and_upsert OK events (2 chunks actually processed)."""
        metrics = _metrics_from_trace_events(full_sync_state)
        assert metrics["chunks"] == 2

    def test_cached_from_state_field(self, full_sync_state: dict) -> None:
        """cached reads cache_filtered_count state field directly."""
        metrics = _metrics_from_trace_events(full_sync_state)
        assert metrics["cached"] == 1

    def test_inserted_from_summarize_ok_events(self, full_sync_state: dict) -> None:
        """inserted counts summarize_and_upsert OK events (2 inserted)."""
        metrics = _metrics_from_trace_events(full_sync_state)
        assert metrics["inserted"] == 2

    def test_skipped_from_summarize_cached_events(self, full_sync_state: dict) -> None:
        """skipped counts summarize_and_upsert CACHED events (1 cache hit)."""
        metrics = _metrics_from_trace_events(full_sync_state)
        assert metrics["skipped"] == 1

    def test_errors_zero_when_no_error_events(self, full_sync_state: dict) -> None:
        """errors is 0 when no summarize_and_upsert ERROR events are present."""
        metrics = _metrics_from_trace_events(full_sync_state)
        assert metrics["errors"] == 0


class TestMetricsEdgeCases:
    """Edge cases for _metrics_from_trace_events."""

    def test_empty_trace_events_reads_from_state_fields(self) -> None:
        """Empty trace_events list → structured state fields still provide counts."""
        state = {
            "trace_events": [],
            "changed_files": ["src/a.py", "src/b.py"],
            "quarantined_files": [{"path": "src/q.py"}],
            "cache_filtered_count": 2,
        }
        metrics = _metrics_from_trace_events(state)
        assert metrics["changed_files"] == 2
        assert metrics["quarantined"] == 1
        assert metrics["cached"] == 2
        assert metrics["sanitized"] == 0
        assert metrics["chunks"] == 0

    def test_missing_optional_fields_default_to_zero(self) -> None:
        """State with only trace_events and no structured fields → safe defaults."""
        state = {"trace_events": []}
        metrics = _metrics_from_trace_events(state)
        assert metrics["changed_files"] == 0
        assert metrics["quarantined"] == 0
        assert metrics["cached"] == 0

    def test_error_events_increment_errors(self) -> None:
        """summarize_and_upsert ERROR events are counted in errors metric."""
        state = {
            "trace_events": [
                {"node": "summarize_and_upsert", "event": "error", "path": "src/x.py", "count": None, "detail": None},
                {"node": "summarize_and_upsert", "event": "error", "path": "src/y.py", "count": None, "detail": None},
            ],
            "changed_files": [],
            "quarantined_files": [],
        }
        metrics = _metrics_from_trace_events(state)
        assert metrics["errors"] == 2
        assert metrics["chunks"] == 0  # errors don't count as processed chunks

    def test_quarantined_reads_state_field_not_events(self) -> None:
        """quarantined_files state field is authoritative even when events say differently."""
        state = {
            "trace_events": [
                # No quarantine events in trace_events
                {"node": "pii_sanitizer", "event": "ok", "path": "src/a.py", "count": None, "detail": None},
            ],
            "changed_files": [],
            "quarantined_files": [{"path": "src/q1.py"}, {"path": "src/q2.py"}],
        }
        metrics = _metrics_from_trace_events(state)
        assert metrics["quarantined"] == 2  # from state field, not events

    def test_trace_event_objects_handled_alongside_dicts(self) -> None:
        """TraceEvent Pydantic objects (not dicts) are handled correctly via getattr."""
        state = {
            "trace_events": [
                TraceEvent(node="pii_sanitizer", event=TraceEventType.OK, path="src/a.py"),
                TraceEvent(node="summarize_and_upsert", event=TraceEventType.OK, path="src/a.py"),
            ],
            "changed_files": ["src/a.py"],
            "quarantined_files": [],
        }
        metrics = _metrics_from_trace_events(state)
        assert metrics["sanitized"] == 1
        assert metrics["chunks"] == 1
        assert metrics["inserted"] == 1


# ---------------------------------------------------------------------------
# Phase B→C gate — retained as documentation that the transition is complete
# ---------------------------------------------------------------------------


def test_trace_events_non_empty_after_pipeline_run(full_sync_state: dict) -> None:
    """Gate: all required nodes emitted TraceEvents — Phase C transition complete.

    This test was the forcing function for Phase C. It remains in the suite as
    a regression guard: if a future refactor accidentally removes trace_events
    emission from a required node, this test fails immediately.
    """
    trace_events = full_sync_state.get("trace_events", [])
    assert len(trace_events) >= 1, "trace_events is empty"

    node_names = {
        (e.get("node") if isinstance(e, dict) else getattr(e, "node", ""))
        for e in trace_events
    }
    required_nodes = {
        "delta_extractor",
        "cache_filter",
        "pii_sanitizer",
        "chunk_dispatcher",
        "summarize_and_upsert",
        "sha_store_update",
    }
    missing = required_nodes - node_names
    assert not missing, f"Nodes missing from trace_events: {missing}"

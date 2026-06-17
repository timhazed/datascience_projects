"""Tests for TraceEvent and TraceEventType (src/models/trace_event.py) — F1 Phase A.

Covers:
  - All six enum values are valid and serialise to their string forms
  - Unknown event string raises ValueError at TraceEvent construction time
  - TraceEvent fields: required node/event; optional path/count/detail
  - Round-trip: model_dump() → model_validate() preserves all fields
"""

import pytest

from src.models.trace_event import TraceEvent, TraceEventType


class TestTraceEventType:
    def test_all_values_are_strings(self) -> None:
        """Every enum member serialises to a plain string (StrEnum mixin)."""
        assert TraceEventType.OK == "ok"
        assert TraceEventType.QUARANTINE == "quarantine"
        assert TraceEventType.SKIP == "skip"
        assert TraceEventType.CACHED == "cached"
        assert TraceEventType.DISPATCHED == "dispatched"
        assert TraceEventType.ERROR == "error"

    def test_unknown_string_raises_value_error(self) -> None:
        """An unrecognised event string raises ValueError — not silently accepted."""
        with pytest.raises(ValueError):
            TraceEvent(node="delta_extractor", event="unknown_event_type")

    def test_valid_string_accepted(self) -> None:
        """Valid string value is accepted and coerced to enum."""
        te = TraceEvent(node="delta_extractor", event="ok")
        assert te.event is TraceEventType.OK


class TestTraceEventModel:
    def test_required_fields_only(self) -> None:
        """TraceEvent can be constructed with only node and event."""
        te = TraceEvent(node="cache_filter", event=TraceEventType.CACHED)
        assert te.node == "cache_filter"
        assert te.event == TraceEventType.CACHED
        assert te.path is None
        assert te.count is None
        assert te.detail is None

    def test_all_fields_set(self) -> None:
        """All optional fields can be populated."""
        te = TraceEvent(
            node="pii_sanitizer",
            event=TraceEventType.QUARANTINE,
            path="src/secrets.py",
            count=1,
            detail="email detected",
        )
        assert te.path == "src/secrets.py"
        assert te.count == 1
        assert te.detail == "email detected"

    def test_round_trip_serialization(self) -> None:
        """model_dump() → model_validate() round-trips all fields without loss."""
        original = TraceEvent(
            node="summarize_and_upsert",
            event=TraceEventType.OK,
            path="src/main.py",
            count=3,
            detail="inserted",
        )
        data = original.model_dump()
        restored = TraceEvent.model_validate(data)
        assert restored == original

    def test_event_serialises_as_string_in_dump(self) -> None:
        """model_dump() emits event as a plain string, not the enum repr."""
        te = TraceEvent(node="chunk_dispatcher", event=TraceEventType.DISPATCHED, count=5)
        dumped = te.model_dump()
        assert dumped["event"] == "dispatched"
        assert isinstance(dumped["event"], str)

    def test_count_zero_is_valid(self) -> None:
        """count=0 is a valid payload — used by chunk_dispatcher on empty-chunks path."""
        te = TraceEvent(node="chunk_dispatcher", event=TraceEventType.SKIP, count=0)
        assert te.count == 0

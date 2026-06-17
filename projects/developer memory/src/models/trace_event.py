"""Structured trace event model for the sync pipeline (F1 rework).

Replaces free-form trace strings as the UI ↔ backend metric contract.
`trace_events` is accumulated via operator.add on SyncState; each node
emits zero or more TraceEvent objects alongside the existing trace string list.

TraceEventType is a closed str Enum — a node emitting an unrecognised event
string raises ValueError at TraceEvent construction time, making event-name
typos visible in tests rather than silently dropping metrics.
"""

from enum import StrEnum

from pydantic import BaseModel


class TraceEventType(StrEnum):
    """Closed enumeration of valid event types for sync pipeline TraceEvents.

    Using str as the mixin lets TraceEvent be serialised to/from JSON without
    a custom encoder — the enum value is the plain string.
    """

    OK = "ok"
    QUARANTINE = "quarantine"
    SKIP = "skip"
    CACHED = "cached"
    DISPATCHED = "dispatched"  # chunk_dispatcher: N chunks sent to summarize_and_upsert
    ERROR = "error"


class TraceEvent(BaseModel):
    """A single structured event emitted by a sync pipeline node.

    All fields except `node` and `event` are optional so nodes can omit
    fields that are not applicable (e.g. `path` for a count-only event).

    Args:
        node: Canonical node name: "delta_extractor", "cache_filter", etc.
        event: Closed enum — typo in a node is a ValueError at construction time.
        path: File path when the event is file-scoped; None for aggregate events.
        count: Numeric payload (e.g. number of chunks dispatched, files filtered).
            None when the event has no numeric dimension.
        detail: Free-form non-metric context for logs only. Never parsed by the UI.
    """

    node: str
    event: TraceEventType
    path: str | None = None
    count: int | None = None
    detail: str | None = None

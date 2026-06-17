"""Developer Memory UI — session state definitions and lifecycle helpers.

Owns the canonical set of session-state keys, their defaults, the idempotent
`init_session_state()` initialiser, and the `_sync_is_active()` predicate.
No imports from other `ui/` modules.
"""

import streamlit as st

# ── Canonical session state schema ────────────────────────────────────────────

# All 11 keys with their initial defaults.  Used by init_session_state() and
# readable as a reference for any code that needs to know which keys exist.
_STATE_DEFAULTS: list[tuple[str, object]] = [
    ("_sync_job_id", None),
    # "queued" | "running" | "done" | "error" | "abandoned"
    ("_sync_status", None),
    # pipeline stage: "queued"|"cloning"|"cache_filtering"|"pii_scanning"|"parsing"|"summarizing"|"done"|"error"
    ("_sync_stage", "queued"),
    ("_sync_result", None),
    ("_sync_error", None),
    ("_sync_chunks_done", 0),
    ("_sync_chunks_total", 0),
    ("_sync_pii_done", 0),      # live PII scan file counter
    ("_sync_pii_total", 0),     # total files entering pii_sanitizer
    ("_sync_wall_start", None), # time.monotonic() captured when job is submitted
    ("_sync_elapsed_secs", None),  # elapsed seconds captured on completion
]


def init_session_state() -> None:
    """Populate st.session_state with default values for any key not yet present.

    Idempotent — keys that already have a value (including falsy defaults) are
    left unchanged.  Safe to call at module level on every Streamlit rerun.
    """
    for key, default in _STATE_DEFAULTS:
        if key not in st.session_state:
            st.session_state[key] = default


def _sync_is_active() -> bool:
    """Return True while a background sync is queued or running.

    Used to disable navigation and show a progress warning in the sidebar.
    """
    return st.session_state["_sync_status"] in ("queued", "running")

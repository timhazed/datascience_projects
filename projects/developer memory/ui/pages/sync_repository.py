"""Developer Memory UI — Sync Repository page renderer and all its display helpers.

Owns the Sync Repository page (`page()`) and the helpers that exclusively serve it:
`_build_stage_label`, `_render_stage_indicator`, `_format_elapsed`,
`_metrics_from_trace_events`, and `_render_sync_progress`.
"""

import os
import time

import streamlit as st

from ui.config import _STAGE_LABELS
from ui.http_client import _mcp_error, _poll_sync_status, _post_sync

# ── Pure display helpers ──────────────────────────────────────────────────────


def _format_elapsed(secs: float) -> str:
    """Format elapsed seconds as 'Xh Ym Zs', omitting leading zero units.

    Args:
        secs: Total elapsed seconds.

    Returns:
        Human-readable duration string e.g. '1h 23m 45s' or '4m 2s' or '38s'.
    """
    total = int(secs)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _build_stage_label(
    stage: str,
    chunks_done: int,
    chunks_total: int,
    pii_done: int = 0,
    pii_total: int = 0,
) -> tuple[str, float]:
    """Return a (display_label, progress_fraction) pair for the current pipeline stage.

    For "pii_scanning", the label includes a live X/N file counter when pii_total > 0,
    and the fraction advances proportionally through the 0.15–0.25 range (capped at 0.25).

    For "summarizing", the label includes a live X/N chunk counter when chunks_total > 0,
    and the fraction advances proportionally through the 0.30–1.0 range.

    Args:
        stage: Server-reported pipeline stage string.
        chunks_done: Chunks processed so far (LLM + upsert combined).
        chunks_total: Total chunks dispatched in this run.
        pii_done: Files scanned by pii_sanitizer so far.
        pii_total: Total files entering pii_sanitizer.

    Returns:
        Tuple of (label string, progress float 0.0–1.0).
    """
    if stage == "pii_scanning" and pii_total > 0:
        label = f"Scanning for PII — {pii_done} / {pii_total} files"
        # Advance proportionally through the 0.15–0.25 range reserved for PII scanning.
        frac = 0.15 + 0.10 * (pii_done / pii_total)
        return label, min(frac, 0.25)

    if stage == "summarizing" and chunks_total > 0:
        label = f"LLM Summarizing — {chunks_done} / {chunks_total} chunks"
        # Map chunk progress into the 0.30–1.0 range (0.30 reserved for pre-LLM stages).
        frac = 0.30 + 0.70 * (chunks_done / chunks_total)
        return label, min(frac, 0.99)  # Cap at 0.99 — 1.0 reserved for "done"

    base_label, base_frac = _STAGE_LABELS.get(stage, ("Processing…", 0.1))
    return base_label, base_frac



def _metrics_from_trace_events(state: dict) -> dict:
    """Extract pipeline metric counts from structured TraceEvent objects.

    Reads structured state fields for counts that are always available
    (changed_files, quarantined_files, cache_filtered_count), then accumulates
    per-event counts from the `trace_events` list emitted by Phase A nodes.

    Args:
        state: SyncState dict containing `trace_events` (list of TraceEvent dicts
            or TraceEvent objects), `changed_files`, `quarantined_files`, and
            `cache_filtered_count`.

    Returns:
        Dict with keys: changed_files, sanitized, quarantined, chunks, cached,
        dispatched, inserted, skipped, updated, errors.
    """
    # Spec §F1 Phase B: changed_files and quarantined are already structured state fields
    # — read them directly, independent of which events were emitted.
    counts: dict = {
        "changed_files": len(state.get("changed_files") or []),
        "sanitized": 0,
        "quarantined": len(state.get("quarantined_files") or []),
        "chunks": 0,
        "cached": state.get("cache_filtered_count", 0),  # cache_filtered_count is NotRequired
        "dispatched": 0,
        "inserted": 0,
        "skipped": 0,
        "updated": 0,
        "errors": 0,
    }

    # Accumulate per-event counts from TraceEvent dicts (JSON-decoded by the MCP client).
    # Each TraceEvent arrives as a dict (after JSON round-trip) or as a TraceEvent object
    # (when the node return value is consumed in the same process without serialisation).
    # trace_events absent in pre-Phase-A server responses; state fields still provide partial counts.
    for ev in state.get("trace_events") or []:
        node = ev.get("node", "") if isinstance(ev, dict) else getattr(ev, "node", "")
        event = ev.get("event", "") if isinstance(ev, dict) else getattr(ev, "event", "")

        if node == "pii_sanitizer" and event == "ok":
            counts["sanitized"] += 1
        elif node == "chunk_dispatcher" and event == "dispatched":
            # chunk_dispatcher emits count=N chunks dispatched; use as a display metric.
            count = ev.get("count") if isinstance(ev, dict) else getattr(ev, "count", None)
            counts["dispatched"] = count or 0
        elif node == "summarize_and_upsert":
            # Spec: chunks = count of summarize_and_upsert OK events (chunks actually processed).
            # This differs from multimodal_parser output count: chunks that fail LLM
            # summarization are absent here but present in the parser count.
            if event == "ok":
                counts["inserted"] += 1
                counts["chunks"] += 1
            elif event == "cached":
                counts["skipped"] += 1
            elif event == "error":
                counts["errors"] += 1

    return counts


def _render_sync_progress(result: dict) -> None:
    """Render a structured progress breakdown from a completed sync result.

    Shows each pipeline stage as a metric row with file/chunk counts, making
    it easy to see how many files were processed at each step.

    Args:
        result: SyncState dict returned by the sync_repository MCP tool.
            Must contain a `trace_events` field (populated by Phase A nodes).
    """
    trace: list[str] = result.get("trace", [])
    upsert_results: list = result.get("upsert_results", [])
    quarantined: list = result.get("quarantined_files", [])
    counts = _metrics_from_trace_events(result)

    # ── Stage metrics ──────────────────────────────────────────────────────────
    st.markdown("### Pipeline Stages")

    col1, col2, col3, col_cache = st.columns(4)
    with col1:
        st.metric("Files discovered", counts["changed_files"])
    with col2:
        st.metric("Cache filtered", counts["cached"],
                  help="Files skipped because this exact version is already indexed.")
    with col3:
        st.metric("PII-safe files", counts["sanitized"])
    with col_cache:
        st.metric("Quarantined", counts["quarantined"],
                  delta=f"-{counts['quarantined']}" if counts["quarantined"] else None,
                  delta_color="inverse")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Chunks parsed", counts["chunks"])
    with col5:
        # Post-§12: summarize+upsert are one collapsed operation; total processed =
        # inserted + skipped (cache hits) + updated (file moved).
        total_processed = counts["inserted"] + counts["skipped"] + counts["updated"]
        st.metric("Chunks processed", total_processed)
    with col6:
        st.metric("Errors", counts["errors"],
                  delta=f"{counts['errors']}" if counts["errors"] else None,
                  delta_color="inverse")

    # ── ChromaDB outcome ───────────────────────────────────────────────────────
    st.markdown("### ChromaDB Outcome")
    inserted = sum(1 for r in upsert_results if isinstance(r, dict) and r.get("action") == "inserted")
    skipped = sum(1 for r in upsert_results if isinstance(r, dict) and r.get("action") == "skipped")
    updated = sum(1 for r in upsert_results if isinstance(r, dict) and r.get("action") == "updated")
    errors_upsert = sum(1 for r in upsert_results if isinstance(r, dict) and r.get("action") == "error")

    col7, col8, col9, col10 = st.columns(4)
    with col7:
        st.metric("Inserted", inserted)
    with col8:
        st.metric("Updated", updated)
    with col9:
        st.metric("Skipped (dup)", skipped)
    with col10:
        st.metric("Upsert errors", errors_upsert,
                  delta=f"{errors_upsert}" if errors_upsert else None,
                  delta_color="inverse")

    # ── Quarantine detail ──────────────────────────────────────────────────────
    if quarantined:
        with st.expander(f"Quarantined files ({len(quarantined)})", expanded=False):
            for f in quarantined:
                path = f.get("path", "unknown") if isinstance(f, dict) else str(f)
                reason = f.get("quarantine_reason", "") if isinstance(f, dict) else ""
                st.markdown(f"- `{path}`" + (f" — {reason}" if reason else ""))

    # ── Full trace ─────────────────────────────────────────────────────────────
    with st.expander("Full execution trace", expanded=False):
        for entry in trace:
            st.text(entry)


def _render_stage_indicator(active_stage: str) -> None:
    """Render a compact pipeline stage indicator showing which stage is active.

    Displays all pipeline stages in order with the active stage highlighted.
    Uses st.caption rows to avoid layout disruption inside st.empty() containers.

    Args:
        active_stage: The current server-reported stage string.
    """
    stages = [
        ("cloning",          "Cloning & Delta"),
        ("cache_filtering",  "Cache Filter"),
        ("pii_scanning",     "PII Scan"),
        ("parsing",          "Parsing"),
        ("summarizing",      "LLM Summarize"),
    ]
    cols = st.columns(len(stages))
    for col, (stage_key, stage_name) in zip(cols, stages, strict=True):
        with col:
            if stage_key == active_stage:
                st.markdown(f"**▶ {stage_name}**")
            else:
                st.caption(stage_name)


# ── Page renderer ─────────────────────────────────────────────────────────────


def page() -> None:
    """Render the Sync Repository page.

    Uses POST /sync (returns job_id immediately) + GET /sync/status polling so the
    browser session cannot orphan or cancel the sync job. The graph runs as a daemon
    thread on the MCP server — independent of this HTTP connection.

    Session state keys used:
        _sync_job_id    — active job UUID4 (None = no job started)
        _sync_status    — last polled status
        _sync_result    — full SyncState dict (set on completion)
        _sync_error     — error message (set on failure)
        _sync_chunks_done / _sync_chunks_total — live upsert progress counters
    """
    st.header("Sync Repository")
    st.markdown(
        "Indexes all files on a branch into Developer Memory. "
        "Re-syncing is idempotent — unchanged files are skipped automatically."
    )

    job_id: str | None = st.session_state["_sync_job_id"]
    status: str | None = st.session_state["_sync_status"]

    # ── Launch form (shown when no job is active or after completion) ──────────
    is_active = status in ("queued", "running")
    is_done = status == "done"
    is_error = status in ("error", "abandoned")

    with st.form("sync_form"):
        repo_url = st.text_input(
            "Repository URL",
            placeholder="https://github.com/owner/repo",
            help="Must be on the allowed hosts list (github.com, gitlab.com, bitbucket.org).",
            disabled=is_active,
        )
        branch = st.text_input("Branch", value="main", disabled=is_active)
        col_submit, col_restart = st.columns([3, 1])
        with col_submit:
            submitted = st.form_submit_button(
                "Sync",
                type="primary",
                disabled=is_active,
                help="Cannot start a new sync while one is already running." if is_active else None,
            )
        with col_restart:
            restart = st.form_submit_button(
                "Restart",
                disabled=is_active or not (is_done or is_error),
                help="Clear the current result and start a fresh sync.",
            )

    if restart:
        for key in ("_sync_job_id", "_sync_status", "_sync_result", "_sync_error",
                    "_sync_wall_start", "_sync_elapsed_secs"):
            st.session_state[key] = None
        st.session_state["_sync_stage"] = "queued"
        st.session_state["_sync_chunks_done"] = 0
        st.session_state["_sync_chunks_total"] = 0
        st.session_state["_sync_pii_done"] = 0
        st.session_state["_sync_pii_total"] = 0
        st.rerun()

    if submitted:
        if not repo_url.strip():
            st.error("Repository URL is required.")
            return
        try:
            new_job_id = _post_sync(repo_url.strip(), branch.strip() or "main")
        except ValueError as exc:
            st.error(f"Validation error: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            _mcp_error(exc)
            return

        st.session_state["_sync_job_id"] = new_job_id
        st.session_state["_sync_status"] = "queued"
        st.session_state["_sync_stage"] = "queued"
        st.session_state["_sync_result"] = None
        st.session_state["_sync_error"] = None
        st.session_state["_sync_chunks_done"] = 0
        st.session_state["_sync_chunks_total"] = 0
        st.session_state["_sync_pii_done"] = 0
        st.session_state["_sync_pii_total"] = 0
        st.session_state["_sync_wall_start"] = time.monotonic()
        st.session_state["_sync_elapsed_secs"] = None
        st.rerun()

    # ── Live progress panel (shown while job is queued or running) ─────────────
    if is_active and job_id:
        status_box = st.empty()
        try:
            payload = _poll_sync_status(job_id)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not reach status endpoint: {exc}. Retrying…")
            # Block this worker for 2 s before retrying — the MCP job continues
            # independently on the server; this sleep only affects poll cadence.
            time.sleep(2.0)
            st.rerun()
            return

        new_status = payload.get("status", status)
        new_stage = payload.get("stage", "queued")
        chunks_done = payload.get("chunks_done", 0)
        chunks_total = payload.get("chunks_total", 0)
        pii_done = payload.get("pii_files_done", 0)
        pii_total = payload.get("pii_files_total", 0)

        # Update session state with latest poll values.
        # Use max() on counters to prevent them from appearing to go backwards
        # when stale poll values arrive out of order during rapid reruns.
        st.session_state["_sync_status"] = new_status
        st.session_state["_sync_stage"] = new_stage
        st.session_state["_sync_chunks_done"] = max(
            chunks_done, st.session_state["_sync_chunks_done"]
        )
        st.session_state["_sync_chunks_total"] = chunks_total
        st.session_state["_sync_pii_done"] = max(pii_done, st.session_state["_sync_pii_done"])
        st.session_state["_sync_pii_total"] = pii_total

        if new_status in ("queued", "running"):
            started_at = payload.get("started_at", "")
            stage_label, frac = _build_stage_label(
                new_stage, chunks_done, chunks_total, pii_done, pii_total
            )

            with status_box.container():
                st.info(
                    f"**{stage_label}**  \n"
                    f"Job `{job_id[:8]}…` — started {started_at[:19].replace('T', ' ')} UTC"
                )
                st.progress(frac)
                # Stage indicator chips
                _render_stage_indicator(new_stage)
                if st.button("Abandon (stop polling this job)", type="secondary"):
                    # Cannot kill the server-side thread, but we stop tracking it.
                    st.session_state["_sync_status"] = "abandoned"
                    st.session_state["_sync_error"] = (
                        "Polling abandoned by user. The sync job continues running on the server."
                    )
                    st.rerun()

            # Auto-rerun every 2 s to refresh progress — standard Streamlit polling
            # pattern; safe because the MCP server runs the sync job in its own
            # daemon thread, independent of this HTTP connection.
            time.sleep(2.0)
            st.rerun()

        elif new_status == "done":
            st.session_state["_sync_result"] = payload.get("result")
            st.session_state["_sync_status"] = "done"
            # Capture elapsed wall time on first completion transition.
            if st.session_state["_sync_wall_start"] is not None:
                st.session_state["_sync_elapsed_secs"] = (
                    time.monotonic() - st.session_state["_sync_wall_start"]
                )
            st.rerun()

        elif new_status == "error":
            st.session_state["_sync_error"] = payload.get("error", "Unknown error")
            st.session_state["_sync_status"] = "error"
            st.rerun()

    # ── Result display (shown after completion) ────────────────────────────────
    if is_done:
        result = st.session_state.get("_sync_result")
        if not isinstance(result, dict):
            st.error("Unexpected response from MCP server.")
            return
        if result.get("error"):
            st.error(f"Sync failed: {result['error']}")
            with st.expander("Execution trace"):
                for entry in result.get("trace", []):
                    st.text(entry)
            return

        upsert_results = result.get("upsert_results", [])
        quarantine_count = len(result.get("quarantined_files", []))
        inserted = sum(1 for r in upsert_results if isinstance(r, dict) and r.get("action") == "inserted")

        # Build elapsed time + model suffix for the completion banner.
        elapsed_secs = st.session_state.get("_sync_elapsed_secs")
        elapsed_str = f" in {_format_elapsed(elapsed_secs)}" if elapsed_secs is not None else ""
        model_name = os.environ.get("OLLAMA_MODEL", "unknown model")
        st.success(
            f"Sync complete — {inserted} new chunks indexed, "
            f"{len(upsert_results) - inserted} already up-to-date, "
            f"{quarantine_count} quarantined.  \n"
            f"Ingest & LLM summarization completed{elapsed_str} · model: `{model_name}`"
        )
        _render_sync_progress(result)

    if is_error:
        err = st.session_state.get("_sync_error", "Unknown error")
        st.error(f"Sync failed: {err}")
        st.caption("Use the Restart button above to try again.")

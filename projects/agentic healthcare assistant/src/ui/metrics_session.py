"""Append metrics to Streamlit session and optional MetricsDB session_log."""

from __future__ import annotations

import logging
from datetime import datetime

import streamlit as st

from src.db.metrics_db import MetricsDB
from src.ui.runtime import _SESSION_ID, settings

logger = logging.getLogger(__name__)


def _record_metric(
    operation: str,
    success: bool,
    latency_ms: float = 0.0,
    error: str = "",
    metrics_db: MetricsDB | None = None,
) -> None:
    """Append one entry to session_metrics and persist to session_log.

    Used by direct DB operations (patient lookup, encounter notes, record updates,
    registration) that bypass the graph and therefore never go through _invoke_and_update.

    Args:
        operation: Human-readable operation name (e.g. "Patient Lookup").
        success: Whether the operation succeeded.
        latency_ms: Optional elapsed time in milliseconds.
        error: Optional error string; empty string if none.
        metrics_db: MetricsDB instance for cross-session persistence.
            Pass None only in contexts where the DB is not yet initialised.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["session_metrics"].append({
        "datetime": now_str,
        "operation": operation,
        "success": success,
        "latency_ms": latency_ms,
        "error": error,
    })
    if metrics_db is not None:
        try:
            metrics_db.log_entry(
                str(settings.db.sqlite_path),
                _SESSION_ID,
                now_str,
                operation,
                success,
                latency_ms,
                error,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[metrics] log_entry failed: %s — %s", type(exc).__name__, exc)

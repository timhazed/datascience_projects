"""graph_config — shared LangGraph invocation config builder.

Single source of truth for constructing the ``config`` dict passed to
``graph.invoke()`` and ``graph.get_state()``.  Always emits
``configurable.thread_id`` so checkpoints are keyed consistently regardless
of whether ``checkpointing_enabled`` is True or False.
"""

from __future__ import annotations

import uuid
from typing import Any

from src.config.settings import Settings


def build_invoke_config(
    patient_id: str | None,
    settings: Settings,
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Return the config dict for a single graph invocation.

    ``thread_id`` is set to the canonical patient slug when one is available.
    When the patient is not yet resolved (e.g. first turn in the CLI), a
    ``session_id`` fallback is used; if that is also absent, a fresh UUID is
    generated so the invocation is still isolated from other threads.

    Args:
        patient_id: Canonical patient slug (e.g. ``"P-001"``), or ``None`` when
            the patient has not been resolved yet.
        settings: Loaded application settings — provides ``graph.recursion_limit``.
        session_id: Optional per-process/per-session UUID string used as the
            thread_id fallback when ``patient_id`` is absent (e.g. UI session).

    Returns:
        Dict suitable for ``graph.invoke(state, config)``, containing
        ``recursion_limit`` and ``configurable.thread_id``.
    """
    thread_id = (
        patient_id.strip()
        if patient_id
        else (session_id or str(uuid.uuid4()))
    )
    return {
        "recursion_limit": settings.graph.recursion_limit,
        "configurable": {"thread_id": thread_id},
    }

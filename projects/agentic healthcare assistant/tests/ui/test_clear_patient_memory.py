"""Phase 4: scoped clear of checkpoint conversation history for the active patient (left panel)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.ui.panels.left_panel import (
    _clear_active_patient_conversation_history,
    _render_left_panel,
)


def _make_graph_with_checkpointer() -> MagicMock:
    """Build a mock CompiledStateGraph with a checkpointer attached."""
    graph = MagicMock()
    graph.checkpointer = MagicMock()
    return graph


def _make_graph_no_checkpointer() -> MagicMock:
    """Build a mock graph without a checkpointer."""
    graph = MagicMock()
    graph.checkpointer = None
    return graph


def test_clear_calls_delete_thread_with_patient_id() -> None:
    """checkpointer.delete_thread is called with the str patient_id (Decision 7)."""
    graph = _make_graph_with_checkpointer()
    session: dict = {
        "chat_history": [("user", "hello")],
        "has_chatted": True,
        "last_planner_output": {"sub_goals": []},
    }
    with patch("src.ui.panels.left_panel.st") as mock_st:
        mock_st.session_state = session
        _clear_active_patient_conversation_history(graph, "P-active")
    graph.checkpointer.delete_thread.assert_called_once_with("P-active")


def test_clear_resets_session_state() -> None:
    """Chat session keys are wiped regardless of checkpointer presence."""
    graph = _make_graph_with_checkpointer()
    session: dict = {
        "chat_history": [("user", "a"), ("assistant", "b")],
        "has_chatted": True,
        "last_planner_output": {"x": 1},
    }
    with patch("src.ui.panels.left_panel.st") as mock_st:
        mock_st.session_state = session
        _clear_active_patient_conversation_history(graph, "P-1")
    assert session["chat_history"] == []
    assert session["has_chatted"] is False
    assert session["last_planner_output"] is None


def test_clear_does_not_affect_other_patients() -> None:
    """delete_thread is invoked only for the exact slug passed in (active patient)."""
    graph = _make_graph_with_checkpointer()
    session: dict = {
        "chat_history": [],
        "has_chatted": False,
        "last_planner_output": None,
    }
    with patch("src.ui.panels.left_panel.st") as mock_st:
        mock_st.session_state = session
        _clear_active_patient_conversation_history(graph, "P-only-this")
    graph.checkpointer.delete_thread.assert_called_once_with("P-only-this")


def test_clear_button_not_rendered_when_no_patient() -> None:
    """Without a resolved name, the left panel must not register the history-clear button."""
    button_keys: list[object] = []

    def _record_button(*args, **kwargs) -> bool:
        button_keys.append(kwargs.get("key"))
        return False

    session: dict = {
        "resolved_patient_id": None,
        "resolved_patient_name": None,
        "_left_panel_matches": [],
        "chat_history": [],
        "has_chatted": False,
        "last_planner_output": None,
        "last_trace": [],
        "completed_tasks": [],
        "session_metrics": [],
        "last_query_for_eval": "",
        "last_answer_for_eval": "",
    }
    patient_db = MagicMock()
    graph = _make_graph_with_checkpointer()

    with patch("src.ui.panels.left_panel.st") as mock_st:
        mock_st.session_state = session
        mock_st.text_input.return_value = ""
        mock_st.button.side_effect = _record_button
        mock_st.success = MagicMock()
        mock_st.warning = MagicMock()
        mock_st.caption = MagicMock()
        mock_st.divider = MagicMock()
        with patch("src.ui.panels.left_panel.settings") as mock_settings:
            mock_settings.db.sqlite_path = Path("/tmp/test.db")
            _render_left_panel(patient_db, graph, metrics_db=None)

    assert "left_clear_history" not in button_keys


def test_clear_helper_skips_delete_when_no_checkpointer() -> None:
    """When graph has no checkpointer, session keys are still cleared."""
    graph = _make_graph_no_checkpointer()
    session: dict = {
        "chat_history": [("user", "x")],
        "has_chatted": True,
        "last_planner_output": {},
    }
    with patch("src.ui.panels.left_panel.st") as mock_st:
        mock_st.session_state = session
        _clear_active_patient_conversation_history(graph, "P-x")
    # No checkpointer → delete_thread not called; session still cleared
    assert session["chat_history"] == []
    assert session["has_chatted"] is False

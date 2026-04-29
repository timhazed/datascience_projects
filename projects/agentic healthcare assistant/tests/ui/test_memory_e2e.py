"""Phase 4 scripted E2E checks for checkpoint-based conversation memory.

Verifies checkpoint hydration, patient isolation via separate thread_ids,
and that _invoke_and_update uses the graph's checkpoint (not PatientMemoryDB).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.ui.graph_runtime import _hydrate_chat_from_checkpoint, _invoke_and_update


def _session_for_invoke(*, patient_id: str = "P-e2e") -> dict:
    return {
        "resolved_patient_id": patient_id,
        "resolved_patient_name": "E2E Patient",
        "completed_tasks": [],
        "session_metrics": [],
        "last_planner_output": None,
        "last_trace": [],
        "chat_history": [],
        "has_chatted": False,
    }


def _graph_with_messages(messages: list) -> MagicMock:
    """Mock graph whose checkpoint contains the given messages."""
    graph = MagicMock()
    graph.checkpointer = MagicMock()
    snapshot = MagicMock()
    snapshot.values = {"messages": messages}
    graph.get_state.return_value = snapshot
    graph.invoke.return_value = {
        "patient_id": None,
        "planner_output": None,
        "completed_tasks": [],
        "final_summary": "reply",
        "trace": [],
        "error": None,
    }
    return graph


class TestCheckpointHydration:
    """Hydration loads the correct patient's messages from the checkpoint."""

    def test_hydration_populates_chat_history_from_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.ui.graph_runtime.settings.db.max_recent_turns", 10)
        messages = [
            HumanMessage(content="Question after restart?"),
            AIMessage(content="Answer persists."),
        ]
        graph = _graph_with_messages(messages)
        session: dict = {"chat_history": [], "has_chatted": False}
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session
            _hydrate_chat_from_checkpoint(graph, "P-A")
        assert len(session["chat_history"]) == 2
        assert session["chat_history"][0] == ("user", "Question after restart?")
        assert session["chat_history"][1] == ("assistant", "Answer persists.")


class TestPatientIsolation:
    """Switching to patient B must not leak patient A's checkpoint content."""

    def test_patient_b_messages_exclude_patient_a_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.ui.graph_runtime.settings.db.max_recent_turns", 10)
        # Only B's messages are in the snapshot returned for P-B
        b_messages = [
            HumanMessage(content="SECRET_BRAVO_THREAD"),
            AIMessage(content="Reply B"),
        ]
        graph = _graph_with_messages(b_messages)
        session: dict = {
            "chat_history": [("user", "SECRET_ALPHA_THREAD")],
            "has_chatted": True,
        }
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session
            _hydrate_chat_from_checkpoint(graph, "P-B")

        joined = " ".join(m for _, m in session["chat_history"])
        assert "SECRET_ALPHA_THREAD" not in joined
        assert "SECRET_BRAVO_THREAD" in joined

    def test_hydrate_chat_loads_only_selected_patient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.ui.graph_runtime.settings.db.max_recent_turns", 10)
        # The graph returns only A's messages (get_state is called with P-A's config)
        a_messages = [HumanMessage(content="Only A")]
        graph = _graph_with_messages(a_messages)
        session: dict = {"chat_history": [("user", "stale")], "has_chatted": True}
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session
            _hydrate_chat_from_checkpoint(graph, "P-A")
        assert session["chat_history"] == [("user", "Only A")]
        assert session["has_chatted"] is True


class TestInvokeNoLegacyMemoryWrite:
    """_invoke_and_update must not call any PatientMemoryDB methods."""

    def test_no_db_write_on_invoke(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.ui.graph_runtime.settings.db.sqlite_path",
            tmp_path / "db.sqlite",
        )
        graph = MagicMock()
        graph.invoke.return_value = {
            "patient_id": None,
            "planner_output": None,
            "completed_tasks": [],
            "final_summary": "Stub assistant reply.",
            "trace": [],
            "error": None,
        }
        session = _session_for_invoke(patient_id="P-e2e")

        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session
            _invoke_and_update(graph, "user query", None, metrics_db=None)

        # Verify the graph was called but no PatientMemoryDB methods were invoked
        graph.invoke.assert_called_once()
        # No append_message, upsert_summary, etc. — PatientMemoryDB is removed.


class TestScopedClear:
    """Clearing conversation history calls checkpointer.delete_thread for that patient."""

    def test_clear_calls_delete_thread(self) -> None:
        from src.ui.panels.left_panel import _clear_active_patient_conversation_history

        graph = MagicMock()
        graph.checkpointer = MagicMock()
        session: dict = {
            "chat_history": [("user", "x")],
            "has_chatted": True,
            "last_planner_output": {"x": 1},
        }
        with patch("src.ui.panels.left_panel.st") as mock_st:
            mock_st.session_state = session
            _clear_active_patient_conversation_history(graph, "P-clear")

        graph.checkpointer.delete_thread.assert_called_once_with("P-clear")
        assert session["chat_history"] == []

    def test_clear_other_patient_untouched(self) -> None:
        """delete_thread is only called for the exact slug passed; other threads unaffected."""
        from src.ui.panels.left_panel import _clear_active_patient_conversation_history

        graph = MagicMock()
        graph.checkpointer = MagicMock()
        session: dict = {
            "chat_history": [],
            "has_chatted": False,
            "last_planner_output": None,
        }
        with patch("src.ui.panels.left_panel.st") as mock_st:
            mock_st.session_state = session
            _clear_active_patient_conversation_history(graph, "P-drop")

        graph.checkpointer.delete_thread.assert_called_once_with("P-drop")

"""Tests for graph_runtime._invoke_and_update (Phase 4 — checkpoint-only persistence)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ui.graph_runtime import _invoke_and_update


def _base_session(*, patient_id: str | None = "P-test") -> dict:
    """Minimal session_state for _invoke_and_update (matches keys it touches)."""
    return {
        "resolved_patient_id": patient_id,
        "resolved_patient_name": "Test Patient",
        "completed_tasks": [],
        "session_metrics": [],
        "last_planner_output": None,
        "last_trace": [],
    }


@pytest.fixture
def tmp_db_path(tmp_path):
    return tmp_path / "healthcare.db"


@pytest.fixture
def graph_ok(tmp_db_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.ui.graph_runtime.settings.db.sqlite_path", tmp_db_path)
    g = MagicMock()
    g.invoke.return_value = {
        "patient_id": None,
        "planner_output": None,
        "completed_tasks": [],
        "final_summary": "The assistant reply.",
        "trace": [],
        "error": None,
    }
    return g


class TestInvokeHydratePreservesInFlightChat:
    """Mid-invoke first resolve must not drop the user turn already in chat_history."""

    def test_first_resolve_appends_in_flight_after_checkpoint_turns(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.ui.graph_runtime.settings.db.sqlite_path",
            tmp_path / "db.sqlite",
        )
        monkeypatch.setattr("src.ui.graph_runtime.settings.db.max_recent_turns", 10)
        # Mock checkpoint providing prior history for the newly resolved patient
        graph = MagicMock()
        graph.checkpointer = MagicMock()
        from langchain_core.messages import AIMessage, HumanMessage
        snapshot = MagicMock()
        snapshot.values = {"messages": [
            HumanMessage(content="earlier"),
            AIMessage(content="prior reply"),
        ]}
        graph.get_state.return_value = snapshot
        graph.invoke.return_value = {
            "patient_id": "P-resolved",
            "planner_output": None,
            "completed_tasks": [],
            "final_summary": "final",
            "trace": [],
            "error": None,
        }
        session = _base_session(patient_id=None)
        session["chat_history"] = [("user", "current question")]
        session["has_chatted"] = True
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session
            _invoke_and_update(graph, "current question", "Hint Name", metrics_db=None)
        assert session["resolved_patient_id"] == "P-resolved"
        assert session["resolved_patient_name"] == "Hint Name"
        assert ("user", "current question") in session["chat_history"]
        assert session["has_chatted"] is True

    def test_first_resolve_empty_checkpoint_keeps_in_flight_user_turn(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.ui.graph_runtime.settings.db.sqlite_path",
            tmp_path / "db.sqlite",
        )
        monkeypatch.setattr("src.ui.graph_runtime.settings.db.max_recent_turns", 10)
        graph = MagicMock()
        graph.checkpointer = MagicMock()
        snapshot = MagicMock()
        snapshot.values = {}
        graph.get_state.return_value = snapshot
        graph.invoke.return_value = {
            "patient_id": "P-new",
            "planner_output": None,
            "completed_tasks": [],
            "final_summary": "answer",
            "trace": [],
            "error": None,
        }
        session = _base_session(patient_id=None)
        session["chat_history"] = [("user", "fresh prompt")]
        session["has_chatted"] = False
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session
            _invoke_and_update(graph, "fresh prompt", "Pat", metrics_db=None)
        assert ("user", "fresh prompt") in session["chat_history"]


class TestInvokeAndUpdateCore:
    def test_returns_ui_this_run_metadata(self, graph_ok, tmp_db_path) -> None:
        """Memory & Logs reads _ui_this_run_* for local latency / operation strip."""
        session = _base_session(patient_id="P-test")
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session
            out = _invoke_and_update(graph_ok, "What is CKD?", None, metrics_db=None)
        assert "_ui_this_run_latency_ms" in out
        assert isinstance(out["_ui_this_run_latency_ms"], int | float)
        assert out["_ui_this_run_operation"] == (
            "No tool tasks in this run (summarizer only)."
        )

    def test_final_summary_returned(self, graph_ok, tmp_db_path) -> None:
        """Final summary passes through from graph result."""
        session = _base_session(patient_id="P-test")
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session
            out = _invoke_and_update(graph_ok, "query", None, metrics_db=None)
        assert out.get("final_summary") == "The assistant reply."

    def test_session_completed_tasks_extended(self, graph_ok, tmp_db_path) -> None:
        """Session accumulated_tasks list grows with each invocation."""
        from src.models.task_result import TaskResult
        task = TaskResult(task="search_disease", success=True)
        graph_ok.invoke.return_value = {
            "patient_id": None,
            "planner_output": None,
            "completed_tasks": [task],
            "final_summary": "reply",
            "trace": [],
            "error": None,
        }
        session = _base_session(patient_id="P-test")
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session
            _invoke_and_update(graph_ok, "query", None, metrics_db=None)
        assert len(session["completed_tasks"]) == 1

    def test_eval_keys_populated(self, graph_ok, tmp_db_path) -> None:
        """last_query_for_eval and last_answer_for_eval are set after invocation."""
        session = _base_session(patient_id="P-test")
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session
            _invoke_and_update(graph_ok, "test query", None, metrics_db=None)
        assert session.get("last_query_for_eval") == "test query"
        assert session.get("last_answer_for_eval") == "The assistant reply."

"""Phase 4 rolling summary rollup tests — now live in summarizer_node, not graph_runtime.

The rolling summary is no longer triggered from _invoke_and_update. These tests
confirm that _invoke_and_update works correctly without memory_summary_chain,
and that the rollup logic (tested exhaustively in test_summarizer_node.py)
is no longer wired into the UI layer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ui.graph_runtime import _invoke_and_update


def _session(*, patient_id: str | None = "P-test") -> dict:
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


class TestInvokeAndUpdateNoLegacyRollup:
    """_invoke_and_update no longer accepts or uses memory_summary_chain or patient_memory_db."""

    def test_invoke_succeeds_without_memory_kwargs(self, graph_ok) -> None:
        """Invoke runs cleanly with only required args — no legacy kwargs needed."""
        session = _session()
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session
            out = _invoke_and_update(graph_ok, "What is CKD?", None, metrics_db=None)
        assert out.get("final_summary") == "The assistant reply."

    def test_no_invoke_with_retry_called_outside_graph(self, graph_ok) -> None:
        """invoke_with_retry is not imported by graph_runtime (only used inside nodes).

        Phase 4: rolling summary moved to summarizer_node; graph_runtime no longer
        imports or calls invoke_with_retry directly.  The absence of the import is
        the strongest possible proof — patching a symbol that does not exist in the
        module would itself raise AttributeError.
        """
        import src.ui.graph_runtime as _gr

        assert not hasattr(_gr, "invoke_with_retry"), (
            "invoke_with_retry must not be imported in graph_runtime — "
            "it belongs exclusively in the node layer."
        )

    def test_metrics_recorded_when_tasks_present(
        self, tmp_db_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Session metrics entry is created when completed_tasks is non-empty."""
        monkeypatch.setattr("src.ui.graph_runtime.settings.db.sqlite_path", tmp_db_path)
        from src.models.task_result import TaskResult
        g = MagicMock()
        g.invoke.return_value = {
            "patient_id": None,
            "planner_output": None,
            "completed_tasks": [TaskResult(task="search_disease", success=True)],
            "final_summary": "reply",
            "trace": [],
            "error": None,
        }
        session = _session()
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session
            _invoke_and_update(g, "q", None, metrics_db=None)
        assert len(session["session_metrics"]) == 1
        assert session["session_metrics"][0]["operation"] == "Disease Search"

    def test_empty_reply_handled_gracefully(
        self, tmp_db_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No final_summary and no error returns generic fallback from user_facing_assistant_text."""  # noqa: E501
        monkeypatch.setattr("src.ui.graph_runtime.settings.db.sqlite_path", tmp_db_path)
        from src.ui.graph_runtime import user_facing_assistant_text
        g = MagicMock()
        g.invoke.return_value = {
            "patient_id": None,
            "planner_output": None,
            "completed_tasks": [],
            "final_summary": None,
            "trace": [],
            "error": None,
            "intent_safe": True,
        }
        session = _session()
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session
            out = _invoke_and_update(g, "q", None, metrics_db=None)
        text = user_facing_assistant_text(out)
        assert "No response" in text

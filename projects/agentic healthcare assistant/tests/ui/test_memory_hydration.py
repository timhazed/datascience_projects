"""Tests for _hydrate_chat_from_checkpoint (Phase 4 checkpoint hydration)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.ui.graph_runtime import (
    _hydrate_chat_from_checkpoint,
    set_resolved_patient_with_hydration,
)


def _make_graph_with_checkpoint(messages: list) -> MagicMock:
    """Build a mock CompiledStateGraph whose get_state returns the given messages."""
    graph = MagicMock()
    graph.checkpointer = MagicMock()
    snapshot = MagicMock()
    snapshot.values = {"messages": messages}
    graph.get_state.return_value = snapshot
    return graph


def _make_graph_no_checkpointer() -> MagicMock:
    """Build a mock graph without a checkpointer attached."""
    graph = MagicMock()
    graph.checkpointer = None
    return graph


def _make_graph_empty_snapshot() -> MagicMock:
    """Build a mock graph where get_state returns an empty snapshot."""
    graph = MagicMock()
    graph.checkpointer = MagicMock()
    snapshot = MagicMock()
    snapshot.values = {}
    graph.get_state.return_value = snapshot
    return graph


@pytest.fixture
def session_state() -> dict:
    return {
        "chat_history": [("user", "stale")],
        "has_chatted": True,
    }


class TestHydrateChatFromCheckpoint:
    def test_hydration_populates_chat_history(
        self, session_state: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.ui.graph_runtime.settings.db.max_recent_turns", 10)
        messages = [
            HumanMessage(content="a"),
            AIMessage(content="b"),
            HumanMessage(content="c"),
        ]
        graph = _make_graph_with_checkpoint(messages)
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session_state
            _hydrate_chat_from_checkpoint(graph, "P-1")
        assert session_state["chat_history"] == [
            ("user", "a"),
            ("assistant", "b"),
            ("user", "c"),
        ]
        assert session_state["has_chatted"] is True

    def test_hydration_empty_snapshot_clears_history(
        self, session_state: dict
    ) -> None:
        graph = _make_graph_empty_snapshot()
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session_state
            _hydrate_chat_from_checkpoint(graph, "P-new")
        assert session_state["chat_history"] == []
        assert session_state["has_chatted"] is False

    def test_patient_switch_ab_no_leakage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.ui.graph_runtime.settings.db.max_recent_turns", 10)
        session = {
            "chat_history": [("user", "only-a"), ("assistant", "reply-a")],
            "has_chatted": True,
        }
        graph = _make_graph_empty_snapshot()
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session
            _hydrate_chat_from_checkpoint(graph, "P-B")
        assert session["chat_history"] == []
        assert session["has_chatted"] is False

    def test_patient_switch_ab_with_b_history(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.ui.graph_runtime.settings.db.max_recent_turns", 10)
        session = {
            "chat_history": [("user", "only-a")],
            "has_chatted": True,
        }
        messages = [
            HumanMessage(content="b1"),
            AIMessage(content="b2"),
        ]
        graph = _make_graph_with_checkpoint(messages)
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session
            _hydrate_chat_from_checkpoint(graph, "P-B")
        assert session["chat_history"] == [("user", "b1"), ("assistant", "b2")]
        assert session["has_chatted"] is True

    def test_no_checkpointer_only_clears(self, session_state: dict) -> None:
        """When graph has no checkpointer, only wipe is performed — no get_state call."""
        graph = _make_graph_no_checkpointer()
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session_state
            _hydrate_chat_from_checkpoint(graph, "P-1")
        assert session_state["chat_history"] == []
        assert session_state["has_chatted"] is False
        graph.get_state.assert_not_called()

    def test_none_patient_id_only_clears(self, session_state: dict) -> None:
        """patient_id=None clears chat without querying checkpoint."""
        graph = MagicMock()
        graph.checkpointer = MagicMock()
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session_state
            _hydrate_chat_from_checkpoint(graph, None)
        assert session_state["chat_history"] == []
        graph.get_state.assert_not_called()

    def test_turns_capped_at_max_recent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the last max_recent_turns pairs are loaded into chat_history."""
        monkeypatch.setattr("src.ui.graph_runtime.settings.db.max_recent_turns", 2)
        messages = []
        for i in range(5):
            messages.append(HumanMessage(content=f"q{i}"))
            messages.append(AIMessage(content=f"a{i}"))
        graph = _make_graph_with_checkpoint(messages)
        session = {"chat_history": [], "has_chatted": False}
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state = session
            _hydrate_chat_from_checkpoint(graph, "P-x")
        # max_recent_turns=2 → last 2 entries from pairs list
        assert len(session["chat_history"]) == 2


class TestSetResolvedPatientWithHydration:
    def test_sets_session_and_calls_hydrate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.ui.graph_runtime.settings.db.max_recent_turns", 10)
        graph = MagicMock()
        session: dict = {
            "resolved_patient_id": None,
            "resolved_patient_name": None,
            "chat_history": [],
            "has_chatted": False,
        }
        with patch("src.ui.graph_runtime._hydrate_chat_from_checkpoint") as mock_h:
            with patch("src.ui.graph_runtime.st") as mock_st:
                mock_st.session_state = session
                set_resolved_patient_with_hydration(graph, "P-99", "Sam Rivera")
        assert session["resolved_patient_id"] == "P-99"
        assert session["resolved_patient_name"] == "Sam Rivera"
        mock_h.assert_called_once_with(graph, "P-99")

    def test_re_resolve_same_id_updates_name_and_hydrates_again(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Selecting the same patient again still refreshes display name and history."""
        monkeypatch.setattr("src.ui.graph_runtime.settings.db.max_recent_turns", 10)
        graph = MagicMock()
        session: dict = {
            "resolved_patient_id": "P-99",
            "resolved_patient_name": "Old Name",
            "chat_history": [("user", "x")],
            "has_chatted": True,
        }
        with patch("src.ui.graph_runtime._hydrate_chat_from_checkpoint") as mock_h:
            with patch("src.ui.graph_runtime.st") as mock_st:
                mock_st.session_state = session
                set_resolved_patient_with_hydration(graph, "P-99", "Sam Rivera")
                set_resolved_patient_with_hydration(graph, "P-99", "Sam Rivera")
        assert session["resolved_patient_id"] == "P-99"
        assert session["resolved_patient_name"] == "Sam Rivera"
        assert mock_h.call_count == 2
        mock_h.assert_called_with(graph, "P-99")


def test_patient_and_doctor_tabs_use_graph_runtime_hydration_helper() -> None:
    """Smoke: tabs import the shared helper; catches broken renames / shadow mocks."""
    from src.ui import graph_runtime
    from src.ui.tabs import doctor_tab, patient_tab

    assert patient_tab.set_resolved_patient_with_hydration is (
        graph_runtime.set_resolved_patient_with_hydration
    )
    assert doctor_tab.set_resolved_patient_with_hydration is (
        graph_runtime.set_resolved_patient_with_hydration
    )

"""Tests for src/utils/history.py — trim_history."""

from src.utils.constants import MAX_HISTORY_TURNS
from src.utils.history import trim_history


def _make_history(n_turns: int) -> list[dict]:
    """Build a history list with n_turns complete turns (user + assistant each)."""
    history = []
    for i in range(n_turns):
        history.append({"role": "user", "content": f"user message {i}"})
        history.append({"role": "assistant", "content": f"assistant message {i}"})
    return history


class TestTrimHistory:
    def test_empty_list_returns_empty(self):
        assert trim_history([]) == []

    def test_under_cap_unchanged(self):
        history = _make_history(MAX_HISTORY_TURNS - 1)
        result = trim_history(history)
        assert result == history

    def test_exactly_at_cap_unchanged(self):
        history = _make_history(MAX_HISTORY_TURNS)
        result = trim_history(history)
        assert result == history

    def test_one_over_cap_drops_oldest_turn(self):
        history = _make_history(MAX_HISTORY_TURNS + 1)
        result = trim_history(history)
        assert len(result) == MAX_HISTORY_TURNS * 2
        # Oldest two messages (index 0 and 1) must have been dropped.
        assert result[0]["content"] == "user message 1"

    def test_does_not_mutate_original(self):
        history = _make_history(MAX_HISTORY_TURNS + 2)
        original_len = len(history)
        trim_history(history)
        assert len(history) == original_len

    def test_custom_max_turns(self):
        history = _make_history(5)
        result = trim_history(history, max_turns=3)
        assert len(result) == 6  # 3 turns × 2 messages

    def test_result_contains_most_recent_turns(self):
        history = _make_history(MAX_HISTORY_TURNS + 3)
        result = trim_history(history)
        # The last message in the trimmed result must be the last in the original.
        assert result[-1] == history[-1]

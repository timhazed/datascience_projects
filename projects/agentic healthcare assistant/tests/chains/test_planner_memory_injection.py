"""Tests for Phase 4 planner prompt memory_context wiring."""

from __future__ import annotations

from src.chains.planner_chain import _PLANNER_PROMPT


def test_planner_prompt_has_memory_context_variable() -> None:
    assert "memory_context" in _PLANNER_PROMPT.input_variables

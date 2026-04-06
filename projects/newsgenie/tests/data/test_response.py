"""Tests for src/data/response.py — IntentSection and SupervisorResponse."""

import pytest
from pydantic import ValidationError

from src.data.enums import AgentName
from src.data.response import IntentSection, SupervisorResponse


def _section(agent: AgentName = AgentName.BUSINESS, query: str = "test") -> IntentSection:
    return IntentSection(agent=agent, query=query)


class TestIntentSection:
    def test_basic_construction(self):
        s = _section(AgentName.SPORTS, "Champions League")
        assert s.agent == AgentName.SPORTS
        assert s.query == "Champions League"
        assert s.articles == []


class TestSupervisorResponse:
    def test_single_section(self):
        resp = SupervisorResponse(
            session_id="s1",
            sections=[_section()],
            sources_used=["newsapi"],
            fallback_used=False,
        )
        assert len(resp.sections) == 1
        assert resp.fallback_used is False

    def test_multiple_sections(self):
        resp = SupervisorResponse(
            session_id="s1",
            sections=[_section(AgentName.BUSINESS), _section(AgentName.SPORTS)],
            sources_used=["newsapi", "guardian"],
            fallback_used=False,
        )
        assert len(resp.sections) == 2

    def test_empty_sections_raises(self):
        with pytest.raises(ValidationError):
            SupervisorResponse(
                session_id="s1",
                sections=[],
                sources_used=[],
                fallback_used=False,
            )

    def test_fallback_flag(self):
        resp = SupervisorResponse(
            session_id="s1",
            sections=[_section()],
            sources_used=[],
            fallback_used=True,
        )
        assert resp.fallback_used is True

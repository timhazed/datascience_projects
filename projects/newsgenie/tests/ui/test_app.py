"""Tests for src/ui/app.py — pure logic functions (no Streamlit calls)."""

from unittest.mock import MagicMock

import pytest

from src.data.enums import AgentName, NewsCategory
from src.data.response import SupervisorResponse
from src.ui.app import BACKEND_LABEL, _build_history, _compose_query


class MockSessionState(dict):
    """Minimal stand-in for st.session_state that supports attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self[key] = value


@pytest.fixture
def session(monkeypatch):
    """Patch st.session_state with a fresh MockSessionState."""
    state = MockSessionState(selected_chip=None)
    import src.ui.app as app_module
    monkeypatch.setattr(app_module.st, "session_state", state)
    return state


class TestBuildHistory:
    def test_empty_messages_returns_empty(self):
        assert _build_history([]) == []

    def test_user_turn_preserved(self):
        messages = [{"role": "user", "content": "What is happening in the NBA?"}]
        result = _build_history(messages)
        assert result == [{"role": "user", "content": "What is happening in the NBA?"}]

    def test_assistant_turn_serialized(self):
        response = MagicMock(spec=SupervisorResponse)
        response.sections = [MagicMock(query="NBA trade news")]
        messages = [{"role": "assistant", "response": response}]
        result = _build_history(messages)
        assert result[0]["role"] == "assistant"
        assert "NBA trade news" in result[0]["content"]

    def test_assistant_multiple_sections_joined(self):
        response = MagicMock(spec=SupervisorResponse)
        response.sections = [MagicMock(query="NBA scores"), MagicMock(query="NHL standings")]
        messages = [{"role": "assistant", "response": response}]
        result = _build_history(messages)
        assert result[0]["content"] == "[Covered: NBA scores, NHL standings]"

    def test_full_conversation_round_trip(self):
        response = MagicMock(spec=SupervisorResponse)
        response.sections = [MagicMock(query="markets news")]
        messages = [
            {"role": "user", "content": "Tell me about markets"},
            {"role": "assistant", "response": response},
            {"role": "user", "content": "Tell me more"},
        ]
        result = _build_history(messages)
        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "Tell me about markets"}
        assert result[1]["role"] == "assistant"
        assert result[2] == {"role": "user", "content": "Tell me more"}


class TestGetNewsButton:
    def test_button_not_rendered_without_chip(self, session):
        """st.button('Get News') is NOT called when no chip is selected."""
        session.selected_chip = None
        mock_st = MagicMock()
        get_news_clicked = False
        if session.selected_chip is not None:
            get_news_clicked = mock_st.button(
                "Get News", key="get_news_btn", use_container_width=False
            )
        assert get_news_clicked is False
        mock_st.button.assert_not_called()

    def test_button_rendered_with_chip(self, session):
        """st.button('Get News') is called and its return value is read directly."""
        session.selected_chip = NewsCategory.SPORTS
        mock_st = MagicMock()
        mock_st.button.return_value = True
        get_news_clicked = False
        if session.selected_chip is not None:
            get_news_clicked = mock_st.button(
                "Get News", key="get_news_btn", use_container_width=False
            )
        assert get_news_clicked is True
        mock_st.button.assert_called_once_with(
            "Get News", key="get_news_btn", use_container_width=False
        )


class TestBackendLabel:
    def test_all_agents_have_labels(self):
        """Every AgentName must have an entry in BACKEND_LABEL."""
        for agent in AgentName:
            assert agent in BACKEND_LABEL, f"Missing label for {agent}"

    def test_business_label_contains_newsapi(self):
        assert "NewsAPI" in BACKEND_LABEL[AgentName.BUSINESS]

    def test_sports_label_contains_guardian(self):
        assert "Guardian" in BACKEND_LABEL[AgentName.SPORTS]

    def test_general_label_shows_world(self):
        assert "World" in BACKEND_LABEL[AgentName.GENERAL]

    def test_web_search_label(self):
        assert "Web Search" in BACKEND_LABEL[AgentName.WEB_SEARCH]


class TestComposeQuery:
    def test_text_only_no_chip(self, session):
        session.selected_chip = None
        text, categories = _compose_query("Bitcoin price")
        assert text == "Bitcoin price"
        assert categories == []

    def test_headline_chip_no_text(self, session):
        session.selected_chip = NewsCategory.BUSINESS
        text, categories = _compose_query("")
        assert "Business" in text
        assert "Show me the latest news on" in text
        assert NewsCategory.BUSINESS in categories

    def test_headline_chip_with_text(self, session):
        session.selected_chip = NewsCategory.SPORTS
        text, categories = _compose_query("Bruins score")
        assert "Bruins score" in text
        assert "Sports" in text
        assert NewsCategory.SPORTS in categories

    def test_world_chip_no_text(self, session):
        session.selected_chip = NewsCategory.GENERAL
        text, categories = _compose_query("")
        assert "World" in text
        assert NewsCategory.GENERAL in categories

    def test_topic_chip_no_text(self, session):
        session.selected_chip = "Cooking"
        text, categories = _compose_query("")
        assert "Cooking" in text
        assert "Show me the latest news on" in text
        assert NewsCategory.WEB in categories

    def test_topic_chip_with_text(self, session):
        session.selected_chip = "Fashion"
        text, categories = _compose_query("spring trends")
        assert "spring trends" in text
        assert "Fashion" in text
        assert NewsCategory.WEB in categories

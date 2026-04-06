"""Tests for src/agents/sports_agent.py."""

from unittest.mock import patch

import httpx

from src.agents.sports_agent import SportsNewsAgent
from src.data.enums import AgentName, NewsCategory

_SETTINGS = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
_SETTINGS.guardian_api_key = "test-key"
_SETTINGS.max_articles_per_agent = 5

_RAW_ARTICLE = {
    "id": "sport/2026/mar/31/champions-league",
    "webTitle": "Champions League Results",
    "webUrl": "https://theguardian.com/sport/cl",
    "webPublicationDate": "2026-03-31T20:00:00Z",
    "sectionName": "Sport",
    "fields": {
        "headline": "CL Results: Real Madrid Win",
        "trailText": "Real Madrid beat Bayern 3-1.",
        "thumbnail": None,
    },
}

_ESPN_RAW = [
    {
        "id": "401803529",
        "shortName": "DAL @ BOS",
        "date": "2026-04-01T00:00:00Z",
        "links": [{"href": "https://espn.com/nhl/game/_/gameId/401803529"}],
        "competitions": [{
            "status": {"type": {"name": "STATUS_FINAL"}},
            "headlines": [{"shortLinkText": "Bruins beat Stars 6-3",
                           "description": "Boston wins at home."}],
            "competitors": [
                {"team": {"displayName": "Boston Bruins", "logo": None}, "score": "6"},
                {"team": {"displayName": "Dallas Stars", "logo": None}, "score": "3"},
            ],
        }],
    }
]


class TestSportsNewsAgent:
    def test_fetch_returns_guardian_and_espn_articles(self):
        with (
            patch("src.agents.sports_agent.guardian_client.fetch_articles",
                  return_value=[_RAW_ARTICLE]),
            patch("src.agents.sports_agent.espn_client.fetch_scores",
                  return_value=_ESPN_RAW),
        ):
            agent = SportsNewsAgent(settings=_SETTINGS)
            result = agent.fetch("hockey")

        assert result.success is True
        assert result.agent == AgentName.SPORTS
        assert result.category == NewsCategory.SPORTS
        providers = [a.provider for a in result.articles]
        assert "guardian" in providers
        assert "espn" in providers
        # Guardian leads
        assert result.articles[0].provider == "guardian"

    def test_fetch_empty_results_returns_success_with_empty_articles(self):
        """Guardian empty + ESPN empty → success=True, articles=[]."""
        with (
            patch("src.agents.sports_agent.guardian_client.fetch_articles", return_value=[]),
            patch("src.agents.sports_agent.espn_client.fetch_scores", return_value=[]),
        ):
            agent = SportsNewsAgent(settings=_SETTINGS)
            result = agent.fetch("noresults")

        assert result.success is True
        assert result.articles == []

    def test_fetch_http_status_error_guardian_warning_espn_called(self):
        """Guardian HTTP 5xx is logged as warning; ESPN fetch still runs; success=True."""
        fake_request = httpx.Request("GET", "https://content.guardianapis.com/search")
        fake_response = httpx.Response(500, request=fake_request)
        exc = httpx.HTTPStatusError("Server error", request=fake_request, response=fake_response)

        with (
            patch("src.agents.sports_agent.guardian_client.fetch_articles", side_effect=exc),
            patch("src.agents.sports_agent.espn_client.fetch_scores", return_value=_ESPN_RAW),
        ):
            agent = SportsNewsAgent(settings=_SETTINGS)
            result = agent.fetch("NHL")

        assert result.success is True
        assert len(result.articles) >= 1
        assert result.articles[0].provider == "espn"

    def test_both_fail_returns_empty_success(self):
        """Both sources returning empty → success=True, articles=[]."""
        with (
            patch("src.agents.sports_agent.guardian_client.fetch_articles", return_value=[]),
            patch("src.agents.sports_agent.espn_client.fetch_scores", return_value=[]),
        ):
            agent = SportsNewsAgent(settings=_SETTINGS)
            result = agent.fetch("cricket")

        assert result.success is True
        assert result.articles == []

    def test_multiple_guardian_articles_all_normalized(self):
        raw2 = {**_RAW_ARTICLE, "webTitle": "Premier League Update",
                "webUrl": "https://theguardian.com/pl"}
        with (
            patch("src.agents.sports_agent.guardian_client.fetch_articles",
                  return_value=[_RAW_ARTICLE, raw2]),
            patch("src.agents.sports_agent.espn_client.fetch_scores", return_value=[]),
        ):
            agent = SportsNewsAgent(settings=_SETTINGS)
            result = agent.fetch("football")

        guardian_articles = [a for a in result.articles if a.provider == "guardian"]
        assert len(guardian_articles) == 2
        assert guardian_articles[0].article_id == "guardian_0"
        assert guardian_articles[1].article_id == "guardian_1"

    def test_espn_fetch_receives_limit_from_settings(self):
        """SportsNewsAgent passes settings.max_articles_per_agent as limit to ESPN."""
        with (
            patch("src.agents.sports_agent.guardian_client.fetch_articles", return_value=[]),
            patch("src.agents.sports_agent.espn_client.fetch_scores", return_value=[]) as mock_espn,
        ):
            agent = SportsNewsAgent(settings=_SETTINGS)
            agent.fetch("NHL")

        mock_espn.assert_called_once_with("NHL", limit=_SETTINGS.max_articles_per_agent)

    def test_latency_ms_is_non_negative(self):
        with (
            patch("src.agents.sports_agent.guardian_client.fetch_articles", return_value=[]),
            patch("src.agents.sports_agent.espn_client.fetch_scores", return_value=[]),
        ):
            agent = SportsNewsAgent(settings=_SETTINGS)
            result = agent.fetch("sports")

        assert result.latency_ms >= 0

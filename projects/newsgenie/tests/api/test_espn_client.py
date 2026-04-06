"""Tests for src/api/espn_client.py."""

import httpx
import respx

from src.api.espn_client import _leagues_for_query, fetch_scores

_BASE = "https://site.api.espn.com/apis/site/v2/sports"

_EVENT = {
    "id": "401803529",
    "shortName": "DAL @ BOS",
    "date": "2026-04-01T00:00:00Z",
    "links": [{"href": "https://espn.com/game/401803529"}],
    "competitions": [{
        "status": {"type": {"name": "STATUS_FINAL"}},
        "headlines": [{"shortLinkText": "Bruins win", "description": "BOS 6, DAL 3"}],
        "competitors": [
            {"team": {"displayName": "Boston Bruins", "logo": None}, "score": "6"},
            {"team": {"displayName": "Dallas Stars", "logo": None}, "score": "3"},
        ],
    }],
}


class TestLeaguesForQuery:
    def test_nhl_keyword_returns_hockey(self):
        result = _leagues_for_query("NHL scores today")
        assert ("hockey", "nhl") in result

    def test_nba_keyword_returns_basketball(self):
        result = _leagues_for_query("NBA playoffs")
        assert ("basketball", "nba") in result

    def test_nfl_keyword_returns_football(self):
        result = _leagues_for_query("NFL trade deadline")
        assert ("football", "nfl") in result

    def test_mlb_keyword_returns_baseball(self):
        result = _leagues_for_query("MLB season")
        assert ("baseball", "mlb") in result

    def test_team_name_maps_to_league(self):
        result = _leagues_for_query("Bruins game tonight")
        assert ("hockey", "nhl") in result

    def test_no_match_returns_all_four_leagues(self):
        result = _leagues_for_query("sports news today")
        leagues = {r[1] for r in result}
        assert leagues == {"nfl", "nba", "mlb", "nhl"}

    def test_multiple_keywords_deduplicates(self):
        result = _leagues_for_query("nhl hockey bruins")
        nhl_count = sum(1 for _, slug in result if slug == "nhl")
        assert nhl_count == 1

    def test_case_insensitive(self):
        result = _leagues_for_query("NHL HOCKEY")
        assert ("hockey", "nhl") in result


class TestFetchScores:
    @respx.mock
    def test_200_returns_events(self):
        respx.get(f"{_BASE}/hockey/nhl/scoreboard").mock(
            return_value=httpx.Response(200, json={"events": [_EVENT]})
        )
        # Remaining leagues return empty
        for sport, league in [("football", "nfl"), ("basketball", "nba"), ("baseball", "mlb")]:
            respx.get(f"{_BASE}/{sport}/{league}/scoreboard").mock(
                return_value=httpx.Response(200, json={"events": []})
            )
        events = fetch_scores("NHL scores", date="20260401")
        assert len(events) == 1
        assert events[0]["id"] == "401803529"

    @respx.mock
    def test_http_error_returns_empty(self):
        for sport, league in [("football", "nfl"), ("basketball", "nba"),
                               ("baseball", "mlb"), ("hockey", "nhl")]:
            respx.get(f"{_BASE}/{sport}/{league}/scoreboard").mock(
                return_value=httpx.Response(500, json={})
            )
        events = fetch_scores("sports news today", date="20260401")
        assert events == []

    @respx.mock
    def test_empty_events_returns_empty_list(self):
        for sport, league in [("football", "nfl"), ("basketball", "nba"),
                               ("baseball", "mlb"), ("hockey", "nhl")]:
            respx.get(f"{_BASE}/{sport}/{league}/scoreboard").mock(
                return_value=httpx.Response(200, json={"events": []})
            )
        events = fetch_scores("sports news today", date="20260401")
        assert events == []

    @respx.mock
    def test_deduplicates_across_leagues(self):
        """Same event id from two league endpoints appears only once."""
        respx.get(f"{_BASE}/hockey/nhl/scoreboard").mock(
            return_value=httpx.Response(200, json={"events": [_EVENT]})
        )
        # Simulate duplicate id from another endpoint (contrived but tests dedup)
        dup_event = {**_EVENT}
        respx.get(f"{_BASE}/football/nfl/scoreboard").mock(
            return_value=httpx.Response(200, json={"events": [dup_event]})
        )
        for sport, league in [("basketball", "nba"), ("baseball", "mlb")]:
            respx.get(f"{_BASE}/{sport}/{league}/scoreboard").mock(
                return_value=httpx.Response(200, json={"events": []})
            )
        events = fetch_scores("nhl nfl", date="20260401")
        assert len(events) == 1

    def test_default_date_is_today(self):
        """fetch_scores with no date does not raise and sets a date."""
        from datetime import UTC, datetime
        from unittest.mock import patch

        from src.api import espn_client

        captured = {}

        def fake_fetch_league(sport, league, date, limit, client):
            captured["date"] = date
            return []

        with patch.object(espn_client, "_fetch_league", side_effect=fake_fetch_league):
            fetch_scores("hockey")

        today = datetime.now(UTC).strftime("%Y%m%d")
        assert captured.get("date") == today

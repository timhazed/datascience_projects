"""Tests for src/utils/normalizer.py — raw dict → NormalizedArticle for each provider."""

from datetime import UTC, datetime

from src.data.enums import NewsCategory
from src.utils.normalizer import normalize_espn, normalize_guardian, normalize_newsapi

# ---------------------------------------------------------------------------
# Fixtures — minimal raw dicts matching the shape each API client returns
# ---------------------------------------------------------------------------

_NEWSAPI_RAW = {
    "source": {"id": "bbc-news", "name": "BBC News"},
    "author": "Test Author",
    "title": "S&P 500 hits record high",
    "description": "Markets surged on Fed rate cut hopes.",
    "url": "https://bbc.com/article/sp500",
    "urlToImage": "https://bbc.com/img/sp500.jpg",
    "publishedAt": "2026-03-31T12:00:00Z",
    "content": None,
}

_GUARDIAN_RAW = {
    "id": "sport/2026/mar/31/champions-league",
    "webTitle": "Champions League Results",
    "webUrl": "https://theguardian.com/sport/champions-league",
    "webPublicationDate": "2026-03-31T20:00:00Z",
    "sectionName": "Sport",
    "fields": {
        "headline": "CL Results: Real Madrid Win",
        "trailText": "Real Madrid beat Bayern 3-1.",
        "thumbnail": "https://theguardian.com/img/cl.jpg",
    },
}


# ---------------------------------------------------------------------------
# normalize_newsapi
# ---------------------------------------------------------------------------


class TestNormalizeNewsapi:
    def test_all_fields_mapped(self):
        result = normalize_newsapi(_NEWSAPI_RAW, NewsCategory.BUSINESS, index=0)
        assert result.article_id == "newsapi_0"
        assert result.title == "S&P 500 hits record high"
        assert result.summary == "Markets surged on Fed rate cut hopes."
        assert result.url == "https://bbc.com/article/sp500"
        assert result.source_name == "BBC News"
        assert result.provider == "newsapi"
        assert result.image_url == "https://bbc.com/img/sp500.jpg"

    def test_category_assigned(self):
        business = normalize_newsapi(_NEWSAPI_RAW, NewsCategory.BUSINESS, index=0)
        general = normalize_newsapi(_NEWSAPI_RAW, NewsCategory.GENERAL, index=0)
        assert business.category == NewsCategory.BUSINESS
        assert general.category == NewsCategory.GENERAL

    def test_index_used_in_article_id(self):
        a = normalize_newsapi(_NEWSAPI_RAW, NewsCategory.BUSINESS, index=3)
        assert a.article_id == "newsapi_3"

    def test_null_description_falls_back_to_title(self):
        raw = {**_NEWSAPI_RAW, "description": None}
        result = normalize_newsapi(raw, NewsCategory.BUSINESS, index=0)
        assert result.summary == raw["title"]

    def test_empty_description_falls_back_to_title(self):
        raw = {**_NEWSAPI_RAW, "description": ""}
        result = normalize_newsapi(raw, NewsCategory.BUSINESS, index=0)
        assert result.summary == raw["title"]

    def test_summary_truncated_to_500_chars(self):
        long_desc = "x" * 600
        raw = {**_NEWSAPI_RAW, "description": long_desc}
        result = normalize_newsapi(raw, NewsCategory.BUSINESS, index=0)
        assert len(result.summary) == 500

    def test_null_source_name_falls_back(self):
        raw = {**_NEWSAPI_RAW, "source": None}
        result = normalize_newsapi(raw, NewsCategory.BUSINESS, index=0)
        assert result.source_name == "NewsAPI"

    def test_null_image_url_is_none(self):
        raw = {**_NEWSAPI_RAW, "urlToImage": None}
        result = normalize_newsapi(raw, NewsCategory.BUSINESS, index=0)
        assert result.image_url is None

    def test_published_at_iso_string_parsed(self):
        result = normalize_newsapi(_NEWSAPI_RAW, NewsCategory.BUSINESS, index=0)
        assert result.published_at.year == 2026
        assert result.published_at.month == 3

    def test_null_published_at_falls_back_to_epoch(self):
        raw = {**_NEWSAPI_RAW, "publishedAt": None}
        result = normalize_newsapi(raw, NewsCategory.BUSINESS, index=0)
        assert result.published_at.year == 1970


# ---------------------------------------------------------------------------
# normalize_guardian
# ---------------------------------------------------------------------------


class TestNormalizeGuardian:
    def test_all_fields_mapped(self):
        result = normalize_guardian(_GUARDIAN_RAW, NewsCategory.SPORTS, index=0)
        assert result.article_id == "guardian_0"
        assert result.title == "CL Results: Real Madrid Win"
        assert result.summary == "Real Madrid beat Bayern 3-1."
        assert result.url == "https://theguardian.com/sport/champions-league"
        assert result.source_name == "Sport"
        assert result.provider == "guardian"
        assert result.image_url == "https://theguardian.com/img/cl.jpg"

    def test_category_assigned(self):
        result = normalize_guardian(_GUARDIAN_RAW, NewsCategory.SPORTS, index=0)
        assert result.category == NewsCategory.SPORTS

    def test_index_used_in_article_id(self):
        result = normalize_guardian(_GUARDIAN_RAW, NewsCategory.SPORTS, index=2)
        assert result.article_id == "guardian_2"

    def test_headline_preferred_over_web_title(self):
        result = normalize_guardian(_GUARDIAN_RAW, NewsCategory.SPORTS, index=0)
        assert result.title == "CL Results: Real Madrid Win"
        assert result.title != "Champions League Results"

    def test_no_headline_falls_back_to_web_title(self):
        raw = {**_GUARDIAN_RAW, "fields": {"trailText": "Some text."}}
        result = normalize_guardian(raw, NewsCategory.SPORTS, index=0)
        assert result.title == "Champions League Results"

    def test_null_trail_text_falls_back_to_title(self):
        raw = {**_GUARDIAN_RAW, "fields": {"headline": "CL Results: Real Madrid Win"}}
        result = normalize_guardian(raw, NewsCategory.SPORTS, index=0)
        assert result.summary == "CL Results: Real Madrid Win"

    def test_no_fields_key_uses_web_title_for_title_and_summary(self):
        raw = {**_GUARDIAN_RAW, "fields": None}
        result = normalize_guardian(raw, NewsCategory.SPORTS, index=0)
        assert result.title == "Champions League Results"
        assert result.summary == "Champions League Results"

    def test_summary_truncated_to_500_chars(self):
        raw = {**_GUARDIAN_RAW, "fields": {"trailText": "y" * 600}}
        result = normalize_guardian(raw, NewsCategory.SPORTS, index=0)
        assert len(result.summary) == 500

    def test_no_thumbnail_is_none(self):
        raw = {**_GUARDIAN_RAW, "fields": {**_GUARDIAN_RAW["fields"], "thumbnail": None}}
        result = normalize_guardian(raw, NewsCategory.SPORTS, index=0)
        assert result.image_url is None

    def test_published_at_iso_string_parsed(self):
        result = normalize_guardian(_GUARDIAN_RAW, NewsCategory.SPORTS, index=0)
        assert result.published_at.year == 2026


# ---------------------------------------------------------------------------
# normalize_espn
# ---------------------------------------------------------------------------

_ESPN_FINAL = {
    "id": "401547378",
    "shortName": "SF VS KC",
    "date": "2024-02-11T23:30:00Z",
    "links": [{"href": "https://espn.com/game/401547378"}],
    "competitions": [{
        "status": {"type": {"name": "STATUS_FINAL"}},
        "headlines": [{"shortLinkText": "KC wins Super Bowl",
                       "description": "KC beats SF 25-22"}],
        "competitors": [
            {"team": {"displayName": "KC Chiefs", "logo": "https://a.espncdn.com/kc.png"},
             "score": "25"},
            {"team": {"displayName": "SF 49ers", "logo": None}, "score": "22"},
        ],
    }],
}

_ESPN_SCHEDULED = {
    "id": "998",
    "shortName": "MTL @ TB",
    "date": "2026-04-02T00:00:00Z",
    "links": [],
    "competitions": [{
        "status": {"type": {"name": "STATUS_SCHEDULED"}},
        "headlines": [],
        "competitors": [
            {"team": {"displayName": "Montreal Canadiens", "logo": None}, "score": "0"},
            {"team": {"displayName": "Tampa Bay Lightning", "logo": None}, "score": "0"},
        ],
    }],
}


class TestNormalizeEspn:
    def test_status_final_with_headlines(self):
        result = normalize_espn(_ESPN_FINAL, index=0)
        assert result.article_id == "espn_0"
        assert result.title == "KC wins Super Bowl"
        assert result.summary == "KC beats SF 25-22"
        assert result.url == "https://espn.com/game/401547378"
        assert result.source_name == "ESPN"
        assert result.provider == "espn"
        assert result.image_url == "https://a.espncdn.com/kc.png"
        assert result.category == NewsCategory.SPORTS

    def test_status_final_no_headlines_falls_back_to_short_name(self):
        event = {**_ESPN_FINAL, "competitions": [{
            **_ESPN_FINAL["competitions"][0],
            "headlines": [],
        }]}
        result = normalize_espn(event, index=1)
        assert result.title == "SF VS KC"
        assert result.article_id == "espn_1"

    def test_status_scheduled_uses_short_name_and_score_string(self):
        result = normalize_espn(_ESPN_SCHEDULED, index=0)
        assert result.title == "MTL @ TB"
        assert "Montreal Canadiens" in result.summary or "0" in result.summary

    def test_empty_competitions_does_not_raise(self):
        event = {"id": "0", "shortName": "X @ Y", "date": "2026-04-01T00:00:00Z",
                 "links": [], "competitions": []}
        result = normalize_espn(event, index=0)
        assert result.title == "X @ Y"
        assert result.provider == "espn"

    def test_missing_fields_fallback_to_defaults(self):
        result = normalize_espn({}, index=0)
        assert result.title == ""
        assert result.provider == "espn"
        assert result.source_name == "ESPN"
        assert result.published_at == datetime.fromtimestamp(0, tz=UTC)

    def test_index_used_in_article_id(self):
        result = normalize_espn(_ESPN_FINAL, index=7)
        assert result.article_id == "espn_7"

    def test_published_at_parsed_as_datetime(self):
        result = normalize_espn(_ESPN_FINAL, index=0)
        assert result.published_at.year == 2024
        assert result.published_at.month == 2

    def test_summary_truncated_to_500_chars(self):
        long_desc = "x" * 600
        event = {**_ESPN_FINAL, "competitions": [{
            **_ESPN_FINAL["competitions"][0],
            "headlines": [{"shortLinkText": "Title", "description": long_desc}],
        }]}
        result = normalize_espn(event, index=0)
        assert len(result.summary) == 500

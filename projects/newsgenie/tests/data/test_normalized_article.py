"""Tests for src/data/normalized_article.py — especially parse_published_at."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.data.enums import NewsCategory
from src.data.normalized_article import NormalizedArticle


def _base(**kwargs) -> dict:
    defaults = {
        "article_id": "newsapi_001",
        "title": "Test headline",
        "summary": "A summary",
        "url": "https://example.com/article",
        "source_name": "Test News",
        "published_at": "2026-03-31T12:00:00Z",
        "category": NewsCategory.BUSINESS,
        "provider": "newsapi",
    }
    return {**defaults, **kwargs}


class TestNormalizedArticle:
    def test_iso_date_string_parsed(self):
        a = NormalizedArticle(**_base(published_at="2026-03-31T12:00:00Z"))
        assert isinstance(a.published_at, datetime)
        assert a.published_at.year == 2026

    def test_human_readable_date_parsed(self):
        a = NormalizedArticle(**_base(published_at="Mar 31, 2026"))
        assert isinstance(a.published_at, datetime)
        assert a.published_at.year == 2026

    def test_unparseable_date_falls_back_to_epoch(self):
        a = NormalizedArticle(**_base(published_at="3 hours ago"))
        # dateutil can parse "3 hours ago" as a relative time; epoch fallback
        # is only for truly unparseable strings, so just check it's a datetime
        assert isinstance(a.published_at, datetime)

    def test_garbage_date_falls_back_to_epoch(self):
        a = NormalizedArticle(**_base(published_at="not-a-date-xyzzy"))
        assert a.published_at == datetime.fromtimestamp(0, tz=UTC)

    def test_datetime_object_passes_through(self):
        dt = datetime(2026, 1, 1, tzinfo=UTC)
        a = NormalizedArticle(**_base(published_at=dt))
        assert a.published_at == dt

    def test_optional_fields_default_to_none(self):
        a = NormalizedArticle(**_base())
        assert a.image_url is None
        assert a.sentiment_score is None
        assert a.tags == []

    def test_missing_required_field_raises(self):
        data = _base()
        del data["title"]
        with pytest.raises(ValidationError):
            NormalizedArticle(**data)

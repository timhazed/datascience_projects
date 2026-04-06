"""Tests for src/data/user_query.py — sanitize_text validator and field constraints."""

import pytest
from pydantic import ValidationError

from src.data.enums import NewsCategory
from src.data.user_query import UserQuery


class TestUserQuerySanitizeText:
    def test_valid_text_is_stripped(self):
        q = UserQuery(text="  hello world  ", session_id="s1")
        assert q.text == "hello world"

    def test_empty_string_raises(self):
        with pytest.raises(ValidationError):
            UserQuery(text="", session_id="s1")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValidationError):
            UserQuery(text="   ", session_id="s1")

    def test_special_chars_only_raises(self):
        with pytest.raises(ValidationError):
            UserQuery(text="!@#$%^&*()", session_id="s1")

    def test_text_exceeding_max_length_raises(self):
        with pytest.raises(ValidationError):
            UserQuery(text="a" * 1001, session_id="s1")

    def test_text_at_max_length_is_accepted(self):
        q = UserQuery(text="a" * 1000, session_id="s1")
        assert len(q.text) == 1000

    def test_alphanumeric_in_special_chars_passes(self):
        q = UserQuery(text="hello!", session_id="s1")
        assert q.text == "hello!"


class TestUserQueryCategories:
    def test_default_categories_is_empty_list(self):
        q = UserQuery(text="news", session_id="s1")
        assert q.categories == []

    def test_categories_accepted(self):
        q = UserQuery(
            text="news",
            session_id="s1",
            categories=[NewsCategory.BUSINESS, NewsCategory.SPORTS],
        )
        assert len(q.categories) == 2

    def test_session_id_stored(self):
        q = UserQuery(text="news", session_id="abc-123")
        assert q.session_id == "abc-123"

"""Tests for src/ui/article_card.py — render_article_card."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from src.data.enums import NewsCategory
from src.data.normalized_article import NormalizedArticle
from src.ui.article_card import render_article_card


def _article(image_url: str | None = None) -> NormalizedArticle:
    return NormalizedArticle(
        article_id="newsapi_1",
        title="Test Headline",
        summary="A summary of the article.",
        url="https://example.com/article",
        source_name="TestSource",
        published_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        category=NewsCategory.BUSINESS,
        provider="newsapi",
        image_url=image_url,
    )


class TestRenderArticleCard:
    def test_renders_title_as_link(self):
        """Article title is rendered as a markdown link."""
        article = _article()
        mock_st = MagicMock()

        with patch("src.ui.article_card.st", mock_st):
            mock_st.columns.return_value = (MagicMock(), MagicMock())
            render_article_card(article)

        # Find any markdown call containing the title and URL
        all_md_calls = [
            str(c) for c in mock_st.mock_calls if "markdown" in str(c).lower()
        ]
        assert any("Test Headline" in c for c in all_md_calls)

    def test_renders_image_when_present(self):
        """st.image is called when image_url is set."""
        article = _article(image_url="https://example.com/img.jpg")
        mock_st = MagicMock()
        col1, col2 = MagicMock(), MagicMock()

        with patch("src.ui.article_card.st", mock_st):
            mock_st.columns.return_value = (col1, col2)
            col1.__enter__ = lambda s: col1
            col1.__exit__ = MagicMock(return_value=False)
            col2.__enter__ = lambda s: col2
            col2.__exit__ = MagicMock(return_value=False)
            render_article_card(article)

        mock_st.image.assert_called_once_with(
            "https://example.com/img.jpg",
            width=220,
            use_container_width=False,
        )

    def test_renders_no_image_caption_when_absent(self):
        """st.caption('No image') is called when image_url is None."""
        article = _article(image_url=None)
        mock_st = MagicMock()
        col1, col2 = MagicMock(), MagicMock()

        with patch("src.ui.article_card.st", mock_st):
            mock_st.columns.return_value = (col1, col2)
            col1.__enter__ = lambda s: col1
            col1.__exit__ = MagicMock(return_value=False)
            col2.__enter__ = lambda s: col2
            col2.__exit__ = MagicMock(return_value=False)
            render_article_card(article)

        mock_st.caption.assert_any_call("No image")

    def test_columns_ratio_is_1_5(self):
        """Two columns are created with [1, 5] ratio (narrower image rail)."""
        article = _article()
        mock_st = MagicMock()

        with patch("src.ui.article_card.st", mock_st):
            mock_st.columns.return_value = (MagicMock(), MagicMock())
            render_article_card(article)

        mock_st.columns.assert_called_once_with([1, 5])

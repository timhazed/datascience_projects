"""Smoke tests for the five thin page modules.

Verifies that each page() function is callable without raising when all Streamlit
widget calls are mocked. No live Streamlit server or MCP server required.

Note: ui.pages.sync_repository is intentionally excluded here — it has its own
dedicated test file (tests/ui/pages/test_sync_repository.py) that exercises both
the pure display helpers and the page() renderer via TestPageCallable.
"""

from unittest.mock import MagicMock, patch


def _make_st_mock(session_state: dict | None = None) -> MagicMock:
    """Return a MagicMock wired to behave like a minimal Streamlit module.

    Args:
        session_state: Optional dict to use as st.session_state. Defaults to empty dict.

    Returns:
        MagicMock with common Streamlit attributes pre-configured.
    """
    mock = MagicMock()
    mock.session_state = session_state if session_state is not None else {}
    # form context manager support
    mock.form.return_value.__enter__ = MagicMock(return_value=mock)
    mock.form.return_value.__exit__ = MagicMock(return_value=False)
    mock.form_submit_button.return_value = False
    mock.columns.return_value = [MagicMock(), MagicMock()]
    mock.button.return_value = False
    return mock


class TestQueryMemoryPageSmoke:
    """Smoke test for ui.pages.query_memory.page()."""

    def test_page_callable_no_submission(self) -> None:
        """page() runs without error when form is not submitted."""
        st_mock = _make_st_mock()
        with patch("ui.pages.query_memory.st", st_mock):
            from ui.pages.query_memory import page
            page()


class TestDeveloperPersonaPageSmoke:
    """Smoke test for ui.pages.developer_persona.page()."""

    def test_page_callable_no_submission(self) -> None:
        """page() runs without error when Generate Persona button is not clicked."""
        st_mock = _make_st_mock()
        st_mock.button.return_value = False
        with patch("ui.pages.developer_persona.st", st_mock):
            from ui.pages.developer_persona import page
            page()


class TestAnalyzeDiffPageSmoke:
    """Smoke test for ui.pages.analyze_diff.page()."""

    def test_page_callable_no_submission(self) -> None:
        """page() runs without error when form is not submitted."""
        st_mock = _make_st_mock()
        with patch("ui.pages.analyze_diff.st", st_mock):
            from ui.pages.analyze_diff import page
            page()


class TestGenerateSkillsPageSmoke:
    """Smoke test for ui.pages.generate_skills.page()."""

    def test_page_callable_no_submission(self) -> None:
        """page() runs without error when form is not submitted."""
        st_mock = _make_st_mock()
        with patch("ui.pages.generate_skills.st", st_mock):
            from ui.pages.generate_skills import page
            page()


class TestPiiReviewQueuePageSmoke:
    """Smoke test for ui.pages.pii_review_queue.page()."""

    def test_page_callable_empty_queue(self) -> None:
        """page() runs without error when quarantine queue is empty."""
        st_mock = _make_st_mock()
        with (
            patch("ui.pages.pii_review_queue.st", st_mock),
            patch("ui.pages.pii_review_queue._get_quarantine", return_value=[]),
        ):
            from ui.pages.pii_review_queue import page
            page()
        # Empty queue path calls st.success
        st_mock.success.assert_called_once()

    def test_page_callable_with_items(self) -> None:
        """page() renders item expanders when quarantine returns items."""
        st_mock = _make_st_mock()
        items = [{"doc_id": "d1", "file_path": "src/secret.py", "quarantine_reason": "PII detected"}]
        with (
            patch("ui.pages.pii_review_queue.st", st_mock),
            patch("ui.pages.pii_review_queue._get_quarantine", return_value=items),
        ):
            from ui.pages.pii_review_queue import page
            page()
        # With items, st.markdown is called at least once (item count line)
        st_mock.markdown.assert_called()

    def test_page_shows_error_on_get_quarantine_failure(self) -> None:
        """page() renders error banner when _get_quarantine raises."""
        st_mock = _make_st_mock()
        with (
            patch("ui.pages.pii_review_queue.st", st_mock),
            patch("ui.pages.pii_review_queue._get_quarantine", side_effect=ConnectionError("refused")),
        ):
            from ui.pages.pii_review_queue import page
            page()
        st_mock.error.assert_called_once()

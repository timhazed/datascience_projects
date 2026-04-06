"""Tests for src/cli/cli.py — Click CLI."""

from datetime import UTC, datetime
from unittest.mock import patch

from click.testing import CliRunner

from src.cli.cli import cli
from src.data import AgentName, NewsCategory, NormalizedArticle
from src.data.response import IntentSection, SupervisorResponse

runner = CliRunner()


def _article(title: str = "Test Headline", provider: str = "newsapi") -> NormalizedArticle:
    return NormalizedArticle(
        article_id=f"{provider}_1",
        title=title,
        summary="A short summary.",
        url="https://example.com/article",
        source_name="TestSource",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        category=NewsCategory.BUSINESS,
        provider=provider,
    )


def _response(
    articles: list[NormalizedArticle] | None = None,
    fallback_used: bool = False,
    session_id: str = "test-session",
) -> SupervisorResponse:
    arts = articles if articles is not None else [_article()]
    return SupervisorResponse(
        session_id=session_id,
        sections=[IntentSection(
            agent=AgentName.BUSINESS,
            query="markets today",
            articles=arts,
        )],
        sources_used=sorted({a.provider for a in arts}),
        fallback_used=fallback_used,
    )


class TestCLIQuery:
    def test_successful_query_exits_zero(self):
        """A valid query with mocked graph returns exit code 0."""
        mock_output = {"final_response": _response()}

        with patch("src.cli.cli._get_graph") as mock_graph_fn:
            mock_graph_fn.return_value.invoke.return_value = mock_output
            result = runner.invoke(cli, ["What's in the markets today?"])

        assert result.exit_code == 0

    def test_article_title_in_output(self):
        """Article title must appear in stdout."""
        mock_output = {"final_response": _response([_article("Nvidia hits record high")])}

        with patch("src.cli.cli._get_graph") as mock_graph_fn:
            mock_graph_fn.return_value.invoke.return_value = mock_output
            result = runner.invoke(cli, ["markets"])

        assert "Nvidia hits record high" in result.output

    def test_fallback_warning_shown(self):
        """When fallback_used=True, the warning text appears in output."""
        mock_output = {"final_response": _response(articles=[], fallback_used=True)}

        with patch("src.cli.cli._get_graph") as mock_graph_fn:
            mock_graph_fn.return_value.invoke.return_value = mock_output
            result = runner.invoke(cli, ["query"])

        assert "one or more" in result.output.lower()

    def test_empty_articles_no_crash(self):
        """A section with no articles must not raise — just print a 'no results' message."""
        mock_output = {"final_response": _response(articles=[])}

        with patch("src.cli.cli._get_graph") as mock_graph_fn:
            mock_graph_fn.return_value.invoke.return_value = mock_output
            result = runner.invoke(cli, ["query"])

        assert result.exit_code == 0

    def test_invalid_query_exits_nonzero(self):
        """A query with only special characters must exit with code 1."""
        result = runner.invoke(cli, ["!!!"])
        assert result.exit_code != 0

    def test_graph_exception_exits_nonzero(self):
        """If the graph raises, the CLI exits with code 1."""
        with patch("src.cli.cli._get_graph") as mock_graph_fn:
            mock_graph_fn.return_value.invoke.side_effect = RuntimeError("graph failure")
            result = runner.invoke(cli, ["markets"])

        assert result.exit_code != 0

    def test_session_id_option_accepted(self):
        """Passing --session-id is accepted and does not cause an error."""
        mock_output = {"final_response": _response(session_id="custom-session")}

        with patch("src.cli.cli._get_graph") as mock_graph_fn:
            mock_graph_fn.return_value.invoke.return_value = mock_output
            result = runner.invoke(cli, ["markets", "--session-id", "custom-session"])

        assert result.exit_code == 0

    def test_session_id_short_option_accepted(self):
        """Passing -s short form is accepted."""
        mock_output = {"final_response": _response(session_id="s123")}

        with patch("src.cli.cli._get_graph") as mock_graph_fn:
            mock_graph_fn.return_value.invoke.return_value = mock_output
            result = runner.invoke(cli, ["markets", "-s", "s123"])

        assert result.exit_code == 0

    def test_session_id_forwarded_to_state(self):
        """The session_id passed via --session-id must reach the graph state."""
        from src.data.state import AgentState

        captured: list[AgentState] = []

        def capture(state):
            captured.append(state)
            return {"final_response": _response(session_id="my-sid")}

        with patch("src.cli.cli._get_graph") as mock_graph_fn:
            mock_graph_fn.return_value.invoke.side_effect = capture
            runner.invoke(cli, ["markets", "--session-id", "my-sid"])

        assert captured and captured[0].query.session_id == "my-sid"

    def test_auto_generated_session_id_when_none_provided(self):
        """If no session_id is given, one is auto-generated (non-empty UUID string)."""
        from src.data.state import AgentState

        captured: list[AgentState] = []

        def capture(state):
            captured.append(state)
            return {"final_response": _response()}

        with patch("src.cli.cli._get_graph") as mock_graph_fn:
            mock_graph_fn.return_value.invoke.side_effect = capture
            runner.invoke(cli, ["markets"])

        assert captured and len(captured[0].query.session_id) > 0

    def test_sources_used_in_output(self):
        """Sources list must appear in the output."""
        mock_output = {"final_response": _response([_article(provider="newsapi")])}

        with patch("src.cli.cli._get_graph") as mock_graph_fn:
            mock_graph_fn.return_value.invoke.return_value = mock_output
            result = runner.invoke(cli, ["markets"])

        assert "newsapi" in result.output

    def test_multiple_sections_rendered(self):
        """A multi-section response renders both section queries."""
        response = SupervisorResponse(
            session_id="s1",
            sections=[
                IntentSection(
                    agent=AgentName.BUSINESS,
                    query="markets update",
                    articles=[_article("Biz headline", "newsapi")],
                ),
                IntentSection(
                    agent=AgentName.SPORTS,
                    query="NBA scores",
                    articles=[_article("Sports headline", "guardian")],
                ),
            ],
            sources_used=["guardian", "newsapi"],
            fallback_used=False,
        )

        with patch("src.cli.cli._get_graph") as mock_graph_fn:
            mock_graph_fn.return_value.invoke.return_value = {"final_response": response}
            result = runner.invoke(cli, ["markets and NBA"])

        assert "markets update" in result.output
        assert "NBA scores" in result.output

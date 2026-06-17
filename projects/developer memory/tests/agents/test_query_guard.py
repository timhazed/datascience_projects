"""Tests for make_query_guard_node (src/agents/query_guard.py).

No external dependencies — pure validation logic.
Coverage target: ≥ 95% (security boundary per spec §10).

Cases:
  - Valid query → query_safe=True
  - Empty string → query_safe=False
  - Whitespace-only → query_safe=False
  - Query > 2000 chars → query_safe=False
  - Shell metacharacter ; → query_safe=False
  - Shell metacharacter | → query_safe=False
  - Shell metacharacter && → query_safe=False
  - Shell metacharacter || → query_safe=False
  - Shell metacharacter > → query_safe=False
  - Shell metacharacter < → query_safe=False
  - Trace entry added on both accept and reject
  - Error message never contains internal state
"""

from src.agents.query_guard import make_query_guard_node


def _state(query: str) -> dict:
    return {"query": query, "query_safe": None, "trace": []}


class TestQueryGuardNode:
    def test_valid_query_accepted(self) -> None:
        """A valid natural-language query sets query_safe=True."""
        node = make_query_guard_node()
        result = node(_state("How are FastAPI routes structured in this repo?"))

        assert result["query_safe"] is True
        assert "error" not in result or result.get("error") is None

    def test_empty_query_rejected(self) -> None:
        """Empty string → query_safe=False."""
        node = make_query_guard_node()
        result = node(_state(""))

        assert result["query_safe"] is False
        assert result.get("error")

    def test_whitespace_only_rejected(self) -> None:
        """Whitespace-only query → query_safe=False."""
        node = make_query_guard_node()
        result = node(_state("   \t\n  "))

        assert result["query_safe"] is False

    def test_query_exceeding_length_limit_rejected(self) -> None:
        """Query > 2000 chars → query_safe=False."""
        node = make_query_guard_node()
        result = node(_state("x" * 2001))

        assert result["query_safe"] is False
        assert "2000" in result["error"]

    def test_query_at_exact_length_limit_accepted(self) -> None:
        """Query of exactly 2000 chars → query_safe=True."""
        node = make_query_guard_node()
        result = node(_state("x" * 2000))

        assert result["query_safe"] is True

    def test_semicolon_rejected(self) -> None:
        """Query with semicolon → query_safe=False (shell metacharacter)."""
        node = make_query_guard_node()
        result = node(_state("show me code; rm -rf /"))

        assert result["query_safe"] is False

    def test_pipe_rejected(self) -> None:
        """Query with | → query_safe=False."""
        node = make_query_guard_node()
        result = node(_state("code | grep secret"))

        assert result["query_safe"] is False

    def test_double_ampersand_rejected(self) -> None:
        """Query with && → query_safe=False."""
        node = make_query_guard_node()
        result = node(_state("find code && execute"))

        assert result["query_safe"] is False

    def test_double_pipe_rejected(self) -> None:
        """Query with || → query_safe=False."""
        node = make_query_guard_node()
        result = node(_state("search || fallback"))

        assert result["query_safe"] is False

    def test_redirect_out_rejected(self) -> None:
        """Query with > → query_safe=False."""
        node = make_query_guard_node()
        result = node(_state("write output > file.txt"))

        assert result["query_safe"] is False

    def test_redirect_in_rejected(self) -> None:
        """Query with < → query_safe=False."""
        node = make_query_guard_node()
        result = node(_state("read < /etc/passwd"))

        assert result["query_safe"] is False

    def test_trace_added_on_accept(self) -> None:
        """Trace gains one entry on acceptance."""
        node = make_query_guard_node()
        result = node(_state("valid query"))

        assert len(result["trace"]) >= 1
        assert any("accept" in t for t in result["trace"])

    def test_trace_added_on_reject(self) -> None:
        """Trace gains one entry on rejection."""
        node = make_query_guard_node()
        result = node(_state(""))

        assert len(result["trace"]) >= 1
        assert any("reject" in t for t in result["trace"])

    def test_backtick_rejected(self) -> None:
        """Query with backtick → query_safe=False (command substitution vector)."""
        node = make_query_guard_node()
        result = node(_state("show me `whoami`"))

        assert result["query_safe"] is False

    def test_dollar_sign_rejected(self) -> None:
        """Query with $ → query_safe=False (variable expansion / $() substitution)."""
        node = make_query_guard_node()
        result = node(_state("find $HOME secrets"))

        assert result["query_safe"] is False

    def test_newline_rejected(self) -> None:
        """Query containing a newline → query_safe=False (multi-command injection vector)."""
        node = make_query_guard_node()
        result = node(_state("valid query\nignore above"))

        assert result["query_safe"] is False

    def test_error_message_sanitized(self) -> None:
        """User-facing error messages contain no internal state or stack traces."""
        node = make_query_guard_node()
        result = node(_state(";drop tables"))

        error = result.get("error", "")
        assert "Traceback" not in error
        assert "Exception" not in error
        assert len(error) < 200  # sanity: not dumping internals

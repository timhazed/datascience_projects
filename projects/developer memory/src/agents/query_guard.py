"""query_guard LangGraph node — Spec §2.2, §6.2.

Validates the query before any ChromaDB or LLM interaction. Rejects queries that are:
  - Empty or whitespace-only
  - Longer than 2000 characters (prompt injection risk)
  - Containing unescaped shell metacharacters (; | && || > <)

User-facing error messages are sanitized — no internal state exposed.
"""

import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Shell metacharacters and prompt-injection vectors (§6.2).
# Covers: classic shell operators (;|<>&&||), command substitution (backtick, $()),
# variable expansion ($), and newline-based multi-command injection.
_SHELL_META_RE = re.compile(r"[;|<>`$\n]|&&|\|\|")

_MAX_QUERY_LENGTH = 2000


def make_query_guard_node() -> Callable[[dict], dict]:
    """Return a query_guard node (no injected dependencies — validation is pure logic).

    Returns:
        LangGraph node function that reads state["query"] and writes query_safe=True/False.
        On rejection, sets state["error"] with a safe user-facing message.
    """

    def query_guard(state: dict) -> dict:
        """Validate the query string against all §6.2 rejection criteria.

        Sets query_safe=True if the query passes all checks, False otherwise.
        Always appends a trace entry describing the outcome.
        """
        query: str = state.get("query", "")
        trace: list[str] = list(state.get("trace", []))

        # ── Guard: empty or whitespace-only ──────────────────────────────
        if not query or not query.strip():
            logger.warning("query_guard: rejected — empty query")
            return {
                "query_safe": False,
                "error": "Query rejected: query must not be empty.",
                "trace": trace + ["query_guard: rejected (empty)"],
            }

        # ── Guard: length limit ───────────────────────────────────────────
        if len(query) > _MAX_QUERY_LENGTH:
            logger.warning("query_guard: rejected — query exceeds %d chars", _MAX_QUERY_LENGTH)
            return {
                "query_safe": False,
                "error": f"Query rejected: query exceeds {_MAX_QUERY_LENGTH} character limit.",
                "trace": trace + ["query_guard: rejected (too long)"],
            }

        # ── Guard: shell metacharacters ───────────────────────────────────
        if _SHELL_META_RE.search(query):
            logger.warning("query_guard: rejected — shell metacharacters detected")
            return {
                "query_safe": False,
                "error": "Query rejected: query contains invalid characters.",
                "trace": trace + ["query_guard: rejected (metacharacters)"],
            }

        logger.debug("query_guard: accepted query (len=%d)", len(query))
        return {
            "query_safe": True,
            "trace": trace + ["query_guard: accepted"],
        }

    return query_guard

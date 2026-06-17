"""semantic_searcher LangGraph node — Spec §2.2, §7.

Embeds the query and performs a semantic search against ChromaDB, with two
independent corpus-driven filter inference paths:

1. Project filter — when the query names a known project (e.g. "Chatbot_DataPipeline"),
   results are restricted to that project's file_path prefix. This is the primary routing
   signal for project-scoped queries and replaces the old static _PROJECT_SCOPE_PATTERNS
   suppression list.

2. Tech filter — when the query names a known technology (e.g. "ChromaDB") and no project
   scope is detected, results are filtered by tech_stack label. Suppressed when a project
   filter is active, since the project-level filter already narrows scope sufficiently.

Both filter sets are fetched from ChromaDB once at node construction time and refreshed
lazily after _CORPUS_TTL_SECONDS, so new projects and technologies become matchable
within one TTL window without requiring a server restart.

Runs after query_guard confirms query_safe=True. ChromaDB errors are non-fatal:
they populate state["error"] and return raw_results=[].
"""

import logging
import re
import time
from collections.abc import Callable

from src.db.chroma_client import ChromaLibrarianClient

logger = logging.getLogger(__name__)

_DEFAULT_N_RESULTS = 20

# How long (seconds) corpus snapshots (tech labels + project names) are considered fresh.
# After this interval the next query transparently re-fetches from ChromaDB.
_CORPUS_TTL_SECONDS = 300

# Synonym override: maps a query term (regex, case-insensitive) to the canonical
# tech_stack label in ChromaDB. Only covers aliases where the user's term differs
# from the indexed label and the alias is long enough to be unambiguous.
# Short abbreviations (e.g. "lc", "lg") are deliberately excluded to prevent
# false positives on unrelated words in free-text queries.
_SYNONYM_OVERRIDES: dict[str, str] = {
    r"\bpyautogen\b": "autogen",
}

# Generic English words that appear as tech_stack labels but are too broad to be
# useful as filters. Matching "database" in "where is Qdrant database used?" would
# return Database-tagged docs instead of Qdrant-tagged docs.
_GENERIC_LABEL_BLOCKLIST: frozenset[str] = frozenset({
    "api", "apis", "ai", "cli", "css", "csv", "database", "documentation",
    "graph", "html", "http", "json", "llm", "llms", "logging", "markdown",
    "orm", "os", "pdf", "python", "rest", "sql", "testing", "yaml",
    "utilities", "configuration", "visualization", "mocking",
})


def _infer_project_filter(
    query: str, corpus_project_prefixes: dict[str, str]
) -> tuple[str, str] | None:
    """Infer a project filter from the query using the corpus project prefix map.

    Checks whether any indexed project name appears in the query text
    (case-insensitive). The separator between words in the project name is treated
    flexibly — underscores, hyphens, and spaces all match each other — so
    "Chatbot DataPipeline", "chatbot_datapipeline", and "chatbot-datapipeline"
    all match the indexed name "Chatbot_DataPipeline".

    Longer project names are checked first to prevent a shorter name that is a
    substring of a longer one from matching prematurely (e.g. "Chatbot" before
    "Chatbot_DataPipeline").

    Args:
        query: The raw query string from the user.
        corpus_project_prefixes: Dict of {project_name: file_path_prefix} fetched
            from ChromaDB at startup via get_distinct_project_prefixes().

    Returns:
        (project_name, file_path_prefix) tuple if matched, else None.
    """
    query_lower = query.lower()
    # Sort longest-first so more-specific names match before shorter substrings
    for name in sorted(corpus_project_prefixes, key=len, reverse=True):
        # Build a flexible separator pattern: "Chatbot_DataPipeline" →
        # "Chatbot[_\-\s]+DataPipeline" so natural language variants all match.
        escaped = re.escape(name)
        flexible = re.sub(r"(?:\\[_\-]|_|-)+", r"[_\\-\\s]+", escaped)
        if re.search(flexible, query_lower, re.IGNORECASE):
            prefix = corpus_project_prefixes[name]
            logger.debug("semantic_searcher: inferred project_filter=%s (prefix=%s)", name, prefix)
            return name, prefix
    return None


def _infer_tech_filter(query: str, corpus_tech_labels: list[str]) -> list[str] | None:
    """Infer a tech_filter from query text using corpus labels + synonym overrides.

    Matching order:
      1. Synonym overrides (_SYNONYM_OVERRIDES) — handles aliases like "pyautogen" → "autogen".
      2. Direct token overlap — checks whether any indexed label appears verbatim in the query
         (case-insensitive whole-word match). Zero maintenance as new technologies are indexed.

    Only the first match is returned to avoid over-constraining the filter.

    Args:
        query: The raw query string from the user.
        corpus_tech_labels: Distinct tech_stack values fetched from ChromaDB at startup.

    Returns:
        Single-element list with the matched label, or None.
    """
    query_lower = query.lower()

    # 1. Check synonym overrides first — map user aliases to canonical corpus labels
    for pattern, canonical_label in _SYNONYM_OVERRIDES.items():
        if re.search(pattern, query_lower, re.IGNORECASE) and canonical_label in corpus_tech_labels:
            logger.debug(
                "semantic_searcher: synonym match — inferred tech_filter=[%s]", canonical_label
            )
            return [canonical_label]

    # 2. Direct overlap: whole-word match of each indexed label against the query.
    # Sort longest-first so specific labels ("Qdrant", "ChromaDB") match before
    # generic single-word labels ("Database", "API") that are substrings of the query.
    # Skip labels in the blocklist — they are too broad to narrow results usefully.
    for label in sorted(corpus_tech_labels, key=len, reverse=True):
        if label.lower() in _GENERIC_LABEL_BLOCKLIST:
            continue
        escaped = re.escape(label)
        if re.search(rf"\b{escaped}\b", query_lower, re.IGNORECASE):
            logger.debug(
                "semantic_searcher: corpus label match — inferred tech_filter=[%s]", label
            )
            return [label]

    return None


def make_semantic_searcher_node(chroma: ChromaLibrarianClient) -> Callable[[dict], dict]:
    """Return a semantic_searcher node with an injected ChromaLibrarianClient.

    Fetches distinct tech labels and project names from ChromaDB once at construction
    time. Both snapshots are refreshed lazily after _CORPUS_TTL_SECONDS so new data
    becomes matchable without a server restart.

    Args:
        chroma: ChromaLibrarianClient instance constructed at server startup.

    Returns:
        LangGraph node function that reads state["query"] and state["tech_filter"]
        and writes state["raw_results"].
    """
    # Shared TTL cache for both corpus snapshots. Using a mutable dict avoids the
    # closure rebinding problem — inner functions mutate values without nonlocal.
    corpus_cache: dict = {
        "tech_labels": chroma.get_distinct_tech_stack(),
        "project_prefixes": chroma.get_distinct_project_prefixes(),
        "fetched_at": time.monotonic(),
    }
    logger.debug(
        "semantic_searcher: loaded %d tech labels, %d projects from corpus",
        len(corpus_cache["tech_labels"]),
        len(corpus_cache["project_prefixes"]),
    )

    def _refresh_corpus_if_stale() -> None:
        """Re-fetch both corpus snapshots if the TTL has expired."""
        if time.monotonic() - corpus_cache["fetched_at"] > _CORPUS_TTL_SECONDS:
            corpus_cache["tech_labels"] = chroma.get_distinct_tech_stack()
            corpus_cache["project_prefixes"] = chroma.get_distinct_project_prefixes()
            corpus_cache["fetched_at"] = time.monotonic()
            logger.debug(
                "semantic_searcher: refreshed corpus (%d tech, %d projects)",
                len(corpus_cache["tech_labels"]),
                len(corpus_cache["project_prefixes"]),
            )

    def semantic_searcher(state: dict) -> dict:
        """Embed the query and retrieve matching documents from ChromaDB.

        Filter inference priority:
          1. Explicit tech_filter in state — used as-is, no inference.
          2. Project filter inferred from query — restricts results to one project's
             file_path prefix. Suppresses tech filter (project scope is sufficient).
          3. Tech filter inferred from query — applied when no project is detected.

        Post-retrieval reranking:
          - When a project filter is active: results are post-filtered to that project's prefix.
          - When a tech filter is active (no project scope): implementation files
            (path starts with "projects/") are promoted before root-level or methodology
            docs to ensure actual usage locations rank above cross-cutting summaries.

        Returns raw_results as a list of ChromaDB result dicts with deserialized tech_stack.
        """
        query: str = state.get("query", "")
        tech_filter: list[str] | None = state.get("tech_filter")
        trace: list[str] = list(state.get("trace", []))

        _refresh_corpus_if_stale()

        # Infer project filter first — takes priority over tech filter.
        # Returns (name, prefix) or None.
        project_match = _infer_project_filter(query, corpus_cache["project_prefixes"])
        if project_match:
            _, project_prefix = project_match
            trace = trace + [f"semantic_searcher: inferred project_filter={project_match[0]}"]

        # Infer tech filter only when no project scope and no explicit filter given.
        # Project scope is sufficient to narrow results; adding a tech filter on top
        # would over-constrain and could drop relevant chunks.
        if not tech_filter and not project_match:
            tech_filter = _infer_tech_filter(query, corpus_cache["tech_labels"])
            if tech_filter:
                trace = trace + [f"semantic_searcher: inferred tech_filter={tech_filter}"]

        # When project filter active, fetch a larger candidate set so that after
        # post-filtering by file_path prefix we still return _DEFAULT_N_RESULTS.
        fetch_n = _DEFAULT_N_RESULTS * 5 if project_match else _DEFAULT_N_RESULTS

        try:
            results = chroma.query(
                text=query,
                tech_filter=tech_filter,
                n_results=fetch_n,
            )
        except Exception as exc:
            logger.error(
                "semantic_searcher failed [%s]: %s", type(exc).__name__, str(exc)[:200]
            )
            return {
                "raw_results": [],
                "error": "Search failed. Please retry.",
                "trace": trace + ["semantic_searcher: error"],
            }

        # Post-filter by project file_path prefix — ChromaDB has no native prefix
        # filter on free-text metadata fields, so this runs in Python after retrieval.
        if project_match:
            results = [
                r for r in results
                if r.get("metadata", {}).get("file_path", "").startswith(project_prefix)
            ][:_DEFAULT_N_RESULTS]

        # When a tech filter is active but no project scope was detected, cross-cutting
        # overview docs (PROJECTS.md, README.md, methodology/) mention many technologies
        # in summary context and rank highly by cosine similarity even though they are
        # not the actual usage location. Re-rank by promoting implementation files
        # (path starts with "projects/") before root-level or methodology docs so that
        # queries like "where is Qdrant used?" return actual code locations first.
        if tech_filter and not project_match:
            impl = [
                r for r in results
                if r.get("metadata", {}).get("file_path", "").startswith("projects/")
            ]
            overview = [
                r for r in results
                if not r.get("metadata", {}).get("file_path", "").startswith("projects/")
            ]
            results = (impl + overview)[:_DEFAULT_N_RESULTS]
            if impl or overview:
                logger.debug(
                    "semantic_searcher: tech-filter rerank — %d impl, %d overview",
                    len(impl),
                    len(overview),
                )

        logger.debug("semantic_searcher: %d results for query (len=%d)", len(results), len(query))
        return {
            "raw_results": results,
            "trace": trace + [f"semantic_searcher: {len(results)} results"],
        }

    return semantic_searcher

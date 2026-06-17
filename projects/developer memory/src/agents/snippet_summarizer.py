"""snippet_summarizer LangGraph node — Spec §2.2, §7.

Synthesizes ChromaDB search results into a coherent developer-facing answer using
the Extract→Contrast→Synthesize prompt (validated in Appendix D). LLM errors are
non-fatal — pipeline returns final_answer=None with error surfaced.
"""

import logging
from collections.abc import Callable

from langchain_ollama import ChatOllama

from src.chains.snippet_summary_chain import build_snippet_summary_chain
from src.utils.retry import invoke_with_retry

logger = logging.getLogger(__name__)


def make_snippet_summarizer_node(llm: ChatOllama) -> Callable[[dict], dict]:
    """Return a snippet_summarizer node with an injected ChatOllama.

    Args:
        llm: ChatOllama instance constructed at server startup. Temperature 0.1 per §5.

    Returns:
        LangGraph node function that reads state["query"] and state["raw_results"]
        and writes state["final_answer"].
    """
    chain = build_snippet_summary_chain(llm)

    def snippet_summarizer(state: dict) -> dict:
        """Synthesize raw ChromaDB results into a user-facing answer.

        Formats raw_results as a plain-text block for the human turn of the prompt.
        External data (query, results) stays in the human turn — never the system prompt.
        """
        query: str = state.get("query", "")
        raw_results: list[dict] = state.get("raw_results", [])
        trace: list[str] = list(state.get("trace", []))

        # Format results for the human turn — file path + intent summary + snippet
        results_text = _format_results(raw_results)

        try:
            answer: str = invoke_with_retry(chain, {"query": query, "results": results_text})
        except Exception as exc:
            logger.error(
                "snippet_summarizer failed [%s]: %s", type(exc).__name__, str(exc)[:200]
            )
            return {
                "final_answer": None,
                "error": "Summarization failed. Please retry.",
                "trace": trace + ["snippet_summarizer: error"],
            }

        logger.debug("snippet_summarizer: answer generated (%d chars)", len(answer))
        return {
            "final_answer": answer,
            "trace": trace + ["snippet_summarizer: ok"],
        }

    return snippet_summarizer


def _format_results(results: list[dict]) -> str:
    """Format ChromaDB result dicts as a plain-text block for prompt injection.

    Each result contributes its file path, intent_summary, key_identifiers, and a
    content excerpt. key_identifiers are included so the synthesizer can cite exact
    class and function names rather than paraphrasing them.
    This output goes into the human turn of the prompt — not the system prompt.
    """
    if not results:
        return "(No results found.)"

    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        meta = r.get("metadata", {})
        path = meta.get("file_path", "unknown")
        summary = meta.get("intent_summary", "")
        key_ids = meta.get("key_identifiers", [])
        # key_identifiers may be a list (deserialized) or a JSON string (not yet unpacked)
        if isinstance(key_ids, str):
            import json
            key_ids = json.loads(key_ids)
        identifiers_str = ", ".join(key_ids) if key_ids else ""
        doc = r.get("document", "")[:500]  # increased from 300 to preserve more code context
        entry = f"Result {i} [{path}]:\n  Intent: {summary}"
        if identifiers_str:
            entry += f"\n  Identifiers: {identifiers_str}"
        entry += f"\n  Code:\n{doc}\n"
        lines.append(entry)

    return "\n".join(lines)

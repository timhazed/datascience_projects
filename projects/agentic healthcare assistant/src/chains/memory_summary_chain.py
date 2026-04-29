"""memory_summary_chain: rolling summary rollup for long-term patient memory.

Chain: memory_summary_prompt | llm | StrOutputParser

Input variables:
    prior_summary: existing rolling summary from SQLite (may be empty).
    recent_turns: formatted recent chat lines (Role: text).
    max_chars: hard cap guidance for output length (typically max_summary_chars).
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from src.chains.chain_prompts import _MEMORY_SUMMARY_PROMPT


def build_memory_summary_chain(llm: BaseChatModel) -> Runnable:
    """Build the rolling memory-summary compression chain.

    Args:
        llm: Chat model using ``temperature_summarizer`` and ``max_tokens_memory_summary``.

    Returns:
        Runnable accepting ``prior_summary``, ``recent_turns``, and ``max_chars``; returns a string.
    """
    return _MEMORY_SUMMARY_PROMPT | llm | StrOutputParser()

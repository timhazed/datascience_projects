"""summary_chain — synthesises all completed task results into a final response.

Chain: summary_prompt | llm | StrOutputParser

Input variables:
    completed_tasks: str — JSON-serialised list of TaskResult dicts
    user_query: str — the original user request
    conversation_memory: str — bounded prior chat (summary + last N turns); may be empty

This is the terminal chain in the graph — it runs after all sub-goals
are complete and composes the final user-facing response from all
TaskResult outcomes. Failed tasks are surfaced clearly rather than silently
dropped.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from src.chains.chain_prompts import _SUMMARY_PROMPT


def build_summary_chain(llm: BaseChatModel) -> Runnable:
    """Build the final response synthesis chain.

    Args:
        llm: LangChain-compatible chat model. Should use a higher max_tokens
            budget (e.g. 700) — the output is a multi-task synthesis that may
            span several result types (history, appointment, search).

    Returns:
        Runnable chain that accepts ``completed_tasks``, ``user_query``, and
        ``conversation_memory`` and returns the final user-facing response string.
    """
    return _SUMMARY_PROMPT | llm | StrOutputParser()

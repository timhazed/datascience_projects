"""search_chain — synthesises web search results into a clinical summary.

Chain: search_summary_prompt | llm | StrOutputParser

Input variables:
    search_results: str — formatted web results (title + snippet per result)
    query:          str — original disease/treatment search query

The chain is used by disease_search_node. The node performs the web search
(Serper/SerpAPI), filters by domain whitelist, formats results, then invokes
this chain to produce the final synthesis.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from src.chains.chain_prompts import _SEARCH_SUMMARY_PROMPT


def build_search_chain(llm: BaseChatModel) -> Runnable:
    """Build the disease search synthesis chain.

    Args:
        llm: LangChain-compatible chat model. Should use a higher max_tokens
            budget (e.g. 650) — the output is a 2-4 paragraph clinical summary
            with inline citations.

    Returns:
        Runnable chain that accepts {"search_results": str, "query": str}
        and returns a string containing the synthesised clinical summary.
    """
    return _SEARCH_SUMMARY_PROMPT | llm | StrOutputParser()

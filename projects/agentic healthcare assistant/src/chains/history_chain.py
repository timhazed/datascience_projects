"""history_chain — answers a clinical query from retrieved patient record chunks.

Chain: history_prompt | llm | StrOutputParser

Input variables:
    retrieved_chunks: str — FAISS similarity search results, concatenated
    query:            str — clinical question about the patient

The chain is used by history_retriever_node. The node performs the FAISS
lookup and formats the chunks before invoking this chain.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from src.chains.chain_prompts import _HISTORY_PROMPT


def build_history_chain(llm: BaseChatModel) -> Runnable:
    """Build the history retrieval summarisation chain.

    Args:
        llm: LangChain-compatible chat model. Should use a moderate max_tokens
            budget (e.g. 400) — the output is a short clinical answer, not a
            full report.

    Returns:
        Runnable chain that accepts {"retrieved_chunks": str, "query": str}
        and returns a string containing the clinical answer.
    """
    return _HISTORY_PROMPT | llm | StrOutputParser()

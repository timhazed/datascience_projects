"""intent_guard_chain — classifies user queries as SAFE or UNSAFE.

Chain: guard_prompt | llm | StrOutputParser | normalizer

The normalizer step strips whitespace and uppercases the output so that
minor LLM formatting variations ("safe", "SAFE  \\n") all resolve to a
canonical "SAFE" or "UNSAFE" string. Normalization is done inside the chain
so callers can do a plain equality check without hidden post-processing
requirements.

Allowed topics: patient medical history, appointment scheduling,
medical information queries, record updates, treatment summaries.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda

from src.chains.chain_prompts import _GUARD_PROMPT

# Normalise the LLM output so callers receive exactly "SAFE" or "UNSAFE".
_normalizer: Runnable = RunnableLambda(lambda s: s.strip().upper())


def build_intent_guard_chain(llm: BaseChatModel) -> Runnable:
    """Build the intent guard classification chain.

    The chain returns a normalised string — either "SAFE" or "UNSAFE" —
    regardless of minor LLM formatting variations. Normalisation is applied
    inside the chain so the graph node can compare with plain equality.

    Args:
        llm: LangChain-compatible chat model instance. Should use a low
            max_tokens budget (e.g. 25) — the only valid outputs are
            "SAFE" or "UNSAFE".

    Returns:
        Runnable chain that accepts {"user_query": str} and returns
        exactly "SAFE" or "UNSAFE" (stripped, uppercased).
    """
    return _GUARD_PROMPT | llm | StrOutputParser() | _normalizer

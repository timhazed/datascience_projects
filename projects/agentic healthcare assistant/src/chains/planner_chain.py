"""planner_chain — decomposes a user query into an ordered list of SubGoals.

Chain: planner_prompt | llm.with_structured_output(PlannerOutput)

Uses structured output (JSON schema mode) to guarantee a typed PlannerOutput.
The planner_node wraps this chain with invoke_with_retry to handle transient
LLM failures.

Security note: patient_context is runtime-resolved data (e.g. a patient name
typed by the user). It is placed in the human turn — not the system turn — so
that a crafted name cannot inject instructions into the privileged system role.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from src.chains.chain_prompts import _PLANNER_PROMPT
from src.models.planner_output import PlannerOutput


def build_planner_chain(llm: BaseChatModel) -> Runnable:
    """Build the planner decomposition chain.

    Uses llm.with_structured_output(PlannerOutput) to enforce the schema on
    every invocation. The LLM must support structured output / JSON schema mode
    (ChatGroq and ChatOpenAI both do via tool-calling).

    Args:
        llm: LangChain-compatible chat model. Use a generous max_tokens budget
            (e.g. 1024+, or 1536+ for reasoning models with structured output).

    Returns:
        Runnable chain that accepts
        ``{"user_query": str, "patient_context": str, "memory_context": str}``
        and returns a PlannerOutput instance.
    """
    return _PLANNER_PROMPT | llm.with_structured_output(PlannerOutput)

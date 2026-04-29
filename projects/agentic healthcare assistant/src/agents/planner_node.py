"""planner_node — decomposes user query into ordered sub-goals.

Wraps build_planner_chain output via invoke_with_retry. On failure, sets
error and empties pending_tasks so the graph can route to END.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.models.graph_state import HealthcareState
from src.models.sub_goal import SubGoal
from src.utils.retry import invoke_with_retry

logger = logging.getLogger(__name__)

# Task names that require a resolved patient — used for resolve_patient prepend rule.
_PLANNER_PATIENT_TASKS: frozenset[str] = frozenset({
    "retrieve_history",
    "update_history",
    "book_appointment",
})


def build_planner_memory_inputs(
    *,
    patient_id: str | None,
    state: HealthcareState,
    max_recent_turns: int,
) -> tuple[str, str]:
    """Build ``patient_context`` and ``memory_context`` from HealthcareState (no DB calls).

    Reads from the checkpoint-restored messages channel and conversation_summary field.
    Always access conversation_summary via state.get() — may be absent from old checkpoints.

    Args:
        patient_id: HealthcareState ``patient_id`` (resolved slug or None).
        state: Current HealthcareState — reads messages and conversation_summary.
        max_recent_turns: Cap on recent turns extracted from state["messages"].

    Returns:
        ``(patient_context, memory_context)``.
    """
    patient_context = (
        f"Known patient ID: {patient_id}" if patient_id else "Patient identity unknown."
    )

    # Filter H/A messages and take the most recent max_recent_turns * 2 (Decision 5).
    pairs = [
        m for m in state.get("messages", [])
        if isinstance(m, HumanMessage | AIMessage)
    ][-max_recent_turns * 2:]
    recent_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in pairs
    )

    summary = state.get("conversation_summary", "") or ""
    parts: list[str] = []
    if summary:
        parts.append(f"Summary: {summary}")
    if recent_text:
        parts.append(recent_text)
    memory_context = "\n\n".join(parts)

    return patient_context, memory_context


def make_planner_node(
    chain: Any,
    max_recent_turns: int = 10,
) -> Callable[[HealthcareState], dict]:
    """Return a planner_node function with the planner chain injected via closure.

    Args:
        chain: Compiled planner chain — prompt | llm.with_structured_output(PlannerOutput).
        max_recent_turns: Cap on recent turns appended to memory_context from checkpoint.

    Returns:
        LangGraph node function compatible with HealthcareState.
    """

    def planner_node(state: HealthcareState) -> dict:
        """Decompose the user query into an ordered list of sub-goals.

        Reads user_query and patient_id from state; builds patient_context string
        and invokes the planner chain. Does not dequeue — it sets pending_tasks
        from scratch.

        Args:
            state: Current HealthcareState.

        Returns:
            Dict with planner_output, pending_tasks, and trace.
            On failure: error, empty pending_tasks, and trace.
        """
        # Accumulate under REPLACE semantics — read prior trace and extend it.
        trace_prior = list(state.get("trace", []))

        patient_id = state.get("patient_id")
        patient_context, memory_context = build_planner_memory_inputs(
            patient_id=patient_id,
            state=state,
            max_recent_turns=max_recent_turns,
        )

        try:
            result = invoke_with_retry(
                chain,
                {
                    "user_query": state["user_query"],
                    "patient_context": patient_context,
                    "memory_context": memory_context,
                },
            )
            # Enforce State Contract: resolve_patient must be first when a patient-
            # dependent task is present and patient_id is not yet resolved.
            # Only prepend when the plan actually needs a patient — not for anonymous
            # queries like pure disease searches that have no patient-dependent tasks.
            needs_patient = any(
                sg.task in _PLANNER_PATIENT_TASKS for sg in result.sub_goals
            )
            if patient_id is None and needs_patient and result.sub_goals:
                if result.sub_goals[0].task != "resolve_patient":
                    logger.warning(
                        "[planner] LLM omitted resolve_patient as first step — prepending"
                    )
                    resolver = SubGoal(task="resolve_patient", order=1, parameters={})
                    reordered = [resolver] + [
                        SubGoal(task=sg.task, order=i + 2, parameters=sg.parameters)
                        for i, sg in enumerate(result.sub_goals)
                    ]
                    result = result.model_copy(update={"sub_goals": reordered})
            logger.info("[planner] decomposed into %d sub-goals", len(result.sub_goals))
            return {
                "planner_output": result,
                "pending_tasks": result.sub_goals,
                "trace": trace_prior + [
                    f"planner: {len(result.sub_goals)} sub-goals — {result.reasoning}"
                ],
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("[planner] failed: %s", exc)
            return {
                "planner_output": None,
                "pending_tasks": [],
                "error": f"Planning failed — please rephrase your request. ({type(exc).__name__})",
                "trace": trace_prior + [f"planner: ERROR — {exc}"],
            }

    return planner_node

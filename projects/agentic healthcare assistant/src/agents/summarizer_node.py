"""summarizer_node — synthesises all completed task results into a final summary.

Terminal node — does not dequeue pending_tasks. Degrades gracefully if the
summary chain fails: produces a plain-text bullet list from completed_tasks
rather than returning None or raising.

Conversation memory is sourced from the LangGraph checkpoint: the ``messages``
channel provides recent H/A turn history, and the ``conversation_summary`` field
holds the rolling condensed summary produced by the memory_summary_chain. Both
are read via ``build_planner_memory_inputs``. The node also triggers a
rolling-summary rollup via ``memory_summary_chain`` every ``cadence`` complete
H/A turn pairs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.planner_node import build_planner_memory_inputs
from src.models.graph_state import HealthcareState
from src.utils.retry import invoke_with_retry

logger = logging.getLogger(__name__)


def make_summarizer_node(
    chain: Any,
    memory_summary_chain: Any | None,
    max_recent_turns: int = 10,
    max_summary_chars: int = 1500,
    cadence: int = 3,
) -> Callable[[HealthcareState], dict]:
    """Return a summarizer_node with the summary chain injected via closure.

    Args:
        chain: Compiled summary chain — summary_prompt | llm | StrOutputParser.
        memory_summary_chain: Rolling memory-summary rollup chain (Phase 2+).
            Pass None to skip rollup (e.g. in tests without checkpointing).
        max_recent_turns: Cap on recent conversation turns shown in summary prompt.
        max_summary_chars: Character cap for rolling summary stored in state.
        cadence: Number of complete H/A turn pairs between rolling-summary rollups.

    Returns:
        LangGraph node function compatible with HealthcareState.
    """

    def summarizer_node(state: HealthcareState) -> dict:
        """Synthesise all completed task results into a final clinical summary.

        Terminal node — does not dequeue pending_tasks. Serialises completed_tasks
        as JSON and passes them with user_query and conversation context to the
        summary chain. Degrades gracefully to a plain-text bullet list if the
        chain fails.

        Conversation memory is built from the checkpoint ``messages`` channel and
        the rolling ``conversation_summary`` field via ``build_planner_memory_inputs``.
        Appends AIMessage to the messages channel so the full turn is recorded.
        Triggers rolling-summary rollup every ``cadence`` complete H/A turn pairs.

        Args:
            state: Current HealthcareState.

        Returns:
            Dict with final_summary, messages (AIMessage appended), trace, and
            optionally conversation_summary when a rollup fires.
        """
        # Accumulate under REPLACE semantics — read prior trace and extend it.
        trace_prior = list(state.get("trace", []))

        completed = state.get("completed_tasks", [])
        tasks_json = json.dumps([t.model_dump() for t in completed], default=str)

        # Phase 4: conversation_memory sourced from checkpoint state via
        # build_planner_memory_inputs — reads both messages channel and
        # conversation_summary (rolling summary) for richer context.
        _, conversation_memory = build_planner_memory_inputs(
            patient_id=state.get("patient_id"),
            state=state,
            max_recent_turns=max_recent_turns,
        )

        try:
            summary = invoke_with_retry(
                chain,
                {
                    "completed_tasks": tasks_json,
                    "user_query": state["user_query"],
                    "conversation_memory": conversation_memory,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[summarizer] chain failed: %s", exc)
            lines = [
                f"- {t.task}: {'OK' if t.success else f'FAILED ({t.error})'}"
                for t in completed
            ]
            summary = "Summary generation failed. Task results:\n" + "\n".join(lines)

        logger.info("[summarizer] summary produced (%d chars)", len(summary))

        # Inject AIMessage so the add_messages reducer records the full turn.
        # The HumanMessage was injected by _build_initial_state before invoke.
        ai_message = AIMessage(content=summary)

        # Count HumanMessage objects in the checkpoint messages channel (Decision 4).
        # This is the authoritative user-turn count — robust against tool/system messages
        # that could corrupt an arithmetic formula assuming strict H/A alternation.
        total_turns = sum(1 for m in state.get("messages", []) if isinstance(m, HumanMessage))

        result: dict = {
            "final_summary": summary,
            "messages": [ai_message],
            "trace": trace_prior + [f"summarizer: summary produced ({len(summary)} chars)"],
        }

        # Rolling-summary rollup — fires every ``cadence`` complete turns when
        # memory_summary_chain is provided. Skipped when checkpointing is disabled
        # (total_turns will always be 1 per invocation with no accumulated state).
        if memory_summary_chain is not None and total_turns > 0 and total_turns % cadence == 0:
            prior_summary = state.get("conversation_summary", "") or ""
            # Filter to H/A pairs only before slicing so tool/system messages are
            # excluded — then take the last max_recent_turns * 2 (Decision 5).
            ha_msgs = [
                m for m in state.get("messages", [])
                if isinstance(m, HumanMessage | AIMessage)
            ][-max_recent_turns * 2:]
            recent_text = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                for m in ha_msgs
            )
            try:
                new_summary = invoke_with_retry(
                    memory_summary_chain,
                    {
                        "prior_summary": prior_summary,
                        "recent_turns": recent_text,
                        "max_chars": max_summary_chars,
                    },
                )
                if new_summary and new_summary.strip() and len(new_summary.strip()) >= 20:
                    capped = new_summary.strip()[:max_summary_chars]
                    result["conversation_summary"] = capped
                    logger.info(
                        "[summarizer] conversation_summary rolled up (%d chars, turn %d)",
                        len(capped),
                        total_turns,
                    )
                else:
                    logger.warning(
                        "[summarizer] rollup returned empty/short output at turn %d — "
                        "keeping prior summary",
                        total_turns,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[summarizer] rollup chain failed at turn %d: %s — keeping prior summary",
                    total_turns,
                    exc,
                )

        return result

    return summarizer_node

"""Tests for summary_chain — FakeListChatModel, no API calls."""

import json

from langchain_core.language_models import FakeListChatModel

from src.chains.summary_chain import _SUMMARY_PROMPT, build_summary_chain


def _fake_llm(response: str) -> FakeListChatModel:
    """Return a FakeListChatModel that always returns the given string."""
    return FakeListChatModel(responses=[response])


_SAMPLE_TASKS = json.dumps([
    {"task": "retrieve_history", "success": True, "result": "Patient has CKD stage 3."},
    {"task": "book_appointment", "success": True, "result": "Booked Dr. Patel on 2026-04-15."},
])

# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_returns_final_summary_string() -> None:
    """Chain returns the LLM string output for a list of completed tasks."""
    summary = "Patient has CKD stage 3. Appointment booked with Dr. Patel on 2026-04-15."
    chain = build_summary_chain(_fake_llm(summary))
    result = chain.invoke({
        "completed_tasks": _SAMPLE_TASKS,
        "user_query": "Book nephrologist and summarize history for Ramesh.",
        "conversation_memory": "",
    })
    assert "CKD" in result or "Patel" in result


def test_returns_string_not_message() -> None:
    """StrOutputParser must unwrap AIMessage — result is a plain string."""
    chain = build_summary_chain(_fake_llm("Summary text."))
    result = chain.invoke({
        "completed_tasks": _SAMPLE_TASKS,
        "user_query": "any query",
        "conversation_memory": "",
    })
    assert isinstance(result, str)


def test_output_matches_llm_response() -> None:
    """Chain output matches the response the fake LLM was configured with."""
    expected = "All tasks completed successfully."
    chain = build_summary_chain(_fake_llm(expected))
    result = chain.invoke({
        "completed_tasks": _SAMPLE_TASKS,
        "user_query": "summary",
        "conversation_memory": "",
    })
    assert result == expected


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_task_result() -> None:
    """Chain handles a single completed task (no multi-task synthesis needed)."""
    single = json.dumps([{"task": "search_disease", "success": True, "result": "ACE inhibitors."}])
    chain = build_summary_chain(_fake_llm("Recommended treatment: ACE inhibitors."))
    result = chain.invoke({
        "completed_tasks": single,
        "user_query": "CKD treatments?",
        "conversation_memory": "",
    })
    assert isinstance(result, str)
    assert len(result) > 0


def test_failed_task_in_results() -> None:
    """Chain handles a mix of successful and failed tasks without crashing."""
    mixed = json.dumps([
        {"task": "retrieve_history", "success": True, "result": "CKD stage 3."},
        {"task": "book_appointment", "success": False, "result": "No slots available."},
    ])
    response = "History retrieved. Booking failed: no slots available."
    chain = build_summary_chain(_fake_llm(response))
    result = chain.invoke({
        "completed_tasks": mixed,
        "user_query": "Get history and book appointment.",
        "conversation_memory": "",
    })
    assert isinstance(result, str)


def test_empty_task_list() -> None:
    """Chain handles an empty completed_tasks list gracefully."""
    chain = build_summary_chain(_fake_llm("No tasks were completed."))
    out = chain.invoke({
        "completed_tasks": "[]",
        "user_query": "do something",
        "conversation_memory": "",
    })
    assert isinstance(out, str)


def test_prompt_has_conversation_memory_input_variable() -> None:
    assert "conversation_memory" in _SUMMARY_PROMPT.input_variables


def test_invoke_with_non_empty_conversation_memory() -> None:
    chain = build_summary_chain(_fake_llm("Uses prior chat."))
    result = chain.invoke({
        "completed_tasks": "[]",
        "user_query": "What did I ask before?",
        "conversation_memory": "User: What causes headaches?",
    })
    assert isinstance(result, str)

"""Tests for disease_search_node — mock search_fn, FakeListChatModel chain."""

from langchain_core.language_models import FakeListChatModel

from src.agents.disease_search_node import make_disease_search_node
from src.chains.search_chain import build_search_chain
from src.models.search_result_item import SearchResultItem
from src.models.sub_goal import SubGoal


def _state(**overrides) -> dict:
    """Build a minimal HealthcareState-compatible dict."""
    base: dict = {
        "user_query": "CKD treatment",
        "patient_id": "P-001",
        "messages": [],
        "planner_output": None,
        "pending_tasks": [],
        "completed_tasks": [],
        "intent_safe": True,
        "final_summary": None,
        "error": None,
        "trace": [],
    }
    base.update(overrides)
    return base


def _sub_goal(query: str = "CKD treatment", max_results: int = 3) -> SubGoal:
    """Build a search_disease SubGoal."""
    return SubGoal(
        task="search_disease",
        parameters={"query": query, "max_results": max_results},
        order=1,
    )


def _fake_items(n: int = 2) -> list[SearchResultItem]:
    """Return n mock SearchResultItem objects."""
    return [
        SearchResultItem(
            title=f"Article {i}",
            url=f"https://example.com/{i}",
            snippet=f"ACE inhibitors for CKD [{i}].",
            source_domain="medlineplus.gov",
        )
        for i in range(1, n + 1)
    ]


def _search_fn(items):
    """Return a search function that always returns items."""
    return lambda query, max_results: items


def _chain(response: str):
    """Return a build_search_chain with FakeListChatModel."""
    return build_search_chain(FakeListChatModel(responses=[response]))


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_returns_synthesis_in_completed_tasks() -> None:
    """Node returns search summary in completed_tasks[0].result['summary']."""
    expected = "ACE inhibitors are first-line for CKD [1][2]."
    node = make_disease_search_node(_search_fn(_fake_items(2)), _chain(expected))
    result = node(_state(pending_tasks=[_sub_goal()]))
    task = result["completed_tasks"][0]
    assert task.success is True
    assert task.result["summary"] == expected


def test_result_count_in_task_result() -> None:
    """result_count matches the number of items returned by search_fn."""
    node = make_disease_search_node(_search_fn(_fake_items(3)), _chain("summary"))
    result = node(_state(pending_tasks=[_sub_goal()]))
    assert result["completed_tasks"][0].result["result_count"] == 3


def test_dequeues_pending_tasks() -> None:
    """Node always dequeues pending_tasks[0]."""
    extra = SubGoal(task="book_appointment", order=2)
    node = make_disease_search_node(_search_fn(_fake_items(1)), _chain("summary"))
    result = node(_state(pending_tasks=[_sub_goal(), extra]))
    assert len(result["pending_tasks"]) == 1
    assert result["pending_tasks"][0] is extra


def test_completed_tasks_is_list_of_one() -> None:
    """completed_tasks is always a list of exactly one TaskResult."""
    node = make_disease_search_node(_search_fn(_fake_items(1)), _chain("x"))
    result = node(_state(pending_tasks=[_sub_goal()]))
    assert isinstance(result["completed_tasks"], list)
    assert len(result["completed_tasks"]) == 1


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_zero_results_returns_failure() -> None:
    """Zero search results → failure TaskResult, not an exception."""
    node = make_disease_search_node(_search_fn([]), _chain("x"))
    result = node(_state(pending_tasks=[_sub_goal()]))
    task = result["completed_tasks"][0]
    assert task.success is False
    assert "no results" in task.error.lower()


def test_search_api_error_returns_failure() -> None:
    """search_fn raising → failure TaskResult with API error message."""

    def _error_fn(query, max_results):
        raise ConnectionError("API timeout")

    node = make_disease_search_node(_error_fn, _chain("x"))
    result = node(_state(pending_tasks=[_sub_goal()]))
    task = result["completed_tasks"][0]
    assert task.success is False
    assert "Search API error" in task.error


def test_dequeues_on_failure() -> None:
    """Node dequeues even when search returns no results."""
    extra = SubGoal(task="book_appointment", order=2)
    node = make_disease_search_node(_search_fn([]), _chain("x"))
    result = node(_state(pending_tasks=[_sub_goal(), extra]))
    assert len(result["pending_tasks"]) == 1


def test_invalid_params_returns_failure() -> None:
    """Malformed parameters (ValidationError) → failure TaskResult, no crash."""
    node = make_disease_search_node(_search_fn(_fake_items(1)), _chain("x"))
    bad = SubGoal(task="search_disease", parameters={"bad_key": 99}, order=1)
    result = node(_state(pending_tasks=[bad]))
    assert result["completed_tasks"][0].success is False
    assert len(result["pending_tasks"]) == 0


def test_empty_pending_tasks_returns_failure() -> None:
    """Empty pending_tasks → failure TaskResult, no IndexError."""
    node = make_disease_search_node(_search_fn(_fake_items(1)), _chain("x"))
    result = node(_state(pending_tasks=[]))
    assert result["completed_tasks"][0].success is False
    assert result["pending_tasks"] == []

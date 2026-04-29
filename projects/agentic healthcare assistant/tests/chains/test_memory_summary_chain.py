"""Tests for memory_summary_chain using FakeListChatModel; no API calls."""

from __future__ import annotations

from langchain_core.language_models import FakeListChatModel

from src.chains.memory_summary_chain import build_memory_summary_chain


def _fake_llm(response: str) -> FakeListChatModel:
    return FakeListChatModel(responses=[response])


def test_build_memory_summary_chain_returns_runnable() -> None:
    chain = build_memory_summary_chain(_fake_llm("compressed summary text here"))
    assert chain is not None
    out = chain.invoke({
        "prior_summary": "Old summary.",
        "recent_turns": "User: Hi\nAssistant: Hello",
        "max_chars": 1500,
    })
    assert isinstance(out, str)


def test_summary_chain_produces_non_empty_string() -> None:
    text = (
        "Patient discussed CKD management; prior valsartan use noted; "
        "follow-up labs ordered."
    )
    chain = build_memory_summary_chain(_fake_llm(text))
    result = chain.invoke({
        "prior_summary": "",
        "recent_turns": "User: What about my kidneys?\nAssistant: Stage 3 CKD discussed.",
        "max_chars": 1500,
    })
    assert result == text
    assert len(result) > 0

"""Tests for src/utils/retry.py — invoke_with_retry()."""

from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.exceptions import OutputParserException

from src.utils.retry import invoke_with_retry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chain_that_succeeds(return_value: str = "ok") -> MagicMock:
    """Chain that succeeds on the first call."""
    chain = MagicMock()
    chain.invoke.return_value = return_value
    return chain


def _chain_that_fails_then_succeeds(exc: Exception, return_value: str = "ok") -> MagicMock:
    """Chain that raises exc on attempt 1, succeeds on attempt 2."""
    chain = MagicMock()
    chain.invoke.side_effect = [exc, return_value]
    return chain


def _chain_that_always_fails(exc: Exception) -> MagicMock:
    """Chain that always raises exc."""
    chain = MagicMock()
    chain.invoke.side_effect = exc
    return chain


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

def test_succeeds_on_first_attempt_no_sleep():
    chain = _chain_that_succeeds("result")
    with patch("src.utils.retry.time.sleep") as mock_sleep:
        result = invoke_with_retry(chain, {"q": "x"})
    assert result == "result"
    mock_sleep.assert_not_called()
    chain.invoke.assert_called_once_with({"q": "x"})


# ---------------------------------------------------------------------------
# Retry on OutputParserException
# ---------------------------------------------------------------------------

def test_recovers_on_second_attempt_output_parser_exception():
    exc = OutputParserException("bad JSON")
    chain = _chain_that_fails_then_succeeds(exc, return_value="recovered")
    with patch("src.utils.retry.time.sleep") as mock_sleep:
        result = invoke_with_retry(chain, {})
    assert result == "recovered"
    mock_sleep.assert_called_once_with(1)


def test_raises_after_three_output_parser_exceptions():
    chain = _chain_that_always_fails(OutputParserException("bad JSON"))
    with patch("src.utils.retry.time.sleep") as mock_sleep:
        with pytest.raises(OutputParserException):
            invoke_with_retry(chain, {})
    assert chain.invoke.call_count == 3
    assert mock_sleep.call_args_list == [call(1), call(2)]


# ---------------------------------------------------------------------------
# Retry on ValueError
# ---------------------------------------------------------------------------

def test_recovers_on_second_attempt_value_error():
    exc = ValueError("invalid output")
    chain = _chain_that_fails_then_succeeds(exc, return_value="recovered")
    with patch("src.utils.retry.time.sleep") as mock_sleep:
        result = invoke_with_retry(chain, {})
    assert result == "recovered"
    mock_sleep.assert_called_once_with(1)


def test_raises_after_three_value_errors():
    chain = _chain_that_always_fails(ValueError("bad"))
    with patch("src.utils.retry.time.sleep") as mock_sleep:
        with pytest.raises(ValueError):
            invoke_with_retry(chain, {})
    assert chain.invoke.call_count == 3
    assert mock_sleep.call_args_list == [call(1), call(2)]


# ---------------------------------------------------------------------------
# Non-retryable exceptions pass through immediately
# ---------------------------------------------------------------------------

def test_non_retryable_passes_through_without_retry():
    chain = _chain_that_always_fails(RuntimeError("network down"))
    with patch("src.utils.retry.time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError, match="network down"):
            invoke_with_retry(chain, {})
    chain.invoke.assert_called_once()
    mock_sleep.assert_not_called()


def test_non_retryable_type_error_passes_through():
    chain = _chain_that_always_fails(TypeError("unexpected arg"))
    with patch("src.utils.retry.time.sleep"):
        with pytest.raises(TypeError):
            invoke_with_retry(chain, {})
    chain.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# Backoff sequence
# ---------------------------------------------------------------------------

def test_backoff_sequence_is_1_then_2_seconds():
    chain = _chain_that_always_fails(ValueError("bad"))
    with patch("src.utils.retry.time.sleep") as mock_sleep:
        with pytest.raises(ValueError):
            invoke_with_retry(chain, {})
    assert mock_sleep.call_args_list == [call(1), call(2)]


def test_inputs_passed_correctly_on_each_attempt():
    inputs = {"query": "test", "context": "ctx"}
    chain = _chain_that_always_fails(ValueError("bad"))
    with patch("src.utils.retry.time.sleep"):
        with pytest.raises(ValueError):
            invoke_with_retry(chain, inputs)
    for c in chain.invoke.call_args_list:
        assert c == call(inputs)

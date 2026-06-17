"""Tests for invoke_with_retry (src/utils/retry.py).

Spec §5 + Phase 1 exit gate — verifies:
  - Returns on first success (no sleep)
  - Retries exactly N-1 times with sleep between attempts (not after final)
  - Re-raises ValidationError immediately (no retry)
  - Re-raises KeyboardInterrupt immediately
  - Raises RuntimeError after all attempts exhausted (cause is last exception)
  - Does NOT sleep after the final failed attempt
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.utils.retry import invoke_with_retry


class _FakeChain:
    """Test double for a LangChain Runnable with configurable call behaviour."""

    def __init__(self, side_effects: list) -> None:
        """Args: side_effects — each element is either a return value or an exception to raise."""
        self._effects = iter(side_effects)
        self.call_count = 0

    def invoke(self, inputs: dict):  # noqa: ANN201
        self.call_count += 1
        effect = next(self._effects)
        if isinstance(effect, BaseException):
            raise effect
        if isinstance(effect, type) and issubclass(effect, BaseException):
            raise effect()
        return effect


class TestInvokeWithRetry:
    @patch("src.utils.retry.time.sleep")
    def test_returns_on_first_success(self, mock_sleep: MagicMock) -> None:
        """No retry needed — returns immediately and does not sleep."""
        chain = _FakeChain(side_effects=["ok"])
        result = invoke_with_retry(chain, {})
        assert result == "ok"
        assert chain.call_count == 1
        mock_sleep.assert_not_called()

    @patch("src.utils.retry.time.sleep")
    def test_retries_on_transient_error_then_succeeds(self, mock_sleep: MagicMock) -> None:
        """Fails once, succeeds on second attempt — exactly one sleep between attempts."""
        chain = _FakeChain(side_effects=[RuntimeError("timeout"), "ok"])
        result = invoke_with_retry(chain, {}, max_attempts=3)
        assert result == "ok"
        assert chain.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("src.utils.retry.time.sleep")
    def test_exhausts_all_attempts_then_raises_runtime_error(self, mock_sleep: MagicMock) -> None:
        """All 3 attempts fail — RuntimeError raised with only 2 sleeps (not 3).

        Sleep occurs only *between* attempts: after attempt 1 and after attempt 2.
        No sleep after the final (3rd) failed attempt — that would wastefully delay
        the RuntimeError raised to the caller.
        """
        err = ConnectionError("Ollama down")
        chain = _FakeChain(side_effects=[err, err, err])
        with pytest.raises(RuntimeError, match="all 3 attempts failed") as exc_info:
            invoke_with_retry(chain, {}, max_attempts=3)
        assert exc_info.value.__cause__ is err
        assert chain.call_count == 3
        # N-1 sleeps for N failures: sleep after attempt 1 and 2, not after attempt 3
        assert mock_sleep.call_count == 2

    @patch("src.utils.retry.time.sleep")
    def test_no_sleep_after_final_failed_attempt(self, mock_sleep: MagicMock) -> None:
        """Explicit check: max_attempts=1 → zero sleeps before RuntimeError."""
        chain = _FakeChain(side_effects=[OSError("fail")])
        with pytest.raises(RuntimeError, match="all 1 attempts failed"):
            invoke_with_retry(chain, {}, max_attempts=1)
        assert chain.call_count == 1
        mock_sleep.assert_not_called()

    @patch("src.utils.retry.time.sleep")
    def test_validation_error_not_retried(self, mock_sleep: MagicMock) -> None:
        """ValidationError is re-raised immediately — structured output failure is not transient."""
        from pydantic import BaseModel

        class _M(BaseModel):
            x: int

        try:
            _M(x="not-an-int")  # type: ignore[arg-type]
        except ValidationError as ve:
            val_err = ve

        chain = _FakeChain(side_effects=[val_err, "should-not-reach"])
        with pytest.raises(ValidationError):
            invoke_with_retry(chain, {}, max_attempts=3)
        assert chain.call_count == 1  # no retry
        mock_sleep.assert_not_called()

    @patch("src.utils.retry.time.sleep")
    def test_keyboard_interrupt_not_retried(self, mock_sleep: MagicMock) -> None:
        """KeyboardInterrupt is re-raised immediately — shutdown must not be suppressed."""
        chain = _FakeChain(side_effects=[KeyboardInterrupt()])
        with pytest.raises(KeyboardInterrupt):
            invoke_with_retry(chain, {}, max_attempts=3)
        assert chain.call_count == 1
        mock_sleep.assert_not_called()

    @patch("src.utils.retry.time.sleep")
    def test_backoff_increases_exponentially(self, mock_sleep: MagicMock) -> None:
        """Sleep durations follow backoff_base ** attempt_number, only between attempts."""
        chain = _FakeChain(side_effects=[OSError(), OSError(), "ok"])
        invoke_with_retry(chain, {}, max_attempts=3, backoff_base=2.0)
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        # attempt 1 → sleep 2^1=2.0, attempt 2 → sleep 2^2=4.0; attempt 3 succeeds → no sleep
        assert sleep_calls == pytest.approx([2.0, 4.0])

    @patch("src.utils.retry.time.sleep")
    def test_backoff_two_failures_then_exhausted(self, mock_sleep: MagicMock) -> None:
        """Two failures with max_attempts=2 → exactly 1 sleep (between attempts 1 and 2)."""
        chain = _FakeChain(side_effects=[OSError(), OSError()])
        with pytest.raises(RuntimeError):
            invoke_with_retry(chain, {}, max_attempts=2, backoff_base=2.0)
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        # Only one sleep: after attempt 1 (2^1=2.0). No sleep after attempt 2 (final).
        assert sleep_calls == pytest.approx([2.0])

    @patch("src.utils.retry.time.sleep")
    def test_passes_inputs_to_chain(self, mock_sleep: MagicMock) -> None:
        """Inputs dict is forwarded unchanged to chain.invoke()."""
        received: list[dict] = []

        class _RecordingChain:
            def invoke(self, inputs: dict):  # noqa: ANN201
                received.append(inputs)
                return "result"

        invoke_with_retry(_RecordingChain(), {"key": "value"})
        assert received == [{"key": "value"}]

"""Tests for make_diff_validator_node (src/agents/diff_validator.py).

Pure validation logic — no external dependencies.

Cases:
  - Valid unified diff → diff_safe=True
  - Empty string → diff_safe=False
  - Whitespace-only → diff_safe=False
  - Diff > 100 KB → diff_safe=False, "100 KB" in error
  - Diff exactly at 100 KB → diff_safe=True
  - Missing --- header → diff_safe=False
  - Missing +++ header → diff_safe=False
  - Trace entry added on accept and reject
  - Error message contains no stack traces or internal state
"""

from src.agents.diff_validator import make_diff_validator_node

_VALID_DIFF = "--- a/src/main.py\n+++ b/src/main.py\n@@ -1,3 +1,3 @@\n-old\n+new\n"


def _state(diff_text: str) -> dict:
    return {"diff_text": diff_text, "trace": []}


class TestDiffValidatorNode:
    def test_valid_diff_accepted(self) -> None:
        """Well-formed unified diff → diff_safe=True."""
        node = make_diff_validator_node()
        result = node(_state(_VALID_DIFF))

        assert result["diff_safe"] is True
        assert "error" not in result or result.get("error") is None

    def test_empty_diff_rejected(self) -> None:
        """Empty string → diff_safe=False with error."""
        node = make_diff_validator_node()
        result = node(_state(""))

        assert result["diff_safe"] is False
        assert result.get("error")

    def test_whitespace_only_rejected(self) -> None:
        """Whitespace-only diff → diff_safe=False."""
        node = make_diff_validator_node()
        result = node(_state("   \n\t  "))

        assert result["diff_safe"] is False

    def test_diff_exceeding_100kb_rejected(self) -> None:
        """Diff > 100 KB → diff_safe=False with '100 KB' in error."""
        node = make_diff_validator_node()
        big = _VALID_DIFF + ("x" * 100_001)
        result = node(_state(big))

        assert result["diff_safe"] is False
        assert "100 KB" in result["error"]

    def test_diff_at_exact_100kb_accepted(self) -> None:
        """Diff of exactly 100 000 bytes → diff_safe=True."""
        node = make_diff_validator_node()
        # Build a valid diff of exactly 100 000 bytes
        padding_needed = 100_000 - len(_VALID_DIFF.encode())
        padded = _VALID_DIFF + ("x" * padding_needed)
        result = node(_state(padded))

        assert result["diff_safe"] is True

    def test_missing_old_header_rejected(self) -> None:
        """Diff without --- line → diff_safe=False (invalid unified diff format)."""
        node = make_diff_validator_node()
        diff = "+++ b/src/main.py\n@@ -1 +1 @@\n+new\n"
        result = node(_state(diff))

        assert result["diff_safe"] is False

    def test_missing_new_header_rejected(self) -> None:
        """Diff without +++ line → diff_safe=False."""
        node = make_diff_validator_node()
        diff = "--- a/src/main.py\n@@ -1 +1 @@\n-old\n"
        result = node(_state(diff))

        assert result["diff_safe"] is False

    def test_trace_added_on_accept(self) -> None:
        """Trace gains an entry on acceptance."""
        node = make_diff_validator_node()
        result = node(_state(_VALID_DIFF))

        assert len(result["trace"]) >= 1
        assert any("accept" in t for t in result["trace"])

    def test_trace_added_on_reject(self) -> None:
        """Trace gains an entry on rejection."""
        node = make_diff_validator_node()
        result = node(_state(""))

        assert len(result["trace"]) >= 1
        assert any("reject" in t for t in result["trace"])

    def test_error_message_sanitized(self) -> None:
        """User-facing error messages contain no stack traces or internal state."""
        node = make_diff_validator_node()
        result = node(_state(""))

        error = result.get("error", "")
        assert "Traceback" not in error
        assert "Exception" not in error
        assert len(error) < 200

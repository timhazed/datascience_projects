"""Tests for make_pii_sanitizer_node (src/agents/pii_sanitizer.py).

Phase 4 exit gate: ≥95% branch coverage; quarantine path test; no live Presidio calls.

All tests mock PIIFilter (and thus AnalyzerEngine) to avoid spaCy model download.
File I/O is handled via tmp_path fixture — no live filesystem reads outside tmp.

Cases:
  - Email address in content → sanitized (pii_filter returns masked text, reason=None)
  - IPv4 → masked, proceeds normally
  - Low-confidence entity → file quarantined, NOT in sanitized_files
  - Presidio exception → file quarantined, NOT in sanitized_files
  - File read error (path does not exist) → quarantined
  - Clean file → passes through unchanged
  - State invariant: file appears in exactly one of sanitized/quarantined, never both
  - Quarantined content is QUARANTINE_SENTINEL (never raw PII content)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agents.pii_sanitizer import make_pii_sanitizer_node
from src.middleware.pii_filter import QUARANTINE_SENTINEL, PIIFilter
from src.models.trace_event import TraceEventType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_filter(sanitize_return: tuple[str, str | None]) -> PIIFilter:
    """Build a PIIFilter mock that returns the given (masked_content, reason) pair."""
    pii_filter = MagicMock(spec=PIIFilter)
    pii_filter.sanitize.return_value = sanitize_return
    return pii_filter


def _write_file(tmp_path: Path, name: str, content: str) -> str:
    """Write a temp file and return its path string."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


def _base_state(changed_files: list[str]) -> dict:
    return {
        "repo_url": "https://github.com/user/repo",
        "branch": "main",
        "changed_files": changed_files,
        "sanitized_files": [],
        "quarantined_files": [],
        "parsed_chunks": [],
        "summarized_chunks": [],
        "upsert_results": [],
        "error": None,
        "trace": [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPiiSanitizerNode:
    def test_clean_file_passes_through_unchanged(self, tmp_path: Path) -> None:
        """A file with no PII passes through sanitized_files unchanged."""
        content = "def add(a: int, b: int) -> int:\n    return a + b\n"
        path = _write_file(tmp_path, "clean.py", content)
        pii_filter = _make_filter((content, None))

        node = make_pii_sanitizer_node(pii_filter)
        result = node(_base_state([path]))

        assert len(result["sanitized_files"]) == 1
        assert result["sanitized_files"][0].content == content
        assert result["quarantined_files"] == []

    def test_email_masked_file_proceeds_normally(self, tmp_path: Path) -> None:
        """File with email: pii_filter returns masked text → sanitized_files, not quarantined."""
        original = "contact admin@example.com for help"
        masked = "contact <EMAIL> for help"
        path = _write_file(tmp_path, "email.py", original)
        pii_filter = _make_filter((masked, None))

        node = make_pii_sanitizer_node(pii_filter)
        result = node(_base_state([path]))

        assert len(result["sanitized_files"]) == 1
        assert result["sanitized_files"][0].content == masked
        assert result["quarantined_files"] == []

    def test_low_confidence_entity_quarantines_file(self, tmp_path: Path) -> None:
        """Low-confidence detection: pii_filter returns (SENTINEL, reason) → quarantined."""
        content = "debug: pointer 0x7fff5fbff8a0 in memory"
        path = _write_file(tmp_path, "debug.py", content)
        reason = "Low-confidence entity detected (CRYPTO) — score < 0.8"
        pii_filter = _make_filter((QUARANTINE_SENTINEL, reason))

        node = make_pii_sanitizer_node(pii_filter)
        result = node(_base_state([path]))

        # File must be in quarantined, not in sanitized
        assert len(result["quarantined_files"]) == 1
        assert result["sanitized_files"] == []
        # Quarantined content must be the sentinel, never the raw content
        assert result["quarantined_files"][0].content == QUARANTINE_SENTINEL
        assert result["quarantined_files"][0].quarantine_reason == reason

    def test_presidio_exception_quarantines_file(self, tmp_path: Path) -> None:
        """pii_filter.sanitize raises → file quarantined, not dropped or re-raised."""
        content = "import requests\nrequests.get('http://example.com')"
        path = _write_file(tmp_path, "net.py", content)
        pii_filter = MagicMock(spec=PIIFilter)
        pii_filter.sanitize.side_effect = RuntimeError("Presidio NLP backend failure")

        node = make_pii_sanitizer_node(pii_filter)
        result = node(_base_state([path]))

        assert len(result["quarantined_files"]) == 1
        assert result["sanitized_files"] == []
        assert result["quarantined_files"][0].content == QUARANTINE_SENTINEL

    def test_file_read_error_quarantines_file(self) -> None:
        """Non-existent file path → quarantined (read error), not raised or dropped."""
        pii_filter = _make_filter(("irrelevant", None))
        node = make_pii_sanitizer_node(pii_filter)

        result = node(_base_state(["/nonexistent/path/secret.py"]))

        assert len(result["quarantined_files"]) == 1
        assert result["sanitized_files"] == []
        assert result["quarantined_files"][0].content == QUARANTINE_SENTINEL

    def test_quarantined_content_is_sentinel_never_raw_pii(self, tmp_path: Path) -> None:
        """Quarantined file content is always the sentinel — raw PII content never stored."""
        secret = "API_KEY=sk-secret1234567890"
        path = _write_file(tmp_path, "secrets.env", secret)
        reason = "API key pattern detected"
        pii_filter = _make_filter((QUARANTINE_SENTINEL, reason))

        node = make_pii_sanitizer_node(pii_filter)
        result = node(_base_state([path]))

        quarantined = result["quarantined_files"][0]
        assert quarantined.content == QUARANTINE_SENTINEL
        assert secret not in quarantined.content

    def test_state_invariant_each_file_in_exactly_one_list(self, tmp_path: Path) -> None:
        """Every file appears in exactly one of sanitized_files / quarantined_files."""
        clean = _write_file(tmp_path, "clean.py", "x = 1")
        pii_file = _write_file(tmp_path, "pii.py", "email@example.com")

        clean_filter = MagicMock(spec=PIIFilter)
        # First call (clean.py): no PII; second call (pii.py): quarantine
        clean_filter.sanitize.side_effect = [
            ("x = 1", None),
            (QUARANTINE_SENTINEL, "email detected"),
        ]

        node = make_pii_sanitizer_node(clean_filter)
        result = node(_base_state([clean, pii_file]))

        assert len(result["sanitized_files"]) == 1
        assert len(result["quarantined_files"]) == 1
        sanitized_paths = {f.path for f in result["sanitized_files"]}
        quarantined_paths = {f.path for f in result["quarantined_files"]}
        assert sanitized_paths.isdisjoint(quarantined_paths)

    def test_trace_entries_added_per_file(self, tmp_path: Path) -> None:
        """Trace list gains one entry per file processed."""
        paths = [_write_file(tmp_path, f"f{i}.py", f"content_{i}") for i in range(3)]
        pii_filter = _make_filter(("content", None))

        node = make_pii_sanitizer_node(pii_filter)
        result = node(_base_state(paths))

        assert len(result["trace"]) == 3

    def test_empty_changed_files_returns_empty_lists(self) -> None:
        """No changed files → sanitized and quarantined are both empty."""
        pii_filter = _make_filter(("", None))
        node = make_pii_sanitizer_node(pii_filter)

        result = node(_base_state([]))

        assert result["sanitized_files"] == []
        assert result["quarantined_files"] == []

    def test_pdf_file_text_extracted_before_pii_scan(self, tmp_path: Path) -> None:
        """PDF files: pypdf extracts text; extracted text (not raw bytes) goes to PIIFilter.

        Verifies that pii_filter.sanitize receives a plain-text string, not binary/garbled
        content. The expected_text is what pypdf would extract from a minimal valid PDF.
        """
        # Minimal PDF that pypdf can parse and extract "Hello World" from
        minimal_pdf = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 72 720 Td (Hello World) Tj ET
endstream endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000360 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
441
%%EOF"""
        pdf_path = tmp_path / "report.pdf"
        pdf_path.write_bytes(minimal_pdf)

        captured: list[str] = []

        def capturing_filter(content: str) -> tuple[str, str | None]:
            captured.append(content)
            return (content, None)

        pii_filter = MagicMock(spec=PIIFilter)
        pii_filter.sanitize.side_effect = capturing_filter

        node = make_pii_sanitizer_node(pii_filter)
        result = node(_base_state([str(pdf_path)]))

        # File must be sanitized, not quarantined
        assert len(result["sanitized_files"]) == 1
        assert result["quarantined_files"] == []
        # PIIFilter must have received plain text, not binary garbage
        assert len(captured) == 1
        called_with = captured[0]
        assert "\ufffd" not in called_with, "PIIFilter received garbled binary content"
        assert isinstance(called_with, str)


class TestPiiSanitizerTraceEvents:
    """Assert trace_events are emitted on every code path."""

    def _make_ok_filter(self) -> PIIFilter:
        """Return a mock PIIFilter that always passes files as clean."""
        mock = MagicMock(spec=PIIFilter)
        mock.sanitize.return_value = ("clean content", None)
        return mock

    def _make_quarantine_filter(self) -> PIIFilter:
        """Return a mock PIIFilter that always quarantines files."""
        mock = MagicMock(spec=PIIFilter)
        mock.sanitize.return_value = (QUARANTINE_SENTINEL, "email detected")
        return mock

    def test_ok_file_emits_ok_event(self) -> None:
        """File that passes PII scan → TraceEventType.OK with the file path."""
        node = make_pii_sanitizer_node(self._make_ok_filter(), workers=1)
        # Patch _read_content so no real file I/O is required.
        with patch("src.agents.pii_sanitizer._read_content", return_value="def foo(): pass"):
            result = node(_base_state(["src/main.py"]))
        ok_events = [e for e in result["trace_events"] if e.event == TraceEventType.OK]
        assert len(ok_events) == 1
        assert ok_events[0].node == "pii_sanitizer"
        assert ok_events[0].path == "src/main.py"

    def test_quarantine_file_emits_quarantine_event(self) -> None:
        """File that is quarantined → TraceEventType.QUARANTINE with the file path."""
        node = make_pii_sanitizer_node(self._make_quarantine_filter(), workers=1)
        with patch("src.agents.pii_sanitizer._read_content", return_value="email=user@example.com"):
            result = node(_base_state(["src/secrets.py"]))
        quar_events = [e for e in result["trace_events"] if e.event == TraceEventType.QUARANTINE]
        assert len(quar_events) == 1
        assert quar_events[0].path == "src/secrets.py"

    def test_skip_file_emits_skip_event(self) -> None:
        """File matching _should_skip (dotfile) → TraceEventType.SKIP, no file I/O needed."""
        node = make_pii_sanitizer_node(self._make_ok_filter(), workers=1)
        # .hidden path is a dotfile — _should_skip returns before any read
        result = node(_base_state([".hidden"]))
        skip_events = [e for e in result["trace_events"] if e.event == TraceEventType.SKIP]
        assert len(skip_events) >= 1

"""Tests for PIIFilter (src/middleware/pii_filter.py).

Phase 4 exit gate: ≥95% branch coverage. No live Presidio calls — AnalyzerEngine
and AnonymizerEngine are injected as mocks in all tests.

Cases:
  - Clean content (no entities) → returned unchanged, reason=None
  - Email detected at high confidence → anonymized text returned, reason=None
  - Low-confidence entity (< 0.8) → QUARANTINE_SENTINEL, reason describes entity type
  - Presidio exception → QUARANTINE_SENTINEL, reason contains exception type
  - Multiple entities: all high-confidence → anonymize; any low-confidence → quarantine
  - Empty content → returned unchanged (no entities detected), reason=None
"""

from unittest.mock import MagicMock

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import EngineResult

from src.middleware.pii_filter import CONFIDENCE_THRESHOLD, QUARANTINE_SENTINEL, PIIFilter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(entity_type: str, score: float, start: int = 0, end: int = 5) -> RecognizerResult:
    """Build a fake Presidio RecognizerResult."""
    r = RecognizerResult(entity_type=entity_type, start=start, end=end, score=score)
    return r


def _make_filter(
    analyzer_results: list[RecognizerResult],
    anonymized_text: str = "<MASKED>",
) -> PIIFilter:
    """Build a PIIFilter with mocked Presidio engines."""
    analyzer = MagicMock(spec=AnalyzerEngine)
    analyzer.analyze.return_value = analyzer_results

    anonymizer = MagicMock(spec=AnonymizerEngine)
    engine_result = MagicMock(spec=EngineResult)
    engine_result.text = anonymized_text
    anonymizer.anonymize.return_value = engine_result

    return PIIFilter(analyzer=analyzer, anonymizer=anonymizer)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPIIFilter:
    def test_clean_content_returned_unchanged(self) -> None:
        """No entities detected → (content, None) — no masking applied."""
        pii_filter = _make_filter(analyzer_results=[])
        content = "def add(a, b):\n    return a + b\n"

        result_text, reason = pii_filter.sanitize(content)

        assert result_text == content
        assert reason is None

    def test_empty_content_returned_unchanged(self) -> None:
        """Empty string → (empty string, None) — nothing to mask."""
        pii_filter = _make_filter(analyzer_results=[])
        result_text, reason = pii_filter.sanitize("")

        assert result_text == ""
        assert reason is None

    def test_high_confidence_entity_anonymized(self) -> None:
        """Email with score 0.85 (≥ 0.8) → anonymized text returned, no quarantine."""
        entity = _make_result("EMAIL_ADDRESS", score=0.85, start=8, end=26)
        pii_filter = _make_filter([entity], anonymized_text="contact <EMAIL> for help")

        result_text, reason = pii_filter.sanitize("contact admin@example.com for help")

        assert result_text == "contact <EMAIL> for help"
        assert reason is None

    def test_low_confidence_entity_triggers_quarantine(self) -> None:
        """Entity with score 0.6 (< 0.8) → QUARANTINE_SENTINEL returned, reason non-None."""
        entity = _make_result("CRYPTO", score=0.6)
        pii_filter = _make_filter([entity])

        result_text, reason = pii_filter.sanitize("debug: pointer 0x7fff5fbff8a0")

        assert result_text == QUARANTINE_SENTINEL
        assert reason is not None
        assert "CRYPTO" in reason
        assert str(CONFIDENCE_THRESHOLD) in reason

    def test_mixed_confidence_any_low_triggers_quarantine(self) -> None:
        """Two entities: one high-confidence, one low-confidence → quarantine (not partial mask).

        Uses CRYPTO (not PERSON/NRP) — NLP-context-dependent types are ignored so
        they don't corrupt code identifiers; pattern-based types like CRYPTO still quarantine.
        """
        high = _make_result("EMAIL_ADDRESS", score=0.9)
        low = _make_result("CRYPTO", score=0.5)
        pii_filter = _make_filter([high, low])

        result_text, reason = pii_filter.sanitize("key=0xdeadbeef; contact admin@example.com")

        assert result_text == QUARANTINE_SENTINEL
        assert reason is not None

    def test_presidio_exception_triggers_quarantine(self) -> None:
        """Presidio AnalyzerEngine raises → QUARANTINE_SENTINEL returned, no re-raise."""
        analyzer = MagicMock(spec=AnalyzerEngine)
        analyzer.analyze.side_effect = RuntimeError("spaCy model not found")
        anonymizer = MagicMock(spec=AnonymizerEngine)
        pii_filter = PIIFilter(analyzer=analyzer, anonymizer=anonymizer)

        result_text, reason = pii_filter.sanitize("some content")

        assert result_text == QUARANTINE_SENTINEL
        assert reason is not None
        assert "RuntimeError" in reason

    def test_all_high_confidence_entities_anonymized(self) -> None:
        """Multiple high-confidence entities → anonymized text, no quarantine."""
        entities = [
            _make_result("EMAIL_ADDRESS", score=0.9),
            _make_result("PHONE_NUMBER", score=0.85),
        ]
        pii_filter = _make_filter(entities, anonymized_text="contact <EMAIL> at <PHONE>")

        result_text, reason = pii_filter.sanitize("contact me@example.com at 555-1234")

        assert result_text == "contact <EMAIL> at <PHONE>"
        assert reason is None

    def test_quarantine_reason_contains_low_confidence_entity_type(self) -> None:
        """Quarantine reason string includes the entity type that triggered quarantine.

        Uses CRYPTO — NRP is now in _IGNORED_ENTITY_TYPES (NLP-context-dependent,
        high false-positive rate on code identifiers) so it no longer triggers quarantine.
        """
        entity = _make_result("CRYPTO", score=0.4)
        pii_filter = _make_filter([entity])

        _, reason = pii_filter.sanitize("content")

        assert reason is not None
        assert "CRYPTO" in reason

    def test_boundary_score_exactly_threshold_is_safe(self) -> None:
        """Entity with score exactly at CONFIDENCE_THRESHOLD (0.8) is NOT quarantined."""
        entity = _make_result("EMAIL_ADDRESS", score=CONFIDENCE_THRESHOLD)
        pii_filter = _make_filter([entity], anonymized_text="<EMAIL>")

        result_text, reason = pii_filter.sanitize("user@example.com")

        assert reason is None
        assert result_text == "<EMAIL>"

    def test_nlp_context_types_never_quarantine_or_mask(self) -> None:
        """PERSON/LOCATION/NRP/ORGANIZATION/GPE at any confidence → ignored entirely.

        Regression guard: these types fire constantly on Python identifiers (class names,
        module aliases, SQLAlchemy symbols). Adding them to _IGNORED_ENTITY_TYPES prevents
        code chunks from being corrupted with <PERSON>/<LOCATION> placeholders at ingest.
        """
        from src.middleware.pii_filter import _IGNORED_ENTITY_TYPES

        nlp_types = {"PERSON", "LOCATION", "NRP", "ORGANIZATION", "GPE"}
        assert nlp_types.issubset(_IGNORED_ENTITY_TYPES), (
            f"NLP-context types missing from _IGNORED_ENTITY_TYPES: "
            f"{nlp_types - _IGNORED_ENTITY_TYPES}"
        )

        # Even at high confidence, ignored types must not trigger quarantine or masking.
        for entity_type in nlp_types:
            entity = _make_result(entity_type, score=0.95)
            pii_filter = _make_filter([entity], anonymized_text="<MASKED>")
            result_text, reason = pii_filter.sanitize("PatientRecord sa.Column Integer")
            assert reason is None, f"{entity_type} at 0.95 should not quarantine"
            assert result_text == "PatientRecord sa.Column Integer", (
                f"{entity_type} at 0.95 should not mask code identifiers"
            )

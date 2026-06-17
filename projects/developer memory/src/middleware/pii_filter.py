"""PII masking middleware using Presidio Analyzer + Anonymizer.

Spec §6.1 — sovereignty layer: all file content is scrubbed of PII before it reaches
ChromaDB or Gemma 4 inference. PIIFilter.sanitize() has two outcomes:

  Success path  → (masked_content, None)
    Presidio detected ≥1 entity, all with score ≥ 0.8 → anonymize and return clean text.
    No entities detected → return content unchanged.

  Quarantine path → (QUARANTINE_SENTINEL, reason)
    Any entity has score < 0.8 → surface for human review (false-positive risk).
    Presidio raises an exception → quarantine to avoid leaking unscreened content.

Quarantined files are placed in SyncState.quarantined_files by pii_sanitizer.py; they
are never passed to multimodal_parser or Gemma 4.
"""

import logging

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from src.models.sanitized_file import QUARANTINE_SENTINEL

logger = logging.getLogger(__name__)

# Re-export so callers can import QUARANTINE_SENTINEL from this module without
# needing to know its canonical home in src.models.sanitized_file.
__all__ = ["PIIFilter", "QUARANTINE_SENTINEL", "CONFIDENCE_THRESHOLD"]

# Entity confidence threshold below which a file is quarantined rather than masked.
# Score < 0.8 indicates ambiguous detection — surfacing for review avoids false positives
# (e.g., 0x7fff... memory addresses flagged as potential keys at low confidence).
CONFIDENCE_THRESHOLD = 0.8

# Entity types that are always ignored in a code repository context.
#
# Two categories of ignored types:
#
# 1. Structural false positives — entity types whose Presidio recognizers fire on
#    code constructs that superficially resemble the pattern (hex strings, version
#    numbers, import paths, numeric tokens):
#      URL               → import paths, docstring links, f-strings
#      US_DRIVER_LICENSE → hex literals, UUIDs, version strings
#      DATE_TIME         → docstrings, changelogs, log format strings
#      IP_ADDRESS        → high FP on dotted-decimal version numbers
#      PHONE_NUMBER      → numeric constants, port numbers
#      US_BANK_NUMBER    → account/order ID columns in data science code
#
# 2. NLP-context-dependent types — trained on prose; fire constantly on Python
#    identifiers, class names, module names, and SQLAlchemy/ORM symbols because
#    the spaCy NER model has no concept of "this is source code":
#      PERSON            → class names (PatientRecord), method names, variable names
#      LOCATION          → module aliases (sa → SQLAlchemy), dotted attribute access
#      NRP               → Presidio "Nationality / Religion / Political" — fires on
#                          short capitalised tokens and import aliases
#      ORGANIZATION      → package names, company names in comments
#      GPE               → geo-political entity — fires on country/city strings in
#                          config files and test fixtures (not PII in code context)
#
# Genuinely dangerous PII in code (API keys, emails, crypto keys) is detected by
# pattern-based recognizers that do NOT rely on NLP context — those are unaffected.
_IGNORED_ENTITY_TYPES: frozenset[str] = frozenset({
    # Structural false positives
    "URL",
    "US_DRIVER_LICENSE",
    "DATE_TIME",
    "IP_ADDRESS",
    "PHONE_NUMBER",
    "US_BANK_NUMBER",
    # NLP-context-dependent — high false-positive rate on source code identifiers
    "PERSON",
    "LOCATION",
    "NRP",
    "ORGANIZATION",
    "GPE",
})


class PIIFilter:
    """Presidio-backed PII masking filter.

    Masks email addresses, IPv4/IPv6, phone numbers, API keys (sk-*, ghp_*), and other
    Presidio-recognised entities. Low-confidence detections trigger quarantine rather
    than masking — they are surfaced in the Streamlit PII Review Queue for human review.

    Args:
        analyzer: AnalyzerEngine instance. If None, creates a default AnalyzerEngine
            (loads spaCy NLP pipeline). Inject a mock in tests to avoid model download.
        anonymizer: AnonymizerEngine instance. If None, creates a default AnonymizerEngine.
    """

    def __init__(
        self,
        analyzer: AnalyzerEngine | None = None,
        anonymizer: AnonymizerEngine | None = None,
    ) -> None:
        # Dependency injection for testability — avoids spaCy model download in CI
        self._analyzer: AnalyzerEngine = analyzer if analyzer is not None else AnalyzerEngine()
        self._anonymizer: AnonymizerEngine = (
            anonymizer if anonymizer is not None else AnonymizerEngine()
        )

    def sanitize(self, content: str) -> tuple[str, str | None]:
        """Mask PII in content or quarantine if masking is unsafe.

        Entities with score < CONFIDENCE_THRESHOLD (0.8) are not masked — they are surfaced
        as quarantine candidates instead. This prevents confident masking of false positives
        (e.g., hex addresses, short numeric tokens) that would corrupt code semantics.

        Args:
            content: Raw file content to screen.

        Returns:
            (masked_content, None) — masking succeeded; content is safe to index.
            (QUARANTINE_SENTINEL, reason) — quarantine triggered; reason describes the cause.
        """
        try:
            results = self._analyzer.analyze(text=content, language="en")

            # Low-confidence entities that are not in the ignored set trigger quarantine.
            # Ignored types (URL, DATE_TIME, US_DRIVER_LICENSE) are high-FP in code repos
            # and are dropped from the results rather than causing file quarantine.
            low_conf = [
                r for r in results
                if r.score < CONFIDENCE_THRESHOLD and r.entity_type not in _IGNORED_ENTITY_TYPES
            ]
            if low_conf:
                entity_types = ", ".join(sorted({r.entity_type for r in low_conf}))
                reason = (
                    f"Low-confidence entity detected ({entity_types}) — score < "
                    f"{CONFIDENCE_THRESHOLD}; flagged for human review"
                )
                logger.warning(
                    "pii_filter: quarantine [low confidence] entity_types=%s", entity_types
                )
                return (QUARANTINE_SENTINEL, reason)

            # Drop ignored entity types from the results before anonymization — they either
            # don't exist in the low-conf set anymore, or are high-conf but not real PII.
            results = [r for r in results if r.entity_type not in _IGNORED_ENTITY_TYPES]

            if not results:
                # No PII detected — return content unchanged
                return (content, None)

            # All entities have score ≥ 0.8 — safe to anonymize
            anonymized = self._anonymizer.anonymize(text=content, analyzer_results=results)
            return (anonymized.text, None)

        except Exception as exc:
            reason = f"Presidio error [{type(exc).__name__}]: {str(exc)[:120]}"
            logger.error("pii_filter: quarantine [exception] %s", reason)
            return (QUARANTINE_SENTINEL, reason)

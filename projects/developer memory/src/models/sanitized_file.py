"""PII-sanitized file model used between pii_sanitizer and multimodal_parser nodes."""

from pydantic import BaseModel

# Module-level constant so pii_filter.py can import a single authoritative definition.
# SanitizedFile.is_quarantined checks content against this value; pii_filter.py returns
# this same string as the sentinel on quarantine. A duplicated string literal would
# silently break is_quarantined if one copy drifted.
QUARANTINE_SENTINEL: str = "[CONTENT_QUARANTINED — PII review required]"


class SanitizedFile(BaseModel):
    """A source file after PII masking by pii_sanitizer.

    Preserves the original file path so multimodal_parser can route by extension
    (.py → python_parser, .md → markdown_parser, .pdf → pdf_parser, etc.).

    When a file is quarantined (masking failed or confidence < 0.8), content is set
    to QUARANTINE_SENTINEL and the file is placed in SyncState.quarantined_files rather
    than sanitized_files.

    Args:
        path: Original file path — preserved for extension-based dispatch.
        content: PII-masked text, or QUARANTINE_SENTINEL if quarantined.
        quarantine_reason: Non-None when this file is quarantined; describes the
            failing entity type or exception that triggered quarantine.
    """

    path: str
    content: str
    quarantine_reason: str | None = None

    @property
    def is_quarantined(self) -> bool:
        """True when this file has been placed in the PII review queue."""
        return self.content == QUARANTINE_SENTINEL

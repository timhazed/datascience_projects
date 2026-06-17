"""PDF file parser — called internally by multimodal_parser (not a LangGraph node).

Spec §4: extracts text from PDF pages using pypdf, then windows into CHUNK_SIZE chunks.
Each page boundary is a preferred split point. Pages larger than CHUNK_SIZE are further
windowed on sentence/paragraph boundaries.

Content type handling:
  bytes  → pypdf extracts text from the raw PDF binary (e.g., HTTP-fetched PDFs).
  str    → content is already-extracted text (pre-extracted by pii_sanitizer); chunked
           directly without re-opening the file. This is the normal pipeline path because
           pii_sanitizer reads PDFs as text via pypdf before PII screening.
"""

import io
import logging

from pypdf import PdfReader

from src.ingest.embeddings import CHUNK_SIZE
from src.models.chunk import ParsedChunk

logger = logging.getLogger(__name__)


def parse_pdf_file(path: str, content: bytes | str) -> list[ParsedChunk]:
    """Extract text from a PDF and return as ParsedChunk objects.

    When content is a str (already-extracted text from pii_sanitizer), the text is
    chunked directly — the file is NOT re-opened. This is the normal pipeline path.

    When content is bytes (e.g., HTTP-fetched PDF), pypdf extracts per-page text via
    a BytesIO wrapper. Pages with no extractable text are skipped (scanned image PDFs).

    Args:
        path: Source file path — preserved in each chunk for metadata.
        content: Already-extracted text string, OR raw PDF bytes.

    Returns:
        List of ParsedChunk objects. Empty list if no text can be extracted.
    """
    if isinstance(content, str):
        # Normal pipeline path: pii_sanitizer pre-extracted text via pypdf.
        # Chunk the extracted text directly — do not re-read the file.
        stripped = content.strip()
        if not stripped:
            return []
        chunks = _window_pages([stripped], path)
        logger.debug("parse_pdf_file: %s (pre-extracted text) → %d chunks", path, len(chunks))
        return chunks

    # Bytes path: extract text from raw PDF binary (e.g., HTTP-fetched content)
    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as exc:
        logger.warning("parse_pdf_file: failed to open %s [%s]: %s", path, type(exc).__name__, exc)
        return []

    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        stripped = text.strip()
        if stripped:
            pages.append(stripped)

    if not pages:
        logger.debug("parse_pdf_file: no extractable text in %s", path)
        return []

    chunks = _window_pages(pages, path)
    logger.debug("parse_pdf_file: %s → %d chunks", path, len(chunks))
    return chunks


def _window_pages(pages: list[str], path: str) -> list[ParsedChunk]:
    """Merge page text and window into CHUNK_SIZE chunks aligned to newlines."""
    chunks: list[ParsedChunk] = []
    buffer = ""

    for page_text in pages:
        candidate = (buffer + "\n" + page_text).strip() if buffer else page_text
        if len(candidate) <= CHUNK_SIZE:
            buffer = candidate
        else:
            if buffer:
                chunks.append(ParsedChunk(path=path, content=buffer))
                buffer = ""
            # Window large pages into CHUNK_SIZE pieces aligned to newlines
            remaining = page_text
            while len(remaining) > CHUNK_SIZE:
                cut = remaining.rfind("\n", 0, CHUNK_SIZE)
                cut = cut if cut > 0 else CHUNK_SIZE
                chunks.append(ParsedChunk(path=path, content=remaining[:cut].strip()))
                remaining = remaining[cut:].strip()
            buffer = remaining

    if buffer:
        chunks.append(ParsedChunk(path=path, content=buffer))

    return chunks

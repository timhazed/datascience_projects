"""Markdown file parser — called internally by multimodal_parser (not a LangGraph node).

Spec §4: splits .md/.markdown files on H1/H2 section boundaries. Each section becomes
a separate ParsedChunk. Sections larger than CHUNK_SIZE are windowed on paragraph
boundaries (double newline) with newline fallback.
"""

import logging
import re

from src.ingest.embeddings import CHUNK_SIZE
from src.models.chunk import ParsedChunk

logger = logging.getLogger(__name__)

# H1 or H2 markdown headers at the start of a line
_HEADER_RE = re.compile(r"^#{1,2}\s+", re.MULTILINE)


def parse_markdown_file(path: str, content: str) -> list[ParsedChunk]:
    """Parse a Markdown file into section-bounded ParsedChunk objects.

    Splits on H1 (# ) and H2 (## ) headers. Content before the first header is
    included as a preamble section if non-empty. Sections exceeding CHUNK_SIZE are
    windowed on paragraph boundaries.

    Args:
        path: Source file path — preserved in each chunk for metadata.
        content: PII-sanitized markdown text.

    Returns:
        List of ParsedChunk objects. Empty list if content is blank.
    """
    stripped = content.strip()
    if not stripped:
        return []

    sections = _split_on_headers(stripped)
    chunks = _merge_and_window(sections, path)
    logger.debug("parse_markdown_file: %s → %d chunks", path, len(chunks))
    return chunks


def _split_on_headers(content: str) -> list[str]:
    """Split content at H1/H2 header boundaries."""
    boundaries = [m.start() for m in _HEADER_RE.finditer(content)]

    if not boundaries:
        return [content]

    sections: list[str] = []
    # Preamble: text before the first header
    if boundaries[0] > 0:
        preamble = content[: boundaries[0]].strip()
        if preamble:
            sections.append(preamble)

    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(content)
        sections.append(content[start:end].strip())

    return sections or [content]


def _merge_and_window(sections: list[str], path: str) -> list[ParsedChunk]:
    """Merge small sections and window large ones to stay within CHUNK_SIZE."""
    chunks: list[ParsedChunk] = []
    buffer = ""

    for section in sections:
        if len(buffer) + len(section) <= CHUNK_SIZE:
            buffer = (buffer + "\n\n" + section).strip() if buffer else section
        else:
            if buffer:
                chunks.append(ParsedChunk(path=path, content=buffer))
                buffer = ""
            # Window on paragraph boundaries (double newline), fall back to newline
            remaining = section
            while len(remaining) > CHUNK_SIZE:
                cut = remaining.rfind("\n\n", 0, CHUNK_SIZE)
                if cut <= 0:
                    cut = remaining.rfind("\n", 0, CHUNK_SIZE)
                if cut <= 0:
                    cut = CHUNK_SIZE
                chunks.append(ParsedChunk(path=path, content=remaining[:cut].strip()))
                remaining = remaining[cut:].strip()
            buffer = remaining

    if buffer:
        chunks.append(ParsedChunk(path=path, content=buffer))

    return chunks

"""Python file parser — called internally by multimodal_parser (not a LangGraph node).

Spec §4: file-type parsers export a parse function directly. multimodal_parser routes
.py files here. Output is a list of ParsedChunk objects with CHUNK_SIZE ≤ 1500 chars
(mxbai-embed-large 512-token context window at ~3 chars/token).

Chunking strategy: split on top-level definition boundaries (class/def at column 0)
when possible; fall back to fixed character windows with newline alignment to preserve
token coherence.
"""

import logging

from src.ingest.embeddings import CHUNK_SIZE
from src.models.chunk import ParsedChunk

logger = logging.getLogger(__name__)


def parse_python_file(path: str, content: str) -> list[ParsedChunk]:
    """Parse a Python source file into semantically-bounded ParsedChunk objects.

    Splits on top-level class/def boundaries first (preferred — keeps logical units
    together). Falls back to fixed-size windows aligned to newlines when no boundaries
    are found (e.g., script files with no top-level definitions).

    Empty content or content consisting only of whitespace produces no chunks.

    Args:
        path: Source file path — preserved in each chunk for metadata.
        content: PII-sanitized Python source text.

    Returns:
        List of ParsedChunk objects. Empty list if content is blank.
    """
    stripped = content.strip()
    if not stripped:
        return []

    segments = _split_on_definitions(stripped)
    chunks = _merge_and_window(segments, path)
    logger.debug("parse_python_file: %s → %d chunks", path, len(chunks))
    return chunks


def _split_on_definitions(content: str) -> list[str]:
    """Split content at top-level class/def/async def boundaries.

    Lines starting a top-level definition (no leading whitespace, starts with
    'class ', 'def ', or 'async def ') are treated as segment boundaries.
    Content before the first definition is included as a preamble segment.
    """
    lines = content.splitlines(keepends=True)
    segments: list[str] = []
    current: list[str] = []

    for line in lines:
        stripped_line = line.lstrip()
        is_boundary = (
            not line[0:1].isspace()
            and (
                stripped_line.startswith("class ")
                or stripped_line.startswith("def ")
                or stripped_line.startswith("async def ")
            )
        )
        if is_boundary and current:
            segments.append("".join(current))
            current = []
        current.append(line)

    if current:
        segments.append("".join(current))

    return segments or [content]


def _merge_and_window(segments: list[str], path: str) -> list[ParsedChunk]:
    """Merge small segments and window large ones to stay within CHUNK_SIZE.

    Adjacent segments are accumulated until the CHUNK_SIZE limit is reached,
    then flushed as a chunk. Segments larger than CHUNK_SIZE are split on
    the nearest preceding newline boundary.
    """
    chunks: list[ParsedChunk] = []
    buffer = ""

    for seg in segments:
        if len(buffer) + len(seg) <= CHUNK_SIZE:
            buffer += seg
        else:
            if buffer:
                chunks.append(ParsedChunk(path=path, content=buffer.strip()))
                buffer = ""
            # Window large segments into CHUNK_SIZE pieces aligned to newlines
            remaining = seg
            while len(remaining) > CHUNK_SIZE:
                cut = remaining.rfind("\n", 0, CHUNK_SIZE)
                cut = cut if cut > 0 else CHUNK_SIZE
                chunks.append(ParsedChunk(path=path, content=remaining[:cut].strip()))
                remaining = remaining[cut:]
            buffer = remaining

    if buffer.strip():
        chunks.append(ParsedChunk(path=path, content=buffer.strip()))

    return chunks

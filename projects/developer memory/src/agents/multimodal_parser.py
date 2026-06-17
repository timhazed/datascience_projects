"""multimodal_parser LangGraph node — Spec §2.1, §4.

SINGLE-WRITER of parsed_chunks: exactly one invocation of this node writes the full
parsed_chunks list to state. chunk_dispatcher reads it once via Send fan-out.
No operator.add reducer on parsed_chunks — SINGLE-WRITER semantics require none.

Routes each SanitizedFile to the appropriate parser function by file extension:
  .py                 → parse_python_file
  .md / .markdown     → parse_markdown_file
  .pdf                → parse_pdf_file
  all others          → treated as plain text; single chunk per file (if non-empty)

Unsupported extensions are not dropped — they are chunked as plain text to capture
README files, configuration, and other developer-authored content.
"""

import logging
from collections.abc import Callable
from pathlib import Path

from src.ingest.embeddings import CHUNK_SIZE
from src.models.chunk import ParsedChunk
from src.models.trace_event import TraceEvent, TraceEventType
from src.utils.parsers.markdown_parser import parse_markdown_file
from src.utils.parsers.pdf_parser import parse_pdf_file
from src.utils.parsers.python_parser import parse_python_file

logger = logging.getLogger(__name__)


def make_multimodal_parser_node() -> Callable[[dict], dict]:
    """Return a multimodal_parser node (no injected dependencies — all parsers are pure).

    Returns:
        LangGraph node function that reads sanitized_files from state and writes
        the full parsed_chunks list in a single return (SINGLE-WRITER contract).
    """

    def multimodal_parser(state: dict) -> dict:
        """Parse all sanitized files into ParsedChunk objects.

        Reads state["sanitized_files"] (list[SanitizedFile]). Quarantined files are
        NOT in sanitized_files — they are in quarantined_files and bypass this node
        entirely (content is the quarantine sentinel, not parseable source).

        Returns parsed_chunks as a flat list. If sanitized_files is empty (e.g., all
        files were quarantined), returns parsed_chunks = [].
        """
        sanitized_files = state.get("sanitized_files", [])
        chunks: list[ParsedChunk] = []
        trace: list[str] = list(state.get("trace", []))

        for sf in sanitized_files:
            ext = Path(sf.path).suffix.lower()
            file_chunks = _dispatch(sf.path, sf.content, ext)
            chunks.extend(file_chunks)
            trace.append(f"multimodal_parser: {sf.path} → {len(file_chunks)} chunks")

        logger.info("multimodal_parser: %d files → %d total chunks", len(sanitized_files), len(chunks))
        return {
            "parsed_chunks": chunks,
            "trace": trace,
            "trace_events": [
                TraceEvent(
                    node="multimodal_parser",
                    event=TraceEventType.OK,
                    count=len(chunks),
                )
            ],
        }

    return multimodal_parser


def _dispatch(path: str, content: str, ext: str) -> list[ParsedChunk]:
    """Route a file to the appropriate parser by extension."""
    if ext == ".py":
        return parse_python_file(path, content)
    if ext in {".md", ".markdown"}:
        return parse_markdown_file(path, content)
    if ext == ".pdf":
        # content is pre-extracted text from pii_sanitizer (str path in pdf_parser).
        # parse_pdf_file's str branch chunks it directly without re-opening the file.
        return parse_pdf_file(path, content)
    # Plain text fallback for .ts, .js, .txt, .yaml, .toml, .json, etc.
    return _plain_text_chunks(path, content)


def _plain_text_chunks(path: str, content: str) -> list[ParsedChunk]:
    """Window plain text content into ParsedChunk objects of at most CHUNK_SIZE chars."""
    stripped = content.strip()
    if not stripped:
        return []
    chunks: list[ParsedChunk] = []
    remaining = stripped
    while len(remaining) > CHUNK_SIZE:
        cut = remaining.rfind("\n", 0, CHUNK_SIZE)
        cut = cut if cut > 0 else CHUNK_SIZE
        chunks.append(ParsedChunk(path=path, content=remaining[:cut].strip()))
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(ParsedChunk(path=path, content=remaining))
    return chunks

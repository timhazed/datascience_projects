"""pii_sanitizer LangGraph node — Spec §2.1, §6.1.

Runs PIIFilter.sanitize() on each changed file. Files are routed to exactly one of:
  - sanitized_files: masking succeeded; proceeds to multimodal_parser
  - quarantined_files: masking failed or low-confidence entity; bypasses parsing/LLM

State invariant: a file appears in exactly one of sanitized_files or quarantined_files.
Quarantined files are later upserted by chroma_upsert with semantic_type="quarantine".

PDF handling: PDFs are binary files — read_text(errors="replace") produces garbled U+FFFD
content. For .pdf files, pypdf extracts page text before PII screening. The extracted text
string is then passed to PIIFilter.sanitize() and stored in SanitizedFile.content.
multimodal_parser receives this pre-extracted text string via the normal pipeline path.
"""

import contextlib
import io
import json
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pypdf import PdfReader

from src.middleware.pii_filter import QUARANTINE_SENTINEL, PIIFilter
from src.models.sanitized_file import SanitizedFile
from src.models.trace_event import TraceEvent, TraceEventType

logger = logging.getLogger(__name__)

# Number of parallel PII scanner workers. Each worker holds its own PIIFilter /
# AnalyzerEngine instance. spaCy en_core_web_lg uses ~500MB per instance, so
# 5 workers = ~2.5GB RAM on top of ChromaDB + Python overhead — enough to OOM
# the Docker Desktop VM on macOS. Default to 2 workers (≈1GB for PII scanning).
# Override with PII_WORKERS env var; PARSER_WORKERS controls the LLM stage separately.
_PII_WORKERS: int = int(os.environ.get("PII_WORKERS", "2"))

# Presidio spaCy model hard limit — texts larger than this cause ValueError.
# We truncate to this many characters before scanning; the LLM downstream
# handles full content in chunks so truncation here only affects PII detection range.
_PRESIDIO_MAX_CHARS: int = 900_000

# Files larger than this byte threshold are skipped entirely before PII scanning.
# Large data files (geographic dumps, training corpora) block the uvicorn event loop
# for 10-30s when read and scanned, causing Docker health check failures and container
# restarts. 3MB covers all source code files while excluding large data files.
_MAX_FILE_BYTES: int = 3 * 1024 * 1024  # 3 MB


def _extract_notebook_source(file_path: str) -> str:
    """Extract only source cells from a Jupyter notebook, discarding outputs.

    Cell outputs (stdout, images, tracebacks, embedded data) can bloat a .ipynb
    to several MB of base64/binary, crashing Presidio. Only code and markdown
    source lines are semantically meaningful for indexing.

    Args:
        file_path: Path to the .ipynb file.

    Returns:
        Concatenated source text from all cells. Empty string if no source found
        or if the file is not valid JSON.
    """
    try:
        nb = json.loads(Path(file_path).read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return ""
    parts = []
    for cell in nb.get("cells", []):
        source = cell.get("source", [])
        if isinstance(source, list):
            parts.append("".join(source))
        elif isinstance(source, str):
            parts.append(source)
    return "\n".join(parts)


# Files that are skipped entirely before PII scanning — they either have no
# semantic value to index or are binary formats that cause Presidio to crash.
_SKIP_SUFFIXES: frozenset[str] = frozenset({
    # Binary image formats — read as text produce MB of garbage that OOM Presidio
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".ico", ".svg",
    # Video / audio
    ".mp4", ".mov", ".avi", ".mkv", ".mp3", ".wav", ".m4a",
    # Binary / compiled artifacts
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".bin", ".whl", ".egg",
    # ML / data binaries
    ".pkl", ".pickle", ".faiss", ".pt", ".pth", ".onnx", ".npy", ".npz", ".mat",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    # Databases
    ".sqlite3", ".sqlite", ".db", ".db-wal", ".db-shm",
    # Office / spreadsheet binaries
    ".xlsx", ".xls", ".docx", ".doc", ".pptx", ".ppt",
    # Font files
    ".ttf", ".woff", ".woff2", ".otf", ".eot",
    # Compiled web / source maps
    ".map", ".br",
    # ML model weights / data files
    ".h5", ".hdf5", ".joblib",
    # Log files — no semantic value, often large
    ".log",
    # CSV / tabular data — training datasets, not source code; high false-positive
    # rate for PHONE_NUMBER and US_BANK_NUMBER on numeric dataset column values,
    # and large files cause Presidio to scan for minutes per file.
    ".csv", ".tsv", ".parquet", ".feather",
    # macOS metadata
    ".DS_Store",
})

# Files skipped by exact name regardless of extension.
_SKIP_NAMES: frozenset[str] = frozenset({
    "poetry.lock", "package-lock.json", "yarn.lock", "Pipfile.lock",
    "Cargo.lock", "composer.lock", "Gemfile.lock",
    ".DS_Store",
})

# Directory components that cause the entire file to be skipped.
# Any file whose path contains one of these as a path segment is excluded.
_SKIP_DIRS: frozenset[str] = frozenset({
    "node_modules", "__pycache__", ".venv", "venv", ".git",
    "dist", "build", ".next", ".nuxt", "coverage", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "htmlcov",
    # Raw dataset/output directories — geographic dumps, training corpora, extracted
    # documents, and pipeline outputs are pathologically slow for Presidio NER.
    # Source code lives in src/, not data/ or output/.
    "data", "dataset", "datasets", "raw", "corpus",
    "output", "extracted", "processed", "artifacts", "cache", "tmp", "temp",
    "input",
})


def _extract_pdf_text(file_path: str) -> str:
    """Extract plain text from a PDF file using pypdf.

    Reads the file as bytes, parses with PdfReader, and concatenates non-empty page text.
    Image-only PDFs (no extractable text) return an empty string. Raises on I/O errors
    or corrupt PDF structure — the caller's exception handler quarantines the file.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        Concatenated text from all pages with extractable content. Empty string if none.
    """
    raw = Path(file_path).read_bytes()
    reader = PdfReader(io.BytesIO(raw))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        stripped = text.strip()
        if stripped:
            pages.append(stripped)
    return "\n".join(pages)


def _should_skip(file_path: str) -> str | None:
    """Return a skip reason string if the file should be excluded, else None.

    Args:
        file_path: Relative file path from the repo root.

    Returns:
        Reason string ("excluded type", "dotfile", "excluded dir") or None if the
        file should proceed to PII scanning.
    """
    p = Path(file_path)
    if p.suffix.lower() in _SKIP_SUFFIXES or p.name in _SKIP_NAMES:
        return "excluded type"
    if any(part.startswith(".") for part in p.parts):
        return "dotfile"
    if any(part in _SKIP_DIRS for part in p.parts):
        return "excluded dir"
    return None


def _read_content(file_path: str, abs_path: str) -> str:
    """Read and return the text content of a file, applying format-specific extraction.

    Args:
        file_path: Relative path (used for extension detection).
        abs_path: Absolute path to read from.

    Returns:
        Extracted text content, truncated to _PRESIDIO_MAX_CHARS if necessary.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        content = _extract_pdf_text(abs_path)
    elif suffix == ".ipynb":
        content = _extract_notebook_source(abs_path)
    else:
        content = Path(abs_path).read_text(encoding="utf-8", errors="replace")

    if len(content) > _PRESIDIO_MAX_CHARS:
        logger.debug(
            "pii_sanitizer: truncating %s (%d chars) to %d for Presidio",
            file_path, len(content), _PRESIDIO_MAX_CHARS,
        )
        content = content[:_PRESIDIO_MAX_CHARS]
    return content


def _scan_file(
    file_path: str,
    repo_root: str,
    pii_filter: PIIFilter,
) -> tuple[str, SanitizedFile | None, SanitizedFile | None, TraceEvent]:
    """Scan a single file through PIIFilter and return the outcome.

    Designed to run in a thread pool worker. Each worker uses its own PIIFilter
    instance — AnalyzerEngine is not thread-safe and must not be shared.

    Args:
        file_path: Relative file path from repo root.
        repo_root: Absolute path to the cloned repo directory.
        pii_filter: PIIFilter instance owned by this worker thread.

    Returns:
        Tuple of (trace_entry, sanitized_file_or_None, quarantined_file_or_None, trace_event).
    """
    skip_reason = _should_skip(file_path)
    if skip_reason:
        return (
            f"pii_sanitizer: skip {file_path}",
            None,
            None,
            TraceEvent(node="pii_sanitizer", event=TraceEventType.SKIP, path=file_path),
        )

    try:
        abs_path = str(Path(repo_root) / file_path) if repo_root else file_path
        content = _read_content(file_path, abs_path)
        clean, reason = pii_filter.sanitize(content)

        if reason is not None:
            logger.warning("pii_sanitizer: quarantine %s — %s", file_path, reason)
            return (
                f"pii_sanitizer: quarantine {file_path}",
                None,
                SanitizedFile(path=file_path, content=QUARANTINE_SENTINEL, quarantine_reason=reason),
                TraceEvent(node="pii_sanitizer", event=TraceEventType.QUARANTINE, path=file_path),
            )
        logger.info("pii_sanitizer: ok %s", file_path)
        return (
            f"pii_sanitizer: ok {file_path}",
            SanitizedFile(path=file_path, content=clean),
            None,
            TraceEvent(node="pii_sanitizer", event=TraceEventType.OK, path=file_path),
        )

    except Exception as exc:
        reason = f"File read error [{type(exc).__name__}]: {str(exc)[:120]}"
        logger.error("pii_sanitizer: quarantine (read error) %s — %s", file_path, reason)
        return (
            f"pii_sanitizer: quarantine (read error) {file_path}",
            None,
            SanitizedFile(path=file_path, content=QUARANTINE_SENTINEL, quarantine_reason=reason),
            TraceEvent(node="pii_sanitizer", event=TraceEventType.ERROR, path=file_path, detail=type(exc).__name__),
        )


def make_pii_sanitizer_node(
    pii_filter: PIIFilter,
    workers: int | None = None,
    pii_progress_callback: Callable[[str, int, int], None] | None = None,
) -> Callable[[dict], dict]:
    """Return a pii_sanitizer node that scans files in parallel using a thread pool.

    Each worker thread owns its own PIIFilter instance — AnalyzerEngine is not
    thread-safe. When pii_filter is a real PIIFilter, additional instances are
    constructed for each extra worker. When pii_filter is a mock (tests), the same
    instance is reused across all workers and workers is forced to 1 to preserve
    side_effect call ordering.

    Args:
        pii_filter: PIIFilter instance (or mock in tests).
        workers: Number of parallel workers. Defaults to _PII_WORKERS (PARSER_WORKERS
            env var). Pass 1 in tests to disable parallelism and keep mock ordering.
        pii_progress_callback: Optional callable invoked after each file is processed.
            Signature: (job_id: str, pii_files_done: int, pii_files_total: int) -> None.
            Callback is skipped when job_id is None (direct MCP tool invocations).

    Returns:
        LangGraph node function that processes all changed_files in parallel and
        splits them into sanitized_files and quarantined_files in state.
    """
    # Detect mocks by checking for the real class (not spec= mocks which pass isinstance).
    # MagicMock(spec=PIIFilter) passes isinstance(x, PIIFilter) — check type directly.
    is_real = type(pii_filter) is PIIFilter
    n_workers = (workers if workers is not None else _PII_WORKERS) if is_real else 1
    worker_filters: list[PIIFilter] = (
        [PIIFilter() for _ in range(n_workers)] if is_real and n_workers > 1
        else [pii_filter] * max(n_workers, 1)
    )

    def pii_sanitizer(state: dict) -> dict:
        """Scan changed_files through PIIFilter in parallel and partition results.

        Files are distributed across _PII_WORKERS threads. Each thread owns its own
        PIIFilter to avoid AnalyzerEngine thread-safety issues. Results are collected
        as futures complete and partitioned into sanitized / quarantined lists.
        """
        # Read job_id at node entry — None means skip all progress callbacks
        job_id: str | None = state.get("_job_id")

        changed_files: list[str] = state.get("changed_files", [])
        sanitized: list[SanitizedFile] = list(state.get("sanitized_files", []))
        quarantined: list[SanitizedFile] = list(state.get("quarantined_files", []))
        trace: list[str] = list(state.get("trace", []))
        repo_root: str = state.get("repo_root", "")

        logger.info("pii_sanitizer: scanning %d files with %d workers", len(changed_files), n_workers)
        try:
            import psutil  # noqa: PLC0415
            proc = psutil.Process()
            rss_mb = proc.memory_info().rss / 1024 / 1024
            avail_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024
            logger.info("MEM [pii start] — process RSS %.0f MB | system available %.1f GB", rss_mb, avail_gb)
        except Exception:  # noqa: BLE001
            pass

        trace_events: list[TraceEvent] = []

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Pre-filter oversized files before dispatching to workers.
            # Size check happens on the main thread so we never queue a file
            # that would block a worker for minutes or spike memory.
            future_to_path = {}
            worker_idx = 0
            for file_path in changed_files:
                abs_path = str(Path(repo_root) / file_path) if repo_root else file_path
                try:
                    file_size = Path(abs_path).stat().st_size
                except OSError:
                    file_size = 0  # let worker handle missing file as a read error
                if file_size > _MAX_FILE_BYTES:
                    logger.info(
                        "pii_sanitizer: skip %s (%.1f MB > %d MB limit)",
                        file_path, file_size / 1024 / 1024, _MAX_FILE_BYTES // (1024 * 1024),
                    )
                    trace.append(f"pii_sanitizer: skip {file_path} (oversized)")
                    trace_events.append(
                        TraceEvent(node="pii_sanitizer", event=TraceEventType.SKIP, path=file_path, detail="oversized")
                    )
                    continue
                future_to_path[executor.submit(
                    _scan_file,
                    file_path,
                    repo_root,
                    worker_filters[worker_idx % n_workers],
                )] = file_path
                worker_idx += 1

            pii_files_total = len(future_to_path)
            pii_files_done = 0

            for future in as_completed(future_to_path):
                trace_entry, san, quar, te = future.result()
                trace.append(trace_entry)
                trace_events.append(te)
                if san is not None:
                    sanitized.append(san)
                if quar is not None:
                    quarantined.append(quar)

                pii_files_done += 1
                if job_id is not None and pii_progress_callback is not None:
                    with contextlib.suppress(Exception):
                        pii_progress_callback(job_id, pii_files_done, pii_files_total)

        logger.info(
            "pii_sanitizer: done — %d sanitized, %d quarantined",
            len(sanitized), len(quarantined),
        )
        return {
            "sanitized_files": sanitized,
            "quarantined_files": quarantined,
            "pii_files_done": pii_files_done,
            "pii_files_total": pii_files_total,
            "trace": trace,
            "trace_events": trace_events,
        }

    return pii_sanitizer

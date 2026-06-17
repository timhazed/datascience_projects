"""Filesystem watchdog handler for local repository change detection.

Spec §9 — DeveloperMemoryWatchdog monitors a local repo directory and triggers
incremental re-ingest when files are created or modified. Used in the long-running
daemon mode where a developer works on a repo locally.

The handler is intentionally simple — it accumulates changed paths and signals a
callback. Actual ingest is handled by the Sync Pipeline, not the watchdog itself.
"""

import logging
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler

logger = logging.getLogger(__name__)

# File extensions to watch — mirrors the multimodal_parser extension set
WATCHED_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".ts", ".js", ".md", ".markdown", ".txt", ".pdf"}
)


class DeveloperMemoryWatchdog(FileSystemEventHandler):
    """Watchdog event handler that triggers re-ingest on file create/modify events.

    Filters events to WATCHED_EXTENSIONS only — ignores IDE artefacts, __pycache__,
    .git directory changes, and build outputs to avoid spurious re-ingest cycles.

    Args:
        on_change: Callback invoked with the changed file path (str) whenever a
            watched file is created or modified. The callback is responsible for
            triggering the Sync Pipeline for that path.
    """

    def __init__(self, on_change: Callable[[str], None]) -> None:
        super().__init__()
        self._on_change = on_change

    def on_created(self, event: FileSystemEvent) -> None:
        """Trigger ingest callback when a new watched file is created."""
        self._handle_event(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Trigger ingest callback when a watched file is modified."""
        self._handle_event(event)

    def _handle_event(self, event: FileSystemEvent) -> None:
        """Filter and dispatch file system events to the on_change callback.

        Ignores directory events, .git paths, __pycache__ directories, and files
        with extensions outside WATCHED_EXTENSIONS.
        """
        if event.is_directory:
            return

        path = Path(str(event.src_path))

        # Skip .git internals and Python cache
        if ".git" in path.parts or "__pycache__" in path.parts:
            return

        if path.suffix.lower() not in WATCHED_EXTENSIONS:
            logger.debug("watchdog: skipping %s (unsupported extension)", path)
            return

        logger.info("watchdog: change detected %s", path)
        self._on_change(str(path))

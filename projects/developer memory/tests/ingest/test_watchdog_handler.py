"""Tests for DeveloperMemoryWatchdog (src/ingest/watchdog_handler.py).

No live filesystem monitoring — events are constructed and dispatched manually.

Cases:
  - .py file created → on_change callback invoked with file path
  - .md file modified → on_change callback invoked
  - .git path → ignored (no callback)
  - __pycache__ path → ignored
  - Unsupported extension (.pyc, .DS_Store) → ignored
  - Directory event → ignored
  - Multiple events for watched files → callback invoked once per event
"""

from unittest.mock import MagicMock

from watchdog.events import FileCreatedEvent, FileModifiedEvent

from src.ingest.watchdog_handler import WATCHED_EXTENSIONS, DeveloperMemoryWatchdog


def _created(path: str) -> FileCreatedEvent:
    event = FileCreatedEvent(path)
    event.is_directory = False
    return event


def _modified(path: str) -> FileModifiedEvent:
    event = FileModifiedEvent(path)
    event.is_directory = False
    return event


def _dir_event(path: str) -> FileCreatedEvent:
    event = FileCreatedEvent(path)
    event.is_directory = True
    return event


class TestDeveloperMemoryWatchdog:
    def test_py_file_created_triggers_callback(self) -> None:
        """.py file create event → on_change called with the file path."""
        callback = MagicMock()
        handler = DeveloperMemoryWatchdog(on_change=callback)

        handler.on_created(_created("/repo/src/main.py"))

        callback.assert_called_once_with("/repo/src/main.py")

    def test_md_file_modified_triggers_callback(self) -> None:
        """.md file modify event → on_change called."""
        callback = MagicMock()
        handler = DeveloperMemoryWatchdog(on_change=callback)

        handler.on_modified(_modified("/repo/README.md"))

        callback.assert_called_once_with("/repo/README.md")

    def test_git_directory_path_ignored(self) -> None:
        """Files inside .git are not forwarded to on_change."""
        callback = MagicMock()
        handler = DeveloperMemoryWatchdog(on_change=callback)

        handler.on_created(_created("/repo/.git/COMMIT_EDITMSG"))

        callback.assert_not_called()

    def test_pycache_path_ignored(self) -> None:
        """Files inside __pycache__ are not forwarded."""
        callback = MagicMock()
        handler = DeveloperMemoryWatchdog(on_change=callback)

        handler.on_modified(_modified("/repo/src/__pycache__/main.cpython-313.pyc"))

        callback.assert_not_called()

    def test_unsupported_extension_ignored(self) -> None:
        """.pyc and .DS_Store files are not forwarded."""
        callback = MagicMock()
        handler = DeveloperMemoryWatchdog(on_change=callback)

        handler.on_created(_created("/repo/src/main.pyc"))
        handler.on_created(_created("/repo/.DS_Store"))

        callback.assert_not_called()

    def test_directory_event_ignored(self) -> None:
        """is_directory=True events are not forwarded."""
        callback = MagicMock()
        handler = DeveloperMemoryWatchdog(on_change=callback)

        handler.on_created(_dir_event("/repo/src/new_package"))
        handler.on_modified(_dir_event("/repo/src/existing_package"))

        callback.assert_not_called()

    def test_multiple_watched_events_all_forwarded(self) -> None:
        """Multiple events on watched files each trigger the callback separately."""
        callback = MagicMock()
        handler = DeveloperMemoryWatchdog(on_change=callback)

        handler.on_created(_created("/repo/src/a.py"))
        handler.on_modified(_modified("/repo/src/b.ts"))
        handler.on_created(_created("/repo/docs/guide.md"))

        assert callback.call_count == 3

    def test_watched_extensions_set_is_non_empty(self) -> None:
        """WATCHED_EXTENSIONS must include at minimum .py and .md."""
        assert ".py" in WATCHED_EXTENSIONS
        assert ".md" in WATCHED_EXTENSIONS

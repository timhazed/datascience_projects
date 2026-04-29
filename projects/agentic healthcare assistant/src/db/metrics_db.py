"""MetricsDB — SQLite persistence for session activity metrics.

Schema:
    session_log (id, session_id, datetime, operation, success, latency_ms, error)

Every metric entry written during a session (patient lookups, bookings, LLM
invocations) is persisted here so historical mean latency per operation is
available across restarts.  The Metrics tab uses get_means() to draw a
reference mean line alongside the current session's latency trend.

session_id is a UUID generated once per app startup so individual runs are
distinguishable in the log.
"""

from __future__ import annotations

from pathlib import Path

from src.db.sqlite_utils import connect_row_factory

_CREATE_SESSION_LOG = """
CREATE TABLE IF NOT EXISTS session_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    datetime    TEXT    NOT NULL,
    operation   TEXT    NOT NULL,
    success     INTEGER NOT NULL,
    latency_ms  REAL    NOT NULL,
    error       TEXT    DEFAULT ''
);
"""

_CREATE_INDEX_OPERATION = (
    "CREATE INDEX IF NOT EXISTS idx_session_log_operation "
    "ON session_log (operation);"
)


class MetricsDB:
    """SQLite persistence layer for session activity metrics.

    init_db() is idempotent — safe to call on every startup.
    log_entry() appends one row per metric event.
    get_means() returns mean latency per operation over all historical sessions,
    excluding sub-millisecond entries (< 1 ms) from DB-only operations that
    would skew LLM operation averages toward zero.
    """

    def init_db(self, db_path: str) -> None:
        """Create the session_log table and index if they do not exist.

        Args:
            db_path: Path to the SQLite database file.
        """
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with connect_row_factory(db_path) as conn:
            conn.execute(_CREATE_SESSION_LOG)
            conn.execute(_CREATE_INDEX_OPERATION)

    def log_entry(
        self,
        db_path: str,
        session_id: str,
        datetime_str: str,
        operation: str,
        success: bool,
        latency_ms: float,
        error: str = "",
    ) -> None:
        """Persist one metric entry to the session_log table.

        Args:
            db_path: Path to the SQLite database file.
            session_id: UUID identifying the current app session.
            datetime_str: ISO-formatted datetime string for the event.
            operation: Human-readable operation name (e.g. "Patient Lookup").
            success: Whether the operation succeeded.
            latency_ms: Elapsed time in milliseconds.
            error: Optional error message; empty string if none.
        """
        with connect_row_factory(db_path) as conn:
            conn.execute(
                "INSERT INTO session_log "
                "(session_id, datetime, operation, success, latency_ms, error) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, datetime_str, operation, int(success), latency_ms, error),
            )

    def get_means(self, db_path: str) -> dict[str, float]:
        """Return mean latency per operation across all historical sessions.

        Only entries with latency_ms >= 1.0 are included so sub-millisecond
        direct DB operations (Patient Lookup, Book Appointment) do not suppress
        the mean for operations that have genuine LLM latency.

        Args:
            db_path: Path to the SQLite database file.

        Returns:
            Dict mapping operation name → mean latency_ms (float).
            Empty dict if no qualifying rows exist.
        """
        if not Path(db_path).exists():
            return {}
        with connect_row_factory(db_path) as conn:
            rows = conn.execute(
                """
                SELECT operation, AVG(latency_ms) AS mean_ms
                FROM   session_log
                WHERE  latency_ms >= 1.0
                GROUP  BY operation
                """,
            ).fetchall()
        return {r["operation"]: round(r["mean_ms"], 1) for r in rows}

    def get_operation_history(
        self, db_path: str, operation: str, limit: int = 50
    ) -> list[dict]:
        """Return recent latency entries for a specific operation.

        Useful for per-operation sparklines or detailed drill-down.

        Args:
            db_path: Path to the SQLite database file.
            operation: Operation name to filter by.
            limit: Maximum number of rows to return (most recent first).

        Returns:
            List of dicts with keys: datetime, session_id, latency_ms, success, error.
        """
        if not Path(db_path).exists():
            return []
        with connect_row_factory(db_path) as conn:
            rows = conn.execute(
                """
                SELECT datetime, session_id, latency_ms, success, error
                FROM   session_log
                WHERE  operation = ?
                ORDER  BY id DESC
                LIMIT  ?
                """,
                (operation, limit),
            ).fetchall()
        return [dict(r) for r in rows]

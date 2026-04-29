"""Shared SQLite connection helpers for DB layer modules."""

from __future__ import annotations

import sqlite3


def connect_row_factory(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with ``sqlite3.Row`` row_factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

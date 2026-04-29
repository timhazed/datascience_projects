"""PatientDB — SQLite CRUD for PatientRecord.

Schema (patients table):
    patient_id TEXT PRIMARY KEY
    phone      TEXT NOT NULL
    email      TEXT
    name       TEXT NOT NULL
    age        INTEGER NOT NULL
    gender     TEXT NOT NULL
    address    TEXT
    summary    TEXT
    conditions TEXT   -- JSON array
    medications TEXT  -- JSON array
    allergies  TEXT   -- JSON array
    notes      TEXT
    last_updated TEXT  -- ISO 8601 datetime

All list fields (conditions, medications, allergies) are stored as JSON arrays
and round-tripped via json.loads / json.dumps on every read and write.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from src.db.sqlite_utils import connect_row_factory
from src.models.patient import PatientRecord

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id   TEXT PRIMARY KEY,
    phone        TEXT NOT NULL,
    email        TEXT,
    name         TEXT NOT NULL,
    age          INTEGER NOT NULL,
    gender       TEXT NOT NULL,
    address      TEXT DEFAULT '',
    summary      TEXT DEFAULT '',
    conditions   TEXT DEFAULT '[]',
    medications  TEXT DEFAULT '[]',
    allergies    TEXT DEFAULT '[]',
    notes        TEXT DEFAULT '',
    last_updated TEXT NOT NULL
);
"""

_CREATE_INDEX_NAME = (
    "CREATE INDEX IF NOT EXISTS idx_patients_name ON patients (name);"
)
_CREATE_INDEX_PHONE = (
    "CREATE INDEX IF NOT EXISTS idx_patients_phone ON patients (phone);"
)

_LIST_FIELDS = frozenset({"conditions", "medications", "allergies"})


class PatientDB:
    """SQLite CRUD layer for PatientRecord.

    All methods are stateless — db_path is passed explicitly so the class
    can be used with :memory: databases in tests without patching globals.
    """

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> PatientRecord:
        """Deserialize a sqlite3.Row into a PatientRecord."""
        d = dict(row)
        for field in _LIST_FIELDS:
            d[field] = json.loads(d[field] or "[]")
        return PatientRecord(**d)

    def init_db(self, db_path: str) -> None:
        """Create the patients table and indexes if they do not exist.

        Args:
            db_path: Path to the SQLite database file.
        """
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with connect_row_factory(db_path) as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX_NAME)
            conn.execute(_CREATE_INDEX_PHONE)

    def create_patient(self, db_path: str, record: PatientRecord) -> None:
        """Insert a new patient row. Skips silently if the slug already exists.

        Args:
            db_path: Path to the SQLite database file.
            record: Fully populated PatientRecord to insert.
        """
        with connect_row_factory(db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO patients
                    (patient_id, phone, email, name, age, gender, address,
                     summary, conditions, medications, allergies, notes, last_updated)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.patient_id,
                    record.phone,
                    record.email,
                    record.name,
                    record.age,
                    record.gender,
                    record.address,
                    record.summary,
                    json.dumps(record.conditions),
                    json.dumps(record.medications),
                    json.dumps(record.allergies),
                    record.notes,
                    record.last_updated.isoformat(),
                ),
            )

    def get_patient(self, db_path: str, patient_id: str) -> PatientRecord | None:
        """Fetch a patient by slug ID.

        Args:
            db_path: Path to the SQLite database file.
            patient_id: Patient slug, e.g. 'P-4a2f91'.

        Returns:
            PatientRecord if found, None otherwise.
        """
        with connect_row_factory(db_path) as conn:
            row = conn.execute(
                "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def update_field(
        self,
        db_path: str,
        patient_id: str,
        field: str,
        value: str,
        operation: str = "append",
    ) -> bool:
        """Append or replace a clinical field on a patient record.

        List fields (conditions, medications, allergies) are parsed from JSON,
        modified, and re-serialised. String fields (summary, notes) are
        concatenated with a newline (append) or replaced directly.

        Args:
            db_path: Path to the SQLite database file.
            patient_id: Patient slug to update.
            field: Field name — one of conditions, medications, allergies,
                notes, summary.
            value: Value to append or replace.
            operation: "append" (default) or "replace".

        Returns:
            True if the row was updated, False if patient_id not found.
        """
        record = self.get_patient(db_path, patient_id)
        if record is None:
            return False

        if field in _LIST_FIELDS:
            current: list[str] = getattr(record, field)
            if operation == "replace":
                new_value_json = json.dumps([value])
            else:
                current.append(value)
                new_value_json = json.dumps(current)
            sql_value: str = new_value_json
        else:
            existing: str = getattr(record, field, "")
            if operation == "replace":
                sql_value = value
            else:
                sql_value = f"{existing}\n{value}".strip() if existing else value

        with connect_row_factory(db_path) as conn:
            conn.execute(
                f"UPDATE patients SET {field} = ?, last_updated = ? WHERE patient_id = ?",  # noqa: S608
                (sql_value, datetime.now().isoformat(), patient_id),
            )
        return True

    def update_notes(self, db_path: str, patient_id: str, notes: str) -> bool:
        """Directly replace the notes field with raw PDF text.

        Args:
            db_path: Path to the SQLite database file.
            patient_id: Patient slug to update.
            notes: Full text to set in the notes column.

        Returns:
            True if updated, False if patient not found.
        """
        with connect_row_factory(db_path) as conn:
            cursor = conn.execute(
                "UPDATE patients SET notes = ?, last_updated = ? WHERE patient_id = ?",
                (notes, datetime.now().isoformat(), patient_id),
            )
        return cursor.rowcount > 0

    def fuzzy_search(self, db_path: str, name: str, limit: int = 5) -> list[PatientRecord]:
        """Search patients by name substring (case-insensitive).

        Args:
            db_path: Path to the SQLite database file.
            name: Partial or full patient name to search.
            limit: Maximum number of results to return.

        Returns:
            List of matching PatientRecord instances, ordered by name.
        """
        with connect_row_factory(db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM patients WHERE LOWER(name) LIKE LOWER(?) ORDER BY name LIMIT ?",
                (f"%{name}%", limit),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def phone_lookup(self, db_path: str, phone: str) -> PatientRecord | None:
        """Look up a patient by exact phone number.

        Args:
            db_path: Path to the SQLite database file.
            phone: Exact phone number string.

        Returns:
            PatientRecord if found, None otherwise.
        """
        with connect_row_factory(db_path) as conn:
            row = conn.execute(
                "SELECT * FROM patients WHERE phone = ?", (phone,)
            ).fetchone()
        return self._row_to_record(row) if row else None

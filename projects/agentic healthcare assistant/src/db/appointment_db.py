"""AppointmentDB — SQLite CRUD for doctors, slots, and bookings.

Schema:
    doctors  (doctor_id, name, specialty)
    slots    (slot_id, doctor_id, specialty, slot_datetime, booked)
    bookings (booking_id, slot_id, patient_id, reason, urgency, booked_at)

Slot seeding: 5 doctors × ~4 slots/day × 86 days = ~1 720 slots.
Slots are generated deterministically from today's date so tests get
consistent results regardless of when they run.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

from src.db.sqlite_utils import connect_row_factory
from src.models.appointment_result import AppointmentResult

_DOCTORS = [
    ("D-001", "Dr. Patel — Nephrology", "Nephrology"),
    ("D-002", "Dr. Singh — Cardiology", "Cardiology"),
    ("D-003", "Dr. Iyer — Endocrinology", "Endocrinology"),
    ("D-004", "Dr. Mehta — General Practice", "General Practice"),
    ("D-005", "Dr. Khan — Pulmonology", "Pulmonology"),
]

# Slot times per day (24-hour format hours)
_SLOT_HOURS = [9, 10, 11, 14]
_SLOT_DAYS_AHEAD = 86  # ~3 months of availability

_CREATE_DOCTORS = """
CREATE TABLE IF NOT EXISTS doctors (
    doctor_id TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    specialty TEXT NOT NULL
);
"""

_CREATE_SLOTS = """
CREATE TABLE IF NOT EXISTS slots (
    slot_id       TEXT PRIMARY KEY,
    doctor_id     TEXT NOT NULL,
    specialty     TEXT NOT NULL,
    slot_datetime TEXT NOT NULL,
    booked        INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (doctor_id) REFERENCES doctors (doctor_id)
);
"""

_CREATE_BOOKINGS = """
CREATE TABLE IF NOT EXISTS bookings (
    booking_id TEXT PRIMARY KEY,
    slot_id    TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    reason     TEXT,
    urgency    TEXT NOT NULL,
    booked_at  TEXT NOT NULL,
    FOREIGN KEY (slot_id) REFERENCES slots (slot_id)
);
"""

_CREATE_INDEX_SLOTS_SPECIALTY = (
    "CREATE INDEX IF NOT EXISTS idx_slots_specialty ON slots (specialty, booked, slot_datetime);"
)


class AppointmentDB:
    """SQLite CRUD layer for appointment scheduling.

    Doctors and slots are seeded once on init_db(); subsequent calls are
    no-ops (IF NOT EXISTS / INSERT OR IGNORE guards).
    """

    def init_db(self, db_path: str) -> None:
        """Create tables, indexes, and seed doctors + slots if tables are empty.

        Slot generation is deterministic from today's date.

        Args:
            db_path: Path to the SQLite database file.
        """
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with connect_row_factory(db_path) as conn:
            conn.execute(_CREATE_DOCTORS)
            conn.execute(_CREATE_SLOTS)
            conn.execute(_CREATE_BOOKINGS)
            conn.execute(_CREATE_INDEX_SLOTS_SPECIALTY)

            # Seed only if tables are empty — idempotent re-runs.
            if conn.execute("SELECT COUNT(*) FROM doctors").fetchone()[0] == 0:
                conn.executemany(
                    "INSERT OR IGNORE INTO doctors VALUES (?, ?, ?)", _DOCTORS
                )
                self._seed_slots(conn)

    def _seed_slots(self, conn: sqlite3.Connection) -> None:
        """Generate ~1 720 slots for all 5 doctors over the next 86 days."""
        today = date.today()
        rows = []
        for doc_id, _, specialty in _DOCTORS:
            for day_offset in range(_SLOT_DAYS_AHEAD):
                slot_date = today + timedelta(days=day_offset)
                # Skip weekends
                if slot_date.weekday() >= 5:
                    continue
                for hour in _SLOT_HOURS:
                    slot_dt = datetime(
                        slot_date.year, slot_date.month, slot_date.day, hour, 0
                    ).isoformat()
                    rows.append((str(uuid.uuid4()), doc_id, specialty, slot_dt, 0))
        conn.executemany(
            "INSERT OR IGNORE INTO slots (slot_id, doctor_id, specialty, slot_datetime, booked) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )

    def query_slots(
        self,
        db_path: str,
        specialty: str,
        preferred_date: date | None = None,
        urgency: str = "routine",
    ) -> list[dict]:
        """Return available slots for a specialty, ordered earliest-first.

        For emergency urgency, returns the single next available slot.
        For urgent, returns slots within 7 days. For routine, all available slots.

        Args:
            db_path: Path to the SQLite database file.
            specialty: Medical specialty to filter by.
            preferred_date: Optional preferred date; slots from this date onward.
            urgency: "routine", "urgent", or "emergency".

        Returns:
            List of slot dicts with slot_id, doctor_id, specialty, slot_datetime.
        """
        from_dt = (
            datetime.combine(preferred_date, datetime.min.time()).isoformat()
            if preferred_date
            else datetime.now().isoformat()
        )

        if urgency == "emergency":
            limit = 1
            to_dt = None
        elif urgency == "urgent":
            limit = 20
            to_dt = (datetime.now() + timedelta(days=7)).isoformat()
        else:
            limit = 50
            to_dt = None

        # Case-insensitive match so planner casing ("endocrinology" vs "Endocrinology") never fails.
        if to_dt:
            sql = (
                "SELECT slot_id, doctor_id, specialty, slot_datetime FROM slots "
                "WHERE LOWER(specialty) = LOWER(?) AND booked = 0 AND slot_datetime >= ? "
                "AND slot_datetime <= ? ORDER BY slot_datetime LIMIT ?"
            )
            params = (specialty, from_dt, to_dt, limit)
        else:
            sql = (
                "SELECT slot_id, doctor_id, specialty, slot_datetime FROM slots "
                "WHERE LOWER(specialty) = LOWER(?) AND booked = 0 AND slot_datetime >= ? "
                "ORDER BY slot_datetime LIMIT ?"
            )
            params = (specialty, from_dt, limit)

        with connect_row_factory(db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def book_slot(
        self,
        db_path: str,
        slot_id: str,
        patient_id: str,
        reason: str = "",
        urgency: str = "routine",
    ) -> AppointmentResult:
        """Atomically book a slot. Fails gracefully if already booked.

        Uses UPDATE … WHERE booked=0 to prevent double-booking without a
        separate SELECT. If 0 rows are affected, returns success=False.

        Args:
            db_path: Path to the SQLite database file.
            slot_id: Slot UUID to book.
            patient_id: Patient slug requesting the booking.
            reason: Brief clinical reason for the visit.
            urgency: Clinical urgency level.

        Returns:
            AppointmentResult with success=True and booking details, or
            success=False with an informative message.
        """
        with connect_row_factory(db_path) as conn:
            cursor = conn.execute(
                "UPDATE slots SET booked = 1 WHERE slot_id = ? AND booked = 0",
                (slot_id,),
            )
            if cursor.rowcount == 0:
                return AppointmentResult(
                    success=False,
                    patient_id=patient_id,
                    message="Slot is no longer available — it may have just been booked.",
                )

            slot_row = conn.execute(
                "SELECT slot_id, doctor_id, specialty, slot_datetime FROM slots "
                "WHERE slot_id = ?",
                (slot_id,),
            ).fetchone()

            doctor_row = conn.execute(
                "SELECT name FROM doctors WHERE doctor_id = ?",
                (slot_row["doctor_id"],),
            ).fetchone()

            booking_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO bookings "
                "(booking_id, slot_id, patient_id, reason, urgency, booked_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (booking_id, slot_id, patient_id, reason, urgency, datetime.now().isoformat()),
            )

        return AppointmentResult(
            success=True,
            appointment_id=booking_id,
            doctor_name=doctor_row["name"],
            slot=slot_row["slot_datetime"],
            patient_id=patient_id,
            message=f"Appointment booked with {doctor_row['name']} on {slot_row['slot_datetime']}.",
        )

    def get_bookings(self, db_path: str, patient_id: str) -> list[dict]:
        """Return all bookings for a patient.

        Args:
            db_path: Path to the SQLite database file.
            patient_id: Patient slug to query.

        Returns:
            List of booking dicts with booking_id, slot_id, doctor name, slot_datetime, urgency.
        """
        with connect_row_factory(db_path) as conn:
            rows = conn.execute(
                """
                SELECT b.booking_id, b.slot_id, d.name AS doctor_name,
                       s.slot_datetime, b.urgency, b.reason, b.booked_at
                FROM bookings b
                JOIN slots s ON b.slot_id = s.slot_id
                JOIN doctors d ON s.doctor_id = d.doctor_id
                WHERE b.patient_id = ?
                ORDER BY s.slot_datetime
                """,
                (patient_id,),
            ).fetchall()
        return [dict(r) for r in rows]

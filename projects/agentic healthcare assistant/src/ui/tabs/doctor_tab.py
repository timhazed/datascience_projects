"""Streamlit Doctor tab — schedule, availability, encounter notes, record updates."""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from src.db.metrics_db import MetricsDB
from src.db.patient_db import PatientDB
from src.ui.components.patient_record import _render_patient_record
from src.ui.constants import _HISTORY_FIELDS
from src.ui.graph_runtime import set_resolved_patient_with_hydration
from src.ui.metrics_session import _record_metric
from src.ui.runtime import settings

logger = logging.getLogger(__name__)


def _render_doctor_availability() -> None:
    """Render a read-only doctor availability table from the SQLite slots table.

    Queries available slot counts grouped by doctor and specialty so clinical
    users can see which physicians have openings before submitting a booking.
    Uses a direct SQLite connection — no graph invocation required.
    """
    db_path = str(settings.db.sqlite_path)
    if not os.path.exists(db_path):
        st.caption(
            "Doctor availability is loading. "
            "If this persists, check the database configuration."
        )
        return

    try:
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT d.name AS doctor, d.specialty,
                       COUNT(s.slot_id)                                 AS total_slots,
                       SUM(CASE WHEN s.booked = 0 THEN 1 ELSE 0 END)   AS available
                FROM doctors d
                JOIN slots s ON d.doctor_id = s.doctor_id
                GROUP BY d.doctor_id
                ORDER BY d.specialty, d.name
                """
            ).fetchall()
    except sqlite3.Error as exc:
        st.warning(f"Could not load doctor availability: {exc}")
        return

    if not rows:
        st.caption("No doctors seeded yet.")
        return

    data = [
        {
            "Doctor": r["doctor"],
            "Specialty": r["specialty"],
            "Available": int(r["available"]),
            "Total slots": int(r["total_slots"]),
        }
        for r in rows
    ]
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


def _render_schedule(days: int, label: str) -> None:
    """Render a schedule table of booked appointments over a date range.

    Queries the bookings, slots, doctors, and patients tables directly so the
    view is always current without requiring a graph invocation.

    Args:
        days: Number of days ahead to include (0 = today only, 6 = Mon–Sun of current week).
        label: Human-readable label for the empty-state message.
    """
    db_path = str(settings.db.sqlite_path)
    if not os.path.exists(db_path):
        st.caption("Database not initialised yet.")
        return

    today = date.today()
    from_dt = datetime.combine(today, datetime.min.time()).isoformat()
    to_dt = (
        datetime.combine(today, datetime.min.time()) + timedelta(days=days)
    ).replace(hour=23, minute=59).isoformat()

    try:
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT d.name  AS doctor,
                       d.specialty,
                       s.slot_datetime,
                       b.patient_id,
                       p.name  AS patient_name,
                       b.urgency,
                       b.reason
                FROM   bookings b
                JOIN   slots    s ON b.slot_id    = s.slot_id
                JOIN   doctors  d ON s.doctor_id  = d.doctor_id
                LEFT JOIN patients p ON b.patient_id = p.patient_id
                WHERE  s.slot_datetime >= ?
                  AND  s.slot_datetime <= ?
                ORDER BY s.slot_datetime, d.name
                """,
                (from_dt, to_dt),
            ).fetchall()
    except sqlite3.Error as exc:
        st.warning(f"Could not load schedule: {exc}")
        return

    if not rows:
        st.info(f"No appointments booked {label}.")
        return

    schedule_rows = []
    for r in rows:
        try:
            dt = datetime.fromisoformat(r["slot_datetime"])
            dt_str = dt.strftime("%a %b %d  %I:%M %p")
        except ValueError:
            dt_str = r["slot_datetime"]
        schedule_rows.append({
            "Date & Time": dt_str,
            "Doctor": r["doctor"],
            "Specialty": r["specialty"],
            "Patient": r["patient_name"] or r["patient_id"],
            "Urgency": (r["urgency"] or "routine").title(),
            "Reason": r["reason"] or "—",
        })

    st.dataframe(pd.DataFrame(schedule_rows), use_container_width=True, hide_index=True)


def _render_tab_doctor(
    graph: CompiledStateGraph,
    patient_db: PatientDB,
    metrics_db: MetricsDB | None = None,
) -> None:
    """Render Tab 2 — Doctor: clinical write actions for physician/admin users.

    Sections:
      - Find a Patient (separate search key from Patient tab)
      - Patient Record (read-only display)
      - Appointment Schedule (Today / This Week)
      - Doctor Availability (slot counts by specialty)
      - Enter Encounter Note (timestamped append to notes)
      - Update Record (structured field edits)

    No graph invocation — all operations are direct DB writes.
    New patient registration belongs in the Patient (MA) tab.

    Args:
        graph: Compiled LangGraph graph — used for checkpoint-based hydration on patient select.
        patient_db: PatientDB for record display and updates.
        metrics_db: MetricsDB for cross-session persistence.
    """
    db_path = str(settings.db.sqlite_path)

    # --- Find a Patient (doc_search key avoids widget key collision with Patient tab) ---
    st.subheader("Find a Patient")
    doc_search_term = st.text_input(
        "Search by name",
        key="doc_search",
        placeholder="Type any part of a name — e.g. 'Ram', 'David', 'Anjali'",
    )
    if doc_search_term:
        matches = patient_db.fuzzy_search(db_path, doc_search_term, limit=20)
        if matches:
            for r in matches:
                col_name, col_info, col_btn = st.columns([3, 2, 1])
                col_name.write(r.name)
                col_info.caption(f"{r.age} · {r.gender}")
                if col_btn.button("Select", key=f"doc_sel_{r.patient_id}"):
                    if r.patient_id != st.session_state.get("resolved_patient_id"):
                        _record_metric("Patient Lookup", success=True, metrics_db=metrics_db)
                    set_resolved_patient_with_hydration(
                        graph,
                        r.patient_id,
                        r.name,
                    )
                    st.rerun()
        else:
            st.info("No patients found matching that name.")

    # --- Patient Record (read-only) ---
    pid = st.session_state.get("resolved_patient_id")
    resolved_display = st.session_state.get("resolved_patient_name", "")

    if pid:
        flash = st.session_state.pop("_doctor_field_update_flash", None)
        if flash:
            st.success(flash)
        record = patient_db.get_patient(db_path, pid)
        st.divider()
        if record:
            st.subheader(f"Record — {record.name}")
            _render_patient_record(record)
            with st.expander("\U0001f4cb Clinical Notes & History", expanded=False):
                if record.notes:
                    st.markdown(record.notes.replace("\n", "  \n"))
                else:
                    st.caption("No clinical notes on file.")
        else:
            st.warning("Patient resolved but record not found in database.")
    else:
        st.info("Find a patient above to view their record.")

    # --- Today's Schedule / This Week's Schedule ---
    st.divider()
    st.subheader("Appointment Schedule")
    sched_tab_today, sched_tab_week = st.tabs(["Today", "This Week"])

    with sched_tab_today:
        _render_schedule(days=0, label="today")

    with sched_tab_week:
        _render_schedule(days=6, label="this week")

    # --- Doctor Availability (open slot counts) ---
    st.divider()
    st.subheader("Doctor Availability")
    st.caption("Open slots remaining by doctor and specialty.")
    _render_doctor_availability()

    # --- Enter Encounter Note ---
    st.divider()
    st.subheader("Enter Encounter Note")
    if not pid:
        st.info("Find a patient above to enable encounter notes.")
    else:
        st.caption(f"Adding note for: **{resolved_display}**")
        note_text = st.text_area(
            "Note",
            key="doc_note",
            max_chars=2000,
            placeholder="Subjective findings, diagnosis, assessment, plan…",
        )
        if st.button("Save Note", disabled=not note_text, key="doc_save_note"):
            # Prepend ISO timestamp header so notes form a dated log rather than a blob
            timestamp_header = (
                f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')} — Clinical encounter]\n"
            )
            with st.spinner("Saving…"):
                try:
                    ok = patient_db.update_field(
                        db_path, pid, "notes", timestamp_header + note_text, "append"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "[doctor_tab] encounter note save failed: %s — %s",
                        type(exc).__name__,
                        exc,
                    )
                    ok = False
            if ok:
                st.success("Encounter note saved.")
                _record_metric("Encounter Note", success=True, metrics_db=metrics_db)
                st.rerun()
            else:
                _record_metric(
                    "Encounter Note", success=False,
                    error="Patient record not found.", metrics_db=metrics_db,
                )
                st.error("Save failed — patient record not found.")

    # --- Update Record ---
    st.divider()
    st.subheader("Update Patient Record")
    if not pid:
        st.info("Find a patient above to enable record updates.")
    else:
        st.caption(f"Updating record for: **{resolved_display}**")
        field_select = st.selectbox("Field to update", _HISTORY_FIELDS, key="doc_field")
        update_value = st.text_area("New value", max_chars=500, key="doc_value")
        op_radio = st.radio(
            "Operation", ["append", "replace"], horizontal=True, key="doc_op"
        )
        if st.button("Update", disabled=not update_value, key="doc_update"):
            with st.spinner("Updating…"):
                try:
                    ok = patient_db.update_field(
                        db_path, pid, field_select, update_value, op_radio
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "[doctor_tab] record update failed: %s — %s", type(exc).__name__, exc
                    )
                    ok = False
            if ok:
                _record_metric("Update Record", success=True, metrics_db=metrics_db)
                st.session_state["_doctor_field_update_flash"] = (
                    f"Field '{field_select}' updated successfully."
                )
                # Rerun so the Record section re-fetches from DB; on the same run, Record is
                # rendered above this button before the DB write, so it would stay stale.
                st.rerun()
            else:
                _record_metric(
                    "Update Record", success=False,
                    error="Patient record not found.", metrics_db=metrics_db,
                )
                st.error("Update failed — patient record not found.")

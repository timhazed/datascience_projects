"""Streamlit Patient tab — MA workflow: find patient, record, appointments, registration."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import sqlite3
import time
from collections import defaultdict
from datetime import datetime

import pandas as pd
import pypdf
import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from src.db.appointment_db import AppointmentDB
from src.db.metrics_db import MetricsDB
from src.db.patient_db import PatientDB
from src.db.patient_vector_store import PatientVectorStore
from src.models.patient import PatientRecord
from src.ui.components.patient_record import _render_patient_record
from src.ui.constants import _SPECIALTIES
from src.ui.graph_runtime import set_resolved_patient_with_hydration
from src.ui.metrics_session import _record_metric
from src.ui.runtime import settings

logger = logging.getLogger(__name__)


def _render_tab_patient(
    graph: CompiledStateGraph,
    patient_db: PatientDB,
    appt_db: AppointmentDB,
    vector_store: PatientVectorStore,
    metrics_db: MetricsDB | None = None,
) -> None:
    """Render Tab 1 — Patient: find patient, view record, appointments, booking, and registration.

    This is the Medical Assistant view. The MA manages the full patient lifecycle —
    finding existing patients, reviewing records, booking appointments, and registering
    new patients with PDF clinical reports.

    Args:
        graph: Compiled LangGraph graph (cached singleton).
        patient_db: PatientDB for direct record lookups.
        appt_db: AppointmentDB for persistent appointment history.
        vector_store: PatientVectorStore for FAISS upsert on new patient registration.
        metrics_db: MetricsDB for cross-session persistence.
    """
    db_path = str(settings.db.sqlite_path)

    # --- Find a Patient — always visible so the MA's first action is search ---
    st.subheader("Find a Patient")
    search_term = st.text_input(
        "Search by name",
        key="hist_search",
        placeholder="Type any part of a name — e.g. 'Ram', 'David', 'Anjali'",
    )
    if search_term:
        matches = patient_db.fuzzy_search(db_path, search_term, limit=20)
        if matches:
            for r in matches:
                col_name, col_info, col_btn = st.columns([3, 2, 1])
                # write() prevents DB-sourced names from being rendered as markdown
                col_name.write(r.name)
                col_info.caption(f"{r.age} · {r.gender}")
                if col_btn.button("Select", key=f"sel_{r.patient_id}"):
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

    # --- My Record — shown when patient is resolved ---
    resolved_pid = st.session_state.get("resolved_patient_id")
    resolved_name = st.session_state.get("resolved_patient_name")

    if resolved_pid:
        record = patient_db.get_patient(db_path, resolved_pid)
        st.divider()
        if record:
            st.subheader(f"Record — {record.name}")
            _render_patient_record(record)

            # Clinical Notes in a dedicated expander below the structured fields.
            # st.markdown with line-break preservation improves readability over st.text().
            with st.expander("\U0001f4cb Clinical Notes & History", expanded=False):
                if record.notes:
                    st.markdown(record.notes.replace("\n", "  \n"))
                else:
                    st.caption("No clinical notes on file.")
        else:
            st.warning("Patient resolved but record not found in database.")

        # --- My Appointments — persistent history from AppointmentDB ---
        st.divider()
        st.subheader("My Appointments")
        try:
            bookings = appt_db.get_bookings(db_path, resolved_pid)
        except Exception as exc:  # noqa: BLE001
            logger.error("[patient_tab] get_bookings failed: %s — %s", type(exc).__name__, exc)
            bookings = []

        if bookings:
            appt_rows = []
            for b in bookings:
                # slot_datetime is stored as ISO string; parse for display
                try:
                    dt = datetime.fromisoformat(b["slot_datetime"])
                    dt_str = dt.strftime("%b %d %Y  %I:%M %p")
                except (ValueError, KeyError):
                    dt_str = b.get("slot_datetime", "—")
                appt_rows.append({
                    "Doctor": b.get("doctor_name", "—"),
                    "Date & Time": dt_str,
                    "Urgency": b.get("urgency", "—").title(),
                    "Reason": b.get("reason", "—"),
                })
            st.dataframe(pd.DataFrame(appt_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No appointments on record.")

        # --- Book Appointment ---
        st.divider()
        _render_book_appointment_form(appt_db, resolved_name, metrics_db=metrics_db)
    else:
        st.info(
            "No patient selected. Use **Find a Patient** above or enter a name in the "
            "**Active Patient** panel on the left."
        )

    # --- Register New Patient — always visible; MA may need to onboard before selecting ---
    st.divider()
    with st.expander("\u2795 Register New Patient"):
        st.caption("Add a new patient to the system. PDF clinical report is required.")
        new_name = st.text_input("Full name *", key="reg_name")
        new_phone = st.text_input("Phone *", key="reg_phone")
        new_email = st.text_input("Email", key="reg_email")
        new_age = st.number_input("Age", min_value=0, max_value=130, value=30, key="reg_age")
        new_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="reg_gender")
        new_address = st.text_input("Address", key="reg_address")
        new_patient_pdf = st.file_uploader(
            "Clinical report (PDF, max 10 MB) *",
            type=["pdf"],
            key="reg_pdf",
        )
        reg_disabled = not new_name or not new_phone or new_patient_pdf is None
        if st.button("Register", disabled=reg_disabled, key="reg_submit"):
            _register_new_patient(
                db_path=db_path,
                patient_db=patient_db,
                vector_store=vector_store,
                metrics_db=metrics_db,
                graph=graph,
                name=new_name,
                phone=new_phone,
                email=new_email or "",
                age=int(new_age),
                gender=new_gender,
                address=new_address,
                pdf_file=new_patient_pdf,
            )


def _render_book_appointment_form(
    appt_db: AppointmentDB,
    resolved_name: str | None,
    metrics_db: MetricsDB | None = None,
) -> None:
    """Render the Book Appointment form with live slot availability.

    Queries available slots directly from AppointmentDB when specialty or
    urgency changes, shows available dates then times as selectboxes, and
    books the chosen slot atomically via appt_db.book_slot() — bypassing the
    graph for this deterministic write.  When specialty, urgency, or the chosen
    date changes, cached Streamlit widget state for date/time pickers is cleared
    so labels and slot indices stay in sync with the newly queried slots.
    A st.rerun() after booking refreshes My Appointments without requiring
    a separate user action.

    Args:
        appt_db: AppointmentDB instance for slot queries and booking.
        resolved_name: Resolved patient display name, or None if not yet resolved.
        metrics_db: MetricsDB for cross-session persistence. Pass None only when
            the DB is unavailable.
    """
    st.subheader("Book Appointment")
    db_path = str(settings.db.sqlite_path)
    resolved_id = st.session_state.get("resolved_patient_id")
    context_label = (
        f"Booking for: **{resolved_name}**" if resolved_name else "No patient resolved yet."
    )
    st.caption(context_label)

    specialty_select = st.selectbox("Specialty", _SPECIALTIES, key="appt_specialty")
    urgency_radio = st.radio(
        "Urgency",
        ["Routine", "Urgent", "Emergency"],
        horizontal=True,
        key="appt_urgency",
    )

    # Streamlit keeps selectbox state by key; the time picker uses integer indices into
    # the current slot list, so a stale index + format_func shows the wrong doctor until
    # the user re-opens the dropdown. Reset date/time when filters change.
    _filter_sig = f"{specialty_select}|{urgency_radio}"
    if st.session_state.get("_appt_filter_sig") != _filter_sig:
        st.session_state["_appt_filter_sig"] = _filter_sig
        st.session_state.pop("appt_date_select", None)
        st.session_state.pop("appt_time_select", None)

    # Query live available slots for the chosen specialty + urgency.
    # Results are cached per run by Streamlit's re-render model — no explicit
    # caching needed because this runs every rerun when the user interacts.
    try:
        available_slots = appt_db.query_slots(
            db_path, specialty_select, urgency=urgency_radio.lower()
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[patient_tab] query_slots failed: %s — %s", type(exc).__name__, exc)
        available_slots = []

    if not available_slots:
        st.warning(
            f"No available slots for {specialty_select} ({urgency_radio}). "
            "Try a different specialty or urgency level."
        )
        return

    # Build a date → list of slots mapping so the user first picks a date,
    # then picks a time from only the slots available on that date.
    slots_by_date: dict[str, list[dict]] = defaultdict(list)
    for s in available_slots:
        try:
            slot_dt = datetime.fromisoformat(s["slot_datetime"])
            date_key = slot_dt.strftime("%a %b %d, %Y")  # e.g. "Mon Apr 14, 2026"
        except (ValueError, KeyError):
            date_key = s["slot_datetime"][:10]
        slots_by_date[date_key].append(s)

    available_dates = list(slots_by_date.keys())
    selected_date_label = st.selectbox(
        "Available date",
        available_dates,
        key="appt_date_select",
    )

    # Same index issue as specialty: changing the date replaces the time-slot list.
    _slot_ctx = f"{specialty_select}|{urgency_radio}|{selected_date_label}"
    if st.session_state.get("_appt_slot_ctx") != _slot_ctx:
        st.session_state["_appt_slot_ctx"] = _slot_ctx
        st.session_state.pop("appt_time_select", None)

    # Resolve doctor names for the time slots on the selected date via a single DB lookup.
    # doctor_id_to_name falls back to the raw doctor_id string if the lookup fails.
    time_slots = slots_by_date[selected_date_label]
    doc_ids = list({s.get("doctor_id", "") for s in time_slots if s.get("doctor_id")})
    doctor_id_to_name: dict[str, str] = {}
    if doc_ids:
        try:
            with contextlib.closing(sqlite3.connect(db_path)) as _conn:
                _conn.row_factory = sqlite3.Row
                placeholders = ",".join("?" * len(doc_ids))
                for row in _conn.execute(
                    f"SELECT doctor_id, name FROM doctors WHERE doctor_id IN ({placeholders})",
                    doc_ids,
                ).fetchall():
                    doctor_id_to_name[row["doctor_id"]] = row["name"]
        except sqlite3.Error:
            pass  # Fall back to doctor_id string if lookup fails

    enriched_labels = []
    for s in time_slots:
        try:
            slot_dt = datetime.fromisoformat(s["slot_datetime"])
            doc_name = doctor_id_to_name.get(s.get("doctor_id", ""), s.get("doctor_id", ""))
            label = f"{slot_dt.strftime('%I:%M %p')}  —  {doc_name}"
        except (ValueError, KeyError):
            label = s["slot_datetime"]
        enriched_labels.append((label, s["slot_id"], s["slot_datetime"]))

    # Index-based selection avoids the label-collision bug that arises when two slots
    # produce the same display string (e.g. doctor lookup failure → both show empty name).
    # format_func renders the label; the selectbox value is the integer index into
    # enriched_labels, so selected_slot is always the exact slot the user chose.
    selected_idx = st.selectbox(
        "Available time",
        options=range(len(enriched_labels)),
        format_func=lambda i: enriched_labels[i][0],
        key="appt_time_select",
    )
    selected_slot = enriched_labels[selected_idx] if enriched_labels else None

    reason_input = st.text_area(
        "Reason",
        max_chars=300,
        key="appt_reason",
        placeholder="e.g. Annual checkup, follow-up for hypertension, CKD review…",
    )

    book_disabled = not resolved_id or selected_slot is None
    if st.button("Book Appointment", disabled=book_disabled, key="appt_book"):
        with st.spinner("Booking…"):
            _book_start = time.perf_counter()
            try:
                appt_result = appt_db.book_slot(
                    db_path,
                    slot_id=selected_slot[1],
                    patient_id=resolved_id,
                    reason=reason_input or "follow-up",
                    urgency=urgency_radio.lower(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("[patient_tab] book_slot failed: %s — %s", type(exc).__name__, exc)
                appt_result = None
            _book_ms = round((time.perf_counter() - _book_start) * 1000, 1)

        if appt_result and appt_result.success:
            try:
                confirmed_dt = datetime.fromisoformat(appt_result.slot)
                confirmed_str = confirmed_dt.strftime("%b %d %Y  %I:%M %p")
            except (ValueError, TypeError):
                confirmed_str = appt_result.slot or selected_slot[2]

            st.success("Appointment booked.")
            with st.container(border=True):
                st.markdown("##### Booking Confirmation")
                st.markdown(f"**Doctor:** {appt_result.doctor_name}")
                st.markdown(f"**Specialty:** {specialty_select}")
                st.markdown(f"**Confirmed slot:** {confirmed_str}")
                st.markdown(f"**Urgency:** {urgency_radio}")
                if reason_input:
                    st.write(f"**Reason:** {reason_input}")

            # Record in session metrics so the Metrics tab captures the booking
            _record_metric(
                "Book Appointment", success=True, latency_ms=_book_ms, metrics_db=metrics_db
            )
            # Rerun immediately so My Appointments refreshes without a separate action
            st.rerun()
        else:
            msg = (appt_result.message if appt_result else "Booking failed — please try again.")
            st.error(msg)


def _register_new_patient(
    *,
    db_path: str,
    patient_db: PatientDB,
    vector_store: PatientVectorStore,
    name: str,
    phone: str,
    email: str,
    age: int,
    gender: str,
    address: str,
    pdf_file: object,
    metrics_db: MetricsDB | None = None,
    graph: CompiledStateGraph,
) -> None:
    """Execute the new patient registration flow.

    Steps:
      1. Generate slug from phone + name via sha256[:6].
      2. INSERT demographics into SQLite (INSERT OR IGNORE — duplicate slug shows info).
      3. Extract PDF text via pypdf.PdfReader.
      4. Update patients.notes with PDF text.
      5. Upsert vector into FAISS with required metadata.
      6. Show st.success or st.error.

    Args:
        db_path: SQLite database path.
        patient_db: PatientDB instance.
        vector_store: PatientVectorStore for FAISS upsert.
        name: Patient full name.
        phone: Patient phone number (used in slug generation).
        email: Optional email address.
        age: Patient age in years.
        gender: Patient gender.
        address: Optional address.
        pdf_file: Streamlit UploadedFile with clinical PDF.
        metrics_db: MetricsDB for cross-session persistence.
        graph: Compiled LangGraph graph — used for checkpoint-based hydration after registration.
    """
    # 1. Derive deterministic slug
    patient_id = "P-" + hashlib.sha256((phone + name).encode()).hexdigest()[:6]

    # 2. INSERT demographics
    record = PatientRecord(
        patient_id=patient_id,
        phone=phone,
        email=email or None,
        name=name,
        age=age,
        gender=gender,
        address=address,
        last_updated=datetime.now(),
    )
    patient_db.create_patient(db_path, record)

    # 3. Extract PDF text (pypdf.PdfReader expects a file-like object)
    try:
        reader = pypdf.PdfReader(pdf_file)
        notes = "\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("[register] PDF extraction failed: %s — %s", type(exc).__name__, exc)
        st.error(f"PDF extraction failed: {type(exc).__name__}")
        return

    # 4. Persist raw PDF text to patients.notes
    patient_db.update_notes(db_path, patient_id, notes)

    # 5. Upsert into FAISS — all 6 required metadata fields must be present
    try:
        vector_store.upsert(
            patient_id=patient_id,
            text=notes,
            metadata={
                "patient_id": patient_id,
                "name": name,
                "age": age,
                "gender": gender,
                "summary": "",        # no summary yet for new patients
                "conditions": "[]",   # no conditions yet
                "source": "pdf",
                "pdf_filename": getattr(pdf_file, "name", "upload.pdf"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[register] FAISS upsert failed: %s — %s", type(exc).__name__, exc)
        st.error(f"FAISS index error: {type(exc).__name__}")
        return

    # Store slug internally; surface the human-readable name — slug never shown in UI
    set_resolved_patient_with_hydration(
        graph,
        patient_id,
        name,
    )
    _record_metric("Register Patient", success=True, metrics_db=metrics_db)
    st.success(f"Patient registered — {name} is now active in this session.")

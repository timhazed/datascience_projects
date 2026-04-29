"""Read-only patient record rendering for Streamlit."""

from __future__ import annotations

import streamlit as st

from src.models.patient import PatientRecord


def _render_patient_record(record: PatientRecord) -> None:
    """Render a PatientRecord as labelled sections instead of raw JSON.

    Displays demographics in a two-column grid, list fields as bullet points,
    and the clinical notes in an expander so long text doesn't dominate the page.

    Args:
        record: PatientRecord fetched from SQLite after successful retrieval.
    """
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Name:** {record.name}")
        st.markdown(f"**Age:** {record.age}")
        st.markdown(f"**Gender:** {record.gender}")
    with col2:
        st.markdown(f"**Phone:** {record.phone}")
        st.markdown(f"**Email:** {record.email or '—'}")
        st.markdown(f"**Address:** {record.address or '—'}")

    if record.summary:
        st.markdown("**Summary**")
        st.info(record.summary)

    for label, items in [
        ("Conditions", record.conditions),
        ("Medications", record.medications),
        ("Allergies", record.allergies),
    ]:
        st.markdown(f"**{label}**")
        if items:
            st.markdown("\n".join(f"- {item}" for item in items))
        else:
            st.caption("None recorded.")

    # Clinical notes are rendered separately by callers so they can be positioned
    # after other sections without duplicating the expander widget.

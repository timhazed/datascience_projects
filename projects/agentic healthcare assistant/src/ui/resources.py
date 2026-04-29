"""Cached Streamlit resources: graph and DB singletons."""

from __future__ import annotations

from typing import Any

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langgraph.graph.state import CompiledStateGraph

from src.db.appointment_db import AppointmentDB
from src.db.initializer import _ensure_db_initialized
from src.db.metrics_db import MetricsDB
from src.db.patient_db import PatientDB
from src.db.patient_vector_store import PatientVectorStore
from src.graph.healthcare_graph import build_graph
from src.ui.runtime import settings
from src.utils.search_provider import SearchProvider


@st.cache_resource(show_spinner="Initialising healthcare system…")
def _load_resources() -> (  # noqa: E501
    tuple[CompiledStateGraph, PatientDB, AppointmentDB, MetricsDB, PatientVectorStore, Any]
):
    """Build and cache graph + DB singletons for the entire server session.

    Calls _ensure_db_initialized() so the first startup seeds SQLite and FAISS.
    Subsequent calls return the cached tuple without re-running.

    Returns:
        Tuple of (
            graph,
            patient_db,
            appt_db,
            metrics_db,
            vector_store,
            checkpointer,
        ).
    """
    embeddings = OpenAIEmbeddings(model=settings.embeddings.model)
    _ensure_db_initialized(settings, embeddings)

    search_provider = SearchProvider(settings.search.provider)
    result = build_graph(settings, search_fn=search_provider.search)

    # Standalone DB instances for direct tab operations (bypass graph for determinism)
    patient_db = PatientDB()
    appt_db = AppointmentDB()
    metrics_db = MetricsDB()
    vector_store = PatientVectorStore(str(settings.db.faiss_path), embeddings)

    return (
        result.graph,
        patient_db,
        appt_db,
        metrics_db,
        vector_store,
        result.checkpointer,
    )

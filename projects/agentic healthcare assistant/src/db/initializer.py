"""initializer — ensures the database is seeded before the first query.

Called by both src/cli/run_query.py and src/ui/resources.py (`_load_resources`) at startup.
Has no Streamlit dependency — safe to import from any entry point.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from src.config.settings import Settings
from src.db.appointment_db import AppointmentDB
from src.db.data_loader import DataLoader
from src.db.metrics_db import MetricsDB
from src.db.patient_db import PatientDB

logger = logging.getLogger(__name__)

_DATASET_DIR = "dataset"

_patient_db = PatientDB()
_appointment_db = AppointmentDB()
_metrics_db = MetricsDB()
_data_loader = DataLoader()


def _is_already_seeded(db_path: str, faiss_path: str) -> bool:
    """Return True if both SQLite and FAISS already contain patient data.

    Both stores must be present: if either is missing the loader must run to
    bring them into a consistent state (e.g. FAISS deleted but SQLite intact).

    Args:
        db_path: SQLite database path.
        faiss_path: FAISS index directory path.
    """
    if not (Path(faiss_path) / "index.faiss").exists():
        return False
    try:
        with sqlite3.connect(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
            return count > 0
    except sqlite3.OperationalError:
        # Table does not exist yet — DB not yet initialized.
        return False


def _ensure_db_initialized(settings: Settings, embeddings_model: object) -> None:
    """Create tables, seed slots, and load all patient data if not already done.

    Skips the expensive data-load step (PDF extraction + OpenAI embedding) when
    both SQLite and the FAISS index already contain patient data from a prior run.
    Schema initialization (CREATE TABLE IF NOT EXISTS, slot seeding) always runs
    so structural changes are picked up without a manual reset.

    Execution order (required):
      1. appointment_db.init_db() — creates doctors, slots, bookings tables + seed
      2. patient_db.init_db()     — creates patients table + indexes
      3. metrics_db.init_db()     — creates session_log table + indexes
      4. data_loader.load_all()   — loads xlsx + PDFs into SQLite + FAISS
                                    (skipped when already seeded)

    Args:
        settings: Loaded Settings instance (provides db paths).
        embeddings_model: LangChain-compatible embeddings instance for FAISS.
    """
    db_path = str(settings.db.sqlite_path)
    faiss_path = str(settings.db.faiss_path)

    logger.info("Initializing appointment DB at %s", db_path)
    _appointment_db.init_db(db_path)

    logger.info("Initializing patient DB at %s", db_path)
    _patient_db.init_db(db_path)

    logger.info("Initializing metrics DB at %s", db_path)
    _metrics_db.init_db(db_path)

    # Drop legacy pre-checkpointing tables if they survived from a prior schema.
    # These tables were written by PatientMemoryDB (removed in Phase 4) and are
    # no longer read or written by any code path. Safe to drop unconditionally.
    with sqlite3.connect(db_path) as _conn:
        _conn.execute("DROP TABLE IF EXISTS patient_memory_messages")
        _conn.execute("DROP TABLE IF EXISTS patient_memory_summary")
        _conn.commit()

    if _is_already_seeded(db_path, faiss_path):
        logger.info("Patient data already seeded — skipping data load")
    else:
        logger.info("Loading patient data from %s", _DATASET_DIR)
        _data_loader.load_all(db_path, faiss_path, _DATASET_DIR, embeddings_model)

    logger.info("DB initialization complete")

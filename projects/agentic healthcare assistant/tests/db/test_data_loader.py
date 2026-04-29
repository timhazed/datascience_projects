"""Tests for DataLoader — real xlsx fixtures + mock PDF extraction, mocked embeddings.

SQLite is real (in_memory_db fixture). FAISS embeddings are mocked.
PDF text extraction (_extract_pdf_text) is patched — we write stub .pdf files
so DataLoader's rglob("*.pdf") discovers them, but avoid a PDF-writing dependency.
No external API calls, no live LLM.

Covered scenarios:
  - xlsx loading: header auto-detection (row 0 and row 2), dedup via INSERT OR IGNORE
  - PDF match to xlsx-backed patient (both naming conventions)
  - PDF-only patient path: demographics extracted and new row inserted
  - Unmatchable PDF skipped with a warning (no crash)
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import openpyxl
import pytest
from langchain_core.embeddings import Embeddings

from src.db.data_loader import DataLoader
from src.db.patient_db import PatientDB

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_PATCH_TARGET = "src.db.data_loader.DataLoader._extract_pdf_text"


def _write_xlsx_original(path: Path, rows: list[dict]) -> None:
    """Write an xlsx with the original_data layout (header on row index 0)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    headers = ["Phone_number", "Email", "Name", "Age", "Gender", "Address", "Summary"]
    ws.append(headers)
    for row in rows:
        ws.append([row.get(h, "") for h in headers])
    wb.save(path)


def _write_xlsx_generated(path: Path, rows: list[dict]) -> None:
    """Write an xlsx with the generated_data layout (header on row index 2)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["records1"])            # row index 0 — label row
    ws.append(["additional_records"])  # row index 1 — label row
    headers = ["Phone_number", "Email", "Name", "Age", "Gender", "Address", "Summary"]
    ws.append(headers)                 # row index 2 — actual header
    for row in rows:
        ws.append([row.get(h, "") for h in headers])
    wb.save(path)


def _stub_pdf(path: Path) -> None:
    """Create a zero-byte stub file with a .pdf extension for glob discovery."""
    path.write_bytes(b"")


def _make_embeddings_mock() -> MagicMock:
    """Return a mock embeddings model that returns a fixed [0.1]*1536 vector."""
    embeddings = MagicMock(spec=Embeddings)
    embeddings.embed_documents.return_value = [[0.1] * 1536]
    embeddings.embed_query.return_value = [0.1] * 1536
    return embeddings


# ---------------------------------------------------------------------------
# Unit tests for pure helpers
# ---------------------------------------------------------------------------


def test_make_slug_deterministic() -> None:
    """_make_slug must return the same value for the same inputs."""
    slug1 = DataLoader._make_slug("555-0100", "Alice Test")
    slug2 = DataLoader._make_slug("555-0100", "Alice Test")
    assert slug1 == slug2
    assert slug1.startswith("P-")
    assert len(slug1) == 8  # "P-" + 6 hex chars


def test_make_slug_differs_by_phone() -> None:
    """Different phone numbers must produce different slugs (with same name)."""
    assert DataLoader._make_slug("555-0001", "Same Name") != DataLoader._make_slug(
        "555-0002", "Same Name"
    )


def test_dob_to_age_valid() -> None:
    """_dob_to_age returns a non-negative integer for a valid DOB string."""
    age = DataLoader._dob_to_age("01/15/1980")
    assert isinstance(age, int)
    assert age >= 40  # At least 40 by 2026


def test_dob_to_age_invalid_returns_zero() -> None:
    """_dob_to_age returns 0 for an unparseable DOB string."""
    assert DataLoader._dob_to_age("not-a-date") == 0


# ---------------------------------------------------------------------------
# DataLoader.load_all — xlsx loading
# ---------------------------------------------------------------------------


def test_load_original_data_header_row_zero(tmp_path: Path, in_memory_db: str) -> None:
    """DataLoader detects the header on row index 0 (original_data layout)."""
    dataset = tmp_path / "original_data"
    dataset.mkdir()
    _write_xlsx_original(
        dataset / "records.xlsx",
        [{"Phone_number": "555-0001", "Name": "Alice Row0", "Age": "30", "Gender": "Female"}],
    )

    with patch(_PATCH_TARGET, return_value=""):
        DataLoader().load_all(
            in_memory_db, str(tmp_path / "faiss"), str(tmp_path), _make_embeddings_mock()
        )

    pdb = PatientDB()
    results = pdb.fuzzy_search(in_memory_db, "Alice Row0")
    assert len(results) == 1
    assert results[0].name == "Alice Row0"


def test_load_generated_data_header_row_two(tmp_path: Path, in_memory_db: str) -> None:
    """DataLoader detects the header on row index 2 (generated_data layout)."""
    dataset = tmp_path / "generated_data"
    dataset.mkdir()
    _write_xlsx_generated(
        dataset / "records.xlsx",
        [{"Phone_number": "555-0002", "Name": "Bob Row2", "Age": "45", "Gender": "Male"}],
    )

    with patch(_PATCH_TARGET, return_value=""):
        DataLoader().load_all(
            in_memory_db, str(tmp_path / "faiss"), str(tmp_path), _make_embeddings_mock()
        )

    pdb = PatientDB()
    results = pdb.fuzzy_search(in_memory_db, "Bob Row2")
    assert len(results) == 1


def test_duplicate_slug_dedup(tmp_path: Path, in_memory_db: str) -> None:
    """Three xlsx rows with identical phone+name (same slug) produce one DB row."""
    dataset = tmp_path / "original_data"
    dataset.mkdir()
    duplicate_row = {
        "Phone_number": "555-DUP", "Name": "Rebeca Nagle", "Age": "35", "Gender": "Female"
    }
    _write_xlsx_original(
        dataset / "records.xlsx",
        [duplicate_row, duplicate_row, duplicate_row],
    )

    with patch(_PATCH_TARGET, return_value=""):
        DataLoader().load_all(
            in_memory_db, str(tmp_path / "faiss"), str(tmp_path), _make_embeddings_mock()
        )

    pdb = PatientDB()
    results = pdb.fuzzy_search(in_memory_db, "Rebeca Nagle")
    assert len(results) == 1


# ---------------------------------------------------------------------------
# DataLoader.load_all — PDF matching
# ---------------------------------------------------------------------------


def test_pdf_match_report_first_last_convention(tmp_path: Path, in_memory_db: str) -> None:
    """report_{First}_{Last}.pdf matched to the xlsx patient updates notes."""
    dataset = tmp_path / "generated_data"
    dataset.mkdir()
    _write_xlsx_original(
        dataset / "records.xlsx",
        [{"Phone_number": "555-0010", "Name": "Carol Evans", "Age": "52", "Gender": "Female"}],
    )
    _stub_pdf(dataset / "report_Carol_Evans.pdf")

    pdf_text = "Clinical findings for Carol Evans."
    with patch(_PATCH_TARGET, return_value=pdf_text):
        DataLoader().load_all(
            in_memory_db, str(tmp_path / "faiss"), str(tmp_path), _make_embeddings_mock()
        )

    pdb = PatientDB()
    results = pdb.fuzzy_search(in_memory_db, "Carol Evans")
    assert len(results) == 1
    assert "Clinical findings" in (results[0].notes or "")


def test_pdf_match_sample_report_firstname_convention(tmp_path: Path, in_memory_db: str) -> None:
    """sample_report_{firstname}.pdf matched to the xlsx patient by first name."""
    dataset = tmp_path / "original_data"
    dataset.mkdir()
    _write_xlsx_original(
        dataset / "records.xlsx",
        [{"Phone_number": "555-0011", "Name": "David Chen", "Age": "60", "Gender": "Male"}],
    )
    _stub_pdf(dataset / "sample_report_david.pdf")

    pdf_text = "Sample report text for David."
    with patch(_PATCH_TARGET, return_value=pdf_text):
        DataLoader().load_all(
            in_memory_db, str(tmp_path / "faiss"), str(tmp_path), _make_embeddings_mock()
        )

    pdb = PatientDB()
    results = pdb.fuzzy_search(in_memory_db, "David Chen")
    assert results[0].notes and "Sample report" in results[0].notes


def test_pdf_match_embeds_into_faiss(tmp_path: Path, in_memory_db: str) -> None:
    """A matched PDF must result in a FAISS vector (embed_documents called)."""
    dataset = tmp_path / "original_data"
    dataset.mkdir()
    _write_xlsx_original(
        dataset / "records.xlsx",
        [{"Phone_number": "555-0012", "Name": "Eve Marsh", "Age": "33", "Gender": "Female"}],
    )
    _stub_pdf(dataset / "report_Eve_Marsh.pdf")

    embeddings = _make_embeddings_mock()
    with patch(_PATCH_TARGET, return_value="Eve Marsh clinical data."):
        DataLoader().load_all(in_memory_db, str(tmp_path / "faiss"), str(tmp_path), embeddings)

    embeddings.embed_documents.assert_called()


# ---------------------------------------------------------------------------
# DataLoader.load_all — PDF-only patient path
# ---------------------------------------------------------------------------

_PDF_ONLY_TEXT = (
    "Jasmine Lee\n"
    "2024-03-15 10:00\n"
    "Location: Main Clinic\n"
    "Patient #: D8199135811  DOB: 12/01/1988\n"
    "Gender: Female\n"
    "Diagnosis: Seasonal allergies, mild asthma.\n"
)


def test_pdf_only_patient_inserted(tmp_path: Path, in_memory_db: str) -> None:
    """A PDF with no matching xlsx row inserts a new patient via PDF-only path."""
    dataset = tmp_path / "original_data"
    dataset.mkdir()
    _write_xlsx_original(dataset / "records.xlsx", [])  # empty xlsx
    _stub_pdf(dataset / "report_Jasmine_Lee.pdf")

    with patch(_PATCH_TARGET, return_value=_PDF_ONLY_TEXT):
        DataLoader().load_all(
            in_memory_db, str(tmp_path / "faiss"), str(tmp_path), _make_embeddings_mock()
        )

    pdb = PatientDB()
    results = pdb.fuzzy_search(in_memory_db, "Jasmine Lee")
    assert len(results) == 1
    assert results[0].gender == "Female"
    assert results[0].age > 0


def test_pdf_only_patient_embedded_in_faiss(tmp_path: Path, in_memory_db: str) -> None:
    """A PDF-only patient must also be indexed in FAISS."""
    dataset = tmp_path / "original_data"
    dataset.mkdir()
    _write_xlsx_original(dataset / "records.xlsx", [])
    _stub_pdf(dataset / "report_Jasmine_Lee.pdf")

    embeddings = _make_embeddings_mock()
    with patch(_PATCH_TARGET, return_value=_PDF_ONLY_TEXT):
        DataLoader().load_all(in_memory_db, str(tmp_path / "faiss"), str(tmp_path), embeddings)

    embeddings.embed_documents.assert_called()


# ---------------------------------------------------------------------------
# DataLoader.load_all — unmatchable / empty PDF
# ---------------------------------------------------------------------------


def test_unmatchable_pdf_skipped_with_warning(
    tmp_path: Path, in_memory_db: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A PDF that cannot be matched and has no parseable header is skipped."""
    dataset = tmp_path / "original_data"
    dataset.mkdir()
    _write_xlsx_original(dataset / "records.xlsx", [])
    _stub_pdf(dataset / "report_unknown_patient.pdf")

    no_header_text = "No structured header here at all."
    loader = DataLoader()
    with caplog.at_level(logging.WARNING, logger="src.db.data_loader"):
        with patch(_PATCH_TARGET, return_value=no_header_text):
            loader.load_all(
                in_memory_db, str(tmp_path / "faiss"), str(tmp_path), _make_embeddings_mock()
            )

    assert any("skip" in record.message.lower() for record in caplog.records)
    pdb = PatientDB()
    assert pdb.fuzzy_search(in_memory_db, "unknown") == []


def test_empty_pdf_skipped_with_warning(
    tmp_path: Path, in_memory_db: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A PDF that yields empty text after extraction is skipped with a warning."""
    dataset = tmp_path / "original_data"
    dataset.mkdir()
    _write_xlsx_original(dataset / "records.xlsx", [])
    _stub_pdf(dataset / "report_empty.pdf")

    loader = DataLoader()
    with caplog.at_level(logging.WARNING, logger="src.db.data_loader"):
        with patch(_PATCH_TARGET, return_value=""):
            loader.load_all(
                in_memory_db, str(tmp_path / "faiss"), str(tmp_path), _make_embeddings_mock()
            )

    assert any("empty" in record.message.lower() for record in caplog.records)


# ---------------------------------------------------------------------------
# _parse_medications_from_notes — unit tests (no DB, no FAISS)
# ---------------------------------------------------------------------------


def test_parse_medications_hp_note_format() -> None:
    """Telmisartan 40mg OD extracted from a standard H&P Plan Notes block."""
    text = (
        "History and Physical Note\n"
        "Patient: Ramesh Kulkarni\n"
        "Plan Notes:\n"
        "Continue Telmisartan 40mg OD. Recommend lifestyle modifications.\n"
    )
    result = DataLoader._parse_medications_from_notes(text)
    assert result == ["Telmisartan 40mg OD"]


def test_parse_medications_multiple_drugs() -> None:
    """Multiple medications in Plan Notes are all captured."""
    text = (
        "Plan Notes:\n"
        "Start Metformin 1000mg BID. Continue Aspirin 81mg daily.\n"
    )
    result = DataLoader._parse_medications_from_notes(text)
    assert "Metformin 1000mg BID" in result
    assert "Aspirin 81mg daily" in result
    assert len(result) == 2


def test_parse_medications_action_words_stripped() -> None:
    """Action verbs (Continue, Start, Stop, etc.) do not appear in drug name."""
    text = "Plan Notes:\nStop Lisinopril 10mg OD.\n"
    result = DataLoader._parse_medications_from_notes(text)
    assert len(result) == 1
    assert result[0].startswith("Lisinopril")
    assert "Stop" not in result[0]


def test_parse_medications_no_plan_section_returns_empty() -> None:
    """Returns [] when no Plan Notes section exists in the text."""
    text = "Subjective Notes:\nPatient feels well. No complaints.\n"
    result = DataLoader._parse_medications_from_notes(text)
    assert result == []


def test_parse_medications_lifestyle_only_no_drugs() -> None:
    """Plan Notes with no drug/dose pattern returns []."""
    text = "Plan Notes:\nLifestyle modifications. Routine labs ordered. Return in 6 months.\n"
    result = DataLoader._parse_medications_from_notes(text)
    assert result == []


def test_parse_medications_dose_units_variety() -> None:
    """Parser handles mcg and g units, not just mg."""
    text = "Plan:\nIncrease Levothyroxine 50mcg OD.\n"
    result = DataLoader._parse_medications_from_notes(text)
    assert len(result) == 1
    assert "Levothyroxine" in result[0]
    assert "50mcg" in result[0]


def test_parse_medications_empty_string_returns_empty() -> None:
    """Empty input returns []."""
    assert DataLoader._parse_medications_from_notes("") == []

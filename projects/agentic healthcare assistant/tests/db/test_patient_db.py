"""Tests for PatientDB — real SQLite via in_memory_db fixture."""

import pytest

from src.db.patient_db import PatientDB
from src.models.patient import PatientRecord


def _make_record(
    patient_id: str = "P-abc123",
    phone: str = "555-0100",
    name: str = "Alice Tester",
    age: int = 40,
    gender: str = "Female",
) -> PatientRecord:
    """Return a minimal PatientRecord for use in tests."""
    return PatientRecord(patient_id=patient_id, phone=phone, name=name, age=age, gender=gender)


@pytest.fixture
def db(in_memory_db: str) -> tuple[PatientDB, str]:
    """Return (PatientDB instance, db_path) ready for use."""
    return PatientDB(), in_memory_db


# ---------------------------------------------------------------------------
# create_patient / get_patient
# ---------------------------------------------------------------------------


def test_create_and_retrieve(db: tuple[PatientDB, str]) -> None:
    """A patient inserted with create_patient can be retrieved with get_patient."""
    pdb, path = db
    record = _make_record()
    pdb.create_patient(path, record)

    result = pdb.get_patient(path, record.patient_id)
    assert result is not None
    assert result.patient_id == record.patient_id
    assert result.name == record.name
    assert result.phone == record.phone


def test_get_patient_not_found(db: tuple[PatientDB, str]) -> None:
    """get_patient returns None for an unknown slug."""
    pdb, path = db
    assert pdb.get_patient(path, "P-does-not-exist") is None


def test_create_patient_idempotent(db: tuple[PatientDB, str]) -> None:
    """create_patient with an existing slug is a no-op (INSERT OR IGNORE)."""
    pdb, path = db
    record = _make_record()
    pdb.create_patient(path, record)
    # Second insert must not raise and must not duplicate
    pdb.create_patient(path, record)

    results = pdb.fuzzy_search(path, "Alice")
    assert len(results) == 1


def test_create_patient_preserves_list_fields(db: tuple[PatientDB, str]) -> None:
    """Conditions, medications, and allergies round-trip through JSON correctly."""
    pdb, path = db
    record = _make_record()
    record = record.model_copy(
        update={
            "conditions": ["hypertension", "CKD"],
            "medications": ["metformin 500mg"],
            "allergies": ["penicillin"],
        }
    )
    pdb.create_patient(path, record)

    result = pdb.get_patient(path, record.patient_id)
    assert result is not None
    assert result.conditions == ["hypertension", "CKD"]
    assert result.medications == ["metformin 500mg"]
    assert result.allergies == ["penicillin"]


# ---------------------------------------------------------------------------
# update_field
# ---------------------------------------------------------------------------


def test_update_field_append_list(db: tuple[PatientDB, str]) -> None:
    """Appending to a list field adds the item without removing existing ones."""
    pdb, path = db
    record = _make_record()
    record = record.model_copy(update={"conditions": ["hypertension"]})
    pdb.create_patient(path, record)

    pdb.update_field(path, record.patient_id, "conditions", "CKD stage 3")

    result = pdb.get_patient(path, record.patient_id)
    assert result is not None
    assert "hypertension" in result.conditions
    assert "CKD stage 3" in result.conditions


def test_update_field_replace_list(db: tuple[PatientDB, str]) -> None:
    """Replace operation on a list field discards previous values."""
    pdb, path = db
    record = _make_record()
    record = record.model_copy(update={"conditions": ["hypertension"]})
    pdb.create_patient(path, record)

    pdb.update_field(path, record.patient_id, "conditions", "diabetes", operation="replace")

    result = pdb.get_patient(path, record.patient_id)
    assert result is not None
    assert result.conditions == ["diabetes"]


def test_update_field_append_string(db: tuple[PatientDB, str]) -> None:
    """Appending to a string field (summary) concatenates with newline."""
    pdb, path = db
    record = _make_record()
    record = record.model_copy(update={"summary": "Initial summary."})
    pdb.create_patient(path, record)

    pdb.update_field(path, record.patient_id, "summary", "Follow-up note.")

    result = pdb.get_patient(path, record.patient_id)
    assert result is not None
    assert "Initial summary." in result.summary
    assert "Follow-up note." in result.summary


def test_update_field_returns_false_for_missing_patient(db: tuple[PatientDB, str]) -> None:
    """update_field returns False when patient_id does not exist."""
    pdb, path = db
    result = pdb.update_field(path, "P-nonexistent", "conditions", "CKD")
    assert result is False


# ---------------------------------------------------------------------------
# update_notes
# ---------------------------------------------------------------------------


def test_update_notes_replaces_content(db: tuple[PatientDB, str]) -> None:
    """update_notes sets the notes field and returns True."""
    pdb, path = db
    record = _make_record()
    pdb.create_patient(path, record)

    ok = pdb.update_notes(path, record.patient_id, "PDF extracted notes here.")
    assert ok is True

    result = pdb.get_patient(path, record.patient_id)
    assert result is not None
    assert result.notes == "PDF extracted notes here."


def test_update_notes_returns_false_for_missing_patient(db: tuple[PatientDB, str]) -> None:
    """update_notes returns False when the patient does not exist."""
    pdb, path = db
    ok = pdb.update_notes(path, "P-ghost", "notes")
    assert ok is False


# ---------------------------------------------------------------------------
# fuzzy_search
# ---------------------------------------------------------------------------


def test_fuzzy_search_case_insensitive(db: tuple[PatientDB, str]) -> None:
    """fuzzy_search finds a patient regardless of query case."""
    pdb, path = db
    pdb.create_patient(path, _make_record(name="Alice Tester"))

    results = pdb.fuzzy_search(path, "alice")
    assert any(r.name == "Alice Tester" for r in results)


def test_fuzzy_search_partial_name(db: tuple[PatientDB, str]) -> None:
    """Partial name match (first name only) returns the patient."""
    pdb, path = db
    pdb.create_patient(path, _make_record(name="Bob Smith", phone="555-0200", patient_id="P-bob1"))

    results = pdb.fuzzy_search(path, "bob")
    assert len(results) >= 1
    assert any(r.name == "Bob Smith" for r in results)


def test_fuzzy_search_no_match(db: tuple[PatientDB, str]) -> None:
    """fuzzy_search returns an empty list when no names match."""
    pdb, path = db
    pdb.create_patient(path, _make_record())

    results = pdb.fuzzy_search(path, "Xzqwerty")
    assert results == []


def test_fuzzy_search_limit(db: tuple[PatientDB, str]) -> None:
    """fuzzy_search respects the limit parameter."""
    pdb, path = db
    for i in range(5):
        pdb.create_patient(
            path,
            _make_record(
                patient_id=f"P-limit{i}",
                phone=f"555-020{i}",
                name=f"Test Patient {i}",
            ),
        )

    results = pdb.fuzzy_search(path, "Test Patient", limit=3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# phone_lookup
# ---------------------------------------------------------------------------


def test_phone_lookup_found(db: tuple[PatientDB, str]) -> None:
    """phone_lookup returns the patient matching the exact phone number."""
    pdb, path = db
    record = _make_record(phone="555-9999")
    pdb.create_patient(path, record)

    result = pdb.phone_lookup(path, "555-9999")
    assert result is not None
    assert result.patient_id == record.patient_id


def test_phone_lookup_not_found(db: tuple[PatientDB, str]) -> None:
    """phone_lookup returns None for a phone number not in the DB."""
    pdb, path = db
    assert pdb.phone_lookup(path, "000-0000") is None

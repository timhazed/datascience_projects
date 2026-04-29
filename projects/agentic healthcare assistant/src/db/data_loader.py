"""DataLoader — ingests records.xlsx + PDF reports into SQLite + FAISS.

Two dataset subdirectories are supported:
  - original_data/  : header on row 1 (index 0); PDFs named sample_report_{first}.pdf
  - generated_data/ : header on row 3 (index 2); PDFs named report_{First}_{Last}.pdf

Two loading paths for PDFs:
  1. xlsx-backed patient  → name substring match → embed PDF text into FAISS
  2. PDF-only patient     → extract demographics from PDF header → INSERT row → embed

PDF-only patients (no xlsx row): Jasmine Lee, Kevin Harris, Mark Evans,
Olivia Turner, Sarah Mitchell. MRN from the PDF header is used as the phone
surrogate for slug generation.

Slug generation: P-<sha256[:6]> of (phone + name).
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import date, datetime
from pathlib import Path

import openpyxl
from pypdf import PdfReader

from src.db.patient_db import PatientDB
from src.db.patient_vector_store import PatientVectorStore
from src.models.patient import PatientRecord

logger = logging.getLogger(__name__)

_PATIENT_DB = PatientDB()

# Compiled once at module load — used by DataLoader._parse_medications_from_notes.
# Strips clinical action verbs that prefix drug names (e.g. "Continue Telmisartan 40mg OD").
_ACTION_WORDS_RE = re.compile(
    r"\b(continue|start|stop|discontinue|add|increase|decrease|switch|hold|initiate)\s+",
    re.IGNORECASE,
)
# Matches "<DrugName> <dose><unit> [<frequency>]" after action-word stripping.
# Covers: "Telmisartan 40mg OD", "Metformin 1000mg BID", "Aspirin 81mg daily"
# Note: word alternatives (daily|once|...) must appear BEFORE [A-Z]{2,4} so the
# regex engine does not greedily consume partial matches (e.g. "dail" from "daily").
_MEDICATION_RE = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+"              # drug name (title-cased)
    r"(\d+(?:\.\d+)?(?:mg|mcg|g|ml|IU|units?))"                  # dose + unit
    r"(?:\s+(daily|once|twice|weekly|[A-Z]{2,4}))?(?=\W|$)",      # optional frequency
    re.IGNORECASE,
)


class DataLoader:
    """Ingests records.xlsx files and PDF reports into SQLite and FAISS.

    Handles both dataset subdirectories, both PDF naming conventions, and
    both xlsx-backed and PDF-only patients. Idempotent — re-running skips
    rows already present in the DB (INSERT OR IGNORE).
    """

    @staticmethod
    def _make_slug(phone: str, name: str) -> str:
        """Derive a deterministic 6-char slug from phone + name."""
        raw = f"{phone}{name}".encode()
        return "P-" + hashlib.sha256(raw).hexdigest()[:6]

    @staticmethod
    def _dob_to_age(dob_str: str) -> int:
        """Convert MM/DD/YYYY DOB string to integer age."""
        result = 0
        try:
            dob = datetime.strptime(dob_str.strip(), "%m/%d/%Y").date()
            today = date.today()
            result = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        except ValueError:
            result = 0

        return result

    @staticmethod
    def _extract_pdf_text(pdf_path: Path) -> str:
        """Extract all text from a PDF, concatenating all pages."""
        reader = PdfReader(str(pdf_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    @staticmethod
    def _parse_medications_from_notes(text: str) -> list[str]:
        """Extract medication entries from the Plan Notes section of a clinical PDF.

        Looks for a "Plan Notes:" (or "Plan:") section and extracts medication-like
        phrases — patterns of the form "<DrugName> <dose><unit> <frequency>".

        Args:
            text: Full extracted PDF text.

        Returns:
            List of medication strings (e.g. ["Telmisartan 40mg OD"]), or [] if
            no medications are found.
        """
        # Isolate the Plan Notes block (stop at next section heading or end of text).
        plan_match = re.search(
            r"(?:Plan Notes?|Plan)\s*:\s*(.*?)(?:\n[A-Z][^\n]*:|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if not plan_match:
            return []

        # Strip action verbs then extract drug/dose/frequency using module-level patterns.
        plan_text = _ACTION_WORDS_RE.sub("", plan_match.group(1))
        results = []
        for m in _MEDICATION_RE.finditer(plan_text):
            drug = m.group(1).strip()
            dose = m.group(2).strip()
            freq = m.group(3).strip() if m.group(3) else ""
            entry = f"{drug} {dose} {freq}".strip() if freq else f"{drug} {dose}"
            results.append(entry)

        return results

    @staticmethod
    def _parse_pdf_only_demographics(text: str, pdf_path: Path) -> dict | None:
        """Extract name, gender, DOB, and MRN from a PDF-only patient header.

        Expected header format (first ~200 chars):
            <Name>
            <date> <time>
            Location: ...
            Patient #: <MRN>  DOB: MM/DD/YYYY
            Gender: <gender>

        Returns a dict with keys: name, mrn, age, gender; or None if parsing fails.
        """
        result = None
        # Name is on the first non-empty line
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        if lines:
            name = lines[0]

            mrn_match = re.search(r"Patient #:\s*(\S+)", text)
            dob_match = re.search(r"DOB:\s*(\d{2}/\d{2}/\d{4})", text)
            gender_match = re.search(r"Gender:\s*(\w+)", text)

            if not (mrn_match and dob_match and gender_match):
                logger.warning("Could not extract demographics from PDF header: %s", pdf_path.name)
            else:
                result = {
                    "name": name,
                    "mrn": mrn_match.group(1),
                    "age": DataLoader._dob_to_age(dob_match.group(1)),
                    "gender": gender_match.group(1),
                }

        return result

    @staticmethod
    def _load_xlsx(xlsx_path: Path) -> list[dict]:
        """Load patient rows from an xlsx file, auto-detecting header row position.

        original_data has header on row index 0; generated_data on row index 2
        (rows 0–1 contain label text).

        Returns list of row dicts with keys matching PatientRecord fields.
        """
        wb = openpyxl.load_workbook(xlsx_path)
        ws = wb.active
        all_rows = list(ws.iter_rows(values_only=True))

        # Detect header row: first row whose first cell equals 'Phone_number'
        header_idx = 0
        for i, row in enumerate(all_rows):
            if row and str(row[0]).strip() == "Phone_number":
                header_idx = i
                break

        headers = [str(h).strip() for h in all_rows[header_idx]]
        records = []
        for row in all_rows[header_idx + 1:]:
            if not any(row):
                continue
            d = dict(zip(headers, row))
            records.append(d)

        return records

    @staticmethod
    def _xlsx_row_to_patient(row: dict) -> PatientRecord:
        """Convert a raw xlsx row dict into a PatientRecord."""
        phone = str(row.get("Phone_number", "")).strip()
        name = str(row.get("Name", "")).strip()
        slug = DataLoader._make_slug(phone, name)

        return PatientRecord(
            patient_id=slug,
            phone=phone,
            email=str(row.get("Email", "") or "").strip() or None,
            name=name,
            age=int(row.get("Age", 0) or 0),
            gender=str(row.get("Gender", "")).strip(),
            address=str(row.get("Address", "") or "").strip(),
            summary=str(row.get("Summary", "") or "").strip(),
        )

    def load_all(
        self,
        db_path: str,
        faiss_path: str,
        dataset_dir: str,
        embeddings_model: object,
    ) -> None:
        """Load all patients from dataset_dir into SQLite and FAISS.

        Processing order:
          1. Discover all xlsx files; load + dedup demographics into SQLite.
          2. For each PDF: match to an xlsx-backed patient by name substring.
             If matched: embed PDF text + update notes.
             If unmatched: attempt PDF-only demographic extraction → INSERT → embed.
          3. Persist the FAISS index.

        Args:
            db_path: SQLite database path.
            faiss_path: FAISS index directory path.
            dataset_dir: Root dataset directory containing subdirectories.
            embeddings_model: LangChain-compatible embeddings instance.
        """
        store = PatientVectorStore(faiss_path, embeddings_model)
        dataset_root = Path(dataset_dir)

        # Step 1: Load all xlsx files → build name→patient map for PDF matching
        patients_by_name: dict[str, PatientRecord] = {}
        for xlsx_path in sorted(dataset_root.rglob("records.xlsx")):
            rows = self._load_xlsx(xlsx_path)
            seen_slugs: set[str] = set()
            for row in rows:
                record = self._xlsx_row_to_patient(row)
                if record.patient_id in seen_slugs:
                    logger.debug(
                        "Dedup: skipping duplicate slug %s (%s)",
                        record.patient_id,
                        record.name,
                    )
                    continue
                seen_slugs.add(record.patient_id)
                _PATIENT_DB.create_patient(db_path, record)
                patients_by_name[record.name.lower()] = record
            logger.info("Loaded %d rows from %s", len(seen_slugs), xlsx_path)

        # Step 2: Process all PDFs
        for pdf_path in sorted(dataset_root.rglob("*.pdf")):
            text = self._extract_pdf_text(pdf_path)
            if not text.strip():
                logger.warning("Empty PDF text, skipping: %s", pdf_path.name)
                continue

            matched_patient = self._match_pdf_to_patient(pdf_path, patients_by_name)

            if matched_patient is not None:
                _PATIENT_DB.update_notes(db_path, matched_patient.patient_id, text)
                # Seed structured medications from Plan Notes if the field is still empty.
                # Only populates on first load — does not overwrite graph-updated values.
                meds = self._parse_medications_from_notes(text)
                if meds:
                    existing = _PATIENT_DB.get_patient(db_path, matched_patient.patient_id)
                    if existing and not existing.medications:
                        for med in meds:
                            _PATIENT_DB.update_field(
                                db_path, matched_patient.patient_id, "medications", med,
                                operation="append",
                            )
                        logger.info(
                            "Seeded medications for %s: %s", matched_patient.name, meds
                        )
                self._upsert_vector(store, matched_patient, text)
            else:
                # PDF-only path: extract demographics and insert a new patient row
                demographics = self._parse_pdf_only_demographics(text, pdf_path)
                if demographics is None:
                    logger.warning("Cannot match or parse PDF, skipping: %s", pdf_path.name)
                    continue
                slug = self._make_slug(demographics["mrn"], demographics["name"])
                meds = self._parse_medications_from_notes(text)
                record = PatientRecord(
                    patient_id=slug,
                    phone=demographics["mrn"],  # MRN as phone surrogate
                    name=demographics["name"],
                    age=demographics["age"],
                    gender=demographics["gender"],
                    notes=text,
                    medications=meds,
                )
                _PATIENT_DB.create_patient(db_path, record)
                patients_by_name[record.name.lower()] = record
                self._upsert_vector(store, record, text)
                logger.info("Inserted PDF-only patient: %s", record.name)

        store.persist()
        logger.info("DataLoader.load_all complete — FAISS index persisted to %s", faiss_path)

    def _match_pdf_to_patient(
        self, pdf_path: Path, patients_by_name: dict[str, PatientRecord]
    ) -> PatientRecord | None:
        """Find a loaded patient whose name appears as a substring of the PDF filename.

        Supports two naming conventions:
          - report_{First}_{Last}.pdf  (generated_data)
          - sample_report_{first}.pdf  (original_data, matched by first name only)

        Args:
            pdf_path: Path to the PDF file.
            patients_by_name: Map of lower-cased patient name → PatientRecord.

        Returns:
            Matched PatientRecord, or None if no match found.
        """
        stem = pdf_path.stem.lower()
        # Strip common prefixes to get name-bearing part
        for prefix in ("report_", "sample_report_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break

        # Convert underscores to spaces for matching
        stem_as_name = stem.replace("_", " ")

        # Try exact match first (full name from filename)
        if stem_as_name in patients_by_name:
            return patients_by_name[stem_as_name]

        # Try substring match (e.g. first-name-only files like sample_report_anjali)
        for patient_name, record in patients_by_name.items():
            if stem_as_name in patient_name or patient_name in stem_as_name:
                return record

        return None

    def _upsert_vector(
        self, store: PatientVectorStore, patient: PatientRecord, text: str
    ) -> None:
        """Embed and add a patient's clinical text into the FAISS store.

        Does not persist to disk — callers must call store.persist() when done
        to avoid writing the index N times during bulk loads.
        """
        metadata = {
            "patient_id": patient.patient_id,
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "summary": patient.summary,
            "conditions": ", ".join(patient.conditions),
        }
        
        store.add(patient.patient_id, text, metadata)

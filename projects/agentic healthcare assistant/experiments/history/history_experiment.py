"""Experiment 1 — Medical History Management.

Research question:
    Can an inline LangChain chain (Groq LLM + FAISS retrieval) accurately answer
    ground-truth QA pairs about patient medical history, and do DB write operations
    (append medication) round-trip losslessly?

Approach:
    1. Write round-trip: append "metformin 500mg" to David Thompson's medications
       → retrieve record → assert the new value is present.
       Latency = wall-clock time for the DB write + read. No LLM evaluator needed.
    2. QA retrieval (5 pairs across 5 patients): FAISS search provides patient context;
       inline Groq chain answers the question; load_evaluator("qa") grades CORRECT=1 / INCORRECT=0.
       Patients: Ramesh Kulkarni, David Thompson, Anjali Mehra (original_data PDF-backed)
                 + Liam Smith, Maya Patel (generated_data summary-backed).
    Success criterion: ≥ 4/5 QA pairs graded CORRECT and all write ops succeed.

Requires:
    GROQ_API_KEY   — for history chain LLM (llama-3.3-70b-versatile)
    OPENAI_API_KEY — for text-embedding-3-small (FAISS similarity search)

Output (experiments/history/data/):
    medical_history_<run_id>.json — one file per test case (6 total per run)

Run:
    GROQ_API_KEY=... OPENAI_API_KEY=... \\
    poetry run python experiments/history/history_experiment.py
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain.evaluation import load_evaluator
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings

from experiments.experiment_runner import ExperimentRunner
from src.db.data_loader import DataLoader
from src.db.patient_db import PatientDB
from src.db.patient_vector_store import PatientVectorStore
from src.models.evaluation import ExperimentMetrics

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — never hardcode strings in chain logic, only here
# ---------------------------------------------------------------------------

_DATASET_DIR = "dataset"  # loads both original_data + generated_data (30 patients)
_HISTORY_MODEL = "llama-3.3-70b-versatile"
_EMBEDDINGS_MODEL = "text-embedding-3-small"
_QA_DATA_PATH = Path("experiments/history/data/history_qanda.jsonl")


def _load_qa_pairs(path: Path) -> list[dict]:
    """Load QA pairs from the experiment's data JSONL file.

    Args:
        path: Path to history_qanda.jsonl.

    Returns:
        List of dicts with 'query' and 'answer' keys.
    """
    pairs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            pairs.append(json.loads(line))
    return pairs


_QA_PAIRS = _load_qa_pairs(_QA_DATA_PATH)

# Write round-trip test parameters
_WRITE_PATIENT_NAME = "David Thompson"
_WRITE_FIELD = "medications"
_WRITE_VALUE = "metformin 500mg"

# ---------------------------------------------------------------------------
# LLM — constructed once at module boundary; not rebuilt per request
# ---------------------------------------------------------------------------

_history_llm = ChatGroq(model=_HISTORY_MODEL, temperature=0.0, max_tokens=300)
_eval_llm = ChatGroq(model=_HISTORY_MODEL, temperature=0.0, max_tokens=100)

# ---------------------------------------------------------------------------
# History retrieval chain — inline; same logic as Phase 5 production module
# ---------------------------------------------------------------------------

_history_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a clinical assistant with access to a patient's medical record. "
        "Answer the question using ONLY the information in the provided record. "
        "Be concise and specific. If the information is not in the record, say so.",
    ),
    (
        "human",
        "Patient record:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    ),
])

_history_chain = _history_prompt | _history_llm | StrOutputParser()

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_dataset(db_path: str, faiss_path: str) -> None:
    """Load original_data patients into temp SQLite + FAISS.

    Initialises both DBs and runs DataLoader.load_all() over original_data only.
    Idempotent — safe to call once per experiment run.
    """
    from src.db.appointment_db import AppointmentDB

    AppointmentDB().init_db(db_path)
    PatientDB().init_db(db_path)
    embeddings = OpenAIEmbeddings(model=_EMBEDDINGS_MODEL)
    DataLoader().load_all(db_path, faiss_path, _DATASET_DIR, embeddings)


def _get_patient_id(db_path: str, name: str) -> str | None:
    """Return the patient_id slug for a patient by exact name lookup."""
    pdb = PatientDB()
    results = pdb.fuzzy_search(db_path, name, limit=1)
    return results[0].patient_id if results else None


def _retrieve_context(faiss_path: str, query: str, k: int = 3) -> str:
    """Retrieve top-k patient record snippets from FAISS as a context string."""
    embeddings = OpenAIEmbeddings(model=_EMBEDDINGS_MODEL)
    store = PatientVectorStore(faiss_path, embeddings)
    docs = store.search(query, k=k)
    if not docs:
        return "No relevant patient records found."
    parts = []
    for doc in docs:
        snippet = (
            f"Patient: {doc.get('name', 'Unknown')} | "
            f"Summary: {doc.get('summary', '')} | "
            f"Conditions: {doc.get('conditions', '')}"
        )
        parts.append(snippet)
    return "\n".join(parts)


def _run_write_test(db_path: str, faiss_path: str) -> dict:
    """Append a medication to David Thompson's record and verify the round-trip.

    Measures wall-clock time of the DB write + read (no LLM invoked).
    Returns a result dict compatible with ExperimentMetrics construction.
    """
    pdb = PatientDB()
    patient_id = _get_patient_id(db_path, _WRITE_PATIENT_NAME)
    if patient_id is None:
        return {
            "case": "write_round_trip",
            "success": False,
            "latency_ms": 0.0,
            "quality_score": None,
            "tool_calls_made": 0,
            "error": f"Patient '{_WRITE_PATIENT_NAME}' not found in DB after load",
        }

    start = time.perf_counter()
    ok = pdb.update_field(db_path, patient_id, _WRITE_FIELD, _WRITE_VALUE, "append")
    record = pdb.get_patient(db_path, patient_id)
    latency_ms = (time.perf_counter() - start) * 1000

    # Also upsert updated record into FAISS so QA on David picks up metformin
    if record is not None:
        embeddings = OpenAIEmbeddings(model=_EMBEDDINGS_MODEL)
        store = PatientVectorStore(faiss_path, embeddings)
        combined_text = f"{record.notes}\nMedications: {', '.join(record.medications)}"
        metadata = {
            "patient_id": record.patient_id,
            "name": record.name,
            "age": record.age,
            "gender": record.gender,
            "summary": record.summary,
            "conditions": ", ".join(record.conditions),
        }
        store.upsert(record.patient_id, combined_text, metadata)

    value_present = (
        record is not None and _WRITE_VALUE in record.medications
    )
    success = ok and value_present

    print(
        f"  [write] {'✓' if success else '✗'} "
        f"{latency_ms:.0f}ms | "
        f"update_ok={ok} value_in_record={value_present}"
    )

    return {
        "case": "write_round_trip",
        "success": success,
        "latency_ms": round(latency_ms, 1),
        "quality_score": None,
        "tool_calls_made": 2,  # 1 write + 1 read
        "error": None if success else "Appended value not found in retrieved record",
    }


def _run_qa_case(
    case: dict,
    faiss_path: str,
    evaluator: object,
) -> dict:
    """Run one QA retrieval test case.

    Workflow: FAISS context retrieval → history_chain.invoke() → evaluator.

    Args:
        case: Dict with 'query' and 'answer' keys.
        faiss_path: Path to the FAISS index directory.
        evaluator: LangChain qa evaluator instance.

    Returns:
        Result dict compatible with ExperimentMetrics construction.
    """
    query = case["query"]
    ground_truth = case["answer"]
    error: str | None = None
    quality_score: float | None = None
    prediction: str | None = None
    tool_calls = 0

    start = time.perf_counter()

    try:
        context = _retrieve_context(faiss_path, query, k=3)
        tool_calls += 1

        # Two attempts — guards against transient Groq 429/503 failures.
        for attempt in range(2):
            try:
                prediction = _history_chain.invoke({"context": context, "question": query})
                tool_calls += 1
                break
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                print(f"    [WARN] LLM error (attempt {attempt + 1}): {exc}")
                if attempt == 0:
                    time.sleep(1)

        if prediction is not None:
            eval_result = evaluator.evaluate_strings(  # type: ignore[attr-defined]
                prediction=prediction,
                input=query,
                reference=ground_truth,
            )
            # score is 1 (CORRECT) or 0 (INCORRECT)
            quality_score = float(eval_result.get("score", 0))
            error = None

    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    latency_ms = (time.perf_counter() - start) * 1000
    success = quality_score is not None and quality_score > 0.0

    grade_label = "CORRECT" if success else "INCORRECT"
    print(
        f"  [qa]    {'✓' if success else '✗'} {latency_ms:.0f}ms | "
        f"{grade_label} | Q: {query[:60]}"
    )
    if prediction:
        print(f"          → {prediction[:100]}{'...' if len(prediction) > 100 else ''}")

    return {
        "case": "qa_retrieval",
        "query": query,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "success": success,
        "latency_ms": round(latency_ms, 1),
        "quality_score": quality_score,
        "tool_calls_made": tool_calls,
        "error": error,
    }


# ---------------------------------------------------------------------------
# HistoryExperiment — inherits ExperimentRunner
# ---------------------------------------------------------------------------


class HistoryExperiment(ExperimentRunner):
    """Medical history fidelity experiment runner.

    Test cases:
      1. Write round-trip: append medication → retrieve → assert present in record.
      2–4. QA retrieval: 3 ground-truth QA pairs graded by load_evaluator("qa").

    Inherits ExperimentRunner for _time_invoke() and _save_results().
    """

    def __init__(self, db_path: str, faiss_path: str) -> None:
        """Initialise with database and FAISS paths.

        The dataset must already be loaded before calling run(). Call
        _load_dataset(db_path, faiss_path) at the experiment boundary.

        Args:
            db_path: Path to the SQLite database file.
            faiss_path: Path to the FAISS index directory.
        """
        self._db_path = db_path
        self._faiss_path = faiss_path
        # Evaluator constructed once — one eval_llm per experiment run
        self._evaluator = load_evaluator("qa", llm=_eval_llm)

    def run(self) -> list[ExperimentMetrics]:
        """Execute all 6 test cases and return one ExperimentMetrics each.

        Execution order:
          1. Write round-trip (modifies DB + FAISS in place)
          2–6. QA pairs: Ramesh, David (picks up appended medication), Anjali,
               Liam Smith, Maya Patel

        Returns:
            List of 4 ExperimentMetrics instances.
        """
        results: list[ExperimentMetrics] = []

        # Test case 1: write round-trip
        write_result = _run_write_test(self._db_path, self._faiss_path)
        results.append(
            ExperimentMetrics(
                experiment_name="medical_history",
                run_id=str(uuid.uuid4()),
                latency_ms=write_result["latency_ms"],
                success=write_result["success"],
                quality_score=write_result["quality_score"],
                tool_calls_made=write_result["tool_calls_made"],
                error=write_result["error"],
            )
        )

        # Test cases 2–4: QA retrieval
        for qa_case in _QA_PAIRS:
            qa_result = _run_qa_case(qa_case, self._faiss_path, self._evaluator)
            results.append(
                ExperimentMetrics(
                    experiment_name="medical_history",
                    run_id=str(uuid.uuid4()),
                    latency_ms=qa_result["latency_ms"],
                    success=qa_result["success"],
                    quality_score=qa_result["quality_score"],
                    tool_calls_made=qa_result["tool_calls_made"],
                    error=qa_result["error"],
                )
            )

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 1 — Medical History Management."""
    print("=== Experiment 1 — Medical History Management ===")
    print(f"Model     : {_HISTORY_MODEL}")
    print(f"Embeddings: {_EMBEDDINGS_MODEL}")
    print(f"Dataset   : {_DATASET_DIR}")
    print(f"QA input  : {_QA_DATA_PATH} ({len(_QA_PAIRS)} pairs)")
    print(f"Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    out_dir = "experiments/history/data"

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "history_exp.db")
        faiss_path = str(Path(tmpdir) / "faiss")

        print("[1/2] Loading dataset...")
        load_start = time.perf_counter()
        _load_dataset(db_path, faiss_path)
        load_ms = (time.perf_counter() - load_start) * 1000
        print(f"      Done in {load_ms:.0f}ms\n")

        print("[2/2] Running test cases...")
        experiment = HistoryExperiment(db_path, faiss_path)
        results = experiment.run()

    experiment._save_results(results, out_dir)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    qa_results = [r for r in results if r.quality_score is not None]
    write_results = [r for r in results if r.quality_score is None]

    write_ok = all(r.success for r in write_results)
    qa_scores = [r.quality_score for r in qa_results if r.quality_score is not None]
    quality_score = statistics.mean(qa_scores) if qa_scores else 0.0
    qa_correct = sum(1 for s in qa_scores if s > 0)
    latencies = [r.latency_ms for r in results]

    print("\n=== Summary ===")
    print(f"Write round-trip  : {'✓ PASS' if write_ok else '✗ FAIL'}")
    print(f"QA correct        : {qa_correct}/{len(qa_results)}")
    print(f"Quality score     : {quality_score:.2f} (avg CORRECT/INCORRECT)")
    print(f"Latency mean      : {statistics.mean(latencies):.0f}ms")
    print(f"Latency median    : {statistics.median(latencies):.0f}ms")
    print(f"Results saved to  : {out_dir}/")
    print()

    # Promotion decision
    overall_pass = write_ok and qa_correct >= 4
    if overall_pass:
        print("[x] Promote to production — write round-trip passes + ≥4/5 QA correct.")
        print("    Next: /architect for history_chain production module (Phase 5).")
    else:
        print("[ ] Keep as reference — below success threshold. Investigate failures above.")


if __name__ == "__main__":
    main()

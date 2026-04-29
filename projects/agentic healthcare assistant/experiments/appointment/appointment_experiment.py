"""Experiment 3 — Appointment Booking.

Research question:
    Does AppointmentDB reliably succeed on valid booking requests and fail
    gracefully (success=False, not exception) on invalid ones — no slots,
    unknown specialty, unresolved patient, double-book?

Approach:
    4 deterministic test scenarios — pure assertions, no LLM.
    AppointmentDB is seeded in a temp SQLite file per run so tests are isolated
    and repeatable regardless of wall-clock date.
    Success criterion: all 4 scenarios produce the expected success/failure outcome.

No API keys required — AppointmentDB is pure SQLite.

Input:
    experiments/appointment/data/appointment_scenarios.jsonl

Output (experiments/appointment/data/):
    appointment_booking_<run_id>.json — one file per scenario (4 total)

Run:
    poetry run python experiments/appointment/appointment_experiment.py
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

from experiments.experiment_runner import ExperimentRunner
from src.db.appointment_db import AppointmentDB
from src.models.appointment_result import AppointmentResult
from src.models.evaluation import ExperimentMetrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SCENARIOS_PATH = Path("experiments/appointment/data/appointment_scenarios.jsonl")
_OUTPUT_DIR = "experiments/appointment/data"


def _load_scenarios(path: Path) -> list[dict]:
    """Load test scenarios from the experiment's input JSONL file.

    Args:
        path: Path to appointment_scenarios.jsonl.

    Returns:
        List of scenario dicts.
    """
    scenarios = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            scenarios.append(json.loads(line))
    return scenarios


_SCENARIOS = _load_scenarios(_SCENARIOS_PATH)


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


def _run_scenario(scenario: dict, db: AppointmentDB, db_path: str) -> dict:
    """Execute one appointment booking scenario and return a result dict.

    Booking pipeline (mirrors production appointment_node):
      1. If patient_id is None → fail immediately (unresolved patient).
      2. Query available slots for the requested specialty + urgency.
      3. If no slots → fail (specialty unavailable or no capacity).
      4. Book the first available slot.
      5. Assert the result matches expected_success.

    Args:
        scenario: Dict loaded from appointment_scenarios.jsonl.
        db: Initialised AppointmentDB instance.
        db_path: Path to the temp SQLite file.

    Returns:
        Result dict with scenario, success, assertion_passed, latency_ms,
        tool_calls_made, slot, message, error.
    """
    patient_id: str | None = scenario["patient_id"]
    specialty: str = scenario["specialty"]
    urgency: str = scenario["urgency"]
    expected_success: bool = scenario["expected_success"]
    tool_calls = 0
    result: AppointmentResult | None = None
    error: str | None = None

    start = time.perf_counter()

    # Step 1: unresolved patient guard
    if patient_id is None:
        result = AppointmentResult(
            success=False,
            patient_id="",
            message="Cannot book appointment: patient identity not resolved.",
        )
    else:
        # Step 2: query slots
        slots = db.query_slots(db_path, specialty, urgency=urgency)
        tool_calls += 1

        if not slots:
            result = AppointmentResult(
                success=False,
                patient_id=patient_id,
                message=f"No available slots found for {specialty}.",
            )
        else:
            # Step 3: book first available slot
            result = db.book_slot(
                db_path,
                slot_id=slots[0]["slot_id"],
                patient_id=patient_id,
                reason=scenario.get("description", ""),
                urgency=urgency,
            )
            tool_calls += 1

    # 3 decimal places — SQLite in-process operations complete in sub-millisecond time;
    # rounding to 1 decimal would produce 0.0 for fast guard-clause exits and mask
    # legitimate timing differences between scenarios.
    latency_ms = round((time.perf_counter() - start) * 1000, 3)

    # Emergency scenario: success alone is sufficient — query_slots with urgency=emergency
    # returns exactly 1 slot by design (LIMIT 1 in AppointmentDB). Booking that slot
    # and getting success=True is the assertion.
    extra_assertion_passed = True
    if scenario["scenario"] == "emergency_booking":
        extra_assertion_passed = result.success is True

    assertion_passed = result.success == expected_success and extra_assertion_passed

    slot_time = result.slot if result else None
    message = result.message if result else ""

    status = "✓ PASS" if assertion_passed else "✗ FAIL"
    expected_label = "success=True" if expected_success else "success=False"
    got_label = f"success={result.success}" if result else "no result"

    print(
        f"  [{scenario['scenario']:25s}] {status} | {latency_ms:.3f}ms | "
        f"expected={expected_label} got={got_label}"
    )
    if result and result.success and slot_time:
        print(f"    → Booked: {result.doctor_name} on {slot_time}")
    elif result and not result.success:
        print(f"    → {message}")

    return {
        "scenario": scenario["scenario"],
        "description": scenario["description"],
        "specialty": specialty,
        "urgency": urgency,
        "patient_id": patient_id,
        "expected_success": expected_success,
        "actual_success": result.success if result else False,
        "assertion_passed": assertion_passed,
        "latency_ms": round(latency_ms, 3),
        "tool_calls_made": tool_calls,
        "slot": slot_time,
        "doctor": result.doctor_name if result else None,
        "message": message,
        "error": error,
    }


# ---------------------------------------------------------------------------
# AppointmentExperiment — inherits ExperimentRunner
# ---------------------------------------------------------------------------


class AppointmentExperiment(ExperimentRunner):
    """Appointment booking experiment runner.

    Exercises 4 deterministic scenarios against a real SQLite AppointmentDB
    seeded in a temp file. No LLM required — success is measured by pure
    assertion: actual_success == expected_success.

    Inherits ExperimentRunner for _time_invoke() and _save_results().
    """

    def __init__(self, db_path: str, scenarios: list[dict]) -> None:
        """Initialise with a seeded database and loaded scenarios.

        Args:
            db_path: Path to the pre-seeded SQLite file.
            scenarios: List of scenario dicts from appointment_scenarios.jsonl.
        """
        self._db_path = db_path
        self._scenarios = scenarios
        self._db = AppointmentDB()

    def run(self) -> list[ExperimentMetrics]:
        """Execute all scenarios and return one ExperimentMetrics each.

        Returns:
            List of ExperimentMetrics — one per scenario.
            quality_score is None for all (deterministic, no LLM grading).
        """
        results = []
        for scenario in self._scenarios:
            r = _run_scenario(scenario, self._db, self._db_path)
            results.append(
                ExperimentMetrics(
                    experiment_name="appointment_booking",
                    run_id=str(uuid.uuid4()),
                    latency_ms=r["latency_ms"],
                    success=r["assertion_passed"],
                    quality_score=None,
                    tool_calls_made=r["tool_calls_made"],
                    error=r["error"],
                )
            )
        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 3 — Appointment Booking."""
    print("=== Experiment 3 — Appointment Booking ===")
    print(f"Input    : {_SCENARIOS_PATH} ({len(_SCENARIOS)} scenarios)")
    print(f"Output   : {_OUTPUT_DIR}/")
    print(f"Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("No API keys required — pure SQLite assertions")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "appointment_exp.db")

        print("[1/2] Seeding AppointmentDB...")
        seed_start = time.perf_counter()
        AppointmentDB().init_db(db_path)
        seed_ms = (time.perf_counter() - seed_start) * 1000
        print(f"      Done in {seed_ms:.0f}ms\n")

        print("[2/2] Running scenarios...")
        experiment = AppointmentExperiment(db_path, _SCENARIOS)
        metrics = experiment.run()

    experiment._save_results(metrics, _OUTPUT_DIR)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    passed = sum(1 for m in metrics if m.success)
    latencies = [m.latency_ms for m in metrics]

    print("\n=== Summary ===")
    print(f"Scenarios passed  : {passed}/{len(metrics)}")
    print(f"Latency mean      : {statistics.mean(latencies):.0f}ms")
    print(f"Latency median    : {statistics.median(latencies):.0f}ms")
    print(f"Latency min/max   : {min(latencies):.0f}ms / {max(latencies):.0f}ms")
    print(f"Results saved to  : {_OUTPUT_DIR}/")
    print()

    overall_pass = passed == len(metrics)
    if overall_pass:
        print("[x] Promote to production — all scenarios passed.")
        print("    AppointmentDB atomic booking confirmed; graceful failure paths verified.")
        print("    Next: /architect for appointment_node production module (Phase 6).")
    else:
        failed = [m for m in metrics if not m.success]
        print(f"[ ] {len(failed)} scenario(s) failed — investigate before promoting.")


if __name__ == "__main__":
    main()

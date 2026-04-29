"""Streamlit Metrics tab — session activity, RAGAS eval, experiment benchmarks."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain_openai import OpenAIEmbeddings

from experiments.appointment.appointment_experiment import AppointmentExperiment
from experiments.history.history_experiment import HistoryExperiment
from experiments.search.search_experiment import SearchExperiment
from src.db.metrics_db import MetricsDB
from src.ui.constants import (
    _EXP_LABELS,
    _EXPERIMENT_LABELS,
    _QUALITY_LEGEND,
    _QUALITY_THRESHOLD,
)
from src.ui.runtime import settings

logger = logging.getLogger(__name__)


def _quality_label(score: float | None, experiment: str) -> str:
    """Return a human-readable quality label for the experiment results table.

    Appointment booking is deterministic — always shows N/A.
    None or NaN scores (e.g. when RAGAS was skipped) also show N/A.
    Scored experiments show a numeric value with a pass/fail gate against 0.80.

    Args:
        score: Quality score (0.0–1.0), None, or float NaN.
        experiment: Experiment name value from ExperimentMetrics.

    Returns:
        Human-readable string for the quality column.
    """
    if experiment == "appointment_booking":
        return "N/A"
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "N/A"
    gate = "✓ Pass" if score >= _QUALITY_THRESHOLD else "✗ Below threshold"
    return f"{score:.2f}  ({gate})"


def _render_tab_metrics(metrics_db: MetricsDB | None = None) -> None:
    """Render Tab 4 — Metrics: live session activity + optional experiment benchmarks.

    Args:
        metrics_db: MetricsDB for querying historical mean latency across sessions.
    """
    # --- How it works legend (collapsed by default — context available but not intrusive) ---
    with st.expander("ℹ️ How metrics work", expanded=False):
        st.markdown(_QUALITY_LEGEND)

    # -----------------------------------------------------------------------
    # Section 1 — Live session activity
    # Every graph invocation automatically appends entries here.
    # -----------------------------------------------------------------------
    st.subheader("This Session")
    session_metrics = st.session_state.get("session_metrics", [])
    if not session_metrics:
        st.info(
            "No activity yet. Use Chat, Appointments, History, or Memory & Logs (sample "
            "scenarios) to see metrics here when tool tasks complete."
        )
    else:
        # Most recent first
        rows = list(reversed(session_metrics))
        live_df = pd.DataFrame(rows)

        # Rename columns for display
        live_df = live_df.rename(columns={
            "datetime": "Time",
            "operation": "Operation",
            "success": "Success",
            "latency_ms": "Latency (ms)",
            "error": "Error",
        })

        # Replace True/False with readable symbols
        live_df["Success"] = live_df["Success"].map({True: "✓", False: "✗"})

        # Sub-millisecond DB operations (booking, lookup) genuinely complete in < 1ms.
        # Show "< 1 ms" instead of "0" so the column is clearly labelled, not blank-looking.
        live_df["Latency (ms)"] = live_df["Latency (ms)"].apply(
            lambda v: "< 1 ms" if v == 0.0 else f"{v} ms"
        )

        # Drop error column if all blank (cleaner when no failures)
        if "Error" in live_df.columns and live_df["Error"].eq("").all():
            live_df = live_df.drop(columns=["Error"])

        st.dataframe(live_df, use_container_width=True, hide_index=True)

        # Latency trend — requires at least 2 points to be meaningful.
        # When historical means are available, a second series shows the mean for each
        # operation alongside the current session's actual latency for comparison.
        if len(session_metrics) >= 2:
            historical_means = (
                metrics_db.get_means(str(settings.db.sqlite_path))
                if metrics_db is not None
                else {}
            )
            chart_rows = []
            for i, m in enumerate(session_metrics):
                row: dict = {"seq": i, "This Session (ms)": m["latency_ms"]}
                mean_val = historical_means.get(m["operation"])
                if mean_val is not None:
                    row["Historical Mean (ms)"] = mean_val
                chart_rows.append(row)
            chart_df = pd.DataFrame(chart_rows).set_index("seq")
            # Only include historical mean column when at least one entry has a value
            display_cols = ["This Session (ms)"]
            has_means = (
                "Historical Mean (ms)" in chart_df.columns
                and chart_df["Historical Mean (ms)"].notna().any()
            )
            if has_means:
                display_cols.append("Historical Mean (ms)")
            st.caption("Latency trend — current session vs. historical mean (chronological order)")
            st.line_chart(chart_df[display_cols])

    # -----------------------------------------------------------------------
    # Section 1b — On-demand RAGAS evaluation of last Chat response
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Evaluate Last Response")
    st.caption(
        "Runs RAGAS Answer Relevancy on the last Chat response — measures how well "
        "the answer addressed the question asked (0.0–1.0). Click after any Chat query."
    )
    last_q = st.session_state.get("last_query_for_eval", "")
    last_a = st.session_state.get("last_answer_for_eval", "")
    eval_disabled = not last_q or not last_a
    if eval_disabled:
        st.caption("Send a message in the Chat tab first to enable evaluation.")
    if st.button("Evaluate last response", disabled=eval_disabled, key="metrics_eval"):
        _run_live_ragas_eval(last_q, last_a)

    # -----------------------------------------------------------------------
    # Section 2 — Experiment benchmarks (optional, run manually)
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Experiment Benchmarks")
    st.caption(
        "Formal offline tests with fixed scenarios. Run to get a baseline score — "
        "results are saved to disk and persist across sessions."
    )

    exp_label = st.selectbox("Select experiment", _EXPERIMENT_LABELS, key="metrics_exp")
    if st.button("Run Experiment", key="metrics_run"):
        _run_experiment(exp_label)

    exp_results = _load_all_experiment_results()
    if not exp_results:
        st.caption("No experiment results yet.")
        return

    raw_df = pd.DataFrame(exp_results)

    # --- Summary cards ---
    total = len(raw_df)
    success_rate = raw_df["success"].sum() / total if total else 0.0
    avg_latency = raw_df["latency_ms"].mean() if total else 0.0

    # Quality average — exclude NaN and appointment_booking (always None)
    if "experiment_name" in raw_df.columns:
        scored_rows = raw_df[raw_df["experiment_name"] != "appointment_booking"]
    else:
        scored_rows = raw_df
    if "quality_score" in scored_rows.columns:
        quality_vals = scored_rows["quality_score"].dropna()
        quality_vals = quality_vals[
            ~quality_vals.apply(lambda v: isinstance(v, float) and math.isnan(v))
        ]
    else:
        quality_vals = pd.Series(dtype=float)
    avg_quality = float(quality_vals.mean()) if len(quality_vals) > 0 else None

    quality_display = "N/A"
    quality_delta = None
    if avg_quality is not None:
        gate_pass = avg_quality >= _QUALITY_THRESHOLD
        quality_display = f"{avg_quality:.2f}"
        quality_delta = "✓ ≥ 0.80 threshold" if gate_pass else "✗ Below 0.80 threshold"

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Success rate", f"{success_rate:.0%}", f"{int(raw_df['success'].sum())}/{total}")
    col_b.metric("Avg latency", f"{avg_latency:.0f} ms")
    col_c.metric(
        "Avg quality (scored)",
        quality_display,
        quality_delta,
        delta_color=(
            "normal" if avg_quality is not None and avg_quality >= _QUALITY_THRESHOLD
            else "inverse"
        ),
    )

    # --- Results table — no experiment_name column; timestamp + human labels ---
    st.subheader("Benchmark runs")
    display_df = raw_df.copy()

    # Human-readable quality column — never shows nan
    if "quality_score" in display_df.columns and "experiment_name" in display_df.columns:
        display_df["quality"] = display_df.apply(
            lambda row: _quality_label(row["quality_score"], row.get("experiment_name", "")),
            axis=1,
        )

    # Friendly operation label — uses module-level _EXP_LABELS constant
    if "experiment_name" in display_df.columns:
        display_df["operation"] = (
            display_df["experiment_name"].map(_EXP_LABELS).fillna(display_df["experiment_name"])
        )

    # Success as symbol
    display_df["success"] = display_df["success"].map({True: "✓", False: "✗"})

    # Sort most recent first; timestamp column
    if "timestamp" in display_df.columns:
        display_df = display_df.sort_values("timestamp", ascending=False)
        display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    show_cols = ["timestamp", "operation", "success", "latency_ms", "quality", "error"]
    existing = [c for c in show_cols if c in display_df.columns]
    st.dataframe(
        display_df[existing].rename(columns={
            "timestamp": "Time",
            "operation": "Operation",
            "success": "Result",
            "latency_ms": "Latency (ms)",
            "quality": "Quality",
            "error": "Error",
        }),
        use_container_width=True,
        hide_index=True,
    )


def _run_live_ragas_eval(query: str, answer: str) -> None:
    """Run RAGAS Answer Relevancy on the last Chat response and display the score.

    Imports ragas lazily inside the function to avoid startup overhead when the
    Metrics tab is never used.  The score is also appended to session_metrics so
    it appears in the "This Session" table.

    Args:
        query: The user question sent to the Chat tab.
        answer: The assistant's response text.
    """
    try:
        import warnings  # noqa: PLC0415,I001
        from datasets import Dataset as HFDataset  # noqa: PLC0415
        from ragas import evaluate as ragas_evaluate  # noqa: PLC0415
        from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: PLC0415
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import answer_relevancy  # noqa: PLC0415
    except ImportError as exc:
        st.error(f"RAGAS evaluation requires additional packages: {exc}")
        return

    with st.spinner("Running RAGAS Answer Relevancy…"):
        eval_start = time.perf_counter()
        try:
            # RAGAS 0.4.x answer_relevancy singleton requires embeddings to be set
            # explicitly; without this the metric internally calls embed_query on a
            # bare OpenAIEmbeddings object that lacks the method in newer versions.
            answer_relevancy.embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(model=settings.embeddings.model)
            )
            dataset = HFDataset.from_dict({
                "question": [query],
                "answer": [answer],
                # answer_relevancy does not require ground_truth but ragas Dataset does
                "contexts": [[answer]],
            })
            result = ragas_evaluate(dataset, metrics=[answer_relevancy])
            raw_score = result["answer_relevancy"]
            # ragas returns numpy float or list; normalise to Python float
            score = float(raw_score[0] if hasattr(raw_score, "__len__") else raw_score)
        except Exception as exc:  # noqa: BLE001
            logger.error("[metrics] RAGAS eval failed: %s — %s", type(exc).__name__, exc)
            st.error(
                f"Evaluation failed ({type(exc).__name__}: {exc}). "
                "Ensure OPENAI_API_KEY is set in .env and the app was restarted after adding it."
            )
            return
        eval_ms = round((time.perf_counter() - eval_start) * 1000, 1)

    gate = "✓ Pass" if score >= _QUALITY_THRESHOLD else "✗ Below threshold"
    st.metric(
        "Answer Relevancy",
        f"{score:.2f}",
        f"{gate} (threshold ≥ {_QUALITY_THRESHOLD})",
        delta_color="normal" if score >= _QUALITY_THRESHOLD else "inverse",
    )

    # Append to live session metrics so it appears in the This Session table
    st.session_state["session_metrics"].append({
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operation": "RAGAS Eval",
        "success": score >= _QUALITY_THRESHOLD,
        "latency_ms": eval_ms,
        "error": "",
    })


def _run_experiment(label: str) -> None:
    """Instantiate and run the selected experiment, showing progress via st.spinner.

    Args:
        label: Experiment label from _EXPERIMENT_LABELS.
    """
    db_path = str(settings.db.sqlite_path)
    faiss_path = str(settings.db.faiss_path)

    with st.spinner(f"Running {label}…"):
        try:
            if label.startswith("Appointment"):
                _run_appointment_experiment(db_path)
            elif label.startswith("Disease"):
                _run_search_experiment()
            else:
                _run_history_experiment(db_path, faiss_path)
            st.success(f"{label} complete — results saved to experiments/*/data/.")
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[metrics] experiment failed: %s — %s", type(exc).__name__, exc
            )
            st.error(f"Experiment failed: {type(exc).__name__} — {exc}")


def _run_appointment_experiment(db_path: str) -> None:
    """Run Experiment 3 (Appointment Booking) and save results.

    Args:
        db_path: Path to the SQLite database.
    """
    scenarios_path = Path("experiments/appointment/data/appointment_scenarios.jsonl")
    if not scenarios_path.exists():
        raise FileNotFoundError(f"Scenarios file not found: {scenarios_path}")

    scenarios = []
    for line in scenarios_path.read_text().splitlines():
        line = line.strip()
        if line:
            scenarios.append(json.loads(line))

    runner = AppointmentExperiment(db_path, scenarios)
    results = runner.run()
    runner._save_results(results, "experiments/appointment/data")


def _run_search_experiment() -> None:
    """Run Experiment 2 (Disease Search) and save results."""
    serper_key = os.getenv("SERPER_API_KEY", "")
    input_path = Path("experiments/search/data/diseases_qanda.jsonl")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    runner = SearchExperiment(input_path, serper_key)
    results = runner.run()
    runner._save_results(results, "experiments/search/data")


def _run_history_experiment(db_path: str, faiss_path: str) -> None:
    """Run Experiment 1 (Medical History) and save results.

    Args:
        db_path: Path to the SQLite database.
        faiss_path: Path to the FAISS index directory.
    """
    runner = HistoryExperiment(db_path, faiss_path)
    results = runner.run()
    runner._save_results(results, "experiments/history/data")


def _load_all_experiment_results() -> list[dict]:
    """Load all ExperimentMetrics JSON files from experiments/*/data/.

    Returns:
        List of dicts, one per JSON file found. Empty list if none exist.
    """
    results: list[dict] = []
    data_dirs = [
        Path("experiments/appointment/data"),
        Path("experiments/history/data"),
        Path("experiments/search/data"),
    ]
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for json_path in sorted(data_dir.glob("*.json")):
            try:
                results.append(json.loads(json_path.read_text()))
            except Exception as exc:  # noqa: BLE001
                logger.warning("[metrics] could not parse %s: %s", json_path, exc)
    return results

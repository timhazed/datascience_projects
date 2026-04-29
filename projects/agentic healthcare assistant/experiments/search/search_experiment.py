"""Experiment 2 — Disease Search relevance.

Phase 1 refactor: SearchExperiment now inherits ExperimentRunner and returns
typed ExperimentMetrics. The rich JSONL output format (with predicted_answer,
serper_results, medline_results) is preserved alongside the typed metrics.

Search sources:
  - Serper  : Google Search via Serper API (SERPER_API_KEY in .env)
  - Medline : NCBI PubMed E-utilities (no API key required)

Input:
  experiments/search/data/<input_file>.jsonl
  Each line: {"query": "...", "answer": "..."}

Output (both in experiments/search/data/):
  <prefix>_run_<datetime>.jsonl   — one record per query with predicted_answer
  <prefix>_stats_<datetime>.jsonl — aggregate run statistics

Run:
  GROQ_API_KEY=... SERPER_API_KEY=... \\
  poetry run python experiments/search/search_experiment.py \\
      [--input experiments/search/data/diseases_qanda.jsonl]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from datasets import Dataset as HFDataset
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness as ragas_faithfulness

from experiments.experiment_runner import ExperimentRunner
from src.models.evaluation import ExperimentMetrics

logger = logging.getLogger(__name__)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "llama-3.3-70b-versatile"
MEDLINE_MAX_RESULTS = 3
SERPER_MAX_RESULTS = 5
NCBI_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
SERPER_URL = "https://google.serper.dev/search"

# ---------------------------------------------------------------------------
# LLM — hardcoded; experiments do not consume the production Settings stack
# ---------------------------------------------------------------------------
_llm = ChatGroq(model=MODEL, temperature=0.3)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a medical information specialist. Using only the search results provided, "
        "write a concise clinical answer to the question. Cite only facts present in the "
        "snippets. Do not add information not found in the search results. "
        "Answer in 1-3 sentences.",
    ),
    (
        "human",
        "Question: {query}\n\nSearch results:\n{snippets}\n\nClinical answer:",
    ),
])

_chain = _PROMPT | _llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def _search_serper(query: str, api_key: str) -> list[dict]:
    """Search via Serper API. Returns list of {title, snippet, link, source}."""
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": SERPER_MAX_RESULTS}
    try:
        resp = httpx.post(SERPER_URL, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "link": r.get("link", ""),
                "source": "serper",
            }
            for r in data.get("organic", [])[:SERPER_MAX_RESULTS]
        ]
    except Exception as exc:  # noqa: BLE001
        print(f"    [WARN] Serper error: {exc}")
        return []


def _search_medline(query: str) -> list[dict]:
    """Search NCBI PubMed via E-utilities (no API key required)."""
    try:
        search_params = urllib.parse.urlencode({
            "db": "pubmed",
            "term": query + " [Title/Abstract]",
            "retmode": "json",
            "retmax": MEDLINE_MAX_RESULTS,
        })
        with urllib.request.urlopen(
            f"{NCBI_SEARCH_URL}?{search_params}", timeout=10
        ) as resp:
            search_data = json.loads(resp.read())

        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        summary_params = urllib.parse.urlencode({
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        })
        with urllib.request.urlopen(
            f"{NCBI_SUMMARY_URL}?{summary_params}", timeout=10
        ) as resp:
            summary_data = json.loads(resp.read())

        results = []
        uids = summary_data.get("result", {}).get("uids", [])
        for uid in uids:
            article = summary_data["result"].get(uid, {})
            title = article.get("title", "")
            source_url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
            authors = ", ".join(
                a.get("name", "") for a in article.get("authors", [])[:3]
            )
            pub_date = article.get("pubdate", "")
            snippet = f"{title} — {authors} ({pub_date})"
            results.append({
                "title": title,
                "snippet": snippet,
                "link": source_url,
                "source": "medline",
            })
        return results

    except Exception as exc:  # noqa: BLE001
        print(f"    [WARN] Medline error: {exc}")
        return []


def _run_query(record: dict, serper_api_key: str) -> dict:
    """Execute search + LLM synthesis for one query record. Returns a rich result dict."""
    query = record["query"]
    ground_truth = record.get("answer", "")

    start = time.perf_counter()

    serper_results = _search_serper(query, serper_api_key)
    medline_results = _search_medline(query)
    all_results = serper_results + medline_results

    predicted_answer: str | None = None
    error: str | None = None

    if all_results:
        snippets_text = "\n".join(
            f"[{i+1}] ({r['source'].upper()}) {r['title']}: {r['snippet']}"
            for i, r in enumerate(all_results)
        )
        # Two attempts — guards against transient Groq 429/503 failures.
        for attempt in range(2):
            try:
                predicted_answer = _chain.invoke({"query": query, "snippets": snippets_text})
                error = None
                break
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                print(f"    [WARN] LLM error (attempt {attempt + 1}): {exc}")
                if attempt == 0:
                    time.sleep(1)
    else:
        error = "No search results returned from either source"

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    success = predicted_answer is not None and len(all_results) > 0

    print(f"    {'✓' if success else '✗'} {latency_ms}ms | "
          f"serper={len(serper_results)} medline={len(medline_results)}")
    if predicted_answer:
        print(f"    → {predicted_answer[:120]}{'...' if len(predicted_answer) > 120 else ''}")

    return {
        "query": query,
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "success": success,
        "latency_ms": latency_ms,
        "serper_results": serper_results,
        "medline_results": medline_results,
        "total_results": len(all_results),
        "error": error,
    }


def _build_stats(results: list[dict], input_file: str, run_dt: str) -> dict:
    """Compute aggregate statistics over all query results."""
    if not results:
        return {
            "run_datetime": run_dt,
            "input_file": input_file,
            "model": MODEL,
            "search_providers": ["serper", "medline"],
            "total_queries": 0,
            "successful_queries": 0,
            "success_rate": 0.0,
            "latency_ms": {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0},
            "avg_serper_results": 0,
            "avg_medline_results": 0,
            "queries_with_zero_results": 0,
        }

    latencies = [r["latency_ms"] for r in results]
    successful = [r for r in results if r["success"]]
    serper_counts = [len(r["serper_results"]) for r in results]
    medline_counts = [len(r["medline_results"]) for r in results]

    return {
        "run_datetime": run_dt,
        "input_file": input_file,
        "model": MODEL,
        "search_providers": ["serper", "medline"],
        "total_queries": len(results),
        "successful_queries": len(successful),
        "success_rate": round(len(successful) / len(results), 3) if results else 0.0,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1),
            "median": round(statistics.median(latencies), 1),
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
            "min": min(latencies),
            "max": max(latencies),
        },
        "avg_serper_results": round(statistics.mean(serper_counts), 1),
        "avg_medline_results": round(statistics.mean(medline_counts), 1),
        "queries_with_zero_results": sum(1 for r in results if r["total_results"] == 0),
    }


# ---------------------------------------------------------------------------
# SearchExperiment — inherits ExperimentRunner (Phase 1 refactor)
# ---------------------------------------------------------------------------

class SearchExperiment(ExperimentRunner):
    """Disease search experiment runner.

    Wraps the search + LLM synthesis pipeline and maps each query result to a
    typed ExperimentMetrics instance for the abstract run() contract.
    The rich JSONL output (predicted_answer, raw results) is saved separately
    by main() using the <prefix>_run_<datetime>.jsonl naming convention.
    """

    def __init__(self, input_path: Path, serper_api_key: str) -> None:
        """Initialise with input data and API key.

        Args:
            input_path: Path to the .jsonl input file.
            serper_api_key: Serper API key (from SERPER_API_KEY env var).
        """
        self._input_path = input_path
        self._serper_api_key = serper_api_key
        self._records = self._load_records()

    @property
    def record_count(self) -> int:
        """Number of test records loaded from the input JSONL file."""
        return len(self._records)

    def _load_records(self) -> list[dict]:
        """Load and parse records from the input JSONL file."""
        records = []
        for line in self._input_path.read_text().splitlines():
            line = line.strip().rstrip(",")
            if line:
                records.append(json.loads(line))
        return records

    def _score_faithfulness(self, rich_results: list[dict]) -> list[float | None]:
        """Compute RAGAS faithfulness for each successful query result.

        Faithfulness measures whether every claim in the predicted answer can
        be inferred from the provided search-result contexts.  A score of 1.0
        means all claims are supported; 0.0 means none are.

        Only queries with a non-empty predicted_answer are evaluated.  Failed
        or empty results receive None.  If the RAGAS evaluation call itself
        fails (e.g. OpenAI quota exceeded), all scores are returned as None
        and the error is logged rather than raised so that ExperimentMetrics
        are still produced for latency/success reporting.

        Args:
            rich_results: List of rich result dicts returned by run_rich().

        Returns:
            List of faithfulness scores (0.0–1.0) or None, one per result.
        """
        # Collect indices of results that have both a predicted answer and contexts
        valid = [
            (i, r) for i, r in enumerate(rich_results)
            if r.get("predicted_answer") and r.get("total_results", 0) > 0
        ]
        if not valid:
            return [None] * len(rich_results)

        # Build RAGAS dataset — contexts are the raw snippets fed to the LLM
        questions = [r["query"] for _, r in valid]
        answers = [r["predicted_answer"] for _, r in valid]
        contexts = [
            [
                f"({x['source'].upper()}) {x['title']}: {x['snippet']}"
                for x in (r["serper_results"] + r["medline_results"])
            ]
            for _, r in valid
        ]
        # ground_truth is required by HFDataset schema; not used by faithfulness metric
        ground_truths = [r.get("ground_truth", "") for _, r in valid]

        dataset = HFDataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        try:
            eval_result = ragas_evaluate(dataset, metrics=[ragas_faithfulness])
            # EvaluationResult is dict-like; "faithfulness" key maps to per-row scores
            faith_scores: list[float] = eval_result["faithfulness"]
            faith_map = {valid[i][0]: score for i, score in enumerate(faith_scores)}
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[search_exp] RAGAS faithfulness evaluation failed: %s — %s",
                type(exc).__name__,
                exc,
            )
            return [None] * len(rich_results)

        return [faith_map.get(i) for i in range(len(rich_results))]

    def run(self) -> list[ExperimentMetrics]:
        """Execute all queries, compute RAGAS faithfulness, and return typed metrics.

        RAGAS faithfulness (0.0–1.0) is computed as a batch after all search +
        synthesis calls complete — this avoids paying the RAGAS evaluation overhead
        per-query and allows a single Dataset.from_dict() call.

        Returns:
            List of ExperimentMetrics — one per test case.  quality_score holds
            the RAGAS faithfulness score; None for queries that failed synthesis.
        """
        rich_results = self.run_rich()
        # Batch-score faithfulness against the search-result contexts used for synthesis
        faith_scores = self._score_faithfulness(rich_results)
        return [
            ExperimentMetrics(
                experiment_name="disease_search",
                run_id=str(uuid.uuid4()),
                latency_ms=r["latency_ms"],
                success=r["success"],
                quality_score=faith_scores[i],
                # 2 search calls (Serper + Medline) + 1 LLM call when synthesis succeeded
                tool_calls_made=2 + (1 if r["predicted_answer"] is not None else 0),
                error=r["error"],
            )
            for i, r in enumerate(rich_results)
        ]

    def run_rich(self) -> list[dict]:
        """Execute all queries and return full rich result dicts.

        Results are cached after the first call — subsequent calls (e.g. from run())
        return the same list without re-invoking the search and LLM APIs.

        Returns:
            List of dicts with query, ground_truth, predicted_answer,
            success, latency_ms, serper_results, medline_results, error.
        """
        if not hasattr(self, "_cached_results"):
            self._cached_results = [
                _run_query(record, self._serper_api_key) for record in self._records
            ]
        return self._cached_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Experiment 2 — Disease Search."""
    parser = argparse.ArgumentParser(description="Disease Search Experiment 2")
    parser.add_argument(
        "--input",
        default="experiments/search/data/diseases_qanda.jsonl",
        help="Path to input .jsonl file",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    serper_api_key = os.getenv("SERPER_API_KEY", "")
    if not serper_api_key:
        raise OSError("SERPER_API_KEY not set in .env")

    prefix = input_path.stem
    run_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = input_path.parent
    run_path = out_dir / f"{prefix}_run_{run_dt}.jsonl"
    stats_path = out_dir / f"{prefix}_stats_{run_dt}.jsonl"

    print("=== Experiment 2 — Disease Search ===")
    print(f"Model  : {MODEL}")
    print(f"Output : {run_path.name}")
    print(f"Stats  : {stats_path.name}")
    print()

    experiment = SearchExperiment(input_path, serper_api_key)

    print(f"Input  : {input_path} ({experiment.record_count} queries)\n")

    # run() executes search + synthesis + RAGAS faithfulness scoring in one call
    metrics = experiment.run()
    rich_results = experiment.run_rich()  # cached — no second API call

    # Write rich JSONL output
    with run_path.open("w") as f:
        for r in rich_results:
            f.write(json.dumps(r) + "\n")

    # Write stats
    stats = _build_stats(rich_results, str(input_path), run_dt)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")

    # Save individual ExperimentMetrics JSON files (one per query)
    experiment._save_results(metrics, str(out_dir))

    # Faithfulness summary — gate against ≥ 0.80 threshold
    faith_values = [m.quality_score for m in metrics if m.quality_score is not None]
    avg_faith = statistics.mean(faith_values) if faith_values else None
    threshold_met = avg_faith is not None and avg_faith >= 0.80

    print("\n=== Summary ===")
    print(
        f"Success rate   : {stats['successful_queries']}/{stats['total_queries']}"
        f" ({stats['success_rate']*100:.0f}%)"
    )
    print(f"Avg latency    : {stats['latency_ms']['mean']}ms")
    print(f"Median latency : {stats['latency_ms']['median']}ms")
    print(f"Avg Serper hits: {stats['avg_serper_results']}")
    print(f"Avg Medline hits: {stats['avg_medline_results']}")
    if avg_faith is not None:
        gate = "✓ PASS (≥ 0.80)" if threshold_met else "✗ FAIL (< 0.80)"
        print(f"RAGAS faithfulness (avg): {avg_faith:.3f} — {gate}")
    else:
        print("RAGAS faithfulness: not computed (check OPENAI_API_KEY)")
    print(f"\nRun output : {run_path}")
    print(f"Stats      : {stats_path}")


if __name__ == "__main__":
    main()

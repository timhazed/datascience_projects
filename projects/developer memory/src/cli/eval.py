"""CLI eval runner for Developer Memory retrieval quality assessment.

Reads a JSONL ground truth dataset, runs each query through the production query
pipeline, grades results against ground truth using three deterministic metrics,
and writes per-case JSONL results + a summary markdown report.

Grading metrics (no LLM-as-judge — all deterministic string matching):
  file_recall_at_k   — did any expected_files appear in top-k retrieved chunks?
  identifier_precision — fraction of expected_identifiers found in answer or chunk metadata
  answer_grounded    — did final_answer cite at least one expected_file path?

Pass thresholds (corpus-level):
  file_recall_at_k   >= 0.75  (75% of applicable cases pass)
  identifier_precision >= 0.65 (mean across applicable cases)
  answer_grounded    >= 0.70  (70% of must-cite cases pass)

Usage:
    poetry run python -m src.cli.eval
    poetry run python -m src.cli.eval --ground-truth experiments/ground_truth/retrieval_eval.jsonl
    poetry run python -m src.cli.eval --cases autogen-001 healthcare-001
    poetry run python -m src.cli.eval --query-types identifier_search pattern_lookup
    poetry run python -m src.cli.eval --no-llm   # retrieval-only, skip synthesis
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# ── Path bootstrap — must happen before any src.* imports ────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

# Load eval-specific env first (src/cli/.env carries GROQ_API_KEY / GROQ_MODEL),
# then root .env as fallback for GITHUB_TOKEN and other shared settings.
_CLI_DIR = _PROJECT_ROOT / "src" / "cli"
load_dotenv(_CLI_DIR / ".env")
load_dotenv(_PROJECT_ROOT / ".env")

os.environ["OLLAMA_HOST"] = "http://localhost:11434"
os.environ["CHROMA_HOST"] = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("devmem.cli.eval")

_RESULTS_DIR = _PROJECT_ROOT / "src" / "cli" / "results"
_DEFAULT_GT_PATH = _PROJECT_ROOT / "src" / "cli" / "inputs" / "retrieval_eval.jsonl"
_DEFAULT_CHROMA_PATH = str(_PROJECT_ROOT / "src" / "cli" / "data" / "chroma")
_DEFAULT_COLLECTION = "eval_collection_v1"
# Read from GROQ_MODEL as set in src/cli/.env.
# gpt-oss-120b is a reasoning model — reasoning_format="hidden" is passed automatically
# in judge_answer() so the answer appears in content rather than reasoning_content.
_DEFAULT_GROQ_JUDGE_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Pass thresholds for overall verdict
_THRESHOLD_FILE_RECALL = 0.75
_THRESHOLD_IDENTIFIER_PRECISION = 0.65
_THRESHOLD_ANSWER_GROUNDED = 0.70


# ── Ground truth loading ──────────────────────────────────────────────────────


def load_ground_truth(path: Path) -> list[dict]:
    """Load and validate the JSONL ground truth dataset.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of case dicts. Each must have id, query, expected_files,
        expected_identifiers, expected_topics, must_cite_file, min_recall_k.

    Raises:
        SystemExit: If the file is missing or any case is malformed.
    """
    required_fields = {"id", "query", "expected_files", "expected_identifiers",
                       "expected_topics", "must_cite_file", "min_recall_k"}
    if not path.exists():
        print(f"ERROR: Ground truth file not found: {path}")
        sys.exit(1)

    cases = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"ERROR: Line {i} is not valid JSON: {exc}")
            sys.exit(1)
        missing = required_fields - case.keys()
        if missing:
            print(f"ERROR: Case on line {i} missing fields: {missing}")
            sys.exit(1)
        cases.append(case)

    logger.info("Loaded %d eval cases from %s", len(cases), path)
    return cases


# ── Query pipeline wiring ─────────────────────────────────────────────────────


def build_pipeline(chroma_path: str, collection: str, model: str, no_llm: bool):
    """Construct the query graph with an eval-isolated ChromaDB.

    Imports src modules after environment variables are set so ChromaLibrarianClient
    uses the eval database, not the production Docker volume.

    Args:
        chroma_path: Local directory for the eval ChromaDB PersistentClient.
        collection: ChromaDB collection name.
        model: Ollama model tag for the snippet_summarizer LLM.
        no_llm: When True, builds a retrieval-only pipeline (no snippet_summarizer).

    Returns:
        (graph, chroma) tuple — compiled LangGraph StateGraph and ChromaLibrarianClient.
    """
    # Verify the ChromaDB path exists and contains a sqlite3 file before proceeding.
    # A missing or empty path means ingest has not been run yet.
    chroma_sqlite = Path(chroma_path) / "chroma.sqlite3"
    if not chroma_sqlite.exists():
        print(f"\nERROR: No ChromaDB found at '{chroma_path}'.")
        print("Run ingest first:")
        print(f"  poetry run python -m src.cli.ingest <REPO_URL> --chroma-path \"{chroma_path}\" --collection {collection}")
        sys.exit(1)

    os.environ["CHROMA_DATA_PATH"] = chroma_path
    os.environ["CHROMA_COLLECTION"] = collection
    os.environ["OLLAMA_MODEL"] = model

    from src.db.chroma_client import ChromaLibrarianClient
    from src.llm.factory import get_llm

    chroma = ChromaLibrarianClient(collection_name=collection)

    # Fail fast if the collection is empty — ingest either didn't run or used a different collection name.
    doc_count = chroma._collection.count()
    if doc_count == 0:
        print(f"\nERROR: Collection '{collection}' at '{chroma_path}' is empty.")
        print("Either ingest hasn't run yet, or a different --collection name was used during ingest.")
        print(f"  poetry run python -m src.cli.ingest <REPO_URL> --chroma-path \"{chroma_path}\" --collection {collection}")
        sys.exit(1)

    llm_model = model

    if no_llm:
        # Build retrieval-only pipeline: stub LLM returns empty string so
        # snippet_summarizer produces final_answer="" with no Ollama call.
        # The grader reads raw_results directly so final_answer content doesn't matter.
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        from src.graphs.query_graph import build_query_graph

        stub_llm = FakeListChatModel(responses=[""])
        graph = build_query_graph(llm=stub_llm, chroma=chroma)
    else:
        from src.graphs.query_graph import build_query_graph
        llm = get_llm(model=llm_model, temperature=0.1, num_ctx=8192)
        graph = build_query_graph(llm=llm, chroma=chroma)

    doc_count = chroma._collection.count()
    logger.info(
        "Pipeline ready — collection=%s docs=%d no_llm=%s",
        collection, doc_count, no_llm,
    )
    if doc_count == 0:
        logger.warning(
            "ChromaDB collection '%s' is empty. Run `python -m src.cli.ingest` first.",
            collection,
        )
    return graph, chroma


# ── Query execution ───────────────────────────────────────────────────────────


def run_query(graph, case: dict) -> dict:
    """Run a single eval case through the query graph.

    Args:
        graph: Compiled LangGraph StateGraph from build_pipeline().
        case: Ground truth case dict.

    Returns:
        Dict with raw_results, final_answer, error, trace, elapsed_seconds.
    """
    t0 = time.monotonic()
    try:
        state = graph.invoke(
            {"query": case["query"], "tech_filter": None, "trace": []},
            config={"recursion_limit": 50},
        )
    except Exception as exc:
        logger.error("Query failed for case %s: [%s] %s", case["id"], type(exc).__name__, exc)
        return {
            "raw_results": [],
            "final_answer": None,
            "error": str(exc),
            "trace": [],
            "elapsed_seconds": round(time.monotonic() - t0, 2),
        }

    return {
        "raw_results": state.get("raw_results", []),
        "final_answer": state.get("final_answer"),
        "error": state.get("error"),
        "trace": state.get("trace", []),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }


# ── LLM-as-judge ─────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are a retrieval quality judge. Your job is to decide whether a model-generated \
answer is grounded in the retrieved documents — meaning it draws its content from \
those documents rather than from general world knowledge alone.

An answer is grounded (YES) if ANY of these are true:
- It references or quotes at least one of the retrieved file names or paths
- It names specific code identifiers (class names, function names, variable names) \
that would only appear in those files
- Its explanation of HOW the system works matches details that come from the retrieved \
files, not generic descriptions

An answer is NOT grounded (NO) only if it could have been written without seeing any \
of the retrieved files at all — i.e. it is entirely generic.

Respond with exactly one word: YES or NO.

Query: {query}

Retrieved files:
{retrieved_files}

Answer:
{answer}

Is the answer grounded in the retrieved content? (YES/NO)"""


def judge_answer(query: str, retrieved_files: list[str], answer: str, judge_model: str) -> int | None:
    """Call Groq LLM to judge whether the answer is grounded in retrieved content.

    Uses a binary YES/NO rubric. Returns 1 (grounded), 0 (not grounded), or None
    on API error so the metric is treated as exempt rather than a hard failure.

    Args:
        query: The original user query.
        retrieved_files: List of file paths returned by the retriever.
        answer: The model's final synthesized answer.
        judge_model: Groq model name (e.g. "openai/gpt-oss-120b").

    Returns:
        1 if grounded, 0 if not, None on API error.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("judge_answer: GROQ_API_KEY not set — skipping judge")
        return None
    if not answer.strip():
        return 0

    from langchain_core.messages import HumanMessage
    from langchain_groq import ChatGroq

    prompt = _JUDGE_PROMPT.format(
        query=query,
        retrieved_files="\n".join(f"  - {f}" for f in retrieved_files[:10]),
        answer=answer[:1500],  # cap to stay within context
    )

    # gpt-oss-* are reasoning models. Use reasoning_effort="medium" (confirmed working
    # in experiments/eval_output_quality.py) — not reasoning_format, which doesn't
    # redirect content correctly for binary YES/NO output. Non-reasoning models ignore
    # reasoning_effort so it's safe to pass unconditionally.
    try:
        llm = ChatGroq(
            model=judge_model,
            temperature=0,
            max_tokens=1024,  # reasoning models use ~74-200 tokens on hidden reasoning before emitting the answer
            api_key=api_key,
            reasoning_effort="medium",
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        verdict = response.content.strip().upper()
        logger.debug("judge_answer: model=%s verdict=%r query=%r", judge_model, verdict, query[:60])
        return 1 if "YES" in verdict else 0
    except Exception as exc:
        logger.warning("judge_answer failed [%s]: %s", type(exc).__name__, str(exc)[:200])
        return None


# ── Grading ───────────────────────────────────────────────────────────────────


def grade_case(case: dict, query_output: dict, llm_judge: bool = False, judge_model: str = "") -> dict:
    """Grade a single case against ground truth.

    Three primary metrics:
      file_recall_at_k     — binary hit: any expected file in top-k retrieved chunks
      identifier_precision — fraction of expected identifiers found in answer + chunk metadata
      answer_grounded      — binary: answer is grounded in retrieved content.
                             When --llm-judge: Groq judge (YES/NO rubric).
                             Otherwise: filename stem of any expected file appears in answer.

    All metrics are None (exempt) when the case has no applicable ground truth
    (e.g. expected_files=[] for cross-repo synthesis cases).

    Args:
        case: Ground truth case dict.
        query_output: Result dict from run_query().
        llm_judge: When True, use Groq LLM-as-judge for answer_grounded instead of stem match.
        judge_model: Groq model name to use when llm_judge=True.

    Returns:
        Dict of metric scores plus diagnostic fields.
    """
    raw_results: list[dict] = query_output["raw_results"]
    final_answer: str = query_output["final_answer"] or ""
    k = case["min_recall_k"]

    # Collect retrieved file paths in rank order
    retrieved_files = [
        r["metadata"].get("file_path", "") for r in raw_results
    ]
    top_k_files = set(retrieved_files[:k])

    # ── Metric 1: File Recall@k ───────────────────────────────────────────
    expected_files: list[str] = case["expected_files"]
    file_recall_at_k = (
        int(bool(set(expected_files) & top_k_files)) if expected_files else None
    )  # None = exempt (cross-repo synthesis)

    # ── Metric 2: Identifier Precision ───────────────────────────────────
    expected_ids: list[str] = case["expected_identifiers"]
    if expected_ids:
        # key_identifiers is stored as a JSON-encoded string in ChromaDB metadata,
        # not a Python list. Parse it before joining.
        def _parse_ids(raw) -> list[str]:
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    return parsed if isinstance(parsed, list) else []
                except (json.JSONDecodeError, ValueError):
                    return []
            return raw or []

        # Search in final_answer AND in key_identifiers metadata of all retrieved chunks
        metadata_ids_text = " ".join(
            " ".join(_parse_ids(r["metadata"].get("key_identifiers")))
            for r in raw_results
        )
        search_corpus = final_answer + " " + metadata_ids_text
        hits = sum(1 for id_ in expected_ids if id_ in search_corpus)
        identifier_precision = round(hits / len(expected_ids), 3)
    else:
        identifier_precision = None  # exempt

    answer_lower = final_answer.lower()

    # ── Metric 3: Answer Grounded (deterministic stem-match) ─────────────
    # Checks whether the filename stem of any expected file appears in the answer.
    # This is more reliable than LLM-as-judge for small models that paraphrase paths
    # rather than quoting them verbatim but still reference the correct file names.
    if case["must_cite_file"] and expected_files:
        answer_grounded = int(any(
            Path(path).stem.lower() in answer_lower
            for path in expected_files
        ))
    else:
        answer_grounded = None  # exempt

    # ── Diagnostic: LLM Judge Score (non-gating) ─────────────────────────
    # When --llm-judge is active, run the Groq judge as an additional signal.
    # Not included in the composite score or verdict — used for model comparison.
    judge_score = None
    if llm_judge and case["must_cite_file"] and expected_files and final_answer.strip():
        judge_score = judge_answer(
            query=case["query"],
            retrieved_files=retrieved_files,
            answer=final_answer,
            judge_model=judge_model,
        )

    # ── Supporting: Topic Coverage ────────────────────────────────────────
    expected_topics: list[str] = case["expected_topics"]
    topics_hit = sum(1 for t in expected_topics if t.lower() in answer_lower)
    topic_coverage = round(topics_hit / len(expected_topics), 3) if expected_topics else None

    # ── Composite score (mean of non-None primary metrics) ────────────────
    primary = [m for m in [file_recall_at_k, identifier_precision, answer_grounded] if m is not None]
    composite = round(sum(primary) / len(primary), 3) if primary else None

    return {
        "file_recall_at_k": file_recall_at_k,
        "identifier_precision": identifier_precision,
        "answer_grounded": answer_grounded,
        "judge_score": judge_score,
        "topic_coverage": topic_coverage,
        "composite_score": composite,
        "retrieved_files_top_k": list(top_k_files),
        "retrieved_files_all": retrieved_files,
    }


# ── Summary and reporting ─────────────────────────────────────────────────────


def _compute_corpus_metrics(case_results: list[dict]) -> dict:
    """Aggregate per-case grades into corpus-level pass/fail metrics.

    Args:
        case_results: List of dicts, each containing the case, query_output, and grades.

    Returns:
        Dict with corpus-level metric values and pass/fail booleans.
    """
    recall_scores = [r["grades"]["file_recall_at_k"] for r in case_results
                     if r["grades"]["file_recall_at_k"] is not None]
    precision_scores = [r["grades"]["identifier_precision"] for r in case_results
                        if r["grades"]["identifier_precision"] is not None]
    grounded_scores = [r["grades"]["answer_grounded"] for r in case_results
                       if r["grades"]["answer_grounded"] is not None]
    judge_scores = [r["grades"]["judge_score"] for r in case_results
                    if r["grades"].get("judge_score") is not None]

    recall_rate = round(sum(recall_scores) / len(recall_scores), 3) if recall_scores else None
    precision_mean = round(sum(precision_scores) / len(precision_scores), 3) if precision_scores else None
    grounded_rate = round(sum(grounded_scores) / len(grounded_scores), 3) if grounded_scores else None
    judge_rate = round(sum(judge_scores) / len(judge_scores), 3) if judge_scores else None

    return {
        "file_recall_rate": recall_rate,
        "file_recall_applicable_cases": len(recall_scores),
        "file_recall_pass": recall_rate >= _THRESHOLD_FILE_RECALL if recall_rate is not None else None,
        "identifier_precision_mean": precision_mean,
        "identifier_precision_applicable_cases": len(precision_scores),
        "identifier_precision_pass": precision_mean >= _THRESHOLD_IDENTIFIER_PRECISION if precision_mean is not None else None,
        "answer_grounded_rate": grounded_rate,
        "answer_grounded_applicable_cases": len(grounded_scores),
        "answer_grounded_pass": grounded_rate >= _THRESHOLD_ANSWER_GROUNDED if grounded_rate is not None else None,
        # LLM judge rate is diagnostic only — not a gating metric
        "judge_rate": judge_rate,
        "judge_applicable_cases": len(judge_scores),
    }


def _verdict(corpus: dict) -> str:
    """Return PASS or FAIL based on all three gating metrics."""
    checks = [corpus["file_recall_pass"], corpus["identifier_precision_pass"], corpus["answer_grounded_pass"]]
    if any(c is False for c in checks):
        return "FAIL"
    if all(c is True for c in checks):
        return "PASS"
    return "INCOMPLETE"  # some metrics had no applicable cases


def save_results(case_results: list[dict], corpus_metrics: dict, config: dict, results_dir: Path = _RESULTS_DIR) -> tuple[Path, Path]:
    """Write per-case JSONL and summary markdown to results_dir.

    Args:
        case_results: List of per-case result dicts.
        corpus_metrics: Aggregated corpus-level metrics dict.
        config: Run configuration dict (model, chroma_path, collection, etc.).
        results_dir: Directory to write output files (default: src/cli/results/).

    Returns:
        (jsonl_path, md_path) tuple.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    model_slug = config["llm_model"].replace(":", "_").replace("/", "_")
    run_id = f"retrieval_eval_{model_slug}_{timestamp}"

    # ── Per-case JSONL ────────────────────────────────────────────────────
    jsonl_path = results_dir / f"{run_id}.jsonl"
    with jsonl_path.open("w") as f:
        for r in case_results:
            row = {
                "case_id": r["case"]["id"],
                "query": r["case"]["query"],
                "query_type": r["case"].get("query_type"),
                "difficulty": r["case"].get("difficulty"),
                "elapsed_seconds": r["query_output"]["elapsed_seconds"],
                "raw_results_count": len(r["query_output"]["raw_results"]),
                "final_answer_preview": (r["query_output"]["final_answer"] or "")[:300],
                "error": r["query_output"]["error"],
                **r["grades"],
            }
            f.write(json.dumps(row) + "\n")

    # ── Summary Markdown ──────────────────────────────────────────────────
    verdict = _verdict(corpus_metrics)
    md_lines = [
        f"# Retrieval Eval Report — {timestamp}",
        "",
        f"**Verdict: {verdict}**",
        "",
        "## Run Configuration",
        f"- Model: `{config['llm_model']}`",
        f"- Embed: `{config['embed_model']}`",
        f"- Judge: `{config.get('judge_model', 'none (deterministic)')}`",
        f"- ChromaDB: `{config['chroma_path']}`",
        f"- Collection: `{config['collection']}`",
        f"- Cases run: {len(case_results)}",
        f"- Total elapsed: {sum(r['query_output']['elapsed_seconds'] for r in case_results):.1f}s",
        "",
        "## Corpus Metrics",
        "",
        "| Metric | Value | Threshold | Pass? |",
        "|--------|-------|-----------|-------|",
    ]

    def _fmt(val):
        if val is None:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.3f}"
        return str(val)

    judge_row = (
        f"| LLM Judge (diagnostic) | {_fmt(corpus_metrics['judge_rate'])} | — | — |"
        if corpus_metrics.get("judge_applicable_cases", 0) > 0 else ""
    )
    md_lines += [
        f"| File Recall@k | {_fmt(corpus_metrics['file_recall_rate'])} | ≥{_THRESHOLD_FILE_RECALL} | {'✓' if corpus_metrics['file_recall_pass'] else '✗'} |",
        f"| Identifier Precision | {_fmt(corpus_metrics['identifier_precision_mean'])} | ≥{_THRESHOLD_IDENTIFIER_PRECISION} | {'✓' if corpus_metrics['identifier_precision_pass'] else '✗'} |",
        f"| Answer Grounded (stem-match) | {_fmt(corpus_metrics['answer_grounded_rate'])} | ≥{_THRESHOLD_ANSWER_GROUNDED} | {'✓' if corpus_metrics['answer_grounded_pass'] else '✗'} |",
    ]
    if judge_row:
        md_lines.append(judge_row)
    md_lines += [
        "",
        "## Per-Case Results",
        "",
        "| ID | Type | Difficulty | Recall@k | ID Prec | Grounded | Judge | Composite | Elapsed |",
        "|----|------|------------|----------|---------|----------|-------|-----------|---------|",
    ]

    for r in case_results:
        c = r["case"]
        g = r["grades"]
        md_lines.append(
            f"| {c['id']} | {c.get('query_type','?')} | {c.get('difficulty','?')} "
            f"| {_fmt(g['file_recall_at_k'])} | {_fmt(g['identifier_precision'])} "
            f"| {_fmt(g['answer_grounded'])} | {_fmt(g.get('judge_score'))} "
            f"| {_fmt(g['composite_score'])} "
            f"| {r['query_output']['elapsed_seconds']:.1f}s |"
        )

    # Failing cases detail
    failing = [r for r in case_results
               if r["grades"]["composite_score"] is not None and r["grades"]["composite_score"] < 0.5]
    if failing:
        md_lines += ["", "## Failing Cases (composite < 0.5)", ""]
        for r in failing[:10]:
            c = r["case"]
            md_lines += [
                f"### {c['id']} — {c['query']}",
                f"- Notes: {c.get('notes', '')}",
                f"- Retrieved: {r['grades']['retrieved_files_all'][:5]}",
                f"- Expected: {c['expected_files']}",
                f"- Final answer preview: {(r['query_output']['final_answer'] or '')[:200]}",
                "",
            ]

    md_path = results_dir / f"{run_id}_summary.md"
    md_path.write_text("\n".join(md_lines))

    logger.info("JSONL results → %s", jsonl_path)
    logger.info("Summary report → %s", md_path)
    return jsonl_path, md_path


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments, run eval, write results."""
    parser = argparse.ArgumentParser(
        prog="devmem-eval",
        description="Grade Developer Memory retrieval quality against a ground truth dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run python -m src.cli.eval
  poetry run python -m src.cli.eval --cases autogen-001 autogen-002
  poetry run python -m src.cli.eval --query-types identifier_search
  poetry run python -m src.cli.eval --no-llm  # retrieval only, skip synthesis
        """,
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=_DEFAULT_GT_PATH,
        metavar="PATH",
        help=f"Path to JSONL ground truth file (default: {_DEFAULT_GT_PATH}).",
    )
    parser.add_argument(
        "--chroma-path",
        default=_DEFAULT_CHROMA_PATH,
        metavar="PATH",
        help=f"Eval ChromaDB directory (default: {_DEFAULT_CHROMA_PATH}).",
    )
    parser.add_argument(
        "--collection",
        default=_DEFAULT_COLLECTION,
        metavar="NAME",
        help=f"ChromaDB collection name (default: {_DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        metavar="ID",
        help="Run only specific case IDs (e.g. autogen-001 healthcare-001).",
    )
    parser.add_argument(
        "--query-types",
        nargs="+",
        metavar="TYPE",
        choices=["tech_lookup", "identifier_search", "pattern_lookup", "cross_repo", "methodology"],
        help="Filter cases by query_type.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        metavar="N",
        help="Override min_recall_k for all cases.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "gemma4:e4b"),
        metavar="TAG",
        help="Ollama model tag for query synthesis (default: $OLLAMA_MODEL or gemma4:e4b).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM synthesis (snippet_summarizer). Grade retrieval metrics only.",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use Groq LLM-as-judge for answer_grounded metric (requires GROQ_API_KEY in src/cli/.env).",
    )
    parser.add_argument(
        "--judge-model",
        default=_DEFAULT_GROQ_JUDGE_MODEL,
        metavar="MODEL",
        help=f"Groq model for LLM-as-judge (default: $GROQ_MODEL or {_DEFAULT_GROQ_JUDGE_MODEL}). "
             "Reasoning models (gpt-oss-*) are supported — reasoning_format=hidden is applied automatically.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_RESULTS_DIR,
        metavar="PATH",
        help=f"Directory for result files (default: {_RESULTS_DIR}).",
    )

    args = parser.parse_args()
    results_dir = args.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load and filter ground truth
    all_cases = load_ground_truth(args.ground_truth)
    cases = all_cases

    if args.cases:
        id_set = set(args.cases)
        cases = [c for c in cases if c["id"] in id_set]
        missing = id_set - {c["id"] for c in cases}
        if missing:
            print(f"WARNING: Case IDs not found in ground truth: {missing}")

    if args.query_types:
        type_set = set(args.query_types)
        cases = [c for c in cases if c.get("query_type") in type_set]

    if args.k is not None:
        for c in cases:
            c["min_recall_k"] = args.k

    if not cases:
        print("No cases to run after filtering.")
        sys.exit(0)

    logger.info("Running %d eval cases", len(cases))

    # Build pipeline
    graph, chroma = build_pipeline(
        chroma_path=args.chroma_path,
        collection=args.collection,
        model=args.model,
        no_llm=args.no_llm,
    )

    if args.llm_judge and not os.environ.get("GROQ_API_KEY"):
        print("ERROR: --llm-judge requires GROQ_API_KEY in src/cli/.env")
        sys.exit(1)

    config = {
        "llm_model": args.model if not args.no_llm else "none (--no-llm)",
        "embed_model": "nomic-embed-text",
        "chroma_path": args.chroma_path,
        "collection": args.collection,
        "judge_model": args.judge_model if args.llm_judge else "none (deterministic)",
    }

    # Run each case
    case_results = []
    for i, case in enumerate(cases, start=1):
        logger.info("[%d/%d] %s — %s", i, len(cases), case["id"], case["query"][:60])
        query_output = run_query(graph, case)
        grades = grade_case(case, query_output, llm_judge=args.llm_judge, judge_model=args.judge_model)
        case_results.append({"case": case, "query_output": query_output, "grades": grades})
        logger.info(
            "  recall@%d=%s  id_prec=%s  grounded=%s  composite=%s  (%.1fs)",
            case["min_recall_k"],
            grades["file_recall_at_k"],
            grades["identifier_precision"],
            grades["answer_grounded"],
            grades["composite_score"],
            query_output["elapsed_seconds"],
        )

    # Aggregate and report
    corpus_metrics = _compute_corpus_metrics(case_results)
    verdict = _verdict(corpus_metrics)

    jsonl_path, md_path = save_results(case_results, corpus_metrics, config, results_dir)

    # Print summary to stdout
    print(f"\n{'='*60}")
    print(f"Verdict: {verdict}")
    print(f"{'='*60}")
    print(f"  File Recall@k:         {corpus_metrics['file_recall_rate']} (threshold ≥{_THRESHOLD_FILE_RECALL})")
    print(f"  Identifier Precision:  {corpus_metrics['identifier_precision_mean']} (threshold ≥{_THRESHOLD_IDENTIFIER_PRECISION})")
    print(f"  Answer Grounded:       {corpus_metrics['answer_grounded_rate']} (threshold ≥{_THRESHOLD_ANSWER_GROUNDED}) [stem-match]")
    if corpus_metrics.get("judge_applicable_cases", 0) > 0:
        print(f"  LLM Judge (diag):      {corpus_metrics['judge_rate']} [{args.judge_model}]")
    print(f"\nResults: {jsonl_path}")
    print(f"Report:  {md_path}")


if __name__ == "__main__":
    main()

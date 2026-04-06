"""
Intent Tester — Benchmarks the NewsGenie supervisor prompt against labeled ground truth.

Research Question:
    How accurately does the supervisor prompt + regex parser route user queries to the
    correct agents (Business News, Sports News, General News, Web Search), and how
    closely does the extracted sub-query match the expected query?

Usage:
    python intent_tester.py --input data/sample_prompts.jsonl
    python intent_tester.py --input ../../data/Intent_Validation_Mini.jsonl
    python intent_tester.py --input data/sample_prompts.jsonl --output /tmp/results/

Input JSONL (one JSON object per line, ground_truth uses supervisor output format):
    {"prompt": "Who won the Super Bowl and what is Bitcoin at?",
     "ground_truth": "INTENT: Sports News | QUERY: Who won the Super Bowl\\nINTENT: Business News | QUERY: current price of Bitcoin"}

Output JSONL adds evaluation fields to each input record.
A companion .stats.json is written alongside the output JSONL.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Supervisor prompt — matches ARCHITECTURE.md §3 (`SUPERVISOR_PROMPT` in supervisor_node).
# Benchmarks use categories_hint="all" (no UI filter). Streamlit passes real categories.
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """
You are a News Facilitator. Your task is to decompose a user query into specific intents for specialized agents: [Business News, Sports News, General News, Web Search].

USER-SELECTED CATEGORIES (from UI): {categories_hint}
If this is not "all", prefer routing into these domains when the query supports it. The system applies a deterministic post-filter as well.

INTENT DEFINITIONS:
1. Business News: Use for EVERYTHING involving money, company earnings, stock prices (e.g., Nvidia, AAPL), cryptocurrency values (e.g., Bitcoin), inflation, the Federal Reserve, exchange rates, supply chain disruptions, and semiconductor industry news (e.g., chip shortages that affect companies or markets).
2. Sports News: Use for scores, game results (e.g., "who won"), team news, athlete updates, and tournament brackets.
3. General News: Use for global headlines, politics, international conflict, and major non-financial/non-sporting events.
4. Web Search: Use ONLY for non-news facts (e.g., "how to bake bread"), local niche data (e.g., "Austin festivals"), or weather.

RULES:
1. Identify up to 3 distinct intents. Do not repeat the same intent/query pair.
2. For each intent, extract the specific sub-query relevant ONLY to that agent.
3. If a query mentions a specific "price," "rate," or "ticker," it MUST be Business News.
4. If a query asks "who won" or for a "score," it MUST be Sports News.
5. If unsure or information is missing, DO NOT GUESS. Default to:
   INTENT: Web Search | QUERY: [Full user sub-query]

6. Respond ONLY in the following format:
INTENT: [Intent Name] | QUERY: [Extracted Sub-query]

USER PROMPT: {prompt}
"""

VALID_INTENTS = {"Business News", "Sports News", "General News", "Web Search"}
INTENT_PATTERN = re.compile(r"INTENT:\s*(.*?)\s*\|\s*QUERY:\s*(.*)")


# ---------------------------------------------------------------------------
# LLM loading
# ---------------------------------------------------------------------------


def load_llm(provider: str, model: str, api_key: str):
    """Load and return the LangChain LLM client for the specified provider."""
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=0.0, max_tokens=300, api_key=api_key)
    elif provider == "groq":
        return ChatGroq(model=model, temperature=0.0, max_tokens=300, api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider '{provider}'. Must be 'openai' or 'groq'.")


# ---------------------------------------------------------------------------
# Parsing — same regex for both LLM output and ground truth
# ---------------------------------------------------------------------------


def parse_intent_string(text: str) -> list[dict]:
    """Parse an INTENT: X | QUERY: Y string into a list of dicts.

    Works on both LLM output and ground_truth since they use the same format.
    Multi-intent inputs use newline-separated pairs.
    Unknown intent values are normalised to 'Web Search'.
    Zero matches → single Web Search fallback using the full input as the query.
    """
    matches = INTENT_PATTERN.findall(text)
    result = []
    for intent, query in matches:
        intent = intent.strip()
        query = query.strip()
        result.append({
            "intent": intent if intent in VALID_INTENTS else "Web Search",
            "query": query,
        })
    return result


# ---------------------------------------------------------------------------
# Supervisor node — mirrors src/graph/supervisor_node.py
# ---------------------------------------------------------------------------


def run_supervisor(prompt: str, llm) -> tuple[list[dict], str]:
    """Send a prompt through the supervisor node.

    Returns (parsed_steps, raw_llm_output).
    parsed_steps is a list of {"intent": str, "query": str} dicts.
    Falls back to a single Web Search step if the LLM returns no parseable output.
    """
    filled = SUPERVISOR_PROMPT.format(prompt=prompt, categories_hint="all")
    response = llm.invoke(filled)
    raw = response.content
    steps = parse_intent_string(raw)
    if not steps:
        steps = [{"intent": "Web Search", "query": prompt}]
    return steps, raw


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two strings based on word tokens."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return round(len(tokens_a & tokens_b) / len(union), 3)


def evaluate(predicted: list[dict], ground_truth: list[dict]) -> dict:
    """Evaluate predicted steps against ground truth steps.

    Intent evaluation uses set comparison (order-independent).
    Query evaluation pairs each matched intent and computes Jaccard similarity.
    """
    pred_intents = [s["intent"] for s in predicted]
    gt_intents = [s["intent"] for s in ground_truth]
    pred_set = set(pred_intents)
    gt_set = set(gt_intents)

    tp_intents = pred_set & gt_set
    fp_intents = pred_set - gt_set
    fn_intents = gt_set - pred_set

    intent_precision = len(tp_intents) / len(pred_set) if pred_set else 0.0
    intent_recall = len(tp_intents) / len(gt_set) if gt_set else 0.0
    intent_f1 = (
        2 * intent_precision * intent_recall / (intent_precision + intent_recall)
        if (intent_precision + intent_recall) > 0
        else 0.0
    )

    # Query similarity — only for intents that matched
    query_similarities = []
    for intent in tp_intents:
        pred_query = next((s["query"] for s in predicted if s["intent"] == intent), "")
        gt_query = next((s["query"] for s in ground_truth if s["intent"] == intent), "")
        query_similarities.append(jaccard_similarity(pred_query, gt_query))

    avg_query_similarity = (
        round(sum(query_similarities) / len(query_similarities), 3)
        if query_similarities
        else 0.0
    )

    return {
        "intent_exact_match": pred_set == gt_set,
        "intent_partial_match": bool(tp_intents),
        "intent_precision": round(intent_precision, 3),
        "intent_recall": round(intent_recall, 3),
        "intent_f1": round(intent_f1, 3),
        "avg_query_similarity": avg_query_similarity,
        "false_positive_intents": sorted(fp_intents),
        "false_negative_intents": sorted(fn_intents),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_statistics(
    results: list[dict], elapsed_seconds: float, provider: str, model: str
) -> dict:
    """Compute aggregate run statistics across all prediction results."""
    n = len(results)
    if n == 0:
        return {}

    errors = [r for r in results if r.get("error")]
    valid = [r for r in results if not r.get("error") and r.get("_metrics")]
    n_valid = len(valid)

    exact_matches = sum(1 for r in valid if r["_metrics"]["intent_exact_match"])
    partial_matches = sum(1 for r in valid if r["_metrics"]["intent_partial_match"])

    precisions = [r["_metrics"]["intent_precision"] for r in valid]
    recalls = [r["_metrics"]["intent_recall"] for r in valid]
    f1s = [r["_metrics"]["intent_f1"] for r in valid]
    query_sims = [r["_metrics"]["avg_query_similarity"] for r in valid]

    # Per-intent accuracy tracking
    intent_tp = {i: 0 for i in VALID_INTENTS}
    intent_gt_total = {i: 0 for i in VALID_INTENTS}
    intent_pred_total = {i: 0 for i in VALID_INTENTS}
    intent_query_sims: dict[str, list[float]] = {i: [] for i in VALID_INTENTS}

    for r in valid:
        for step in r["_gt_parsed"]:
            i = step["intent"]
            if i in VALID_INTENTS:
                intent_gt_total[i] += 1
        for step in r["_pred_parsed"]:
            i = step["intent"]
            if i in VALID_INTENTS:
                intent_pred_total[i] += 1
        for intent in set(s["intent"] for s in r["_pred_parsed"]) & set(
            s["intent"] for s in r["_gt_parsed"]
        ):
            if intent in VALID_INTENTS:
                intent_tp[intent] += 1
                pred_q = next(
                    (s["query"] for s in r["_pred_parsed"] if s["intent"] == intent), ""
                )
                gt_q = next(
                    (s["query"] for s in r["_gt_parsed"] if s["intent"] == intent), ""
                )
                intent_query_sims[intent].append(jaccard_similarity(pred_q, gt_q))

    per_intent = {}
    for intent in sorted(VALID_INTENTS):
        gt = intent_gt_total[intent]
        pred = intent_pred_total[intent]
        tp = intent_tp[intent]
        prec = tp / pred if pred else 0.0
        rec = tp / gt if gt else 0.0
        sims = intent_query_sims[intent]
        per_intent[intent] = {
            "ground_truth_count": gt,
            "predicted_count": pred,
            "true_positives": tp,
            "intent_precision": round(prec, 3),
            "intent_recall": round(rec, 3),
            "avg_query_similarity": round(sum(sims) / len(sims), 3) if sims else 0.0,
        }

    return {
        "run_timestamp": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "total_prompts": n,
        "valid_prompts": n_valid,
        "errors": len(errors),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "intent_exact_match_count": exact_matches,
        "intent_exact_match_accuracy": round(exact_matches / n_valid if n_valid else 0, 3),
        "intent_partial_match_count": partial_matches,
        "intent_partial_match_accuracy": round(partial_matches / n_valid if n_valid else 0, 3),
        "avg_intent_precision": round(sum(precisions) / len(precisions) if precisions else 0, 3),
        "avg_intent_recall": round(sum(recalls) / len(recalls) if recalls else 0, 3),
        "avg_intent_f1": round(sum(f1s) / len(f1s) if f1s else 0, 3),
        "avg_query_similarity": round(sum(query_sims) / len(query_sims) if query_sims else 0, 3),
        "per_intent": per_intent,
    }


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------


def build_output_path(input_path: Path, output_dir: Path, provider: str, model: str) -> Path:
    """Return {output_dir}/{input_stem}_RUN_{datetime}.jsonl"""
    model_leaf = model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{input_path.stem}_RUN_{provider}_{model_leaf}_{timestamp}.jsonl"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_statistics(stats: dict) -> None:
    """Print a formatted statistics summary to stdout."""
    n = stats["valid_prompts"]
    print()
    print("=" * 66)
    print("  RUN STATISTICS")
    print("=" * 66)
    print(f"  Provider / Model    : {stats['provider']} / {stats['model']}")
    print(f"  Run timestamp       : {stats['run_timestamp']}")
    print(f"  Total prompts       : {stats['total_prompts']}")
    print(f"  Errors              : {stats['errors']}")
    print(f"  Elapsed             : {stats['elapsed_seconds']}s")
    print()
    print(
        f"  Intent exact match  : {stats['intent_exact_match_count']}/{n}"
        f"  ({stats['intent_exact_match_accuracy'] * 100:.1f}%)"
    )
    print(
        f"  Intent partial match: {stats['intent_partial_match_count']}/{n}"
        f"  ({stats['intent_partial_match_accuracy'] * 100:.1f}%)"
    )
    print(f"  Avg intent precision: {stats['avg_intent_precision']:.3f}")
    print(f"  Avg intent recall   : {stats['avg_intent_recall']:.3f}")
    print(f"  Avg intent F1       : {stats['avg_intent_f1']:.3f}")
    print(f"  Avg query similarity: {stats['avg_query_similarity']:.3f}  (Jaccard on word tokens)")
    print()
    print(
        f"  {'Intent':<22} {'GT':>4} {'Pred':>5} {'TP':>4}"
        f" {'Prec':>6} {'Recall':>7} {'Q-Sim':>7}"
    )
    print("  " + "-" * 60)
    for intent, m in stats["per_intent"].items():
        print(
            f"  {intent:<22} {m['ground_truth_count']:>4} {m['predicted_count']:>5}"
            f" {m['true_positives']:>4} {m['intent_precision']:>6.3f}"
            f" {m['intent_recall']:>7.3f} {m['avg_query_similarity']:>7.3f}"
        )
    print("=" * 66)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the intent tester CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark the NewsGenie supervisor prompt against labeled ground truth."
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: data/ next to this script)",
    )
    args = parser.parse_args()

    # -- Environment ----------------------------------------------------------
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("SUPERVISOR_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY" if provider == "openai" else "GROQ_API_KEY", "")

    if provider not in ("openai", "groq"):
        print(f"ERROR: LLM_PROVIDER='{provider}' is not supported. Use 'openai' or 'groq'.")
        sys.exit(1)

    if not api_key:
        key_name = "OPENAI_API_KEY" if provider == "openai" else "GROQ_API_KEY"
        print(f"ERROR: {key_name} is not set in the environment or .env file.")
        sys.exit(1)

    # -- Paths ----------------------------------------------------------------
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    script_dir = Path(__file__).parent
    output_dir = Path(args.output).expanduser().resolve() if args.output else script_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(input_path, output_dir, provider, model)

    print(f"Provider  : {provider}")
    print(f"Model     : {model}")
    print(f"Input     : {input_path}")
    print(f"Output    : {output_path}")
    print()

    # -- Load LLM -------------------------------------------------------------
    try:
        llm = load_llm(provider, model, api_key)
    except Exception as exc:
        print(f"ERROR: Failed to load LLM: {exc}")
        sys.exit(1)

    # -- Read input JSONL -----------------------------------------------------
    records = []
    with input_path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"WARNING: Skipping line {line_no} — invalid JSON: {exc}")

    if not records:
        print("ERROR: No valid records found in input file.")
        sys.exit(1)

    print(f"Loaded {len(records)} prompts. Running...\n")

    # -- Run predictions ------------------------------------------------------
    results = []
    start = time.perf_counter()

    for i, record in enumerate(records, 1):
        prompt = record.get("prompt", "")
        ground_truth_raw = record.get("ground_truth", "")
        ground_truth_parsed = parse_intent_string(ground_truth_raw)

        label = prompt[:55] + "..." if len(prompt) > 55 else prompt
        print(f"  [{i:3d}/{len(records)}] {label:<58}", end="", flush=True)

        try:
            predicted_parsed, raw_output = run_supervisor(prompt, llm)
            metrics = evaluate(predicted_parsed, ground_truth_parsed)

            # Output record — clean and minimal
            result = {
                "prompt": prompt,
                "ground_truth": ground_truth_raw,
                "predicted_intent": raw_output.strip(),
            }
            # Internal fields carried for statistics — not written to output JSONL
            result["_gt_parsed"] = ground_truth_parsed
            result["_pred_parsed"] = predicted_parsed
            result["_metrics"] = metrics

            symbol = (
                "✓" if metrics["intent_exact_match"]
                else ("~" if metrics["intent_partial_match"] else "✗")
            )
            pred_intents = [s["intent"] for s in predicted_parsed]
            print(f" {symbol}  {pred_intents}")

        except Exception as exc:
            result = {
                "prompt": prompt,
                "ground_truth": ground_truth_raw,
                "predicted_intent": "",
                "_gt_parsed": ground_truth_parsed,
                "_pred_parsed": [],
                "_metrics": None,
                "error": str(exc),
            }
            print(f" ✗  ERROR: {exc}")

        results.append(result)

    elapsed = time.perf_counter() - start

    # -- Write output JSONL — only the three user-facing fields ---------------
    with output_path.open("w") as f:
        for r in results:
            output_record = {
                "prompt": r["prompt"],
                "ground_truth": r["ground_truth"],
                "predicted_intent": r["predicted_intent"],
            }
            if r.get("error"):
                output_record["error"] = r["error"]
            f.write(json.dumps(output_record) + "\n")

    print(f"\nResults written  → {output_path}")

    # -- Statistics -----------------------------------------------------------
    stats = compute_statistics(results, elapsed, provider, model)
    print_statistics(stats)

    stats_path = output_path.with_suffix(".stats.json")
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Stats JSON       → {stats_path}")
    print()


if __name__ == "__main__":
    main()

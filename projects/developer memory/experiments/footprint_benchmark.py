"""
experiments/footprint_benchmark.py

Measures and compares the runtime footprint of gemma4:e4b vs gemma4:26b
inside a Docker-containerised Ollama environment (or native Ollama).

Metrics collected per model, per prompt tier:
  - Cold-start load time (seconds, first inference)
  - First-token latency (ms)
  - Inference throughput (tokens/sec) — from Ollama's own eval_duration
  - Ollama process RSS delta (MB) after model load
  - Docker container memory (MB) via `docker stats --no-stream` (if --container given)
  - Peak RSS during long-context inference

Usage:
    # Native Ollama (M4 Max recommended path):
    python experiments/footprint_benchmark.py

    # Against a containerised Ollama (uncomment ollama service in docker-compose.yml first):
    python experiments/footprint_benchmark.py \\
        --host http://localhost:11434 \\
        --container developer_memory_ollama

    # Benchmark only the quantized model:
    python experiments/footprint_benchmark.py --models gemma4:e4b

Output:
    experiments/results/footprint_<timestamp>.json
    experiments/results/footprint_<timestamp>.csv
"""

import argparse
import csv
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import psutil

# ── Constants ──────────────────────────────────────────────────────────────────

MODELS: list[str] = ["gemma4:e4b", "gemma4:26b"]

# Three prompt tiers stress short, medium, and long context windows.
PROMPTS: dict[str, str] = {
    "short": "What is a Python decorator? Answer in two sentences.",
    "medium": (
        "Explain the architectural tradeoffs between a monolithic FastAPI application "
        "and a microservices architecture orchestrated by LangGraph agents. "
        "Focus on latency, maintainability, and developer experience."
    ),
    "long": (
        "You are a principal software architect. Review the following design decision: "
        "We are building a developer memory system that uses ChromaDB for vector storage, "
        "Ollama for local LLM inference, and LangGraph for multi-agent orchestration. "
        "The system must satisfy: (1) >50 tok/s throughput on Apple M4 Max, "
        "(2) zero-trust local-only data sovereignty, "
        "(3) incremental git ingestion with SHA-256 idempotency, "
        "(4) PII masking with a quarantine review queue. "
        "Identify three architectural risks and propose specific mitigations, "
        "referencing design patterns (CQRS, saga, bulkhead, circuit breaker) where applicable. "
        "Then estimate the peak unified memory footprint in GB under full load."
    ),
}

NUM_WARM_RUNS: int = 2   # averaged warm-run count per prompt tier
RESULTS_DIR: Path = Path(__file__).parent / "results"


# ── Ollama HTTP API helpers ────────────────────────────────────────────────────

def pull_model_if_missing(host: str, model: str, timeout: float = 600.0) -> None:
    """Pull model if not already present in Ollama's local store."""
    tags = httpx.get(f"{host}/api/tags", timeout=10.0).json()
    present = {m["name"] for m in tags.get("models", [])}
    if model in present:
        print(f"  {model} already present — skipping pull.")
        return
    print(f"  Pulling {model} (this may take several minutes)...")
    with httpx.stream("POST", f"{host}/api/pull", json={"name": model}, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            chunk = json.loads(line) if line else {}
            status = chunk.get("status", "")
            if status:
                print(f"    {status}", end="\r", flush=True)
    print()


def unload_model(host: str, model: str) -> None:
    """Evict model from Ollama unified memory (keep_alive=0)."""
    httpx.post(
        f"{host}/api/generate",
        json={"model": model, "keep_alive": 0},
        timeout=30.0,
    )
    time.sleep(2)  # allow Ollama time to release memory before next benchmark


def run_inference(host: str, model: str, prompt: str, timeout: float = 180.0) -> dict:
    """
    Stream a single inference request and return timing + token stats.

    Returns:
        first_token_ms  : wall-clock ms from request send to first non-empty token
        tokens_per_sec  : Ollama's own eval_count / eval_duration (most accurate)
        eval_count      : total tokens generated (from Ollama final event)
        elapsed_s       : total wall-clock seconds for full response
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": 300},
    }
    first_token_ms: float | None = None
    eval_count: int = 0
    eval_duration_ns: int = 0
    start = time.perf_counter()

    with httpx.stream("POST", f"{host}/api/generate", json=payload, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if first_token_ms is None and chunk.get("response"):
                first_token_ms = (time.perf_counter() - start) * 1000
            if chunk.get("done"):
                eval_count = chunk.get("eval_count", 0)
                eval_duration_ns = chunk.get("eval_duration", 0)

    elapsed_s = time.perf_counter() - start
    tps = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0.0

    return {
        "first_token_ms": round(first_token_ms or 0.0, 1),
        "tokens_per_sec": round(tps, 2),
        "eval_count": eval_count,
        "elapsed_s": round(elapsed_s, 2),
    }


# ── Memory measurement helpers ─────────────────────────────────────────────────

def ollama_rss_mb() -> float:
    """Return RSS (MB) of the Ollama server process via psutil. 0.0 if not found."""
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            if "ollama" in (proc.info["name"] or "").lower():
                return proc.info["memory_info"].rss / (1024 ** 2)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return 0.0


def docker_container_mem_mb(container: str) -> float:
    """
    Return container memory usage in MB via `docker stats --no-stream`.
    Returns 0.0 if Docker is unavailable or the container is not running.
    """
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
            capture_output=True,
            text=True,
            timeout=10,
        )
        raw = result.stdout.strip().split("/")[0].strip()  # e.g. "4.2GiB" or "512MiB"
        if "GiB" in raw:
            return float(raw.replace("GiB", "")) * 1024
        if "MiB" in raw:
            return float(raw.replace("MiB", ""))
        if "GB" in raw:
            return float(raw.replace("GB", "")) * 1024
        if "MB" in raw:
            return float(raw.replace("MB", ""))
    except Exception:
        pass
    return 0.0


def sample_memory(container: str | None) -> dict:
    """Snapshot current memory from both sources."""
    return {
        "ollama_rss_mb": round(ollama_rss_mb(), 1),
        "docker_mb": round(docker_container_mem_mb(container), 1) if container else 0.0,
    }


# ── Per-model benchmark ────────────────────────────────────────────────────────

def benchmark_model(host: str, model: str, container: str | None) -> dict:
    """Run the full benchmark suite for one model. Returns structured result dict."""
    print(f"\n{'=' * 62}")
    print(f"  Benchmarking: {model}")
    print(f"{'=' * 62}")

    result: dict = {
        "model": model,
        "timestamp": datetime.now(UTC).isoformat(),
        "host": host,
        "container": container,
    }

    # ── Baseline memory before any model activity ──────────────────────────────
    mem_baseline = sample_memory(container)
    print(f"  Baseline RSS : {mem_baseline['ollama_rss_mb']:.0f} MB")

    # ── Cold start (model load + first inference) ──────────────────────────────
    print("  [cold] short prompt...", end=" ", flush=True)
    cold = run_inference(host, model, PROMPTS["short"])
    mem_loaded = sample_memory(container)
    cold["rss_delta_mb"] = round(mem_loaded["ollama_rss_mb"] - mem_baseline["ollama_rss_mb"], 1)
    cold["docker_delta_mb"] = round(mem_loaded["docker_mb"] - mem_baseline["docker_mb"], 1)
    print(f"{cold['tokens_per_sec']} tok/s  |  Δ RSS {cold['rss_delta_mb']:.0f} MB")
    result["cold_start"] = cold

    # ── Warm runs ──────────────────────────────────────────────────────────────
    for tier in ("medium", "long"):
        runs: list[dict] = []
        for i in range(NUM_WARM_RUNS):
            print(f"  [warm {i + 1}/{NUM_WARM_RUNS}] {tier} prompt...", end=" ", flush=True)
            run = run_inference(host, model, PROMPTS[tier])
            runs.append(run)
            print(f"{run['tokens_per_sec']} tok/s")

        # Average the warm runs
        avg: dict = {
            k: round(sum(r[k] for r in runs) / len(runs), 2)
            for k in ("first_token_ms", "tokens_per_sec", "eval_count", "elapsed_s")
        }
        result[f"{tier}_prompt"] = avg

    # ── Peak memory after long-context inference ───────────────────────────────
    mem_peak = sample_memory(container)
    result["memory"] = {
        "baseline_rss_mb": mem_baseline["ollama_rss_mb"],
        "loaded_rss_mb": mem_loaded["ollama_rss_mb"],
        "rss_delta_mb": cold["rss_delta_mb"],
        "peak_rss_mb": mem_peak["ollama_rss_mb"],
        "docker_loaded_mb": mem_loaded["docker_mb"],
        "docker_peak_mb": mem_peak["docker_mb"],
    }

    print(f"  Peak RSS     : {mem_peak['ollama_rss_mb']:.0f} MB")

    # ── Evict model to isolate next benchmark ──────────────────────────────────
    print(f"  Evicting {model} from memory...")
    unload_model(host, model)

    return result


# ── Output ────────────────────────────────────────────────────────────────────

def write_results(results: list[dict]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    json_path = RESULTS_DIR / f"footprint_{ts}.json"
    json_path.write_text(json.dumps(results, indent=2))

    csv_path = RESULTS_DIR / f"footprint_{ts}.csv"
    rows: list[dict] = []
    for r in results:
        base = {
            "model": r["model"],
            "timestamp": r["timestamp"],
            "host": r["host"],
            "container": r["container"] or "native",
        }
        for tier, key in [
            ("short_cold", "cold_start"),
            ("medium_warm", "medium_prompt"),
            ("long_warm", "long_prompt"),
        ]:
            row = {**base, "prompt_tier": tier}
            row.update(r.get(key, {}))
            row.update({f"mem_{k}": v for k, v in r.get("memory", {}).items()})
            rows.append(row)

    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return json_path, csv_path


def print_summary(results: list[dict]) -> None:
    header = f"\n{'Model':<24} {'RSS delta':>10} {'Peak RSS':>10} {'Short tok/s':>12} {'Long tok/s':>11} {'1st tok ms':>11}"
    print(header)
    print("─" * len(header.rstrip()))
    for r in results:
        mem = r.get("memory", {})
        print(
            f"{r['model']:<24}"
            f"{mem.get('rss_delta_mb', 0):>9.0f}MB"
            f"{mem.get('peak_rss_mb', 0):>9.0f}MB"
            f"{r.get('cold_start', {}).get('tokens_per_sec', 0):>12.1f}"
            f"{r.get('long_prompt', {}).get('tokens_per_sec', 0):>11.1f}"
            f"{r.get('cold_start', {}).get('first_token_ms', 0):>10.0f}ms"
        )

    # Interpretation note
    if len(results) == 2:
        a, b = results[0], results[1]
        tps_a = a.get("cold_start", {}).get("tokens_per_sec", 0)
        tps_b = b.get("cold_start", {}).get("tokens_per_sec", 0)
        rss_a = a.get("memory", {}).get("rss_delta_mb", 0)
        rss_b = b.get("memory", {}).get("rss_delta_mb", 0)
        faster = a["model"] if tps_a >= tps_b else b["model"]
        leaner = a["model"] if rss_a <= rss_b else b["model"]
        print(f"\n  Faster      : {faster} ({max(tps_a, tps_b):.1f} tok/s)")
        print(f"  Leaner      : {leaner} ({min(rss_a, rss_b):.0f} MB RSS delta)")
        print(
            "  Recommendation: if available memory < 16 GB, prefer the leaner model. "
            "See _check_memory() in server.py."
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure footprint of gemma4:e4b vs gemma4:26b in Docker or native Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--host", default="http://localhost:11434", help="Ollama base URL"
    )
    parser.add_argument(
        "--container", default=None,
        help="Docker container name for memory stats (e.g. developer_memory_ollama). "
             "Omit when benchmarking native Ollama.",
    )
    parser.add_argument(
        "--models", nargs="+", default=MODELS,
        help=f"Space-separated list of Ollama model tags (default: {' '.join(MODELS)})",
    )
    args = parser.parse_args()

    print("Developer Memory — LLM Footprint Benchmark")
    print(f"  Ollama host : {args.host}")
    print(f"  Container   : {args.container or '(native — psutil RSS)'}")
    print(f"  Models      : {args.models}")
    print(f"  Warm runs   : {NUM_WARM_RUNS} per prompt tier")

    # Verify Ollama is reachable before doing anything
    try:
        httpx.get(f"{args.host}/api/tags", timeout=5.0).raise_for_status()
    except Exception as exc:
        raise SystemExit(f"\nCannot reach Ollama at {args.host}.\n  {exc}") from exc

    # Pull any missing models upfront — isolates pull time from inference timing
    print("\nChecking models...")
    for model in args.models:
        pull_model_if_missing(args.host, model)

    # Run benchmarks sequentially — never concurrently (would pollute memory readings)
    results: list[dict] = []
    for model in args.models:
        results.append(benchmark_model(args.host, model, args.container))

    json_path, csv_path = write_results(results)
    print_summary(results)
    print(f"\n  JSON → {json_path}")
    print(f"  CSV  → {csv_path}")


if __name__ == "__main__":
    main()

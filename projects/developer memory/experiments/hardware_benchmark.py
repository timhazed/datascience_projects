"""
experiments/hardware_benchmark.py

Measures LLM inference performance across a configurable prompt suite and
records a full hardware profile alongside every metric. Purpose: produce
results that can be compared across machines — M4 Max (Metal), other Apple
Silicon, Intel Mac, Linux CPU, Linux + NVIDIA GPU — to quantify the real
inference gap attributable to the GPU backend.

Apple Silicon deployment context (see docs/AgenticArchitecture.md §11):
  - Hybrid (Mode A): native `ollama serve` on the Mac for full Metal/ANE;
    containerised apps call it via `http://host.docker.internal:11434`.
    Expect roughly 50+ tok/s on M4-class hardware with Gemma 4 26B when GPU-bound.
  - Docker Model Runner (Mode B): vllm-metal backend, zero-copy / unified-memory
    path inside the container — typically ~40–45 tok/s for the same model class
    (all-in-Docker tradeoff vs native Ollama).
  - Sanity checks: `ollama ps` should show the model using GPU on the host;
    set `OLLAMA_ORIGINS="*"` on the host so containers can call native Ollama.
    Sustained warm tok/s well below ~30 on an M4 Max with Gemma 4 usually means
    CPU fallback or a mis-pointed `--host`.

Metrics collected per model, per prompt:
  - First-token latency (ms)
  - Inference throughput (tok/s)     — from Ollama's eval_duration (most accurate)
  - Prompt eval throughput (tok/s)   — from Ollama's prompt_eval_duration
  - eval_count (tokens generated)
  - prompt_eval_count (tokens in prompt)
  - Total elapsed wall-clock time (s)

Hardware profile captured per run:
  - Platform, OS version, architecture (arm64 / x86_64)
  - CPU model string + physical / logical core counts
  - Apple Silicon chip generation (M1–M4, Pro/Max/Ultra) if applicable
  - System / Unified Memory (GB)
  - Inferred GPU backend: Metal | CUDA | CPU
  - Ollama version

Usage:
    # Run with built-in prompt suite against local Ollama:
    python experiments/hardware_benchmark.py

    # Specify model:
    python experiments/hardware_benchmark.py --model gemma4:26b

    # Load custom prompts from a JSON file (list of strings):
    python experiments/hardware_benchmark.py --prompts experiments/prompts.json

    # Compare two result files side-by-side:
    python experiments/hardware_benchmark.py --compare \\
        experiments/results/hardware_machineA.json \\
        experiments/results/hardware_machineB.json

    # Mode A (hybrid): native Ollama on localhost:11434 (default host for --deployment-mode hybrid)
    python experiments/hardware_benchmark.py --deployment-mode hybrid

    # Mode B (model-runner): Docker Model Runner / vllm-metal published on host (default :12434)
    python experiments/hardware_benchmark.py --deployment-mode model-runner

    # Run hybrid + model-runner in one session (each against its default or env-configured URL)
    python experiments/hardware_benchmark.py --all-deployment-modes

    # Override default URLs (see DEFAULT_HOST_BY_MODE / env keys in source)
    export HARDWARE_BENCHMARK_HOST_MODEL_RUNNER=http://localhost:12434

Output:
    experiments/results/hardware_<timestamp>.json
    experiments/results/hardware_<timestamp>.csv
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import psutil

# ── Default model ──────────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemma4:26b"
MODEL_MAPPING = {
    "gemma4:e4b": "huggingface.co/lmstudio-community/gemma-3n-e4b-it-mlx-4bit", # Updated
    "gemma4:26b": "huggingface.co/cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
}
NUM_WARM_RUNS = 3   # averaged per prompt; first run is cold-start, remainder are warm
RESULTS_DIR = Path(__file__).parent / "results"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLLAMA_BOOTSTRAP_SCRIPT = REPO_ROOT / "scripts" / "run_ollama.zsh"

# Deployment modes (§11 of AgenticArchitecture.md)
DEPLOYMENT_MODES = {
    "hybrid":       "Mode A — Native Ollama + Containerised App Stack (~50–100 tok/s)",
    "model-runner": "Mode B — Docker Model Runner / vllm-metal (~40–45 tok/s)",
}

# Default Ollama HTTP base URL per mode when --host is omitted.
# Map host ports to your compose file: e.g. publish Docker Model Runner on :12434.
DEFAULT_HOST_BY_MODE: dict[str, str] = {
    "hybrid": os.environ.get(
        "HARDWARE_BENCHMARK_HOST_HYBRID", "http://localhost:11434"
    ),
    "model-runner": os.environ.get(
        "HARDWARE_BENCHMARK_HOST_MODEL_RUNNER", "http://localhost:12434"
    ),
}


def resolve_ollama_host(deployment_mode: str, host_override: str | None) -> str:
    """Return the Ollama base URL for this mode; strip trailing slashes."""
    base = (host_override or DEFAULT_HOST_BY_MODE[deployment_mode]).strip()
    return base.rstrip("/")


def verify_ollama_reachable(host: str) -> None:
    httpx.get(f"{host}/api/tags", timeout=5.0).raise_for_status()


def start_ollama_via_script(script_path: str, mode: str, model: str) -> None:
    """Start Ollama for a mode/model by delegating to scripts/run_ollama.zsh."""
    print(f"\nStarting Ollama via script: {script_path} {mode} {model}")
    try:
        result = subprocess.run(
            ["/bin/zsh", script_path, mode, model],
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        raise SystemExit(
            f"\nFailed to start Ollama via script for deployment-mode={mode}.\n"
            f"  Script: {script_path}\n"
            f"  Args  : {mode} {model}\n"
            f"  Detail: {detail}"
        ) from exc

    output = (result.stdout or "").strip()
    if output:
        print(output)

# ── Prompt suite ───────────────────────────────────────────────────────────────
# Eight prompts designed to stress different latency and throughput regimes:
#   tier 1–2: tokenisation + first-token latency dominates
#   tier 3–5: throughput at moderate output length
#   tier 6–8: long output + reasoning — saturates memory bandwidth

DEFAULT_PROMPTS: list[dict] = [
    {
        "id": "p1_ultra_short",
        "label": "Ultra-short (1 token out)",
        "text": "Reply with only the word: yes",
    },
    {
        "id": "p2_short_factual",
        "label": "Short factual (2 sentences)",
        "text": "What is a Python decorator? Answer in two sentences.",
    },
    {
        "id": "p3_structured",
        "label": "Structured output (JSON)",
        "text": (
            "Return a JSON object with keys: language, paradigm, typing_discipline, "
            "year_created — for Python. No prose, just the JSON."
        ),
    },
    {
        "id": "p4_code_short",
        "label": "Short code generation",
        "text": (
            "Write a Python function that takes a list of integers and returns "
            "the two largest values without using sort(). Include type annotations."
        ),
    },
    {
        "id": "p5_reasoning",
        "label": "Multi-step reasoning",
        "text": (
            "A repository has 3 microservices. Service A calls Service B, which calls "
            "Service C. Service C has a 200ms P99 latency. Service B adds 50ms. "
            "What is the P99 end-to-end latency observed by a caller of Service A? "
            "Show your reasoning step by step."
        ),
    },
    {
        "id": "p6_code_long",
        "label": "Long code generation",
        "text": (
            "Implement a Python class `RateLimiter` using the token bucket algorithm. "
            "It must be thread-safe, support burst capacity, and expose an async "
            "`acquire(tokens: int = 1) -> float` method that returns the wait time. "
            "Include docstrings and type annotations."
        ),
    },
    {
        "id": "p7_architecture",
        "label": "Architecture analysis (long context in)",
        "text": (
            "You are a principal software architect. Review this design: "
            "A developer memory system uses ChromaDB for vector storage, "
            "Ollama for local LLM inference, and LangGraph for multi-agent orchestration. "
            "Requirements: (1) >50 tok/s throughput on Apple M4 Max, "
            "(2) zero-trust local-only data sovereignty, "
            "(3) incremental git ingestion with SHA-256 idempotency, "
            "(4) PII masking with quarantine review queue. "
            "Identify three architectural risks. For each: name the risk, describe the "
            "failure mode, and propose a specific mitigation referencing a design pattern."
        ),
    },
    {
        "id": "p8_synthesis",
        "label": "Long synthesis (max output)",
        "text": (
            "Write a detailed technical blog post explaining how Apple Silicon's Unified "
            "Memory Architecture eliminates the CPU-GPU memory copy bottleneck for LLM "
            "inference. Cover: traditional discrete GPU memory transfers, how UMA works, "
            "why memory bandwidth (not FLOPS) is the binding constraint for transformer "
            "inference, and quantitative estimates for a 26B parameter model. "
            "Target audience: senior software engineers. Minimum 600 words."
        ),
    },
]


# ── Hardware profile ───────────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: float = 5.0) -> str:
    """Run a shell command; return stdout stripped, or '' on any failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _apple_silicon_chip() -> str:
    """
    Return the Apple Silicon chip identifier (e.g. 'Apple M4 Max') or '' on
    non-Apple-Silicon hardware. Parses `sysctl hw.model` which returns strings
    like 'Mac16,11' — map via `system_profiler` for the marketing name.
    """
    # system_profiler is the most reliable source of the marketing chip name
    raw = _run(["system_profiler", "SPHardwareDataType"], timeout=10.0)
    for line in raw.splitlines():
        if "Chip" in line or "Processor Name" in line:
            return line.split(":", 1)[-1].strip()
    return ""


def _gpu_info() -> str:
    """
    Return a brief GPU description:
      - macOS arm64 → 'Metal (Apple GPU)' + chip name
      - macOS x86_64 → 'Metal (Intel/AMD)'
      - Linux with nvidia-smi → 'CUDA: <gpu name>'
      - fallback → 'CPU only'
    """
    system = platform.system()
    arch = platform.machine()

    if system == "Darwin":
        chip = _apple_silicon_chip()
        if arch == "arm64":
            return f"Metal — {chip}" if chip else "Metal (Apple Silicon)"
        # Intel Mac: Metal but not ANE
        gpu_raw = _run(["system_profiler", "SPDisplaysDataType"], timeout=10.0)
        for line in gpu_raw.splitlines():
            if "Chipset Model" in line:
                return f"Metal (Intel/AMD) — {line.split(':', 1)[-1].strip()}"
        return "Metal (Intel/AMD)"

    if system == "Linux":
        nvidia = _run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
        )
        if nvidia:
            return f"CUDA — {nvidia.splitlines()[0]}"
        # AMD ROCm
        rocm = _run(["rocm-smi", "--showproductname"])
        if rocm:
            return f"ROCm — {rocm.splitlines()[0]}"

    return "CPU only"


def _ollama_version(host: str) -> str:
    """Fetch Ollama version string from /api/version."""
    try:
        data = httpx.get(f"{host}/api/version", timeout=5.0).json()
        return data.get("version", "unknown")
    except Exception:
        return "unknown"


def collect_hardware_profile(host: str) -> dict:
    """Return a dict describing the current machine's hardware and inference backend."""
    vm = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()

    profile = {
        "hostname": platform.node(),
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_model": platform.processor() or _run(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": round(cpu_freq.max, 0) if cpu_freq else None,
        "ram_gb": round(vm.total / (1024 ** 3), 1),
        "ram_available_gb": round(vm.available / (1024 ** 3), 1),
        "gpu_backend": _gpu_info(),
        "apple_chip": _apple_silicon_chip() if platform.system() == "Darwin" else "",
        "ollama_version": _ollama_version(host),
    }

    # Memory bandwidth estimate (informational — actual measured in benchmark)
    # M4 Max: ~546 GB/s; M4 Pro: ~273 GB/s; M4: ~120 GB/s; Intel Mac: ~50 GB/s
    chip = profile["apple_chip"].upper()
    if "M4 MAX" in chip or "M4MAX" in chip:
        profile["est_memory_bandwidth_gbs"] = 546
    elif "M4 PRO" in chip or "M4PRO" in chip:
        profile["est_memory_bandwidth_gbs"] = 273
    elif "M4 ULTRA" in chip or "M4ULTRA" in chip:
        profile["est_memory_bandwidth_gbs"] = 819
    elif "M4" in chip:
        profile["est_memory_bandwidth_gbs"] = 120
    elif "M3 MAX" in chip:
        profile["est_memory_bandwidth_gbs"] = 400
    elif "M3 PRO" in chip:
        profile["est_memory_bandwidth_gbs"] = 150
    elif "M3" in chip:
        profile["est_memory_bandwidth_gbs"] = 100
    elif "M2 MAX" in chip:
        profile["est_memory_bandwidth_gbs"] = 400
    elif "M2 PRO" in chip:
        profile["est_memory_bandwidth_gbs"] = 200
    elif "M2 ULTRA" in chip:
        profile["est_memory_bandwidth_gbs"] = 800
    elif "M2" in chip:
        profile["est_memory_bandwidth_gbs"] = 100
    elif "M1 MAX" in chip:
        profile["est_memory_bandwidth_gbs"] = 400
    elif "M1 PRO" in chip:
        profile["est_memory_bandwidth_gbs"] = 200
    elif "M1 ULTRA" in chip:
        profile["est_memory_bandwidth_gbs"] = 800
    elif "M1" in chip:
        profile["est_memory_bandwidth_gbs"] = 68
    else:
        profile["est_memory_bandwidth_gbs"] = None   # unknown / non-Apple

    return profile


# ── Ollama inference helpers ───────────────────────────────────────────────────

def pull_model_if_missing(host: str, model: str, timeout: float = 600.0) -> None:
    """Pull model if not already present in Ollama's local store."""
    if ":12434" in host:
        print(f"  Ensuring {model} is pulled in Docker...")
        subprocess.run(["docker", "model", "pull", model], check=True)
        return

    tags = httpx.get(f"{host}/api/tags", timeout=10.0).json()
    present = {m["name"] for m in tags.get("models", [])}
    if model in present:
        print(f"  {model} already present — skipping pull.")
        return
    print(f"  Pulling {model}...")
    with httpx.stream("POST", f"{host}/api/pull", json={"name": model}, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            chunk = json.loads(line) if line else {}
            status = chunk.get("status", "")
            if status:
                print(f"    {status}", end="\r", flush=True)
    print()


def run_inference(host: str, model: str, prompt: str, timeout: float = 300.0) -> dict:
    """
    Stream one inference request; return timing and token stats.

    Returns:
        first_token_ms       : wall-clock ms to first non-empty token
        tokens_per_sec       : eval_count / eval_duration (Ollama native metric)
        prompt_tokens_per_sec: prompt_eval_count / prompt_eval_duration
        eval_count           : tokens generated
        prompt_eval_count    : tokens in prompt
        elapsed_s            : total wall-clock seconds
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": 1024},
    }
    first_token_ms: float | None = None
    eval_count = 0
    eval_duration_ns = 0
    prompt_eval_count = 0
    prompt_eval_duration_ns = 0
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
                prompt_eval_count = chunk.get("prompt_eval_count", 0)
                prompt_eval_duration_ns = chunk.get("prompt_eval_duration", 0)

    elapsed_s = time.perf_counter() - start
    tps = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0.0
    ptps = (
        (prompt_eval_count / (prompt_eval_duration_ns / 1e9))
        if prompt_eval_duration_ns > 0
        else 0.0
    )

    return {
        "first_token_ms": round(first_token_ms or 0.0, 1),
        "tokens_per_sec": round(tps, 2),
        "prompt_tokens_per_sec": round(ptps, 2),
        "eval_count": eval_count,
        "prompt_eval_count": prompt_eval_count,
        "elapsed_s": round(elapsed_s, 2),
    }


def unload_model(host: str, model: str) -> None:
    """Evict model from Ollama memory to get a clean cold-start for the next model."""
    httpx.post(
        f"{host}/api/generate",
        json={"model": model, "keep_alive": 0},
        timeout=30.0,
    )
    time.sleep(2)


# ── Benchmark runner ───────────────────────────────────────────────────────────

def benchmark_model(
    host: str,
    model: str,
    prompts: list[dict],
    warm_runs: int,
) -> dict:
    """Run the full prompt suite for one model. Returns structured result dict."""
    print(f"\n{'=' * 68}")
    print(f"  Model: {model}")
    print(f"{'=' * 68}")

    prompt_results: list[dict] = []

    for prompt in prompts:
        pid = prompt["id"]
        label = prompt["label"]
        text = prompt["text"]

        runs: list[dict] = []
        for i in range(warm_runs):
            tag = "cold" if i == 0 else f"warm {i}/{warm_runs - 1}"
            print(f"  [{pid}] {label} — {tag}...", end=" ", flush=True)
            run = run_inference(host, model, text)
            runs.append(run)
            print(f"{run['tokens_per_sec']} tok/s  |  1st token {run['first_token_ms']:.0f}ms")

        # Cold start = run[0]; warm average = mean of run[1:]
        cold = runs[0]
        warm_rows = runs[1:] if len(runs) > 1 else runs
        warm_avg = {
            k: round(sum(r[k] for r in warm_rows) / len(warm_rows), 2) for k in cold
        }

        prompt_results.append({
            "prompt_id": pid,
            "prompt_label": label,
            "prompt_chars": len(text),
            "cold": cold,
            "warm_avg": warm_avg,
        })

    return {
        "model": model,
        "timestamp": datetime.now(UTC).isoformat(),
        "host": host,
        "warm_runs": warm_runs,
        "prompts": prompt_results,
        "deployment_mode": None,   # populated by main() after args are parsed
    }


# ── Output ─────────────────────────────────────────────────────────────────────

def write_results(hw: dict, results: list[dict]) -> tuple[Path, Path]:
    """Write JSON + CSV to experiments/results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    payload = {"hardware": hw, "results": results}
    json_path = RESULTS_DIR / f"hardware_{ts}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    csv_path = RESULTS_DIR / f"hardware_{ts}.csv"
    rows: list[dict] = []
    for r in results:
        base = {
            "hostname": hw["hostname"],
            "apple_chip": hw["apple_chip"] or hw["cpu_model"],
            "gpu_backend": hw["gpu_backend"],
            "deployment_mode": r.get("deployment_mode") or "unspecified",
            "benchmark_host": r.get("host", ""),
            "ram_gb": hw["ram_gb"],
            "est_bw_gbs": hw.get("est_memory_bandwidth_gbs", ""),
            "model": r["model"],
            "timestamp": r["timestamp"],
        }
        for p in r["prompts"]:
            row = {
                **base,
                "prompt_id": p["prompt_id"],
                "prompt_label": p["prompt_label"],
                "prompt_chars": p["prompt_chars"],
                "cold_first_token_ms": p["cold"]["first_token_ms"],
                "cold_tokens_per_sec": p["cold"]["tokens_per_sec"],
                "cold_eval_count": p["cold"]["eval_count"],
                "cold_elapsed_s": p["cold"]["elapsed_s"],
                "warm_first_token_ms": p["warm_avg"]["first_token_ms"],
                "warm_tokens_per_sec": p["warm_avg"]["tokens_per_sec"],
                "warm_prompt_tokens_per_sec": p["warm_avg"]["prompt_tokens_per_sec"],
                "warm_eval_count": p["warm_avg"]["eval_count"],
                "warm_elapsed_s": p["warm_avg"]["elapsed_s"],
            }
            rows.append(row)

    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return json_path, csv_path


def print_summary(hw: dict, results: list[dict]) -> None:
    """Print hardware profile + per-prompt throughput table."""
    print(f"\n{'─' * 68}")
    print("  Hardware Profile")
    print(f"{'─' * 68}")
    print(f"  Host         : {hw['hostname']}")
    print(f"  Chip         : {hw['apple_chip'] or hw['cpu_model']}")
    print(f"  GPU backend  : {hw['gpu_backend']}")
    print(f"  RAM          : {hw['ram_gb']} GB ({hw['ram_available_gb']} GB available)")
    if hw.get("est_memory_bandwidth_gbs"):
        print(f"  Est. BW      : {hw['est_memory_bandwidth_gbs']} GB/s (theoretical peak)")
    print(f"  Ollama       : {hw['ollama_version']}")

    for r in results:
        mode = r.get("deployment_mode") or "unspecified"
        mode_desc = DEPLOYMENT_MODES.get(mode, mode)
        print(f"\n{'─' * 68}")
        print(f"  Model          : {r['model']}")
        print(f"  Deployment mode: {mode_desc}")
        print(f"  Runs/prompt    : {r['warm_runs']} (run[0]=cold, rest=warm)")
        header = f"  {'Prompt':<32} {'Cold tok/s':>10} {'Warm tok/s':>10} {'1st tok ms':>11} {'Eval tok':>9}"
        print(header)
        print(f"  {'─' * 72}")
        for p in r["prompts"]:
            print(
                f"  {p['prompt_label']:<32}"
                f"{p['cold']['tokens_per_sec']:>10.1f}"
                f"{p['warm_avg']['tokens_per_sec']:>10.1f}"
                f"{p['warm_avg']['first_token_ms']:>10.0f}ms"
                f"{p['warm_avg']['eval_count']:>9}"
            )

        warm_tps = [p["warm_avg"]["tokens_per_sec"] for p in r["prompts"]]
        print(f"\n  Warm avg tok/s across all prompts : {sum(warm_tps) / len(warm_tps):.1f}")
        print(f"  Warm peak tok/s                   : {max(warm_tps):.1f}")
        print(f"  Warm min tok/s                    : {min(warm_tps):.1f}")


def compare_results(paths: list[str]) -> None:
    """
    Load two or more result JSON files and print a side-by-side comparison.
    Useful for comparing M4 Max vs another machine without re-running.
    """
    loaded = []
    for p in paths:
        data = json.loads(Path(p).read_text())
        hw = data["hardware"]
        label = hw.get("apple_chip") or hw.get("cpu_model") or hw["hostname"]
        backend = hw["gpu_backend"]
        loaded.append({"label": f"{label} [{backend}]", "hw": hw, "results": data["results"]})

    # Collect all prompt IDs from the first file to drive the table
    prompt_ids = [p["prompt_id"] for p in loaded[0]["results"][0]["prompts"]]
    prompt_labels = {
        p["prompt_id"]: p["prompt_label"]
        for p in loaded[0]["results"][0]["prompts"]
    }

    print(f"\n{'═' * 80}")
    print("  Hardware Comparison — Warm tok/s per prompt")
    print(f"{'═' * 80}")

    # Header
    col_w = 16
    header = f"  {'Prompt':<30}" + "".join(f"{m['label'][:col_w]:>{col_w}}" for m in loaded)
    print(header)
    print(f"  {'─' * (30 + col_w * len(loaded))}")

    for pid in prompt_ids:
        label = prompt_labels.get(pid, pid)[:28]
        row = f"  {label:<30}"
        for m in loaded:
            # Find matching prompt in this result file
            tps = next(
                (
                    p["warm_avg"]["tokens_per_sec"]
                    for p in m["results"][0]["prompts"]
                    if p["prompt_id"] == pid
                ),
                None,
            )
            row += f"{tps:>{col_w}.1f}" if tps is not None else f"{'N/A':>{col_w}}"
        print(row)

    # Speedup row (first vs each subsequent)
    if len(loaded) >= 2:
        print(f"\n  Speedup ({loaded[0]['label'][:20]} / others):")
        for other in loaded[1:]:
            speedups = []
            for pid in prompt_ids:
                tps_a = next(
                    (p["warm_avg"]["tokens_per_sec"]
                     for p in loaded[0]["results"][0]["prompts"]
                     if p["prompt_id"] == pid), 0
                )
                tps_b = next(
                    (p["warm_avg"]["tokens_per_sec"]
                     for p in other["results"][0]["prompts"]
                     if p["prompt_id"] == pid), 0
                )
                if tps_b > 0:
                    speedups.append(tps_a / tps_b)
            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                print(f"    vs {other['label'][:40]}: {avg_speedup:.1f}× avg speedup")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure LLM inference performance with full hardware profiling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--host",
        default=None,
        metavar="URL",
        help=(
            "Ollama base URL. If omitted, the URL is chosen from --deployment-mode "
            "(hybrid → localhost:11434, model-runner → localhost:12434) or "
            "HARDWARE_BENCHMARK_HOST_HYBRID / HARDWARE_BENCHMARK_HOST_MODEL_RUNNER."
        ),
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model tag to benchmark (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--prompts", default=None,
        help="Path to a JSON file containing a list of prompt objects "
             '(each with "id", "label", "text" keys). '
             "Omit to use the built-in 8-prompt suite.",
    )
    parser.add_argument(
        "--warm-runs", type=int, default=NUM_WARM_RUNS,
        help=f"Total runs per prompt (run[0]=cold, rest averaged as warm). "
             f"Default: {NUM_WARM_RUNS}",
    )
    parser.add_argument(
        "--deployment-mode",
        choices=list(DEPLOYMENT_MODES.keys()),
        default="hybrid",
        help=(
            "Which deployment topology to benchmark: selects the default Ollama URL "
            "unless --host is set. Choices: "
            + " | ".join(f"{k}: {v}" for k, v in DEPLOYMENT_MODES.items())
        ),
    )
    parser.add_argument(
        "--all-deployment-modes",
        action="store_true",
        help=(
            "Run the full benchmark once per mode (hybrid, then model-runner), each "
            "against its default or env-configured host. Cannot be used with --host."
        ),
    )
    parser.add_argument(
        "--ollama-script-path",
        default=str(DEFAULT_OLLAMA_BOOTSTRAP_SCRIPT),
        help=(
            "Path to the Ollama bootstrap script that is run before each benchmark mode. "
            f"Default: {DEFAULT_OLLAMA_BOOTSTRAP_SCRIPT}"
        ),
    )
    parser.add_argument(
        "--compare", nargs="+", metavar="JSON_FILE",
        help="Compare two or more existing result JSON files side-by-side. "
             "No inference is run when this flag is provided.",
    )
    args = parser.parse_args()

    # ── Compare mode — no inference needed ────────────────────────────────────
    if args.compare:
        compare_results(args.compare)
        return

    if args.all_deployment_modes and args.host is not None:
        raise SystemExit(
            "Cannot use --host with --all-deployment-modes "
            "(each mode targets its own default URL; override with "
            "HARDWARE_BENCHMARK_HOST_HYBRID / HARDWARE_BENCHMARK_HOST_MODEL_RUNNER)."
        )

    modes_to_run = (
        list(DEPLOYMENT_MODES.keys()) if args.all_deployment_modes else [args.deployment_mode]
    )

    # ── Load prompt suite ──────────────────────────────────────────────────────
    if args.prompts:
        prompts = json.loads(Path(args.prompts).read_text())
        print(f"  Loaded {len(prompts)} prompts from {args.prompts}")
    else:
        prompts = DEFAULT_PROMPTS
        print(f"  Using built-in {len(prompts)}-prompt suite")

    print("\nDeveloper Memory — Hardware Inference Benchmark")
    if len(modes_to_run) > 1:
        print(f"  Modes        : {', '.join(modes_to_run)} (--all-deployment-modes)")
    else:
        print(f"  Mode         : {DEPLOYMENT_MODES[modes_to_run[0]]}")
    print(f"  Model        : {args.model}")
    print(f"  Runs/prompt  : {args.warm_runs} (run[0]=cold)")

    results: list[dict] = []
    hw_shared: dict | None = None

    for mode in modes_to_run:
        start_ollama_via_script(args.ollama_script_path, mode, args.model)

        host = resolve_ollama_host(mode, args.host)
        mode_desc = DEPLOYMENT_MODES[mode]

        print(f"\n{'═' * 68}")
        print(f"  Deployment   : {mode_desc}")
        print(f"  Ollama URL   : {host}")

        try:
            verify_ollama_reachable(host)
        except Exception as exc:
            raise SystemExit(
                f"\nCannot reach Ollama at {host} (deployment-mode={mode}).\n"
                f"  Start the server for this mode or set the matching "
                f"HARDWARE_BENCHMARK_HOST_* env var.\n"
                f"  {exc}"
            ) from exc

        hw = collect_hardware_profile(host)
        if hw_shared is None:
            hw_shared = hw

        print(f"  Chip         : {hw['apple_chip'] or hw['cpu_model']}")
        print(f"  GPU backend  : {hw['gpu_backend']}")
        print(f"  RAM          : {hw['ram_gb']} GB")
        if hw.get("est_memory_bandwidth_gbs"):
            print(f"  Est. BW      : {hw['est_memory_bandwidth_gbs']} GB/s")
        print(f"  Ollama       : {hw['ollama_version']}")

        print("\nChecking model...")
        current_model = args.model
        if mode == "model-runner":
            current_model = MODEL_MAPPING.get(args.model, args.model)
            print(f"  Mapping {args.model} -> {current_model} for Docker Runner")

        pull_model_if_missing(host, current_model)

        result = benchmark_model(host, current_model, prompts, args.warm_runs)
        result["deployment_mode"] = mode
        results.append(result)

    assert hw_shared is not None

    # ── Write output ───────────────────────────────────────────────────────────
    json_path, csv_path = write_results(hw_shared, results)
    print_summary(hw_shared, results)
    print(f"\n  JSON → {json_path}")
    print(f"  CSV  → {csv_path}")
    print(
        "\n  Tip: run on another machine, then compare with:\n"
        "  python experiments/hardware_benchmark.py --compare "
        "<this_file>.json <other_file>.json"
    )


if __name__ == "__main__":
    main()

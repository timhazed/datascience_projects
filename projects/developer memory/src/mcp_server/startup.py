"""Startup utilities for the Developer Memory MCP server.

Owns: logging configuration, signal handlers, memory preflight.

This module is the root of the import DAG — it imports nothing from other
mcp_server modules. All other mcp_server modules may safely import from here
without risking circular imports.

Startup sequence (executed at module import time):
  1. logging.basicConfig() configured — must run before any other module touches logging.
  2. _install_signal_handlers() called — SIGTERM/SIGINT handlers installed exactly once.
"""

import logging
import os
import pathlib
import signal
import sys

import psutil

logging.basicConfig(
    level=logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    # Override any handler uvicorn may have installed before us.
    force=True,
)
# Presidio is noisy — suppress per-file "Fetching recognizers" INFO spam and the
# one-time startup WARNINGs about non-English recognizers not loaded in the en registry.
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
# Suppress uvicorn access log health-check spam — /health is polled every few seconds.
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _log_memory(label: str = "") -> None:
    """Log current process RSS and system available memory via psutil.

    Args:
        label: Optional label appended to the log line, e.g. "after PIIFilter init".
    """
    try:
        proc = psutil.Process()
        rss_mb = proc.memory_info().rss / 1024 / 1024
        vm = psutil.virtual_memory()
        avail_gb = vm.available / 1024 / 1024 / 1024
        used_pct = vm.percent
        logger.info(
            "MEM%s — process RSS %.0f MB | system available %.1f GB (%.0f%% used)",
            f" [{label}]" if label else "",
            rss_mb,
            avail_gb,
            used_pct,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("MEM — could not read memory stats: %s", exc)


def _install_signal_handlers() -> None:
    """Install SIGTERM/SIGINT handlers that log and exit cleanly.

    Called once at module import time. Logs the signal name so exits are
    visible in Docker logs and are distinguishable from crashes.
    """

    def _handle(signum: int, frame: object) -> None:
        sig_name = signal.Signals(signum).name
        _log_memory(f"at {sig_name}")
        logger.warning("SHUTDOWN — received %s, exiting cleanly", sig_name)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


# Install signal handlers once at import time.
_install_signal_handlers()


def _check_memory() -> str:
    """Abort or degrade gracefully if Unified Memory is critically low.

    Reads available system memory via psutil and applies two thresholds:
      - < 8 GB  → raises RuntimeError (cannot run safely; Ollama alone needs ~10 GB)
      - < 16 GB → halves PARSER_WORKERS, switches model to gemma4:e4b (quantized fallback)
      - ≥ 16 GB → returns the configured OLLAMA_MODEL unchanged

    On M4 Max, ChromaDB + Ollama (gemma4:26b) + Streamlit can consume ~20–28 GB at peak.
    If available memory drops below 16 GB, reduce PARSER_WORKERS to avoid swap thrashing
    during large-repo ingestion.

    When running inside Docker, psutil reports the container's memory limit (typically 6–8 GB),
    not the host's physical RAM. Ollama runs natively on the host, so the container process
    itself has a small footprint. The check is skipped inside containers — the host operator
    is responsible for ensuring adequate RAM before running `make up`.

    Returns:
        Model tag string: "gemma4:26b" (full model) or "gemma4:e4b" (quantized fallback).

    Raises:
        RuntimeError: If available memory is below 8 GB — only enforced outside Docker.
    """
    # /.dockerenv is created by the Docker runtime in every container.
    # Skip the RAM check here — the host has adequate memory; the container limit is irrelevant.
    if pathlib.Path("/.dockerenv").exists():
        configured_model = os.environ.get("OLLAMA_MODEL", "gemma4:26b")
        logger.info("Running inside Docker — skipping host RAM check; model=%s", configured_model)
        return configured_model

    available_gb = psutil.virtual_memory().available / (1024**3)

    if available_gb < 8:
        raise RuntimeError(
            f"Insufficient Unified Memory: {available_gb:.1f} GB available, 8 GB required. "
            "Close other applications before starting Developer Memory."
        )

    configured_model = os.environ.get("OLLAMA_MODEL", "gemma4:26b")

    # Models that are already low-footprint — no fallback needed even under memory pressure.
    _SMALL_MODELS = {"gemma4:e4b", "phi4-mini", "phi4-mini:latest", "gpt-oss:20b"}
    configured_base = configured_model.split(":")[0]

    if (
        available_gb < 16
        and configured_model not in _SMALL_MODELS
        and configured_base not in _SMALL_MODELS
    ):
        # Quantized fallback: 9.6 GB footprint vs ~20 GB for the full MoE model.
        # Halve PARSER_WORKERS to reduce peak memory during parallel chunk parsing.
        degraded_workers = max(2, int(os.environ.get("PARSER_WORKERS", "6")) // 2)
        os.environ["PARSER_WORKERS"] = str(degraded_workers)
        active_model = "gemma4:e4b"
        logger.warning(
            "Low memory (%.1f GB available) — PARSER_WORKERS reduced to %d; "
            "falling back to gemma4:e4b (configured model was %s)",
            available_gb,
            degraded_workers,
            configured_model,
        )
        return active_model

    return configured_model

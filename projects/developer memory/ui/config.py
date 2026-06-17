"""Developer Memory UI — server URL configuration and pipeline stage label constants.

Owns `_SERVER_BASE`, `_MCP_URL`, and `_STAGE_LABELS`. Root of the import DAG — no
imports from other `ui/` modules.
"""

import os

# ── Server URLs ───────────────────────────────────────────────────────────────

_SERVER_BASE: str = os.environ.get("MCP_SERVER_URL", "http://localhost:9000")
_MCP_URL: str = _SERVER_BASE + "/mcp"

# ── Stage labels ──────────────────────────────────────────────────────────────

# Maps server stage string → (display label, progress bar fraction).
# Fractions are approximate — summarizing dominates total wall time.
_STAGE_LABELS: dict[str, tuple[str, float]] = {
    "queued":          ("Waiting to start…", 0.0),
    "cloning":         ("Cloning repository & computing delta…", 0.05),
    "cache_filtering": ("Filtering cached files…", 0.10),
    "pii_scanning":    ("Scanning files for PII…", 0.15),
    "parsing":         ("Parsing files into chunks…", 0.27),
    "summarizing":     ("LLM Summarizing chunks…", 0.30),
    "done":            ("Complete", 1.0),
    "error":           ("Failed", 0.0),
}

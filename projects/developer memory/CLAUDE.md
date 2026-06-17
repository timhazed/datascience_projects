# Developer Memory — Claude Context

**Project:** Developer Memory MCP v2.5 — Sovereign Developer Memory & Coaching System  
**Stack:** Python 3.13, FastMCP, LangGraph 1.x, ChromaDB 0.5.20, Ollama (native macOS), Streamlit  
**Entry points:** `streamlit run ui/streamlit_app.py` | `make up` (Docker stack)

---

## Architecture in One Paragraph

Five LangGraph pipelines (sync, query, persona, diff, skills) run behind a FastMCP HTTP server at port 9000. The Streamlit UI at port 8501 talks to the MCP server only via HTTP — it never imports `src.*` directly. ChromaDB (`developer_memory_v1` collection) is the vector store; Ollama runs natively on macOS and is reached by Docker containers via `host.docker.internal`. All graphs are compiled once at startup as singletons in `src/mcp_server/container.py`.

---

## Directory Ownership

| Path | Owns |
|------|------|
| `src/agents/` | 20 node factories (`make_*_node`) |
| `src/chains/` | 5 LLM chain builders (`build_*_chain`) |
| `src/graphs/` | 5 StateGraph builders (`build_*_graph`) |
| `src/models/` | All Pydantic models and TypedDicts (state, chunk, analysis, persona, skills, quarantine, trace_event) |
| `src/db/` | `ChromaLibrarianClient`, `SHAStore` |
| `src/mcp_server/` | `server.py` (tools + routes), `container.py` (singletons), `sync_jobs.py` (job lifecycle), `startup.py` (preflight) |
| `src/middleware/` | `PIIFilter` (Presidio + spaCy), `validate_target_path` |
| `src/config/` | `Settings` (pydantic-settings, env-validated) |
| `src/llm/` | `get_llm()` factory (ChatOllama) |
| `src/utils/` | `git_utils`, `retry`, `parsers/` (pdf, markdown, python) |
| `ui/` | Streamlit UI only — no `src.*` imports except through HTTP |
| `tests/` | Mirrors `src/`; `tests/ui/`, `tests/contracts/` |
| `docs/` | Architecture specs — read before implementing anything |

---

## Non-Negotiable Rules

**Imports:** UI never imports `src.*`. `container.py` is the only place singletons are constructed.

**Patch targets:** Always patch the name in the **consuming** module's namespace, not the definition site. `server.py` imports `run_sync_job as _run_sync_job` — patch `src.mcp_server.server._run_sync_job`, not `src.mcp_server.sync_jobs.run_sync_job`.

**No live calls in tests:** No real HTTP, git clone, subprocess, thread, or filesystem writes outside `tmp_path`. Mock `time.sleep` in any test that would block.

**Coverage:** 85% floor (`--cov-fail-under=85`) on `src/` only. `ui/`, `src/cli/`, `experiments/` are omitted.

**Ruff:** `line-length=100`, `select=["E","F","I","UP","B","SIM"]`. Run `ruff check src/ tests/ ui/` before every commit.

**Graphs:** Recursion limit 10,000 (large repos). Every Send payload and every sequential node return dict must include `"trace_events": []` when emitting nothing.

**State fan-out:** `upsert_results`, `trace`, `trace_events` use `operator.add` reducers. `parsed_chunks` is SINGLE-WRITER (only `multimodal_parser` writes it).

---

## Key Invariants to Never Break

- `_sanitize_result()` in `server.py` recursively converts Pydantic models before JSON serialization — all MCP tool returns go through it.
- `sha_store_update` writes `commit_sha` **only** on clean completion (no errors, no upsert errors).
- `release_quarantine_item` uses get-merge-update — `collection.update()` replaces the entire metadata dict, so always `get()` first to preserve existing fields.
- `QuarantineItem.doc_id` comes from `request.path_params["doc_id"]` not the request body.
- PII confidence threshold: 0.8. QUARANTINE_SENTINEL = `"[CONTENT_QUARANTINED — PII review required]"`.
- Deviation alert threshold: `deviation_score > 0.65`.
- ChromaDB idempotency: same `content_hash` → `action="skipped"`, same SHA different path → `action="updated"`, new → `action="inserted"`.

---

## Docs to Read Before Any Implementation

- `docs/AgenticArchitecture.md` — full node/state/graph spec (primary reference)
- `docs/AgenticArchitectureRework.md` — F6/F2/F1 remediation specs (implement F6→F2→F1 in order)
- `docs/SyncPipelineImprovements.md` — cache_filter, SHA freshness, git timeout specs
- `docs/UIRefactor.md` — ui/ package structure and phase plan
- `docs/ServerRefactor.md` — mcp_server/ package decomposition spec

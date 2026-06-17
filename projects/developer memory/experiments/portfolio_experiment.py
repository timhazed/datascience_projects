"""
experiments/portfolio_experiment.py

Ingests source files from multiple projects into a local ChromaDB collection,
then runs semantic queries against the indexed content using Ollama for both
embedding and synthesis.

The pipeline is intentionally minimal — no LangChain, no LangGraph:
    file → intent summary (Ollama) → embed (Ollama) → ChromaDB upsert
    query → embed (Ollama) → ChromaDB search → synthesis (Ollama) → result

This lets us validate whether intent-summarised vector search actually answers
portfolio-level questions before committing to the full architecture.

Usage:
    # Ingest all three projects then run all queries:
    python experiments/portfolio_experiment.py --model gemma4:26b --all

    # Ingest only (idempotent — skips files already indexed):
    python experiments/portfolio_experiment.py --model gemma4:e4b --ingest

    # Query only (requires a prior ingest run):
    python experiments/portfolio_experiment.py --model gemma4:26b --query

    # Run a specific subset of queries:
    python experiments/portfolio_experiment.py --model gemma4:26b --query --queries pq001,pq012,pq020

    # Force re-ingest (discards existing index for this model):
    python experiments/portfolio_experiment.py --model gemma4:26b --ingest --force-reingest

    # Use a different embedding model:
    python experiments/portfolio_experiment.py --model gemma4:26b --all --embed-model mxbai-embed-large

Output:
    experiments/results/portfolio_<model_slug>_<timestamp>.json
"""

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import chromadb
import httpx

# ── Paths ──────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = _SCRIPT_DIR / "results"
INPUTS_DIR = _SCRIPT_DIR / "inputs"
CHROMA_DIR = RESULTS_DIR / "chroma_portfolio"
QUERIES_FILE = INPUTS_DIR / "portfolio_queries.jsonl"

# Projects to ingest — paths relative to the repo parent directory.
_PROJECTS_ROOT = _SCRIPT_DIR.parent.parent
PROJECTS: dict[str, Path] = {
    "agentic_healthcare_assistant": _PROJECTS_ROOT / "agentic healthcare assistant",
    "newsgenie": _PROJECTS_ROOT / "newsgenie",
    "exercise_and_health_coach": _PROJECTS_ROOT / "exercise_and_health_coach",
}

# ── File walk config ───────────────────────────────────────────────────────────

INCLUDE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".md", ".ts", ".js", ".yaml", ".yml", ".toml",
})
# Named non-dot directories to skip. Dot directories (any component starting with ".")
# are skipped automatically in _walk_project — no need to list them here.
SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", "venv", "node_modules", "dist", "build", "tmp",
})
MAX_FILE_BYTES = 50_000   # skip files larger than 50 KB
CHUNK_SIZE = 3_000        # characters per chunk — nomic-embed-text cap is 2048 tokens; at ~2 chars/token for dense code, 3000 chars ≈ 1500 tokens
CHUNK_OVERLAP = 150       # overlap between consecutive chunks

# ── Ollama config ──────────────────────────────────────────────────────────────

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"  # 2048-token context window; handles full 6000-char chunks

_NUM_PREDICT_SYNTHESIS: dict[str, int] = {
    "gemma4:e4b": 2048,
    "gemma4:26b": 2048,
}

_SYNTHESIS_PROMPT = """\
### SYSTEM ROLE
You are a Senior Software Architect specializing in Portfolio Synthesis. \
Your goal is to provide a cross-project evaluation based on provided technical summaries.

### CORE TASK
Analyze the {{query}} by synthesizing information across the retrieved projects. \
You must follow this multi-step reasoning process:
1. **Extraction:** Identify which projects contain data relevant to the query.
2. **Contrast:** Highlight differences in patterns, tech stacks, or quality between these projects.
3. **Synthesis:** Provide a ranked judgment or an integrated answer as requested.

### CONSTRAINTS
- **Groundedness:** Answer ONLY using the provided summaries. \
If information is missing, state: "Insufficient data in context."
- **Citations:** Every claim must be followed by the project name in brackets, \
e.g., [agentic_healthcare_assistant].
- **Format:** Use Markdown headers for different sections and a Summary Table \
if comparing multiple projects.

### CONTEXT
Retrieved Project Summaries:
{context}

---
### USER QUERY
{query}"""

# ── Ollama helpers ─────────────────────────────────────────────────────────────

def _check_embed_model(embed_model: str, client: httpx.Client) -> None:
    """Preflight: verify the embed model is available before starting any ingest/query work.

    Uses GET /api/tags to list pulled models. Fails fast with a clear message if the
    requested embed model is not present, rather than surfacing errors on every file.
    """
    r = client.get(f"{OLLAMA_HOST}/api/tags", timeout=10.0)
    r.raise_for_status()
    available = [m["name"] for m in r.json().get("models", [])]
    # Normalise: "nomic-embed-text" and "nomic-embed-text:latest" are the same model.
    def _base(name: str) -> str:
        return name.split(":")[0]
    if not any(_base(a) == _base(embed_model) for a in available):
        pulled = ", ".join(available) or "(none)"
        raise RuntimeError(
            f"\n  Embed model '{embed_model}' is not available in Ollama.\n"
            f"  Models currently pulled: {pulled}\n\n"
            f"  Pull a dedicated embedding model (generative models cannot be used for embeddings):\n"
            f"      ollama pull all-minilm          #  46 MB — fast, good for experimentation\n"
            f"      ollama pull nomic-embed-text    # 274 MB — better retrieval quality\n"
            f"      ollama pull mxbai-embed-large   # 670 MB — highest quality (used in prod spec)\n\n"
            f"  Then re-run with:  --embed-model all-minilm"
        )


def _ollama_embed(text: str, embed_model: str, client: httpx.Client) -> list[float]:
    """Return embedding vector via POST /api/embed.

    Spec: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    Request:  {"model": str, "input": str}
    Response: {"embeddings": [[float, ...]]}
    """
    r = client.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": embed_model, "input": text},
        timeout=30.0,
    )
    if not r.is_success:
        print(f"  [embed 400 detail] status={r.status_code} body={r.text[:300]!r}", file=sys.stderr)
        print(f"  [embed 400 input]  len={len(text)} first_100={text[:100]!r}", file=sys.stderr)
    r.raise_for_status()
    return r.json()["embeddings"][0]


def _ollama_chat(
    prompt: str,
    model: str,
    num_predict: int,
    client: httpx.Client,
    as_json: bool = False,
) -> str:
    """Call Ollama /api/chat (non-streaming) and return the message content string."""
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": num_predict,
        },
    }
    if as_json:
        payload["format"] = "json"
    r = client.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120.0)
    r.raise_for_status()
    return r.json()["message"]["content"]



# ── File chunking ──────────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks of CHUNK_SIZE characters."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _walk_project(name: str, root: Path) -> list[dict]:
    """Return a list of chunk dicts for all indexable files under root."""
    if not root.exists():
        print(f"  [warn] {name}: directory not found at {root}", file=sys.stderr)
        return []
    chunks: list[dict] = []
    for fpath in sorted(root.rglob("*")):
        if not fpath.is_file():
            continue
        # Skip named directories (e.g. __pycache__, node_modules) and any dot directory
        # (e.g. .venv, .git, .idea, .env) — check every component of the relative path.
        rel_parts = fpath.relative_to(root).parts
        if any(p in SKIP_DIRS or (p.startswith(".") and p not in (".", "..")) for p in rel_parts):
            continue
        if fpath.suffix not in INCLUDE_EXTENSIONS:
            continue
        if fpath.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if not content:
            continue
        rel = str(fpath.relative_to(root))
        for i, chunk in enumerate(_chunk_text(content)):
            chunks.append({"project": name, "path": rel, "chunk_index": i, "content": chunk})
    return chunks


# ── ChromaDB ───────────────────────────────────────────────────────────────────

def _get_collection(model: str) -> chromadb.Collection:
    """Return a persistent ChromaDB collection scoped to the generation model.

    Separate collections per model prevent cross-contamination between e4b and 26b
    intent summaries when comparing query quality.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9]", "_", model)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma.get_or_create_collection(
        name=f"portfolio_{slug}",
        metadata={"hnsw:space": "cosine"},
    )


# ── Ingest ─────────────────────────────────────────────────────────────────────

def run_ingest(model: str, embed_model: str, force: bool = False, limit: int | None = None) -> dict:
    """Ingest all project files into ChromaDB. Returns ingest stats dict.

    Args:
        limit: If set, only ingest the first N files across all projects (for quick smoke tests).
    """
    if force:
        # Delete the existing collection so dimension constraints from a prior embed model are cleared.
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9]", "_", model)
        chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col_name = f"portfolio_{slug}"
        existing = [c.name for c in chroma.list_collections()]
        if col_name in existing:
            chroma.delete_collection(col_name)
            print(f"[force-reingest] deleted existing collection '{col_name}'")
    collection = _get_collection(model)
    stats: dict = {"projects": {}, "total_chunks": 0, "skipped": 0, "errors": 0}
    t0 = time.monotonic()
    files_processed = 0

    with httpx.Client() as client:
        _check_embed_model(embed_model, client)
        for project_name, project_path in PROJECTS.items():
            if limit is not None and files_processed >= limit:
                break
            print(f"\n── {project_name} ──")
            # Group chunks by file path so we count files, not chunks
            all_chunks = _walk_project(project_name, project_path)
            # Apply file-level limit
            if limit is not None:
                file_paths: list[str] = []
                seen: set[str] = set()
                for c in all_chunks:
                    if c["path"] not in seen:
                        if files_processed + len(seen) >= limit:
                            break
                        seen.add(c["path"])
                    file_paths.append(c["path"])
                allowed = set(list(seen)[:limit - files_processed])
                all_chunks = [c for c in all_chunks if c["path"] in allowed]
                files_processed += len(allowed)

            ingested = skipped = errors = 0

            for chunk in all_chunks:
                content_hash = hashlib.sha256(chunk["content"].encode()).hexdigest()
                doc_id = f"{content_hash}_{chunk['chunk_index']}"

                if not force and collection.get(ids=[doc_id])["ids"]:
                    skipped += 1
                    continue

                # nomic-embed-text: 2048-token context. At ~4 chars/token, 6000-char chunks ≈ 1500 tokens — within limit.
                # Strip null bytes only; no truncation needed.
                embed_text = (
                    chunk["content"]
                    .replace("\x00", "")
                    .encode("utf-8", errors="ignore")
                    .decode("utf-8")
                )
                try:
                    embedding = _ollama_embed(embed_text, embed_model, client)
                except httpx.HTTPError as exc:
                    errors += 1
                    print(f"  [embed error] {chunk['path']}:{chunk['chunk_index']} — {exc}")
                    continue

                collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk["content"]],
                    metadatas=[{
                        "project": chunk["project"],
                        "path": chunk["path"],
                        "chunk_index": chunk["chunk_index"],
                        "indexed_at": datetime.now(UTC).isoformat(),
                    }],
                )
                ingested += 1
                print(f"  ✓  {chunk['path']}:{chunk['chunk_index']}")

            stats["projects"][project_name] = {
                "ingested": ingested, "skipped": skipped, "errors": errors,
            }
            stats["total_chunks"] += ingested
            stats["skipped"] += skipped
            stats["errors"] += errors
            print(f"  → {ingested} ingested, {skipped} skipped, {errors} errors")

    stats["duration_s"] = round(time.monotonic() - t0, 2)
    total = stats["total_chunks"]
    duration = stats["duration_s"]
    print(f"\n[ingest complete] {total} chunks indexed in {duration}s")
    return stats


# ── Query ──────────────────────────────────────────────────────────────────────

def run_query(
    model: str,
    embed_model: str,
    query_ids: list[str] | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Run portfolio queries against the indexed ChromaDB collection."""
    collection = _get_collection(model)
    if collection.count() == 0:
        print("[warn] collection is empty — run --ingest first", file=sys.stderr)
        return []

    num_predict = _NUM_PREDICT_SYNTHESIS.get(model, 1500)

    # Load query corpus
    if not QUERIES_FILE.exists():
        print(f"[error] queries file not found: {QUERIES_FILE}", file=sys.stderr)
        return []
    queries: list[dict] = []
    with QUERIES_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    if query_ids:
        id_set = set(query_ids)
        queries = [q for q in queries if q["id"] in id_set]
        missing = id_set - {q["id"] for q in queries}
        if missing:
            print(f"[warn] query IDs not found: {sorted(missing)}", file=sys.stderr)

    results: list[dict] = []
    effective_k = min(top_k, collection.count())

    with httpx.Client() as client:
        _check_embed_model(embed_model, client)
        for q in queries:
            print(f"\n── {q['id']} [{q['difficulty']}] ──")
            print(f"   {q['query']}")
            t0 = time.monotonic()

            # Embed query
            try:
                q_vec = _ollama_embed(q["query"], embed_model, client)
            except httpx.HTTPError as exc:
                print(f"  [embed error] {exc}", file=sys.stderr)
                results.append({**q, "retrieved_chunks": [], "answer": None, "error": str(exc), "latency_s": 0.0})
                continue

            # Retrieve from ChromaDB
            search = collection.query(
                query_embeddings=[q_vec],
                n_results=effective_k,
                include=["documents", "metadatas", "distances"],
            )
            chunks: list[dict] = []
            for doc, meta, dist in zip(
                search["documents"][0],
                search["metadatas"][0],
                search["distances"][0],
                strict=False,
            ):
                chunks.append({
                    "project": meta.get("project", ""),
                    "path": meta.get("path", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "content_preview": doc[:600],
                    "cosine_distance": round(dist, 4),
                })

            # Format context for synthesis prompt — raw content excerpt per chunk
            context_lines: list[str] = []
            for i, c in enumerate(chunks, 1):
                context_lines.append(
                    f"[{i}] Project: {c['project']} | File: {c['path']} "
                    f"(distance: {c['cosine_distance']})\n"
                    f"{c['content_preview']}"
                )
            context = "\n\n".join(context_lines)

            # Synthesize answer
            synthesis_prompt = _SYNTHESIS_PROMPT.format(query=q["query"], context=context)
            try:
                answer = _ollama_chat(synthesis_prompt, model, num_predict, client, as_json=False)
            except httpx.HTTPError as exc:
                answer = f"ERROR: {exc}"
                print(f"  [synthesis error] {exc}", file=sys.stderr)

            latency = round(time.monotonic() - t0, 2)
            print(f"  {len(chunks)} chunks retrieved | {latency}s")
            # Print first 250 chars of answer as preview
            preview = answer[:250].replace("\n", " ")
            print(f"  ↳ {preview}{'…' if len(answer) > 250 else ''}")

            results.append({
                "id": q["id"],
                "query": q["query"],
                "category": q["category"],
                "difficulty": q["difficulty"],
                "expected_projects": q.get("expected_projects", []),
                "notes": q.get("notes", ""),
                "retrieved_chunks": chunks,
                "answer": answer,
                "latency_s": latency,
            })

    return results


# ── Results ────────────────────────────────────────────────────────────────────

def save_results(
    model: str,
    embed_model: str,
    ingest_stats: dict | None,
    query_results: list[dict],
) -> Path:
    """Write run results to experiments/results/portfolio_<slug>_<timestamp>.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9]", "_", model)
    out = RESULTS_DIR / f"portfolio_{slug}_{ts}.json"
    payload = {
        "run_id": f"portfolio_{slug}_{ts}",
        "model": model,
        "embed_model": embed_model,
        "timestamp": datetime.now(UTC).isoformat(),
        "projects": list(PROJECTS.keys()),
        "ingest_stats": ingest_stats,
        "query_results": query_results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n[saved] {out}")
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Portfolio experiment: ingest real project source files into ChromaDB "
            "and run semantic portfolio queries via Ollama."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default="gemma4:26b",
        choices=["gemma4:e4b", "gemma4:26b"],
        help="Ollama model used for intent summarization and query synthesis (default: gemma4:26b)",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        metavar="MODEL",
        help=f"Ollama embedding model (default: {DEFAULT_EMBED_MODEL}). "
             "Must be pulled: ollama pull <model>",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--ingest",
        action="store_true",
        help="Walk project directories and ingest intent summaries into ChromaDB",
    )
    mode.add_argument(
        "--query",
        action="store_true",
        help="Run portfolio queries against previously indexed ChromaDB data",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Run ingest then query in sequence",
    )

    parser.add_argument(
        "--queries",
        default=None,
        metavar="IDS",
        help="Comma-separated query IDs to run, e.g. --queries pq001,pq012 "
             "(default: all 20 queries)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="K",
        help="Number of ChromaDB chunks to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--force-reingest",
        action="store_true",
        help="Delete the existing collection and re-ingest all files from scratch",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit ingest to first N files (for smoke testing — e.g. --limit 5)",
    )

    args = parser.parse_args()
    query_ids = [q.strip() for q in args.queries.split(",")] if args.queries else None

    ingest_stats: dict | None = None
    query_results: list[dict] = []

    if args.ingest or args.all:
        ingest_stats = run_ingest(
            model=args.model,
            embed_model=args.embed_model,
            force=args.force_reingest,
            limit=args.limit,
        )

    if args.query or args.all:
        query_results = run_query(
            model=args.model,
            embed_model=args.embed_model,
            query_ids=query_ids,
            top_k=args.top_k,
        )

    if ingest_stats is not None or query_results:
        save_results(args.model, args.embed_model, ingest_stats, query_results)


if __name__ == "__main__":
    main()

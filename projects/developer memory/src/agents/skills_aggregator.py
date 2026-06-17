"""skills_aggregator LangGraph node — Spec §3b, §10 Phase 2.

Queries ChromaDB with three targeted semantic_type filters to gather concrete structural
evidence (file paths, class/function names, module layout) for skills_synthesizer.

Replaces the single broad query with Logic/Config/Interface queries and source-type-first
ranking so Python/notebook chunks dominate over README/markdown chunks.
"""

import json
import logging
from collections.abc import Callable

from src.db.chroma_client import ChromaLibrarianClient
from src.models.skills import SkillsData
from src.utils.chroma_aggregation import aggregate_chroma_docs

logger = logging.getLogger(__name__)

# ── Caps ──────────────────────────────────────────────────────────────────────
# All caps are module-level constants so tests can assert against them by name.
# See §4 token budget for derivation — these keep total input under ~8,984 tokens.
_MAX_LOGIC_CHUNKS = 20
_MAX_PATTERN_SUMMARIES = 20
_MAX_FILE_MANIFEST = 100
_MAX_IDENTIFIER_FILES = 100
_MAX_IDENTIFIERS_PER_FILE = 5
# Token budget: num_ctx=16384 minus num_predict=4096 = 12,288 input tokens.
# Char→token ratio for this mixed prose+identifier payload is ~4 chars/token.
# Guard triggers at 42,000 chars ≈ 10,500 tokens, leaving ~1,800 token safety margin.
# (Prior value of 24,000 used chars÷3 ratio appropriate only for dense code — was 2× too aggressive.)
_CONTEXT_CHAR_LIMIT = 42_000


def make_skills_aggregator_node(chroma: ChromaLibrarianClient) -> Callable[[dict], dict]:
    """Return a skills_aggregator node with an injected ChromaLibrarianClient.

    Runs three targeted ChromaDB queries (Logic/Config/Interface) and aggregates
    the results into a SkillsData with concrete structural fields for synthesis.

    Args:
        chroma: ChromaLibrarianClient instance constructed at server startup.

    Returns:
        LangGraph node function that writes state["skills_data"] (SkillsData | None).
    """

    # ── Private helpers — all inside the factory closure ─────────────────────
    # These are only ever called from skills_aggregator below and operate on
    # ChromaDB result dicts — module cohesion rule satisfied.

    def _is_real_identifier(s: str) -> bool:
        """Return True if s is a class/function name, not an import or comment.

        Filters out import lines, comment lines, dunder names, and digit-leading
        strings that appear as key_identifiers in ChromaDB metadata.

        Args:
            s: A string from the key_identifiers JSON list.

        Returns:
            True if s represents a real class or function name.
        """
        return (
            not s.startswith(("import ", "from ", "#"))
            and len(s) > 2
            and not s.startswith(("__", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))
        )

    def _infer_source_type(file_path: str) -> str:
        """Return source type label based on file extension.

        Args:
            file_path: Path string from ChromaDB metadata.

        Returns:
            'python' for .py files, 'notebook' for .ipynb, 'other' for everything else.
        """
        if file_path.endswith(".py"):
            return "python"
        if file_path.endswith(".ipynb"):
            return "notebook"
        return "other"

    def _build_chunk_candidates(logic_results: list[dict]) -> list[dict]:
        """Build ranked candidate dicts from Logic query results.

        Deserializes key_identifiers from JSON (ChromaDB stores it as a string —
        unlike tech_stack which _unpack_query_results already deserializes).
        Filters import lines from key_identifiers before returning.

        Args:
            logic_results: Result dicts from chroma.query(where={"semantic_type": "Logic"}).

        Returns:
            List of dicts with keys: file_path, intent_summary, key_identifiers
            (filtered real names only), tech_stack (already list), source_type.
        """
        candidates = []
        for doc in logic_results:
            meta = doc.get("metadata", {})
            fp = meta.get("file_path", "")
            # key_identifiers is stored as json.dumps(list) — NOT deserialized by
            # _unpack_query_results (only tech_stack is). Must call json.loads here.
            raw_ids = meta.get("key_identifiers", "[]")
            try:
                all_ids: list[str] = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
            except (json.JSONDecodeError, TypeError):
                all_ids = []
            real_ids = [s for s in all_ids if _is_real_identifier(s)]
            candidates.append(
                {
                    "file_path": fp,
                    "intent_summary": meta.get("intent_summary", ""),
                    "key_identifiers": real_ids,
                    "tech_stack": meta.get("tech_stack", []),
                    "source_type": _infer_source_type(fp),
                }
            )
        return candidates

    def _rank_chunks(candidates: list[dict]) -> list[dict]:
        """Sort candidates: Python/notebook before other, then by identifier count.

        Ranking priority (descending):
          1. source_type in ('python', 'notebook') — README/md pushed to bottom
          2. len(key_identifiers) — real class/function name density
          3. len(tech_stack) — tiebreaker only

        Tech-stack cardinality as primary sort is explicitly prohibited (confirmed
        by live measurement to surface 14/30 .md chunks with the old strategy).

        Args:
            candidates: Output of _build_chunk_candidates.

        Returns:
            Sorted list; caller slices to _MAX_LOGIC_CHUNKS.
        """
        return sorted(
            candidates,
            key=lambda c: (
                0 if c["source_type"] in ("python", "notebook") else 1,
                -len(c["key_identifiers"]),
                -len(c["tech_stack"]),
            ),
        )

    def _build_identifier_index(logic_results: list[dict]) -> dict[str, list[str]]:
        """Build a file_path → class/function names index from Logic results.

        Deserializes key_identifiers from JSON per document (same caveat as
        _build_chunk_candidates — ChromaDB does not deserialize this field).
        Deduplicates per file, filters import lines, caps per file and total.

        Args:
            logic_results: Result dicts from chroma.query(where={"semantic_type": "Logic"}).

        Returns:
            Dict mapping file_path to up to _MAX_IDENTIFIERS_PER_FILE real identifiers.
            Total files capped at _MAX_IDENTIFIER_FILES.
        """
        # Use insertion-order dict to preserve frequency ordering (earlier = more common)
        file_ids: dict[str, list[str]] = {}
        for doc in logic_results:
            meta = doc.get("metadata", {})
            fp = meta.get("file_path", "")
            if not fp:
                continue
            raw_ids = meta.get("key_identifiers", "[]")
            try:
                all_ids: list[str] = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
            except (json.JSONDecodeError, TypeError):
                all_ids = []
            real_ids = [s for s in all_ids if _is_real_identifier(s)]
            if fp not in file_ids:
                file_ids[fp] = []
            for id_ in real_ids:
                if id_ not in file_ids[fp]:
                    file_ids[fp].append(id_)

        # Cap per-file and total-files
        result: dict[str, list[str]] = {}
        for fp, ids in file_ids.items():
            if len(result) >= _MAX_IDENTIFIER_FILES:
                break
            capped = ids[:_MAX_IDENTIFIERS_PER_FILE]
            if capped:
                result[fp] = capped
        return result

    def _build_file_manifest(logic_results: list[dict]) -> list[str]:
        """Deduplicate file paths from Logic results, Python/notebook first.

        Args:
            logic_results: Result dicts from chroma.query(where={"semantic_type": "Logic"}).

        Returns:
            Deduplicated list of file paths, .py/.ipynb before others, capped at
            _MAX_FILE_MANIFEST.
        """
        seen: set[str] = set()
        py_nb: list[str] = []
        other: list[str] = []
        for doc in logic_results:
            fp = doc.get("metadata", {}).get("file_path", "")
            if not fp or fp in seen:
                continue
            seen.add(fp)
            if fp.endswith((".py", ".ipynb")):
                py_nb.append(fp)
            else:
                other.append(fp)
        combined = py_nb + other
        return combined[:_MAX_FILE_MANIFEST]

    def _build_module_groups(file_paths: list[str]) -> dict[str, list[str]]:
        """Group file paths by their parent directory (up to two levels deep).

        Strips the filename first, then uses up to two directory components as the key:
          - 'src/agents/foo.py' → dir_parts=['src','agents'] → key='src/agents'
          - 'src/model.py'      → dir_parts=['src']          → key='src'
          - 'model.py'          → dir_parts=[]               → key='(root)'

        This ensures keys are always directories, never file paths — matching the
        SkillsData.module_groups description (e.g. 'src/agents', not 'src/model.py').

        Args:
            file_paths: Deduplicated list from _build_file_manifest.

        Returns:
            Dict mapping directory key → list of file paths under that directory.
        """
        groups: dict[str, list[str]] = {}
        for fp in file_paths:
            parts = fp.split("/")
            # Strip filename — group by parent directory, not by the file's own path
            dir_parts = parts[:-1]
            if len(dir_parts) >= 2:
                # e.g. ['src', 'agents', ...] → 'src/agents'
                key = "/".join(dir_parts[:2])
            elif len(dir_parts) == 1:
                # e.g. ['src'] → 'src'
                key = dir_parts[0]
            else:
                # bare filename with no directory component
                key = "(root)"
            groups.setdefault(key, []).append(fp)
        return groups

    def _build_pattern_summaries(
        logic_results: list[dict],
        config_results: list[dict],
        interface_results: list[dict],
    ) -> list[str]:
        """Collect deduplicated intent_summary strings across all three query types.

        Covers Logic + Config + Interface signals. Deduplicates in order so Logic
        summaries appear first (they are the primary structural evidence).

        Args:
            logic_results: Logic query results.
            config_results: Config query results.
            interface_results: Interface query results.

        Returns:
            Deduplicated list of non-empty intent_summary strings, capped at
            _MAX_PATTERN_SUMMARIES.
        """
        seen: set[str] = set()
        summaries: list[str] = []
        for doc in logic_results + config_results + interface_results:
            s = doc.get("metadata", {}).get("intent_summary", "")
            if s and s not in seen:
                seen.add(s)
                summaries.append(s)
                if len(summaries) >= _MAX_PATTERN_SUMMARIES:
                    break
        return summaries

    def _apply_overflow_guard(
        logic_chunks: list[dict],
        identifier_index: dict[str, list[str]],
        file_manifest: list[str],
    ) -> tuple[list[dict], dict[str, list[str]], list[str]]:
        """Reduce logic_chunks and identifier_index if rendered size exceeds budget.

        Estimates token count as chars÷3 (conservative for code strings — shorter
        tokens than prose). Triggers if the three largest components together exceed
        _CONTEXT_CHAR_LIMIT chars. Reduces to fallback caps to stay inside num_ctx=16384.

        Args:
            logic_chunks: Ranked top chunks after initial cap.
            identifier_index: File→identifiers map after per-file cap.
            file_manifest: Deduplicated file path list.

        Returns:
            3-tuple (logic_chunks, identifier_index, file_manifest) — file_manifest
            is returned unchanged; only logic_chunks and identifier_index are reduced.
        """
        rendered_size = (
            len(str(identifier_index))
            + len(str(logic_chunks))
            + len(str(file_manifest))
        )
        # chars÷3 is conservative for code; true token count may be lower
        if rendered_size // 3 > _CONTEXT_CHAR_LIMIT // 3:
            logger.warning(
                "skills_aggregator: overflow guard triggered (rendered_size=%d chars) — "
                "reducing logic_chunks to 10, identifier_index to 50 files",
                rendered_size,
            )
            logic_chunks = logic_chunks[:10]
            identifier_index = dict(list(identifier_index.items())[:50])
        return logic_chunks, identifier_index, file_manifest

    # ── Node inner function ───────────────────────────────────────────────────

    def skills_aggregator(state: dict) -> dict:
        """Query ChromaDB with three targeted queries and aggregate skills evidence.

        Guard clauses handle query errors and empty collections; happy path builds
        all SkillsData fields and applies overflow guard before returning.

        Args:
            state: LangGraph state dict — reads 'trace'.

        Returns:
            Dict with 'skills_data' (SkillsData | None), optional 'error', and 'trace'.
        """
        trace: list[str] = list(state.get("trace", []))

        # Guard clause — query errors → early return (non-fatal; no stack trace exposed)
        try:
            logic_results = chroma.query(
                text="class function method implementation algorithm data pipeline",
                where={"semantic_type": {"$eq": "Logic"}},
                n_results=500,
            )
            config_results = chroma.query(
                text="configuration settings constants environment variables",
                where={"semantic_type": {"$eq": "Config"}},
                n_results=50,
            )
            interface_results = chroma.query(
                text="API endpoint interface protocol abstract class",
                where={"semantic_type": {"$eq": "Interface"}},
                n_results=50,
            )
        except Exception as exc:
            logger.error(
                "skills_aggregator query failed [%s]: %s", type(exc).__name__, str(exc)[:200]
            )
            return {
                "skills_data": None,
                "error": "No indexed content found.",
                "trace": trace + ["skills_aggregator: error"],
            }

        # Guard clause — empty collection → early return
        if not logic_results and not config_results and not interface_results:
            return {
                "skills_data": None,
                "error": "No indexed content found.",
                "trace": trace + ["skills_aggregator: no content"],
            }

        # Happy path — build all SkillsData fields sequentially
        candidates = _build_chunk_candidates(logic_results)
        ranked = _rank_chunks(candidates)
        logic_chunks = ranked[:_MAX_LOGIC_CHUNKS]

        identifier_index = _build_identifier_index(logic_results)
        file_manifest = _build_file_manifest(logic_results)
        module_groups = _build_module_groups(file_manifest)
        # Notebooks are a subset of file_manifest; extract after dedup
        notebook_files = [fp for fp in file_manifest if fp.endswith(".ipynb")]
        pattern_summaries = _build_pattern_summaries(
            logic_results, config_results, interface_results
        )
        logic_chunks, identifier_index, file_manifest = _apply_overflow_guard(
            logic_chunks, identifier_index, file_manifest
        )

        # has_source_code: True if any Python/notebook Logic chunk has real identifiers
        has_source_code = any(
            c["source_type"] in ("python", "notebook")
            and any(_is_real_identifier(id_) for id_ in c["key_identifiers"])
            for c in logic_chunks
        )

        # repo_url from first Logic result (all chunks in a repo share the same URL)
        repo_url = logic_results[0]["metadata"].get("repo_url", "") if logic_results else ""

        # aggregate_chroma_docs: tech_stack already deserialized by _unpack_query_results;
        # semantic_counter gives distribution across all three query types combined.
        all_results = logic_results + config_results + interface_results
        semantic_counter, tech_counter, _ = aggregate_chroma_docs(all_results)

        skills_data = SkillsData(
            tech_stacks=[t for t, _ in tech_counter.most_common()],
            semantic_type_distribution=dict(semantic_counter),
            pattern_summaries=pattern_summaries,
            doc_count=len(all_results),
            file_manifest=file_manifest,
            module_groups=module_groups,
            identifier_index=identifier_index,
            logic_chunks=logic_chunks,
            notebook_files=notebook_files,
            repo_url=repo_url,
            has_source_code=has_source_code,
        )
        logger.debug(
            "skills_aggregator: %d docs → %d tech stacks, %d files, has_source_code=%s",
            len(all_results),
            len(skills_data.tech_stacks),
            len(file_manifest),
            has_source_code,
        )
        return {
            "skills_data": skills_data,
            "trace": trace + [f"skills_aggregator: {len(all_results)} docs"],
        }

    return skills_aggregator

"""skills_synthesizer LangGraph node — Spec §3d, §10 Phase 4.

Pre-builds both chain variants at factory time (source-code and docs-only) — never
per-request. Selects the appropriate chain at invoke time based on has_source_code.
All 13 _HUMAN template variables are assembled by _build_chain_inputs() inside the
factory closure, keeping the node function itself a thin dispatcher.
"""

import logging
from collections.abc import Callable

from langchain_ollama import ChatOllama

from src.chains.skills_chain import build_skills_chain
from src.models.skills import SkillsData
from src.utils.retry import invoke_with_retry

logger = logging.getLogger(__name__)


def make_skills_synthesizer_node(llm: ChatOllama) -> Callable[[dict], dict]:
    """Return a skills_synthesizer node with injected ChatOllama.

    Pre-builds both chain variants at factory construction time — one for source-code
    repos (six sections) and one for documentation-only repos (three sections).
    Chain selection and input assembly happen inside the node function per-request;
    chain construction never does.

    Args:
        llm: ChatOllama singleton (temperature=0.2, num_ctx=16384, num_predict=4096).
            Built once at server startup and injected — never constructed here.

    Returns:
        LangGraph node function that reads state["skills_data"] and writes
        state["skills_markdown"].
    """
    # Both chains built at factory time — never rebuilt per request.
    chain_source = build_skills_chain(llm, has_source_code=True)
    chain_docs = build_skills_chain(llm, has_source_code=False)

    def _build_chain_inputs(skills_data: SkillsData) -> dict:
        """Assemble all 13 _HUMAN template variables from a SkillsData instance.

        Formats structured fields (lists, dicts) as human-readable strings so the
        LLM receives concrete evidence — not Python repr. External data stays in
        this function (human turn) and never reaches the system prompt.

        Args:
            skills_data: Aggregated skills evidence from skills_aggregator.

        Returns:
            Dict with all 13 keys expected by the _HUMAN prompt template.
        """
        # Module groups: "src/agents: [foo.py, bar.py]" one per line
        module_groups_text = "\n".join(
            f"{pkg}: [{', '.join(files)}]"
            for pkg, files in skills_data.module_groups.items()
        ) or "(none)"

        # File manifest: newline-separated paths
        file_manifest_text = "\n".join(skills_data.file_manifest) or "(none)"

        # Notebook files: newline-separated paths
        notebook_files_text = "\n".join(skills_data.notebook_files) or "(none)"

        # Identifier index: "src/model.py: [MyModel, train]" one per line
        identifier_index_text = "\n".join(
            f"{fp}: [{', '.join(ids)}]"
            for fp, ids in skills_data.identifier_index.items()
        ) or "(none)"

        # Logic chunks: "src/model.py | MyModel, train | trains a CNN" one per line
        # Use local variables in f-string — avoids .format() raising KeyError/IndexError
        # if intent_summary or file_path contains literal braces from JSON-heavy content.
        def _fmt_chunk(c: dict) -> str:
            fp = c.get("file_path", "")
            ids = ", ".join(c.get("key_identifiers", [])) or "(no identifiers)"
            summary = c.get("intent_summary", "")
            return f"{fp} | {ids} | {summary}"

        logic_chunks_text = "\n".join(_fmt_chunk(c) for c in skills_data.logic_chunks) or "(none)"

        # Tech stacks: bullet list
        tech_text = (
            "\n".join(f"- {t}" for t in skills_data.tech_stacks) or "(none detected)"
        )

        return {
            "repo_url": skills_data.repo_url,
            "doc_count": skills_data.doc_count,
            "tech_count": len(skills_data.tech_stacks),
            "module_groups": module_groups_text,
            "file_manifest_count": len(skills_data.file_manifest),
            "file_manifest": file_manifest_text,
            "notebook_count": len(skills_data.notebook_files),
            "notebook_files": notebook_files_text,
            "identifier_index": identifier_index_text,
            "logic_chunks": logic_chunks_text,
            "tech_stacks": tech_text,
            "semantic_type_distribution": str(skills_data.semantic_type_distribution),
            "pattern_summaries": (
                "\n".join(f"- {p}" for p in skills_data.pattern_summaries) or "(none)"
            ),
        }

    def skills_synthesizer(state: dict) -> dict:
        """Generate PROJECT_SKILLS.md Markdown from SkillsData.

        Selects chain variant based on has_source_code, assembles all 13 prompt
        variables via _build_chain_inputs, and writes the result to skills_markdown.
        LLM errors are non-fatal — returns skills_markdown=None with error surfaced.

        Args:
            state: LangGraph state dict — reads 'skills_data' and 'trace'.

        Returns:
            Dict with 'skills_markdown' (str | None), optional 'error', and 'trace'.
        """
        skills_data: SkillsData | None = state.get("skills_data")
        trace: list[str] = list(state.get("trace", []))

        # Guard clause — no aggregated data → early return (non-fatal)
        if skills_data is None:
            return {
                "skills_markdown": None,
                "error": "No skills data available for synthesis.",
                "trace": trace + ["skills_synthesizer: skipped (no skills_data)"],
            }

        # Select pre-built chain variant — never construct a new chain here
        chain = chain_source if skills_data.has_source_code else chain_docs

        try:
            markdown: str = invoke_with_retry(chain, _build_chain_inputs(skills_data))
        except Exception as exc:
            logger.error(
                "skills_synthesizer failed [%s]: %s", type(exc).__name__, str(exc)[:200]
            )
            return {
                "skills_markdown": None,
                "error": "Skills synthesis failed. Please retry.",
                "trace": trace + ["skills_synthesizer: error"],
            }

        # Normalize encoding corruption from gpt-oss:20b output — the model emits UTF-8
        # arrow/dash characters that get mangled when Ollama streams bytes decoded as Latin-1.
        # Replace only the known corrupted multi-byte sequences; leave other characters intact.
        markdown = (
            markdown
            .replace("\u0096", "–")   # Windows-1252 en-dash → Unicode en-dash
            .replace("\u0097", "—")   # Windows-1252 em-dash → Unicode em-dash
            .replace("\u0092", "'")   # Windows-1252 right single quote → apostrophe
            .replace("â\x80\x94", "—")   # UTF-8 em-dash decoded as Latin-1
            .replace("â\x80\x93", "–")   # UTF-8 en-dash decoded as Latin-1
            .replace("â\x86\x92", "→")   # UTF-8 rightwards arrow decoded as Latin-1
        )

        logger.debug("skills_synthesizer: generated %d chars of Markdown", len(markdown))
        return {
            "skills_markdown": markdown,
            "trace": trace + ["skills_synthesizer: ok"],
        }

    return skills_synthesizer

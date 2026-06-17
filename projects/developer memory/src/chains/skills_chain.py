"""Chain for PROJECT_SKILLS.md generation — Spec §3c, §10 Phase 3.

Produces free-text Markdown via StrOutputParser. Two system prompt variants are
pre-built at factory time — one for source-code repos (six sections) and one for
documentation-only repos (three sections). Temperature 0.2 per §4: enough creativity
for readable prose while staying grounded in indexed evidence.

The skills_synthesizer node calls build_skills_chain at factory construction time
(never per-request) and writes the result via file_exporter.
"""

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

# ── System prompts — selected once at chain-build time, never at request time ─

_SYSTEM_SOURCE_CODE = (
    "You are a technical documentation generator producing a PROJECT_SKILLS.md file "
    "for use as a Claude AI context document — a digital twin that lets Claude reason "
    "about this codebase without re-reading source files.\n\n"
    "The output must be valid Markdown with these six required sections:\n\n"
    "## Purpose\n"
    "One concise paragraph: what this codebase does and why it exists.\n\n"
    "## Architecture\n"
    "Key architectural decisions and system topology grounded in the symbol index, "
    "module structure, and source chunks. Name concrete components, pipelines, or layers "
    "you can identify from the file paths and class/function names provided.\n\n"
    "## Tech Stack\n"
    "Each technology with a one-line note on *how* it is used in this project specifically "
    "— not a generic description. Omit technologies with no supporting evidence.\n\n"
    "## Coding Patterns\n"
    "Named patterns with concrete explanations tied to what the indexed code actually does. "
    "Reference specific class names, function names, or file paths from the symbol index.\n\n"
    "## Conventions & Tendencies\n"
    "Recurring design decisions: naming conventions, error handling style, test strategy, "
    "module structure, dependency injection approach — inferred from the source chunks "
    "and convention patterns provided.\n\n"
    "## Working With This Codebase\n"
    "Practical notes for an AI assistant: what to check before making changes, "
    "what invariants must not be broken, what the dominant extension points are.\n\n"
    "Rules:\n"
    "- Every claim must be grounded in the provided symbol index, source chunks, or "
    "tech stack data. Do not invent components, frameworks, or patterns not evidenced.\n"
    "- Be specific and concrete — reference actual file paths, class names, and function "
    "names from the symbol index. Generic descriptions have no value in this context.\n"
    "- Ignore summaries that describe general methodology, principles, or guidelines "
    "(e.g. 'this document explains SOLID principles'). Only reason from summaries that "
    "describe concrete implementation — actual classes, functions, or data flows.\n"
    "- Ignore tech stack entries that come from dependency directories (.venv, node_modules, "
    "site-packages) rather than the project's own source code.\n"
    "- If the evidence is insufficient to fill a section, write "
    "'Insufficient indexed implementation data' rather than inventing plausible content.\n"
    "- Aim for 600–1200 words total. Dense and precise beats long and vague."
)

_SYSTEM_DOCS_ONLY = (
    "You are a technical documentation generator producing a PROJECT_SKILLS.md file "
    "for use as a Claude AI context document.\n\n"
    "> This repository contains primarily documentation. The following describes the "
    "documented methodology and projects.\n\n"
    "The output must be valid Markdown with these three required sections:\n\n"
    "## Purpose\n"
    "One concise paragraph: what this documentation repository covers and why it exists.\n\n"
    "## Documented Topics\n"
    "Key subjects, methodologies, or project descriptions covered in the documentation, "
    "drawn from the intent summaries and pattern data provided.\n\n"
    "## Working With This Repository\n"
    "Practical notes for an AI assistant reading or extending this documentation: "
    "what the dominant topics are, how the content is organised, what to look for.\n\n"
    "Rules:\n"
    "- Every claim must be grounded in the provided data. Do not invent content.\n"
    "- Aim for 300–600 words total."
)

# ── Human turn — all external data; new variables added in Phase 3 ─────────────
# External data (file paths, identifiers, chunks) goes in the human turn only —
# never in the system prompt. This prevents prompt injection from repo content.
_HUMAN = (
    "Repository: {repo_url}\n"
    "Indexed: {doc_count} documents, {tech_count} unique technologies.\n\n"
    "=== MODULE STRUCTURE ===\n"
    "Top-level packages/directories:\n{module_groups}\n\n"
    "Source files ({file_manifest_count} total, capped at 100):\n{file_manifest}\n\n"
    "Jupyter notebooks ({notebook_count}):\n{notebook_files}\n\n"
    "=== SYMBOL INDEX (file → classes/functions, capped at 100 files × 5 each) ===\n"
    "{identifier_index}\n\n"
    "=== TOP SOURCE CHUNKS (Python/notebook first, ranked by symbol density) ===\n"
    "{logic_chunks}\n\n"
    "=== TECH STACK ===\n"
    "{tech_stacks}\n\n"
    "=== CODE COMPOSITION ===\n"
    "{semantic_type_distribution}\n\n"
    "=== CONVENTION PATTERNS (config + interface signals) ===\n"
    "{pattern_summaries}\n\n"
    "Generate PROJECT_SKILLS.md with all required sections. "
    "Stop after Working With This Codebase. Do not repeat any section. "
    "End the document with exactly one blank line after the last paragraph."
)


def build_skills_chain(llm: ChatOllama, *, has_source_code: bool = True) -> Runnable:
    """Build the skills synthesis chain for source-code or docs-only repos.

    Selects the appropriate system prompt at chain-build time — never per-request.
    The prompt is assembled here (not at module level) so that the has_source_code
    flag can control which system constant is used without rebuilding the LLM.

    Args:
        llm: ChatOllama singleton (temperature=0.2, num_ctx=16384, num_predict=4096).
            Built once at server startup and injected — never constructed here.
        has_source_code: Selects system prompt variant. True = six-section source code
            digital twin (_SYSTEM_SOURCE_CODE). False = documentation-only three-section
            variant (_SYSTEM_DOCS_ONLY). Keyword-only to prevent accidental positional
            misuse.

    Returns:
        A LangChain Runnable: dict → str (Markdown). Accepts all 13 _HUMAN variables.
    """
    # System prompt selected at build time — not at invoke time. This is intentional:
    # the chain is built once per has_source_code variant, never rebuilt per request.
    system = _SYSTEM_SOURCE_CODE if has_source_code else _SYSTEM_DOCS_ONLY
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", _HUMAN)])
    # StrOutputParser: free-text Markdown generation — no structured schema needed.
    return prompt | llm | StrOutputParser()

"""Experiment: Compare current vs. proposed intent_summary_chain annotation.

Goal: Verify that adding a `key_identifiers` field to _ChunkAnnotation preserves
concrete API names (e.g. AssistantAgent, UserProxyAgent) that the current prompt
drops — improving code search recall.

Test files: 3 real Python files from the autogen_crop_yield_simple_agent project.
Output: side-by-side comparison of current vs. proposed annotation, saved to
experiments/results/annotation_prompt_<timestamp>.json

Usage:
    poetry run python experiments/annotation_prompt_experiment.py
"""

import json
import logging
import os
import pathlib
from datetime import UTC, datetime
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = os.getenv("OLLAMA_MODEL", "gemma4:26b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

AUTOGEN_DIR = pathlib.Path(
    "/Users/timhazed/src/datascience/datascience_projects/projects"
    "/LLM Framework Benchmarking/autogen_crop_yield_simple_agent"
)

TEST_FILES = [
    AUTOGEN_DIR / "agents" / "prediction_agent.py",
    AUTOGEN_DIR / "agents" / "base_agent.py",
    AUTOGEN_DIR / "agents" / "data_preparation_agent.py",
]

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# ── Current schema (no key_identifiers) ──────────────────────────────────────


class CurrentAnnotation(BaseModel):
    """Mirrors the current _ChunkAnnotation in intent_summary_chain.py."""

    intent_summary: str = Field(
        description=(
            "Explain *why* this code exists and what developer intent it encodes. "
            "Focus on the rationale and design decision, not a description of what the code does. "
            "2–4 sentences."
        ),
    )
    tech_stack: list[str] = Field(
        default_factory=list,
        description=(
            "Technology and framework identifiers detected in this chunk, "
            "e.g. ['FastAPI', 'Pydantic', 'LangChain']. Empty list if none detected."
        ),
    )
    semantic_type: Literal["Logic", "Config", "Boilerplate", "Interface"] = Field(
        default="Logic",
        description=(
            "Classify the chunk: "
            "Logic = algorithmic/business logic code; "
            "Config = configuration, settings, constants; "
            "Boilerplate = scaffolding, generated code, imports; "
            "Interface = API boundaries, protocol definitions, abstract classes."
        ),
    )


# ── Proposed schema (adds key_identifiers) ───────────────────────────────────


class ProposedAnnotation(BaseModel):
    """Proposed _ChunkAnnotation with key_identifiers field added."""

    intent_summary: str = Field(
        description=(
            "Explain *why* this code exists and what developer intent it encodes. "
            "Focus on the rationale and design decision, not a description of what the code does. "
            "2–4 sentences."
        ),
    )
    tech_stack: list[str] = Field(
        default_factory=list,
        description=(
            "Technology and framework identifiers detected in this chunk, "
            "e.g. ['FastAPI', 'Pydantic', 'LangChain']. Empty list if none detected."
        ),
    )
    semantic_type: Literal["Logic", "Config", "Boilerplate", "Interface"] = Field(
        default="Logic",
        description=(
            "Classify the chunk: "
            "Logic = algorithmic/business logic code; "
            "Config = configuration, settings, constants; "
            "Boilerplate = scaffolding, generated code, imports; "
            "Interface = API boundaries, protocol definitions, abstract classes."
        ),
    )
    key_identifiers: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete identifiers present in this chunk that a developer might search for: "
            "class names, function names, method names, and framework-specific API names "
            "(e.g. ['AssistantAgent', 'UserProxyAgent', 'initiate_chat', 'PredictionAgent', 'autogen']). "
            "Include the actual names verbatim — do not paraphrase or generalize."
        ),
    )


# Refined uses ProposedAnnotation schema — same fields, new few-shot system prompt.


# ── Prompt (shared by both variants) ─────────────────────────────────────────

_SYSTEM_CURRENT = (
    "You are a code intelligence analyst specialising in developer intent. "
    "Your job is to infer *why* a piece of code was written — the rationale, design decision, "
    "or problem it solves — not to describe what the code does syntactically. "
    "Identify the technologies and frameworks present, and classify the chunk semantically."
)

_SYSTEM_PROPOSED = (
    "You are a code intelligence analyst specialising in developer intent. "
    "Your job is to infer *why* a piece of code was written — the rationale, design decision, "
    "or problem it solves — not to describe what the code does syntactically. "
    "Identify the technologies and frameworks present, classify the chunk semantically, "
    "and extract every concrete identifier (class names, function names, API names) verbatim from the code."
)

_HUMAN = (
    "File: {path}\n\n"
    "```\n{content}\n```\n\n"
    "Annotate this code chunk with: intent_summary, tech_stack, and semantic_type."
)

_HUMAN_PROPOSED = (
    "File: {path}\n\n"
    "```\n{content}\n```\n\n"
    "Annotate this code chunk.\n\n"
    "For key_identifiers: scan the code and list every class name, function name, method name, "
    "and third-party API name you can find — copy them verbatim as they appear in the code. "
    "Examples of what to include: class definitions (class Foo:), instantiated objects "
    "(autogen.AssistantAgent(...)), called methods (.initiate_chat(...)), imported names. "
    "Do NOT paraphrase — copy the exact identifier string."
)

_SYSTEM_REFINED = (
    "You are a code intelligence analyst specialising in developer intent. "
    "Your job is to infer *why* a piece of code was written — the rationale, design decision, "
    "or problem it solves — not to describe what the code does syntactically. "
    "Identify the technologies and frameworks present, classify the chunk semantically, "
    "and extract every concrete identifier (class names, function names, API names) verbatim from the code.\n\n"
    "Example:\n\n"
    "File: src/auth/session.py\n"
    "```python\n"
    "def validate_token(token: str):\n"
    "    # Check expiry to prevent stale session hijacking\n"
    "    return redis_client.get('auth_' + token)\n"
    "```\n\n"
    "intent_summary: Validates session tokens against a Redis cache to prevent stale session hijacking.\n"
    "tech_stack: [\"Python\", \"Redis\"]\n"
    "semantic_type: Logic\n"
    'key_identifiers: ["validate_token", "redis_client", "get"]'
)

_HUMAN_REFINED = (
    "File: {path}\n\n"
    "```\n{content}\n```\n\n"
    "Annotate this code chunk following the format in the system example.\n\n"
    "For key_identifiers: scan the code and list every class name, function name, method name, "
    "and third-party API name verbatim — copy them exactly as they appear (case-sensitive). "
    "Do NOT paraphrase or omit PascalCase names like AssistantAgent or UserProxyAgent."
)

_PROMPT_CURRENT = ChatPromptTemplate.from_messages([("system", _SYSTEM_CURRENT), ("human", _HUMAN)])
_PROMPT_PROPOSED = ChatPromptTemplate.from_messages([("system", _SYSTEM_PROPOSED), ("human", _HUMAN_PROPOSED)])
_PROMPT_REFINED = ChatPromptTemplate.from_messages([("system", _SYSTEM_REFINED), ("human", _HUMAN_REFINED)])


# ── Helpers ───────────────────────────────────────────────────────────────────


def _truncate(content: str, max_chars: int = 6000) -> str:
    """Truncate and strip file content to stay within context budget.

    Strips leading/trailing whitespace — Gemma models are sensitive to noisy
    whitespace inside code blocks when attempting structured output.
    """
    content = content.strip()
    if len(content) <= max_chars:
        return content
    logger.warning("Truncating content to %d chars", max_chars)
    return content[:max_chars] + "\n# [truncated for experiment]"


def _annotate(chain: Any, path: str, content: str) -> dict:
    """Run the chain and return the annotation as a dict. Retries once on parse failure."""
    for attempt in range(2):
        try:
            result = chain.invoke({"path": path, "content": content})
            return result.model_dump()
        except Exception as e:
            if attempt == 0:
                logger.warning("Attempt 1 failed (%s), retrying...", type(e).__name__)
            else:
                logger.error("Both attempts failed: %s", e)
                return {"intent_summary": "ERROR", "tech_stack": [], "semantic_type": "Logic", "key_identifiers": []}
    return {}


def _search_hit(annotation: dict, targets: list[str]) -> list[str]:
    """Return which target identifiers appear anywhere in the annotation values."""
    annotation_text = json.dumps(annotation).lower()
    return [t for t in targets if t.lower() in annotation_text]


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run annotation experiment on test files and save results."""
    logger.info("Model: %s @ %s", MODEL, OLLAMA_HOST)

    llm = ChatOllama(
        model=MODEL,
        temperature=0.0,
        num_predict=1024,
        num_ctx=8192,
        base_url=OLLAMA_HOST,
        request_timeout=120.0,
    )

    chain_current = _PROMPT_CURRENT | llm.with_structured_output(CurrentAnnotation)
    chain_proposed = _PROMPT_PROPOSED | llm.with_structured_output(ProposedAnnotation)
    chain_refined = _PROMPT_REFINED | llm.with_structured_output(ProposedAnnotation)

    # Identifiers we expect a search to find in these files
    search_targets = [
        "AssistantAgent",
        "UserProxyAgent",
        "PredictionAgent",
        "BaseAgentConfig",
        "initiate_chat",
        "autogen",
    ]

    results = []

    for fpath in TEST_FILES:
        if not fpath.exists():
            logger.warning("File not found: %s", fpath)
            continue

        content = _truncate(fpath.read_text())
        rel_path = str(fpath.relative_to(AUTOGEN_DIR.parent.parent.parent))
        logger.info("Annotating: %s", rel_path)

        logger.info("  → current schema")
        current = _annotate(chain_current, rel_path, content)

        logger.info("  → proposed schema")
        proposed = _annotate(chain_proposed, rel_path, content)

        logger.info("  → refined schema")
        refined = _annotate(chain_refined, rel_path, content)

        current_hits = _search_hit(current, search_targets)
        proposed_hits = _search_hit(proposed, search_targets)
        refined_hits = _search_hit(refined, search_targets)

        result = {
            "file": rel_path,
            "current": current,
            "proposed": proposed,
            "refined": refined,
            "search_recall": {
                "targets": search_targets,
                "current_hits": current_hits,
                "proposed_hits": proposed_hits,
                "refined_hits": refined_hits,
                "current_recall": len(current_hits) / len(search_targets),
                "proposed_recall": len(proposed_hits) / len(search_targets),
                "refined_recall": len(refined_hits) / len(search_targets),
            },
        }
        results.append(result)

        # Print summary inline
        print(f"\n{'='*60}")
        print(f"FILE: {fpath.name}")
        print(f"{'='*60}")
        print(f"\n[CURRENT] intent_summary:\n  {current.get('intent_summary', '')}")
        print(f"[CURRENT] tech_stack: {current.get('tech_stack', [])}")
        print(f"[CURRENT] hits: {current_hits} ({len(current_hits)}/{len(search_targets)})")
        print(f"\n[PROPOSED] intent_summary:\n  {proposed.get('intent_summary', '')}")
        print(f"[PROPOSED] key_identifiers: {proposed.get('key_identifiers', [])}")
        print(f"[PROPOSED] hits: {proposed_hits} ({len(proposed_hits)}/{len(search_targets)})")
        print(f"\n[REFINED] design_intent:\n  {refined.get('design_intent', '')}")
        print(f"[REFINED] chunk_classification: {refined.get('chunk_classification', '')}")
        print(f"[REFINED] key_identifiers: {refined.get('key_identifiers', [])}")
        print(f"[REFINED] hits: {refined_hits} ({len(refined_hits)}/{len(search_targets)})")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"annotation_prompt_{ts}.json"
    out_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "model": MODEL,
                "search_targets": search_targets,
                "files_tested": len(results),
                "results": results,
            },
            indent=2,
        )
    )
    logger.info("Results saved to %s", out_path)

    # Aggregate recall
    if results:
        avg_current = sum(r["search_recall"]["current_recall"] for r in results) / len(results)
        avg_proposed = sum(r["search_recall"]["proposed_recall"] for r in results) / len(results)
        avg_refined = sum(r["search_recall"]["refined_recall"] for r in results) / len(results)
        print(f"\n{'='*60}")
        print(f"AGGREGATE RECALL across {len(results)} files")
        print(f"  Current schema:  {avg_current:.0%}")
        print(f"  Proposed schema: {avg_proposed:.0%}")
        print(f"  Refined schema:  {avg_refined:.0%}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

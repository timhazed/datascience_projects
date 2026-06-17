"""State TypedDict for the Skills Export Pipeline (generate_skills_pkg MCP tool).

Spec §3 — SkillsState flows through:
  skills_cache_guard → (hit) → END
  skills_cache_guard → (miss) → skills_aggregator → skills_synthesizer → file_exporter → END
"""

import operator
from typing import Annotated, NotRequired, TypedDict

from src.models.skills import SkillsData


class SkillsState(TypedDict):
    """Pipeline state for generate_skills_pkg.

    Fields:
        target_path: Resolved, validated write destination for PROJECT_SKILLS.md.
            Pre-validated by validate_target_path before pipeline invocation.
        skills_data: Aggregated ChromaDB data from skills_aggregator, or None if no content found.
            Conditional edge routes to END with error if None.
        skills_markdown: Generated PROJECT_SKILLS.md content from skills_synthesizer.
        export_result: Write outcome from file_exporter: {"path": str, "bytes_written": int}.
            None until file_exporter runs, or populated by skills_cache_guard on cache hit.
        error: Non-fatal error message.
        trace: Append-only execution log.
        repo_url: Git repository URL — set by skills_cache_guard from ChromaDB probe.
        branch: Branch name — set by skills_cache_guard from ChromaDB probe.
        synced_sha: Current commit SHA from ChromaDB metadata — set by skills_cache_guard.
        cache_hit: True when skills_sha matches synced_sha and cached file is copied.
            Defaults to False in the initial state dict passed by generate_skills_pkg.
        force: True when caller requests bypass of the SHA gate (always synthesize).
            Defaults to False.

    All five new fields are NotRequired so existing invocation sites that omit them
    remain valid. Node implementations read them with state.get("field", default).
    """

    target_path: str
    skills_data: SkillsData | None
    skills_markdown: str | None
    export_result: dict | None
    error: str | None
    trace: Annotated[list[str], operator.add]

    # ── new fields — all NotRequired for backward compatibility ───────────────
    repo_url: NotRequired[str]
    branch: NotRequired[str]
    synced_sha: NotRequired[str]
    cache_hit: NotRequired[bool]
    force: NotRequired[bool]

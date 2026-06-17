"""Skills data model for the skills export pipeline."""

from pydantic import BaseModel, Field


class SkillsData(BaseModel):
    """Aggregated skills evidence produced by skills_aggregator from ChromaDB.

    Extended with structural fields to give skills_synthesizer concrete architectural
    evidence: file paths, symbol names, module groupings, and ranked source chunks.
    All new fields use default_factory defaults so existing callers constructing
    SkillsData(tech_stacks=[...]) continue to work without changes.
    """

    # --- existing fields (unchanged) ---
    tech_stacks: list[str] = Field(
        default_factory=list,
        description="Deduplicated technology names sorted by frequency across all indexed docs.",
    )
    semantic_type_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each semantic_type label (e.g. 'Logic', 'Config') across all docs.",
    )
    pattern_summaries: list[str] = Field(
        default_factory=list,
        description="Top representative intent summaries (Logic + Config + Interface).",
    )
    doc_count: int = Field(0, description="Total number of indexed documents analyzed.")

    # --- new fields ---
    file_manifest: list[str] = Field(
        default_factory=list,
        description=(
            "Deduplicated source file paths (capped at 100), sorted by Logic chunk frequency "
            "(proxy for file complexity). Python and notebook files listed before others."
        ),
    )
    module_groups: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Files grouped by top-level directory. Key = directory name (e.g. 'src/agents'), "
            "value = list of file paths. Allows synthesizer to describe package structure."
        ),
    )
    identifier_index: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Map of file_path → list of key_identifiers (class/function names). "
            "Capped at 100 files × 5 identifiers each. Import statements filtered out. "
            "Gives synthesizer concrete symbol names per module (~4,124 tokens at cap)."
        ),
    )
    logic_chunks: list[dict] = Field(
        default_factory=list,
        description=(
            "Top-20 Logic chunks ranked: Python/notebook source first, then by "
            "len(key_identifiers) descending. Each dict: file_path, intent_summary, "
            "key_identifiers, tech_stack, source_type ('python'|'notebook'|'other'). "
            "Primary evidence source for Architecture and Coding Patterns sections."
        ),
    )
    notebook_files: list[str] = Field(
        default_factory=list,
        description=(
            "Deduplicated .ipynb file paths. Listed separately so synthesizer can describe "
            "notebook-based exploratory workflows distinct from production source modules."
        ),
    )
    repo_url: str = Field(
        default="",
        description="Repository URL from indexed chunk metadata, for provenance.",
    )
    has_source_code: bool = Field(
        default=True,
        description=(
            "True if any Logic chunk from a .py or .ipynb file has at least one "
            "non-import key_identifier. False signals a documentation-only repo — "
            "synthesizer uses docs-only system prompt variant."
        ),
    )

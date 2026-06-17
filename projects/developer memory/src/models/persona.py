"""Persona models for the Developer Memory tendency and persona synthesis pipeline.

TendencyData and PersonaProfile are co-located because they form the two stages
of the persona pipeline: raw aggregation (TendencyData) → LLM synthesis (PersonaProfile).
"""

from pydantic import BaseModel, Field


class TendencyData(BaseModel):
    """Aggregated developer tendency data produced by tendency_scanner.

    Built from ChromaDB query results with temporal weighting (last 6 months = 3× weight).
    Passed to persona_synthesizer as prompt context — never returned directly to MCP clients.

    Args:
        scope: Query scope used — "project", "file", or "author".
        doc_count: Total number of ChromaDB documents analyzed.
        semantic_type_distribution: Count of each semantic_type label across all docs.
        tech_stack_frequency: Count of each tech stack item across all docs.
        dominant_patterns: Top N intent summary excerpts capturing recurring patterns.
        weighted_docs: Raw ChromaDB result dicts with "weight" key from apply_temporal_weight().
    """

    scope: str = Field(description="Query scope: 'project', 'file', or 'author'.")
    doc_count: int = Field(default=0, description="Total number of indexed documents analyzed.")
    semantic_type_distribution: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Count of each semantic_type label across all analyzed docs, "
            "e.g. {'Logic': 42, 'Config': 8}."
        ),
    )
    tech_stack_frequency: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Count of each tech stack item across all analyzed docs, "
            "e.g. {'FastAPI': 20, 'Pydantic': 18}."
        ),
    )
    dominant_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Top intent summary excerpts capturing the most frequently observed "
            "developer patterns and design decisions."
        ),
    )
    weighted_docs: list[dict] = Field(
        default_factory=list,
        description=(
            "Raw ChromaDB result dicts enriched with a 'weight' key "
            "from apply_temporal_weight() — recent docs weighted 3×."
        ),
    )


class PersonaProfile(BaseModel):
    """Structured LLM output from persona_synthesizer — the developer's stylistic profile.

    Used as the structured output target for with_structured_output(PersonaProfile).
    Field description= strings are injected into the JSON Schema that Gemma 4 receives
    as its output contract. Also loaded by persona_loader for use as context in
    coaching_analyzer.

    Args:
        style_summary: Narrative description of the developer's coding style and philosophy.
        dominant_patterns: Key coding patterns observed (e.g. "uses_type_annotations").
        tech_preferences: Technologies the developer reaches for most frequently.
        coaching_notes: Areas where the developer shows growth opportunities or inconsistencies.
    """

    style_summary: str = Field(
        description=(
            "2–4 sentence narrative describing this developer's overall coding style, "
            "philosophy, and characteristic approach to problem solving."
        ),
    )
    dominant_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Key coding and design patterns consistently observed across the developer's work. "
            "Each entry should be a concise phrase (3–8 words) distilled from the dominant "
            "pattern evidence provided, e.g. 'applies SOLID principles', "
            "'uses factory pattern for LLM instantiation', 'guards intent before action'. "
            "Do NOT return an empty list — extract at least 3 patterns from the evidence."
        ),
    )
    tech_preferences: list[str] = Field(
        default_factory=list,
        description=(
            "Technologies and frameworks the developer reaches for most frequently, "
            "ranked by frequency, e.g. ['FastAPI', 'Pydantic', 'LangChain']."
        ),
    )
    coaching_notes: list[str] = Field(
        default_factory=list,
        description=(
            "Specific, actionable observations grounded only in the provided tendency data. "
            "Note patterns that appear less consistently than the dominant ones, or areas "
            "where the data shows mixed signals. Do NOT invent gaps or weaknesses not "
            "evidenced in the data. If the data shows strong consistent practice, say so."
        ),
    )

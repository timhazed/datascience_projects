"""State TypedDict for the Persona Pipeline (get_dev_persona MCP tool).

Spec §3 — PersonaState flows through: tendency_scanner → persona_synthesizer.
"""

import operator
from typing import Annotated, TypedDict

from src.models.persona import PersonaProfile, TendencyData


class PersonaState(TypedDict):
    """Pipeline state for get_dev_persona.

    Fields:
        scope: Query scope — "project", "file", or "author".
        recency_months: Lookback window for temporal weighting. Docs within this window
            receive 3× weight in tendency aggregation. Default: 6.
        tendency_data: Aggregated ChromaDB patterns from tendency_scanner, or None if
            insufficient indexed data exists.
        persona_profile: LLM-synthesized developer profile from persona_synthesizer.
        error: Non-fatal error — "Insufficient indexed data for persona" if no data found.
        trace: Append-only execution log.
    """

    scope: str
    recency_months: int
    tendency_data: TendencyData | None
    persona_profile: PersonaProfile | None
    error: str | None
    trace: Annotated[list[str], operator.add]

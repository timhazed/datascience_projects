"""Shared enum/Literal definitions imported by multiple models.

Add new field names here only; all consumers update automatically.
"""

from typing import Literal

# Single canonical definition — imported by HistoryUpdateRequest and UpdateHistoryParams.
HistoryField = Literal["conditions", "medications", "allergies", "notes", "summary"]

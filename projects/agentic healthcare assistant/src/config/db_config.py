"""DBConfig — SQLite and FAISS path configuration."""

from pathlib import Path

from pydantic import BaseModel, Field


class DBConfig(BaseModel):
    """File-system paths for the SQLite patient database and FAISS index."""

    sqlite_path: Path = Field(
        description="Path to the SQLite database file, e.g. 'data/healthcare.db'"
    )
    faiss_path: Path = Field(
        description="Directory path for the FAISS index files, e.g. 'data/faiss'"
    )
    max_recent_turns: int = Field(
        default=10,
        ge=1,
        description="Max conversation turns loaded into UI and planner context",
    )
    max_summary_chars: int = Field(
        default=1500,
        ge=1,
        description="Character cap for rolling summary stored in DB",
    )
    memory_summary_cadence_turns: int = Field(
        default=3,
        ge=1,
        description="Refresh rolling summary every K successful turns",
    )
    checkpoint_db_path: str = Field(
        default="data/checkpoints.db",
        description="Path to the SQLite database file for LangGraph SqliteSaver checkpoints",
    )
    checkpointing_enabled: bool = Field(
        default=False,
        description="Enable LangGraph SqliteSaver checkpointing; disabled until Phase 4 cutover",
    )

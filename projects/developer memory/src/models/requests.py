"""MCP tool request models — validated at the tool boundary before pipeline invocation.

These four request types are co-located because they all serve the same purpose:
validating and sanitizing MCP client input before it enters any LangGraph pipeline.
Each class applies security guards (URL allowlist, path traversal) at instantiation time.
"""

from pydantic import BaseModel, Field, field_validator

from src.middleware.path_guard import validate_target_path
from src.middleware.url_guard import validate_repo_url


class SyncRequest(BaseModel):
    """Validated input for the sync_repository MCP tool.

    Args:
        repo_url: Git repository URL — must be on the ALLOWED_HOSTS allowlist (§6.3a).
        branch: Branch to sync. Defaults to "main".
    """

    repo_url: str
    branch: str = "main"

    @field_validator("repo_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Allowlist-based validation — scheme check alone is insufficient against injection."""
        return validate_repo_url(v)


class QueryRequest(BaseModel):
    """Validated input for the query_memory MCP tool.

    Args:
        query: Natural-language query string — further validated by query_guard node (§6.2).
        tech_filter: Optional list of tech stack labels to narrow the ChromaDB search.
    """

    query: str
    tech_filter: list[str] | None = None


class DiffRequest(BaseModel):
    """Validated input for the analyze_diff MCP tool.

    Args:
        diff_text: Unified diff text — further validated by diff_validator node (§6.3).
    """

    diff_text: str


class SkillsRequest(BaseModel):
    """Validated input for the generate_skills_pkg MCP tool.

    Args:
        target_path: Write destination for PROJECT_SKILLS.md — must resolve under
            SKILLS_EXPORT_DIR to prevent path traversal (§6.3b).
    """

    target_path: str = Field(description="Destination path for PROJECT_SKILLS.md")

    @field_validator("target_path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Guard against path traversal before write — raises ValueError with safe message."""
        return validate_target_path(v)

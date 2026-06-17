"""Application settings loaded from environment / .env file.

Spec §8 — pydantic-settings reads OLLAMA_MODEL, CHROMA_COLLECTION, PARSER_WORKERS,
and SKILLS_EXPORT_DIR from the environment. Field names map to env vars via automatic
uppercasing (ollama_model → OLLAMA_MODEL).

F6 rework: Settings now owns CHROMA_HOST and CHROMA_DATA_PATH so that URL-parse logic
lives in exactly one place. ChromaLibrarianClient reads from Settings() instead of
calling os.environ directly.
"""

from urllib.parse import urlparse

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for Developer Memory MCP server.

    All fields are read from environment variables (or .env file). The field name
    is uppercased to form the env var name: ollama_model → OLLAMA_MODEL.

    Args:
        ollama_model: Ollama model tag. "gemma4:26b" when RAM ≥ 16 GB; overridden to
            "gemma4:e4b" at runtime by _check_memory() when RAM < 16 GB.
        chroma_collection: ChromaDB collection name (§9 schema).
        parser_workers: Number of parallel worker processes for multimodal_parser.
            Automatically halved by _check_memory() when RAM < 16 GB.
        skills_export_dir: Root directory for skills export write guard (§6.3b).
            Empty string means Path.cwd() is used as the export root.
        github_token: Optional GitHub personal access token for private repo access.
        chroma_host: Full ChromaDB URL (e.g. "http://chromadb:8000"). Empty string
            means local PersistentClient mode using chroma_data_path.
        chroma_data_path: Local filesystem path for ChromaDB PersistentClient when
            chroma_host is empty. Ignored when chroma_host is set.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_model: str = "gemma4:26b"
    ollama_skills_model: str = ""  # Override for skills synthesis; falls back to ollama_model if empty
    chroma_collection: str = "developer_memory_v1"
    parser_workers: int = 6
    skills_export_dir: str = ""
    github_token: str = ""

    # GIT_CLONE_TIMEOUT_SECS: timeout in seconds for git clone subprocess. 0 = no timeout.
    git_clone_timeout_secs: int = 120
    # CACHE_FILTER_ENABLED: set False to force full re-index (e.g. after schema migration).
    cache_filter_enabled: bool = True
    # SHA_FRESHNESS_ENABLED: set False for offline/local repos or to disable ls-remote check.
    sha_freshness_enabled: bool = True

    # F6: ChromaDB connection — owned here so URL-parse logic lives in one place only.
    chroma_host: str = ""          # Full URL e.g. "http://chromadb:8000"; empty = local mode
    chroma_data_path: str = "chroma_data"  # Local path used when chroma_host is empty

    @model_validator(mode="after")
    def _validate_chroma_host(self) -> "Settings":
        """Fail fast at startup when CHROMA_HOST is set but malformed.

        Bare hostnames without a scheme (e.g. "chromadb:8000") will silently parse
        to hostname=None in urlparse — this validator rejects them with an actionable
        message before any connection is attempted.

        Returns:
            Self (unchanged) when chroma_host is empty or a valid HTTP/HTTPS URL.

        Raises:
            ValueError: When chroma_host is non-empty but not a valid HTTP URL.
        """
        host = self.chroma_host
        if not host:
            return self
        parsed = urlparse(host)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(
                f"CHROMA_HOST must be a full HTTP URL (e.g. 'http://chromadb:8000'). "
                f"Got: {host!r}. Bare hostnames without 'http://' are not supported."
            )
        return self

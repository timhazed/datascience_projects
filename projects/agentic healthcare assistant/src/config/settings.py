"""Settings — root configuration loaded from config.yaml.

Usage:
    from src.config import load_settings, Settings
    settings = load_settings()
    settings = Settings.load("path/to/config.yaml")

    # Equivalent:
    from src.config.settings import load_settings
"""

from pathlib import Path

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from src.config.db_config import DBConfig
from src.config.embeddings_config import EmbeddingsConfig
from src.config.graph_config import GraphConfig
from src.config.provider_config import ProviderConfig
from src.config.search_config import SearchConfig


class Settings(BaseSettings):
    """Root settings model. Loads from config.yaml; env vars may override scalar fields.

    The provider dict must contain exactly one key ("groq" or "openai") — this key
    determines which LLM factory path get_llm() uses at graph construction time.
    """

    provider: dict[str, ProviderConfig] = Field(
        description="Exactly one provider key ('groq' or 'openai') with its ProviderConfig"
    )
    search: SearchConfig = Field(description="Web search provider selection")
    embeddings: EmbeddingsConfig = Field(
        default_factory=EmbeddingsConfig,
        description="Embeddings provider and model for FAISS",
    )
    db: DBConfig = Field(description="SQLite and FAISS file-system paths")
    graph: GraphConfig = Field(
        default_factory=GraphConfig,
        description="LangGraph execution limits and related options",
    )
    
    model_config = {"arbitrary_types_allowed": True}

    _VALID_PROVIDERS = frozenset({"groq", "openai"})

    @model_validator(mode="after")
    def _validate_exactly_one_provider(self) -> "Settings":
        """Enforce exactly one valid provider key ('groq' or 'openai')."""
        if len(self.provider) != 1:
            raise ValueError(
                "Exactly one provider must be configured ('groq' or 'openai'), "
                f"got {len(self.provider)}: {list(self.provider)}"
            )
        unknown = set(self.provider) - self._VALID_PROVIDERS
        if unknown:
            raise ValueError(
                f"Unknown provider(s): {unknown}. Expected one of {self._VALID_PROVIDERS}"
            )
        return self

    @property
    def provider_name(self) -> str:
        """Return the active provider name ('groq' or 'openai')."""
        return next(iter(self.provider))

    @property
    def provider_config(self) -> ProviderConfig:
        """Return the active ProviderConfig instance."""
        return self.provider[self.provider_name]

    @classmethod
    def load(cls, config_path: str | Path = "config.yaml") -> "Settings":
        """Load Settings from a YAML file.

        Args:
            config_path: Path to the YAML config file. Defaults to 'config.yaml' in cwd.

        Returns:
            Fully validated Settings instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValidationError: If required fields are missing or values are invalid.
        """
        path = Path(config_path)
        # If the path is relative and not found in cwd, resolve it relative to
        # the project root (two directories above src/config/settings.py).
        # This handles Streamlit launching from a different working directory.
        if not path.is_absolute() and not path.exists():
            path = Path(__file__).parent.parent.parent / config_path
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls(**data)


def load_settings(config_path: str | Path = "config.yaml") -> Settings:
    """Compatibility wrapper around Settings.load()."""
    return Settings.load(config_path)

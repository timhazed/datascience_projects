from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderConfig(BaseModel):
    """LLM provider configuration."""

    model: str
    temperature: float = 0.3
    max_tokens: int = 2048


class AgentConfig(BaseModel):
    """Per-agent configuration overrides."""

    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class ExerciseConfig(BaseModel):
    """Exercise domain thresholds."""

    min_sets_per_muscle_group: int = 18
    max_sets_per_muscle_group: int = 24
    target_rpe_min: int = 7
    target_rpe_max: int = 9


class RecoveryConfig(BaseModel):
    """Recovery domain thresholds."""

    static_hold_min_seconds: int = 30
    static_hold_max_seconds: int = 60
    modality_order: list[str] = Field(
        default_factory=lambda: ["smr", "dynamic_mobility", "static_stretching"]
    )


class WorkflowConfig(BaseModel):
    """Workflow execution settings."""

    max_retries: int = 3
    timeout_seconds: int = 120


class Settings(BaseSettings):
    """Application settings loaded from .env and config.yaml."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys from .env
    openai_api_key: str = ""

    # Optional overrides from .env
    log_level: str = "INFO"
    config_path: str = "config.yaml"

    # Loaded from config.yaml
    default_provider: str = "openai"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    exercise: ExerciseConfig = Field(default_factory=ExerciseConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)

    def model_post_init(self, __context: Any) -> None:
        """Load YAML config after .env is loaded."""
        self._load_yaml_config()

    def _load_yaml_config(self) -> None:
        """Load configuration from YAML file."""
        config_path = Path(self.config_path)

        if not config_path.exists():
            return

        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        # Update settings from YAML
        if "default_provider" in config:
            self.default_provider = config["default_provider"]

        if "providers" in config:
            self.providers = {
                name: ProviderConfig(**cfg) for name, cfg in config["providers"].items()
            }

        if "agents" in config:
            self.agents = {name: AgentConfig(**cfg) for name, cfg in config["agents"].items()}

        if "exercise" in config:
            self.exercise = ExerciseConfig(**config["exercise"])

        if "recovery" in config:
            self.recovery = RecoveryConfig(**config["recovery"])

        if "workflow" in config:
            self.workflow = WorkflowConfig(**config["workflow"])

    def get_provider_config(self, provider_name: str | None = None) -> ProviderConfig:
        """Get configuration for a specific provider."""
        name = provider_name or self.default_provider
        if name not in self.providers:
            raise ValueError(f"Unknown provider: {name}")
        return self.providers[name]

    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Get configuration for a specific agent."""
        return self.agents.get(agent_name, AgentConfig())

    def get_llm_params(self, agent_name: str) -> dict[str, Any]:
        """Get merged LLM parameters for an agent (provider defaults + agent overrides)."""
        agent_cfg = self.get_agent_config(agent_name)
        provider_name = agent_cfg.provider or self.default_provider
        provider_cfg = self.get_provider_config(provider_name)

        return {
            "provider": provider_name,
            "model": agent_cfg.model or provider_cfg.model,
            "temperature": (
                agent_cfg.temperature
                if agent_cfg.temperature is not None
                else provider_cfg.temperature
            ),
            "max_tokens": agent_cfg.max_tokens or provider_cfg.max_tokens,
        }


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

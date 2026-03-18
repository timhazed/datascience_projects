from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings
import yaml


class ProviderConfig(BaseModel):
    """Config for a single LLM provider (openai, groq, etc.)."""
    model: str = Field(default="gpt-3.5-turbo", min_length=1)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int | None = Field(default=None, description="Max tokens in response (None = model default)")


class StoreConfig(BaseModel):
    """Config for a single vector store (faiss, chroma, etc.)."""
    persist_directory: str = Field(default="data/faiss", min_length=1)


class Settings(BaseSettings):
    provider: dict[str, ProviderConfig] = Field(default_factory=dict)
    store: dict[str, StoreConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_exactly_one(self) -> "Settings":
        if len(self.provider) != 1:
            raise ValueError(
                "Exactly one provider must be configured (openai OR groq), "
                f"got {len(self.provider)}: {list(self.provider)}"
            )
        if len(self.store) != 1:
            raise ValueError(
                "Exactly one store must be configured (faiss OR chroma), "
                f"got {len(self.store)}: {list(self.store)}"
            )
        return self

    @property
    def provider_name(self) -> str:
        return next(iter(self.provider))

    @property
    def provider_config(self) -> ProviderConfig:
        return self.provider[self.provider_name]

    @property
    def store_name(self) -> str:
        return next(iter(self.store))

    @property
    def store_config(self) -> StoreConfig:
        return self.store[self.store_name]


def load_settings(config_path: str = "config.yaml") -> Settings:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return Settings(**config)
"""EmbeddingsConfig — embedding model provider and model name."""

from typing import Literal

from pydantic import BaseModel, Field


class EmbeddingsConfig(BaseModel):
    """Configuration for the embeddings model used by the FAISS patient vector store."""

    provider: Literal["openai"] = Field(
        default="openai",
        description="Embeddings provider — fixed to 'openai' (text-embedding-3-small)",
    )
    model: str = Field(
        default="text-embedding-3-small",
        description="Embeddings model name; OPENAI_API_KEY always required",
    )

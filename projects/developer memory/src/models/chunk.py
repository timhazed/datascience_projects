"""Chunk models for the Developer Memory ingest pipeline.

ParsedChunk → SummarizedChunk → UpsertResult is the ingest processing pipeline.
These three types are co-located because they form a single cohesive processing unit.
"""

import hashlib
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class ParsedChunk(BaseModel):
    """A raw text chunk produced by multimodal_parser from a sanitized source file.

    Args:
        path: Source file path — used to route by extension and preserved in metadata.
        content: PII-sanitized chunk text, ready for LLM summarization.
    """

    path: str
    content: str


class SummarizedChunk(BaseModel):
    """Structured output from intent_summarizer — the LLM-generated rationale for a chunk.

    This model is used as the structured output target for with_structured_output().
    Field description= strings are injected into the JSON Schema that Gemma 4 receives
    as its output contract. content_hash is computed deterministically from content
    for ChromaDB idempotency.

    Args:
        content: The original PII-sanitized chunk text (passed through from ParsedChunk).
        path: Source file path (passed through for ChromaDB file_path metadata).
        intent_summary: Gemma 4-generated rationale explaining *why* this code exists.
        tech_stack: Detected technology/framework identifiers (e.g. ["FastAPI", "Pydantic"]).
        semantic_type: Chunk classification used for query filtering and persona aggregation.
        key_identifiers: Verbatim class/function/API names for code search recall.
        author_identity: Developer identity from git blame/commit metadata; defaults to "unknown"
            when not available at summarization time.
    """

    content: str
    path: str
    intent_summary: str = Field(
        description=(
            "Explain *why* this code exists and what developer intent it encodes. "
            "Focus on the rationale and design decision, not a description of what the code does. "
            "2–4 sentences."
        ),
    )
    tech_stack: list[str] = Field(
        default_factory=list,
        description=(
            "Technology and framework identifiers detected in this chunk, "
            "e.g. ['FastAPI', 'Pydantic', 'LangChain']. Empty list if none detected."
        ),
    )
    semantic_type: Literal["Logic", "Config", "Boilerplate", "Interface", "quarantine"] = Field(
        default="Logic",
        description=(
            "Classify the chunk: "
            "Logic = algorithmic/business logic code; "
            "Config = configuration, settings, constants; "
            "Boilerplate = scaffolding, generated code, imports; "
            "Interface = API boundaries, protocol definitions, abstract classes."
        ),
    )
    key_identifiers: list[str] = Field(
        default_factory=list,
        description=(
            "Verbatim class, function, and API names extracted from this chunk for code search recall. "
            "e.g. ['AssistantAgent', 'UserProxyAgent', 'initiate_chat']."
        ),
    )
    author_identity: str = "unknown"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        """SHA-256 of content — serves as the ChromaDB document id for idempotency."""
        return hashlib.sha256(self.content.encode()).hexdigest()


class UpsertResult(BaseModel):
    """Outcome of a single ChromaDB upsert operation, returned by chroma_upsert.

    Args:
        content_hash: SHA-256 of the upserted content — the ChromaDB document id.
        file_path: Source file path for traceability.
        action: Whether the document was inserted (new), skipped (duplicate), or
            updated (file moved — metadata-only update per §9 idempotency contract).
        error: Non-None when the upsert failed; pipeline continues but surfaces the error.
    """

    content_hash: str
    file_path: str
    action: Literal["inserted", "skipped", "updated", "error"]
    error: str | None = None

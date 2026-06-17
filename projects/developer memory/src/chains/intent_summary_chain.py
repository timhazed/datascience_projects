"""Chain for intent summarization — Spec §5, §7.

Builds the LangChain Runnable used by intent_summarizer to generate structured
annotations for a single ParsedChunk. Returns a _ChunkAnnotation (LLM-generated
fields only). The node assembles the full SummarizedChunk by combining the annotation
with the pass-through content and path — the LLM never echoes back input data.

Structured output target: _ChunkAnnotation (internal — tightly coupled to this chain).
The node builds SummarizedChunk from the annotation + chunk.content + chunk.path.
"""

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Internal structured output model ─────────────────────────────────────────
# Tightly coupled to this chain — only intent_summarizer uses it.
# Co-located here per the "tightly related small types" exception.


class _ChunkAnnotation(BaseModel):
    """LLM-generated annotation fields for a ParsedChunk.

    Used as the with_structured_output target. content and path are pass-through
    from ParsedChunk and are NOT included here — the LLM should not echo input data.
    """

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
    semantic_type: Literal["Logic", "Config", "Boilerplate", "Interface"] = Field(
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
            "Concrete identifiers present in this chunk that a developer might search for: "
            "class names, function names, method names, and framework-specific API names. "
            "Copy them verbatim, case-sensitive — do not paraphrase. "
            "e.g. ['AssistantAgent', 'UserProxyAgent', 'initiate_chat', 'PredictionAgent']."
        ),
    )


# ── Prompt template ──────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a code intelligence analyst specialising in developer intent. "
    "Your job is to infer *why* a piece of code was written — the rationale, design decision, "
    "or problem it solves — not to describe what the code does syntactically. "
    "Identify the technologies and frameworks present, classify the chunk semantically, "
    "and extract every concrete identifier (class names, function names, API names) verbatim from the code.\n\n"
    "Example:\n\n"
    "File: src/auth/session.py\n"
    "```python\n"
    "def validate_token(token: str):\n"
    "    # Check expiry to prevent stale session hijacking\n"
    "    return redis_client.get('auth_' + token)\n"
    "```\n\n"
    "intent_summary: Validates session tokens against a Redis cache to prevent stale session hijacking.\n"
    "tech_stack: [\"Python\", \"Redis\"]\n"
    "semantic_type: Logic\n"
    'key_identifiers: ["validate_token", "redis_client", "get"]'
)

_HUMAN = (
    "File: {path}\n\n"
    "```\n{content}\n```\n\n"
    "Annotate this code chunk following the format in the system example.\n\n"
    "For key_identifiers: scan the code and list every class name, function name, method name, "
    "and third-party API name verbatim — copy them exactly as they appear (case-sensitive). "
    "Do NOT paraphrase or omit PascalCase names like AssistantAgent or UserProxyAgent."
)

_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


def build_intent_summary_chain(llm: ChatOllama) -> Runnable:
    """Build the intent summarization chain — called once at graph construction time.

    The returned chain accepts {"content": str, "path": str} and returns a
    _ChunkAnnotation with LLM-generated intent_summary, tech_stack, semantic_type,
    and key_identifiers. The caller (intent_summarizer node) assembles the full SummarizedChunk.

    Args:
        llm: ChatOllama instance constructed at server startup and injected here.
            Must not be reconstructed per invocation.

    Returns:
        A LangChain Runnable: dict → _ChunkAnnotation.
    """
    # with_structured_output enforces the JSON schema at inference time.
    # Bind num_predict=1024 per §5: preamble + JSON output budget for gemma4:26b.
    structured_llm = llm.with_structured_output(_ChunkAnnotation)
    # External data (content, path) go into the human turn — never the system prompt.
    return _PROMPT | structured_llm

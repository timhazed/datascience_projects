"""LLM singleton factory for all Developer Memory pipelines.

Spec §5 — ChatOllama is constructed once per (model, temperature, num_ctx) combination
via lru_cache and injected into node factories at graph construction time. It is never
reconstructed inside a node function that runs per request.

Usage at server startup (src/server.py):
    _model = _check_memory()
    _llm_exact    = get_llm(model=_model, temperature=0.0, num_ctx=8192)
    _llm_light    = get_llm(model=_model, temperature=0.1, num_ctx=8192)
    _llm_creative = get_llm(model=_model, temperature=0.3, num_ctx=8192)
    _llm_balanced = get_llm(model=_model, temperature=0.2, num_ctx=16384)
"""

import os
from functools import cache

from langchain_ollama import ChatOllama


@cache  # cache key = (model, temperature, num_ctx, num_predict); maxsize=1 would evict
def get_llm(
    model: str = "gemma4:26b",
    temperature: float = 0.0,
    num_ctx: int = 8192,
    num_predict: int = 2048,
) -> ChatOllama:
    """Return a cached ChatOllama instance keyed by (model, temperature, num_ctx, num_predict).

    Reads OLLAMA_HOST from the environment (default: http://localhost:11434).
    In Mode A hybrid deployment, containers set OLLAMA_HOST=http://host.docker.internal:11434
    to reach the native macOS Ollama process. The experiment and the MCP server
    both use the same factory, so both resolve the same host.

    maxsize=None ensures each unique argument combination is cached independently.
    num_ctx must be set at construction time — ChatOllama does not expose it as a
    per-call parameter. Nodes that require a larger context window (e.g. skills_synthesizer
    at 16384) must request their own singleton at startup rather than sharing the default.
    num_predict caps max output tokens; skills_synthesizer uses 4096 to avoid repetition loops.

    Args:
        model: Ollama model tag, e.g. "gemma4:26b" or "gemma4:e4b".
        temperature: Sampling temperature. 0.0 for deterministic structured output nodes;
            higher for creative synthesis nodes (see §5 per-node table).
        num_ctx: Context window size in tokens. Default 8192. skills_synthesizer uses 16384.
        num_predict: Max tokens to generate. Default 2048. skills_synthesizer uses 4096.

    Returns:
        A ChatOllama instance — the same object on every call with identical args.
    """
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return ChatOllama(
        model=model,
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=num_ctx,
        base_url=base_url,
        request_timeout=120.0,  # 2-minute hard deadline per chunk — prevents infinite stalls
    )

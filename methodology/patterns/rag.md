# RAG Pattern

*This document reflects current thinking on designing retrieval-augmented generation systems. Updated as new retrieval strategies and production failure modes are identified through project work.*

---

## What RAG Is

Retrieval-Augmented Generation is an architecture pattern that grounds LLM responses in a specific document corpus. Rather than relying solely on the model's parametric knowledge, RAG retrieves relevant passages from a controlled document set and provides them as context for generation.

**Core value:** Responses are grounded in verifiable source material, reducing hallucination and enabling domain-specific accuracy that a general model cannot achieve through training alone.

**Core limitation:** The quality of every response is bounded by the quality of retrieval. A RAG system with poor retrieval produces confident responses based on irrelevant context.

---

## When to Use RAG

Use RAG when:
- Responses must be grounded in a specific, controlled document corpus
- The corpus changes more frequently than the model can be fine-tuned
- Responses must be traceable to source documents
- The domain is narrow enough that a well-curated corpus can cover it adequately

Do not use RAG when:
- The task requires general reasoning across the model's full training knowledge
- The corpus cannot be curated to sufficient quality (garbage in, garbage out)
- Latency requirements cannot accommodate a retrieval step
- Fine-tuning on a stable, high-quality dataset would produce better results at lower operational complexity

See [workflows/decision_guide.md](../workflows/decision_guide.md) for the full decision framework.

---

## System Architecture

```
Documents
    ↓
[Loader]          — load documents from source
    ↓
[Chunker]         — split into retrievable units
    ↓
[Embedder]        — encode chunks as dense vectors
    ↓
[Vector Store]    — index for similarity search

                        ←──── Query
[Retriever]       — find relevant chunks
    ↓
[Context Builder] — assemble prompt with retrieved chunks
    ↓
[LLM]             — generate grounded response
    ↓
[Guardrail]       — validate and safety-check output
    ↓
                        Response ────→
```

---

## Document Loading

Define the loader strategy per source type:

| Source | Loader | Preprocessing |
|--------|--------|---------------|
| PDF | PyPDFLoader or pdfplumber | Page extraction, header/footer removal |
| Plain text | TextLoader | Encoding normalization |
| Web | WebBaseLoader | HTML stripping |
| Structured (CSV, JSON) | CSVLoader / JSONLoader | Field selection |

**Metadata extraction:** Define what metadata to extract per document at load time. At minimum: source file, page number, and ingestion date. Metadata that is not extracted at load time is not available at retrieval time.

---

## Chunking Strategy

Chunking is the most consequential design decision in a RAG system. Chunk size determines what information is co-located in a single retrieval unit.

**Fixed-size chunking**

Split on token count with overlap. Use when documents are uniform in structure.

- `chunk_size`: 200–500 tokens for most use cases. Measure in tokens, not characters (4 characters ≈ 1 token is a rough estimate; use a tokenizer for accuracy).
- `chunk_overlap`: 10–20% of chunk size. Overlap preserves context at boundaries.

**Paragraph-aware chunking**

Split on structural boundaries (double newlines, headings). Use when documents have meaningful paragraph structure — technical documentation, legal text, policies.

**Semantic chunking**

Embed first, split on semantic boundaries. Use when information density is high and fixed-size boundaries cut through related content. Computationally expensive; justified when retrieval precision is critical.

**Choosing chunk size:**

Smaller chunks: higher retrieval precision, lower context density per chunk. Risk: relevant information is split across chunk boundaries.

Larger chunks: more context per chunk, but lower retrieval precision (chunks contain more irrelevant content). Risk: relevant information is diluted by surrounding context.

Evaluate both directions against your actual document corpus. The right chunk size is empirically determined, not set by convention.

---

## Embedding Model

| Option | Use When | Dimension |
|--------|----------|-----------|
| OpenAI `text-embedding-3-small` | Moderate quality, low cost | 1536 |
| OpenAI `text-embedding-3-large` | High quality, higher cost | 3072 |
| HuggingFace `all-MiniLM-L6-v2` | Local, low cost, moderate quality | 384 |
| HuggingFace `BAAI/bge-large-en` | High quality, local | 1024 |

The embedding model used for indexing and the model used for query embedding must be the same. Mismatched models produce meaningless similarity scores.

---

## Vector Store Selection

| Option | Use When | Persistence |
|--------|----------|-------------|
| FAISS | Local development, < 100k documents | File-based |
| ChromaDB | Local development, metadata filtering needed | SQLite or HTTP |
| Qdrant | Production, scale, advanced filtering | Docker or Cloud |
| Pinecone | Managed production, no self-hosting | Cloud |

**Access pattern:** The vector store is accessed through an abstraction layer — a factory or interface. Switching stores should require a configuration change, not a code rewrite.

**Rebuild vs. incremental update:** Define this strategy before deployment. Most vector stores support both; the choice depends on document update frequency and index size.

---

## Retrieval

**Similarity search** — returns the top-k chunks by cosine similarity. Simple and effective for most use cases.

**Maximum Marginal Relevance (MMR)** — balances relevance and diversity. Reduces redundancy when multiple chunks from the same document are highly similar to the query.

**Hybrid search** — combines dense (embedding) and sparse (BM25/keyword) retrieval. Improves recall on queries where keyword matching outperforms semantic similarity.

**Relevance score threshold:** Set a minimum similarity score below which chunks are discarded. Returning low-relevance chunks is worse than returning nothing — the LLM will use them and produce context-contaminated responses.

Calibrate the threshold against your actual query distribution. A threshold set too high produces empty context; too low allows noise.

---

## Context Assembly

Retrieved chunks are assembled into the prompt in a defined order. Options:

- **Chronological** — by document date. Use when temporal order matters.
- **By relevance score** — highest-scoring chunks first. The LLM attends more strongly to early context.
- **By source** — group chunks from the same document. Use when document-level coherence matters.

Set a hard limit on the number of chunks included in a single prompt. Calculate the maximum token consumption of the context block at this limit, and verify it fits within the context window along with the system prompt, query, and output budget.

---

## Graceful Degradation

A RAG system must handle retrieval failure gracefully:

- **No chunks above threshold:** Return a response that acknowledges the limitation honestly. Do not generate a response based on parametric knowledge when the system is designed to be grounded.
- **Retrieval timeout:** Surface a specific error. Do not silently return an empty context.
- **Vector store unavailable:** Fail fast with a clear error. Do not attempt generation without context in a grounded system.

---

## Common Failure Modes

**Chunking breaks meaningful units.** A fixed-size chunk splits a question from its answer, or a procedure from its prerequisites. Retrieval returns partial context and the LLM fills the gap with hallucination.

**No score threshold.** Chunks with very low similarity scores are included. The LLM receives irrelevant context and generates a response that uses it.

**Context window overflow.** Too many chunks + long system prompt + long query exceeds the context window. The model silently truncates early chunks — often the most relevant ones.

**Embedding model mismatch.** Index was built with one model; queries use another. Similarity scores are meaningless.

**Missing metadata.** Source attribution, page numbers, or dates were not captured at load time. The system cannot provide citations or filter by document property.

**Retrieval evaluated only end-to-end.** High end-to-end scores mask retrieval failures where the LLM compensates with parametric knowledge. Evaluate retrieval precision and recall independently.

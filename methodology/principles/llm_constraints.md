# LLM Constraints

*This document reflects current understanding of the structural constraints that LLMs impose on system design. These constraints are not bugs to work around — they are properties of the technology that must be designed for explicitly. Updated as model capabilities evolve and new constraint patterns are identified in production.*

---

## Why This Document Exists

Engineers who treat LLMs as deterministic, infinitely capable black boxes build systems that fail in production in ways that are difficult to diagnose. The constraints documented here are not edge cases — they are inherent properties of LLM-based systems that require explicit design attention.

Understanding these constraints is the difference between a system that is robust to expected model behavior and one that surprises its operators every week.

---

## Non-Determinism

LLMs are probabilistic. Given the same input, they do not always produce the same output. At temperature 0 most providers return consistent results, but consistency is not guaranteed — it is a statistical tendency, not a contract.

**Design implications:**
- Never assert equality on LLM outputs in tests. Test structure, type, and semantic properties.
- Systems that depend on exact output reproducibility (e.g., deterministic audit trails) cannot be built on raw LLM output without an additional validation layer.
- Evaluation must be conducted over multiple samples, not a single run. A single passing run does not establish that a system is reliable.

**Mitigation:** Use temperature 0 for structured output tasks. Use LLM-as-judge for evaluation rather than string matching. Design output schemas that constrain the space of valid responses.

---

## Token Limits

Every LLM has a context window — a maximum number of tokens it can process in a single call, spanning both input and output. Exceeding this limit causes hard failures or silent truncation depending on the provider.

**Input token budget:**

The prompt, system message, conversation history, retrieved context, and format instructions all consume input tokens. At maximum input conditions (large documents, long conversation history, many retrieved chunks), the total must fit within the context window with room for the output.

Calculate the maximum input size explicitly. Do not discover the limit in production.

**Output token budget:**

`max_tokens` limits the response length. A response that hits the limit mid-output is truncated. For structured outputs — JSON, XML, Pydantic models — partial responses are invalid and unparseable.

Calculate `max_tokens` from output structure:

```
max_tokens = (avg_fields × avg_tokens_per_field × num_items) × 1.3
```

Never set `max_tokens` to an arbitrary round number without showing this calculation.

**Context window management strategies:**

| Strategy | Use When |
|----------|----------|
| Fixed window | Input is bounded and fits comfortably |
| Sliding window | Conversation history; keep most recent N turns |
| Summarization | Long conversation history; compress older turns |
| Chunk filtering | RAG; discard chunks below relevance threshold |

---

## Latency

LLM API calls are slow relative to traditional software operations. Typical latencies range from hundreds of milliseconds to several seconds depending on model size, provider, and output length.

**Design implications:**
- Do not place synchronous LLM calls in hot paths where response time is critical.
- In multi-agent systems, parallelize independent agent calls where the pipeline allows it.
- Set explicit timeouts on all API calls. An unresponsive API without a timeout blocks the entire pipeline.
- Communicate latency expectations to users. A UI that shows nothing for three seconds while an LLM generates a response feels broken.

**Measuring latency:**
- Measure p50, p95, and p99, not just average. LLM latency distributions are heavy-tailed.
- Measure end-to-end latency including retrieval, chain construction, and parsing — not just the API call.

---

## Cost

LLM API pricing is typically token-based. Cost scales with input length, output length, and query volume. A system that is affordable at development scale can be expensive at production scale.

**Calculate cost before deployment:**

```
cost_per_query = (input_tokens × input_price_per_token) + (output_tokens × output_price_per_token)
monthly_cost = cost_per_query × expected_monthly_queries
```

**Cost levers:**

| Lever | Effect | Trade-off |
|-------|--------|-----------|
| Smaller model | Lower cost | Lower capability |
| Shorter prompts | Lower input cost | Less instruction, potentially lower quality |
| Lower `max_tokens` | Lower output cost | Risk of truncation if set too low |
| Caching repeated inputs | Reduced API calls | Requires cache management |
| Batching | Throughput efficiency | Increased latency |

**Cost monitoring:** Instrument every LLM call with token counts. Alert on anomalous consumption. A prompt injection that causes runaway token usage will show up in cost monitoring before it shows up in quality metrics.

---

## Structured Output Reliability

LLMs do not reliably produce valid structured output without explicit mechanisms to enforce it. A prompt that says "respond in JSON" will produce JSON most of the time — not all of the time.

**Reliability by approach:**

| Approach | Reliability | Notes |
|----------|-------------|-------|
| Raw prompt instruction ("respond in JSON") | Low | No enforcement mechanism |
| Format instructions injected via parser | Medium | Improves but does not guarantee |
| Provider-native JSON mode | High | Provider-specific; verify support |
| Function calling / tool use | High | Strongly typed; provider-specific |

**Design implications:**
- Always validate structured outputs through a Pydantic model. Do not assume the model produced valid output.
- Implement a retry loop on parse failure. A single retry resolves the majority of transient structured output failures.
- Field descriptions in Pydantic schemas must be specific enough to constrain model behavior. Vague descriptions produce vague output.

---

## Prompt Sensitivity

Small changes to a prompt can produce significantly different outputs. A prompt that works reliably with one model version may degrade with the next. This sensitivity is not predictable from first principles — it must be measured.

**Design implications:**
- Treat prompt changes as code changes. Version them, review them, test them.
- When a model is upgraded by the provider, re-evaluate structured output reliability, response quality, and format compliance before releasing.
- Do not optimize prompts by intuition alone. Measure the effect of each change against an evaluation dataset.

---

## Distribution Shift

A model's behavior degrades when the inputs it receives differ from the distribution it was trained or prompted for. This can happen gradually as user behavior evolves, or suddenly when a new input type is introduced.

**Design implications:**
- Evaluation datasets must be updated periodically to reflect the current distribution of real inputs.
- Monitor production inputs for patterns that differ from the evaluation distribution.
- The LLM-as-judge layer is the most reliable signal of output quality degradation under distribution shift.

---

## Provider Dependency

LLM providers update models, change APIs, deprecate endpoints, and experience outages. A system tightly coupled to a specific provider is exposed to all of these risks.

**Design implications:**
- Abstract the LLM provider behind an interface. Swapping providers should require a configuration change, not a code rewrite.
- Test against provider API changes before they reach production.
- Have a fallback plan for provider outages: a secondary provider, degraded behavior, or a graceful failure message.

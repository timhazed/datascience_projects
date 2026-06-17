# Output Quality Evaluation — Final Findings
**Date:** 2026-05-01  
**Eval script:** `experiments/eval_output_quality.py`  
**Corpus:** 24 cases across 5 chain types  
**Runs per case:** 2  
**Judge:** Groq `openai/gpt-oss-120b`, `reasoning_effort=medium`, evidence-based rubric  
**Reference results:** `quality_20260501_155658.json` (e4b), `quality_20260501_163334.json` (26b)

---

## Decision Outcome

**The original e4b recommendation is reversed.**

The 16.7pp compliance gap that favoured e4b in earlier runs was entirely caused by a
`num_predict` starvation bug: `intent_summarizer` was capped at 512 tokens, which 26b's
thinking preamble exhausted before any JSON was emitted. With the cap raised to 1024,
both models achieve **100% compliance across all chains**.

The decision is now quality-based, not compliance-based.

---

## Compliance (clean runs — all fixes applied)

| Chain | e4b | 26b |
|---|---|---|
| intent_summarizer | 100% | 100% |
| snippet_summarizer | 100% | 100% |
| persona_synthesizer | 100% | 100% |
| coaching_analyzer | 100% | 100% |
| skills_synthesizer | 100% | 100% |

**Structured chains overall:** both 100% — gap 0.0pp.

---

## Rubric Quality (LLM-as-judge, all chains)

| Chain | e4b | 26b | Delta |
|---|---|---|---|
| intent_summarizer | 60% | **79%** | 26b +19pp |
| snippet_summarizer | 60% | **67%** | 26b +7pp |
| persona_synthesizer | 42% | **57%** | 26b +15pp |
| coaching_analyzer | **67%** | 53% | e4b +14pp |
| skills_synthesizer | **78%** | 70% | e4b +8pp |

26b produces higher-quality output on the synthesis and summarisation chains
(intent, snippet, persona). e4b leads on the analytical reasoning chains
(coaching, skills).

---

## Throughput

Both models run at **~75 tok/s sustained** on M4 Max — bandwidth-bound, not
compute-bound. No meaningful throughput difference. 26b's higher per-case wall-clock
time on `intent_summarizer` (~8.4s vs ~1.9s for e4b) is the thinking preamble
consuming the extra token budget, not slower generation.

---

## Recommendation by Use Case

| Priority | Recommendation | Rationale |
|---|---|---|
| High-volume ingest (`intent_summarizer`) | **26b** | +19pp rubric — summarises WHY code exists, not just WHAT it does |
| Developer persona synthesis | **26b** | +15pp rubric — more coherent archetypes, less fabrication |
| Coaching feedback quality | **e4b** | +14pp rubric — more specific rationales |
| Resource-constrained deployment | **e4b** | Smaller footprint, same throughput, competitive on coaching |

If a single model must be chosen: **26b** for quality-first deployments,
**e4b** for resource-constrained or coaching-primary deployments.

---

## Key Bugs Fixed During This Session

### 1. `num_predict` starvation (critical — invalidated earlier results)
`intent_summarizer` was capped at 512 tokens. 26b's thinking preamble consumed the
entire budget, producing empty responses and 0% compliance. Raised to 1024.
**This was the root cause of the original e4b recommendation.**

### 2. `<think>` block stripping
26b prepends `<think>...</think>` reasoning tokens before JSON output. The validator
now strips these via regex before `json.loads`, then seeks to the first `{` for any
remaining preamble.

### 3. Section-header parser bug in heuristic scorer
Multi-`##` criteria (e.g. `"output contains ## Tech Stack, ## Patterns, ## Tendencies"`)
broke the extractor, producing 0% rubric scores on `skills_synthesizer` even when
sections were present. Fixed to extract only leading title-case words after `##`.

### 4. All chains moved to LLM-as-judge
Heuristic scorer cannot evaluate negative assertions, numeric thresholds, or semantic
criteria reliably. All five chains now use Groq judge; heuristic is fallback only.

### 5. Stricter judge prompt with evidence requirement
Judge now requires a direct quote as evidence for every PASS verdict. Numeric threshold
criteria and negative assertions are explicitly called out in the rules.

### 6. `response_format` + `reasoning_effort=high` conflict
Groq returns 400 `json_validate_failed` on long inputs with both set. Resolved by
using `reasoning_effort=medium` which stays within context limits.

### 7. Groq retry logic
Exponential backoff (5 attempts) on HTTP 429, 500, 503, 504 and `TimeoutException`.
Respects `retry-after` header.

### 8. `skills_synthesizer` system prompt
Added explicit 400-word minimum and "WHY not just WHAT" per-bullet instruction.
Model was consistently producing ~325 words without it.

### 9. Rolling failures JSONL log + `--cases` CLI flag
Per-case failures with fail counts and full response written to
`results/failures_<timestamp>.jsonl` after each case. `--cases` enables single-case
targeted reruns for isolating fixes.

---

## Persistent Corpus/Rubric Issues

These fail on both models every run. The rubric criteria ask for information absent
from the input or are structurally unevaluable. **Fix in corpus, not model prompts.**

| Case | Criterion | Root cause |
|---|---|---|
| `intent_sum_001` | `tech_stack includes Stripe/payments/financial` | Model outputs `"Payment processing"` — rubric too literal |
| `intent_sum_004` | `semantic_type is Config or Logic` | Model outputs `Boilerplate` — valid classification, enum too narrow |
| `snippet_sum_001` | `quarantine-not-drop philosophy` | Philosophy not derivable from mechanics-only snippets |
| `snippet_sum_002` | `operator.add reducer role` | `operator.add` not present in provided snippets |
| `snippet_sum_004` | `graceful degradation` | Context not in snippets |
| `coaching_001/003` | `names sovereignty / PII risk` | Sovereignty context absent from persona JSON |
| `skills_syn_001/002` | 400-word minimum | Model still ~380 words after prompt fix; needs another iteration |

---

## Genuine Model Gaps (both models)

| Case | Gap |
|---|---|
| `intent_sum_002/005/006–008` | Abstracts tech names (`"Configuration Management"`) instead of citing specifics (`ChromaDB`, `Ollama`) |
| `persona_syn_002/003` | Fabricates patterns not in tendency data; omits sample-size confidence qualifier |
| `coaching_001/003` | Rationales correct in direction but never name the specific violation (sovereignty, PII) |
| `coaching_004–006` | Rationales vague — state the what but not the why (e.g. why `isinstance` beats string comparison) |

---

## Next Steps

1. Fix corpus rubric criteria for the unanswerable cases listed above.
2. Enrich `coaching_001/003` persona JSON with data sovereignty context so the model
   can reason about it.
3. Investigate `persona_syn_002/003` corpus inputs — determine if minimal tendency
   data is deliberate (hallucination resistance testing) or needs enriching.
4. Run full `--runs 5` dual-model eval once corpus fixes are in to get stable scores.

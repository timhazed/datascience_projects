# Case Study: Agentic Healthcare Assistant

**Project:** [Agentic Healthcare Assistant](../../projects/agentic%20healthcare%20assistant/)
**Completed:** April 2026

---

## Problem

Healthcare administration requires coordinating at least four distinct operations from a single patient encounter: booking appointments, retrieving patient history, updating records, and researching current treatment guidelines. Each lives in a different system. Staff context-switch across all of them to answer one patient question.

Existing AI tools compound this rather than solve it. A chatbot answers a factual question but cannot book an appointment. A scheduling system finds open slots but knows nothing about the patient's conditions. A search engine returns clinical guidelines but cannot cross-reference them against a specific patient's current medications.

The goal was a single conversational interface that accepts a natural-language query, decomposes it into an ordered plan, dispatches each sub-task to the correct tool, and returns one synthesized response.

---

## Success Criteria

Defined upfront in Architecture.md before any code was written:

| Criterion | Threshold |
|-----------|-----------|
| Appointment booking success rate | >= 90% of valid slot requests |
| Disease search relevance (RAGAS faithfulness) | >= 0.80 |
| History retrieval fidelity | All structured fields round-trip losslessly |
| Graph execution latency P95 (Groq) | < 8s for single-tool path |
| Test coverage | >= 85% overall; >= 90% on tool and chain modules |

---

## Architecture Decisions

Each decision below was a deliberate tradeoff. None of them were defaults.

### 1. Intent guard before any tool runs

Every query passes through an LLM-as-judge node before the planner sees it. The guard classifies the input as SAFE or UNSAFE. On UNSAFE, the graph routes to END immediately; no tool is invoked and no patient data is touched.

The alternative was embedding content filtering inside each tool node. That approach would require every node to implement its own guardrails and would still allow the planner to run and decompose adversarial inputs into sub-goals before anything blocked them. Putting the guard first meant one enforcement point, tested once, with a single failure mode.

### 2. Sequential task dispatch, not parallel

The planner decomposes a query into an ordered queue of sub-goals. The graph processes them one at a time. This was a conscious choice against async fan-out.

The reason is dependency. A query like "book a nephrologist for my father and summarize his treatment options" requires patient resolution before booking (needs patient_id), and history retrieval before synthesis (needs the patient context). Most multi-step clinical queries have this shape. Parallel dispatch would require explicit dependency tracking between sub-goals. Sequential dispatch makes the dependency implicit in the ordering the planner assigns, which the LLM can reason about naturally.

The cost is latency. The benefit is that every node receives a fully resolved state from prior nodes without coordination logic.

### 3. Asymmetric LLM usage

Not every node uses an LLM. This was the most important structural decision in the graph.

| Node | LLM? | Reason |
|------|------|--------|
| `intent_guard` | Yes | Classification requires language understanding |
| `planner` | Yes | Query decomposition is open-ended |
| `patient_resolver` | No | SQLite fuzzy name match; deterministic |
| `history_retriever` | Yes | Synthesizes across structured fields and FAISS chunks |
| `history_writer` | No | SQLite update + FAISS upsert; no generation needed |
| `appointment_node` | No | Pure slot query and insert; correctness requires determinism |
| `disease_search_node` | Yes | Synthesizes across Serper + Medline results |
| `summarizer` | Yes | Composes all task outputs into a coherent response |

Appointment booking and patient resolution were deliberately kept LLM-free. Booking correctness is binary: either the right slot was claimed atomically or it was not. Introducing an LLM into that path adds latency, token cost, and a failure mode (malformed output) for a task that does not need generation. Experiment 3 confirmed this: sub-2ms latency, 4/4 scenarios, zero API calls.

### 4. Per-node token budgets derived from first principles

Token limits were not defaulted to 512 or 1024 uniformly. Each node's budget was calculated from the expected output structure:

- Planner: 5 sub-goals x 68 tokens/sub-goal + 45 wrapper tokens = ~385 tokens, x1.3 headroom = 1024. On Groq reasoning models (`gpt-oss-120b`), internal reasoning trace consumes part of the budget before visible output; values below 1024 truncate the tool JSON mid-structure.
- History summarizer: 10 fields x 30 tokens/field = 300, x1.3 = 400.
- Disease search summary: 8 snippets x 60 tokens/snippet = 480, x1.3 = 650.

This mattered in practice. Early runs with a uniform 512 token limit caused the planner to truncate multi-step PlannerOutput JSON on reasoning-model configs, producing a `tool_use_failed` error from the Groq API. Raising the planner budget to 1024 (and 1536 in the config for reasoning-model overrides) resolved it.

### 5. Domain whitelist on search results before the LLM sees them

Disease search runs parallel Serper (Google) and Medline (NCBI PubMed) fetches. Before results reach the LLM, a `TRUSTED_MEDICAL_DOMAINS` filter deprioritizes consumer-facing content. Medline results are placed first in the combined context.

This came directly from experiment findings. The rheumatoid arthritis query in Experiment 2 consistently led with NSAIDs rather than DMARDs (methotrexate) as first-line, because Serper returns consumer-web content for broad clinical management queries. The domain filter and Medline priority ordering mitigated this in production, though the experiment script (which calls Serper directly without the filter) still shows the gap. It is a documented known limitation, not a silent failure.

---

## What Failed and Changed

### Hand-rolled conversation summary replaced by LangGraph SqliteSaver

The first implementation of conversation memory was hand-written. A rolling summary string was maintained in application memory, truncated on a character limit, and serialized to a file between sessions. It worked in isolation.

The problems appeared when the UI needed to load prior conversation history at patient selection time. The hand-rolled system required the application to read the file, deserialize the summary, reconstruct the message pairs, and hydrate the chat view. Every piece of that pipeline was custom code that could drift out of sync with the actual state the graph had produced.

The replacement was LangGraph `SqliteSaver` checkpointing with `thread_id = patient_id`. The checkpoint became the single source of truth. The UI reads the `messages` channel directly from the checkpoint at patient selection. The rolling summary is written to `state["conversation_summary"]` by the summarizer node at a configurable cadence (every 3 turns by default) and persists in the checkpoint automatically. Between cadence fires, LangGraph preserves the last written value without any application code managing it.

The practical improvement: prior conversation context is available immediately when a patient is selected, without a separate load step, and without any risk of the summary diverging from the message history. A patient can be deselected and reselected and the graph state is exactly where it was left.

One subtlety that surfaced during this migration: `conversation_summary` is a `NotRequired` field in `HealthcareState`. Checkpoints created before the field was introduced do not have it. All access to the field now uses `state.get("conversation_summary", "")` rather than direct key access, which would raise a `KeyError` on older checkpoints. This is documented as an invariant in the state contract.

### Experiment 2 ran twice

The disease search experiment had a baseline run on 2026-04-09 and a follow-up on 2026-04-14. The baseline used manual QA only: 20 queries, all returned results, assessed by reading the outputs. That was not sufficient. It confirmed the tool did not crash; it did not confirm the LLM stayed within the retrieved content.

The follow-up added automated RAGAS faithfulness scoring, which measures what fraction of claims in the LLM summary are traceable to the retrieved snippets. The score was 0.960 against a 0.80 threshold. More concretely: no hallucinated treatment options were observed across 20 queries.

The follow-up also switched Medline from title-only search to `[Title/Abstract]`, recovering 2 queries that had previously returned zero Medline results (from 7 to 5 zero-result queries). The tradeoff was +377ms mean latency, from 1,660ms to 2,037ms. Both are well inside the 8s P95 threshold.

### QAEvalChain deprecated in LangChain 0.3

Experiment 1 was designed around `QAEvalChain` for grading history retrieval answers against ground truth. It was deprecated in LangChain 0.3. The replacement was `load_evaluator("qa")`, which accepts the same prediction/input/reference interface but uses a separate evaluator LLM instance constructed at temperature 0.0. The logic was equivalent; the import path and instantiation changed. No findings were affected.

---

## Experiments as Validation Gates

The three experiments were run before production code was finalized. Each was a promotion decision: does this tool's behavior meet the architecture thresholds, or does the design need to change before production use?

**Experiment 1 — Medical History (2026-04-10)**
5/5 QA pairs correct, write round-trip lossless at 6ms, mean QA latency 605ms. The write test ran before the QA cases, so the appended medication ("metformin 500mg") was immediately retrievable via FAISS re-index, confirming the full round-trip: DB write → FAISS upsert → retrieval → LLM synthesis. Promoted on first run.

**Experiment 2 — Disease Search (2026-04-09 baseline, 2026-04-14 final)**
Required iteration (see above). Final: RAGAS faithfulness 0.960, 20/20 queries, mean latency 2,037ms, P95 2,855ms. Known gap: rheumatoid arthritis query does not reliably surface DMARDs as first-line; consumer-web Serper results for broad management queries remain the root cause. Domain whitelist mitigates it in production. Promoted after second run.

**Experiment 3 — Appointment Booking (2026-04-10)**
4/4 scenarios, mean latency 1ms, no LLM calls, no API keys required. Atomic double-booking prevention confirmed end-to-end (validated separately in unit tests). Promoted on first run.

---

## Outcome

All three experiments met their architecture thresholds. The graph was promoted to production with the exact tool implementations validated in the experiments: no new capability was required between experiment and production, only extraction of the inline experiment code into production modules.

The two lasting design lessons:

**Let the evaluation method match what the tool actually does.** Appointment booking is deterministic: pure assertions are correct. History retrieval has ground truth: a string-match evaluator is correct. Disease search is a RAG pipeline where hallucination is the failure mode: RAGAS faithfulness is correct. Using the same evaluator for all three would have obscured real behavior in at least two of them.

**Checkpointing is infrastructure, not an optimization.** Hand-rolling conversation state management created a synchronization problem that did not exist in the initial single-session prototype and only appeared when the UI needed to restore prior state. Moving to SqliteSaver removed an entire class of bugs by making the graph's own checkpoint the source of truth for the UI layer.

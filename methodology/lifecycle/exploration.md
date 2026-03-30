# Exploration

*This document reflects current thinking on how to structure exploratory work so that findings are actionable and the decision to build — or not build — is made deliberately.*

---

## Purpose

Exploration is the phase where a hypothesis is validated before any production design begins. Its purpose is not to produce working software. It is to answer a specific question with enough confidence to make a build/no-build decision.

The most expensive mistake in AI development is building a production system around an assumption that a two-hour experiment would have invalidated. Exploration exists to prevent that.

---

## When to Use

Use exploration when:
- The feasibility of a core capability is uncertain (e.g., can this model produce reliable structured output for this domain?)
- Multiple approaches exist and the trade-offs are not clear without empirical data
- A new framework, model, or data source needs to be evaluated before committing to it
- A proof of concept is needed before investing in full architecture and implementation

Exploration is the right starting point for notebook-based experiments, benchmarking studies, and feasibility probes. It is not the right phase for systems that already have a clear design — those go directly to architecture.

---

## Inputs

- A single, clearly stated research question
- The success criteria that will answer it
- The constraints under which the experiment must run (cost, latency, data availability)

---

## Outputs

- Documented findings: what worked, what failed, and what surprised you
- Quantitative results where applicable (accuracy, latency, parse success rate, cost per call)
- A promotion decision: build, hold, or kill (see below)

---

## The Five Steps

### 1. State the Research Question

One sentence. Precisely scoped. Answerable.

Good: "Does chunking at 300 tokens with 15% overlap produce higher retrieval precision than 500 tokens with 10% overlap on this document set?"

Too broad: "What chunking strategy should I use?"

Unanswerable: "Is RAG good for this use case?"

If you cannot state the question in one sentence, the exploration is not ready to begin.

### 2. Define Success Criteria

Specify what a successful outcome looks like before running the experiment. This prevents post-hoc rationalization of results.

- What metric are you measuring?
- What threshold constitutes success?
- What is the minimum acceptable result?

Example: "Success = ≥ 90% parse success rate at ≤ 1.5s median latency across 50 test inputs."

### 3. Implement

Write the minimum code needed to answer the question. Keep it runnable top-to-bottom. Comment on surprising behavior as you discover it — observations made in the moment are more accurate than reconstructions after the fact.

Standards that apply even in exploration:
- Use environment variables for API keys. Never hardcode credentials.
- Timestamp results. Models and APIs change; undated results become misleading.
- Print intermediate outputs. Show your reasoning in the notebook.

Standards that are intentionally relaxed:
- Full Pydantic schema design is not required
- SOLID compliance is not enforced
- Production-grade error handling is not expected

### 4. Document Findings

Record what you learned, not just what you measured. The most valuable findings are often qualitative: unexpected model behavior, a framework limitation that changes the design, a data quality issue that invalidates the original approach.

Structure findings as:
- What worked as expected
- What failed and why
- What surprised you
- Quantitative results with context (sample size, input distribution, conditions)

### 5. Make the Promotion Decision

Every exploration ends with an explicit decision. There is no neutral outcome.

**Promote to production** — findings justify building this properly. Identify the appropriate architecture pattern and move to the architecture phase.

**Hold** — findings are promising but incomplete. Specify exactly what additional validation is needed before promotion.

**Kill** — the approach does not meet success criteria or has a fundamental limitation that cannot be designed around. Document why clearly, so the same path is not re-explored.

A kill decision is not a failure. It is the most efficient possible outcome — weeks of design and implementation time recovered before they were spent.

---

## Early Kill Criteria

These observations during exploration should trigger a kill or hold decision immediately, before completing the full experiment:

- The model cannot produce valid structured output for the domain at an acceptable success rate, even with prompt engineering
- Retrieval precision is below an acceptable threshold across all chunking strategies tested
- API cost or latency at expected query volume exceeds budget constraints
- The required data is not available, not clean enough, or not licensed for the intended use
- A core framework capability assumed in the design does not exist or does not behave as expected

---

## Common Failure Modes

**Exploring without a question.** "Playing around with the data" produces observations but not decisions. Without a defined question and success criteria, exploration has no exit condition.

**Moving to architecture on partial results.** A 70% success rate in exploration is not a green light for production. Know your threshold before you start, and hold to it.

**Not documenting what failed.** Failed experiments are as valuable as successful ones. An undocumented failure will be repeated.

**Conflating exploration with implementation.** If the notebook is growing into a production system, stop and move to the architecture phase. Exploration code that becomes production code skips every quality gate that follows.

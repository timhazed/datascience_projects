# Critique

*This document is a living record of accumulated failure modes and the structural patterns behind them. It is updated as new failure classes are discovered through project work.*

---

## Purpose

Critique is a structured adversarial review of a proposed design, conducted before any implementation begins. Its purpose is to find every way the system can fail — technically, architecturally, and operationally — while the cost of change is still low.

Most engineering teams skip this phase or conflate it with code review. Code review catches implementation errors. Critique catches design errors. These are not the same problem, and they cannot be solved with the same tool.

The output of critique is not a refined design. It is a verified design — one that has been stress-tested against a taxonomy of known failure modes and either hardened or replaced.

---

## When to Use

Critique is triggered after architecture and before implementation. It is not optional.

Additional triggers:
- A significant change to an existing system's data flow or component boundaries
- Integration of a new LLM provider, retrieval mechanism, or agent role
- Any system where incorrect output has downstream consequences (safety-critical, financial, user-facing)

---

## Inputs

- The architecture specification (component diagram, data flow, interface contracts)
- The Pydantic schemas defining data at each boundary
- The proposed chain or agent topology
- Any assumptions stated during architecture that have not been verified

---

## Outputs

- A prioritized list of verified findings, each with a structural root cause
- A set of required changes before implementation may proceed
- A list of assumptions that were verified and can be closed
- An explicit record of what was checked and how

---

## The Cardinal Rule

**Unverified critiques are noise. Verified critiques are value.**

Every finding must be grounded in evidence: a file read, a type trace, a framework behavior confirmed from source. Critiques based on assumption or memory are not findings — they are questions that need to become findings.

---

## Failure Mode Taxonomy

### LLM and Chain Failures

**Hallucination**
The model produces output that is syntactically valid but semantically incorrect. This is most dangerous in structured output scenarios where the parser succeeds but the values are wrong.
- Check: Are Field descriptions specific enough to constrain model behavior?
- Check: Is there a validation layer that catches plausible-but-wrong values, not just malformed ones?

**Malformed output**
The model returns output that fails to parse. This is common when prompt templates and parser expectations are misaligned.
- Check: Are format instructions injected correctly into the prompt?
- Check: Is there a retry loop on parse failure, or does the chain surface a raw exception?

**Retrieval failure (RAG systems)**
The retriever returns chunks that are syntactically relevant but semantically unhelpful. The LLM receives context that does not support the query.
- Check: Is chunk size calibrated to the query type?
- Check: Is there a relevance score threshold below which chunks are discarded?
- Check: Does the system degrade gracefully when retrieval returns nothing useful?

**Context window overflow**
Retrieved chunks, conversation history, and system prompt together exceed the model's context limit. The model silently truncates context, producing responses that ignore part of the input.
- Check: Has the total prompt size been calculated at maximum input conditions?
- Check: Is there a strategy for managing context at scale (compression, sliding window, summarization)?

**Token budget exhaustion**
The model truncates its response because `max_tokens` is set too low. This is especially dangerous for structured outputs where partial JSON is invalid.
- Check: Is `max_tokens` calculated from output structure, or assigned as an arbitrary round number?
- Formula: `(avg_fields × avg_tokens_per_field × num_items) × 1.3`

### Agent and Orchestration Failures

**Agent misrouting**
The routing logic sends a query to the wrong agent or tool. This is most common when intent classification is ambiguous or when agent role boundaries are not clearly defined.
- Check: Are routing conditions mutually exclusive?
- Check: Is there a fallback path when no route matches?
- Check: What does the system do when routing confidence is low?

**State corruption**
Shared mutable state between agents causes one agent's output to overwrite or contaminate another's. This is difficult to reproduce and often produces intermittent failures.
- Check: Is state passed explicitly between agents, or accessed globally?
- Check: Is each agent's input schema validated before execution?

**Infinite loops**
Conditional routing creates a cycle — agent A routes to agent B, which routes back to agent A. This is most common in self-correcting agent loops without iteration limits.
- Check: Is there a maximum iteration count on every loop?
- Check: Is there a circuit breaker that exits on repeated identical states?

**Silent failure**
An agent fails, but the pipeline continues with degraded or missing output rather than surfacing an error. Downstream agents receive partial state and produce incorrect results.
- Check: Is each agent's output validated before being passed to the next stage?
- Check: Are failures logged with enough context to reconstruct the failure path?

### Data and Pipeline Failures

**Data leakage**
Training data, evaluation data, or user data crosses boundaries it should not. In AI systems this includes prompt injection (user input altering system behavior), PII in logs, and evaluation contamination.
- Check: Is user input sanitized before injection into prompts?
- Check: Are evaluation datasets isolated from retrieval indexes?

**Overfitting to evaluation data**
The system performs well on the evaluation set but fails on real inputs because the evaluation set was not representative. This is especially common when evaluation data was generated by the same model being evaluated.
- Check: Was the evaluation set constructed independently of the model?
- Check: Does the evaluation set cover the full distribution of expected inputs, including edge cases?

**Schema drift**
An upstream data source changes its structure. Downstream Pydantic models silently fail to validate or validate incorrectly because field names or types have changed.
- Check: Are upstream schema changes detected before they reach the model layer?
- Check: Are validation errors surfaced immediately, or swallowed silently?

### Architecture Failures

**Single responsibility violations**
A class or function mixes concerns — loading data, calling the LLM, and formatting output in the same unit. This makes the system difficult to test, extend, and debug.
- Check: Does each component have a single reason to change?
- Check: Can each component be tested in isolation?

**Missing abstraction boundaries**
High-level orchestration depends directly on a specific LLM provider or vector store. Swapping providers requires changes throughout the codebase rather than in one place.
- Check: Is the LLM provider accessed through an interface, not a hardcoded constructor?
- Check: Is the vector store accessed through a factory or abstraction layer?

**No retry strategy**
LLM API calls fail transiently. A system with no retry logic produces user-facing errors for failures that would resolve on a second attempt.
- Check: Does every chain invocation have a minimum two-attempt retry loop?
- Check: Is the exception type caught specific, or does the retry swallow all errors?

---

## Early Kill Criteria

These findings indicate that the proposed design should be rearchitected before implementation proceeds. They are not items to address in implementation — they are blockers.

- The chain topology cannot handle the token budget at maximum input size
- Routing logic has unresolvable ambiguity (two agents legitimately match the same input class)
- The system requires shared mutable state between agents with no isolation mechanism
- The evaluation strategy cannot distinguish a working system from a plausible-sounding one
- The architecture depends on a framework capability that has not been confirmed to exist

---

## Critique Output Format

Each finding should be recorded as:

```
Finding: [short label]
Severity: Blocker | High | Medium | Low
Location: [file, function, or component]
Observation: [what was found, with evidence]
Root Cause: [structural explanation — why does this exist?]
Required Change: [what must change before implementation proceeds]
```

Findings at Blocker or High severity must be resolved before implementation begins. Medium and Low findings are addressed during implementation or review.

---

## What Critique Is Not

Critique is not code review. Code review verifies that an implementation matches its specification. Critique verifies that the specification is sound before implementation begins.

Critique is not a gatekeeping exercise. Its purpose is to surface problems while they are cheap to fix — not to delay or block progress. A critique session that produces no findings is a successful critique session.

Critique is not a one-person job. The most effective critiques involve someone who did not write the architecture, because the author's mental model blinds them to the gaps in it.

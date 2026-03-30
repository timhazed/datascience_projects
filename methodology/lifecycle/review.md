# Review

*This document reflects current standards for code review in AI systems, including the checks that are unique to LLM-based architectures. Updated as new anti-patterns are identified in practice.*

---

## Purpose

Review is a formal verification that the implementation matches the specification and meets quality standards. It is the last gate before a system is considered complete.

Review catches what critique cannot: implementation bugs, anti-patterns introduced during coding, security issues, and gaps between what the specification intended and what the code delivers. Critique operates on design. Review operates on code.

---

## When to Use

Review is conducted after implementation is complete and before the system is deployed or handed off. It applies to all new systems and to any change that touches component boundaries, data flow, or LLM interactions.

---

## Inputs

- The implementation (source code and tests)
- The verified architecture specification
- The critique findings and their resolutions

---

## Outputs

- A prioritized list of findings with required changes
- Confirmation that the implementation matches the specification
- A sign-off that the system meets the quality standards defined in the principles

---

## Mandatory Pre-Flight

Do not review from memory. Before issuing any finding:
1. Read the file being reviewed
2. Trace data flow through the full chain — prompt construction, LLM invocation, output parsing, downstream consumption
3. Grep for all call sites of modified functions
4. Check `pyproject.toml` for dependency versions relevant to the code

Findings issued without reading the code are noise.

---

## Review Checklist

### Philosophy Compliance

- [ ] **Single entry / single exit**: non-trivial functions have a single conceptual exit. Guard clauses at the top of a function are acceptable. Multiple `return` statements scattered through complex body logic are a finding.
- [ ] **One class per file**: no file defines more than one public class. Internal `_Classes` co-located with their parent are the only exception.
- [ ] **SRP**: each class has one reason to change. No class mixes data loading, LLM invocation, and output formatting.
- [ ] **OCP**: new behavior is added by extension, not by modifying existing classes.
- [ ] **DIP**: high-level modules depend on abstractions, not concrete provider implementations.
- [ ] **No duplication**: shared logic is extracted, not copied. Prompt templates live in one location.
- [ ] **Docstrings**: every public function and method has a docstring.
- [ ] **Inline comments**: complex sequences have explanatory comments.

### LangChain and LLM Checks

- [ ] Is every chain constructed at module level or via factory? No per-request construction.
- [ ] Does every `chain.invoke()` call have a retry loop with a minimum of two attempts?
- [ ] Is the exception type in the retry loop specific, not a bare `except Exception`?
- [ ] Are all LLM outputs validated through a Pydantic model? No raw string parsing.
- [ ] Are Field descriptions in Pydantic schemas specific enough to constrain LLM output?
- [ ] Is `max_tokens` calculated using the formula, not assigned as an arbitrary number? Show the math.
- [ ] Is the model name a configurable variable, not a hardcoded string?
- [ ] Is temperature appropriate for the task? Structured output uses 0.
- [ ] Is the LLM provider accessed through an abstraction layer?

### RAG-Specific Checks

- [ ] Is chunk size defined in tokens, not characters?
- [ ] Is there a relevance score threshold that filters low-quality retrieval results?
- [ ] Does the system degrade gracefully when retrieval returns nothing?
- [ ] Is the vector store accessed through a factory or abstraction, not hardcoded?
- [ ] Does metadata flow correctly from document load through to retrieval results?

### Agent-Specific Checks

- [ ] Does every agent have a validated Pydantic input and output schema?
- [ ] Are routing conditions mutually exclusive? Is there a fallback for unmatched inputs?
- [ ] Is there a maximum iteration count on every agent loop?
- [ ] Is state passed explicitly between agents, not accessed globally?
- [ ] Is each agent's output validated before being passed to the next stage?

### Safety and Security

- [ ] Is user input sanitized before injection into prompts? No raw string interpolation of user input.
- [ ] Is there an intent filter that rejects off-topic queries before the main pipeline runs?
- [ ] Is there an LLM-as-judge layer for safety-critical or domain-sensitive outputs?
- [ ] Do error messages avoid exposing stack traces, internal state, or model behavior?
- [ ] Are API keys and credentials loaded from environment variables, not hardcoded?
- [ ] Is PII excluded from logs?

### Test Coverage

- [ ] Does test coverage meet or exceed the floor defined in the architecture specification?
- [ ] Does `tests/` mirror the structure of `src/`?
- [ ] Are LLM calls mocked in unit tests, with integration tests covering real API behavior?
- [ ] Do tests cover the failure paths — malformed LLM output, retrieval failure, API timeout — not just the happy path?

### Project Artifacts

- [ ] `README.md` is present and describes how to install, configure, and run the system
- [ ] `pyproject.toml` is present with all dependencies pinned
- [ ] `.gitignore` excludes `.env`, `__pycache__`, and build artifacts

---

## Finding Format

```
Finding: [short label]
Severity: Blocker | High | Medium | Low
File: [path:line]
Observation: [what was found]
Root Cause: [why it exists — structural explanation]
Required Change: [what must change]
```

**Blocker** — the system will produce incorrect results or fail in production. Must be resolved before delivery.

**High** — violates a non-negotiable standard (SOLID, no duplication, retry strategy). Must be resolved before delivery.

**Medium** — quality gap that will cause friction or fragility over time. Should be resolved before delivery; may be deferred with explicit acknowledgment.

**Low** — style or documentation gap. May be addressed in a follow-up.

---

## What Review Is Not

Review is not a design discussion. If a finding reveals a design flaw — not an implementation error — the correct response is to return to architecture, not to negotiate the finding in review. The specification is the authority.

Review is not punishment. Its purpose is to catch problems while they are still cheap to fix. A review with many findings against a first implementation is expected. A review with many findings against a mature system is a signal that the lifecycle is not being followed.

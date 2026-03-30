# Implementation

*This document reflects current standards for turning a verified specification into production-grade code. Standards are updated as new patterns prove their value in practice.*

---

## Purpose

Implementation is the phase where a verified specification becomes working software. Its inputs are a specification that has passed critique. Its output is code that meets that specification, accompanied by tests that prove it does.

The key constraint of this phase: the specification is not renegotiated during implementation. If an implementation decision reveals a flaw in the specification, the correct response is to pause, revisit the architecture, and run critique again on the changed design — not to silently diverge from the spec in the code.

---

## When to Use

Implementation begins only after:
- The architecture specification is complete
- Critique has been completed and all Blocker and High severity findings are resolved
- The implementation is decomposed into tickets (defined in the architecture specification)

---

## Inputs

- The verified architecture specification
- The implementation ticket decomposition
- The existing codebase (read it — do not assume its state)

---

## Outputs

- Working code that implements the specification
- Tests that cover the implementation at the defined coverage floor
- Updated `README.md` if the project structure changes

---

## Standards

These standards apply to every implementation. They are not optional, and they are not adjusted based on time pressure. A system that violates them has technical debt from its first commit.

### Single Entry / Single Exit

Every function and method has one entry point and one return path. Guard clauses — early returns at the top of a function for invalid or missing input — are acceptable. Multiple `return` statements scattered through complex body logic are a violation.

This constraint exists because functions with multiple exits are harder to reason about, harder to test, and harder to instrument. The cognitive load of tracing all possible exit paths compounds as systems grow.

### One Class Per File

No file defines more than one public class. Internal classes prefixed with `_` and co-located with their parent are the only exception.

### SOLID

**Single Responsibility Principle** — each class has one reason to change. Never mix concerns in one class. Loading data, calling an LLM, and formatting output are three different responsibilities.

**Open/Closed Principle** — new behavior is added by extension, not by modifying existing classes.

**Liskov Substitution Principle** — subclasses are substitutable for their base class without breaking behavior.

**Interface Segregation Principle** — abstract classes do not force implementers to define methods they don't use.

**Dependency Inversion Principle** — high-level modules depend on abstractions, not concrete implementations. The orchestration layer depends on an LLM interface, not a hardcoded provider constructor.

### No Duplication

Shared logic lives in `utils/` or a base class, not copied across modules. Prompt templates live in one canonical location. If the same logic appears in two places, one of them is wrong.

### Tests Written Alongside Code

Tests are not written after implementation is complete. Each ticket includes the tests that verify its deliverable. The coverage floor is defined before implementation begins — minimum 80%, higher for safety-critical or LLM-judge layers.

`tests/` mirrors `src/`. A module at `src/chains/retrieval_chain.py` has a test at `tests/chains/test_retrieval_chain.py`.

### Documented Code

Every public function and method has a docstring: what it does, its parameters, and what it returns. Inline comments on complex sequences — chain construction, retry logic, state routing — explain the why, not the what.

---

## LangChain Implementation Standards

### Chain Construction

Chains are constructed at module level (singleton pattern) or injected via factory at the application boundary. A chain is never rebuilt per request.

```python
# Correct — module-level singleton
_chain = prompt | llm | parser

def invoke(input: InputModel) -> OutputModel:
    return _chain.invoke(input)
```

### Retry Strategy

Every `chain.invoke()` call has a minimum two-attempt retry loop. LLM APIs fail transiently. Silent single-attempt failures produce user-facing errors for problems that would resolve automatically.

The exception type caught must be specific. Catching all exceptions in a retry loop masks real bugs.

```python
for attempt in range(2):
    try:
        result = chain.invoke(input)
        break
    except OutputParserException:
        if attempt == 1:
            raise
```

### Structured Output

All LLM outputs are validated through a Pydantic model. No raw string parsing. Field descriptions in the schema serve as LLM format instructions — they must be specific enough to constrain model behavior.

### Provider Abstraction

The LLM provider is accessed through a factory or interface, not hardcoded in chain definitions. Swapping providers should require changes in one place.

---

## Ticket-Driven Delivery

Implementation proceeds ticket by ticket. Each ticket is:
- Completed before the next begins
- Accompanied by its tests before moving on
- Reviewed against the specification before closure

This constraint prevents "almost done" situations where 90% of the system is built but untested. Each ticket produces a working, tested increment.

---

## When the Spec and Reality Diverge

It is common to discover during implementation that a design assumption was wrong — a framework behaves differently than specified, a data structure requires a different shape, a token budget calculation needs revision.

The correct response:
1. Stop implementation on the affected ticket
2. Document the finding precisely (what the spec assumed, what reality requires)
3. Return to architecture, update the specification, run critique on the changed section
4. Resume implementation against the updated spec

Do not silently implement something different from the spec. Spec drift makes critique and review impossible.

---

## Common Failure Modes

**Building before critique completes.** Critique exists to catch design flaws before implementation. Starting implementation before critique finishes means building on an unverified foundation.

**Skipping tests under time pressure.** Tests skipped in implementation are tests that don't exist. They cannot be "added later" without understanding what the code does — and that understanding degrades over time.

**Hardcoding configuration.** Model names, API endpoints, and environment-specific values hardcoded in the implementation create systems that cannot be configured without code changes.

**God classes.** A class that grows to handle multiple responsibilities is a sign that the architecture was underspecified or that the spec is being ignored. Stop and refactor to the single responsibility principle before continuing.

**Per-request chain construction.** Rebuilding the chain on every invocation adds latency and obscures errors. Chains are constructed once.

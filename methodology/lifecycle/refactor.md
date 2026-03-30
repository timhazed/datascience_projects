# Refactor

*This document reflects current thinking on surgical system improvement — changing what is causing friction or fragility without breaking what works. Updated as new refactor patterns prove their value in practice.*

---

## Purpose

Refactor is the phase where a specific part of an existing system is redesigned to reduce friction, eliminate fragility, or align with updated standards — without breaking the system's current behavior.

Refactor is not a rewrite. A rewrite discards working code and replaces it. A refactor preserves behavior while improving the structure behind it. The cardinal rule: **working code at every step**.

The trigger for refactor is always specific. "The code is messy" is not a trigger. "This class has three reasons to change and adding a new retrieval strategy requires modifying it in four places" is a trigger.

---

## When to Use

Refactor is warranted when:
- A component is difficult to test because of tight coupling or mixed responsibilities
- Extending the system with new behavior requires modifying existing code rather than extending it
- The same logic appears in multiple places and the copies have diverged
- Evaluation findings identify a component as the source of recurring failures
- A standard has been updated and existing code does not comply

Refactor is not warranted when:
- The code works, is well-tested, and does not need to change
- The "improvement" is aesthetic preference rather than structural necessity
- The change is a complete replacement of an existing design

---

## Inputs

- A specific, scoped diagnosis of what is causing friction or fragility
- The current code (read it — understand it fully before proposing any change)
- The tests that cover the affected area

---

## Outputs

- A target state design for the affected subsystem
- An incremental migration plan from current state to target state
- Updated code with passing tests at every step

---

## Phase 1: Diagnosis

Understand the current state before proposing any change. Every refactor that goes wrong does so because the engineer did not fully understand what they were changing.

**Required before designing the target state:**
1. Read every file in the affected subsystem
2. Grep for all call sites of the classes and functions being changed
3. Identify all consumers — internal modules, CLI, UI, tests
4. Read the tests covering the affected area
5. Document the root cause of the friction or fragility

**Diagnosis output — the smell catalogue:**

| File | Location | Smell | Root Cause |
|------|----------|-------|------------|
| ... | ... | SRP violation — class loads data and calls LLM | No separation between I/O and chain logic |
| ... | ... | Duplicated prompt template | Template copied instead of extracted to shared location |

Do not design the target state until the diagnosis is complete.

---

## Phase 2: Target State Design

Define where the subsystem should end up. Apply the same standards as the architecture phase — this is a re-architecture of a bounded scope.

The target state design answers:
- What are the new component boundaries?
- What interfaces exist between them?
- What is the data flow?
- What changes for consumers of the affected components?

The target state must be achievable in incremental steps, each of which leaves the system in a working state.

---

## Phase 3: Incremental Migration

Define the sequence of steps from current state to target state. Each step is:
- A single, coherent change
- Verified by a passing test suite
- Committed independently

This structure means the refactor can be paused at any step without leaving the system broken. It also makes the change reviewable — each commit has a clear, focused purpose.

**Migration step format:**

```
Step N: [short label]
Change: [what is modified]
Tests: [what must pass after this step]
Verifies: [which aspect of the target state is delivered]
```

---

## Common Structural Smells and Their Fixes

**SRP violation — class with multiple responsibilities**

Symptom: Adding a new feature requires changing a class for unrelated reasons. Tests for one responsibility break when another responsibility changes.

Fix: Extract each responsibility into its own class. Define an interface if multiple implementations are possible. Inject dependencies.

**God class**

Symptom: One class owns a large fraction of the system's logic. Its test file is the largest in the project. New developers cannot find where to add new behavior.

Fix: Identify the distinct responsibilities inside the class. Extract them one at a time, starting with the most isolated. Do not extract everything at once.

**Prompt template duplication**

Symptom: The same prompt structure appears in multiple files with minor variations. A change to the shared logic requires hunting down every copy.

Fix: Extract to a single canonical location. Parameterize the variations. Reference from all consumers.

**Missing abstraction — hardcoded provider**

Symptom: The LLM provider is instantiated directly in chain definitions. Swapping providers requires changes throughout the codebase.

Fix: Introduce a factory or interface. Move provider construction to the application boundary. Chain definitions depend on the interface.

**Missing retry logic**

Symptom: LLM API failures produce user-facing errors. The failure rate is non-zero in production.

Fix: Wrap every `chain.invoke()` in a retry loop with a minimum of two attempts. Catch specific exception types.

**Test-last implementation**

Symptom: The coverage floor is not met. Tests are concentrated on the happy path. Failure paths are untested.

Fix: Identify untested paths from the coverage report. Write tests for each. Do not change behavior — only add coverage.

---

## What Not to Refactor

**Working code that does not need to change.** The goal of a refactor is to reduce friction or fragility in a specific area. Refactoring code that is not causing problems adds risk without benefit.

**Aesthetic preferences.** Variable naming, comment style, and formatting are not refactor triggers unless they cause genuine ambiguity. Apply a formatter and move on.

**Everything at once.** A refactor that touches every file in the project is a rewrite. Refactors are scoped. If the scope keeps expanding, stop and reassess whether the right answer is a new design rather than a repair of the old one.

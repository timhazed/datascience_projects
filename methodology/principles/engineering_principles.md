# Engineering Principles

*These principles apply to every system built under this methodology, regardless of stack, domain, or scale. They are not aspirational guidelines — they are non-negotiable standards. This document is updated as principles are refined through production experience.*

---

## Why These Principles Exist

AI systems introduce new complexity — non-determinism, latency variability, context limits, retrieval quality — but they are still software. The structural problems that make traditional software hard to maintain, test, and extend make AI systems worse, faster. A god class in a web application is a maintenance burden. A god class in an agent pipeline is a debugging nightmare.

These principles apply SOLID and classical software design to the specific constraints of AI systems. They do not replace domain judgment — they create the structure within which judgment can be applied reliably.

---

## Core Principles

### Single Entry / Single Exit

Every function and method has one entry point and one return path.

Guard clauses — early returns at the top of a function for invalid or missing input — are acceptable and encouraged. They handle the exceptional path explicitly and immediately, leaving the main body to handle only the valid case.

Multiple `return` statements scattered through complex body logic are a violation. They create hidden exit paths that are easy to miss when reading, testing, or instrumenting code.

**Why this matters in AI systems:** Chain functions, agent handlers, and parsing utilities that branch in complex ways produce LLM interactions that are difficult to trace when they fail. A single exit point means one place to log, one place to instrument, one place to handle errors.

### One Class Per File

No file defines more than one public class. Internal classes prefixed with `_` and co-located with their parent are the only exception.

**Why this matters:** Files that contain multiple classes grow in ambiguous directions. When a file owns one class, its purpose is unambiguous and its growth is constrained.

### SOLID

| Acronym | Full Name | Core Idea |
|---------|-----------|-----------|
| SRP | Single Responsibility Principle | A class has one reason to change |
| OCP | Open/Closed Principle | Open for extension, closed for modification |
| LSP | Liskov Substitution Principle | Subclasses are substitutable for their base class |
| ISP | Interface Segregation Principle | Interfaces don't force unused method implementations |
| DIP | Dependency Inversion Principle | Depend on abstractions, not concrete implementations |

These principles are referenced throughout the lifecycle and review checklists by acronym. The definitions above are the canonical reference.

**Single Responsibility Principle (SRP)**

Each class has one reason to change. Never mix concerns in a single class.

Common violations in AI systems:
- A class that loads documents, chunks them, and embeds them (three responsibilities)
- A class that constructs the prompt, calls the LLM, and parses the output (three responsibilities)
- An agent that handles routing logic and domain expertise (two responsibilities)

Each concern is a candidate for its own class. When a class needs to change because a retrieval strategy changed, that change should not require touching the embedding logic.

**Open/Closed Principle (OCP)**

New behavior is added by extension, not by modifying existing classes.

In practice: use abstract base classes or protocols to define component interfaces. New implementations — a new LLM provider, a new chunking strategy, a new retrieval mechanism — are new classes that implement the interface, not modifications to existing ones.

**Liskov Substitution Principle (LSP)**

Subclasses are substitutable for their base class without breaking behavior.

If a `VectorRetriever` base class defines `retrieve(query: str) -> list[Document]`, any subclass must satisfy that contract. A subclass that requires additional setup steps or returns a different type violates this principle.

**Interface Segregation Principle (ISP)**

Abstract classes do not force implementers to define methods they don't use.

If an abstract `DocumentProcessor` defines both `load()` and `chunk()`, an implementer that only needs to load documents is forced to implement a `chunk()` method that does nothing. Split the interface.

**Dependency Inversion Principle (DIP)**

High-level modules depend on abstractions, not concrete implementations.

The orchestration layer — the code that sequences agents, chains, or pipeline stages — depends on interfaces, not on `ChatGroq`, `FAISS`, or any other specific provider. Provider selection happens at the application boundary, not inside the pipeline.

**Why this matters:** A pipeline that hardcodes `ChatGroq` cannot be tested without an API call, cannot be swapped to a different provider without rewriting chain definitions, and cannot have its LLM behavior mocked in unit tests.

### No Duplication

Shared logic lives in `utils/` or a base class. Prompt templates live in one canonical location. Copy-pasted code is a synchronization problem waiting to happen.

When the same logic appears in two places, the copies will diverge. When they diverge, the system behaves differently in different contexts — and the difference is invisible until it causes a failure.

### Tests for Everything

Unit tests are written alongside code, not after. Every ticket includes the tests that verify its deliverable. The coverage floor is defined before implementation begins. Tests are added as defects are processed and resolved to ensure no regressive behavior in future builds.

**Minimum coverage floor: 80%.** Systems with LLM-as-judge layers or safety-critical outputs: 90% or higher.

Tests cover failure paths — malformed LLM output, retrieval failure, API timeout, invalid user input — not just the happy path. A test suite that covers only the happy path is not a safety net; it is a false sense of security.

### Documented Code

Every public function and method has a docstring: what it does, its parameters, and what it returns.

Inline comments explain complex sequences — chain construction, retry logic, state routing, conditional branching. Comments explain the *why*, not the *what*. If the code clearly shows what is happening, the comment explains why it is happening that way.

---

## Quantitative Standards

### Token Budget Calculation

`max_tokens` is never assigned as an arbitrary round number. It is calculated:

```
max_tokens = (avg_fields × avg_tokens_per_field × num_items) × 1.3
```

The 1.3 safety margin is mandatory. Show the calculation in the architecture specification and in code comments where `max_tokens` is set. An arbitrary number is an undocumented assumption that will produce silent truncation failures.

### Retry Strategy

Every `chain.invoke()` call has a minimum two-attempt retry loop. LLM APIs fail transiently. A single-attempt call produces user-facing errors for failures that would resolve automatically on a second attempt.

The exception type caught must be specific — not a bare `except Exception`. Catching all exceptions in a retry loop masks bugs that are not transient API failures.

### Coverage Floor

The test coverage target is defined before implementation begins. Minimum: 80%. The floor does not move based on delivery pressure — it is a pre-condition for a system being considered complete.

---

## Project Artifacts

Every project requires these artifacts before it is considered complete:

- `README.md` — installation, configuration, and usage instructions
- `pyproject.toml` — all dependencies declared and pinned
- `.gitignore` — excludes `.env`, `__pycache__`, build artifacts, and credentials

These are not optional. A project without them cannot be reproduced, deployed, or handed off.

---

## Standard Directory Structure

```
project/
├── src/
│   ├── agents/
│   ├── chains/
│   ├── models/
│   ├── utils/
│   └── app.py
├── tests/           (mirrors src/)
├── pyproject.toml
├── .gitignore
└── README.md
```

`tests/` mirrors `src/`. A module at `src/chains/retrieval_chain.py` has a corresponding test at `tests/chains/test_retrieval_chain.py`. This structure makes it immediately visible when a module has no tests.

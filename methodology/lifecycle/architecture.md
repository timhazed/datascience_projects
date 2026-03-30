# Architecture

*This document reflects current thinking on what a complete system specification contains and why each section is required. It is updated as new stack patterns and design constraints emerge from production experience.*

---

## Purpose

Architecture is the phase where a system is fully specified before any code is written. It defines components, interfaces, data flow, and constraints at a level of detail sufficient for implementation to begin without ambiguity.

The architecture phase does not produce code. It produces a specification that answers three questions: what does the system do, how is it structured, and where will it fail? The last question is answered in the critique phase that follows — but the specification must be complete enough to make critique meaningful.

A specification that is too vague defers decisions to implementation, where they are more expensive to change. A specification that is too rigid constrains implementation unnecessarily. The target is a specification that defines all structural decisions while leaving implementation details to the implementer.

---

## When to Use

Architecture is required for any system that:
- Will be used by more than one person
- Has multiple interacting components
- Will be extended or maintained beyond its initial delivery
- Has safety-critical, financial, or user-facing outputs

Architecture is optional for pure exploration work, one-off scripts, and standalone notebooks. See [exploration.md](exploration.md) for the appropriate phase for those.

---

## Inputs

- A completed exploration phase with a promotion decision, or a clear problem statement with defined success criteria
- Known constraints: stack, deployment environment, cost budget, latency requirements
- Any existing code in the project that may be reused or extended

---

## Outputs

A complete technical specification with all required sections (see below). This specification is the primary input to the critique phase.

---

## Mandatory Pre-Flight

Before writing any specification:

1. Read all relevant existing files in the project. Design from the current state of the codebase, not from memory.
2. Identify reusable patterns — chains, parsers, Pydantic models, UI components, utility functions — before designing new ones.
3. Confirm that proposed dependencies are available in the project's package manager.
4. Verify that any framework capability the design depends on actually exists and behaves as expected.

Specifications built on unverified assumptions produce critiques that are entirely blockers. Do the pre-flight.

---

## Required Specification Sections

### 1. Goal and Scope

- What problem does this system solve?
- What is explicitly out of scope?
- Success criteria, stated as measurable outcomes

### 2. Data Models

Define all input and output schemas before designing components. Data models are the contracts between components — everything else is built around them.

Use Pydantic v2. Field descriptions should be specific enough to serve as LLM format instructions where applicable.

```python
class InputModel(BaseModel):
    field: type = Field(description="...")

class OutputModel(BaseModel):
    field: type = Field(description="...")
```

Define models for every boundary in the system: user input, LLM input, LLM output, component handoffs, and final output.

### 3. Component Topology

An ASCII diagram of the full system:

```
UserInput → ComponentA → ComponentB → ComponentC → Output
                ↑
         external dependency
```

For each component:
- Its single responsibility
- Its input schema
- Its output schema
- Its dependencies (LLM, vector store, external API)

### 4. LLM Configuration

- Model name as a configurable variable, not a hardcoded string
- Temperature with justification (0 for structured output, higher for generative tasks)
- `max_tokens` calculation — never an arbitrary round number:

```
max_tokens = (avg_fields × avg_tokens_per_field × num_items) × 1.3
```

Show the calculation. The 1.3 safety margin is mandatory.

- LLM construction strategy: singleton at module level or injected via factory. One construction, not one per request.
- Retry strategy on every chain invocation: minimum two attempts, specific exception type caught, backoff if applicable.

### 5. Retrieval Design (RAG systems)

- Document sources and loader strategy
- Chunking approach: fixed-size, paragraph-aware, or semantic — with justification
- Chunk size in tokens (not characters), overlap percentage
- Vector store selection with justification
- Embedding model, dimension, and batch strategy
- Retrieval mechanism: similarity search, MMR, hybrid — with score threshold
- Metadata extracted and how it flows through to results

### 6. Agent Roster (agent systems)

For each agent:
- Role and single responsibility
- Input schema and output schema
- LLM configuration
- Tools or capabilities available to it

Orchestration flow as an ASCII diagram. Define handoff data structures, routing conditions, fallback paths, and iteration limits on any loop.

### 7. Safety and Guardrails

For every user-facing output:
- Input validation: Pydantic constraints, length limits, sanitization
- Intent filter: what queries should be rejected before invoking the main pipeline?
- LLM-as-judge layer: what does the judge check, and what happens on a fail verdict?
- Error surface: what does the user see when something fails? No stack traces, no internal state.

### 8. Project Structure

Standard layout:

```
project/
├── src/
│   ├── agents/         (agent implementations)
│   ├── chains/         (LangChain chain definitions)
│   ├── models/         (Pydantic schemas)
│   ├── utils/          (shared utilities)
│   └── app.py          (entry point)
├── tests/              (mirrors src/)
├── pyproject.toml
├── .gitignore
└── README.md
```

`tests/` mirrors `src/`. Every module has a corresponding test file.

### 9. Implementation Tickets

Decompose the specification into discrete, independently deliverable units of work. Each ticket should be completable in a single session without requiring changes to other tickets.

Format:
```
Ticket N: [short label]
Scope: [what this ticket delivers]
Depends on: [ticket numbers, if any]
Tests required: [what must be tested]
```

---

## Common Failure Modes

**Designing before reading the codebase.** Specifications that duplicate existing patterns or conflict with existing code waste implementation time and produce confusing PRs.

**Underspecifying data models.** Vague field descriptions produce vague LLM output. Every field that an LLM will populate needs a description that constrains it.

**Omitting the token budget calculation.** "I'll set `max_tokens` to 2000 and see if it's enough" is not a calculation. This produces silent truncation failures in production.

**Specifying without verifying.** A specification that depends on a framework behavior that was not confirmed will produce a blocker in critique. Confirm first, specify second.

**Skipping implementation tickets.** A specification without a decomposition produces an implementation that attempts to build everything at once. Tickets enforce incremental delivery and make progress visible.

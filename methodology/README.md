# AI Systems Engineering Methodology

*This methodology is a living document. It reflects current thinking, refined continuously through production experience and experimentation. When a project invalidates an assumption or surfaces a better pattern, this document is updated. It is never finished.*

---

I built this methodology because most AI projects fail for the same reasons: they skip design, skip failure analysis, and treat evaluation as an afterthought. This directory codifies how I approach AI system design — not as a checklist to follow blindly, but as a repeatable operating system for building reliable, maintainable systems under real-world constraints.

This is not a generic AI tutorial. It reflects decisions made on real projects, with real failure modes, under real cost and latency constraints.

---

## The Lifecycle

Every system I build follows this sequence:

```
Exploration → Architecture → Critique → Implementation → Review → Refactor
```

Each phase has a defined purpose, clear inputs and outputs, and explicit exit criteria. The sequence is not ceremonial — each phase prevents a class of failures that the next phase cannot catch.

```
┌─────────────┐
│ Exploration │  Validate the hypothesis. Establish a baseline.
│             │  Decide whether to build at all.
└──────┬──────┘
       │
┌──────▼──────┐
│Architecture │  Define components, interfaces, and data flow
│             │  before any code is written.
└──────┬──────┘
       │
┌──────▼──────┐
│   Critique  │  Find every way the design can fail.
│             │  This phase happens before implementation, not after.
└──────┬──────┘
       │
┌──────────────┐
│Implementation│  Build to the specification. Enforce standards.
│             │  Write tests alongside code.
└──────┬──────┘
       │
┌──────▼──────┐
│   Review    │  Verify the implementation matches the specification.
│             │  Catch bugs, anti-patterns, and quality gaps.
└──────┬──────┘
       │
┌──────▼──────┐
│  Evaluation │  Measure system behavior in production conditions.
│             │  Confirm the system does what it claims to do.
└──────┬──────┘
       │
┌──────▼──────┐
│   Refactor  │  Improve what is causing friction or fragility.
│             │  Working code at every step — no big bang rewrites.
└─────────────┘
```

---

## Philosophy

Three principles shape every decision in this methodology:

**Build for failure first.** The most valuable engineering work happens before implementation begins. Anticipating failure modes, questioning assumptions, and establishing kill criteria prevents wasted cycles, not just bad outcomes.

**AI systems are structurally different from traditional software.** Non-determinism, token budgets, retrieval quality, and model behavior under distribution shift are not edge cases — they are first-class engineering concerns. They require explicit design attention at every phase.

**Reproducibility requires structure.** A system that cannot be explained, extended, or handed off is a liability. Every design decision should be derivable from the structure of the code, not from institutional memory.

---

## Directory Structure

```
methodology/
├── README.md                   ← this file
│
├── lifecycle/                  ← the core operating system
│   ├── exploration.md
│   ├── architecture.md
│   ├── critique.md
│   ├── implementation.md
│   ├── review.md
│   ├── evaluation.md
│   └── refactor.md
│
├── principles/                 ← the engineering doctrine
│   ├── engineering_principles.md
│   ├── llm_constraints.md
│   └── safety_and_guardrails.md
│
├── patterns/                   ← reusable system designs
│   ├── rag.md
│   ├── agentic_systems.md
│   ├── multi_agent.md
│   └── evaluation_patterns.md
│
├── workflows/                  ← decision guides for system selection
│   └── decision_guide.md
│
└── case_studies/               ← methodology applied to real projects
    ├── roof_cnn.md
    ├── rag_pipeline.md
    └── llm_benchmarking.md
```

---

## Where to Start

**If you want to understand the approach:** Read [lifecycle/critique.md](lifecycle/critique.md) first. It is the phase most engineers skip, and the one that prevents the most expensive mistakes.

**If you want to understand the standards:** Read [principles/engineering_principles.md](principles/engineering_principles.md). These apply to every system, regardless of stack or domain.

**If you want to understand system selection:** Read [workflows/decision_guide.md](workflows/decision_guide.md). It maps problem types to the appropriate architecture pattern.

**If you want to see the methodology applied:** Browse [case_studies/](case_studies/).

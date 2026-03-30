# Decision Guide

*This document provides the decision framework for selecting the right architecture pattern, lifecycle entry point, and evaluation strategy for a given problem. Updated as new patterns prove their value and existing ones reveal their limits.*

---

## How to Use This Guide

Every AI system starts with a problem statement. This guide maps that problem to the right approach. The questions are ordered — answer them in sequence and stop when you have a clear answer.

---

## Part 1: Should You Build at All?

Before selecting a pattern, confirm the problem is worth building a system for.

**Question 1.1 — Is the requirement clear?**

Can you state what the system must do in two sentences, with measurable success criteria? If not, the problem is not yet defined well enough to build anything. Spend more time in requirements before starting exploration.

**Question 1.2 — Is an AI system the right tool?**

AI systems are justified when:
- The task requires language understanding, generation, or reasoning
- The input space is too large or varied for deterministic rules
- A simpler tool (search, a database query, a rule engine) cannot meet the requirement

If a deterministic system can solve the problem, build that instead. AI systems add complexity, cost, non-determinism, and operational burden. They are not the default choice.

**Question 1.3 — Is there data available to validate the approach?**

Without representative data, exploration cannot produce reliable conclusions. If data is not available, the first task is data acquisition — not system design.

---

## Part 2: Where Do You Enter the Lifecycle?

**Feasibility is uncertain → Start with Exploration**

If a core capability is unproven — can this model handle this domain? does retrieval work well enough on this corpus? — start with exploration. A time-boxed experiment (hours, not days) answers the question before committing to design.

See [lifecycle/exploration.md](../lifecycle/exploration.md).

**Feasibility is established → Start with Architecture**

If the approach is proven (either through prior work or exploration), move directly to architecture. Skipping architecture to "just start building" produces systems that are difficult to extend, test, or hand off.

See [lifecycle/architecture.md](../lifecycle/architecture.md).

---

## Part 3: Which Architecture Pattern?

Work through these questions in order.

### Question 3.1 — Does the system need to retrieve from a document corpus?

**Yes** — the system must answer questions grounded in specific documents, policies, or domain knowledge that the model does not have in its training data.

→ Consider RAG. Continue to Question 3.2.

**No** → Continue to Question 3.3.

---

### Question 3.2 — Is grounding in source documents the primary requirement?

**Yes** — the system must return answers traceable to specific documents, and responses that are not grounded should be explicitly rejected.

→ Use the **RAG Pattern**. See [patterns/rag.md](../patterns/rag.md).

**No, but retrieval augments a more complex workflow** → Continue to Question 3.3 and incorporate retrieval as a tool in the agent design.

---

### Question 3.3 — Is the execution path fixed, or does it depend on intermediate results?

**Fixed** — the steps are always the same, only the data changes. Input goes in, output comes out, via a defined sequence of transformations.

→ Use a **pipeline** (a fixed chain of LangChain components). This is the simplest, most reliable option.

**Variable** — the correct sequence of steps cannot be determined at design time; it depends on what the model learns during execution.

→ Consider an agentic approach. Continue to Question 3.4.

---

### Question 3.4 — How many distinct responsibilities does the system need to perform?

**One clearly defined responsibility** — the system does one thing (parse intent, generate a plan, audit an output).

→ Use the **Single-Agent Pattern**. See [patterns/agentic_systems.md](../patterns/agentic_systems.md).

**Multiple distinct responsibilities** — the system needs to parse intent, perform domain reasoning, and validate safety. These are different jobs requiring different expertise.

→ Use the **Multi-Agent Pattern**. See [patterns/multi_agent.md](../patterns/multi_agent.md).

---

### Question 3.5 — Does the output have safety or quality requirements that require independent validation?

**Yes** — the output is safety-critical, domain-sensitive, or user-facing in a way that requires an audit before delivery.

→ Add a **Gatekeeper Agent** as the final stage regardless of the pattern chosen. See [patterns/multi_agent.md](../patterns/multi_agent.md#the-reference-pattern-intake--specialist--gatekeeper).

**No** → Structural output validation via Pydantic is sufficient.

---

## Part 4: Pattern Selection Summary

| Problem Type | Pattern | Reference |
|--------------|---------|-----------|
| Fixed pipeline, no retrieval | Chain pipeline | architecture.md |
| Grounded Q&A from documents | RAG | patterns/rag.md |
| Dynamic task, single responsibility | Single agent | patterns/agentic_systems.md |
| Dynamic task, multiple responsibilities | Multi-agent | patterns/multi_agent.md |
| Any of the above, safety-critical output | + Gatekeeper | patterns/multi_agent.md |

When in doubt, choose the simpler pattern. A pipeline is easier to debug than an agent. A single agent is easier to debug than a multi-agent system. Complexity is added when the simpler approach demonstrably cannot solve the problem — not before.

---

## Part 5: Evaluation Strategy Selection

| Output Type | Primary Strategy | Secondary |
|-------------|-----------------|-----------|
| Categorical / classification | Exact match, F1 | Human spot-check |
| Structured output | Parse success rate, field validity | LLM-as-judge on semantics |
| Open-ended generation | LLM-as-judge | Human evaluation for calibration |
| RAG responses | Retrieval precision/recall + context faithfulness | LLM-as-judge for answer quality |
| Safety-critical output | LLM-as-judge (calibrated) | Human evaluation mandatory |

See [lifecycle/evaluation.md](../lifecycle/evaluation.md) for the full evaluation strategy and [patterns/evaluation_patterns.md](../patterns/evaluation_patterns.md) for implementation patterns.

---

## Common Decision Errors

**Choosing agents when a pipeline suffices.** Agents introduce non-determinism and complexity that pipelines do not. If the execution path is known in advance, a pipeline is always the better choice.

**Choosing RAG when fine-tuning is appropriate.** RAG requires a curated, maintained document corpus. If the domain knowledge is stable and high-quality training data exists, fine-tuning may produce better results with lower operational complexity.

**Skipping the gatekeeper for "simple" systems.** Safety requirements do not depend on system complexity. A simple system with user-facing outputs that could cause harm requires a gatekeeper.

**Building before the pattern is chosen.** Starting implementation without a clear architecture pattern produces systems that are coherent locally but incoherent at the system level. Choose the pattern, design the system, then build it.

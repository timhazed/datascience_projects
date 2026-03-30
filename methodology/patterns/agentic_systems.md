# Agentic Systems Pattern

*This document reflects current thinking on when and how to design systems that use LLM agents for autonomous reasoning and action. Updated as agentic patterns mature and new failure modes are identified in practice.*

---

## What an Agentic System Is

An agentic system is one where an LLM is given tools, context, and an objective — and is responsible for deciding which tools to use and in what order to achieve the objective. Unlike a fixed pipeline, the execution path is determined at runtime by the model.

**Core value:** Agentic systems handle tasks where the correct sequence of steps is not known in advance — tasks that require reasoning about what to do next based on intermediate results.

**Core risk:** The same flexibility that makes agents powerful makes them unpredictable. An agent that can take actions can take the wrong actions. Every agentic system must be designed with explicit constraints on what the agent can do and how failures are detected and handled.

---

## When Agents Are Justified

Use an agentic approach when:
- The task requires dynamic decision-making that cannot be encoded as a fixed pipeline
- The correct sequence of steps depends on intermediate results that are not known at design time
- Multiple tools or data sources must be combined in ways that vary by query

Do not use agents when:
- A fixed pipeline can reliably handle the task
- The steps are always the same — only the data changes
- The risk of an incorrect action is high and the action is difficult to reverse
- Latency requirements are strict (agents are inherently slower than pipelines)

**The default should be a pipeline.** Agents introduce complexity, non-determinism, and failure modes that pipelines do not. Agents are justified when the problem genuinely cannot be solved with a fixed pipeline. When in doubt, start with a pipeline.

See [workflows/decision_guide.md](../workflows/decision_guide.md) for the full decision framework.

---

## Agent Anatomy

Every agent has four components:

**Role** — a single, clearly defined responsibility. An agent that is a domain expert and a router and a safety checker has three responsibilities and will do all three poorly.

**Tools** — the actions the agent can take. Define tools explicitly and restrict them to what the agent's role requires. A tool the agent should not use should not be available to it.

**Input schema** — a Pydantic model defining what the agent receives. Validated before the agent executes.

**Output schema** — a Pydantic model defining what the agent produces. Validated before the output is passed downstream.

---

## Single-Agent Pattern

A single agent with a defined tool set, operating on a clear objective.

```
UserInput → [InputValidation] → [Agent] → [OutputValidation] → Response
                                    ↕
                              [Tool 1]
                              [Tool 2]
                              [Tool N]
```

Use this pattern when:
- The task is well-defined and self-contained
- The agent's tool set is small and well-understood
- The output can be validated structurally

---

## Design Constraints

**Limit the tool set.** More tools means more decisions, more failure modes, and more surface area for the agent to go wrong. An agent with 15 tools will use them inconsistently. An agent with 3 well-designed tools will use them reliably.

**Define tool interfaces precisely.** Every tool has a typed input and a typed output. A tool that returns an ambiguous string gives the agent insufficient information to decide what to do next.

**Set iteration limits.** Any agent loop must have a maximum iteration count. An agent reasoning loop without an iteration limit will run until the context window is exhausted or the API budget is spent.

**Validate outputs.** The agent's output is validated through a Pydantic model before it is returned or passed downstream. An agent that produces output that cannot be parsed has failed, regardless of how reasonable the reasoning trace looks.

**Log reasoning traces.** Agent reasoning steps are the primary diagnostic tool when an agent fails. Log them with enough context to reconstruct the failure path.

---

## Tool Design Principles

A tool is a function the agent can call. It should be:

**Narrow in scope.** One tool does one thing. A tool that searches a database and formats the results for display is two tools.

**Reliable.** A tool that fails intermittently produces agent behavior that is difficult to interpret. Tool failures should be handled within the tool and surfaced as a structured error result, not as an exception that halts the agent.

**Idempotent where possible.** If an agent calls the same tool twice with the same inputs, the second call should produce the same result. Tools with side effects require additional design care.

**Fast relative to LLM calls.** A tool that takes 30 seconds to execute will produce agent trajectories that are dominated by waiting. Set timeouts on tool calls.

---

## Safety Constraints

**Irreversible actions require confirmation.** An agent that can send emails, modify databases, or call external APIs is an agent that can cause real-world harm from a reasoning error. Design confirmation steps for high-consequence, irreversible actions.

**Constrain the action space.** Define explicitly what the agent is allowed to do. An agent with write access to a database should have that access restricted to the specific tables and operations its role requires.

**Monitor for anomalous behavior.** An agent that is calling tools far more frequently than expected, or calling tools in unusual combinations, may be stuck in a reasoning loop or misinterpreting its objective. Alert on statistical anomalies in tool call patterns.

---

## Evaluation

Agentic systems require evaluation that goes beyond structural output validation. An agent can produce a correctly typed, Pydantic-valid output while having made poor decisions throughout the reasoning trace. Evaluation must measure both the quality of the final output and the soundness of the path taken to produce it.

### What to Evaluate

**Tool selection accuracy** — did the agent select the right tool for each step? Compare agent tool call sequences against reference trajectories on a labeled dataset.

**Routing accuracy** — for systems with intent classification or conditional routing, measure how often the agent routes to the correct path. This is the most directly measurable behavioral metric.

**Output quality** — use LLM-as-judge to evaluate whether the final output meets domain-specific quality criteria. See [patterns/evaluation_patterns.md](evaluation_patterns.md).

**Iteration efficiency** — how many reasoning steps does the agent take to reach a valid output? Agents that take consistently more steps than expected are a signal of ambiguous tool descriptions or underspecified objectives.

**Failure rate by category** — track what types of failures occur (reasoning loops, tool errors, parse failures, output rejections) and their relative frequency. This distribution guides where to invest in improvements.

### Eval Sets Are Not Static

An eval set defined at launch reflects what was known at launch. As the system operates in production, it encounters inputs and failure patterns that were not anticipated during design. Eval sets must grow to reflect this.

**Expand evals when:**
- A new failure mode is identified in production — add cases that cover it
- Business requirements change — new success criteria require new measurements
- The input distribution shifts — new query types need representation in the eval set
- The gatekeeper rejection rate changes significantly — investigate and add cases that explain the shift

**Business metric alignment:** Evals should connect to the metrics that matter to the project's stakeholders — task completion rate, user satisfaction signals, escalation rate, or domain-specific accuracy. An eval suite that measures technical correctness but not business outcomes is incomplete. Revisit the alignment between technical evals and business metrics at each significant project iteration.

**Treat the eval set as a first-class artifact.** Version it alongside the code. When the system changes in a way that invalidates an eval case, update or retire the case explicitly — don't let stale cases produce misleading passing results.

---

## Common Failure Modes

**Reasoning loops.** The agent reaches a state where it keeps calling the same tool or sequence of tools without making progress. Without an iteration limit, this runs until the context or budget is exhausted.

**Tool selection errors.** The agent selects a tool that is technically available but wrong for the current step. This is most common when the tool set is large or tool descriptions are ambiguous.

**Context accumulation.** As the reasoning trace grows, early context is pushed out of the context window. The agent "forgets" the original objective or earlier tool results and makes decisions based on incomplete context.

**Overconfident failure.** The agent produces an output that looks complete and well-reasoned but is built on a reasoning error in the middle of the trace. Structural output validation catches format failures; it does not catch semantic errors in the reasoning.

**Hallucinated tool calls.** The agent invokes a tool with parameters it fabricated rather than retrieved. Tool input validation catches type errors; it does not catch plausible-but-wrong values.

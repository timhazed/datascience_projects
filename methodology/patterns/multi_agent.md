# Multi-Agent Systems Pattern

*This document reflects current thinking on designing systems where multiple specialized agents collaborate to produce an output. Updated as orchestration patterns and failure isolation strategies mature through production experience.*

---

## What Multi-Agent Systems Are

A multi-agent system is an architecture where multiple specialized agents operate in sequence or in parallel, each responsible for a distinct phase of processing, with a final gatekeeper that validates the combined output before it reaches the user.

**Core value:** Specialization. A single agent asked to parse user intent, perform domain reasoning, and audit safety simultaneously will do all three less reliably than three agents each doing one thing. Multi-agent design applies the Single Responsibility Principle at the agent level.

**Core complexity:** More agents means more failure surfaces, more state to manage between agents, and more opportunity for errors to compound across stages. Every additional agent must justify its existence by improving output quality in a measurable way.

---

## The Reference Pattern: Intake → Specialist → Gatekeeper

This is the default multi-agent topology. It is the starting point for any system that requires domain expertise and safety validation.

```
UserInput
    ↓
[Intake Agent]         — parse and validate user intent
    ↓
ParsedIntent
    ↓
[Specialist Agent(s)]  — domain expertise and generation
    ↓
DomainOutput
    ↓
[Gatekeeper Agent]     — safety audit and quality check
    ↓
ConditionalRouter: PASS → return output | FAIL → rejection message
    ↓
Response
```

**Intake Agent:** Responsible for understanding what the user actually wants and validating that the request is coherent and in scope. Output is a structured, validated representation of user intent — not a response, just a parsed objective. This agent should be fast and cheap.

**Specialist Agent(s):** Responsible for generating the domain output. Receives the structured intent from intake. Has access to domain knowledge, tools, or retrieval context. May be multiple sequential or parallel agents when the domain has distinct sub-problems.

**Gatekeeper Agent:** Responsible for auditing the specialist output before it reaches the user. Checks for safety issues, factual inconsistencies, scope violations, and quality thresholds. Returns a structured verdict. Does not modify the output — it approves or rejects it.

---

## Agent Roster Design

For each agent, define before implementation begins:

| Agent | Role | Input | Output | LLM Config |
|-------|------|-------|--------|------------|
| Intake | Parse and validate user intent | Raw user input | Structured intent model | temp=0 |
| Specialist | Domain generation | Structured intent | Domain output model | task-appropriate |
| Gatekeeper | Safety and quality audit | Domain output | Verdict model | temp=0 |

Each agent:
- Has a Pydantic input schema and a Pydantic output schema
- Has a single responsibility
- Has an LLM temperature appropriate for its task (0 for structured, higher for generative)
- Has a defined behavior when it cannot produce valid output

---

## State Management

Define the state object that is passed through the pipeline. All agent-to-agent communication happens through this state — not through shared global state.

```python
class PipelineState(BaseModel):
    user_input: str
    parsed_intent: ParsedIntent | None = None
    domain_output: DomainOutput | None = None
    verdict: GatekeeperVerdict | None = None
```

**Why explicit state matters:** Shared global state between agents creates invisible dependencies. One agent's output can overwrite another's in ways that are difficult to trace. Explicit state passing makes the data flow auditable.

**State validation:** Validate state at each handoff. Before the Specialist runs, confirm that `parsed_intent` is present and valid. Before the Gatekeeper runs, confirm that `domain_output` is present and valid. A missing field is a pipeline failure, not an edge case.

---

## Orchestration Flow

Define the full execution sequence, including:

- The conditions under which each agent is invoked
- The routing logic at each decision point
- What happens on agent failure
- What happens on gatekeeper rejection

```
Intake executes
  → success: continue to Specialist
  → failure: return "Unable to process your request. Please rephrase."

Specialist executes
  → success: continue to Gatekeeper
  → failure: return "An error occurred generating a response. Please try again."

Gatekeeper executes
  → PASS: return domain output
  → FAIL: return rejection message (specific to failure reason, no internal detail)
```

---

## Parallel Execution

When the pipeline includes multiple independent specialists, they can execute in parallel to reduce total latency.

Use parallel execution when:
- Two agents operate on the same input but produce independent outputs
- Their outputs are combined by a subsequent aggregation step
- They have no dependency on each other's intermediate results

Do not use parallel execution when:
- One agent's output is the other's input
- The agents share mutable state
- The downstream combination logic cannot handle partial results

---

## Failure Isolation

Each agent is isolated from the others. An exception in one agent is caught, logged, and converted to a structured failure state — it does not propagate as an exception to the next stage.

```
agent.execute(state)
  → success: update state, continue
  → exception: log with full context, set failure state, return to router
```

The router handles the failure state. The pipeline does not crash because one agent threw an exception.

---

## Gatekeeper Design

The gatekeeper is the most important agent in a safety-critical multi-agent system. It is the last line of defense before output reaches the user.

**What the gatekeeper checks (examples by domain):**

| Domain | Checks |
|--------|--------|
| Medical / health | Advice that contradicts source documents; missing safety disclaimers; specific dosage recommendations without professional qualification |
| Financial | Specific investment recommendations presented as advice; missing risk disclosures |
| General | Off-topic content; harmful content; factual claims not supported by context |

**Gatekeeper output schema:**

```python
class GatekeeperVerdict(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    reason: str = Field(description="Specific reason for FAIL verdict, empty string for PASS")
    flags: list[str] = Field(description="Specific issues identified")
```

**Calibration:** Validate the gatekeeper against a representative sample of both good and bad outputs before deploying. A gatekeeper with a high false rejection rate degrades user experience. A gatekeeper with a high false acceptance rate provides no safety value.

---

## Evaluation

Multi-agent systems require evaluation at two levels: each agent individually, and the pipeline end-to-end. End-to-end evaluation alone is insufficient — a high end-to-end score can mask a failure in one agent that is compensated for by another, leaving the system fragile to any change in the compensating agent.

### Per-Agent Evaluation

Each agent in the pipeline has its own eval set, covering:

**Intake Agent** — intent classification accuracy, structured output validity, handling of ambiguous or out-of-scope inputs. The intake agent's eval set is typically the most straightforward to construct because ground truth is well-defined.

**Specialist Agent(s)** — domain output quality, measured by LLM-as-judge against domain-specific criteria. Measure consistency: does the agent produce outputs of equivalent quality across the full range of expected inputs, or does quality degrade on edge cases?

**Gatekeeper Agent** — verdict accuracy against a labeled dataset of known-good and known-bad outputs. Calibrate against human ratings before trusting the gatekeeper in production. Track false rejection rate and false acceptance rate independently — they represent different failure modes with different consequences.

### Pipeline Evaluation

End-to-end evaluation measures whether the full pipeline, operating as a system, produces acceptable outputs across the expected input distribution.

**What to measure:**
- End-to-end success rate (output reaches the user without pipeline failure)
- Gatekeeper PASS rate (percentage of outputs approved by the gatekeeper)
- Latency at each stage and end-to-end (identify which agent is the bottleneck)
- Cost per successful output (token consumption across all agents)

### Eval Sets Must Evolve with the System and the Business

Eval sets defined at launch are a starting point, not a finished artifact. Multi-agent systems are particularly sensitive to distribution shift — a change in how users phrase requests, a new use case introduced by stakeholders, or a change in the underlying domain knowledge can all invalidate assumptions baked into the original eval set.

**Expand evals when:**
- A new failure mode is identified in production — add cases that exercise it
- Business requirements change — new success criteria require new measurements in the eval set
- A new agent role is added to the pipeline — that agent needs its own eval set from day one
- The gatekeeper rejection rate rises or falls significantly — investigate the cause and add cases that represent the change

**Align evals with business metrics.** Technical evals measure what the system does. Business metrics measure whether it does what the project needs. These are not the same thing. A pipeline that achieves 95% structural validity may still produce outputs that fail to meet domain accuracy requirements that stakeholders actually care about. At each significant project iteration, review whether the technical evals are predictive of the business metrics — and update them if they are not.

**Version and maintain the eval set.** Eval datasets are code. They live in the repository, they are reviewed when modified, and they are never silently changed to make a metric look better. When an eval case is retired because the system's scope changed, the retirement is documented.

---

## Common Failure Modes

**Intake passes ambiguous intent downstream.** The specialist receives a poorly specified objective and generates an output that is technically correct but not what the user wanted. Intake validation must catch ambiguity, not just structural errors.

**Specialist output validated structurally but wrong semantically.** A Pydantic model confirms the output has the right shape. The gatekeeper — not the output schema — is responsible for semantic correctness.

**Gatekeeper over-rejects.** A gatekeeper tuned too conservatively rejects legitimate outputs, producing a system that frustrates users with constant refusals. Monitor rejection rates in production and calibrate.

**State not validated at handoffs.** A missing field in the pipeline state is passed to the next agent, which fails with a cryptic error rather than a clear "expected field X, received None."

**No iteration limit on retry loops.** An intake agent that fails and retries without limit will loop indefinitely. Every retry has a maximum count.

**Agent roles bleed.** The specialist starts doing safety checks because the system prompt was underspecified. The gatekeeper starts rephrasing the output instead of just auditing it. Clear role definitions with specific output schemas prevent this.

# Evaluation Patterns

*This document reflects current implementation patterns for evaluating AI systems. Evaluation tooling and best practices evolve rapidly — this document is updated as better methods are validated in practice.*

---

## Overview

Evaluation patterns are reusable implementations of the evaluation strategies defined in [lifecycle/evaluation.md](../lifecycle/evaluation.md). This document focuses on how to build these mechanisms, not when to use them.

---

## LLM-as-Judge Pattern

An LLM judge evaluates generated outputs against defined criteria. This pattern scales to open-ended outputs that cannot be matched against a fixed reference.

### Minimal Judge Implementation

```python
from pydantic import BaseModel, Field
from typing import Literal

class JudgeVerdict(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    reason: str = Field(
        description="Specific reason for the verdict. Empty string if PASS."
    )
    flags: list[str] = Field(
        description="List of specific issues identified. Empty list if PASS."
    )

class JudgeInput(BaseModel):
    query: str = Field(description="The original user query")
    response: str = Field(description="The generated response to evaluate")
    criteria: str = Field(description="The evaluation criteria to apply")
```

### Judge Prompt Structure

The judge prompt must be specific and unambiguous. Vague criteria produce inconsistent verdicts.

```
System: You are an impartial evaluator. Your task is to assess whether a response
meets the specified criteria. You must return a structured verdict.

Evaluate the following response against the criteria provided.
Return PASS only if the response fully satisfies all criteria.
Return FAIL if the response fails on any criterion.

Query: {query}
Response: {response}
Criteria: {criteria}
```

### Calibration Protocol

Before using a judge in production, validate it against human ratings:

1. Collect 50–100 examples with human-assigned verdicts (PASS/FAIL)
2. Run the judge on the same examples
3. Calculate agreement rate and Cohen's Kappa
4. Acceptable threshold: > 80% agreement, Kappa > 0.6
5. Review disagreements to identify systematic judge errors
6. Revise criteria definitions where disagreements cluster

A judge that has not been calibrated is an unvalidated assumption.

---

## Retrieval Evaluation Pattern

Retrieval quality must be evaluated independently of generation quality. End-to-end RAG evaluation masks retrieval failures where the LLM compensates with parametric knowledge.

### What to Measure

**Retrieval Precision@k:** Of the top-k retrieved chunks, what fraction were relevant?

```
precision@k = relevant_retrieved / k
```

**Retrieval Recall@k:** Of all relevant chunks in the corpus, what fraction appeared in the top-k?

```
recall@k = relevant_retrieved / total_relevant
```

**Context Faithfulness:** Does the generated answer reflect only the retrieved context, or does it introduce information not present in the chunks?

**Answer Relevance:** Is the generated answer responsive to the actual question asked?

### Building a Retrieval Evaluation Dataset

1. Select a representative sample of queries (50+ for initial evaluation)
2. For each query, manually identify the relevant chunks in the corpus
3. Run the retriever and record which chunks are returned
4. Calculate precision and recall against the manual relevance labels

This dataset is reusable. As the corpus and retrieval configuration change, re-run evaluation against the same query set to detect regressions.

---

## Structured Output Reliability Pattern

Measures the rate at which the LLM produces parseable, schema-valid structured output.

### What to Measure

**Parse success rate:** Percentage of outputs that parse successfully against the target schema.

**Field-level validity rate:** For each field in the output schema, what percentage of outputs contain a valid value for that field? (Identifies which fields are most prone to generation errors.)

**Retry impact:** Parse success rate on first attempt vs. after one retry. Quantifies the value of the retry loop.

### Measurement Protocol

```
total_attempts = N
first_attempt_successes = 0
retry_successes = 0
failures = 0

for each test input:
    attempt 1:
        try parse → first_attempt_success
        except ParseError:
            attempt 2:
                try parse → retry_success
                except ParseError → failure

parse_success_rate = (first_attempt_successes + retry_successes) / total_attempts
first_attempt_rate = first_attempt_successes / total_attempts
retry_value = retry_successes / (total_attempts - first_attempt_successes)
```

Acceptable thresholds depend on the domain. For structured outputs in production systems: first-attempt parse success rate > 90%, total success rate (with retry) > 97%.

---

## Red Flag Detection Pattern

A lightweight classifier that flags outputs containing specific risk patterns before they reach the LLM-as-judge.

**Use when:** The system operates in a domain with well-defined red flags that can be detected with a rule-based or lightweight classifier before invoking a more expensive judge.

**Pattern:**

```python
RED_FLAG_PATTERNS = [
    r"specific dosage of \d+",          # medical dosage without qualification
    r"guaranteed (return|profit)",       # financial guarantee claims
    r"you should (stop|start) taking",   # direct medical instruction
]

def detect_red_flags(text: str) -> list[str]:
    flags = []
    for pattern in RED_FLAG_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            flags.append(pattern)
    return flags
```

Red flag detection runs before the LLM judge. Outputs that trigger red flags are rejected immediately, without incurring the latency and cost of a full judge evaluation.

This pattern works best for well-defined, rule-based risk categories. It is not a substitute for the LLM judge — it is a fast pre-filter.

---

## Evaluation Reporting

Every evaluation run produces a report with:

- Run date and system version
- Dataset description (size, construction method, coverage)
- Results per metric with confidence intervals where applicable
- Comparison to previous run (regression detection)
- Failure analysis: top failure categories with examples
- Go / No-Go recommendation with justification

Store evaluation reports alongside the code. They are the evidence that the system works.

---

## Anti-Pattern: Self-Evaluation

Do not use the same model to generate outputs and evaluate them. Self-evaluation introduces self-preference bias — the model rates its own outputs more favorably than an independent judge would.

Use a different model for judging, or use human raters for calibration, whenever the evaluation dataset is used to make a deployment decision.

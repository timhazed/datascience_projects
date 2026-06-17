# Retrieval Eval Report — 20260505_205100

**Verdict: FAIL**

## Run Configuration
- Model: `gemma4:e4b`
- Embed: `nomic-embed-text`
- Judge: `openai/gpt-oss-120b`
- ChromaDB: `/Users/timhazed/src/research_projects/developer memory/src/cli/data/chroma`
- Collection: `developer_memory_v1`
- Cases run: 20
- Total elapsed: 405.7s

## Corpus Metrics

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| File Recall@k | 0.812 | ≥0.75 | ✓ |
| Identifier Precision | 0.854 | ≥0.65 | ✓ |
| Answer Grounded (stem-match) | 0.571 | ≥0.7 | ✗ |
| LLM Judge (diagnostic) | 0.429 | — | — |

## Per-Case Results

| ID | Type | Difficulty | Recall@k | ID Prec | Grounded | Judge | Composite | Elapsed |
|----|------|------------|----------|---------|----------|-------|-----------|---------|
| autogen-001 | identifier_search | simple | 1 | 1.000 | N/A | N/A | 1.000 | 19.3s |
| autogen-002 | pattern_lookup | semantic | 1 | 1.000 | 1 | 1 | 1.000 | 13.8s |
| autogen-003 | identifier_search | semantic | 1 | 1.000 | N/A | N/A | 1.000 | 13.1s |
| healthcare-001 | pattern_lookup | semantic | 1 | 1.000 | 1 | 1 | 1.000 | 22.1s |
| healthcare-002 | methodology | synthesis | 1 | 1.000 | 0 | 0 | 0.667 | 22.6s |
| healthcare-003 | pattern_lookup | semantic | 0 | 0.667 | N/A | N/A | 0.334 | 19.8s |
| healthcare-004 | identifier_search | simple | 1 | 1.000 | 1 | 1 | 1.000 | 20.0s |
| newsgenie-001 | pattern_lookup | semantic | 1 | 0.667 | N/A | N/A | 0.834 | 22.5s |
| newsgenie-002 | identifier_search | simple | 0 | 0.667 | 1 | 0 | 0.556 | 18.9s |
| hr-001 | pattern_lookup | semantic | 1 | 1.000 | N/A | N/A | 1.000 | 18.8s |
| crewai-001 | pattern_lookup | semantic | 1 | 0.750 | N/A | N/A | 0.875 | 18.1s |
| cross-001 | cross_repo | synthesis | N/A | 0.800 | N/A | N/A | 0.800 | 26.5s |
| cross-002 | cross_repo | synthesis | N/A | 1.000 | N/A | N/A | 1.000 | 36.1s |
| cross-003 | cross_repo | simple | N/A | 0.800 | N/A | N/A | 0.800 | 12.0s |
| cross-004 | cross_repo | synthesis | N/A | 0.500 | N/A | N/A | 0.500 | 26.1s |
| methodology-001 | methodology | synthesis | 0 | N/A | 0 | 0 | 0.000 | 17.6s |
| methodology-002 | methodology | semantic | 1 | N/A | N/A | N/A | 1.000 | 11.8s |
| methodology-003 | methodology | semantic | 1 | N/A | 0 | 0 | 0.500 | 14.0s |
| pattern-001 | pattern_lookup | synthesis | 1 | 1.000 | N/A | N/A | 1.000 | 25.3s |
| pattern-002 | pattern_lookup | semantic | 1 | 0.667 | N/A | N/A | 0.834 | 27.3s |

## Failing Cases (composite < 0.5)

### healthcare-003 — How does the healthcare assistant guard against off-topic or unsafe queries
- Notes: Safety pattern — widened to k=10 since test files rank above source files for this query
- Retrieved: ['projects/agentic healthcare assistant/docs/Architecture.md', 'projects/agentic healthcare assistant/README.pdf', 'projects/agentic healthcare assistant/tests/agents/test_intent_guard_node.py', 'projects/agentic healthcare assistant/tests/agents/test_intent_guard_node.py', 'projects/agentic healthcare assistant/docs/Architecture.md']
- Expected: ['projects/agentic healthcare assistant/src/agents/intent_guard_node.py', 'projects/agentic healthcare assistant/src/chains/intent_guard_chain.py']
- Final answer preview: The healthcare assistant employs a multi-layered approach, centered on a dedicated **intent guardrail**, to prevent processing of off-topic or unsafe queries.

1.  **Intent Classification:** The syste

### methodology-001 — Summarize my agentic healthcare assistant case study
- Notes: The query that failed in production — case study markdown must surface; widened to k=10 since architecture/README docs dominate top results
- Retrieved: ['projects/agentic healthcare assistant/experiments/appointment/findings/findings.md', 'projects/agentic healthcare assistant/docs/Architecture.md', 'projects/agentic healthcare assistant/docs/Architecture.md', 'projects/agentic healthcare assistant/docs/Architecture.md', 'projects/agentic healthcare assistant/README.pdf']
- Expected: ['methodology/case_studies/agentic_healthcare_assistant.md']
- Final answer preview: The agentic healthcare assistant is a sophisticated, LangGraph-orchestrated system designed to manage complex patient care tasks from a single natural language interface. Its core mechanism involves a

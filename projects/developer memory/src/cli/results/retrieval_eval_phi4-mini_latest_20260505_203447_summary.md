# Retrieval Eval Report — 20260505_203447

**Verdict: PASS**

## Run Configuration
- Model: `phi4-mini:latest`
- Embed: `nomic-embed-text`
- Judge: `openai/gpt-oss-120b`
- ChromaDB: `/Users/timhazed/src/research_projects/developer memory/src/cli/data/chroma`
- Collection: `developer_memory_v1`
- Cases run: 20
- Total elapsed: 116.0s

## Corpus Metrics

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| File Recall@k | 0.812 | ≥0.75 | ✓ |
| Identifier Precision | 0.825 | ≥0.65 | ✓ |
| Answer Grounded (stem-match) | 0.714 | ≥0.7 | ✓ |
| LLM Judge (diagnostic) | 0.857 | — | — |

## Per-Case Results

| ID | Type | Difficulty | Recall@k | ID Prec | Grounded | Judge | Composite | Elapsed |
|----|------|------------|----------|---------|----------|-------|-----------|---------|
| autogen-001 | identifier_search | simple | 1 | 1.000 | N/A | N/A | 1.000 | 3.4s |
| autogen-002 | pattern_lookup | semantic | 1 | 1.000 | 1 | 1 | 1.000 | 3.3s |
| autogen-003 | identifier_search | semantic | 1 | 1.000 | N/A | N/A | 1.000 | 3.6s |
| healthcare-001 | pattern_lookup | semantic | 1 | 1.000 | 1 | 1 | 1.000 | 5.6s |
| healthcare-002 | methodology | synthesis | 1 | 1.000 | 1 | 0 | 1.000 | 7.7s |
| healthcare-003 | pattern_lookup | semantic | 0 | 0.667 | N/A | N/A | 0.334 | 5.9s |
| healthcare-004 | identifier_search | simple | 1 | 1.000 | 1 | 1 | 1.000 | 5.6s |
| newsgenie-001 | pattern_lookup | semantic | 1 | 0.667 | N/A | N/A | 0.834 | 6.6s |
| newsgenie-002 | identifier_search | simple | 0 | 0.667 | 0 | 1 | 0.222 | 8.3s |
| hr-001 | pattern_lookup | semantic | 1 | 1.000 | N/A | N/A | 1.000 | 6.6s |
| crewai-001 | pattern_lookup | semantic | 1 | 0.500 | N/A | N/A | 0.750 | 5.2s |
| cross-001 | cross_repo | synthesis | N/A | 0.800 | N/A | N/A | 0.800 | 7.0s |
| cross-002 | cross_repo | synthesis | N/A | 0.750 | N/A | N/A | 0.750 | 5.6s |
| cross-003 | cross_repo | simple | N/A | 0.800 | N/A | N/A | 0.800 | 3.5s |
| cross-004 | cross_repo | synthesis | N/A | 0.500 | N/A | N/A | 0.500 | 7.3s |
| methodology-001 | methodology | synthesis | 0 | N/A | 0 | 1 | 0.000 | 5.8s |
| methodology-002 | methodology | semantic | 1 | N/A | N/A | N/A | 1.000 | 4.9s |
| methodology-003 | methodology | semantic | 1 | N/A | 1 | 1 | 1.000 | 6.1s |
| pattern-001 | pattern_lookup | synthesis | 1 | 1.000 | N/A | N/A | 1.000 | 7.5s |
| pattern-002 | pattern_lookup | semantic | 1 | 0.667 | N/A | N/A | 0.834 | 6.4s |

## Failing Cases (composite < 0.5)

### healthcare-003 — How does the healthcare assistant guard against off-topic or unsafe queries
- Notes: Safety pattern — widened to k=10 since test files rank above source files for this query
- Retrieved: ['projects/agentic healthcare assistant/docs/Architecture.md', 'projects/agentic healthcare assistant/README.pdf', 'projects/agentic healthcare assistant/tests/agents/test_intent_guard_node.py', 'projects/agentic healthcare assistant/tests/agents/test_intent_guard_node.py', 'projects/agentic healthcare assistant/docs/Architecture.md']
- Expected: ['projects/agentic healthcare assistant/src/agents/intent_guard_node.py', 'projects/agentic healthcare assistant/src/chains/intent_guard_chain.py']
- Final answer preview: The healthcare assistant employs multiple safety mechanisms to handle off-topic or unsafe queries: 

1. Intent guards use Large Language Models for classification of user inputs as safe (`intent_guard

### newsgenie-002 — How does the newsgenie system fetch news from the Guardian API
- Notes: API client lookup — guardian_client.py is the sole file; widened to k=10 since architecture docs rank above source for this query
- Retrieved: ['projects/newsgenie/docs/ARCHITECTURE.md', 'projects/newsgenie/tests/agents/test_business_agent.py', 'PROJECTS.md', 'projects/newsgenie/docs/ARCHITECTURE.md', 'projects/newsgenie/src/graph/agent_nodes.py']
- Expected: ['projects/newsgenie/src/api/guardian_client.py']
- Final answer preview: The newsgenie system fetches Guardian-related sports content using both The Guardian and ESPN APIs, with articles from The Guardian prioritized over ESPN scorecards; HTTP errors are treated as warning

### methodology-001 — Summarize my agentic healthcare assistant case study
- Notes: The query that failed in production — case study markdown must surface; widened to k=10 since architecture/README docs dominate top results
- Retrieved: ['projects/agentic healthcare assistant/experiments/appointment/findings/findings.md', 'projects/agentic healthcare assistant/docs/Architecture.md', 'projects/agentic healthcare assistant/docs/Architecture.md', 'projects/agentic healthcare assistant/docs/Architecture.md', 'projects/agentic healthcare assistant/README.pdf']
- Expected: ['methodology/case_studies/agentic_healthcare_assistant.md']
- Final answer preview: The agentic healthcare assistant project is designed to streamline patient management by integrating appointment booking, history retrieval and updating, as well as disease information search into one

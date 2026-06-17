# Retrieval Eval Report — 20260505_205748

**Verdict: PASS**

## Run Configuration
- Model: `gpt-oss:20b`
- Embed: `nomic-embed-text`
- Judge: `openai/gpt-oss-120b`
- ChromaDB: `/Users/timhazed/src/research_projects/developer memory/src/cli/data/chroma`
- Collection: `developer_memory_v1`
- Cases run: 20
- Total elapsed: 198.0s

## Corpus Metrics

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| File Recall@k | 0.812 | ≥0.75 | ✓ |
| Identifier Precision | 0.844 | ≥0.65 | ✓ |
| Answer Grounded (stem-match) | 0.857 | ≥0.7 | ✓ |
| LLM Judge (diagnostic) | 1.000 | — | — |

## Per-Case Results

| ID | Type | Difficulty | Recall@k | ID Prec | Grounded | Judge | Composite | Elapsed |
|----|------|------------|----------|---------|----------|-------|-----------|---------|
| autogen-001 | identifier_search | simple | 1 | 1.000 | N/A | N/A | 1.000 | 16.3s |
| autogen-002 | pattern_lookup | semantic | 1 | 1.000 | 1 | 1 | 1.000 | 8.5s |
| autogen-003 | identifier_search | semantic | 1 | 0.750 | N/A | N/A | 0.875 | 8.4s |
| healthcare-001 | pattern_lookup | semantic | 1 | 1.000 | 1 | 1 | 1.000 | 10.0s |
| healthcare-002 | methodology | synthesis | 1 | 1.000 | 1 | 1 | 1.000 | 11.7s |
| healthcare-003 | pattern_lookup | semantic | 0 | 0.667 | N/A | N/A | 0.334 | 8.5s |
| healthcare-004 | identifier_search | simple | 1 | 1.000 | 1 | 1 | 1.000 | 8.4s |
| newsgenie-001 | pattern_lookup | semantic | 1 | 1.000 | N/A | N/A | 1.000 | 8.4s |
| newsgenie-002 | identifier_search | simple | 0 | 0.667 | 1 | 1 | 0.556 | 10.8s |
| hr-001 | pattern_lookup | semantic | 1 | 1.000 | N/A | N/A | 1.000 | 8.2s |
| crewai-001 | pattern_lookup | semantic | 1 | 0.750 | N/A | N/A | 0.875 | 7.9s |
| cross-001 | cross_repo | synthesis | N/A | 0.800 | N/A | N/A | 0.800 | 13.6s |
| cross-002 | cross_repo | synthesis | N/A | 0.750 | N/A | N/A | 0.750 | 9.9s |
| cross-003 | cross_repo | simple | N/A | 0.800 | N/A | N/A | 0.800 | 5.3s |
| cross-004 | cross_repo | synthesis | N/A | 0.500 | N/A | N/A | 0.500 | 11.9s |
| methodology-001 | methodology | synthesis | 0 | N/A | 0 | 1 | 0.000 | 9.5s |
| methodology-002 | methodology | semantic | 1 | N/A | N/A | N/A | 1.000 | 8.3s |
| methodology-003 | methodology | semantic | 1 | N/A | 1 | 1 | 1.000 | 8.1s |
| pattern-001 | pattern_lookup | synthesis | 1 | 1.000 | N/A | N/A | 1.000 | 11.0s |
| pattern-002 | pattern_lookup | semantic | 1 | 0.667 | N/A | N/A | 0.834 | 13.3s |

## Failing Cases (composite < 0.5)

### healthcare-003 — How does the healthcare assistant guard against off-topic or unsafe queries
- Notes: Safety pattern — widened to k=10 since test files rank above source files for this query
- Retrieved: ['projects/agentic healthcare assistant/docs/Architecture.md', 'projects/agentic healthcare assistant/README.pdf', 'projects/agentic healthcare assistant/tests/agents/test_intent_guard_node.py', 'projects/agentic healthcare assistant/tests/agents/test_intent_guard_node.py', 'projects/agentic healthcare assistant/docs/Architecture.md']
- Expected: ['projects/agentic healthcare assistant/src/agents/intent_guard_node.py', 'projects/agentic healthcare assistant/src/chains/intent_guard_chain.py']
- Final answer preview: The assistant uses a dedicated **intent‑guard node** that runs an LLM‑based classifier (`_GUARD_PROMPT`) on every user query.  
If the classifier returns “UNSAFE” (e.g., an off‑topic question or a req

### methodology-001 — Summarize my agentic healthcare assistant case study
- Notes: The query that failed in production — case study markdown must surface; widened to k=10 since architecture/README docs dominate top results
- Retrieved: ['projects/agentic healthcare assistant/experiments/appointment/findings/findings.md', 'projects/agentic healthcare assistant/docs/Architecture.md', 'projects/agentic healthcare assistant/docs/Architecture.md', 'projects/agentic healthcare assistant/docs/Architecture.md', 'projects/agentic healthcare assistant/README.pdf']
- Expected: ['methodology/case_studies/agentic_healthcare_assistant.md']
- Final answer preview: The **Agentic Healthcare Assistant** is a LangGraph‑based clinical assistant that decomposes natural‑language requests into a sequential plan (e.g., `resolve_patient → retrieve_history → book_appointm

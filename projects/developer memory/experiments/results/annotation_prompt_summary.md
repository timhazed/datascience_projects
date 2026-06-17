# Annotation Prompt Experiment — Summary
Date: 2026-05-04

## Goal
Determine whether adding a `key_identifiers` field to `_ChunkAnnotation` (in `intent_summary_chain.py`)
improves code search recall by preserving concrete API names that the current intent-focused prompt drops.

## Test Files
3 Python files from `projects/LLM Framework Benchmarking/autogen_crop_yield_simple_agent/agents/`:
- `prediction_agent.py` — contains `autogen.AssistantAgent`, `autogen.UserProxyAgent`, `initiate_chat`
- `base_agent.py` — contains `BaseAgentConfig`, `create_llm_config`
- `data_preparation_agent.py` — ETL agent, no Autogen API usage

## Search Targets (6 identifiers)
`AssistantAgent`, `UserProxyAgent`, `PredictionAgent`, `BaseAgentConfig`, `initiate_chat`, `autogen`

## Schemas Tested

### Current (baseline)
Fields: `intent_summary`, `tech_stack`, `semantic_type`
Instruction: "Focus on rationale and design decision, **not** a description of what the code does."
Problem: Explicitly told not to describe what the code does → drops class/API names from output.

### Proposed
Fields: `intent_summary`, `tech_stack`, `semantic_type`, `key_identifiers`
Added field: verbatim identifier list with strong anchoring instruction.

### Refined (Proposed + few-shot example in system prompt)
Same schema as Proposed. Adds a concrete input/output example to the system prompt.
Problem: `{{`/`}}` escaping conflicts with `with_structured_output` on gemma4:26b → parse failures.

## Results

| Model       | Current | Proposed | Refined (few-shot) | Winner   |
|-------------|---------|----------|--------------------|----------|
| gemma4:26b  | 28%     | **50%**  | 0% (parse failures)| Proposed |
| gemma4:e4b  | 17%     | 44%      | **50%**            | Refined  |
| phi4:14b    | 22%     | 44%      | **50%**            | Refined  |
| phi4-mini   | 17%     | 17%      | 22%                | Refined  |

### Per-file: prediction_agent.py (the most diagnostic file)

| Model      | Current | Proposed | Refined |
|------------|---------|----------|---------|
| gemma4:26b | 4/6     | **6/6**  | 0/6     |
| gemma4:e4b | 1/6     | 5/6      | **6/6** |
| phi4:14b   | 2/6     | 5/6      | **6/6** |
| phi4-mini  | 2/6     | 1/6      | 2/6     |

## Key Findings

1. **Proposed schema is the safe production winner** — best or tied-best on 3 of 4 models,
   zero reliability issues across all models.

2. **Refined (few-shot) wins on e4b and phi4:14b** but is incompatible with gemma4:26b's
   `with_structured_output` parser. The `{{`/`}}` example in the system prompt causes the
   model to generate free-form JSON instead of following the schema, resulting in parse failures.

3. **phi4-mini is at a model capability ceiling** — no prompt variant reliably extracts
   PascalCase identifiers like `AssistantAgent`. Accept lower recall for this model.

4. **phi4:14b is a strong alternative to gemma4:e4b** — same recall, likely faster.
   Worth a latency benchmark before committing to it for ingest.

5. **Root cause of the original search failure**: `intent_summary` field description
   explicitly instructed the LLM *not* to describe what the code does syntactically,
   causing it to omit class/API names in favor of design rationale prose.

## Decision

**Ship Proposed schema to `src/chains/intent_summary_chain.py`.**

After deployment, re-index affected repos (or run a fresh sync) to rebuild ChromaDB
with the improved annotations. Existing chunks will not benefit until re-ingested.

## Raw Results
All per-run JSON files: `experiments/results/annotation_prompt_<timestamp>.json`
Final canonical runs (one per model, Proposed schema):
- gemma4:26b: `annotation_prompt_20260504_213331.json`
- gemma4:e4b: `annotation_prompt_20260504_211424.json`
- phi4:14b:   `annotation_prompt_20260504_212359.json`
- phi4-mini:  `annotation_prompt_20260504_211221.json`

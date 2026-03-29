# Exercise and Recovery Coach

A SOLO (Spec-driven, Agentic, Audit-heavy) multi-agent system for personalized exercise and recovery planning, built on Python 3.13 and LangChain.

## Overview

The system coordinates four specialized AI agents through an intent-based routing system:

| Agent | Role | Output |
|-------|------|--------|
| **Intake Agent** | Extracts structured context from free-text conversation | `UserContext` |
| **Kinesiologist Specialist** | Produces evidence-based workout programs | `WorkoutPlan` |
| **Recovery Specialist** | Creates sequenced mobility/recovery routines | `RecoveryPlan` |
| **Clinical Gatekeeper** | Audits plans for safety and red flags | `AuditLog` |

The **State Router** classifies user intent and directs requests to the appropriate workflow while enforcing safety blocks for medical red flags via a four-layer `SafetyJudge` (rule-based fast path + LLM semantic backstop).

LLM provider is **OpenAI-only** (Groq removed due to lack of strict JSON schema support). All agents return validated Pydantic models via LangChain's `with_structured_output()`.

## Features

- **Strict JSON outputs** validated against `WorkoutPlan` and `RecoveryPlan` Pydantic schemas
- **Four-layer safety system** via `SafetyJudge`:
  - L1: `RedFlagScanner` on stored user context (known conditions)
  - L1b: `RedFlagScanner` deep scan on the current message text (first-turn detection)
  - L2: Acute-pattern string list (chest pain, electrical zap, etc.)
  - L3: LLM semantic backstop for novel phrasings (e.g. "fluttering in my chest")
  - Volume and ratio validators for exercise prescription
  - Clinical Gatekeeper LLM audit on every generated plan
- **Multi-turn conversational intake** with context persistence and minimal-intake guardrails
- **Intent classification** routing to integrated, recovery-only, or safety-block workflows
- **Dual interfaces**: CLI and Gradio web UI
- **Validation runner** for benchmarking intent routing, safety blocking, and output quality

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd exercise-and-health-coach

# Install dependencies with Poetry
poetry install

# Copy environment template and add API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
```

## Quick Start

### CLI Interface

```bash
poetry run python -m src.cli.app
```

**CLI options:**

| Option | Description |
|--------|-------------|
| `-m`, `--message` | Process a single message (non-interactive) |
| `-v`, `--verbose` | Show detailed JSON output (workout plan, recovery plan, audit log) |
| `-d`, `--debug` | Show LangChain debug output (LLM prompts and responses) |
| `--log-level` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `WARNING`) |

```bash
# Interactive mode
poetry run python -m src.cli.app

# Single message
poetry run python -m src.cli.app -m "I want to build muscle"

# Verbose: show JSON plans in output
poetry run python -m src.cli.app -v

# Debug: show LLM prompts and responses
poetry run python -m src.cli.app -d
```

### Gradio Web Interface

```bash
poetry run python -m src.ui.app
```

Then open http://localhost:7860 in your browser.

## Project Structure

```
src/
├── agents/                     # AI agent implementations
│   ├── base_agent.py           # Abstract base with structured output + retry logic
│   ├── intake_agent.py         # Context extraction from conversation
│   ├── kinesiologist_specialist.py  # Workout plan generation
│   ├── recovery_specialist.py  # Recovery/mobility plan generation
│   └── clinical_gatekeeper.py  # Safety audit agent
├── config/                     # Settings management
│   └── settings.py             # Pydantic settings + config loader
├── llm/                        # LLM factory (OpenAI-only)
│   └── llm_factory.py          # get_llm(), get_llm_for_agent()
├── models/                     # Pydantic schemas and enums
│   ├── schemas.py              # UserContext, WorkoutPlan, RecoveryPlan, AuditLog
│   └── enums.py                # FitnessGoal, MuscleGroup, Equipment, etc.
├── orchestrator/               # Request coordination
│   ├── coach.py                # ExerciseCoach main entry point
│   ├── state_router.py         # Intent-based workflow routing
│   └── intent_classifier.py    # NLP intent classification
├── prompts/                    # Prompt templates and helpers
├── skills/                     # Runtime-loaded SKILL.md persona files
│   └── skill_loader.py         # load_skill() utility
├── state/                      # Conversation state management
│   └── conversation_state.py   # ConversationState, create_session()
├── util/                       # Utilities
│   ├── validation_runner.py         # Benchmark runner for quality evaluation
│   └── kinesiologist_eval_runner.py # Router routing accuracy evaluation (pipeline + router modes)
├── validators/                 # Safety and constraint validators
│   ├── base.py                 # BaseValidator, ValidatorChain
│   ├── red_flag_scanner.py     # Medical red flag detection
│   ├── safety_judge.py         # Four-layer safety screener (L1–L3 regex + L4 LLM)
│   ├── volume_validator.py     # Exercise volume constraints (18-24 sets/muscle)
│   └── ratio_validator.py      # Push:pull ratio balance (1:1 target)
├── workflows/                  # Plan generation workflows
│   ├── base.py                 # BaseWorkflow, WorkflowResult
│   ├── integrated_workflow.py  # Exercise + Recovery + Audit
│   └── recovery_only_workflow.py  # Recovery + Audit
├── cli/                        # Command-line interface
│   └── app.py
└── ui/                         # Gradio web interface
    └── app.py

skills/                         # Agent persona definitions
├── kinesiologist/SKILL.md      # S&C coach persona + volume/ratio rules
├── recovery/SKILL.md           # Recovery specialist persona + modality rules
├── gatekeeper/SKILL.md         # Safety auditor persona + blocking criteria
└── intake/SKILL.md             # Intake coordinator persona + extraction rules

data/                           # Validation datasets
├── Validation_Questions.jsonl  # Full question set for benchmarking
└── Validation_Questions_Mini.jsonl  # Quick smoke test set
```

## Configuration

Configuration is loaded from `config.yaml` (override via `CONFIG_PATH` env var).

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required) |
| `CONFIG_PATH` | Path to config file (optional, defaults to `config.yaml`) |
| `LOG_LEVEL` | Logging level (optional, defaults to `WARNING`) |

### Sample config.yaml

```yaml
default_provider: "openai"

providers:
  openai:
    model: "gpt-4o-mini"
    temperature: 0.3
    max_tokens: 2048

agents:
  intake:
    provider: "openai"
    temperature: 0.2    # Low for accurate extraction
  kinesiologist:
    provider: "openai"
    temperature: 0.4    # Slightly creative for exercise variety
  recovery:
    provider: "openai"
    temperature: 0.3
  gatekeeper:
    provider: "openai"
    temperature: 0.1    # Very conservative for safety
  safety:
    provider: "openai"
    temperature: 0.0    # Deterministic — binary emergency decision
    max_tokens: 5       # YES/NO = 1 token; hard cap prevents accidental verbosity

exercise:
  min_sets_per_muscle_group: 18
  max_sets_per_muscle_group: 24
  target_rpe_min: 7
  target_rpe_max: 9

recovery:
  static_hold_min_seconds: 30
  static_hold_max_seconds: 60
  modality_order:
    - "smr"
    - "dynamic_mobility"
    - "static_stretching"

workflow:
  max_retries: 3
  timeout_seconds: 120
```

## Safety Features

### 1. SafetyJudge (`safety_judge.py`) — Four-layer screener

Every message passes through `SafetyJudge.assess()` before any plan is generated. Layers run fast → slow; the first non-empty result short-circuits the remaining layers so the LLM is only called when the rule-based layers produce no signal.

| Layer | Mechanism | Catches |
|-------|-----------|---------|
| **L1** `_scan_context` | `RedFlagScanner` on stored `UserContext` | Conditions already in the user's profile (prior turns) |
| **L1b** `_scan_message_deep` | `RedFlagScanner` via temp context with message appended | First-turn messages: "I have arrhythmia", cardiac terms |
| **L2** `_scan_message_regex` | Plain-string `_ACUTE_PATTERNS` list | Known acute signals: "chest pain", "electrical zap", "dizzy" |
| **L3** `_run_llm_assessment` | YES/NO LLM prompt; 2-attempt retry; fail-open | Novel phrasings not in any list: "fluttering in my chest" |

- LLM is **not called** when L1, L1b, or L2 fires — no cost for known patterns
- On LLM failure (both retries exhausted), returns `[]` (fail-open) — routing is never crashed
- `temperature: 0.0`, `max_tokens: 5` — deterministic, minimal-cost binary decision

### 2. Red Flag Scanner (`red_flag_scanner.py`)

Rule-based pattern matching underpinning L1 and L1b:

| Category | Examples | Action |
|----------|----------|--------|
| **Neurological** | Numbness, tingling, radiating pain, sciatica | Block |
| **Cardiovascular** | Chest pain, shortness of breath, palpitations, arrhythmia | Block |
| **Acute Injury** | Recent fracture, torn ligament, acute inflammation | Block |
| **Inflammatory** | Active infection, severe swelling, fever | Block |

### 3. Volume Validator (`volume_validator.py`)

Enforces evidence-based volume constraints:
- **Target**: 18-24 sets per muscle group per week
- **Over-volume**: Triggers retry (blocking error)
- **Under-volume**: Warning only (non-blocking)

### 4. Ratio Validator (`ratio_validator.py`)

Ensures balanced programming:
- **Push:Pull ratio**: Target 1:1 (acceptable 0.8-1.2)
- **Quad:Hip ratio**: Target balanced (acceptable 0.7-1.4)
- All violations generate warnings (non-blocking)

### 5. Clinical Gatekeeper Agent

LLM-based audit of every generated plan:
- Reviews for contraindications and safety concerns
- Can reject or request modifications
- Generates `AuditLog` with status (APPROVED, MODIFIED, REJECTED)

## Evaluation Runners

### Kinesiologist Evaluation Runner

Deterministic, LLM-free benchmark of the `StateRouter`'s routing decisions against kinesiologist ground truth. No API key required — the runner simulates intake extraction with regex heuristics.

```bash
# Pipeline mode (default) — pre-populates state from each message; full-pipeline simulation
poetry run python -m src.util.kinesiologist_eval_runner

# Router mode — routes with empty state; tests router inference in isolation
poetry run python -m src.util.kinesiologist_eval_runner --mode router

# Custom input file
poetry run python -m src.util.kinesiologist_eval_runner \
    --input data/Kinesiologist_and_Recovery_full_evaluation.json

# Save report to a specific directory
poetry run python -m src.util.kinesiologist_eval_runner --output-dir results/
```

**Two evaluation modes:**

| Mode | State at routing | Use case |
|------|-----------------|----------|
| `pipeline` (default) | Pre-populated from message via regex extraction | End-to-end routing accuracy |
| `router` | Empty (no context) | Router inference logic in isolation |

**Evaluation case format (JSON array):**

```json
[
  {
    "question": "I'm 38, no injuries. I slept on my neck wrong. I can't look over my left shoulder.",
    "predicted_response_failure": "What is your weight and fitness goal?",
    "ground_truth": "Trigger Recovery Workflow. Zero biometrics required for acute neck stiffness."
  }
]
```

A timestamped JSON report is written to the output directory after each run.

### Validation Runner

Benchmark the full pipeline (LLM calls included) against a labeled JSONL question set to measure routing accuracy, red-flag detection, and false positive rates.

```bash
# Full validation run
poetry run python -m src.util.validation_runner data/Validation_Questions.jsonl -v

# Quick smoke test
poetry run python -m src.util.validation_runner data/Validation_Questions_Mini.jsonl -v

# Custom output directory
poetry run python -m src.util.validation_runner data/Validation_Questions.jsonl -o results/ -v
```

Use `-v` to print progress (e.g. `[1/10] Processing: ...`) to stdout during the run.

**Summary statistics:**
- **Intent accuracy**: Percentage of correct intent classifications
- **Red-flag misses**: Questions that should trigger SAFETY_BLOCK but did not
- **False blocks**: Non-red-flag questions incorrectly blocked

Use these metrics to track regressions when adjusting prompts, thresholds, or model parameters.

## Workflow Types

The `StateRouter` classifies requests into one of six workflows:

| Workflow | Trigger | Agents Used |
|----------|---------|-------------|
| `INTEGRATED` | Exercise request with complete intake | Kinesiologist + Recovery + Gatekeeper |
| `RECOVERY_ONLY` | Recovery/mobility request (acute or complete intake) | Recovery + Gatekeeper |
| `RECOVERY_FOLLOWUP` | Short follow-up question in an active recovery thread | Direct response, no plan regeneration |
| `INTAKE_NEEDED` | Missing required user context | Intake Agent prompts for info |
| `SAFETY_BLOCK` | Red flags detected by SafetyJudge (L1–L3) | Immediate rejection with guidance |
| `GENERAL_RESPONSE` | Q&A, clarification, greeting | Direct response, no plan generation |

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src --cov-report=term-missing

# Run specific test file
poetry run pytest tests/unit/test_agents.py -v

# Run integration tests only
poetry run pytest tests/integration/ -v
```

### Code Quality

```bash
# Lint with Ruff
poetry run ruff check src tests

# Format with Ruff
poetry run ruff format src tests
```

### Coverage Requirements

Minimum 85% coverage enforced. Excluded from coverage:
- `src/cli/*`
- `src/ui/*`
- `*/__init__.py`

## Architecture

High-level flow:

```
User Input
    │
    ▼
┌─────────────┐
│ Intake Agent│ ──▶ Extract UserContext
└─────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ StateRouter                                  │
│                                              │
│  IntentClassifier ──▶ EXERCISE / RECOVERY /  │
│                        INTEGRATED / INTAKE   │
│                                              │
│  Inference (no intake friction):             │
│    _GOAL_VOCABULARY dict                     │
│    \bgym\b + _GYM_NEG_PATTERN               │
│    experience level heuristics               │
│                                              │
│  SafetyJudge.assess(message, context)        │
│    L1  RedFlagScanner on UserContext         │
│    L1b RedFlagScanner deep scan on message   │
│    L2  _ACUTE_PATTERNS string list           │
│    L3  LLM YES/NO backstop (novel symptoms)  │
└──────────────────────────────────────────────┘
    │
    ├─── SAFETY_BLOCK ──────▶ Reject with guidance
    │
    ├─── INTAKE_NEEDED ─────▶ Prompt for missing info
    │
    ├─── GENERAL_RESPONSE ──▶ Direct Q&A response
    │
    ├─── RECOVERY_FOLLOWUP ─▶ Short follow-up (active recovery thread)
    │
    └─── INTEGRATED / RECOVERY_ONLY
            │
            ▼
    ┌───────────────┐
    │   Workflow    │ ──▶ Generate plans
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │  Gatekeeper   │ ──▶ Audit for safety
    └───────────────┘
            │
            ▼
    Structured JSON Response
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed documentation.

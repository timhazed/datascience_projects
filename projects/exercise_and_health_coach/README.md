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

The **State Router** classifies user intent and directs requests to the appropriate workflow while enforcing safety blocks for medical red flags.

LLM provider is **OpenAI-only** (Groq removed due to lack of strict JSON schema support). All agents return validated Pydantic models via LangChain's `with_structured_output()`.

## Features

- **Strict JSON outputs** validated against `WorkoutPlan` and `RecoveryPlan` Pydantic schemas
- **Multi-layer safety system**:
  - Rule-based red flag scanner (neurological, cardiovascular, acute injury, inflammatory)
  - Volume and ratio validators for exercise prescription
  - Clinical Gatekeeper audit on every generated plan
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
├── skills/                     # Runtime-loaded SKILL.md persona files
│   └── skill_loader.py         # load_skill() utility
├── state/                      # Conversation state management
│   └── conversation_state.py   # ConversationState, create_session()
├── util/                       # Utilities
│   └── validation_runner.py    # Benchmark runner for quality evaluation
├── validators/                 # Safety and constraint validators
│   ├── base.py                 # BaseValidator, ValidatorChain
│   ├── red_flag_scanner.py     # Medical red flag detection
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

### 1. Red Flag Scanner (`red_flag_scanner.py`)

Rule-based pattern matching for medical concerns that require professional evaluation:

| Category | Examples | Action |
|----------|----------|--------|
| **Neurological** | Numbness, tingling, radiating pain, sciatica | Block |
| **Cardiovascular** | Chest pain, shortness of breath, palpitations | Block |
| **Acute Injury** | Recent fracture, torn ligament, acute inflammation | Block |
| **Inflammatory** | Active infection, severe swelling, fever | Block |

### 2. Volume Validator (`volume_validator.py`)

Enforces evidence-based volume constraints:
- **Target**: 18-24 sets per muscle group per week
- **Over-volume**: Triggers retry (blocking error)
- **Under-volume**: Warning only (non-blocking)

### 3. Ratio Validator (`ratio_validator.py`)

Ensures balanced programming:
- **Push:Pull ratio**: Target 1:1 (acceptable 0.8-1.2)
- **Quad:Hip ratio**: Target balanced (acceptable 0.7-1.4)
- All violations generate warnings (non-blocking)

### 4. Clinical Gatekeeper Agent

LLM-based audit of every generated plan:
- Reviews for contraindications and safety concerns
- Can reject or request modifications
- Generates `AuditLog` with status (APPROVED, MODIFIED, REJECTED)

## Validation Runner

Benchmark the system against a labeled question set to measure routing accuracy, red-flag detection, and false positive rates.

### Running Validation

```bash
# Full validation run
poetry run python -m src.util.validation_runner data/Validation_Questions.jsonl -v

# Quick smoke test
poetry run python -m src.util.validation_runner data/Validation_Questions_Mini.jsonl -v

# Custom output directory
poetry run python -m src.util.validation_runner data/Validation_Questions.jsonl -o results/ -v
```

Use `-v` to print progress (e.g. `[1/10] Processing: ...`) to stdout during the run.

### Question Format (JSONL)

Each line is a JSON object with:

```json
{
  "question": "I'm 32, 85kg, intermediate. No injuries. Goal is vertical jump for volleyball.",
  "expected_intent": "exercise_request",
  "expected_response": "Optional reference response for manual review"
}
```

**Valid `expected_intent` values:**
- `exercise_request` - Workout plan requested
- `recovery_request` - Recovery/mobility plan requested
- `integrated_request` - Both exercise and recovery
- `red_flag_check` - Should trigger safety block
- `general_question` - Q&A, no plan generation

### Output Metrics

The runner generates a timestamped JSONL with per-question results:

```json
{
  "question": "...",
  "expected_intent": "exercise_request",
  "predicted_intent": "exercise_request",
  "workflow": "integrated",
  "predicted_response": "...",
  "is_rejected": false,
  "has_workout_plan": true,
  "has_recovery_plan": true,
  "audit_status": "approved"
}
```

**Summary statistics:**
- **Intent accuracy**: Percentage of correct intent classifications
- **Red-flag misses**: Questions with `expected_intent: red_flag_check` that were NOT blocked
- **False blocks**: Non-red-flag questions that were incorrectly blocked

Use these metrics to track regressions when adjusting prompts, thresholds, or model parameters.

## Workflow Types

The `StateRouter` classifies requests into one of five workflows:

| Workflow | Trigger | Agents Used |
|----------|---------|-------------|
| `INTEGRATED` | Exercise request with complete intake | Kinesiologist + Recovery + Gatekeeper |
| `RECOVERY_ONLY` | Recovery/mobility request | Recovery + Gatekeeper |
| `INTAKE_NEEDED` | Missing required user context | Intake Agent prompts for info |
| `SAFETY_BLOCK` | Red flags detected | Immediate rejection with guidance |
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
┌─────────────┐
│ StateRouter │ ──▶ Classify intent + check red flags
└─────────────┘
    │
    ├─── SAFETY_BLOCK ──▶ Reject with guidance
    │
    ├─── INTAKE_NEEDED ──▶ Prompt for missing info
    │
    ├─── GENERAL_RESPONSE ──▶ Direct Q&A response
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

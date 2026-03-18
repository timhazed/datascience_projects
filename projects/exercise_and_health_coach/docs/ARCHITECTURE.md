# Architecture

## System Overview

The Exercise and Recovery Coach implements a **Parallel-Sequential Flow** architecture following the SOLO Protocol (Spec-driven, Agentic, Audit-heavy).

```
┌─────────────────────────────────────────────────────────────────┐
│                        ExerciseCoach                            │
│                     (Main Orchestrator)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       IntakeAgent                               │
│              (Context Extraction from Conversation)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       StateRouter                               │
│        (IntentClassifier + RedFlagScanner + Routing)            │
└─────────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────┬───────────┼───────────┬─────────────┐
    ▼             ▼           ▼           ▼             ▼
INTEGRATED   RECOVERY    INTAKE      SAFETY       GENERAL
             _ONLY       _NEEDED     _BLOCK       _RESPONSE
```

## Workflow Types

The `StateRouter` classifies requests into one of five workflows:

| Workflow | Trigger | Action |
|----------|---------|--------|
| `INTEGRATED` | Exercise request + complete context | Kinesiologist → Recovery → Gatekeeper |
| `RECOVERY_ONLY` | Recovery/mobility request | Recovery → Gatekeeper |
| `INTAKE_NEEDED` | Missing required user context | Prompt for age, goals, equipment, etc. |
| `SAFETY_BLOCK` | Red flags detected | Immediate rejection with medical referral |
| `GENERAL_RESPONSE` | Q&A, greeting, clarification | Direct response, no plan generation |

### Workflow A: Integrated (Exercise + Recovery)

Used when user requests workout/exercise planning with complete context.

```
User Request
     │
     ▼
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   Intake    │ ──▶ │  Kinesiologist  │ ──▶ │    Recovery      │
│   Agent     │     │   Specialist    │     │   Specialist     │
└─────────────┘     └─────────────────┘     └──────────────────┘
                           │                        │
                           │    WorkoutPlan         │   RecoveryPlan
                           ▼                        ▼
                    ┌──────────────────────────────────────┐
                    │        Clinical Gatekeeper           │
                    │     (Safety Audit + Approval)        │
                    └──────────────────────────────────────┘
                                    │
                         ┌──────────┼──────────┐
                         ▼          ▼          ▼
                    APPROVED    MODIFIED    REJECTED
```

### Workflow B: Recovery Only

Used for mobility, stretching, or recovery requests.

```
User Request (pain/stiffness/recovery)
     │
     ▼
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Intake    │ ──▶ │    Recovery      │ ──▶ │    Clinical      │
│   Agent     │     │   Specialist     │     │   Gatekeeper     │
└─────────────┘     └──────────────────┘     └──────────────────┘
```

## Agent Architecture

### Base Agent Pattern

All agents inherit from `BaseAgent[T]` which provides:

```python
class BaseAgent(ABC, Generic[T]):
    """Abstract base for all agents."""

    def __init__(
        self,
        llm: BaseChatModel,
        max_retries: int = 3,
        validators: ValidatorChain | None = None,
    ):
        self.llm = llm
        self.max_retries = max_retries
        self.validators = validators

    # Core properties (implemented by subclasses)
    @property
    def name(self) -> str: ...
    @property
    def output_schema(self) -> type[T]: ...
    @property
    def system_prompt(self) -> str: ...

    # Shared functionality
    def invoke(**kwargs) -> T           # Sync execution with retry
    async def ainvoke(**kwargs) -> T    # Async execution with retry

    # Extensibility points
    def format_user_message(**kwargs) -> str
    def build_prompt(**kwargs) -> list[Message]
```

### Structured Output

Agents use LangChain's `with_structured_output()` with strict JSON schema mode:

```python
@property
def structured_llm(self) -> BaseChatModel:
    """Get LLM configured for structured output."""
    if self._structured_llm is None:
        self._structured_llm = self.llm.with_structured_output(
            self.output_schema,
            method="json_schema",  # Strict JSON schema enforcement
        )
    return self._structured_llm
```

This ensures:
- LLM output conforms exactly to the Pydantic schema
- No manual JSON parsing or string extraction
- Type-safe results with full Pydantic validation

### Retry Logic

Failed LLM calls are retried up to 3 times (configurable):

```python
for attempt in range(1, self.max_retries + 1):
    try:
        messages = self.build_prompt(**kwargs)
        result = self.structured_llm.invoke(messages)

        # Run validators if configured
        if self.validators:
            validation = self.validators.validate(result)
            if not validation.is_valid:
                continue  # Retry

        return result
    except (ValidationError, Exception) as e:
        last_error = e
        # Retry on next iteration

raise AgentExecutionError(f"Failed after {max_retries} attempts")
```

### Validator Integration

Agents can have validators wired into their processing chain:

```python
class KinesiologistSpecialist(BaseAgent[WorkoutPlan]):
    def __init__(self, llm, ...):
        validators = ValidatorChain()
        validators.add(VolumeValidator())   # 18-24 sets/muscle
        validators.add(RatioValidator())    # 1:1 push:pull
        super().__init__(llm=llm, validators=validators)
```

Available validators:
- **VolumeValidator**: Enforces 18-24 sets per muscle group (blocking on over-volume)
- **RatioValidator**: Checks push:pull ratio (1:1 target, 0.8-1.2 acceptable)
- **RedFlagScanner**: Pattern-based medical red flag detection

## Intent Classification

The `IntentClassifier` categorizes user messages into intent types:

```python
class IntentType(str, Enum):
    EXERCISE_REQUEST = "exercise_request"      # Workout plan needed
    RECOVERY_REQUEST = "recovery_request"      # Mobility/stretch plan
    INTEGRATED_REQUEST = "integrated_request"  # Both workout + recovery
    GENERAL_QUESTION = "general_question"      # Q&A about fitness
    CLARIFICATION = "clarification"            # Follow-up info
    GREETING = "greeting"                      # Hello, thanks, etc.
```

Intent classification feeds into the `StateRouter` to determine the appropriate workflow.

## Skill System

Each agent loads its persona and operational logic from a `SKILL.md` file at runtime:

```
skills/
├── kinesiologist/SKILL.md   # S&C coach persona + volume/ratio rules
├── recovery/SKILL.md        # Recovery specialist + modality sequence
├── gatekeeper/SKILL.md      # Safety auditor + blocking criteria
└── intake/SKILL.md          # Intake coordinator + extraction rules
```

This allows:
- Non-code changes to agent behavior
- Version-controlled skill definitions
- Easy A/B testing of prompts

```python
# In agent __init__
self._skill_content = load_skill("kinesiologist")

@property
def system_prompt(self) -> str:
    return f"{self._skill_content}\n{OPERATIONAL_INSTRUCTIONS}"
```

## State Management

### ConversationState

Persists across conversation turns:

```python
@dataclass
class ConversationState:
    session_id: str
    user_context: UserContext      # Accumulated user info
    turns: list[ConversationTurn]  # Message history
    intake_complete: bool          # Ready for workout generation?

    def update_context(self, new_context: UserContext):
        """Merge new info without losing existing data."""
        self.user_context = self.user_context.merge(new_context)
```

### Context Accumulation

The intake agent progressively builds context:

```
Turn 1: "I want to build muscle"
        → extracted: {goals: [HYPERTROPHY]}

Turn 2: "I'm 30 years old"
        → extracted: {biometrics: {age: 30}}
        → accumulated: {goals: [HYPERTROPHY], biometrics: {age: 30}}

Turn 3: "intermediate lifter with access to a gym"
        → extracted: {experience: "intermediate", equipment: [...]}
        → accumulated: complete context → ready for workout
```

## Safety Architecture

### Multi-Layer Protection

```
Layer 1: RedFlagScanner (Rule-based)
         └─ Pattern matching for critical symptoms
         └─ Fast, deterministic, no LLM needed
         └─ Runs BEFORE workflow execution

Layer 2: ClinicalGatekeeper (LLM-based)
         └─ Nuanced analysis of context + plan
         └─ Can suggest modifications
         └─ Provides clinical rationale
         └─ Runs AFTER plan generation
```

### Red Flag Categories

| Category | Examples | Action |
|----------|----------|--------|
| Neurological | Numbness, tingling, shooting pain, sciatica | HARD-STOP → Neurologist |
| Cardiovascular | Chest pain, palpitations, fainting, SOB | HARD-STOP → Cardiologist |
| Acute Injury | Pop sound, can't bear weight, acute swelling | HARD-STOP → Orthopedist |
| Inflammatory | Fever + joint pain, pulsing swelling | HARD-STOP → GP |

### Routing Decision Flow

```python
def route(message: str, state: ConversationState) -> RoutingDecision:
    # Step 1: Classify intent
    intent = intent_classifier.classify(message)

    # Step 2: Check for safety concerns (RedFlagScanner)
    safety_concerns = red_flag_scanner.scan(state.user_context, message)
    if safety_concerns:
        return RoutingDecision(
            workflow=WorkflowType.SAFETY_BLOCK,
            safety_concerns=safety_concerns,
        )

    # Step 3: Handle non-action intents
    if intent in (GENERAL_QUESTION, CLARIFICATION, GREETING):
        return RoutingDecision(workflow=WorkflowType.GENERAL_RESPONSE)

    # Step 4: Check intake completeness
    missing = get_missing_required_fields(state.user_context)
    if missing:
        return RoutingDecision(
            workflow=WorkflowType.INTAKE_NEEDED,
            missing_fields=missing,
        )

    # Step 5: Route to appropriate workflow
    if intent == RECOVERY_REQUEST:
        return RoutingDecision(workflow=WorkflowType.RECOVERY_ONLY)
    else:
        return RoutingDecision(workflow=WorkflowType.INTEGRATED)
```

## Data Models

### Core Schemas

```
UserContext
├── biometrics: Biometrics (age, weight, height, sex)
├── medical_history: MedicalHistory
├── fitness_goals: list[FitnessGoal]
├── available_equipment: list[Equipment]
├── experience_level: ExperienceLevel
├── pain_areas: list[str]
└── specific_requests: list[str]

WorkoutPlan
├── plan_id: str
├── blocks: list[ExerciseBlock]
│   └── exercises: list[Exercise]
│       ├── name, sets, reps, rest_seconds
│       ├── movement_pattern: MovementPattern
│       ├── primary_muscles: list[MuscleGroup]
│       └── alternatives: list[str]
├── total_duration_minutes: int
├── difficulty_level: str
└── rationale: str

RecoveryPlan
├── plan_id: str
├── blocks: list[RecoveryBlock]
│   ├── modality: RecoveryModality (SMR, DYNAMIC_MOBILITY, STATIC_STRETCHING)
│   └── exercises: list[RecoveryExercise]
│       ├── name, duration_seconds, sets
│       ├── target_areas: list[str]
│       └── intensity: str (light, moderate, deep)
├── total_duration_minutes: int
├── target_areas: list[str]
└── rationale: str

AuditLog
├── status: AuditStatus (APPROVED, MODIFIED, REJECTED)
├── red_flags: list[RedFlag]
├── modifications: list[str]
├── warnings: list[str]
└── approval_notes / rejection_reason: str
```

## CLI Observability

Verbose and debug behavior is centralized in `src/cli/app.py` via LangChain globals:

| Flag | LangChain | Effect |
|------|-----------|--------|
| `-v` (verbose) | — | CLI-only: prints workout plan, recovery plan, and audit log as JSON in the response |
| `-d` (debug) | `set_debug(True)` | Adds `ConsoleCallbackHandler`; prints LLM prompts and responses (`[llm/start]`, `[llm/end]`) |

Agents and workflows do not accept verbose parameters. The CLI sets `set_debug()` before creating the coach so that all LLM invocations (including `with_structured_output`) emit trace output when debug is enabled.

The validation runner uses `-v` for its own progress output (e.g. `[1/10] Processing: ...`), not for LangChain tracing.

## LLM Configuration

### OpenAI-Only Provider

The system uses OpenAI exclusively (Groq was removed due to lack of strict JSON schema support):

```python
def get_llm_for_agent(agent_name: str) -> BaseChatModel:
    """Factory for agent-specific LLM configuration."""
    settings = get_settings()
    params = settings.get_llm_params(agent_name)

    return ChatOpenAI(
        model=params["model"],           # e.g., "gpt-4o-mini"
        temperature=params["temperature"],
        max_tokens=params["max_tokens"],
        api_key=settings.openai_api_key,
    )
```

### Temperature Guidelines

| Agent | Temperature | Rationale |
|-------|-------------|-----------|
| Intake | 0.2 | Accurate extraction, minimal hallucination |
| Kinesiologist | 0.4 | Balanced creativity for exercise variety |
| Recovery | 0.3 | Consistent modality sequencing |
| Gatekeeper | 0.1 | Conservative, deterministic safety decisions |

## Validation Runner

Benchmark the system against labeled question sets:

```bash
poetry run python -m src.util.validation_runner data/Validation_Questions.jsonl -v
```

### Metrics Tracked

- **Intent accuracy**: Percentage of correct intent classifications
- **Red-flag misses**: Questions that should have blocked but didn't
- **False blocks**: Legitimate requests incorrectly blocked

### Question Format (JSONL)

```json
{
  "question": "I'm 32, 85kg, intermediate. No injuries. Goal is hypertrophy.",
  "expected_intent": "exercise_request"
}
```

Use validation runs to track regressions when adjusting prompts, thresholds, or model parameters.

## Testing Strategy

### Unit Tests
- Each agent tested in isolation with mocked LLM
- Validators tested with known good/bad inputs
- State management tested for context accumulation
- Intent classification tested with labeled examples

### Integration Tests
- Multi-turn intake flow
- Complete workflow A (exercise + recovery + audit)
- Complete workflow B (recovery + audit)
- Safety blocking for red flags
- Intent routing accuracy

### Coverage Target
- Minimum 85% line coverage
- Critical paths (safety, routing) at 100%
- Excluded: `src/cli/*`, `src/ui/*`, `*/__init__.py`

# Refactor Plan: Inference Vocabulary, Equipment Generics, Action Verb Guard + LLM Safety Backstop

**Date:** 2026-03-27 (revision 3 — all critique findings addressed)
**Baseline:** 17/30 passed on 30-case kinesiologist eval (pipeline mode)
**Target:** ≥ 22/30 passed after all steps complete

---

## Phase 1: Diagnosis

### What breaks and why

Four structural failure categories identified across 13 eval failures:

1. **Goal inference — first-match `if/elif` chain** (`state_router.py:286-354`). The function returns on the first matching branch. New vocabulary requires a new branch. "bicep burnout", "metabolic circuit", "explosive power", "lose fat" all produce `None` because they fall through every branch. Adding a new term means editing a non-obvious `if/elif` tree.

2. **Equipment inference — exact phrase list** (`state_router.py:435-448`). The gym-phrases list requires exact substring matches: `"gym session"`, `"at the gym"`, `"gym workout"`. Any variation (`"for the gym"`, `"my gym"`, `"going to gym"`) silently falls through. The fix is always "add another phrase" — there is no upper bound.

3. **IntentClassifier INTAKE_UPDATE false wins** (`intent_classifier.py:334-349`). The classifier *already has* an INTAKE_UPDATE demotion block (lines 334-349): when INTAKE_UPDATE wins but another intent scores `> 0.2`, the action intent wins. The structural problem is that EXERCISE_REQUEST scores **zero** for messages like "I want to lose fat" or "Improving explosive power for basketball" — those phrases match no EXERCISE_REQUEST pattern, so the demotion never fires.

4. **Safety check — closed regex catalogue** (`state_router.py:241-278`). The `acute_patterns` list in `_check_safety()` and `RED_FLAG_PATTERNS` in `red_flag_scanner.py` are finite enumerations. "fluttering in my chest" is not in either. Any novel symptom requires a new entry. Cardiac/neurological presentations are too varied to enumerate.

### Smell catalogue

| File | Line | Smell | Root Cause |
|------|------|-------|------------|
| `src/orchestrator/state_router.py` | 286-354 | `if/elif` chain — 8 branches with embedded term lists | No single-responsibility vocabulary table |
| `src/orchestrator/state_router.py` | 435-448 | Hardcoded exact gym phrase list (7 phrases) | Open/Closed violated — every new phrase = code change |
| `src/orchestrator/intent_classifier.py` | 334-349 | INTAKE_UPDATE demotion fires only when EXERCISE_REQUEST scores > 0.2 | EXERCISE_REQUEST patterns don't cover action verbs, sport goals, skill learning |
| `src/orchestrator/state_router.py` | 241-278 | `acute_patterns` — inline list with 16 entries | Closed enumeration of open-world medical symptoms |
| `src/validators/red_flag_scanner.py` | 66-87 | `RED_FLAG_PATTERNS` covers regex but not semantic symptoms | Regex cannot enumerate all cardiac/neurological presentations |

### What must NOT change (Preservation Contracts)

```
PRESERVED — public signatures, callers must not change:
  StateRouter()                                              ← zero-argument constructor — 17 call sites
  StateRouter.route(message, state) → RoutingDecision
  StateRouter._infer_goal_from_message(message) → FitnessGoal | None   ← tests call directly
  StateRouter._infer_equipment_from_message(message) → list[Equipment] | None  ← tests call directly
  IntentClassifier.classify(message) → ClassifiedIntent
  IntentClassifier.is_exercise_related / is_recovery_related / needs_action
  RedFlagScanner.validate(data) → ValidationResult   ← clinical_gatekeeper.py calls directly
  All fields of ClassifiedIntent, RoutingDecision, WorkflowType

PRESERVED — behaviour contracts:
  test_strength_not_overridden_by_rehab_term: "I have scoliosis … my main goal is strength" → STRENGTH
  test_oa_abbreviation_does_not_false_match: "I coach a road running group" ≠ REHABILITATION
  All 31 tests in tests/unit/test_kinesiologist_eval.py
  All tests in tests/unit/test_orchestration.py

FREE TO CHANGE — internal implementation:
  The if/elif body of _infer_goal_from_message
  The gym phrase list inside _infer_equipment_from_message
  INTENT_PATTERNS[EXERCISE_REQUEST] contents (additive only)
  recovery_override_terms contents (additive only)
  _check_safety() internal flow (delegating to SafetyJudge)
```

---

## Phase 2: Target State

### Architecture

```
CURRENT (fragile)                         TARGET (maintainable)
─────────────────                         ─────────────────────────────────────

state_router.py                           state_router.py
  _infer_goal_from_message()               _GOAL_VOCABULARY: dict[FitnessGoal, list[str]]
    if "hypertrophy" in msg: ...    →→→    _infer_goal_from_message()
    elif "strength" in msg: ...              iterate _GOAL_VOCABULARY in priority order
    elif "lose weight" in msg: ...           return first match (rehab last — preserved)
    ...8 branches...

  _infer_equipment_from_message()          _GYM_NEG_PATTERN: re.Pattern  (replaces phrase list)
    if phrase in ["gym session",    →→→    _infer_equipment_from_message()
        "at the gym", ...]                   re.search(r'\bgym\b') and not _GYM_NEG_PATTERN.search()
    ...7 exact phrases...                    + "chair" → BODYWEIGHT
                                             + "plyo box" → BODYWEIGHT

intent_classifier.py                      intent_classifier.py
  INTENT_PATTERNS[EXERCISE_REQUEST]        INTENT_PATTERNS[EXERCISE_REQUEST]
    — no sport/skill/loss-verb coverage      + sport patterns (basketball, volleyball…)
  demotion block fires only when    →→→    + skill/learning patterns (l-sit, learn to…)
    EXERCISE_REQUEST > 0.2                 + fat-loss verb patterns (lose fat, burn fat…)
                                           demotion block fires for these new cases

state_router.py                           src/validators/safety_judge.py  ← NEW FILE
  _check_safety()                    →→→    SafetyJudge
    L1: RedFlagScanner on context            L1:  _scan_context()   — RedFlagScanner on stored context
    L2: inline acute_patterns                L1b: _scan_message_deep() — RedFlagScanner on message text
    ← no LLM backstop →                     L2:  _scan_message_regex() — _ACUTE_PATTERNS inline list
                                             L3:  _run_llm_assessment() — LLM semantic backstop
                                           Single exit: assess() returns first non-empty layer result
                                           state_router._check_safety() delegates to SafetyJudge
```

### LLM Safety Judge — detailed design

The judge is a four-layer, single-exit contract. Layers are ordered fast → slow. The LLM is the catch-all.

```
message + context arrive at _check_safety()
       │
       ▼
assess(message, context) → list[str]          ← one call site; single exit
  return (
      self._scan_context(context)             # L1:  RedFlagScanner on stored context
      or self._scan_message_deep(             # L1b: RedFlagScanner on message text (PRESERVED)
             message, context)
      or self._scan_message_regex(message)    # L2:  _ACUTE_PATTERNS inline list
      or self._run_llm_assessment(message)    # L3:  LLM semantic backstop
  )
       │
       └── Empty list [] means safe.
           First non-empty list short-circuits remaining layers.
```

**Layer descriptions:**

| Layer | Method | Mechanism | Catches |
|-------|--------|-----------|---------|
| L1 | `_scan_context(context)` | `RedFlagScanner.validate(context)` | Conditions already stored in user profile |
| L1b | `_scan_message_deep(message, context)` | Append message to `medical_history.notes` in a temp context; `validate()` again | `RED_FLAG_PATTERNS` regexes on message text — e.g. "arrhythmia" in message |
| L2 | `_scan_message_regex(message)` | `_ACUTE_PATTERNS` string list (moved verbatim from `_check_safety`) | Known acute signals: "electrical zap", "chest pain", "dizzy" |
| L3 | `_run_llm_assessment(message)` | Plain YES/NO prompt; 2-attempt retry | Novel phrasings: "fluttering in chest + sweaty" |

**Why L1b is not collapsed into L1:** L1 scans the stored `UserContext` that the LLM intake agent has accumulated across turns. L1b scans the *current message* — it catches red flags in a user's first message before anything is stored. These are two distinct inputs. The deep scan at `state_router.py:269-278` implements L1b; it must be preserved verbatim in `SafetyJudge._scan_message_deep()`.

**LLM is NOT called when** L1, L1b, or L2 fires. All existing safety tests pass with zero LLM calls.

**LLM IS called when** all three regex layers return empty on a message being routed to an action workflow.

**Failure mode:** On LLM exception (both retry attempts exhausted), `_run_llm_assessment()` logs a warning and returns `[]` (fail-open). The routing pipeline is not crashed.

**Why no structured output:** The decision is binary. A YES/NO prompt requires no JSON schema binding, no `with_structured_output`, and no model feature gating. Parsing is `response.content.strip().upper().startswith("YES")`. The concern string is derived from the original message, not from LLM output.

### New interface contracts

```python
# NEW FILE — src/validators/safety_judge.py
# One public class only (philosophy: one class per file)

_SAFETY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a safety screener for an exercise coaching app. "
     "Does this message describe a medical emergency or acute symptom "
     "requiring IMMEDIATE medical attention? "
     "Reply YES if so, NO if not. No other text."),
    ("human", "{message}"),
])

_ACUTE_PATTERNS: list[str] = [
    # Moved verbatim from state_router._check_safety() lines 242-262
    "chest pain", "can't breathe", "shortness of breath", "breathless",
    "dizzy", "lightheaded", "numbness", "shooting pain", "radiating pain",
    "severe pain", "emergency", "heart attack", "stroke",
    "electrical zap", "zapping", "zap when", "sharp zap",
    "tingling", "pins and needles",
]


class SafetyJudge:
    """Four-layer safety screener: regex fast-path (×3) + LLM semantic backstop.

    Layers L1, L1b, L2 run synchronously with no LLM cost. Layer L3 (LLM)
    is invoked only when all three regex layers return empty. assess() has a
    single exit point — the first non-empty concern list short-circuits.

    StateRouter constructs SafetyJudge with no arguments. The LLM is resolved
    internally via get_llm_for_agent("safety") — all 17 StateRouter() call
    sites remain zero-argument.
    """

    def __init__(self) -> None:
        """Load LLM from config internally. No arguments required."""
        # get_settings() is the module singleton — do not call Settings() directly.
        self._red_flag_scanner = RedFlagScanner(strict_mode=True)
        self._chain = _SAFETY_PROMPT | get_llm_for_agent("safety")  # no settings arg → singleton

    def assess(self, message: str, context: UserContext) -> list[str]:
        """Return concern strings. Empty list means safe. Never raises.

        Single exit. Layers evaluated left to right; first non-empty result
        short-circuits. Python list truthiness: [] is falsy, non-empty is truthy.
        """
        return (
            self._scan_context(context)
            or self._scan_message_deep(message, context)
            or self._scan_message_regex(message)
            or self._run_llm_assessment(message)
        )

    def _scan_context(self, context: UserContext) -> list[str]:
        """L1: RedFlagScanner on stored user context. Fast, no LLM."""
        result = self._red_flag_scanner.validate(context)
        return result.errors if not result.is_valid else []

    def _scan_message_deep(self, message: str, context: UserContext) -> list[str]:
        """L1b: RedFlagScanner on message text via temp context.

        Preserves the deep-scan logic at state_router.py:269-278.
        Catches RED_FLAG_PATTERNS matches on first-turn messages where
        no user context has been stored yet (e.g. "I have arrhythmia").
        """
        if not message:
            return []
        temp_context = UserContext(
            medical_history=context.medical_history.model_copy(deep=True),
        )
        notes = temp_context.medical_history.notes or ""
        temp_context.medical_history.notes = f"{notes}\n{message}".strip()
        result = self._red_flag_scanner.validate(temp_context)
        return result.errors if not result.is_valid else []

    def _scan_message_regex(self, message: str) -> list[str]:
        """L2: _ACUTE_PATTERNS string list on raw message. Fast, no LLM."""
        msg_lower = message.lower()
        for pattern in _ACUTE_PATTERNS:
            if pattern in msg_lower:
                return [f"Acute symptom mentioned: {pattern}"]
        return []

    def _run_llm_assessment(self, message: str) -> list[str]:
        """L3: LLM semantic backstop. 2-attempt retry; fail-open on both failures.

        Token budget: response is always YES or NO (1 token). max_tokens=5
        provides a hard cap with margin. temperature=0.0 enforces determinism.
        Formula: 1 token × 1.3 safety margin = 2 → set max_tokens=5.
        """
        for attempt in range(2):
            try:
                response = self._chain.invoke({"message": message})
                if response.content.strip().upper().startswith("YES"):
                    return [f"Potential medical emergency: {message[:120]}"]
                return []
            except Exception as exc:  # noqa: BLE001
                if attempt == 1:
                    logger.warning(
                        "SafetyJudge LLM assessment failed after 2 attempts: %s",
                        type(exc).__name__,
                    )
        return []  # fail-open — do not crash routing pipeline


# CHANGED — src/orchestrator/state_router.py

# __init__:
#   self._safety_judge = SafetyJudge()    ← added; zero-arg; preserves all 17 call sites
#   self.red_flag_scanner = ...           ← REMOVED (SafetyJudge._red_flag_scanner owns it)
#   import re                             ← ADDED (required for Step 2)

# _check_safety(self, user_context, message):
#   return self._safety_judge.assess(message, user_context)  ← message first, context second
#   # All previous body removed — assess() reproduces it exactly

# import RedFlagScanner removed from state_router.py top-level imports
#   (RedFlagScanner now lives inside SafetyJudge)
```

### Philosophy compliance targets

| Smell | Fix | Philosophy |
|-------|-----|------------|
| `if/elif` goal chain — OCP violated | `_GOAL_VOCABULARY` dict — new goals added to table, no logic changes | OCP — open for extension, closed for modification |
| Exact gym phrase list | `\bgym\b` regex + `_GYM_NEG_PATTERN` compiled regex | OCP — new negation case = one word added to alternation |
| EXERCISE_REQUEST vocabulary gap | Additive patterns for sport/skill/loss verbs | No existing patterns changed; extension only |
| Closed safety enumeration | Four-layer SafetyJudge with LLM backstop | DIP — router depends on SafetyJudge, not inline regex |
| `_check_safety` multiple concerns (scan context + scan message + deep scan) | All delegated to `SafetyJudge.assess()` | SRP — router routes; judge judges |
| `assess()` multiple mid-logic returns | `or` chain → single-exit; four private helper methods | Single exit rule |

---

## Phase 3: Migration Steps

**Strangler fig assessment: YES.** Every step is additive or an internal implementation swap. `StateRouter` and `IntentClassifier` public APIs are unchanged throughout. No step leaves the system broken.

---

### Step 0: Verification baseline

```bash
poetry run pytest --tb=short -q
# Expected: all tests pass (31 in test_kinesiologist_eval.py + full suite)

poetry run python -m src.util.kinesiologist_eval_runner
# Expected: 17/30 passed — record as pre-refactor baseline

poetry run python -m src.util.kinesiologist_eval_runner \
    --mode router --input data/Kinesiologist_and_Recovery_evaluation.json
# Expected: 11/15 passed — record as secondary baseline
```

---

### Step 1: Goal vocabulary dict `[state_router.py]`

**What changes:** Replace `_infer_goal_from_message` `if/elif` body with a `_GOAL_VOCABULARY` dict.

**The dict must include ALL existing terms plus the new additions below.** An implementer must not only add new terms — all existing terms from every branch of the current `if/elif` (lines 286-354) must be carried into the dict. The table below is the authoritative full inventory.

**Iteration order: `HYPERTROPHY → STRENGTH → WEIGHT_LOSS → ENDURANCE → ATHLETIC_PERFORMANCE → REHABILITATION`**
- Rehab is last — preserves the contract: "I have scoliosis but my main goal is strength" → `STRENGTH`
- Python 3.7+ dict insertion order is guaranteed — no additional sorting needed

| Goal | Carry-forward (all existing terms) | New terms added |
|------|------------------------------------|-----------------|
| `HYPERTROPHY` | `"hypertrophy"`, `"build muscle"`, `"muscle gain"`, `"push day"`, `"pull day"`, `"leg day"`, `"arm day"`, `"arms day"`, `"chest day"`, `"back day"`, `"shoulder day"`, `"shoulders day"`, `"bro split"`, `"bro-split"`, `"ppl"`, `"push/pull/legs"`, `"push pull legs"`, `"boulder shoulders"`, `"bigger"`, `"mass"`, `"size"` | `"bicep"`, `"biceps"`, `"burnout"`, `"peak"`, `"pump"`, `"bicep peak"` |
| `STRENGTH` | `"strength"`, `"stronger"` | *(none)* |
| `WEIGHT_LOSS` | `"lose weight"`, `"weight loss"`, `"fat loss"` | `"lose fat"`, `"burn fat"`, `"metabolic"`, `"circuit"`, `"hiit"`, `"burn calories"` |
| `ENDURANCE` | `"endurance"`, `"marathon"`, `"run"`, `"running"`, `"jog"`, `"jogging"`, `"miles"`, `" 5k"`, `" 10k"`, `"half marathon"`, `"cardio"` | *(none)* |
| `ATHLETIC_PERFORMANCE` | `"performance"`, `"sport"`, `"handstand"`, `"muscle up"`, `"muscle-up"`, `"pistol squat"`, `"planche"`, `"front lever"`, `"vertical jump"`, `"box jump"` | `"explosive"`, `"basketball"`, `"volleyball"`, `"plyo"`, `"agility"`, `"speed"` |
| `REHABILITATION` | `"fix my shoulder"`, `"fix my hip"`, `"fix my knee"`, `"fix my back"`, `"fix my posture"`, `"fix my scapula"`, `"correct my posture"`, `"winged scapula"`, `"scapula"`, `"scoliosis"`, `"posture"`, `"bone density"`, `"osteopenia"`, `"osteoporosis"`, `"osteoarthritis"`, `"osteoarthritic"`, `"arthritis"`, `"postpartum"`, `"post-partum"`, `"postnatal"` | *(none)* |

- Preserve the "OA omitted intentionally" comment for `REHABILITATION`

**New `_infer_goal_from_message()` body:**
```python
msg = message.lower()
for goal, terms in _GOAL_VOCABULARY.items():
    if any(t in msg for t in terms):
        return goal
return None
```

**Bridge:** None — `@staticmethod`, implementation-only change.

**Gate:**
```bash
poetry run pytest --tb=short -q
# Verify specifically:
#   test_strength_not_overridden_by_rehab_term
#   test_oa_abbreviation_does_not_false_match
#   test_winged_scapula_infers_rehabilitation
#   test_bone_density_infers_rehabilitation
#   test_postpartum_infers_rehabilitation
```

---

### Step 2: Equipment inference — gym keyword + negation `[state_router.py]`

**What changes:** Replace exact gym phrase list with `\bgym\b` regex + compiled negation pattern.

**Checklist before editing:**
- Add `import re` to `state_router.py` imports (currently absent — required for `re.search` and `re.compile`)
- Do not modify `_HOME_EQUIPMENT_TERMS` (already contains `"gym"` for the `at home` guard — unrelated to the new gym detection)

- New module-level compiled pattern:
  ```python
  _GYM_NEG_PATTERN: re.Pattern = re.compile(
      r"\b(no|not|don't|doesn't|can't|cannot|without|avoid|skip|lack|intimidated)"
      r"\b[\w\s]{0,25}\bgym\b",
      re.IGNORECASE,
  )
  ```
  **Why `"intimidated"` is in the alternation:** "I'm intimidated by the gym" must not infer full gym. The word "intimidated" is not a grammatical negation, so it was not covered by the first regex draft. Including it explicitly ensures the pattern matches "intimidated … gym" within 25 characters.

  A compiled pattern is used rather than a phrase list because:
  - The phrase list had no upper bound — every new phrasing required a new entry
  - A single proximity regex covers present and future variants without new entries
  - `re.compile` at module load time means zero per-call compilation cost

- Replace existing gym phrase block (lines 435-448):
  ```python
  if re.search(r'\bgym\b', msg, re.IGNORECASE) and not _GYM_NEG_PATTERN.search(msg):
      return [Equipment.BARBELL, Equipment.DUMBBELL, Equipment.MACHINE,
              Equipment.CABLE, Equipment.KETTLEBELL]
  ```

- Add to specific-equipment detection block:
  ```python
  if "chair" in msg or "chairs" in msg:
      equipment.append(Equipment.BODYWEIGHT)
  if "plyo box" in msg or "plyo boxes" in msg:
      equipment.append(Equipment.BODYWEIGHT)  # No PLYO_BOX enum; plyos are bodyweight
  ```

- All existing location/context guards (`"at home"`, `_HOME_EQUIPMENT_TERMS`) unchanged

**Bridge:** None — `@staticmethod`, implementation-only change.

**Gate:**
```bash
poetry run pytest --tb=short -q
# Verify specifically:
#   test_gym_session_infers_full_gym
#   test_at_home_with_dumbbells_not_bodyweight
#   test_outdoor_running_still_infers_treadmill
#   test_park_pullup_bar_infers_bodyweight
#   test_intimidated_by_the_gym_does_not_infer_gym  ← new test from Step 2 list
```

---

### Step 3: Action verb patterns in EXERCISE_REQUEST `[intent_classifier.py]`

**What changes:** Add patterns to `INTENT_PATTERNS[EXERCISE_REQUEST]` (additive — no existing pattern removed or modified).

Add to `INTENT_PATTERNS[IntentType.EXERCISE_REQUEST]`:
```python
# Fat-loss action verbs — covers "I want to lose fat", "help me burn fat"
r"\b(i\s*want\s*to|i\s*need\s*to|help\s*me)\s*(lose|burn|cut|lean|slim)\b",
# Sport-specific goals — covers "explosive power for basketball", "volleyball training"
r"\b(explosive|plyometric|plyo)\b",
r"\b(basketball|volleyball|soccer|football|rugby|sport[s]?)\b",
# Skill learning — covers "Learning the L-Sit progression", "learning to handstand"
# NOTE: suffix (progression|skill|movement) is NON-OPTIONAL in both patterns to
# prevent false matches on "I am learning I have a knee condition" (optional suffix
# allows zero-length match; non-optional eliminates the false positive).
r"\b(learning\s+to|learn\s+to|working\s+on)\s+(the\s+)?[\w\-]+\s*(progression|skill|movement)\b",
r"\b(learn|learning)\s+(the\s+)?[\w\-]+\s+progression\b",
```

**Mechanism:** The classifier already has an INTAKE_UPDATE demotion block at lines 334-349 that fires when any other intent scores `> 0.2`. These additions make `EXERCISE_REQUEST` score `> 0.2` for the cases where `INTAKE_UPDATE` previously won uncontested. No new control flow is added.

**Bridge:** None — additive pattern additions to an existing dict.

**Gate:**
```bash
poetry run pytest --tb=short -q
# Verify specifically:
#   test_intake_update_detection (still passes — pure "I am 66" still INTAKE_UPDATE)
#   test_bare_age_classifies_as_intake_update
#   TestIntentClassifier (all)
#   TestAgeAndInjuryClassification (all)
#   test_learning_condition_message_is_not_exercise_request  ← false-positive guard
```

---

### Step 4a: Add `safety` agent to `config.yaml` `[config.yaml]`

**What changes:** Add the `safety` agent entry to `config.yaml` so `get_llm_for_agent("safety")` receives `temperature: 0.0` instead of the provider default (`0.3`).

The `safety` key **must** be nested under the `agents:` section. `settings.py:101-102` only parses `config["agents"]` into `self.agents`; a key at the YAML root is silently ignored.

```yaml
# In config.yaml — under the existing agents: section
agents:
  intake:
    temperature: 0.2
  kinesiologist:
    temperature: 0.4
  recovery:
    temperature: 0.3
  gatekeeper:
    temperature: 0.1
  safety:                 # ← ADD THIS
    temperature: 0.0
    max_tokens: 5         # YES/NO = 1 token; budget: 1 × 1.3 = 2 → set 5 for margin
```

**Why `temperature: 0.0`:** A safety screener must be deterministic. Any temperature above 0 introduces variance into a binary emergency decision.

**Why `max_tokens: 5`:** Token budget calculation: 1 token (YES or NO) × 1.3 safety margin = 2 tokens. Setting 5 provides headroom while preventing accidental inheritance of the provider default (2048).

**Gate:**
```bash
poetry run python -c "
from src.config.settings import get_settings
s = get_settings()
p = s.get_llm_params('safety')
assert p['temperature'] == 0.0, f'Expected 0.0, got {p[\"temperature\"]}'
assert p['max_tokens'] == 5, f'Expected 5, got {p[\"max_tokens\"]}'
print('PASS: safety agent config correct')
"
```

---

### Step 4b: SafetyJudge — LLM backstop `[NEW: src/validators/safety_judge.py]`

**What changes:** Extract all safety logic from `_check_safety()` into a new `SafetyJudge` class with a four-layer, single-exit implementation.

**New file:** `src/validators/safety_judge.py` — one public class only (see Phase 2 interface contracts for the complete pseudocode).

**Key implementation notes:**
- `SafetyJudge.__init__()` calls `get_llm_for_agent("safety")` with **no second argument** — the factory's default is `get_settings()` (the module singleton). Do NOT pass `Settings()` directly; that bypasses the singleton and creates a redundant load of `.env` + `config.yaml`.
- `assess(self, message: str, context: UserContext)` — **`message` is the first positional argument, `context` is second.** The call site in `state_router._check_safety()` must pass arguments in this order: `self._safety_judge.assess(message, user_context)`.
- `_ACUTE_PATTERNS` is moved **verbatim** from `_check_safety()` lines 242-262 — not rewritten as regex. The current patterns are plain strings matched with `in message_lower`. Moving them to a module constant preserves semantics; `re.search` is not used here.
- `_scan_message_deep()` preserves the deep-scan at `state_router.py:269-278` exactly — temp context, `medical_history.notes` append, `RedFlagScanner.validate()` call.

**Update** `src/orchestrator/state_router.py`:
- Add `import re` to imports (required for Step 2; if Step 2 is already merged, this is already done)
- `__init__`: add `self._safety_judge = SafetyJudge()` — zero arguments, preserving all 17 call sites
- `__init__`: remove `self.red_flag_scanner = RedFlagScanner(strict_mode=True)` — `SafetyJudge` owns it now
- `_check_safety()`: replace entire body with `return self._safety_judge.assess(message, user_context)`
- Remove `from src.validators.red_flag_scanner import RedFlagScanner` import (no longer used directly)

**Update** `src/validators/__init__.py`: export `SafetyJudge`.

**Bridge:** `_check_safety()` public signature is unchanged — all router callers unaffected. `StateRouter()` zero-argument constructor preserved.

**Gate:**
```bash
poetry run pytest --tb=short -q
# Verify specifically:
#   test_route_safety_block
#   test_route_acute_symptoms_in_message
#   test_route_dizziness_and_shortness_of_breath_blocks
#   test_route_shooting_pain_blocks
#   TestSafetyBlock.test_electrical_zap_routes_safety_block
#   test_state_router_constructor_still_zero_args
```

---

### Step 5: Eval runner — `_PAIN_AREA_KEYWORDS` update `[kinesiologist_eval_runner.py]`

**What changes:** Add `"scapula"` to `_PAIN_AREA_KEYWORDS` in the eval runner.

```python
("scapula", "scapula"),
("scapula", "winged scapula"),
```

This fixes Case 7: the pipeline extractor populates `pain_areas`, satisfying `has_health_acknowledgment()` for the `RECOVERY_ONLY` path. The LLM intake agent would extract this naturally; the eval runner simulation was missing it.

**Gate:**
```bash
poetry run python -m src.util.kinesiologist_eval_runner
# Expected: ≥ 22/30 passed (up from 17/30 pre-refactor baseline)

poetry run python -m src.util.kinesiologist_eval_runner \
    --mode router --input data/Kinesiologist_and_Recovery_evaluation.json
# Expected: 11/15 passed (unchanged — router-only baseline preserved)

poetry run pytest --tb=short -q
# Expected: all tests pass
```

---

## Phase 4: Regression Guard

### Existing tests — must pass at every step

| Test file | Coverage area | Key tests to preserve |
|-----------|--------------|----------------------|
| `tests/unit/test_kinesiologist_eval.py` | Router inference, acute bypass, safety, equipment, goal | `test_strength_not_overridden_by_rehab_term`, `test_oa_abbreviation_does_not_false_match`, `test_gym_session_infers_full_gym`, `test_at_home_with_dumbbells_not_bodyweight` |
| `tests/unit/test_orchestration.py` | Classifier, router, safety block | All `TestIntentClassifier`, all `TestStateRouter`, all `TestAgeAndInjuryClassification` |
| `tests/unit/test_validators.py` | RedFlagScanner | All — must pass after `SafetyJudge` wraps the scanner |

### New tests required

**Step 1 — `TestGoalVocabularyDict`** (add to `tests/unit/test_kinesiologist_eval.py`):
```python
def test_lose_fat_infers_weight_loss(router)
def test_metabolic_circuit_infers_weight_loss(router)
def test_explosive_power_infers_athletic_performance(router)
def test_basketball_infers_athletic_performance(router)
def test_bicep_burnout_infers_hypertrophy(router)
def test_rehab_still_last_when_strength_also_present(router)  # preservation
# Verify carry-forward: all existing passing tests still pass (no new tests for carry-forwards)
```

**Step 2 — `TestEquipmentNegation`** (add to `tests/unit/test_kinesiologist_eval.py`):
```python
def test_for_the_gym_infers_full_gym(router)
# Test message: "25, 90kg, advanced. Push day for the gym."
# Must NOT contain "bodyweight" — tests the \bgym\b regex, not the early-exit

def test_intimidated_by_the_gym_does_not_infer_gym(router)
# IMPORTANT: test message must NOT contain "bodyweight", "no equipment", or other
# early-exit terms — the test must exercise _GYM_NEG_PATTERN specifically:
# e.g. router._infer_equipment_from_message("I'm intimidated by the gym, beginner workout?")
# assert equipment is None or Equipment.BARBELL not in (equipment or [])

def test_i_dont_go_to_gym_does_not_infer_gym(router)
# e.g. "I don't go to the gym, I work out at home."
# Note: "at home" guard also fires here — acceptable as belt-and-suspenders

def test_chair_infers_bodyweight(router)
def test_plyo_boxes_infers_bodyweight(router)
```

**Step 3 — `TestActionVerbIntentGuard`** (add to `tests/unit/test_orchestration.py`):
```python
def test_i_want_to_lose_fat_is_exercise_request(classifier)
def test_improving_explosive_power_is_exercise_request(classifier)
def test_learning_lsit_progression_is_exercise_request(classifier)
def test_learning_condition_message_is_not_exercise_request(classifier)
# Message: "I am learning I have a knee condition." — must NOT classify as EXERCISE_REQUEST
# Verifies non-optional suffix guards against false positive

def test_i_am_30_beginner_with_action_is_exercise_not_intake(classifier)
```

**Step 4b — `TestSafetyJudge`** (new file: `tests/unit/test_safety_judge.py`):
```python
def test_known_context_pattern_hits_l1_no_llm_call(mock_llm)
# Stored context with arrhythmia → _scan_context fires; LLM not invoked

def test_message_deep_scan_catches_arrhythmia_in_message(mock_llm)
# Message "I have arrhythmia" with no stored context → _scan_message_deep fires
# Specifically verifies L1b (not caught by _ACUTE_PATTERNS which has no "arrhythmia")

def test_acute_pattern_hits_l2_no_llm_call(mock_llm)
# Message "electrical zap in my elbow" → _scan_message_regex fires; LLM not invoked

def test_novel_cardiac_symptom_reaches_llm(mock_llm)
# "fluttering in chest + sweaty" → L1/L1b/L2 all empty → LLM invoked

def test_llm_yes_response_returns_concern_string(mock_llm)
# LLM returns "YES" → ["Potential medical emergency: ..."]

def test_llm_no_response_returns_empty(mock_llm)
# LLM returns "NO" → []

def test_llm_exception_attempt1_retries(mock_llm)
# First call raises → second attempt made (verify 2 calls total)

def test_llm_exception_both_attempts_returns_empty_fail_open(mock_llm)
# Both calls raise → [] returned, no exception propagated

def test_assess_message_first_context_second_signature()
# assess(message="...", context=UserContext()) — positional order correct

def test_state_router_constructor_still_zero_args()
# StateRouter() — no TypeError; constructs SafetyJudge internally

def test_safety_judge_integrates_with_router_flutter_routes_safety_block(mock_llm)
# mock LLM returns YES for flutter message → router.route(...) → SAFETY_BLOCK
```

### Regression verification command

```bash
# Run after every step — must pass before proceeding to the next
poetry run pytest --tb=short -q

# Full eval check (run after Step 5)
poetry run python -m src.util.kinesiologist_eval_runner
poetry run python -m src.util.kinesiologist_eval_runner \
    --mode router --input data/Kinesiologist_and_Recovery_evaluation.json
```

---

## Predicted outcome

| Eval | Before | After |
|------|--------|-------|
| 30-case pipeline mode | 17/30 (57%) | ≥ 22/30 (73%) |
| 15-case router mode | 11/15 (73%) | 11/15 (73%) — unchanged |

### Cases newly resolved by step

| Step | Cases fixed | Mechanism |
|------|-------------|-----------|
| 1 — Goal vocab | Case 25 (partial), Case 28 (partial) | "metabolic" → WEIGHT_LOSS; "l-sit" → ATHLETIC_PERFORMANCE |
| 2 — Equipment | Case 16, Case 25 | "for the gym" → FULL GYM; "chair" → BODYWEIGHT |
| 3 — Action verb | Case 28 | "learning" in patterns → EXERCISE_REQUEST wins demotion |
| 4b — Safety LLM | Case 20 | "fluttering in chest + sweaty" caught by LLM judge |
| 5 — Eval runner | Case 7 | "scapula" → pain_areas → health ack satisfied |

### Remaining failures after all steps (8 cases)

These are genuine router policy gaps, not inference gaps — they require a separate policy decision:

| Case | Blocker | Required fix (out of scope) |
|------|---------|----------------------------|
| 1 | Body weight absent; gym equipment makes it mandatory | Weight bypass for gym + advanced + specific muscle target |
| 10 | Age + experience absent from message | Lenient gate when equipment + request type both known |
| 13 | "Can I squat?" should be GENERAL_QUESTION | Classifier: technique questions ≠ EXERCISE_REQUEST |
| 18 | Age absent from message | Same as Case 10 |
| 19 | Body weight absent; KETTLEBELL not bodyweight | Weight bypass for single-implement requests |
| 22 | "push-ups" tips classifier to EXERCISE_REQUEST; should be RECOVERY | Classifier: pain + alternative question = RECOVERY |
| 23 | No action verb; "plyo boxes" unrecognised; weight absent | Composite fix across classifier + equipment + weight policy |
| 27 | Goal genuinely unstated | Goal default for beginner + bodyweight + no explicit goal |

---

## Critique revision log

| Revision | Finding | Resolution |
|----------|---------|-----------|
| Rev 1 → Rev 2 | `StateRouter.__init__` breaking change (17 call sites) | `SafetyJudge.__init__()` takes no args |
| Rev 1 → Rev 2 | `safety` missing from `config.yaml` | Step 4a added |
| Rev 1 → Rev 2 | `import re` missing from `state_router.py` | Step 2 checklist item added |
| Rev 1 → Rev 2 | `learning` pattern false-positive | Non-optional suffix in both patterns |
| Rev 1 → Rev 2 | `_GYM_NEGATIONS: list[str]` → should be compiled regex | `_GYM_NEG_PATTERN: re.Pattern` |
| Rev 1 → Rev 2 | No retry in `_run_llm_assessment` | `for attempt in range(2)` loop added |
| Rev 1 → Rev 2 | Structured JSON output not needed for binary decision | `SafetyAssessment(BaseModel)` removed; YES/NO plain text |
| Rev 2 → Rev 3 | `assess()` parameter order swapped at call site | Interface and call site aligned: `assess(message, context)` |
| Rev 2 → Rev 3 | Deep scan (state_router:269-278) dropped silently | L1b `_scan_message_deep()` added; all four layers documented |
| Rev 2 → Rev 3 | `_GOAL_VOCABULARY` incomplete — only new terms listed | Full carry-forward inventory table added to Step 1 |
| Rev 2 → Rev 3 | `_GYM_NEG_PATTERN` doesn't cover "intimidated" | `"intimidated"` added to alternation; test tightened |
| Rev 2 → Rev 3 | `Settings()` bypasses singleton | `get_llm_for_agent("safety")` with no second arg |
| Rev 2 → Rev 3 | Step 4a YAML nesting unspecified | Exact `agents:` structure shown with full context |
| Rev 2 → Rev 3 | `assess()` multiple mid-logic returns | Four private helpers; `or` chain → single-exit |
| Rev 2 → Rev 3 | Token budget formula not shown | `max_tokens=5`; formula: `1 × 1.3 = 2 → set 5` |

# Refactor Plan: Intake Friction — Intelligent Inference & Acute Bypass

**Source**: Kinesiologist & Recovery Specialist evaluation of `Validation_Questions_Run_20260326_164519.jsonl`
**Evaluation file**: `data/Kinesiologist_and_Recovery_evaluation.json` (15 cases)
**Date**: 2026-03-26

---

## Phase 1: Diagnosis

### Root Cause Summary

Mapping all 15 kinesiologist findings to structural root causes in the router/classifier pipeline:

| # | Question | Failure | Root Cause |
|---|----------|---------|------------|
| 1 | "gym session" bicep burnout | Asked for equipment | RC-A: "gym session" not in `_infer_equipment_from_message` |
| 2 | Neck stuck from sleeping wrong | Asked for weight + goals | RC-B: no recovery pattern for "slept on X wrong" + acute bypass missing |
| 3 | Handstand at home | Asked for equipment | RC-A: "at home" alone doesn't infer bodyweight |
| 4 | Shins throbbing after run | Asked for age + experience | RC-B: `is_complete_for_recovery()` blocks on missing age |
| 5 | CNS feels fried, training for meet | Asked for goals + equipment | RC-C: "cns feels fried" doesn't match "cns fried" substring |
| 6 | T2 Diabetes, weight loss, beginner | Asked for equipment | RC-A: medical condition + goal + beginner should infer bodyweight/walking |
| 7 | Winged scapula with dumbbells | Asked for training days | RC-D: "winged scapula" infers no goal; `fitness_goals` missing → INTAKE_NEEDED; "training days" question is LLM-level (out-of-router scope) — router fix reduces friction but follow-up question is a separate tuning concern |
| 8 | 12-hr shift, eyes tired, headache | Asked for health conditions | RC-B: minimal intake gate fires; shift recovery should bypass age gate |
| 9 | 4mo postpartum, cleared to train | Asked for equipment | RC-A: "postpartum" context implies home/bodyweight |
| 10 | Park with pull-up bar | Asked for weight + experience | RC-A + RC-E: "pull-up bar" not inferred; weight required even for bodyweight — **partially resolved**: equipment + weight fixed by Steps 1–2; experience level and age gates remain as intentional intake (out of scope for this refactor) |
| 11 | Osteopenia, bone density goal | Asked workout/recovery/both? | RC-D: "bone density" infers no goal; routed to GENERAL_RESPONSE |
| 12 | Electrical zap in elbow | Asked for training goals | Likely stale — `electrical zap` IS in `acute_patterns`; **verify only** |
| 13 | Scoliosis, can I squat? | Asked for fitness goals | RC-D: "squats" triggers EXERCISE_REQUEST; intake fires before advisory route |
| 14 | Concert all night, feet throbbing | Asked for weight + experience | RC-B: minimal intake fires; "throbbing" recovery override present but age absent |
| 15 | OA + resistance bands | Asked for session duration | RC-D: "OA" infers no goal → INTAKE_NEEDED; session_duration asked by LLM (out-of-router scope) |

### Five Root Causes

**RC-A — Equipment inference gaps** (cases 1, 3, 6, 9, 10)
`state_router.py:_infer_equipment_from_message` — patterns for "gym session", "at home" alone,
"at a park", "pull-up bar", "postpartum", and "diabetes" + beginner context are absent. Note: case 6
goal inference ("weight loss") already works; equipment is the only missing field. Note: case 10 is
partially resolved — experience level and age gates remain as intentional intake after this refactor.

**RC-B — Acute recovery receives no bypass** (cases 2, 4, 8, 14)
Two gates fire sequentially: (1) router's `is_complete_for_recovery()` at line 133 returns
`INTAKE_NEEDED` when age/health is absent; (2) `coach.py:_create_minimal_intake_prompt` fires its
own Tier 1 gate for any non-safety intent. Neither has an acute-injury bypass. The kinesiologist's
ground truth: acute environmental/injury pain warrants template-speed response, not intake
questioning.

**RC-C — CNS/deload literal match failure** (case 5)
`intent_classifier.py:recovery_override_terms` contains `"cns fried"` as a literal substring.
`"my CNS feels fried"` does not contain this substring. The term needs variants. Additionally,
"training for a meet" adds EXERCISE_REQUEST score, drowning the recovery signal.

**RC-D — Goal inference too narrow** (cases 7, 11, 13, 15)
`_infer_goal_from_message` has no patterns for: corrective/postural work ("fix my", "winged
scapula"), bone health ("bone density", "osteopenia"), medical fitness ("OA", "osteoarthritis",
"scoliosis can I"), or general health goals. `FitnessGoal.REHABILITATION` exists but is never
inferred. When `fitness_goals` is empty, exercise path returns `INTAKE_NEEDED`.

**RC-E — Weight required even when context is bodyweight-only** (cases 4, 10, 14)
`get_missing_required_fields_for_exercise()` unconditionally requires `weight_kg`. When equipment
is inferred as `[Equipment.BODYWEIGHT]`, weight is neither necessary nor helpful for prescription.

### Smell Catalogue

| File | Line | Smell | Root Cause |
|------|------|-------|------------|
| state_router.py | 338–387 | Incomplete equipment inference — misses "gym session", "at home", "pull-up bar", "park", "postpartum" | No patterns for location-implied equipment |
| state_router.py | 133–140 | `is_complete_for_recovery()` blocks all recovery on missing age, no acute bypass | Intake gate doesn't distinguish urgent vs routine |
| coach.py | 120–131 | Minimal intake gate has no acute bypass for recovery requests | Tier 1 gate doesn't distinguish intent urgency |
| intent_classifier.py | 244–246 | `"cns fried"` literal substring — misses "cns feels fried", "cns is fried" | String-literal approach breaks on natural language variants |
| state_router.py | 254–304 | `_infer_goal_from_message` has no rehab/medical/corrective patterns | Goal inference never suggests `FitnessGoal.REHABILITATION` |
| schemas.py | 115–129 | Weight always required regardless of equipment context | Missing equipment-aware branching in missing fields computation |

### Preservation Contracts

```
PRESERVED (must not change):
- route(message, state) → RoutingDecision  (signature)
- process_message / aprocess_message  (signatures + return types)
- WorkflowType enum values (no new variants)
- CoachResponse, CoachOutput, IntakeResponse  (all fields)
- is_complete_for_exercise() / is_complete_for_recovery()  (signatures — callers in tests)
- get_missing_required_fields_for_exercise() / _for_recovery()  (signatures)
- All 228 existing tests must pass at every step

FREE TO CHANGE (internal implementation only):
- Bodies of _infer_equipment_from_message, _infer_goal_from_message, _infer_experience_from_message
- recovery_override_terms list in intent_classifier.py
- Logic inside get_missing_required_fields_for_exercise()
- Tier 1 gate skip conditions inside _create_minimal_intake_prompt and process_message
- New private helpers (_is_acute_recovery_request) — no external callers
- RoutingDecision gains is_acute: bool = False (additive field, default value — no existing
  test constructors break; follows the same pattern as safety_concerns: list[str])
```

---

## Phase 2: Target State Design

```
CURRENT (over-gating)                     TARGET (inference-first)
──────────────────────────────────────    ────────────────────────────────────────
message → intake_agent                    message → intake_agent
  └─ router.route()                         └─ router.route()
       ├─ safety check                           ├─ safety check
       ├─ infer equipment (limited)    →→→        ├─ infer equipment (expanded RC-A)
       │    "bodyweight" "gym access"             │    + "gym session" "at home"
       │                                          │    + "pull-up bar" "park"
       │                                          │    + "postpartum" context
       ├─ infer goal (limited)         →→→        ├─ infer goal (expanded RC-D)
       │    strength/endurance/etc.               │    + rehab: "fix my", "OA", "scoliosis"
       │                                          │    + "bone density" → REHABILITATION
       ├─ is_complete_for_recovery()   →→→        ├─ if _is_acute_recovery(msg)
       │    always requires age                   │       → bypass age/health gate
       │                                          │    else → is_complete_for_recovery()
       └─ get_missing_for_exercise()   →→→        └─ get_missing_for_exercise()
            always requires weight                     if bodyweight-only → skip weight

coach._create_minimal_intake_prompt()     coach._create_minimal_intake_prompt()
  └─ fires for all recovery intents   →→→   └─ bypassed for acute recovery signals
```

### Interface Contracts

```python
# PRESERVED — signature unchanged
def get_missing_required_fields_for_exercise(self) -> list[str]: ...
def is_complete_for_recovery(self) -> bool: ...
def route(self, message: str, state: ConversationState) -> RoutingDecision: ...

# ADDITIVE — new field on RoutingDecision (default=False, backwards-compatible)
# Follows the existing safety_concerns: list[str] pattern — signals travel through
# the decision object, not via cross-module private method calls.
@dataclass
class RoutingDecision:
    ...
    is_acute: bool = False  # NEW — True when acute bypass fired in router

# NEW private helper (StateRouter only — no external callers)
@staticmethod
def _is_acute_recovery_request(message: str) -> bool:
    """True when message describes acute pain/fatigue warranting zero-friction response."""

# CHANGED — body only, signature preserved
def _infer_equipment_from_message(message: str) -> list[Equipment] | None: ...
def _infer_goal_from_message(message: str) -> FitnessGoal | None: ...
```

### Philosophy Compliance Targets

| Smell | Fix | Philosophy |
|-------|-----|------------|
| Literal "cns fried" substring | Add variants: "cns feels fried", "cns is fried", "nervous system fried" | No fragile string matching |
| Equipment inference misses 5 contexts | Add patterns: gym session, at home, park, pull-up bar, postpartum | No duplication — single inference site |
| Goal inference misses rehab category | Add: "fix my", "OA", "scoliosis", "bone density", "winged scapula" → `REHABILITATION` | No duplication — `_infer_goal_from_message` is the single place |
| Weight required for bodyweight workouts | `if all equipment is BODYWEIGHT → skip weight_kg` | Single exit (function returns list built once) |
| Two independent acute-bypass sites | Extract `_is_acute_recovery_request()` called from both router and coach | SRP — detection logic in one place |

---

## Phase 3: Migration Steps

**Strangler Fig Assessment: YES.** All changes are to internal method bodies or new private helpers.
Old and new implementations coexist at every step. No bridge methods required.

### Step 0 — Verification Baseline

```bash
poetry run pytest --tb=short -q
# Gate: 228 passed, 0 failed
```

Write target-state tests for all 15 eval cases before making any change. Add to
`tests/unit/test_kinesiologist_eval.py`. Tests for cases that are currently broken must use
`@pytest.mark.xfail(strict=True, reason="Fixed in Step N — see REFACTOR_INTAKE_AGENT_FRICTION.md")`
so that: (a) they fail visibly now without breaking CI, (b) they turn green exactly when the fixing
step lands, and (c) CI catches a wrong-step fix if a test passes too early. Do NOT write
characterization tests that assert the current broken behaviour — this creates false green and
looks like a regression when the bug is fixed.

---

### Step 1 — Expand equipment inference (RC-A)

**File**: `src/orchestrator/state_router.py:_infer_equipment_from_message`

Additions:
- `"gym session"`, `"at the gym"`, `"at my gym"`, `"gym workout"` → full gym equipment list
- `"at home"` — **do NOT add to the existing early-return bodyweight list** (lines 343–355).
  That block immediately returns and skips the specific-equipment loop. "I train at home with
  dumbbells" would incorrectly return `[Equipment.BODYWEIGHT]`. Instead, add as a separate
  conditional with an equipment-exclusion guard, inserted between the full-gym block and the
  specific-equipment loop:

  ```python
  # Module-level constant (define once, reused by both "at home" and "diabetes" guards)
  # Includes "gym" for messages like "I have diabetes and go to a gym" — bare "gym" does
  # NOT match the full-gym early-return phrases ("full gym", "gym access", etc.) so it
  # must be excluded here to prevent incorrect BODYWEIGHT inference for gym-goers.
  # Known edge case: "I train at home, no gym" contains "gym" → guard skips bodyweight
  # inference. Acceptable — "no equipment", "no weights", "bodyweight only" cover this
  # phrasing already (early-return list at state_router.py:343–355).
  _HOME_EQUIPMENT_TERMS = [
      "barbell", "dumbbell", "dumbbells", "db ", "kettlebell",
      "bands", "resistance band", "treadmill", "cable", "machine", "gym",
  ]

  # In _infer_equipment_from_message — after the full-gym early-return, before the loop:
  if ("at home" in msg or "at a park" in msg or "in the park" in msg
          or "pull-up bar" in msg or "pullup bar" in msg) \
          and not any(t in msg for t in _HOME_EQUIPMENT_TERMS):
      return [Equipment.BODYWEIGHT]

  if ("postpartum" in msg or "post-partum" in msg or "postnatal" in msg) \
          and not any(t in msg for t in _HOME_EQUIPMENT_TERMS):
      return [Equipment.BODYWEIGHT]

  if "diabetes" in msg \
          and not any(t in msg for t in _HOME_EQUIPMENT_TERMS):
      return [Equipment.BODYWEIGHT]
  ```

  Using `_HOME_EQUIPMENT_TERMS` as a single module-level constant covers all three guards —
  no duplication. Case 6 goal inference ("weight loss") already works; equipment is the only
  missing field. Step 3 must NOT add "diabetes" → GENERAL_FITNESS.

- `"at a park"`, `"in the park"`, `"pull-up bar"`, `"pullup bar"` — included in the same guard
  block above. **Do not add bare `"outdoor"`** — conflicts with existing `"outdoor running"` →
  TREADMILL check at `state_router.py:372`.
- `"postpartum"`, `"post-partum"`, `"postnatal"` — in same guard block above.
- **Do NOT add `"resistance bands"`** — `state_router.py:380` already matches it via the substring
  `"resistance band" in msg` (superstring match). Adding it is a no-op.

Note: all inference methods (`_infer_equipment_from_message`, `_infer_goal_from_message`,
`_infer_experience_from_message`) are gated on `is_exercise_intent` at `state_router.py:105–120`.
They only fire for `EXERCISE_REQUEST` / `INTEGRATED_REQUEST` intents. If a postpartum or diabetes
message routes as `RECOVERY_REQUEST`, equipment and goal inference are intentionally skipped (recovery
doesn't require equipment or fitness goals). This is by design; do not move inference outside the
`is_exercise_intent` guard.

**Tests added**: `test_gym_session_infers_full_gym`, `test_at_home_infers_bodyweight`,
`test_park_pullup_bar_infers_bodyweight`, `test_postpartum_infers_bodyweight`,
`test_diabetes_beginner_infers_bodyweight`,
`test_outdoor_running_still_infers_treadmill` (regression guard against "outdoor" collision)

**Gate**: full suite passes before Step 2.

---

### Step 2 — Weight bypass for bodyweight workouts (RC-E)

**File**: `src/models/schemas.py:get_missing_required_fields_for_exercise`

Change: wrap the `weight_kg` check — only add "weight" to missing if equipment is non-empty AND
`Equipment.BODYWEIGHT` is NOT the only item.

The guard must be placed **inside** the existing `if not self.biometrics.is_complete():` block.
`biometrics.is_complete()` returns `age is not None and weight_kg is not None` (`schemas.py:32–34`).
When weight is None, the outer block fires; the inner guard then skips adding "weight" for
bodyweight-only users. Age collection is unaffected.

```python
# TARGET — full function body context shown to prevent wrong-level placement
def get_missing_required_fields_for_exercise(self) -> list[str]:
    missing = []
    if not self.biometrics.is_complete():          # ← outer guard unchanged
        if self.biometrics.age is None:
            missing.append("age")
        is_bodyweight_only = (                     # ← NEW: placed inside outer guard
            bool(self.available_equipment)
            and all(e == Equipment.BODYWEIGHT for e in self.available_equipment)
        )
        if self.biometrics.weight_kg is None and not is_bodyweight_only:
            missing.append("weight")
    if not self.fitness_goals:
        missing.append("fitness_goals")
    if self.experience_level is None:
        missing.append("experience_level")
    if not self.available_equipment:
        missing.append("equipment")
    return missing
```

After Step 1 infers bodyweight equipment, Step 2 immediately unlocks those users from the weight gate.

**Tests added**: `test_bodyweight_only_skips_weight_requirement`,
`test_gym_equipment_still_requires_weight`

**Gate**: full suite passes before Step 3.

---

### Step 3 — Expand goal inference (RC-D)

**File**: `src/orchestrator/state_router.py:_infer_goal_from_message`

Additions using `FitnessGoal.REHABILITATION` (already in enum):

**Insertion point**: append these patterns after the `skill_patterns` block (before `return None`
at `state_router.py:304`). They must come AFTER all existing goal checks — HYPERTROPHY, STRENGTH,
WEIGHT_LOSS, ENDURANCE, ATHLETIC_PERFORMANCE — so that a message containing both a standard goal
term ("strength") and a rehab term ("scoliosis") returns the standard goal, not REHABILITATION.

- Specific compound body-part forms only — do NOT use bare `"fix my"` as a substring.
  `"fix my" in msg` matches "fix my PR" and "fix my 5K time" → incorrect REHABILITATION.
  Use explicit compounds: `"fix my shoulder"`, `"fix my hip"`, `"fix my knee"`,
  `"fix my back"`, `"fix my posture"`, `"fix my scapula"`, `"correct my posture"`.
- `"winged scapula"`, `"scapula"`, `"scoliosis"`, `"posture"` → `REHABILITATION`
- `"bone density"`, `"osteopenia"`, `"osteoporosis"` → `REHABILITATION`
- `"osteoarthritis"`, `"osteoarthritic"`, `"arthritis"` → `REHABILITATION`
  - **Do NOT add bare `"OA"`**: `_infer_goal_from_message` lowercases msg at line 256
    (`msg = message.lower()`). `"OA"` (uppercase) never matches a lowercase string.
    `"oa"` (lowercase) false-matches `"coach"`, `"road"`, `"board"`, `"soak"`. Use the full
    medical terms; the LLM intake agent handles the abbreviation before the router runs.
- `"postpartum"`, `"post-partum"`, `"postnatal"` → `REHABILITATION`
  - Goal inference correctly sets `fitness_goals`; equipment ([BODYWEIGHT]) is handled by Step 1.
  - Both must be set for the exercise path to proceed — Step 1 and Step 3 are complementary here.
  - Scope constraint: `_infer_goal_from_message` only fires when `is_exercise_intent = True`
    (`state_router.py:105–108`). A postpartum message that routes as `RECOVERY_REQUEST` (e.g., "hip
    pain postpartum") will not have `fitness_goals` inferred — but recovery doesn't check
    `fitness_goals`, so this is correct and intentional.

Do NOT add `"diabetes"` or `"metabolic"` here. Case 6 goal inference ("weight loss") already works;
the failure was equipment-only and is fully addressed in Step 1.

**Tests added**: `test_winged_scapula_infers_rehabilitation`, `test_bone_density_infers_rehabilitation`,
`test_oa_infers_rehabilitation`, `test_postpartum_infers_rehabilitation`

**Gate**: full suite passes before Step 4.

---

### Step 4 — CNS/deload detection fix + neck acute language (RC-C + partial RC-B)

**File**: `src/orchestrator/intent_classifier.py`

Additions to `recovery_override_terms` (fire only when no exercise terms present):
- `"cns feels fried"`, `"cns is fried"`, `"nervous system fried"`, `"neurologically fried"`
- `"system feels fried"` (catchall)
- `"slept on my neck wrong"`, `"slept wrong"`, `"woke up with a stiff"`, `"can't look over"`
- `"neck is stuck"`, `"neck stuck"`, `"can't turn my"`, `"slept on it wrong"`

Additions to `INTENT_PATTERNS[IntentType.RECOVERY_REQUEST]` (score-based path, fire even
when exercise terms co-occur — confirmed API: `INTENT_PATTERNS: dict[IntentType, list[str]]`
at `intent_classifier.py:21`):
```python
# Neck/sleep recovery patterns
r"\b(slept\s*(on\s*)?(\w+\s*)?(wrong|badly|funny))\b",
r"\b(woke\s*up\s*with)\s*(a\s*)?(stiff|sore|tight)\b",
r"\b(can'?t\s*(look|turn|rotate|move))\s*(my\s*)?(\w+\s*)?(over|to|left|right)\b",
# CNS fatigue pattern — REQUIRED for case 5 ("CNS feels fried, training for a meet")
# The existing r"\b(deload|cns\s*(fried|fatigue))\b" uses \s* (whitespace-only) and
# does NOT match "cns feels fried" (intervening non-whitespace word). This new pattern
# accumulates recovery score through normal scoring so recovery can compete with or
# tie exercise score → INTEGRATED_REQUEST, avoiding the INTAKE_NEEDED blocker.
r"\b(cns|nervous\s+system)\s+\w*\s*(feels?\s+)?(fried|toast|wiped|drained)\b",
```

**Why INTENT_PATTERNS and not recovery_override_terms for CNS case 5**: the override
gate at `intent_classifier.py:264` blocks when `any(ex in msg for ex in EXERCISE_TERMS)`.
"Training for a meet" contains exercise terms → override never fires. The INTENT_PATTERNS
path accumulates score regardless; recovery and exercise tie at 0.5/0.5 → `INTEGRATED_REQUEST`.

**Step 4 is a prerequisite for Step 5, not a standalone fix for case 5.** `INTEGRATED_REQUEST`
still passes through the exercise intake gate at `state_router.py:122`, which returns
`INTAKE_NEEDED` for empty state. The `INTAKE_NEEDED` bypass for case 5 is Step 5b's new
exercise+acute block inserted at line 121 (before the intake gate). Step 4 ensures the
classifier produces `INTEGRATED_REQUEST` so that `is_exercise_intent = True` in Step 5b's
new block — without Step 4, the message might not reach that code path at all.

**Tests added**: `test_cns_feels_fried_classifies_recovery`,
`test_slept_on_neck_wrong_classifies_recovery`, `test_cant_look_over_shoulder_classifies_recovery`

**Gate**: full suite passes before Step 5.

---

### Step 5 — Acute recovery bypass (RC-B)

**Files**: `src/orchestrator/state_router.py` + `src/orchestrator/coach.py`

**5a** — Add `_is_acute_recovery_request(message: str) -> bool` as a private static method on
`StateRouter`. Acute signals (any one sufficient):
- `"throbbing"` + present in message
- `"can't look"`, `"can't turn"`, `"can't move"` + body part
- `"slept on X wrong"` / `"slept wrong"` / `"woke up with"`
- `"after shift"`, `"after concert"`, `"all night"` + fatigue body part
- `"tension headache"` — required for case 8 ("I just finished a 12-hour shift. My eyes are
  tired and I have a tension headache."): `"after shift"` does NOT appear in that message
  ("finished a … shift" ≠ "after shift"), so without this entry the acute bypass never fires
  and `test_shift_headache_bypasses_intake` fails. Verified empirically:
  `"tension headache" in msg.lower()` → `True`. With this signal added, all five Step-5
  bypass tests pass: `test_shin_throbbing_bypasses_age_gate`, `test_neck_stuck_bypasses_intake`,
  `test_concert_fatigue_bypasses_intake`, `test_shift_headache_bypasses_intake`,
  `test_cns_fried_exercise_intent_bypasses_intake` — confirmed by live simulation against
  `data/Kinesiologist_and_Recovery_evaluation.json`.
- `"neck is stuck"`, `"jaw is stuck"`, `"back is stuck"` — require explicit body-part prefix;
  do NOT use bare `"stuck"` alone (a message like "I'm stuck on what exercises to do for my
  shoulders" contains "stuck" and a body part but is not acute). These specific forms are also
  added to `recovery_override_terms` in Step 4 alongside the other neck-specific terms.
- `"locked up"`, `"seized up"`
- CNS deload signals: `"cns feels fried"`, `"cns is fried"`, `"nervous system fried"`,
  `"cns fried"` — kinesiologist ground truth for case 5: "CNS Fried state is a specialized
  status request; deliver Deload Protocol immediately." CNS deload warrants zero-friction
  response equivalent to acute pain.
  - **Do NOT include bare `"cns feels"`**: `"after deload my cns feels good"` matches it →
    false acute bypass → user asking for a workout gets RECOVERY_ONLY instead. The specific
    four-term list above is exhaustive and covers all eval-file variants.

**5b** — Two changes to `route()`, applied in the same commit:

**Change 1 — Exercise+acute bypass block at `state_router.py:121`.**
Insert immediately after the `_infer_equipment_from_message` call (line 120), **before** the
exercise intake gate at line 122. This ordering is critical: line 122 returns `INTAKE_NEEDED`
for any incomplete exercise context — the new block must intercept before that gate fires.

```python
# INSERT at state_router.py:121 — after equipment inference (line 120),
# before the exercise intake gate (line 122).
# Handles case 5: "CNS feels fried, training for a meet" → INTEGRATED_REQUEST intent
# (exercise-term co-occurrence prevents recovery_override_terms from firing), but
# kinesiologist ground truth is immediate deload.
# _is_acute_recovery_request is NOT gated on exercise terms.
# NOTE: is_exercise_intent is already computed at line 101; is_recovery_intent is NOT
# yet needed here (it is assigned at line 132 for the recovery block that follows).
if is_exercise_intent and self._is_acute_recovery_request(message):
    return RoutingDecision(
        workflow=WorkflowType.RECOVERY_ONLY,
        intent=intent,
        reason="Acute CNS/deload state — bypassing exercise intake gate",
        missing_fields=[],
        safety_concerns=[],
        is_acute=True,
    )
# Line 122 continues: if is_exercise_intent and not is_complete -> INTAKE_NEEDED
```

**Change 2 — Collapse recovery blocks at lines 132–168.**
Replace the recovery intake check (lines 132–140) and the RECOVERY_ONLY return (lines 161–168)
with a single `if is_recovery_intent:` block. **Delete lines 161–168** in the same commit.

```python
# TARGET — replaces lines 132–140 AND lines 161–168
if is_recovery_intent:
    is_acute = self._is_acute_recovery_request(message)
    if not is_acute and not state.user_context.is_complete_for_recovery():
        return RoutingDecision(workflow=WorkflowType.INTAKE_NEEDED, ...)
    # Non-acute + complete context, or acute → return RECOVERY_ONLY immediately.
    # is_acute travels on RoutingDecision; coach.py reads it without coupling to
    # StateRouter internals (mirrors the existing safety_concerns pattern).
    return RoutingDecision(workflow=WorkflowType.RECOVERY_ONLY, ..., is_acute=is_acute)

# DELETE lines 161–168 (unreachable after Change 2 — every RECOVERY_REQUEST exits above):
# if intent.intent == IntentType.RECOVERY_REQUEST:
#     return RoutingDecision(workflow=WorkflowType.RECOVERY_ONLY, ...)
#
# Verify deletion:
#   grep -n 'if intent.intent == IntentType.RECOVERY_REQUEST' src/orchestrator/state_router.py
# → must return no output.
```

Note: `ruff check` RET505 will NOT catch the lines 161–168 dead code (RET505 targets `else`
after `return`, not standalone unreachable `if` blocks). The `grep` command is the explicit gate.

**5c** — In `coach.py:process_message` and `aprocess_message`, extend `skip_intake_check`:
```python
# routing.is_acute is set by StateRouter in 5b — no cross-module private access.
# This mirrors the existing pattern: routing.safety_concerns carries safety signals
# from the router to the coach without coupling to StateRouter internals.
skip_intake_check = (
    is_definitional
    or routing.workflow == WorkflowType.SAFETY_BLOCK
    or routing.is_acute
)
```

**Tests added**: `test_shin_throbbing_bypasses_age_gate`, `test_neck_stuck_bypasses_intake`,
`test_concert_fatigue_bypasses_intake`, `test_shift_headache_bypasses_intake`,
`test_cns_fried_exercise_intent_bypasses_intake` (INTEGRATED_REQUEST + CNS acute → RECOVERY_ONLY),
`test_acute_bypass_does_not_affect_routine_recovery` (non-acute with no age still → INTAKE_NEEDED)

**Gate**: full suite passes.

---

### Step 6 — Verify case 12 (no code change)

Run live validation against: `"I have a sharp, electrical zap in my elbow when I grip heavy weights."`
and confirm `WorkflowType.SAFETY_BLOCK` is returned. The pattern `"electrical zap"` is already in
`acute_patterns`. If confirmed working, close as stale finding. If failing, add `"electrical"` and
`"zap"` as individual terms (belt-and-suspenders).

---

### Cutover Definition (per step)

Each step is done when:
1. All 228+ existing tests pass
2. Newly added kinesiologist-eval tests for that step pass
3. `poetry run ruff check src/ tests/` exits 0
4. Step 5 only: `grep -n 'if intent.intent == IntentType.RECOVERY_REQUEST' src/orchestrator/state_router.py` returns no output (dead code at lines 161–168 deleted). Note: `ruff` RET505 does NOT catch this automatically — this grep is the explicit gate.

---

## Phase 4: Regression Guard

### Existing Baseline Tests

- `tests/unit/test_orchestration.py` — `TestStateRouter`, `TestRecoveryFollowupRouting`, `TestAgeAndInjuryClassification`
- `tests/unit/test_models.py` — `has_health_acknowledgment`, `get_missing_required_fields_for_exercise/recovery`
- `tests/unit/test_coach_transcript.py` — end-to-end routing + async path
- `tests/integration/test_workflows.py` — full workflow integration

### New Test File

Write **before Step 1**: `tests/unit/test_kinesiologist_eval.py`

Each of the 15 eval cases becomes a target-state test. For cases that are currently broken, use
`@pytest.mark.xfail(strict=True, reason="Fixed in Step N — see REFACTOR_INTAKE_AGENT_FRICTION.md")`.

`xfail(strict=True)` guarantees:
- The test fails loudly now (current broken behavior is not silently green)
- CI turns the test green exactly when the fixing step lands
- CI fails the build if a test passes *before* its step (catches wrong-step fixes)

Do NOT write tests that assert the current broken behaviour (e.g. `assert result == INTAKE_NEEDED`
for a case that should route differently). Writing broken-behaviour assertions as "characterization
tests" creates false green today and looks like a regression when the bug is fixed.

### Critical New Tests Per Step

| Step | Test | Assertion |
|------|------|-----------|
| 1 | `test_gym_session_infers_full_gym` | `available_equipment` contains `BARBELL` |
| 1 | `test_at_home_infers_bodyweight` | `available_equipment == [BODYWEIGHT]` |
| 2 | `test_bodyweight_only_skips_weight` | `get_missing_required_fields_for_exercise()` omits `"weight"` |
| 3 | `test_winged_scapula_infers_rehabilitation` | `_infer_goal_from_message` returns `FitnessGoal.REHABILITATION` |
| 4 | `test_cns_feels_fried_classifies_recovery` | `classify("CNS feels fried")` returns `RECOVERY_REQUEST` (isolated, no exercise terms) |
| 5 | `test_cns_fried_training_for_meet_not_intake` | `route(CASE_5_FULL_MSG, empty_state).workflow == RECOVERY_ONLY` AND `.is_acute == True` — CNS deload signals trigger the new `is_exercise_intent + is_acute` bypass in Step 5b, overriding the exercise intake gate |
| 5 | `test_shin_throbbing_bypasses_age_gate` | `route(msg, state_no_age)` returns `RECOVERY_ONLY`, not `INTAKE_NEEDED` |
| 5 | `test_shift_headache_bypasses_intake` | `route(CASE_8_MSG, state_no_age)` returns `RECOVERY_ONLY` — relies on `"tension headache"` in signal list; without it returns `INTAKE_NEEDED` |
| 5 | `test_acute_bypass_does_not_affect_routine_recovery` | Non-acute recovery with no age still → `INTAKE_NEEDED` |

### Verification Command

Run after every step — must pass before proceeding:

```bash
poetry run pytest --tb=short -q && poetry run ruff check src/ tests/
```

---

> **Ready for `/critique`** — before implementing, stress-test the migration plan,
> especially the preservation contracts and the step ordering.
>
> After `/critique` returns **"Plan is sound, proceed to implementation"**, begin Step 0.
>
> After all migration steps are complete, run `/review` on the modified files before
> calling the refactor done.

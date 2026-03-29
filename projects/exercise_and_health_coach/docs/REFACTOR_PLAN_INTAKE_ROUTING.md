# Refactor plan: Intake + intent routing (conversational recovery & follow-ups)

Revised after `/critique` (rounds 2 + 3 + 4): addressed `last_output` dead-code, circular import on `WorkflowType`, inter-step regression window, underspecified follow-up detection, `RoutingDecision` missing `mode` field, SKILL.md as the real prompt target, YAGNI `active_thread`, split xfail tests, `CoachOutput` construction guard (`audit is not None`, full field mapping, `session_id`), `last_workflow` ordering fix (field added in Step 3 not Step 0.5), `has_health_acknowledgment()` shared helper for `is_complete_for_recovery()` / `_check_minimal_intake()` divergence, inverted follow-up cue detection, `CLARIFICATION` deferred to v2, and two-layer LLM mocking strategy for transcript tests.

---

## Phase 1: Diagnosis

### What breaks (observed)

Source: `data/problem_prompt_and_response.txt`.

1. After the recovery-specific welcome, **"I am 66 and I tweaked my left hamstring"** yields the generic **"What would you like to work on?"** menu instead of continuing the recovery thread or acknowledging extracted context.
2. After a recovery plan is delivered, **"What about icing the hamstring?"** again yields the same generic menu — no use of session context or the prior plan.

### Root causes (structural)

| Area | File / lines | Issue |
|------|----------------|-------|
| Intent: age without "years" | `intent_classifier.py` — `INTAKE_UPDATE` pattern `r"\b(i\s*am\|i'm)\s*\d+\s*(years?\|yrs?)\b"` | Bare age ("I am 66") does not match. |
| Intent: empty scores → GENERAL_QUESTION | `intent_classifier.py` ~256–262 | When few patterns hit, classifier returns `GENERAL_QUESTION` with low confidence. |
| Router: GENERAL_QUESTION always menu | `state_router.py` 75–83 | `GENERAL_QUESTION` / `CLARIFICATION` → `GENERAL_RESPONSE` with no use of state (no continuation, no `last_output`). |
| General response | `coach.py` `_create_general_response` 459–470 | Only definitional shortcuts; everything else → fixed generic menu (465–470). |
| Recovery "completeness" | `schemas.py` `is_complete_for_recovery()` 103–109 | Returns `True` always — router 117–124 never enforces recovery-specific gaps; misleading vs minimal intake elsewhere. |
| INTAKE_UPDATE + missing fields | `state_router.py` 156–165 + `UserContext.get_missing_required_fields()` | Missing fields are exercise-oriented (weight, goals, experience, equipment). Recovery-only users who trigger `INTAKE_UPDATE` are routed to `INTAKE_NEEDED` and asked for gym equipment and fitness goals. |
| Intake extraction | `intake_agent.py` system prompt from `SKILL.md` + merge in `conversation_state.py` | If the LLM does not map "tweaked hamstring" into `pain_areas` / `injuries` / notes, `minimal_intake_complete` may stay false — compounded by intent issues above. |
| `last_output` never written | `coach.py` `_handle_routing` 217–260 | `ConversationState.last_output` is declared (`conversation_state.py:49`) but never assigned after a workflow completes. Follow-up tie-back always sees `None`. |

### Smell catalogue

| File | Smell | Root cause |
|------|--------|------------|
| `intent_classifier.py` | Brittle regex; wrong fallback | Single-message, stateless classification |
| `state_router.py` | Early exit for GENERAL_* | No session phase, `last_workflow`, or follow-up policy |
| `state_router.py` | `WorkflowType` defined here | Belongs in `models/enums.py`; current location will cause circular import when `ConversationState` needs `last_workflow: WorkflowType` |
| `coach.py` | God orchestration | Routing + intake + generic UX in one place without a "continuation" policy |
| `schemas.py` | `is_complete_for_recovery` stub | Placeholder never aligned with router/coach |

### Preservation contracts

**PRESERVE**

- `ExerciseCoach.process_message` / `aprocess_message` signatures and return types.
- `CoachResponse`, `ConversationState`, `UserContext`, `IntakeResponse` field shapes (additive fields OK if backward compatible).
- Workflow contracts: `IntegratedWorkflow` / `RecoveryOnlyWorkflow` execute inputs and `WorkflowResult` usage.
- Gradio/CLI: same entry points and `gr.State` session pattern.
- Existing safety path: `SAFETY_BLOCK`, red-flag scanner, minimal intake when still required for safety.

**FREE TO CHANGE**

- `IntentClassifier` internals and pattern sets; optional API e.g. `classify(message, state)`.
- `StateRouter.route` ordering and branching when state is considered.
- `_create_general_response` behavior for scoped follow-ups.
- `is_complete_for_recovery` implementation (must match real rules).
- `WorkflowType` location (move to `enums.py` — all import sites updated).
- Tests and new small modules (e.g. `continuation.py`, `intent_context.py`).

---

## Phase 2: Target state (revised)

### Conceptual delta

| Current | Target |
|---------|--------|
| `classify(message)` only | `classify(message, state)` **or** pre-step: `continuation_detector(message, state)` — **without** overriding definitional `GENERAL_QUESTION` from the classifier. |
| `GENERAL_QUESTION` → always menu | Tiered handling: definitional → **recovery follow-up** (when `last_workflow` + cues) → contextual `INTAKE_UPDATE` completion → menu. |
| No workflow memory | **`last_workflow`** set when a workflow **finishes successfully**; use **`last_output`** for content; continuation reads workflow first, falls back to `last_output` for migration/legacy. |
| `last_output` never populated | Written in `coach.py` `_handle_routing` (sync) and `_ahandle_routing` (async) after each successful workflow result. |
| `INTAKE_UPDATE` regex misses "I am 66" | Age patterns: "I'm 66", "age 66", etc., without requiring "years". |
| `get_missing_required_fields()` one shape | **Exercise** vs **recovery** missing-field policies; recovery `INTAKE_UPDATE` must not require goals/equipment. |
| `is_complete_for_recovery() == True` | Reflects age + health/pain (aligned with `_check_minimal_intake` / one shared helper). **Intentional UX change:** users who previously got instant recovery plans will now be prompted for age + pain area first. Welcome/onboarding text must be updated to set this expectation. |

### Follow-up content policy

1. **Templates first** for icing/heat/frequency-style questions: general principles, **one reviewed disclaimer** (not individualized medical advice; when in doubt see a clinician), and **tie-back** to `last_output.user_message` when available.
2. **No new full plans** on the follow-up path unless the user explicitly requests a new plan and routing sends them through the recovery workflow again.
3. **Optional small LLM later** (not required for v1): only when `last_output` has enough recovery context and the question is clearly in-domain; cap length; system prompt forbids diagnosis and defers red flags. Default ship path is **template + disclaimer** for consistency and liability bounds.

### Router ordering (must hold)

Order is **conceptual**; implement inside `StateRouter` / coach without breaking safety.

1. **Safety** — unchanged (`SAFETY_BLOCK`).
2. **Definitional `GENERAL_QUESTION`** — classifier already runs definitional patterns before scoring (`intent_classifier.py` 148–163). **Do not** let continuation logic override these (e.g. "What is RPE?" stays definition path → `GENERAL_RESPONSE`, no exercise intake).
3. **`GENERAL_QUESTION` + recovery thread + follow-up cues** — all **four** must be true: (a) `last_workflow == WorkflowType.RECOVERY_ONLY`, (b) message word count ≤ 12, (c) message does **not** contain an `exercise_terms` guard word (see canonical list in Step 3 sub-step 6), (d) message contains `?` **or** starts with a question word (`what`, `how`, `should`, `can`, `is`, `when`, `where`, `why`, `does`). Condition (d) prevents declarative requests like "I need a recovery plan for my hamstring" from false-positiving into the follow-up path. Use the **inverted approach**: any short *question* in a recovery thread is a follow-up *unless* it contains `exercise_terms`. Do not use an allowlist of recovery cue words — an allowlist misses common follow-ups like "Can I massage it?", "Should I rest it?", "Is it OK to keep running?". Extract `exercise_terms` into a shared utility (`src/util/term_lists.py`) consumed by both the classifier override (lines 245–253) and this check. Route to `WorkflowType.RECOVERY_FOLLOWUP`.
4. **`CLARIFICATION`** — **deferred to v2.** Short confirmations like "yes"/"ok" after a recovery plan are ambiguous without additional product design. For v1, `CLARIFICATION` falls through to the tiered `_create_general_response` path unchanged. Log this as a known gap.
5. **Other `GENERAL_QUESTION` / `CLARIFICATION`** — tiered `_create_general_response` (definitional → contextual → menu).
6. **Exercise / recovery intake and workflows** — as today, with aligned `is_complete_for_recovery` and split missing-field lists.

### `INTAKE_UPDATE` completion path

When intake is complete but the user only sent demographics/injury, `state_router.py` 168–175 returns `GENERAL_RESPONSE` ("ready for request"). **Coach** must respond consistently: **continue recovery thread** (e.g. offer to generate or refine recovery plan) — **not** the generic menu. Fold this into tiered `_create_general_response`.

### Intent: injury keywords & tie-breaks

- Add cues for tweak / strained / pulled / hamstring (and similar) toward `RECOVERY_REQUEST` or `INTAKE_UPDATE` with recovery weighting; respect existing **recovery vs exercise** tie-break (`intent_classifier.py` 288–305).
- Extend the **substring recovery override** list (201–253) **carefully** — keep the **`exercise_terms`** guard so phrases like "stretch for my workout" do not flip incorrectly.
- **Tests**: non-questions ("I tweaked my hamstring") vs questions ("What about icing?") to catch overlap with `GENERAL_QUESTION` patterns that require `?`.

### Schema changes (new and additive)

| Change | Type | Location | Notes |
|--------|------|----------|-------|
| Move `WorkflowType` | Relocation | `models/enums.py` | **Prerequisite for Step 3.** Eliminates `conversation_state → state_router → conversation_state` circular import. Update all import sites: `state_router.py`, `coach.py`, any tests. |
| `WorkflowType.RECOVERY_FOLLOWUP` | New enum variant | `models/enums.py` | Replaces the underspecified `mode=follow_up` concept. Explicit variant makes routing testable without a `mode` field hack. `_handle_routing` in `coach.py` dispatches to `_create_recovery_followup_response`. |
| `ConversationState.last_workflow` | New field (additive) | `conversation_state.py` | `last_workflow: WorkflowType \| None = None`. Set when `RECOVERY_ONLY` or `INTEGRATED` workflow completes successfully. Reset on `reset()`. |
| `ConversationState.last_output` | Write path (field exists) | `coach.py` | Already declared at `conversation_state.py:49`. **Must be assigned** in `_handle_routing` and `_ahandle_routing` after `_workflow_result_to_response`. |

---

## Phase 3: Migration plan (strangler)

**Gate after every step:** `poetry run pytest --tb=short -q`

### Step −1 — Baseline verification (before anything else)

Run `poetry run pytest --tb=short -q` against the unmodified codebase. All existing tests must be green before Step 0 begins. If any fail, resolve them now — those failures are not part of this refactor. Without a green baseline, Step 0's `xfail` tests will mask pre-existing failures as expected.

### Step 0 — Characterization / transcript tests **first**

Add **two separate** failing tests (not one composite xfail) so each fix can be independently verified:

- **Test A** (`test_age_intake_does_not_produce_generic_menu`): replay the turn "I am 66 and I tweaked my left hamstring" and assert the response does **not** contain the generic "What would you like to work on?" menu. Mark `xfail` until Steps 1–2 land.
- **Test B** (`test_icing_followup_does_not_produce_generic_menu`): replay a session that delivers a recovery plan, then asks "What about icing the hamstring?" and assert the response is icing-oriented (contains "ice" or a disclaimer), **not** the generic menu. Mark `xfail` until Steps 3–4 land. **Important:** Write Test B's routing assertion using the string value — `assert routing.workflow.value == "recovery_followup"` — **not** `WorkflowType.RECOVERY_FOLLOWUP`, which does not exist until Step 3 sub-step 2. A module-level reference to a missing enum variant causes pytest collection failure for the entire test file, making Test A uncollectable as well. Update the assertion to the enum reference in Step 3 sub-step 2 when the variant is added.

Keeping them separate ensures Step 1 can turn Test A green without hiding a regression in a composite assert.

### Step 0.5 — Write `last_output` after workflow completion *(new — prerequisite for Step 3)*

`last_output` is declared in `ConversationState` but never written. Without this, the follow-up tie-back in Step 3 always reads `None`. This step covers **`last_output` only** — `last_workflow` is added to `ConversationState` in Step 3 sub-step 3, where the field is actually declared.

**In `coach.py` `_handle_routing`** (sync, lines ~248–253) and **`_ahandle_routing`** (async twin), after each workflow branch calls `_workflow_result_to_response(result)`, add the `last_output` write before returning:

```python
# Guard: only write when audit is present (error results have audit=None)
if result.audit is not None:
    state.last_output = CoachOutput(
        session_id=state.session_id,
        audit=result.audit,
        user_message=result.user_message,
        workout_plan=result.workout_plan,
        recovery_plan=result.recovery_plan,
        follow_up_questions=result.follow_up_questions or [],
    )
```

Apply identically to both the `INTEGRATED` branch (line ~246) and the `RECOVERY_ONLY` branch (line ~253). `session_id` comes from `state.session_id` — it is not in `WorkflowResult`.

**Test gates:**
- Happy path: after a mocked recovery workflow completes with a valid `AuditLog`, assert `state.last_output is not None` and `state.last_output.user_message == result.user_message`.
- Failure path: after a mocked workflow returns `_create_error_result(...)` (audit=None), assert `state.last_output` remains `None` — no crash, no partial write.

### Step 1 — Fix `INTAKE_UPDATE` age patterns

Patterns for "I am 66", "I'm 66", "age 66", etc., without requiring "years".

> **Inter-step regression warning:** After this step, "I am 66 and tweaked my hamstring" correctly classifies as `INTAKE_UPDATE`. However, `state_router.py:158` then calls `get_missing_required_fields()`, which checks weight/goals/experience/equipment — all exercise-oriented. All will be missing → router returns `INTAKE_NEEDED` → user is asked for gym equipment and fitness goals. This is **worse** than the current generic-menu behavior. Test A will change from one failure mode to another. This regression is closed in Step 4. Document it in the test run notes so no one is surprised.

### Step 2 — Recovery / injury keywords

Classifier cues as above; tune combined messages to prefer action intent per 288–305; add tie-break tests.

### Step 3 — `last_workflow` + state-aware routing *(depends on Step 0.5)*

**Prerequisite sub-steps (do in order):**

1. **Move `WorkflowType` to `models/enums.py`** — update imports in `state_router.py`, `coach.py`, and all test files. Run suite. Must be green before continuing.
2. **Add `WorkflowType.RECOVERY_FOLLOWUP`** to the enum.
3. **Add `last_workflow: WorkflowType | None = None`** to `ConversationState`. No circular import because `WorkflowType` now lives in `enums.py`. Also add `self.last_workflow = None` to `ConversationState.reset()`.
4. **Wire `last_workflow` write and follow-up routing together**: in `coach.py` `_handle_routing` and `_ahandle_routing`, immediately after the `state.last_output` write added in Step 0.5, set `state.last_workflow = routing.workflow` (which is `WorkflowType.RECOVERY_ONLY` or `INTEGRATED` at that point). Then in `StateRouter.route`, after the definitional check and before the generic `GENERAL_RESPONSE` return, apply the inverted follow-up check (§ router ordering point 3 above). Route to `WorkflowType.RECOVERY_FOLLOWUP`.
5. **Add `_create_recovery_followup_response`** to `coach.py`: template + disclaimer + tie-back to `state.last_output.user_message` (guarded: `if state.last_output is not None`).
6. **Extract `exercise_terms` list** to `src/util/term_lists.py` so both the classifier override (`intent_classifier.py:244`) and the router continuation check use the same guard. The **canonical list** is:
   ```python
   EXERCISE_TERMS = ["workout", "training", "lift", "program", "sets", "reps"]
   ```
   The classifier currently uses 4 terms (`workout, training, lift, program`). Adding `sets` and `reps` is a deliberate expansion — do it at the same time as extraction so the classifier and router are always in sync. Update `intent_classifier.py:244` to import `EXERCISE_TERMS` from `term_lists.py` instead of defining a local list. Add regression test: "5 sets of squats" with `last_workflow=RECOVERY_ONLY` → does **not** route to `RECOVERY_FOLLOWUP`.

Run full suite after each sub-step.

### Step 4 — `is_complete_for_recovery` + split missing fields

**UX change notice:** Fixing this stub changes the behavior for users who open with an unqualified "Give me a recovery plan." They will now see an intake prompt asking for age + pain area — a gate that never existed before. Update the welcome message (Gradio UI and CLI) to frame this as a quick two-question setup, not a blocker. Exercise-path users will also see a changed flow: `INTAKE_UPDATE` now acknowledges receipt when recovery-level fields are met, deferring exercise-specific field prompts to when the user actually requests a workout.

**Update existing tests first (before touching implementation):**
- `tests/unit/test_models.py:160-162` (`test_recovery_requires_less`): add `pain_areas=["hamstring"]` so it has both age AND health acknowledgment — `UserContext(biometrics=Biometrics(age=30), pain_areas=["hamstring"])`.
- `tests/unit/test_models.py:164-166` (`test_recovery_with_pain_areas`): add `biometrics=Biometrics(age=30)` so it satisfies the age requirement.
- Run suite after test updates — they must still pass against the current stub (`return True`) before the implementation changes. This preserves the characterization-first methodology.

**Update `get_missing_required_fields()` test call sites before deleting the old method:**
- `tests/unit/test_models.py:144,158` — replace `get_missing_required_fields()` with `get_missing_required_fields_for_exercise()` (both test the exercise-complete path).
- `tests/integration/test_workflows.py:34,41,46` — same replacement. Run suite after each file. Delete the old method only once all call sites are updated and suite is green.

**Update `is_complete_for_exercise()` at `schemas.py:99-101` before deleting the old method:**
- `is_complete_for_exercise()` currently calls `self.get_missing_required_fields()` directly. Rename this call to `self.get_missing_required_fields_for_exercise()`. Do this before deleting the old method — `conversation_state.py:152` calls `is_complete_for_exercise()`, so a silent `AttributeError` here would disable the exercise completeness gate entirely without a loud test failure.

- **Extract a shared helper `UserContext.has_health_acknowledgment() -> bool`** in `schemas.py`. This is the single authoritative definition. It mirrors the current logic in `_check_minimal_intake` (`conversation_state.py:162-174`) exactly: returns `True` when any of `medical_history.conditions`, `medical_history.injuries`, `pain_areas`, `medical_history.no_concerns_reported` are populated, OR when `medical_history.notes` contains one of the acceptance keywords (`"none"`, `"no "`, `"healthy"`, `"no concerns"`, `"no issues"`). The keyword filter on notes is **intentional and preserved** — raw notes text is not sufficient; the user must have stated there are no concerns or named a condition/area.
- **Implement `is_complete_for_recovery()`** to return `True` when `biometrics.age is not None` and `self.has_health_acknowledgment()`.
- **Update `_check_minimal_intake()`** in `conversation_state.py` to delegate to `self.user_context.has_health_acknowledgment()` instead of re-implementing the logic. This eliminates the divergence.
- Rename/split `get_missing_required_fields()`:
  - `get_missing_required_fields_for_exercise()` — current behavior (weight, goals, experience, equipment). Update `state_router.py:108` to call this version for exercise intents.
  - `get_missing_required_fields_for_recovery()` — age + pain/injury acknowledgment only. Update `state_router.py:158` INTAKE_UPDATE path to call this version.
- Verify all call sites: `state_router.py:108`, `state_router.py:158`, `conversation_state.py:152` (calls `is_complete_for_exercise()` — updated above), `schemas.py:99-101` (`is_complete_for_exercise()` itself — updated above).

This step closes the inter-step regression introduced at Step 1.

### Step 5 — Intake reliability

- **Target file: `skills/intake/SKILL.md`** (loaded by `intake_agent.py:38` via `load_skill("intake")`). Editing `INTAKE_INSTRUCTIONS` in Python code alone has no effect — the SKILL.md prompt dominates.
- Add explicit extraction examples for muscle injury language: "tweaked", "strained", "pulled", "[muscle name] is sore/tight/hurts" → map to `pain_areas` or `medical_history.injuries`; use `raw_extraction_notes` when ambiguous.
- **Tests:** mock the LLM response and assert that a message like "I tweaked my left hamstring" produces an `IntakeResponse` with `pain_areas` or `medical_history.injuries` populated.

### Step 6 — Cleanup

- Remove duplicate health checks if consolidated; ensure `minimal_intake_complete` and router share one definition.
- Delete any temporary delegation shims introduced during migration.
- Verify `active_thread` is not present anywhere — it was considered and rejected (YAGNI: `last_workflow` + `last_output` cover the session thread without an additional concept).

**Cutover:** Both transcript tests green; no regression on `Questions_Run`-style JSONL (run validation runner on a mini set).

---

## Phase 4: Regression guard

**Baseline:** `poetry run pytest --tb=short -q` after every step.

### New unit tests

| Test | File | Added in |
|------|------|----------|
| Test A: age + hamstring turn → no generic menu | `tests/unit/test_coach_transcript.py` | Step 0 |
| Test B: icing follow-up → no generic menu | `tests/unit/test_coach_transcript.py` | Step 0 |
| `last_output` written after recovery workflow (happy path) | `tests/unit/test_coach_transcript.py` | Step 0.5 |
| `last_output` stays `None` when workflow returns error (audit=None) | `tests/unit/test_coach_transcript.py` | Step 0.5 |
| `last_workflow` set to `RECOVERY_ONLY` after recovery workflow | `tests/unit/test_orchestration.py` | Step 3 sub-step 4 |
| Import sanity: `WorkflowType` importable from `enums` without `state_router` | `tests/unit/test_models.py` | Step 3 sub-step 1 |
| "What about icing?" after recovery plan → `RECOVERY_FOLLOWUP` | `tests/unit/test_orchestration.py` | Step 3 |
| "What about stretching before my workout?" → does **not** route to `RECOVERY_FOLLOWUP` | `tests/unit/test_orchestration.py` | Step 3 |
| "I need a recovery plan for my hamstring" with `last_workflow=RECOVERY_ONLY` → does **not** route to `RECOVERY_FOLLOWUP` (declarative, no `?`) | `tests/unit/test_orchestration.py` | Step 3 |
| "5 sets of squats" with `last_workflow=RECOVERY_ONLY` → does **not** route to `RECOVERY_FOLLOWUP` (`exercise_terms` guard) | `tests/unit/test_orchestration.py` | Step 3 sub-step 6 |
| "I am 66… hamstring" → `INTAKE_UPDATE` or `RECOVERY_REQUEST` | `tests/unit/test_intent_classifier.py` | Step 1 |
| "What about icing…?" → not `GENERAL_QUESTION` with generic fallback | `tests/unit/test_intent_classifier.py` | Step 2 |
| "What is RPE?" → definitional path, no intake triggered | `tests/unit/test_intent_classifier.py` | Step 3 |
| `has_health_acknowledgment()` — False with raw notes (no keyword); True with `pain_areas` | `tests/unit/test_models.py` | Step 4 |
| `intake_agent` mock: "tweaked my left hamstring" → `pain_areas` or `medical_history.injuries` populated | `tests/unit/test_agents.py` | Step 5 |

### JSONL sampling regression tests (`tests/integration/test_validation_sample.py`)

Pin a **random sample of ~10 records** from each run file as static pytest fixtures (hardcoded in the test file — not re-read from disk at test time). These assert **intent classification and workflow routing only** — no LLM calls, no network required.

**Source files to sample from:**
```
data/Validation_Questions_Run_20260316_111401.jsonl    — sample ~10
data/Validation_Questions2_Run_20260317_091803.jsonl   — sample ~10
```

**Selection criteria when picking the sample (choose manually, then pin):**
- At least 2 `recovery_request` records — guard the recovery classification path
- At least 2 `exercise_request` records — guard the exercise path
- At least 1 `integrated_request` record
- Prefer records where `expected_intent != predicted_intent` in the run — these are already known edge cases
- Include at least 1 record that involves an injury or pain mention

**What each fixture record asserts:**
1. `IntentClassifier.classify(question).intent == expected_intent` — classification did not regress.
2. `StateRouter.route(question, fresh_state).workflow == WorkflowType(record["workflow"])` — routing did not regress. (`record["workflow"]` is the string value from the JSONL; `WorkflowType(...)` converts it to the enum for comparison.)

No LLM is called. These tests run on every `pytest` invocation and act as a canary for unintended classification shifts introduced by regex changes in Steps 1–2.

### Transcript replay (`tests/integration/test_coach_transcript.py`)

Both Test A and Test B go through `ExerciseCoach.process_message`, which makes **two independent LLM calls per turn**: (1) `IntakeAgent.extract_from_message` and (2) the workflow `execute` method. Both must be mocked — mocking only the workflow still leaves live intake agent calls that require a network connection.

**Required mocks for each test:**
- `IntakeAgent.extract_from_message` → return a fixed `IntakeResponse` (e.g. `extracted_context=UserContext(biometrics=Biometrics(age=66), pain_areas=["left hamstring"]), confidence_score=0.9`)
- `RecoveryOnlyWorkflow.execute` → return a fixed `WorkflowResult` with a valid `AuditLog` (for Test B, the turn that delivers the recovery plan)

**Test A** (`test_age_intake_does_not_produce_generic_menu`): can alternatively test `StateRouter.route(message, state)` directly — no LLM mock needed at all since the router is pure logic. This is preferred for Test A because it isolates the routing fix from any intake agent interaction.

**Test B** (`test_icing_followup_does_not_produce_generic_menu`): pre-populate `state.last_workflow = WorkflowType.RECOVERY_ONLY` and `state.last_output` with a stub `CoachOutput` directly, then call `process_message("What about icing the hamstring?", state)` with only the intake agent mocked. This avoids needing to replay the full prior recovery plan turn.

Assert:
- Test A: response does not contain the generic "What would you like to work on?" menu string.
- Test B: response contains "ice" or a medical disclaimer; does not contain the generic menu string.

### Manual cutover check

Gradio session replay of `data/problem_prompt_and_response.txt` — end-to-end with the live LLM — before merging each phase.

---

## Decisions log

| Topic | Decision |
|-------|----------|
| Follow-ups LLM vs templates | **Templates + one disclaimer** for v1; optional capped LLM only with strict guardrails later. |
| `last_workflow` vs `last_output` | **Explicit `last_workflow`** on success; **`last_output`** for content; continuation checks workflow first, `last_output` fallback. |
| Router ordering | **Safety → definitional → recovery follow-up → other GENERAL/CLARIFICATION (tiered, CLARIFICATION unchanged for v1) → workflows.** |
| `mode` field vs new `WorkflowType` | **`WorkflowType.RECOVERY_FOLLOWUP`** (option b). More explicit, directly testable, avoids adding a loosely typed `mode: str \| None` field to `RoutingDecision`. |
| `WorkflowType` location | Move to **`models/enums.py`** to eliminate the `conversation_state → state_router → conversation_state` circular import that `last_workflow` field would otherwise cause. |
| `active_thread` field | **Rejected — YAGNI.** `last_workflow` + `last_output` fully cover the session thread. Remove from spec. |
| Step 0 xfail test structure | **Two separate tests** (Test A: age/intake; Test B: icing follow-up) so each step's progress is independently observable and no intermediate regression hides in a composite assert. |
| `is_complete_for_recovery()` UX delta | **Intentional behavioral change.** Document in welcome message. Users who previously skipped intake will now see a quick age + pain prompt before their first recovery plan. |
| `has_health_acknowledgment()` authority | Extract as a method on `UserContext` in `schemas.py`. Single definition used by both `is_complete_for_recovery()` and `_check_minimal_intake()`. Notes acceptance requires keyword filter — raw notes text is not sufficient. |
| Follow-up cue detection strategy | **Inverted approach.** Any short (≤12 word) question in a recovery thread is treated as a follow-up unless it contains `exercise_terms`. Allowlist rejected — too narrow, misses "massage", "rest", "elevate", "keep running". |
| `CLARIFICATION` continuation | **Deferred to v2.** Ambiguous without additional product design. Falls through to tiered `_create_general_response` unchanged. |
| Transcript test mocking strategy | Test A: test `StateRouter.route()` directly (no LLM). Test B: pre-populate `state.last_workflow` + `state.last_output` directly, then call `process_message` with intake agent mocked. Avoids full multi-turn replay complexity. |
| `CoachOutput` construction guard | Only assign `state.last_output` when `result.audit is not None`. Full field mapping: `session_id=state.session_id`, `audit`, `user_message`, `workout_plan`, `recovery_plan`, `follow_up_questions or []`. |
| Follow-up question signal (condition d) | Message must contain `?` or start with a question word (`what`, `how`, `should`, `can`, `is`, `when`, `where`, `why`, `does`). Prevents declarative statements from false-positiving into `RECOVERY_FOLLOWUP`. |
| `exercise_terms` canonical list | `["workout", "training", "lift", "program", "sets", "reps"]` — 6 terms. Classifier updated from 4 → 6 at same time as extraction to `term_lists.py`. |

---

## Summary

The bot feels non-conversational because (a) "I am 66" misses `INTAKE_UPDATE`, (b) weak matches fall through to `GENERAL_QUESTION`, (c) `GENERAL_QUESTION` / generic coach path ignores session memory — including after a recovery plan when the user asks about icing, and (d) `last_output` is never written so tie-back is structurally impossible. This plan fixes classification, adds **`last_workflow`**-aware continuation with **template-first** follow-ups and **preserved definitional behavior**, moves `WorkflowType` to `enums.py` to avoid a circular import, adds `WorkflowType.RECOVERY_FOLLOWUP` as an explicit dispatch target, wires the `last_output` write path, aligns recovery completeness with **split missing-field policies**, and uses **two separate transcript-first tests** so green means end-to-end behavior, not only intent labels.

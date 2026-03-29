import re
from dataclasses import dataclass

from src.models.enums import Equipment, ExperienceLevel, FitnessGoal, IntentType, WorkflowType
from src.models.schemas import UserContext
from src.orchestrator.intent_classifier import ClassifiedIntent, IntentClassifier
from src.state.conversation_state import ConversationState
from src.util.term_lists import EXERCISE_TERMS
from src.validators.safety_judge import SafetyJudge

# Question words that signal a follow-up question (condition d of follow-up check)
_QUESTION_WORDS = {"what", "how", "should", "can", "is", "when", "where", "why", "does"}

# Equipment terms that indicate the user has access to specific equipment beyond bodyweight.
# Used by the location/context guards in _infer_equipment_from_message to avoid incorrectly
# inferring BODYWEIGHT for a user who mentions "at home" but also mentions "dumbbells", or
# who has "diabetes" but goes to a gym.
# Note: "gym" is included so that "I have diabetes and train at a gym" is not inferred as
# bodyweight-only. Known edge case: "at home, no gym" contains "gym" and skips the guard;
# this is acceptable because "no equipment", "no weights", and "bodyweight only" cover that
# phrasing via the early-return list above.
_HOME_EQUIPMENT_TERMS = [
    "barbell", "dumbbell", "dumbbells", "db ", "kettlebell",
    "bands", "resistance band", "treadmill", "cable", "machine", "gym",
]

# Goal vocabulary: ordered HYPERTROPHY → STRENGTH → WEIGHT_LOSS → ENDURANCE →
# ATHLETIC_PERFORMANCE → REHABILITATION. REHABILITATION is last so that a message
# containing both a standard goal term ("strength") and a rehab term ("scoliosis")
# returns the standard goal — not REHABILITATION.  Python 3.7+ dict insertion order
# is guaranteed; no additional sorting is needed.
#
# "OA" is intentionally omitted from REHABILITATION: lowercasing "OA" → "oa"
# false-matches "coach", "road", "board", etc.  The LLM intake agent expands
# abbreviations before the router runs; use full medical terms only.
_GOAL_VOCABULARY: dict[FitnessGoal, list[str]] = {
    FitnessGoal.HYPERTROPHY: [
        "hypertrophy", "build muscle", "muscle gain",
        "push day", "pull day", "leg day", "arm day", "arms day",
        "chest day", "back day", "shoulder day", "shoulders day",
        "bro split", "bro-split", "ppl", "push/pull/legs", "push pull legs",
        "boulder shoulders", "bigger", "mass", "size",
        "bicep", "biceps", "burnout", "peak", "pump", "bicep peak",
    ],
    FitnessGoal.STRENGTH: [
        "strength", "stronger",
    ],
    FitnessGoal.WEIGHT_LOSS: [
        "lose weight", "weight loss", "fat loss",
        "lose fat", "burn fat", "metabolic", "circuit", "hiit", "burn calories",
    ],
    FitnessGoal.ENDURANCE: [
        "endurance", "marathon", "run", "running", "jog", "jogging",
        "miles", " 5k", " 10k", "half marathon", "cardio",
    ],
    FitnessGoal.ATHLETIC_PERFORMANCE: [
        "performance", "sport",
        "handstand", "muscle up", "muscle-up", "pistol squat",
        "planche", "front lever", "vertical jump", "box jump",
        "explosive", "basketball", "volleyball", "plyo", "agility", "speed",
    ],
    FitnessGoal.REHABILITATION: [
        # Corrective compound forms (bare "fix my" not used — matches "fix my PR")
        "fix my shoulder", "fix my hip", "fix my knee", "fix my back",
        "fix my posture", "fix my scapula", "correct my posture",
        # Postural / structural
        "winged scapula", "scapula", "scoliosis", "posture",
        # Bone health
        "bone density", "osteopenia", "osteoporosis",
        # Arthritic conditions
        "osteoarthritis", "osteoarthritic", "arthritis",
        # Postpartum
        "postpartum", "post-partum", "postnatal",
    ],
}

# Negation/avoidance pattern for gym keyword inference.  Matches "intimidated by the gym",
# "no gym", "don't go to the gym", etc., within a 25-character window before "gym".
# "intimidated" is included explicitly because it is not a grammatical negation but must
# still prevent full-gym inference for users who describe gym avoidance.
_GYM_NEG_PATTERN: re.Pattern = re.compile(
    r"\b(no|not|don't|doesn't|can't|cannot|without|avoid|skip|lack|intimidated)"
    r"\b[\w\s]{0,25}\bgym\b",
    re.IGNORECASE,
)


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    workflow: WorkflowType
    intent: ClassifiedIntent
    reason: str
    missing_fields: list[str]
    safety_concerns: list[str]
    is_acute: bool = False  # True when _is_acute_recovery_request fired; read by coach.py


class StateRouter:
    """
    Routes user requests to appropriate workflows based on:
    - User intent (exercise, recovery, integrated)
    - Conversation state (intake completeness)
    - Safety checks (red flags)
    """

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self._safety_judge = SafetyJudge()

    def route(
        self,
        message: str,
        state: ConversationState,
    ) -> RoutingDecision:
        """
        Determine the appropriate workflow for a user message.

        Args:
            message: User's input message.
            state: Current conversation state.

        Returns:
            RoutingDecision with workflow type and context.
        """
        # Step 1: Classify intent
        intent = self.intent_classifier.classify(message)

        # Step 2: Check for safety concerns
        safety_concerns = self._check_safety(state.user_context, message)
        if safety_concerns:
            return RoutingDecision(
                workflow=WorkflowType.SAFETY_BLOCK,
                intent=intent,
                reason="Red flag symptoms detected requiring medical clearance",
                missing_fields=[],
                safety_concerns=safety_concerns,
            )

        # Step 3: Handle non-action intents
        if intent.intent in (IntentType.GENERAL_QUESTION, IntentType.CLARIFICATION):
            # Recovery follow-up check (router ordering §3).
            # All four conditions must hold:
            #   (a) last_workflow == RECOVERY_ONLY
            #   (b) message word count ≤ 12
            #   (c) message does NOT contain exercise_terms
            #   (d) message contains '?' OR starts with a question word
            last_wf = getattr(state, "last_workflow", None)
            if last_wf == WorkflowType.RECOVERY_ONLY:
                msg_lower = message.lower().strip()
                word_count = len(msg_lower.split())
                has_exercise_term = any(t in msg_lower for t in EXERCISE_TERMS)
                first_word = msg_lower.split()[0] if msg_lower else ""
                has_question_signal = "?" in message or first_word in _QUESTION_WORDS
                if word_count <= 12 and not has_exercise_term and has_question_signal:
                    return RoutingDecision(
                        workflow=WorkflowType.RECOVERY_FOLLOWUP,
                        intent=intent,
                        reason="Short question in active recovery thread",
                        missing_fields=[],
                        safety_concerns=[],
                    )

            return RoutingDecision(
                workflow=WorkflowType.GENERAL_RESPONSE,
                intent=intent,
                reason="General question or clarification",
                missing_fields=[],
                safety_concerns=[],
            )

        # Step 4: Check intake completeness for action intents
        is_exercise_intent = intent.intent in (
            IntentType.EXERCISE_REQUEST,
            IntentType.INTEGRATED_REQUEST,
        )
        if is_exercise_intent and not state.user_context.fitness_goals:
            inferred_goal = self._infer_goal_from_message(message)
            if inferred_goal and inferred_goal not in state.user_context.fitness_goals:
                state.user_context.fitness_goals.append(inferred_goal)

        # Infer experience level for skill-based requests
        if is_exercise_intent and state.user_context.experience_level is None:
            inferred_level = self._infer_experience_from_message(message)
            if inferred_level:
                state.user_context.experience_level = inferred_level

        # Infer equipment from message to reduce redundant intake
        if is_exercise_intent and not state.user_context.available_equipment:
            inferred_equipment = self._infer_equipment_from_message(message)
            if inferred_equipment:
                state.user_context.available_equipment.extend(inferred_equipment)

        # Acute bypass for exercise intents — fires BEFORE the exercise intake gate so that
        # a CNS-deload or acute-injury message classified as EXERCISE/INTEGRATED_REQUEST is
        # not incorrectly blocked by a missing goals/weight/equipment check.
        if is_exercise_intent and self._is_acute_recovery_request(message):
            return RoutingDecision(
                workflow=WorkflowType.RECOVERY_ONLY,
                intent=intent,
                reason="Acute CNS/deload state — bypassing exercise intake gate",
                missing_fields=[],
                safety_concerns=[],
                is_acute=True,
            )

        if is_exercise_intent and not state.user_context.is_complete_for_exercise():
            missing = state.user_context.get_missing_required_fields_for_exercise()
            return RoutingDecision(
                workflow=WorkflowType.INTAKE_NEEDED,
                intent=intent,
                reason=f"Need more information: {', '.join(missing)}",
                missing_fields=missing,
                safety_concerns=[],
            )

        is_recovery_intent = intent.intent == IntentType.RECOVERY_REQUEST
        if is_recovery_intent:
            is_acute = self._is_acute_recovery_request(message)
            if not is_acute and not state.user_context.is_complete_for_recovery():
                return RoutingDecision(
                    workflow=WorkflowType.INTAKE_NEEDED,
                    intent=intent,
                    reason="Need basic information for recovery plan",
                    missing_fields=["age or pain areas"],
                    safety_concerns=[],
                )
            return RoutingDecision(
                workflow=WorkflowType.RECOVERY_ONLY,
                intent=intent,
                reason="Recovery/mobility-only request",
                missing_fields=[],
                safety_concerns=[],
                is_acute=is_acute,
            )

        # Step 5: Route to appropriate workflow
        if intent.intent == IntentType.EXERCISE_REQUEST:
            return RoutingDecision(
                workflow=WorkflowType.INTEGRATED,  # Always include recovery
                intent=intent,
                reason="Exercise request - generating integrated plan",
                missing_fields=[],
                safety_concerns=[],
            )

        if intent.intent == IntentType.INTEGRATED_REQUEST:
            return RoutingDecision(
                workflow=WorkflowType.INTEGRATED,
                intent=intent,
                reason="Integrated exercise and recovery request",
                missing_fields=[],
                safety_concerns=[],
            )

        # Step 6: Intake update - continue collecting
        if intent.intent == IntentType.INTAKE_UPDATE:
            # Use recovery-level missing fields for INTAKE_UPDATE so recovery-path users
            # are not asked for weight/goals/equipment (exercise-specific fields).
            # Exercise-specific fields are only requested when the user makes an exercise request.
            missing = state.user_context.get_missing_required_fields_for_recovery()
            if missing:
                return RoutingDecision(
                    workflow=WorkflowType.INTAKE_NEEDED,
                    intent=intent,
                    reason="Continuing intake - still need more info",
                    missing_fields=missing,
                    safety_concerns=[],
                )
            else:
                # Intake complete, but no action requested - just acknowledge
                return RoutingDecision(
                    workflow=WorkflowType.GENERAL_RESPONSE,
                    intent=intent,
                    reason="Information received - ready for workout/recovery request",
                    missing_fields=[],
                    safety_concerns=[],
                )

        # Default fallback
        return RoutingDecision(
            workflow=WorkflowType.GENERAL_RESPONSE,
            intent=intent,
            reason="Unable to determine specific workflow",
            missing_fields=[],
            safety_concerns=[],
        )

    def _check_safety(self, user_context: UserContext, message: str) -> list[str]:
        """Check for safety concerns via SafetyJudge (four-layer, single-exit).

        Delegates to SafetyJudge.assess() which runs: L1 context scan,
        L1b message deep scan, L2 acute-pattern regex, L3 LLM backstop.
        """
        return self._safety_judge.assess(message, user_context)

    @staticmethod
    def _infer_goal_from_message(message: str) -> FitnessGoal | None:
        """Lightweight goal inference to reduce unnecessary intake friction.

        Iterates _GOAL_VOCABULARY in priority order (HYPERTROPHY → STRENGTH →
        WEIGHT_LOSS → ENDURANCE → ATHLETIC_PERFORMANCE → REHABILITATION) and returns
        the first matching goal.  REHABILITATION is last — preserving the contract
        that "I have scoliosis but my main goal is strength" returns STRENGTH.
        """
        msg = message.lower()
        for goal, terms in _GOAL_VOCABULARY.items():
            if any(t in msg for t in terms):
                return goal
        return None

    @staticmethod
    def _infer_experience_from_message(message: str) -> ExperienceLevel | None:
        """Infer experience level from message context to reduce intake friction."""
        msg = message.lower()

        # Explicit mentions
        if "beginner" in msg or "new to" in msg or "just starting" in msg:
            return ExperienceLevel.BEGINNER
        if "intermediate" in msg:
            return ExperienceLevel.INTERMEDIATE
        if "advanced" in msg or "experienced" in msg:
            return ExperienceLevel.ADVANCED

        # Learning/progression language implies beginner-to-intermediate
        learning_patterns = [
            "learn", "how to do", "want to do", "progression",
            "steps", "drills for", "first time",
        ]
        if any(term in msg for term in learning_patterns):
            return ExperienceLevel.BEGINNER

        # Advanced workout splits imply intermediate+
        advanced_patterns = [
            "ppl", "push/pull/legs", "bro split", "5x5",
            "powerlifting", "meet", "competition",
        ]
        if any(term in msg for term in advanced_patterns):
            return ExperienceLevel.INTERMEDIATE

        return None

    @staticmethod
    def _is_acute_recovery_request(message: str) -> bool:
        """Return True when the message describes an acute pain or fatigue state.

        Acute signals warrant a zero-friction response — no intake questions.
        Any single signal is sufficient. Signals are specific enough to avoid
        false positives on routine exercise messages.
        """
        msg = message.lower()
        acute_signals = [
            # Acute pain / injury
            "throbbing",
            "can't look", "can't turn", "can't move",
            "neck is stuck", "jaw is stuck", "back is stuck",
            "locked up", "seized up",
            # Sleep / positional injury
            "slept on my neck wrong", "slept wrong", "woke up with",
            # Post-event fatigue
            "after shift", "after concert", "all night",
            "tension headache",  # case 8: "finished a 12-hour shift … tension headache"
            # CNS deload — kinesiologist ground truth: deliver deload protocol immediately.
            # "cns feels" intentionally omitted to prevent false match on
            # "after deload my cns feels good".
            "cns feels fried", "cns is fried", "nervous system fried", "cns fried",
        ]
        return any(signal in msg for signal in acute_signals)

    @staticmethod
    def _infer_equipment_from_message(message: str) -> list[Equipment] | None:
        """Infer equipment from message to reduce redundant intake questions."""
        msg = message.lower()

        # No equipment / bodyweight only
        if any(
            term in msg
            for term in [
                "no equipment",
                "bodyweight only",
                "bodyweight",
                "at home with no equipment",
                "no weights",
                "chair and a wall",
                "chair and wall",
            ]
        ):
            return [Equipment.BODYWEIGHT]

        # Full gym / gym access — \bgym\b with negation guard.
        # _GYM_NEG_PATTERN matches "no gym", "intimidated by the gym", "don't go to the gym",
        # etc., preventing false positive full-gym inference for users who mention the gym
        # only to express avoidance.
        if re.search(r"\bgym\b", msg, re.IGNORECASE) and not _GYM_NEG_PATTERN.search(msg):
            return [
                Equipment.BARBELL,
                Equipment.DUMBBELL,
                Equipment.MACHINE,
                Equipment.CABLE,
                Equipment.KETTLEBELL,
            ]

        # Location / context guards — infer BODYWEIGHT when the user's phrasing implies
        # no access to loaded equipment. Each guard checks _HOME_EQUIPMENT_TERMS to avoid
        # incorrectly inferring bodyweight for users who mention specific equipment too.
        if (
            "at home" in msg
            or "at a park" in msg
            or "in the park" in msg
            or "pull-up bar" in msg
            or "pullup bar" in msg
        ) and not any(t in msg for t in _HOME_EQUIPMENT_TERMS):
            return [Equipment.BODYWEIGHT]

        if (
            "postpartum" in msg or "post-partum" in msg or "postnatal" in msg
        ) and not any(t in msg for t in _HOME_EQUIPMENT_TERMS):
            return [Equipment.BODYWEIGHT]

        if "diabetes" in msg and not any(t in msg for t in _HOME_EQUIPMENT_TERMS):
            return [Equipment.BODYWEIGHT]

        # Specific equipment mentions
        equipment = []
        if "treadmill" in msg or "outdoor running" in msg:
            equipment.append(Equipment.TREADMILL)
        if "dumbbell" in msg or "dumbbells" in msg or "db " in msg:
            equipment.append(Equipment.DUMBBELL)
        if "kettlebell" in msg or "kb " in msg:
            equipment.append(Equipment.KETTLEBELL)
        if "barbell" in msg:
            equipment.append(Equipment.BARBELL)
        if "bands" in msg or "resistance band" in msg:
            equipment.append(Equipment.BANDS)
        if "medicine ball" in msg or "med ball" in msg:
            equipment.append(Equipment.MEDICINE_BALL)
        if "foam roller" in msg or "roller" in msg:
            equipment.append(Equipment.FOAM_ROLLER)
        if "chair" in msg or "chairs" in msg:
            equipment.append(Equipment.BODYWEIGHT)
        if "plyo box" in msg or "plyo boxes" in msg:
            # No dedicated PLYO_BOX enum; plyometric box work is bodyweight
            equipment.append(Equipment.BODYWEIGHT)

        return equipment if equipment else None

    def should_generate_plan(self, decision: RoutingDecision) -> bool:
        """Check if the routing decision requires plan generation."""
        return decision.workflow in (
            WorkflowType.INTEGRATED,
            WorkflowType.RECOVERY_ONLY,
            # RECOVERY_FOLLOWUP uses a template response, not a full plan generation
        )

    def needs_intake(self, decision: RoutingDecision) -> bool:
        """Check if more intake information is needed."""
        return decision.workflow == WorkflowType.INTAKE_NEEDED

    def is_blocked(self, decision: RoutingDecision) -> bool:
        """Check if request is blocked due to safety."""
        return decision.workflow == WorkflowType.SAFETY_BLOCK

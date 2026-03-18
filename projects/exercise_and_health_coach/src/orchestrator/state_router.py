from dataclasses import dataclass
from enum import Enum

from src.models.enums import Equipment, ExperienceLevel, FitnessGoal, IntentType
from src.models.schemas import UserContext
from src.orchestrator.intent_classifier import ClassifiedIntent, IntentClassifier
from src.state.conversation_state import ConversationState
from src.validators.red_flag_scanner import RedFlagScanner


class WorkflowType(str, Enum):
    """Available workflow types."""

    INTEGRATED = "integrated"  # Exercise + Recovery
    RECOVERY_ONLY = "recovery_only"
    INTAKE_NEEDED = "intake_needed"  # More info required
    SAFETY_BLOCK = "safety_block"  # Red flags detected
    GENERAL_RESPONSE = "general_response"  # Q&A, not plan generation


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    workflow: WorkflowType
    intent: ClassifiedIntent
    reason: str
    missing_fields: list[str]
    safety_concerns: list[str]


class StateRouter:
    """
    Routes user requests to appropriate workflows based on:
    - User intent (exercise, recovery, integrated)
    - Conversation state (intake completeness)
    - Safety checks (red flags)
    """

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.red_flag_scanner = RedFlagScanner(strict_mode=True)

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

        if is_exercise_intent and not state.user_context.is_complete_for_exercise():
            missing = state.user_context.get_missing_required_fields()
            return RoutingDecision(
                workflow=WorkflowType.INTAKE_NEEDED,
                intent=intent,
                reason=f"Need more information: {', '.join(missing)}",
                missing_fields=missing,
                safety_concerns=[],
            )

        is_recovery_intent = intent.intent == IntentType.RECOVERY_REQUEST
        if is_recovery_intent and not state.user_context.is_complete_for_recovery():
            return RoutingDecision(
                workflow=WorkflowType.INTAKE_NEEDED,
                intent=intent,
                reason="Need basic information for recovery plan",
                missing_fields=["age or pain areas"],
                safety_concerns=[],
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

        if intent.intent == IntentType.RECOVERY_REQUEST:
            return RoutingDecision(
                workflow=WorkflowType.RECOVERY_ONLY,
                intent=intent,
                reason="Recovery/mobility-only request",
                missing_fields=[],
                safety_concerns=[],
            )

        # Step 6: Intake update - continue collecting
        if intent.intent == IntentType.INTAKE_UPDATE:
            # Check what's still missing after this update would be applied
            missing = state.user_context.get_missing_required_fields()
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
        """Check for safety concerns in user context and message."""
        concerns = []

        # Scan user context
        scan_result = self.red_flag_scanner.validate(user_context)
        if not scan_result.is_valid:
            concerns.extend(scan_result.errors)

        # Quick scan of current message for acute symptoms
        acute_patterns = [
            "chest pain",
            "can't breathe",
            "shortness of breath",
            "breathless",
            "dizzy",
            "lightheaded",
            "numbness",
            "shooting pain",
            "radiating pain",
            "severe pain",
            "emergency",
            "heart attack",
            "stroke",
            # Neurological red flags (nerve compression indicators)
            "electrical zap",
            "zapping",
            "zap when",
            "sharp zap",
            "tingling",
            "pins and needles",
        ]
        message_lower = message.lower()
        for pattern in acute_patterns:
            if pattern in message_lower:
                concerns.append(f"Acute symptom mentioned: {pattern}")

        # Deep scan the current message through the red flag scanner without mutating state
        if message:
            temp_context = UserContext(
                medical_history=user_context.medical_history.model_copy(deep=True),
            )
            notes = temp_context.medical_history.notes or ""
            temp_context.medical_history.notes = f"{notes}\n{message}".strip()
            msg_scan = self.red_flag_scanner.validate(temp_context)
            if not msg_scan.is_valid:
                concerns.extend(msg_scan.errors)

        return concerns

    @staticmethod
    def _infer_goal_from_message(message: str) -> FitnessGoal | None:
        """Lightweight goal inference to reduce unnecessary intake friction."""
        msg = message.lower()

        # Explicit goal mentions
        if any(term in msg for term in ["hypertrophy", "build muscle", "muscle gain"]):
            return FitnessGoal.HYPERTROPHY
        if "strength" in msg or "stronger" in msg:
            return FitnessGoal.STRENGTH
        if any(term in msg for term in ["lose weight", "weight loss", "fat loss"]):
            return FitnessGoal.WEIGHT_LOSS
        if any(
            term in msg
            for term in [
                "endurance",
                "marathon",
                "run",
                "running",
                "jog",
                "jogging",
                "miles",
                "5k",
                "10k",
                "half marathon",
                "cardio",
            ]
        ):
            return FitnessGoal.ENDURANCE
        if "performance" in msg or "sport" in msg:
            return FitnessGoal.ATHLETIC_PERFORMANCE

        # Implicit goals from workout-type keywords (reduce intake friction for advanced users)
        # These workout splits imply hypertrophy/strength focus
        hypertrophy_workout_patterns = [
            "push day", "pull day", "leg day", "arm day", "arms day",
            "chest day", "back day", "shoulder day", "shoulders day",
            "bro split", "bro-split", "ppl", "push/pull/legs", "push pull legs",
            "boulder shoulders", "bigger", "mass", "size",
        ]
        if any(term in msg for term in hypertrophy_workout_patterns):
            return FitnessGoal.HYPERTROPHY

        # Skill/movement goals imply athletic performance
        skill_patterns = [
            "handstand", "muscle up", "muscle-up", "pistol squat",
            "planche", "front lever", "vertical jump", "box jump",
        ]
        if any(term in msg for term in skill_patterns):
            return FitnessGoal.ATHLETIC_PERFORMANCE

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

        # Full gym / gym access
        if any(
            term in msg
            for term in ["full gym", "gym access", "full gym access", "have a gym"]
        ):
            return [
                Equipment.BARBELL,
                Equipment.DUMBBELL,
                Equipment.MACHINE,
                Equipment.CABLE,
                Equipment.KETTLEBELL,
            ]

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

        return equipment if equipment else None

    def should_generate_plan(self, decision: RoutingDecision) -> bool:
        """Check if the routing decision requires plan generation."""
        return decision.workflow in (
            WorkflowType.INTEGRATED,
            WorkflowType.RECOVERY_ONLY,
        )

    def needs_intake(self, decision: RoutingDecision) -> bool:
        """Check if more intake information is needed."""
        return decision.workflow == WorkflowType.INTAKE_NEEDED

    def is_blocked(self, decision: RoutingDecision) -> bool:
        """Check if request is blocked due to safety."""
        return decision.workflow == WorkflowType.SAFETY_BLOCK

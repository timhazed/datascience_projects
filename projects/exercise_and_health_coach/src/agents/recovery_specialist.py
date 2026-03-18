from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.base_agent import BaseAgent
from src.config.settings import Settings, get_settings
from src.models.schemas import RecoveryPlan, UserContext, WorkoutPlan
from src.skills.loader import load_skill

# Additional operational instructions appended to SKILL.md
RECOVERY_INSTRUCTIONS = """
## Clinical Recovery Logic

### 1. Modality Specificity
- **SMR:** 30-60s on trigger points. Avoid direct pressure on the lumbar spine or joints.
- **Dynamic:** 8-12 reps. Focus on 'lubricating' the joint capsule.
- **Static:** 30-60s. Only prescribe post-workout or for standalone mobility.

### 2. The "Nerve Safety" Filter
- **CRITICAL:** If user has 'zap,' 'tingling,' or 'numbness,' do NOT prescribe stretching/SMR
  for that limb. Substitute with 'Breathing' and 'Active Recovery' (light walking).

### 3. Output Requirements
- Organise JSON blocks strictly: SMR -> Dynamic -> Static.
- Include 'Rationale' explaining why specific antagonists were chosen to balance the workout.
"""


class RecoverySpecialist(BaseAgent[RecoveryPlan]):
    """
    Recovery specialist agent for mobility and recovery prescription.

    Creates evidence-based recovery programs with:
    - Proper modality sequencing (SMR → Dynamic → Static)
    - Duration guidelines (30-60s static holds)
    - Target area selection based on workout or user needs
    - Equipment adaptation
    """

    def __init__(
        self,
        llm: BaseChatModel,
        settings: Settings | None = None,
        max_retries: int = 3,
    ):
        self._settings = settings or get_settings()

        # Load SKILL.md at runtime
        self._skill_content = load_skill("recovery")

        # Recovery plans don't need validation like exercise plans
        super().__init__(llm=llm, max_retries=max_retries, validators=None)

    @property
    def name(self) -> str:
        return "recovery_specialist"

    @property
    def output_schema(self) -> type[RecoveryPlan]:
        return RecoveryPlan

    @property
    def system_prompt(self) -> str:
        """Build system prompt from SKILL.md + operational instructions."""
        return f"{self._skill_content}\n{RECOVERY_INSTRUCTIONS}"

    def format_user_message(self, **kwargs: Any) -> str:
        """
        Format request for recovery prescription.

        Args:
            user_context: UserContext with user information.
            workout_plan: Optional WorkoutPlan to base recovery on.
            specific_request: Optional specific recovery request.
            standalone: Whether this is standalone recovery (not post-workout).

        Returns:
            Formatted prompt for recovery prescription.
        """
        user_context: UserContext = kwargs.get("user_context")
        workout_plan: WorkoutPlan | None = kwargs.get("workout_plan")
        specific_request: str = kwargs.get("specific_request", "")
        standalone: bool = kwargs.get("standalone", workout_plan is None)

        if not user_context:
            raise ValueError("user_context is required")

        parts = []

        # Context type
        if standalone:
            parts.append("## Recovery Type: Standalone Mobility Session")
        else:
            parts.append("## Recovery Type: Post-Workout Recovery")

        # User info
        parts.append("\n## User Profile")
        if user_context.biometrics.age:
            parts.append(f"- Age: {user_context.biometrics.age}")

        # Pain and tightness areas
        if user_context.pain_areas:
            parts.append(f"- Pain/Tightness Areas: {', '.join(user_context.pain_areas)}")

        # Medical considerations
        if user_context.medical_history.injuries:
            injuries = ", ".join(user_context.medical_history.injuries)
            parts.append(f"- Injuries (BE CAUTIOUS): {injuries}")
        if user_context.medical_history.conditions:
            parts.append(f"- Conditions: {', '.join(user_context.medical_history.conditions)}")

        # Equipment
        recovery_equipment = []
        if user_context.available_equipment:
            for eq in user_context.available_equipment:
                if eq.value in ["foam_roller", "lacrosse_ball", "bands"]:
                    recovery_equipment.append(eq.value)

        if recovery_equipment:
            parts.append(f"- Recovery Equipment: {', '.join(recovery_equipment)}")
        else:
            parts.append("- Recovery Equipment: None (bodyweight only)")

        # Workout context
        if workout_plan and not standalone:
            parts.append("\n## Preceding Workout Summary")
            parts.append(self._summarize_workout(workout_plan))

        # Specific request
        if specific_request:
            parts.append(f"\n## Specific Request\n{specific_request}")

        if user_context.specific_requests:
            recovery_requests = [r for r in user_context.specific_requests
                               if any(kw in r.lower() for kw in
                                     ["stretch", "mobility", "recovery", "tight", "stiff", "sore"])]
            if recovery_requests:
                parts.append(f"\n## User Requests\n- {chr(10).join(recovery_requests)}")

        # Task
        parts.append("\n## Task")
        if standalone:
            parts.append("Create a complete standalone recovery/mobility session.")
            parts.append("Address the user's pain areas and common problem spots.")
        else:
            parts.append("Create a post-workout recovery routine.")
            parts.append("Focus on the primary movers from the workout.")

        parts.append("Follow the modality sequence: SMR → Dynamic Mobility → Static Stretching")
        min_hold = self._settings.recovery.static_hold_min_seconds
        max_hold = self._settings.recovery.static_hold_max_seconds
        parts.append(f"Use {min_hold}-{max_hold} second holds for static stretches.")

        return "\n".join(parts)

    def _summarize_workout(self, workout_plan: WorkoutPlan) -> str:
        """Summarize workout for recovery context."""
        lines = []

        # Get all muscles worked
        muscle_sets = workout_plan.get_sets_by_muscle()
        primary_muscles = sorted(
            muscle_sets.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 muscles

        lines.append("Primary muscles worked:")
        for muscle, sets in primary_muscles:
            lines.append(f"- {muscle.value}: {sets} sets")

        # Movement patterns
        patterns = set()
        for block in workout_plan.blocks:
            for ex in block.exercises:
                patterns.add(ex.movement_pattern.value)

        lines.append(f"\nMovement patterns: {', '.join(patterns)}")

        return "\n".join(lines)

    def create_recovery_plan(
        self,
        user_context: UserContext,
        workout_plan: WorkoutPlan | None = None,
        specific_request: str = "",
    ) -> RecoveryPlan:
        """
        Create a recovery plan.

        Convenience method with explicit parameters.

        Args:
            user_context: Complete user context.
            workout_plan: Optional preceding workout to base recovery on.
            specific_request: Optional specific recovery request.

        Returns:
            RecoveryPlan with proper modality sequencing.
        """
        print("Processing started by Recovery Specialist")
        return self.invoke(
            user_context=user_context,
            workout_plan=workout_plan,
            specific_request=specific_request,
            standalone=workout_plan is None,
        )

    async def acreate_recovery_plan(
        self,
        user_context: UserContext,
        workout_plan: WorkoutPlan | None = None,
        specific_request: str = "",
    ) -> RecoveryPlan:
        """Async version of create_recovery_plan."""
        print("Processing started by Recovery Specialist")
        return await self.ainvoke(
            user_context=user_context,
            workout_plan=workout_plan,
            specific_request=specific_request,
            standalone=workout_plan is None,
        )

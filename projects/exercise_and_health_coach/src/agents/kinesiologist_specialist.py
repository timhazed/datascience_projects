from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.base_agent import BaseAgent
from src.config.settings import Settings, get_settings
from src.models.schemas import UserContext, WorkoutPlan
from src.skills.loader import load_skill
from src.validators.base import ValidatorChain
from src.validators.ratio_validator import RatioValidator
from src.validators.volume_validator import VolumeValidator

# Additional operational instructions appended to SKILL.md
KINESIOLOGIST_INSTRUCTIONS = """
## Clinical Programming Instructions

### 1. Goal-Action Alignment (SAID Principle)
- **Running/Endurance:** 'Main Work' MUST be 'cardio' (e.g., Tempo Run, Intervals).
- **Strength/Hypertrophy:** Prioritize compound movements: Squat, Hip Hinge, Push/Pull.
- **RPE Logic:** RPE 5-7 endurance, 7-9 hypertrophy, 8-10 strength.

### 2. Biomechanical Balance & Safety
- **Push:Pull Ratio:** Balance every 'push' with a 'pull' in the same plane.
- **Constraint Adherence:** Use ONLY available equipment. No barbell if 'medicine ball only.'
- **Injury Contraindication:** Avoid direct stress on injured joints (e.g., Lunge vs Squat).

### 3. Output Requirements
- **Workout Blocks:** warm_up, main_work, accessory, finisher.
- **Exercise Details:** sets, reps, rest, RPE, alternatives.
- **Rationale:** Explain how plan meets goals and respects constraints.

### 4. Schema Compliance
- **Strict Enum Usage:** Use designated enums. Map 'lat' to 'back', 'abs' to 'core'.
"""

class KinesiologistSpecialist(BaseAgent[WorkoutPlan]):
    """
    Senior S&C Coach agent for exercise prescription.

    Creates evidence-based exercise programs with:
    - Volume control (18-24 sets/muscle group)
    - Movement pattern balance (Push:Pull 1:1)
    - Equipment adaptation
    - Experience-appropriate exercise selection
    """

    def __init__(
        self,
        llm: BaseChatModel,
        settings: Settings | None = None,
        max_retries: int = 3,
    ):
        self._settings = settings or get_settings()

        # Load SKILL.md at runtime
        self._skill_content = load_skill("kinesiologist")

        # Build validator chain
        validators = ValidatorChain()
        validators.add(VolumeValidator(settings=self._settings))
        validators.add(RatioValidator())

        super().__init__(llm=llm, max_retries=max_retries, validators=validators)

    @property
    def name(self) -> str:
        return "kinesiologist_specialist"

    @property
    def output_schema(self) -> type[WorkoutPlan]:
        return WorkoutPlan

    @property
    def system_prompt(self) -> str:
        """Build system prompt from SKILL.md + operational instructions."""
        return f"{self._skill_content}\n{KINESIOLOGIST_INSTRUCTIONS}"

    def format_user_message(self, **kwargs: Any) -> str:
        """
        Format user context into exercise prescription request.

        Args:
            user_context: UserContext with all user information.
            specific_request: Optional specific workout request.

        Returns:
            Formatted prompt for exercise prescription.
        """
        user_context: UserContext = kwargs.get("user_context")
        specific_request: str = kwargs.get("specific_request", "")

        if not user_context:
            raise ValueError("user_context is required")

        parts = ["## User Profile"]

        # Biometrics
        if user_context.biometrics.age:
            parts.append(f"- Age: {user_context.biometrics.age}")
        if user_context.biometrics.weight_kg:
            parts.append(f"- Weight: {user_context.biometrics.weight_kg} kg")
        if user_context.biometrics.sex:
            parts.append(f"- Sex: {user_context.biometrics.sex}")

        # Experience
        if user_context.experience_level:
            parts.append(f"- Experience Level: {user_context.experience_level}")
        if user_context.training_days_per_week:
            parts.append(f"- Training Days/Week: {user_context.training_days_per_week}")
        if user_context.session_duration_minutes:
            parts.append(f"- Session Duration: {user_context.session_duration_minutes} minutes")

        # Goals
        if user_context.fitness_goals:
            goals = [g.value for g in user_context.fitness_goals]
            parts.append(f"- Goals: {', '.join(goals)}")

        # Equipment
        if user_context.available_equipment:
            equipment = [e.value for e in user_context.available_equipment]
            parts.append(f"- Available Equipment: {', '.join(equipment)}")
        else:
            parts.append("- Available Equipment: bodyweight only")

        # Medical considerations
        parts.append("\n## Medical Considerations")
        if user_context.medical_history.conditions:
            parts.append(f"- Conditions: {', '.join(user_context.medical_history.conditions)}")
        if user_context.medical_history.injuries:
            parts.append(f"- Injuries: {', '.join(user_context.medical_history.injuries)}")
        if user_context.pain_areas:
            pain = ", ".join(user_context.pain_areas)
            parts.append(f"- Pain Areas (AVOID DIRECT LOADING): {pain}")
        if user_context.medical_history.contraindications:
            contras = ", ".join(user_context.medical_history.contraindications)
            parts.append(f"- Contraindications: {contras}")

        if not any([
            user_context.medical_history.conditions,
            user_context.medical_history.injuries,
            user_context.pain_areas,
            user_context.medical_history.contraindications,
        ]):
            parts.append("- No reported medical concerns")

        # Specific request
        if specific_request:
            parts.append(f"\n## Specific Request\n{specific_request}")

        if user_context.specific_requests:
            parts.append(f"\n## User Preferences\n- {chr(10).join(user_context.specific_requests)}")

        # Task
        parts.append("\n## Task")
        parts.append("Create a complete workout plan appropriate for this user.")
        parts.append("Include warm-up, main work, and any necessary accessory work.")
        parts.append("Provide alternatives for key exercises.")

        return "\n".join(parts)

    def create_workout(
        self,
        user_context: UserContext,
        specific_request: str = "",
    ) -> WorkoutPlan:
        """
        Create a workout plan for the user.

        Convenience method with explicit parameters.

        Args:
            user_context: Complete user context.
            specific_request: Optional specific workout request.

        Returns:
            Validated WorkoutPlan.
        """
        print("Processing started by Kinesiologist Specialist")
        return self.invoke(
            user_context=user_context,
            specific_request=specific_request,
        )

    async def acreate_workout(
        self,
        user_context: UserContext,
        specific_request: str = "",
    ) -> WorkoutPlan:
        """Async version of create_workout."""
        print("Processing started by Kinesiologist Specialist")
        return await self.ainvoke(
            user_context=user_context,
            specific_request=specific_request,
        )

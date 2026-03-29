from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.base_agent import BaseAgent
from src.models.schemas import IntakeResponse, UserContext
from src.skills.loader import load_skill

# Additional operational instructions appended to SKILL.md
INTAKE_INSTRUCTIONS = """
## Advanced Extraction Guidelines

### 1. Unit Normalization
- Weight: Convert lbs/stone to kg. 
- Height: Convert feet/inches to cm.

### 2. Efficiency Override (Anti-Friction)
- If user provides complete bio-snapshot (e.g., '30y, 80kg, no injuries, want hypertrophy'),
  set confidence_score to 1.0 and skip all clarification questions.
- If request is 'Recovery' or 'SMR', skip questions about 'Weight' or 'Training Days per Week'.

### 3. Red Flag Signaling
- If neurological (zaps, tingling) or cardiac (fluttering) symptoms detected, note in
  'raw_extraction_notes' to alert the Gatekeeper.
"""

class IntakeAgent(BaseAgent[IntakeResponse]):
    """
    Agent for extracting user context from natural language.

    Analyzes user messages to build a complete UserContext,
    identifying what information is present and what needs clarification.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        max_retries: int = 3,
    ):
        # Load SKILL.md at runtime
        self._skill_content = load_skill("intake")

        super().__init__(llm=llm, max_retries=max_retries, validators=None)

    @property
    def name(self) -> str:
        return "intake_agent"

    @property
    def output_schema(self) -> type[IntakeResponse]:
        return IntakeResponse

    @property
    def system_prompt(self) -> str:
        """Build system prompt from SKILL.md + operational instructions."""
        return f"{self._skill_content}\n{INTAKE_INSTRUCTIONS}"

    def format_user_message(self, **kwargs: Any) -> str:
        """
        Format user message for intake extraction.

        Args:
            user_message: The user's input message.
            existing_context: Optional existing UserContext to build upon.
            conversation_history: Optional previous conversation for context.

        Returns:
            Formatted prompt for extraction.
        """
        user_message = kwargs.get("user_message", "")
        existing_context = kwargs.get("existing_context")
        conversation_history = kwargs.get("conversation_history", "")

        parts = []

        if conversation_history:
            parts.append(f"## Previous Conversation\n{conversation_history}\n")

        if existing_context and isinstance(existing_context, UserContext):
            parts.append("## Already Known Information")
            parts.append(self._format_existing_context(existing_context))
            parts.append("")

        parts.append("## Current User Message")
        parts.append(user_message)
        parts.append("")
        parts.append("Extract any new information from the current message.")
        parts.append("Only include fields where you found new information.")

        return "\n".join(parts)

    def _format_existing_context(self, ctx: UserContext) -> str:
        """Format existing context for the prompt."""
        lines = []

        if ctx.biometrics.age:
            lines.append(f"- Age: {ctx.biometrics.age}")
        if ctx.biometrics.weight_kg:
            lines.append(f"- Weight: {ctx.biometrics.weight_kg} kg")
        if ctx.biometrics.height_cm:
            lines.append(f"- Height: {ctx.biometrics.height_cm} cm")
        if ctx.biometrics.sex:
            lines.append(f"- Sex: {ctx.biometrics.sex}")

        if ctx.fitness_goals:
            goals = [g.value for g in ctx.fitness_goals]
            lines.append(f"- Goals: {', '.join(goals)}")

        if ctx.experience_level:
            lines.append(f"- Experience: {ctx.experience_level}")

        if ctx.available_equipment:
            equipment = [e.value for e in ctx.available_equipment]
            lines.append(f"- Equipment: {', '.join(equipment)}")

        if ctx.medical_history.conditions:
            lines.append(f"- Conditions: {', '.join(ctx.medical_history.conditions)}")

        if ctx.medical_history.injuries:
            lines.append(f"- Injuries: {', '.join(ctx.medical_history.injuries)}")

        if ctx.pain_areas:
            lines.append(f"- Pain areas: {', '.join(ctx.pain_areas)}")

        return "\n".join(lines) if lines else "No information collected yet."

    def extract_from_message(
        self,
        user_message: str,
        existing_context: UserContext | None = None,
        conversation_history: str = "",
    ) -> IntakeResponse:
        """
        Extract context from a user message.

        Convenience method with explicit parameters.

        Args:
            user_message: The user's input.
            existing_context: Previously collected context.
            conversation_history: Summary of previous conversation.

        Returns:
            IntakeResponse with extracted information.
        """
        return self.invoke(
            user_message=user_message,
            existing_context=existing_context,
            conversation_history=conversation_history,
        )

    async def aextract_from_message(
        self,
        user_message: str,
        existing_context: UserContext | None = None,
        conversation_history: str = "",
    ) -> IntakeResponse:
        """Async version of extract_from_message."""
        return await self.ainvoke(
            user_message=user_message,
            existing_context=existing_context,
            conversation_history=conversation_history,
        )

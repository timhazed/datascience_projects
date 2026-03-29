from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.models.enums import IntentType, WorkflowType
from src.models.schemas import CoachOutput, UserContext


class ConversationTurn(BaseModel):
    """Single turn in a conversation."""

    turn_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    timestamp: datetime = Field(default_factory=datetime.now)
    user_message: str
    assistant_response: str
    intent: IntentType | None = None
    context_updates: dict[str, Any] = Field(default_factory=dict)


class ConversationState(BaseModel):
    """
    Session state that persists across conversation turns.

    Used with Gradio's gr.State() to maintain context between messages.
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Accumulated user context from intake
    user_context: UserContext = Field(default_factory=UserContext)

    # Conversation history (limited to recent turns)
    turns: list[ConversationTurn] = Field(default_factory=list)
    max_turns_to_keep: int = Field(default=20, exclude=True)

    # Workflow state
    intake_complete: bool = False
    minimal_intake_complete: bool = False  # Tier 1: age + health concerns confirmed

    # Previous outputs for reference
    last_output: CoachOutput | None = None

    # Last workflow that completed successfully (used for follow-up routing)
    last_workflow: WorkflowType | None = None

    def add_turn(
        self,
        user_message: str,
        assistant_response: str,
        intent: IntentType | None = None,
        context_updates: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        """Add a new conversation turn and update state."""
        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=assistant_response,
            intent=intent,
            context_updates=context_updates or {},
        )
        self.turns.append(turn)
        self.updated_at = datetime.now()

        # Trim old turns if needed
        if len(self.turns) > self.max_turns_to_keep:
            self.turns = self.turns[-self.max_turns_to_keep :]

        return turn

    def update_context(self, updates: UserContext) -> None:
        """Merge new context into existing user context."""
        # Update biometrics
        if updates.biometrics.age is not None:
            self.user_context.biometrics.age = updates.biometrics.age
        if updates.biometrics.weight_kg is not None:
            self.user_context.biometrics.weight_kg = updates.biometrics.weight_kg
        if updates.biometrics.height_cm is not None:
            self.user_context.biometrics.height_cm = updates.biometrics.height_cm
        if updates.biometrics.sex is not None:
            self.user_context.biometrics.sex = updates.biometrics.sex
        if updates.biometrics.resting_heart_rate is not None:
            self.user_context.biometrics.resting_heart_rate = updates.biometrics.resting_heart_rate

        # Update medical history (append, don't replace)
        if updates.medical_history.conditions:
            existing = set(self.user_context.medical_history.conditions)
            self.user_context.medical_history.conditions.extend(
                c for c in updates.medical_history.conditions if c not in existing
            )
        if updates.medical_history.injuries:
            existing = set(self.user_context.medical_history.injuries)
            self.user_context.medical_history.injuries.extend(
                i for i in updates.medical_history.injuries if i not in existing
            )
        if updates.medical_history.medications:
            existing = set(self.user_context.medical_history.medications)
            self.user_context.medical_history.medications.extend(
                m for m in updates.medical_history.medications if m not in existing
            )
        if updates.medical_history.contraindications:
            existing = set(self.user_context.medical_history.contraindications)
            self.user_context.medical_history.contraindications.extend(
                c for c in updates.medical_history.contraindications if c not in existing
            )
        if updates.medical_history.no_concerns_reported:
            self.user_context.medical_history.no_concerns_reported = True
        if updates.medical_history.notes:
            if self.user_context.medical_history.notes:
                self.user_context.medical_history.notes += f"\n{updates.medical_history.notes}"
            else:
                self.user_context.medical_history.notes = updates.medical_history.notes

        # Update fitness goals (append unique)
        if updates.fitness_goals:
            existing = set(self.user_context.fitness_goals)
            self.user_context.fitness_goals.extend(
                g for g in updates.fitness_goals if g not in existing
            )

        # Update equipment (append unique)
        if updates.available_equipment:
            existing = set(self.user_context.available_equipment)
            self.user_context.available_equipment.extend(
                e for e in updates.available_equipment if e not in existing
            )

        # Update single-value fields (replace if provided)
        if updates.experience_level is not None:
            self.user_context.experience_level = updates.experience_level
        if updates.training_days_per_week is not None:
            self.user_context.training_days_per_week = updates.training_days_per_week
        if updates.session_duration_minutes is not None:
            self.user_context.session_duration_minutes = updates.session_duration_minutes

        # Update lists (append unique)
        if updates.specific_requests:
            existing = set(self.user_context.specific_requests)
            self.user_context.specific_requests.extend(
                r for r in updates.specific_requests if r not in existing
            )
        if updates.pain_areas:
            existing = set(self.user_context.pain_areas)
            self.user_context.pain_areas.extend(
                p for p in updates.pain_areas if p not in existing
            )

        # Check if intake is now complete
        self.intake_complete = self.user_context.is_complete_for_exercise()
        self._check_minimal_intake()
        self.updated_at = datetime.now()

    def _check_minimal_intake(self) -> None:
        """Check if minimal safety intake (Tier 1) is complete."""
        # Delegates to the single authoritative definition on UserContext.
        has_age = self.user_context.biometrics.age is not None
        self.minimal_intake_complete = has_age and self.user_context.has_health_acknowledgment()

    def set_minimal_intake_complete(self) -> None:
        """Manually mark minimal intake as complete (e.g., after explicit confirmation)."""
        self.minimal_intake_complete = True
        self.updated_at = datetime.now()

    def get_conversation_summary(self, max_turns: int = 5) -> str:
        """Get a summary of recent conversation for context."""
        recent = self.turns[-max_turns:] if self.turns else []
        if not recent:
            return "No previous conversation."

        lines = []
        for turn in recent:
            lines.append(f"User: {turn.user_message[:100]}...")
            lines.append(f"Assistant: {turn.assistant_response[:100]}...")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset state for a new session while keeping session_id."""
        self.user_context = UserContext()
        self.turns = []
        self.intake_complete = False
        self.minimal_intake_complete = False
        self.last_output = None
        self.last_workflow = None
        self.updated_at = datetime.now()


def create_session() -> ConversationState:
    """Factory function to create a new conversation session."""
    return ConversationState()

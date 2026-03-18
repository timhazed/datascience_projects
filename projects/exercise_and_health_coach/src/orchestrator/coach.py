import logging
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.intake_agent import IntakeAgent
from src.config.settings import Settings, get_settings
from src.llm.llm_factory import get_llm_for_agent
from src.models.enums import AuditStatus, IntentType
from src.models.schemas import AuditLog, CoachOutput, RecoveryPlan, UserContext, WorkoutPlan
from src.orchestrator.state_router import RoutingDecision, StateRouter, WorkflowType
from src.state.conversation_state import ConversationState, create_session
from src.workflows.base import WorkflowResult
from src.workflows.integrated_workflow import IntegratedWorkflow
from src.workflows.recovery_only_workflow import RecoveryOnlyWorkflow

logger = logging.getLogger(__name__)


@dataclass
class CoachResponse:
    """Response from the coach to be displayed to user."""

    message: str
    needs_more_info: bool = False
    missing_fields: list[str] = field(default_factory=list)
    is_rejected: bool = False
    workout_plan: WorkoutPlan | None = None
    recovery_plan: RecoveryPlan | None = None
    audit: AuditLog | None = None
    follow_up_questions: list[str] = field(default_factory=list)


class ExerciseCoach:
    """
    Main entry point for the Exercise and Health Coach system.

    Coordinates:
    - Intake: Extracts user context from conversation
    - Routing: Determines appropriate workflow
    - Execution: Runs workout/recovery workflows
    - Response: Formats output for user
    """

    def __init__(
        self,
        settings: Settings | None = None,
        intake_llm: BaseChatModel | None = None,
    ):
        self._settings = settings or get_settings()

        # Initialize components
        self.intake_agent = IntakeAgent(
            llm=intake_llm or get_llm_for_agent("intake", self._settings),
        )
        self.router = StateRouter()

        # Workflows (lazy init to avoid LLM creation until needed)
        self._integrated_workflow: IntegratedWorkflow | None = None
        self._recovery_workflow: RecoveryOnlyWorkflow | None = None

    @property
    def integrated_workflow(self) -> IntegratedWorkflow:
        """Get or create integrated workflow."""
        if self._integrated_workflow is None:
            self._integrated_workflow = IntegratedWorkflow(settings=self._settings)
        return self._integrated_workflow

    @property
    def recovery_workflow(self) -> RecoveryOnlyWorkflow:
        """Get or create recovery workflow."""
        if self._recovery_workflow is None:
            self._recovery_workflow = RecoveryOnlyWorkflow(settings=self._settings)
        return self._recovery_workflow

    def process_message(
        self,
        message: str,
        state: ConversationState | None = None,
    ) -> tuple[CoachResponse, ConversationState]:
        """
        Process a user message through the coaching system.

        Args:
            message: User's input message.
            state: Current conversation state (created if None).

        Returns:
            Tuple of (CoachResponse, updated ConversationState).
        """
        # Initialize state if needed
        if state is None:
            state = create_session()

        logger.info(f"[Coach] Processing message: {message[:50]}...")

        # Step 1: Extract context from message
        intake_response = self._run_intake(message, state)
        if intake_response.extracted_context:
            state.update_context(intake_response.extracted_context)

        # Capture explicit “no health concerns” statements to reduce intake friction
        self._record_health_denial_if_present(message, state)

        # Step 2: Route to appropriate action
        routing = self.router.route(message, state)

        # Step 2.5: Check minimal intake (Tier 1 - safety check)
        # Skip intake check for: definitional questions, safety blocks, general responses
        if not state.minimal_intake_complete:
            is_definitional = (
                routing.intent.intent == IntentType.GENERAL_QUESTION
                and "definitional_question" in routing.intent.keywords_matched
            )
            skip_intake_check = (
                is_definitional
                or routing.workflow == WorkflowType.SAFETY_BLOCK
            )
            if not skip_intake_check:
                minimal_prompt = self._create_minimal_intake_prompt(state, routing)
                if minimal_prompt is not None:
                    state.add_turn(
                        user_message=message,
                        assistant_response=minimal_prompt.message,
                        intent=IntentType.INTAKE_UPDATE,
                    )
                    return minimal_prompt, state
        logger.info(f"[Coach] Routing decision: {routing.workflow.value}")

        # Step 3: Handle based on routing
        response = self._handle_routing(message, state, routing)

        # Step 4: Update conversation history
        state.add_turn(
            user_message=message,
            assistant_response=response.message,
            intent=routing.intent.intent,
        )

        return response, state

    async def aprocess_message(
        self,
        message: str,
        state: ConversationState | None = None,
    ) -> tuple[CoachResponse, ConversationState]:
        """Async version of process_message."""
        if state is None:
            state = create_session()

        # Extract context
        intake_response = await self.intake_agent.aextract_from_message(
            user_message=message,
            existing_context=state.user_context,
            conversation_history=state.get_conversation_summary(),
        )
        if intake_response.extracted_context:
            state.update_context(intake_response.extracted_context)

        # Route
        routing = self.router.route(message, state)

        # Check minimal intake (Tier 1 - safety check)
        # Skip intake check for: definitional questions, safety blocks, general responses
        if not state.minimal_intake_complete:
            is_definitional = (
                routing.intent.intent == IntentType.GENERAL_QUESTION
                and "definitional_question" in routing.intent.keywords_matched
            )
            skip_intake_check = (
                is_definitional
                or routing.workflow == WorkflowType.SAFETY_BLOCK
            )
            if not skip_intake_check:
                minimal_prompt = self._create_minimal_intake_prompt(state, routing)
                if minimal_prompt is not None:
                    state.add_turn(
                        user_message=message,
                        assistant_response=minimal_prompt.message,
                        intent=IntentType.INTAKE_UPDATE,
                    )
                    return minimal_prompt, state

        # Handle
        response = await self._ahandle_routing(message, state, routing)

        # Update history
        state.add_turn(
            user_message=message,
            assistant_response=response.message,
            intent=routing.intent.intent,
        )

        return response, state

    def _run_intake(self, message: str, state: ConversationState):
        """Run intake agent to extract context."""
        try:
            return self.intake_agent.extract_from_message(
                user_message=message,
                existing_context=state.user_context,
                conversation_history=state.get_conversation_summary(),
            )
        except Exception as e:
            logger.warning(f"[Coach] Intake extraction failed: {e}")
            # Return empty response on failure
            from src.models.schemas import IntakeResponse
            return IntakeResponse(
                extracted_context=UserContext(),
                clarification_needed=[],
                confidence_score=0.0,
            )

    def _handle_routing(
        self,
        message: str,
        state: ConversationState,
        routing: RoutingDecision,
    ) -> CoachResponse:
        """Handle the routing decision."""

        # Safety block
        if routing.workflow == WorkflowType.SAFETY_BLOCK:
            return self._create_safety_response(routing)

        # Need more info
        if routing.workflow == WorkflowType.INTAKE_NEEDED:
            return self._create_intake_prompt(routing, state)

        # General response (Q&A)
        if routing.workflow == WorkflowType.GENERAL_RESPONSE:
            return self._create_general_response(routing, state)

        # Execute workflow
        # Use accumulated specific_requests from context, or current message
        specific_request = self._get_specific_request(message, state)

        if routing.workflow == WorkflowType.INTEGRATED:
            result = self.integrated_workflow.execute(
                user_context=state.user_context,
                specific_request=specific_request,
            )
            return self._workflow_result_to_response(result)

        if routing.workflow == WorkflowType.RECOVERY_ONLY:
            result = self.recovery_workflow.execute(
                user_context=state.user_context,
                specific_request=specific_request,
            )
            return self._workflow_result_to_response(result)

        # Fallback
        fallback_msg = (
            "I'm not sure how to help with that. "
            "Could you tell me more about what you're looking for?"
        )
        return CoachResponse(message=fallback_msg)

    async def _ahandle_routing(
        self,
        message: str,
        state: ConversationState,
        routing: RoutingDecision,
    ) -> CoachResponse:
        """Async version of handle_routing."""
        if routing.workflow == WorkflowType.SAFETY_BLOCK:
            return self._create_safety_response(routing)

        if routing.workflow == WorkflowType.INTAKE_NEEDED:
            return self._create_intake_prompt(routing, state)

        if routing.workflow == WorkflowType.GENERAL_RESPONSE:
            return self._create_general_response(routing, state)

        # Use accumulated specific_requests from context, or current message
        specific_request = self._get_specific_request(message, state)

        if routing.workflow == WorkflowType.INTEGRATED:
            result = await self.integrated_workflow.aexecute(
                user_context=state.user_context,
                specific_request=specific_request,
            )
            return self._workflow_result_to_response(result)

        if routing.workflow == WorkflowType.RECOVERY_ONLY:
            result = await self.recovery_workflow.aexecute(
                user_context=state.user_context,
                specific_request=specific_request,
            )
            return self._workflow_result_to_response(result)

        return CoachResponse(
            message="I'm not sure how to help with that. Could you tell me more?",
        )

    def _create_safety_response(self, routing: RoutingDecision) -> CoachResponse:
        """Create response for safety-blocked request."""
        lines = ["**Safety Notice**\n"]
        consult_msg = (
            "Based on what you've shared, I recommend consulting with a "
            "healthcare provider before starting an exercise program.\n"
        )
        lines.append(consult_msg)

        if routing.safety_concerns:
            lines.append("**Concerns identified:**")
            for concern in routing.safety_concerns[:5]:
                lines.append(f"- {concern}")

        clearance_msg = (
            "\nYour health and safety come first. "
            "Please get medical clearance before we proceed."
        )
        lines.append(clearance_msg)

        return CoachResponse(
            message="\n".join(lines),
            is_rejected=True,
        )

    def _create_minimal_intake_prompt(
        self, state: ConversationState, routing: RoutingDecision
    ) -> CoachResponse:
        """Create prompt for minimal safety intake (Tier 1).
        Recovery intent: age + pain/health only (no weight/experience/equipment).
        Exercise intent: age + health; full profile requested in one message."""
        questions = []
        missing = []

        if state.user_context.biometrics.age is None:
            questions.append("How old are you?")
            missing.append("age")

        # Check if health concerns have been addressed (conditions, injuries, pain_areas)
        has_health_info = (
            bool(state.user_context.medical_history.conditions)
            or bool(state.user_context.medical_history.injuries)
            or bool(state.user_context.pain_areas)
            or bool(
                state.user_context.medical_history.notes
                and any(
                    word in state.user_context.medical_history.notes.lower()
                    for word in ["none", "no ", "healthy", "no concerns", "no issues"]
                )
            )
        )
        if not has_health_info:
            questions.append(
                "Do you have any health conditions, injuries, or concerns? (Say 'none' if clear)"
            )
            missing.append("health_concerns")

        if not questions:
            # Minimal intake is complete
            state.set_minimal_intake_complete()
            return None

        # Recovery: age + pain only. Do NOT ask for weight/experience/equipment.
        is_recovery = routing.intent.intent == IntentType.RECOVERY_REQUEST
        if is_recovery:
            intro = (
                "Welcome! For your recovery routine, I just need:\n\n"
                "- **Age**\n"
                "- **Areas of tightness or discomfort** (or say 'none' if fine)\n\n"
                "Example: \"I'm 55, my wrist hurts when I do push-ups.\""
            )
        else:
            # Exercise: consolidated prompt for full profile
            intro = (
                "Welcome! To create your personalized plan, please share in one message:\n\n"
                "- **Age** and any **health conditions or injuries** (say 'none' if clear)\n"
                "- **Weight** (kg or lbs)\n"
                "- **Experience level** (beginner, intermediate, or advanced)\n"
                "- **Equipment** (treadmill, gym, outdoor running, bodyweight only, etc.)\n\n"
                "Example: \"I'm 45, no health concerns, 180 lbs, advanced, I have a treadmill.\""
            )
        return CoachResponse(
            message=intro,
            needs_more_info=True,
            missing_fields=missing,
            follow_up_questions=questions,
        )

    def _create_intake_prompt(
        self,
        routing: RoutingDecision,
        state: ConversationState,
    ) -> CoachResponse:
        """Create prompt for missing information."""
        missing_fields = routing.missing_fields

        # Friendly prompts for each field
        prompts = {
            "age": "How old are you?",
            "weight": "What's your current weight (in kg or lbs)?",
            "fitness_goals": (
                "What are your fitness goals? "
                "(e.g., build muscle, improve mobility, lose weight, run 5 miles)"
            ),
            "experience_level": (
                "How would you describe your training experience? "
                "(beginner, intermediate, or advanced)"
            ),
            "equipment": (
                "What equipment do you have access to? "
                "(e.g., treadmill, gym, outdoor running, bodyweight only, dumbbells)"
            ),
            "age or pain areas": (
                "Could you tell me your age, or describe where "
                "you're experiencing tightness/discomfort?"
            ),
        }

        intro = "To create a personalized plan for you, I need a bit more information:\n"
        lines = [intro]

        questions = []
        for field_name in missing_fields:
            default_prompt = f"Could you tell me about your {field_name.replace('_', ' ')}?"
            prompt = prompts.get(field_name, default_prompt)
            questions.append(prompt)
            lines.append(f"- {prompt}")

        return CoachResponse(
            message="\n".join(lines),
            needs_more_info=True,
            missing_fields=missing_fields,
            follow_up_questions=questions,
        )

    def _create_general_response(
        self,
        routing: RoutingDecision,
        state: ConversationState,
    ) -> CoachResponse:
        """Create response for general questions or acknowledgments."""
        intent = routing.intent.intent

        if intent == IntentType.CLARIFICATION:
            clarification_msg = (
                "Got it! Is there anything else you'd like to add, "
                "or would you like me to create a workout or recovery plan?"
            )
            return CoachResponse(message=clarification_msg)

        if intent == IntentType.INTAKE_UPDATE:
            if state.intake_complete:
                complete_msg = (
                    "Thanks for that information! I have everything I need. "
                    "Would you like me to create a workout plan, a recovery routine, or both?"
                )
                return CoachResponse(message=complete_msg)
            else:
                return self._create_intake_prompt(routing, state)

        # General question - check if it's a definitional question we can answer
        definitional_answer = self._get_definition_if_known(routing.intent.raw_message)
        if definitional_answer:
            return CoachResponse(message=definitional_answer)

        # Generic help response
        return CoachResponse(
            message="I'm here to help with exercise and recovery planning. You can ask me to:\n\n"
            "- Create a workout plan (just tell me your goals and experience level)\n"
            "- Design a recovery/mobility routine\n"
            "- Help with specific muscle groups or movement patterns\n\n"
            "What would you like to work on?",
        )

    def _workflow_result_to_response(self, result: WorkflowResult) -> CoachResponse:
        """Convert workflow result to coach response."""
        return CoachResponse(
            message=result.user_message,
            needs_more_info=False,
            is_rejected=result.is_rejected,
            workout_plan=result.workout_plan,
            recovery_plan=result.recovery_plan,
            audit=result.audit,
            follow_up_questions=result.follow_up_questions or [],
        )

    def _get_specific_request(self, message: str, state: ConversationState) -> str:
        """
        Get the specific request to pass to workflow.

        Uses accumulated specific_requests from UserContext if available,
        otherwise uses the current message (filtering out generic confirmations).
        """
        # Check if we have accumulated specific requests from intake
        if state.user_context.specific_requests:
            return "; ".join(state.user_context.specific_requests)

        # Filter out generic confirmation messages
        generic_patterns = ["both", "yes", "yeah", "sure", "go ahead", "ok", "okay"]
        message_lower = message.lower().strip()
        if message_lower in generic_patterns:
            # Look for original request in conversation history
            for turn in reversed(state.turns):
                # Find the first substantive user message
                turn_msg = turn.user_message.lower()
                if not any(p in turn_msg for p in generic_patterns) and len(turn_msg) > 20:
                    return turn.user_message
            return ""

        return message

    def _get_definition_if_known(self, message: str) -> str | None:
        """
        Return a definition if the user is asking about a known fitness term.

        Returns None if the term is not in our knowledge base.
        """
        # Common fitness term definitions
        definitions = {
            "smr": (
                "**SMR (Self-Myofascial Release)** is a technique using tools like "
                "foam rollers or lacrosse balls to apply pressure to tight muscles "
                "and fascia. It helps reduce muscle tension, improve blood flow, "
                "and increase range of motion. Common before or after workouts."
            ),
            "doms": (
                "**DOMS (Delayed Onset Muscle Soreness)** is muscle pain that "
                "develops 12-24 hours after exercise, peaking around 24-72 hours. "
                "It's caused by microscopic muscle damage from unfamiliar or intense "
                "exercise. Rest, light movement, and stretching can help recovery."
            ),
            "diaphragmatic breathing": (
                "**Diaphragmatic breathing** (belly breathing) is a technique where "
                "you breathe deeply using your diaphragm rather than shallow chest "
                "breathing. It activates the parasympathetic nervous system, reducing "
                "stress and promoting recovery. Practice: inhale through nose letting "
                "belly expand, exhale slowly through mouth."
            ),
            "hypertrophy": (
                "**Hypertrophy** is the increase in muscle size through resistance "
                "training. It occurs when muscle protein synthesis exceeds breakdown. "
                "Typically achieved with moderate weights (65-85% 1RM), 8-12 reps, "
                "and 3-5 sets per exercise with adequate protein intake."
            ),
            "rpe": (
                "**RPE (Rate of Perceived Exertion)** is a scale (typically 1-10) "
                "measuring how hard you feel you're working. RPE 7 means you could "
                "do 3 more reps, RPE 9 means 1 more rep, RPE 10 is absolute failure. "
                "Useful for autoregulating training intensity."
            ),
            "progressive overload": (
                "**Progressive overload** is gradually increasing training demands "
                "over time to continue making gains. Can be achieved by adding weight, "
                "reps, sets, or reducing rest periods. Essential principle for "
                "strength and muscle development."
            ),
        }

        message_lower = message.lower()

        # Find ALL matching terms in the message
        matched_definitions = []
        for term, definition in definitions.items():
            if term in message_lower:
                matched_definitions.append(definition)

        if matched_definitions:
            return "\n\n".join(matched_definitions)

        return None

    def _record_health_denial_if_present(self, message: str, state: ConversationState) -> None:
        """
        Heuristic: if the user explicitly denies health issues, record it so
        minimal intake doesn't loop unnecessarily.
        """
        if state.user_context.medical_history.no_concerns_reported:
            return

        text = message.lower()
        if any(
            phrase in text
            for phrase in [
                "no health concerns",
                "no health issues",
                "no injuries",
                "no medical conditions",
                "healthy, no issues",
            ]
        ):
            state.user_context.medical_history.no_concerns_reported = True
            if not state.user_context.medical_history.notes:
                state.user_context.medical_history.notes = "User reported no health concerns."
            return

        # More general pattern: "none" followed by a health keyword
        if "none" in text:
            health_keywords = ["injur", "condition", "concern", "issue", "medical"]
            if any(kw in text for kw in health_keywords):
                state.user_context.medical_history.no_concerns_reported = True
                if not state.user_context.medical_history.notes:
                    state.user_context.medical_history.notes = (
                        "User reported none for health concerns."
                    )

    def create_output(
        self,
        response: CoachResponse,
        state: ConversationState,
    ) -> CoachOutput:
        """Create formal CoachOutput from response."""
        return CoachOutput(
            session_id=state.session_id,
            timestamp=datetime.now(),
            workout_plan=response.workout_plan,
            recovery_plan=response.recovery_plan,
            audit=response.audit or AuditLog(status=AuditStatus.APPROVED),
            user_message=response.message,
            follow_up_questions=response.follow_up_questions,
        )

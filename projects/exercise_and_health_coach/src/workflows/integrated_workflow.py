import logging
from datetime import datetime

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.clinical_gatekeeper import ClinicalGatekeeper
from src.agents.kinesiologist_specialist import KinesiologistSpecialist
from src.agents.recovery_specialist import RecoverySpecialist
from src.config.settings import Settings, get_settings
from src.exceptions import AgentExecutionError
from src.llm.llm_factory import get_llm_for_agent
from src.models.enums import AuditStatus
from src.models.schemas import UserContext
from src.workflows.base import BaseWorkflow, WorkflowResult

logger = logging.getLogger(__name__)


class IntegratedWorkflow(BaseWorkflow):
    """
    Workflow A: Integrated Exercise + Recovery

    Steps:
    1. KinesiologistSpecialist generates exercise block
    2. (Validators run within kinesiologist agent)
    3. RecoverySpecialist generates recovery based on workout
    4. ClinicalGatekeeper audits the complete bundle
    """

    def __init__(
        self,
        settings: Settings | None = None,
        kinesiologist_llm: BaseChatModel | None = None,
        recovery_llm: BaseChatModel | None = None,
        gatekeeper_llm: BaseChatModel | None = None,
    ):
        self._settings = settings or get_settings()
        
        # Initialize agents with appropriate LLMs
        self.kinesiologist = KinesiologistSpecialist(
            llm=kinesiologist_llm or get_llm_for_agent("kinesiologist", self._settings),
            settings=self._settings,
        )
        self.recovery_specialist = RecoverySpecialist(
            llm=recovery_llm or get_llm_for_agent("recovery", self._settings),
            settings=self._settings,
        )
        self.gatekeeper = ClinicalGatekeeper(
            llm=gatekeeper_llm or get_llm_for_agent("gatekeeper", self._settings),
            settings=self._settings,
        )

    @property
    def name(self) -> str:
        return "integrated"

    def execute(
        self,
        user_context: UserContext,
        specific_request: str = "",
    ) -> WorkflowResult:
        """
        Execute integrated workflow.

        Args:
            user_context: Complete user context.
            specific_request: Optional specific request.

        Returns:
            WorkflowResult with workout and recovery plans.
        """
        logger.info(f"[{self.name}] Starting integrated workflow")

        try:
            # Step 1: Quick safety check
            if not self.gatekeeper.quick_safety_check(user_context):
                logger.warning(f"[{self.name}] Quick safety check failed")
                audit = self.gatekeeper.audit(user_context)
                return WorkflowResult(
                    success=False,
                    workflow_name=self.name,
                    timestamp=datetime.now(),
                    audit=audit,
                    user_message=self._format_rejection_message(audit),
                )

            # Step 2: Generate workout plan
            logger.info(f"[{self.name}] Generating workout plan")
            workout_plan = self.kinesiologist.create_workout(
                user_context=user_context,
                specific_request=specific_request,
            )

            # Step 3: Generate recovery plan based on workout
            logger.info(f"[{self.name}] Generating recovery plan")
            recovery_plan = self.recovery_specialist.create_recovery_plan(
                user_context=user_context,
                workout_plan=workout_plan,
            )

            # Step 4: Full audit of both plans
            logger.info(f"[{self.name}] Running clinical audit")
            audit = self.gatekeeper.audit(
                user_context=user_context,
                workout_plan=workout_plan,
                recovery_plan=recovery_plan,
            )

            # Step 5: Create result based on audit
            if audit.status == AuditStatus.REJECTED:
                return WorkflowResult(
                    success=False,
                    workflow_name=self.name,
                    timestamp=datetime.now(),
                    workout_plan=workout_plan,
                    recovery_plan=recovery_plan,
                    audit=audit,
                    user_message=self._format_rejection_message(audit),
                )

            logger.info(f"[{self.name}] Workflow completed successfully")
            return WorkflowResult(
                success=True,
                workflow_name=self.name,
                timestamp=datetime.now(),
                workout_plan=workout_plan,
                recovery_plan=recovery_plan,
                audit=audit,
                user_message=self._format_success_message(workout_plan, recovery_plan, audit),
            )

        except AgentExecutionError as e:
            logger.error(f"[{self.name}] Agent error: {e}")
            return self._create_error_result(
                f"Unable to generate your workout plan. Please try rephrasing your request "
                f"or simplify your requirements. (Technical: {e})"
            )

        except Exception as e:
            logger.error(f"[{self.name}] Unexpected error: {e}", exc_info=True)
            # Return graceful error instead of raising to avoid crashing the session
            return self._create_error_result(
                "I encountered an unexpected issue creating your plan. "
                "This might be due to complex requirements. Try a simpler request."
            )

    async def aexecute(
        self,
        user_context: UserContext,
        specific_request: str = "",
    ) -> WorkflowResult:
        """Async version of execute."""
        logger.info(f"[{self.name}] Starting async integrated workflow")

        try:
            # Step 1: Quick safety check
            if not self.gatekeeper.quick_safety_check(user_context):
                audit = await self.gatekeeper.aaudit(user_context)
                return WorkflowResult(
                    success=False,
                    workflow_name=self.name,
                    timestamp=datetime.now(),
                    audit=audit,
                    user_message=self._format_rejection_message(audit),
                )

            # Step 2: Generate workout plan
            workout_plan = await self.kinesiologist.acreate_workout(
                user_context=user_context,
                specific_request=specific_request,
            )

            # Step 3: Generate recovery plan
            recovery_plan = await self.recovery_specialist.acreate_recovery_plan(
                user_context=user_context,
                workout_plan=workout_plan,
            )

            # Step 4: Full audit
            audit = await self.gatekeeper.aaudit(
                user_context=user_context,
                workout_plan=workout_plan,
                recovery_plan=recovery_plan,
            )

            if audit.status == AuditStatus.REJECTED:
                return WorkflowResult(
                    success=False,
                    workflow_name=self.name,
                    timestamp=datetime.now(),
                    workout_plan=workout_plan,
                    recovery_plan=recovery_plan,
                    audit=audit,
                    user_message=self._format_rejection_message(audit),
                )

            return WorkflowResult(
                success=True,
                workflow_name=self.name,
                timestamp=datetime.now(),
                workout_plan=workout_plan,
                recovery_plan=recovery_plan,
                audit=audit,
                user_message=self._format_success_message(workout_plan, recovery_plan, audit),
            )

        except AgentExecutionError as e:
            logger.error(f"[{self.name}] Agent error in async: {e}")
            return self._create_error_result(
                f"Unable to generate your workout plan. Please try rephrasing your request "
                f"or simplify your requirements. (Technical: {e})"
            )

        except Exception as e:
            logger.error(f"[{self.name}] Unexpected error in async: {e}", exc_info=True)
            return self._create_error_result(
                "I encountered an unexpected issue creating your plan. "
                "This might be due to complex requirements. Try a simpler request."
            )

    def _format_success_message(self, workout, recovery, audit) -> str:
        """Format success message for user."""
        lines = ["Here's your personalized training plan:\n"]

        # Workout summary
        lines.append("**Workout**")
        lines.append(f"- Duration: {workout.total_duration_minutes} minutes")
        lines.append(f"- Total sets: {workout.get_total_sets()}")
        lines.append(f"- Difficulty: {workout.difficulty_level}")

        # Exercise list
        for block in workout.blocks:
            lines.append(f"\n*{block.block_name.replace('_', ' ').title()}*")
            for ex in block.exercises:
                lines.append(f"  - {ex.name}: {ex.sets}x{ex.reps}")

        # Recovery summary
        lines.append("\n**Recovery**")
        lines.append(f"- Duration: {recovery.total_duration_minutes} minutes")
        for block in recovery.blocks:
            lines.append(f"\n*{block.modality.value.replace('_', ' ').title()}*")
            for ex in block.exercises:
                lines.append(f"  - {ex.name}: {ex.duration_seconds}s")

        # Audit notes
        if audit.status == AuditStatus.MODIFIED:
            note = audit.approval_notes or "Some modifications were made for safety."
            lines.append(f"\n**Note:** {note}")

        if audit.warnings:
            lines.append("\n**Warnings:**")
            for warning in audit.warnings[:3]:  # Limit to 3
                lines.append(f"- {warning}")

        return "\n".join(lines)

    def _format_rejection_message(self, audit) -> str:
        """Format rejection message for user."""
        lines = ["**Safety Notice**\n"]
        lines.append("I'm unable to provide a workout plan due to safety concerns:\n")

        if audit.rejection_reason:
            lines.append(audit.rejection_reason)

        if audit.red_flags:
            lines.append("\n**Concerns identified:**")
            for flag in audit.red_flags[:5]:
                lines.append(f"- {flag.category.value}: {flag.recommendation}")

        lines.append(
            "\nPlease consult with a healthcare provider before starting an exercise program."
        )

        return "\n".join(lines)

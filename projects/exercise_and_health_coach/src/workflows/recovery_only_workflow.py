import logging
from datetime import datetime

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.clinical_gatekeeper import ClinicalGatekeeper
from src.agents.recovery_specialist import RecoverySpecialist
from src.config.settings import Settings, get_settings
from src.exceptions import AgentExecutionError
from src.llm.llm_factory import get_llm_for_agent
from src.models.enums import AuditStatus
from src.models.schemas import UserContext
from src.workflows.base import BaseWorkflow, WorkflowResult

logger = logging.getLogger(__name__)


class RecoveryOnlyWorkflow(BaseWorkflow):
    """
    Workflow B: Recovery Only

    Steps:
    1. RecoverySpecialist generates standalone mobility routine
    2. ClinicalGatekeeper performs "Acute Symptom Check"

    Used for:
    - Mobility/flexibility requests
    - Recovery routines
    - Pain/tightness management
    """

    def __init__(
        self,
        settings: Settings | None = None,
        recovery_llm: BaseChatModel | None = None,
        gatekeeper_llm: BaseChatModel | None = None,
    ):
        self._settings = settings or get_settings()

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
        return "recovery_only"

    def execute(
        self,
        user_context: UserContext,
        specific_request: str = "",
    ) -> WorkflowResult:
        """
        Execute recovery-only workflow.

        Args:
            user_context: User context (may be minimal for recovery).
            specific_request: Optional specific request.

        Returns:
            WorkflowResult with recovery plan.
        """
        logger.info(f"[{self.name}] Starting recovery-only workflow")

        try:
            # Step 1: Quick safety check (less strict for recovery)
            # Recovery can still help with some conditions, but acute symptoms need referral
            if not self.gatekeeper.quick_safety_check(user_context):
                logger.warning(f"[{self.name}] Safety concerns detected")
                audit = self.gatekeeper.audit(user_context)

                # For recovery, we may still provide guidance with modifications
                if audit.status == AuditStatus.REJECTED and self._has_critical_flags(audit):
                    return WorkflowResult(
                        success=False,
                        workflow_name=self.name,
                        timestamp=datetime.now(),
                        audit=audit,
                        user_message=self._format_rejection_message(audit),
                    )

            # Step 2: Generate recovery plan
            logger.info(f"[{self.name}] Generating recovery plan")
            recovery_plan = self.recovery_specialist.create_recovery_plan(
                user_context=user_context,
                specific_request=specific_request,
            )

            # Step 3: Audit recovery plan
            logger.info(f"[{self.name}] Running clinical audit")
            audit = self.gatekeeper.audit(
                user_context=user_context,
                recovery_plan=recovery_plan,
            )

            if audit.status == AuditStatus.REJECTED:
                return WorkflowResult(
                    success=False,
                    workflow_name=self.name,
                    timestamp=datetime.now(),
                    recovery_plan=recovery_plan,
                    audit=audit,
                    user_message=self._format_rejection_message(audit),
                )

            logger.info(f"[{self.name}] Workflow completed successfully")
            return WorkflowResult(
                success=True,
                workflow_name=self.name,
                timestamp=datetime.now(),
                recovery_plan=recovery_plan,
                audit=audit,
                user_message=self._format_success_message(recovery_plan, audit),
            )

        except AgentExecutionError as e:
            logger.error(f"[{self.name}] Agent error: {e}")
            return self._create_error_result(
                f"Unable to generate your recovery plan. Please try rephrasing your request. "
                f"(Technical: {e})"
            )

        except Exception as e:
            logger.error(f"[{self.name}] Unexpected error: {e}", exc_info=True)
            return self._create_error_result(
                "I encountered an unexpected issue creating your recovery plan. "
                "Please try again with a simpler request."
            )

    async def aexecute(
        self,
        user_context: UserContext,
        specific_request: str = "",
    ) -> WorkflowResult:
        """Async version of execute."""
        logger.info(f"[{self.name}] Starting async recovery-only workflow")

        try:
            # Quick safety check
            if not self.gatekeeper.quick_safety_check(user_context):
                audit = await self.gatekeeper.aaudit(user_context)
                if audit.status == AuditStatus.REJECTED and self._has_critical_flags(audit):
                    return WorkflowResult(
                        success=False,
                        workflow_name=self.name,
                        timestamp=datetime.now(),
                        audit=audit,
                        user_message=self._format_rejection_message(audit),
                    )

            # Generate recovery plan
            recovery_plan = await self.recovery_specialist.acreate_recovery_plan(
                user_context=user_context,
                specific_request=specific_request,
            )

            # Audit
            audit = await self.gatekeeper.aaudit(
                user_context=user_context,
                recovery_plan=recovery_plan,
            )

            if audit.status == AuditStatus.REJECTED:
                return WorkflowResult(
                    success=False,
                    workflow_name=self.name,
                    timestamp=datetime.now(),
                    recovery_plan=recovery_plan,
                    audit=audit,
                    user_message=self._format_rejection_message(audit),
                )

            return WorkflowResult(
                success=True,
                workflow_name=self.name,
                timestamp=datetime.now(),
                recovery_plan=recovery_plan,
                audit=audit,
                user_message=self._format_success_message(recovery_plan, audit),
            )

        except AgentExecutionError as e:
            logger.error(f"[{self.name}] Agent error in async: {e}")
            return self._create_error_result(
                f"Unable to generate your recovery plan. Please try rephrasing your request. "
                f"(Technical: {e})"
            )

        except Exception as e:
            logger.error(f"[{self.name}] Unexpected error in async: {e}", exc_info=True)
            return self._create_error_result(
                "I encountered an unexpected issue creating your recovery plan. "
                "Please try again with a simpler request."
            )

    def _has_critical_flags(self, audit) -> bool:
        """Check if audit has critical red flags."""
        return audit.has_critical_flags() if audit else False

    def _format_success_message(self, recovery, audit) -> str:
        """Format success message for user."""
        lines = ["Here's your personalized recovery routine:\n"]

        lines.append(f"**Total Duration:** {recovery.total_duration_minutes} minutes\n")

        if recovery.target_areas:
            lines.append(f"**Target Areas:** {', '.join(recovery.target_areas)}\n")

        for block in recovery.blocks:
            modality_name = block.modality.value.replace("_", " ").title()
            lines.append(f"**{modality_name}**")

            for ex in block.exercises:
                intensity = f" ({ex.intensity})" if ex.intensity != "moderate" else ""
                lines.append(f"  - {ex.name}: {ex.duration_seconds}s{intensity}")

                if ex.breathing_cue:
                    lines.append(f"    *{ex.breathing_cue}*")

            lines.append("")

        # Notes
        if recovery.rationale:
            lines.append(f"**Why this routine:** {recovery.rationale}")

        if audit.status == AuditStatus.MODIFIED:
            note = audit.approval_notes or "Some modifications were made for safety."
            lines.append(f"\n**Note:** {note}")

        if audit.warnings:
            lines.append("\n**Things to watch:**")
            for warning in audit.warnings[:3]:
                lines.append(f"- {warning}")

        return "\n".join(lines)

    def _format_rejection_message(self, audit) -> str:
        """Format rejection message for user."""
        lines = ["**Safety Notice**\n"]
        consult_msg = (
            "Based on the information provided, I recommend consulting with a "
            "healthcare provider before starting this recovery routine:\n"
        )
        lines.append(consult_msg)

        if audit.rejection_reason:
            lines.append(audit.rejection_reason)

        if audit.red_flags:
            lines.append("\n**Concerns:**")
            for flag in audit.red_flags[:5]:
                lines.append(f"- {flag.recommendation}")

        lines.append("\nGentle movement may still be beneficial, but please get clearance first.")

        return "\n".join(lines)

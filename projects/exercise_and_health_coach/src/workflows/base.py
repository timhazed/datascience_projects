from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from src.models.enums import AuditStatus
from src.models.schemas import AuditLog, RecoveryPlan, UserContext, WorkoutPlan


@dataclass
class WorkflowResult:
    """Result from workflow execution."""

    success: bool
    workflow_name: str
    timestamp: datetime

    # Generated plans
    workout_plan: WorkoutPlan | None = None
    recovery_plan: RecoveryPlan | None = None

    # Audit result
    audit: AuditLog | None = None

    # For error cases
    error_message: str | None = None

    # For user communication
    user_message: str = ""
    follow_up_questions: list[str] | None = None

    @property
    def is_approved(self) -> bool:
        """Check if workflow result is approved."""
        if self.audit is None:
            return False
        return self.audit.status in (AuditStatus.APPROVED, AuditStatus.MODIFIED)

    @property
    def is_rejected(self) -> bool:
        """Check if workflow result is rejected."""
        if self.audit is None:
            return False
        return self.audit.status == AuditStatus.REJECTED


class BaseWorkflow(ABC):
    """
    Abstract base class for workflows.

    Workflows orchestrate multiple agents in sequence to produce
    a complete plan (workout, recovery, or both).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Workflow identifier."""
        pass

    @abstractmethod
    def execute(
        self,
        user_context: UserContext,
        specific_request: str = "",
    ) -> WorkflowResult:
        """
        Execute the workflow.

        Args:
            user_context: Complete user context.
            specific_request: Optional specific request from user.

        Returns:
            WorkflowResult with generated plans and audit.
        """
        pass

    @abstractmethod
    async def aexecute(
        self,
        user_context: UserContext,
        specific_request: str = "",
    ) -> WorkflowResult:
        """Async version of execute."""
        pass

    def _create_error_result(self, error_message: str) -> WorkflowResult:
        """Create an error result."""
        return WorkflowResult(
            success=False,
            workflow_name=self.name,
            timestamp=datetime.now(),
            error_message=error_message,
            user_message=f"I encountered an error: {error_message}",
        )

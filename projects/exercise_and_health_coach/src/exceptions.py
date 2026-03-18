from typing import Any


class CoachException(Exception):
    """Base exception for all coach-related errors."""

    pass


class ValidationError(CoachException):
    """Errors during input/output validation."""

    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(f"Validation error for '{field}': {message}")


class AgentExecutionError(CoachException):
    """Errors from agent execution."""

    def __init__(self, agent_name: str, message: str, original_error: Exception | None = None):
        self.agent_name = agent_name
        self.original_error = original_error
        super().__init__(f"Agent '{agent_name}' error: {message}")


class WorkflowError(CoachException):
    """Errors during workflow execution."""

    def __init__(self, workflow_name: str, step: str, message: str):
        self.workflow_name = workflow_name
        self.step = step
        super().__init__(f"[{workflow_name}] Failed at {step}: {message}")


class RedFlagError(CoachException):
    """Raised when critical red flags require immediate handling."""

    def __init__(self, flags: list[str], message: str = "Critical red flags detected"):
        self.flags = flags
        super().__init__(f"{message}: {', '.join(flags)}")


class IntakeIncompleteError(CoachException):
    """Raised when user context is incomplete for workflow execution."""

    def __init__(self, missing_fields: list[str]):
        self.missing_fields = missing_fields
        super().__init__(f"Intake incomplete. Missing fields: {', '.join(missing_fields)}")


class ConfigurationError(CoachException):
    """Errors in system configuration."""

    def __init__(self, message: str):
        super().__init__(f"Configuration error: {message}")


class LLMError(CoachException):
    """Errors from LLM provider."""

    def __init__(self, provider: str, message: str, status_code: int | None = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"LLM error ({provider}): {message}")

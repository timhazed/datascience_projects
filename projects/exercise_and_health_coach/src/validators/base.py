from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ValidationResult(BaseModel):
    """Result of a validation check."""

    is_valid: bool
    errors: list[str] = []
    warnings: list[str] = []
    suggestions: list[str] = []
    metadata: dict[str, Any] = {}


class BaseValidator(ABC, Generic[T]):
    """
    Abstract base class for validators.

    Validators check domain-specific rules and can either:
    - Return errors (blocking)
    - Return warnings (non-blocking)
    - Return suggestions for improvement
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Validator identifier."""
        pass

    @abstractmethod
    def validate(self, data: T) -> ValidationResult:
        """
        Validate the input data.

        Args:
            data: The Pydantic model to validate.

        Returns:
            ValidationResult with is_valid status and any errors/warnings.
        """
        pass

    def __call__(self, data: T) -> ValidationResult:
        """Allow validators to be called directly."""
        return self.validate(data)


class ValidatorChain:
    """
    Chain multiple validators together.

    Runs all validators and aggregates results.
    """

    def __init__(self, validators: list[BaseValidator] | None = None):
        self.validators: list[BaseValidator] = validators or []

    def add(self, validator: BaseValidator) -> "ValidatorChain":
        """Add a validator to the chain."""
        self.validators.append(validator)
        return self

    def validate(self, data: Any) -> ValidationResult:
        """
        Run all validators and aggregate results.

        Returns combined result - is_valid is False if ANY validator fails.
        """
        all_errors: list[str] = []
        all_warnings: list[str] = []
        all_suggestions: list[str] = []
        all_metadata: dict[str, Any] = {}

        for validator in self.validators:
            result = validator.validate(data)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            all_suggestions.extend(result.suggestions)
            all_metadata[validator.name] = result.metadata

        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            suggestions=all_suggestions,
            metadata=all_metadata,
        )

    def __call__(self, data: Any) -> ValidationResult:
        """Allow chain to be called directly."""
        return self.validate(data)

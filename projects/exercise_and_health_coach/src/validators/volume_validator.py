from src.config.settings import Settings, get_settings
from src.models.schemas import WorkoutPlan
from src.validators.base import BaseValidator, ValidationResult


class VolumeValidator(BaseValidator[WorkoutPlan]):
    """
    Validates exercise volume per muscle group.

    Checks that weekly volume falls within configured thresholds:
    - Minimum sets per muscle group (default: 18)
    - Maximum sets per muscle group (default: 24)

    Under-volume generates warnings, over-volume generates errors.
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self.min_sets = self._settings.exercise.min_sets_per_muscle_group
        self.max_sets = self._settings.exercise.max_sets_per_muscle_group

    @property
    def name(self) -> str:
        return "volume_validator"

    def validate(self, data: WorkoutPlan) -> ValidationResult:
        """
        Validate volume per muscle group.

        Args:
            data: WorkoutPlan to validate.

        Returns:
            ValidationResult with volume analysis.
        """
        errors: list[str] = []
        warnings: list[str] = []
        suggestions: list[str] = []

        # Get sets by muscle group
        muscle_sets = data.get_sets_by_muscle()

        # Track volume analysis
        under_volume: dict[str, int] = {}
        over_volume: dict[str, int] = {}
        optimal_volume: dict[str, int] = {}

        for muscle, sets in muscle_sets.items():
            muscle_name = muscle.value

            if sets < self.min_sets:
                under_volume[muscle_name] = sets
                warnings.append(
                    f"{muscle_name.title()}: {sets} sets is below minimum ({self.min_sets}). "
                    f"Consider adding {self.min_sets - sets} more sets."
                )
            elif sets > self.max_sets:
                over_volume[muscle_name] = sets
                errors.append(
                    f"{muscle_name.title()}: {sets} sets exceeds maximum ({self.max_sets}). "
                    f"Reduce by {sets - self.max_sets} sets to prevent overtraining."
                )
            else:
                optimal_volume[muscle_name] = sets

        # Provide suggestions for under-volume muscles
        if under_volume:
            suggestions.append(
                f"Consider adding exercises targeting: {', '.join(under_volume.keys())}"
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            metadata={
                "total_sets": data.get_total_sets(),
                "sets_by_muscle": {m.value: s for m, s in muscle_sets.items()},
                "under_volume": under_volume,
                "over_volume": over_volume,
                "optimal_volume": optimal_volume,
                "thresholds": {"min": self.min_sets, "max": self.max_sets},
            },
        )


class SingleSessionVolumeValidator(BaseValidator[WorkoutPlan]):
    """
    Validates volume for a single training session.

    Uses adjusted thresholds appropriate for per-session (not weekly) volume.
    Typically 4-8 sets per muscle group per session.
    """

    def __init__(
        self,
        min_sets_per_session: int = 4,
        max_sets_per_session: int = 8,
    ):
        self.min_sets = min_sets_per_session
        self.max_sets = max_sets_per_session

    @property
    def name(self) -> str:
        return "single_session_volume_validator"

    def validate(self, data: WorkoutPlan) -> ValidationResult:
        """Validate per-session volume."""
        errors: list[str] = []
        warnings: list[str] = []

        muscle_sets = data.get_sets_by_muscle()

        for muscle, sets in muscle_sets.items():
            if sets > self.max_sets:
                warnings.append(
                    f"{muscle.value.title()}: {sets} sets in one session may be excessive. "
                    f"Consider spreading across multiple days."
                )

        # Check total volume
        total = data.get_total_sets()
        if total > 30:
            warnings.append(
                f"Total volume ({total} sets) is high for a single session. "
                f"Consider splitting into multiple workouts."
            )

        return ValidationResult(
            is_valid=True,  # Session volume is advisory, not blocking
            errors=errors,
            warnings=warnings,
            metadata={
                "total_sets": total,
                "sets_by_muscle": {m.value: s for m, s in muscle_sets.items()},
            },
        )

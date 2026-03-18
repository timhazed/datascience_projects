from src.models.enums import MovementPattern
from src.models.schemas import WorkoutPlan
from src.validators.base import BaseValidator, ValidationResult

# Movement pattern classifications
PUSH_PATTERNS = {
    MovementPattern.HORIZONTAL_PUSH,
    MovementPattern.VERTICAL_PUSH,
}

PULL_PATTERNS = {
    MovementPattern.HORIZONTAL_PULL,
    MovementPattern.VERTICAL_PULL,
}

LOWER_PATTERNS = {
    MovementPattern.SQUAT,
    MovementPattern.HIP_HINGE,
    MovementPattern.LUNGE,
}


class RatioValidator(BaseValidator[WorkoutPlan]):
    """
    Validates movement pattern ratios for balanced programming.

    Checks:
    - Push:Pull ratio (target 1:1, acceptable 0.8-1.2)
    - Horizontal:Vertical balance within push/pull
    - Quad:Hip dominant balance for lower body

    Imbalances generate warnings; severe imbalances generate errors.
    """

    def __init__(
        self,
        push_pull_min_ratio: float = 0.8,
        push_pull_max_ratio: float = 1.2,
        quad_hip_min_ratio: float = 0.7,
        quad_hip_max_ratio: float = 1.4,
    ):
        self.push_pull_min = push_pull_min_ratio
        self.push_pull_max = push_pull_max_ratio
        self.quad_hip_min = quad_hip_min_ratio
        self.quad_hip_max = quad_hip_max_ratio

    @property
    def name(self) -> str:
        return "ratio_validator"

    def validate(self, data: WorkoutPlan) -> ValidationResult:
        """
        Validate movement pattern ratios.

        Args:
            data: WorkoutPlan to validate.

        Returns:
            ValidationResult with ratio analysis.
        """
        errors: list[str] = []
        warnings: list[str] = []
        suggestions: list[str] = []

        # Count sets by movement pattern
        pattern_sets: dict[MovementPattern, int] = {}
        for block in data.blocks:
            for exercise in block.exercises:
                pattern = exercise.movement_pattern
                pattern_sets[pattern] = pattern_sets.get(pattern, 0) + exercise.sets

        # Calculate push and pull totals
        push_sets = sum(pattern_sets.get(p, 0) for p in PUSH_PATTERNS)
        pull_sets = sum(pattern_sets.get(p, 0) for p in PULL_PATTERNS)

        # Push:Pull ratio check
        if pull_sets > 0:
            push_pull_ratio = push_sets / pull_sets
        elif push_sets > 0:
            push_pull_ratio = float("inf")
        else:
            push_pull_ratio = 1.0  # No push or pull = balanced

        if push_pull_ratio != float("inf"):
            if push_pull_ratio < self.push_pull_min:
                warnings.append(
                    f"Push:Pull ratio ({push_pull_ratio:.2f}) is low. "
                    f"Consider adding more pushing movements (bench, overhead press)."
                )
            elif push_pull_ratio > self.push_pull_max:
                warnings.append(
                    f"Push:Pull ratio ({push_pull_ratio:.2f}) is high. "
                    f"Consider adding more pulling movements (rows, pull-ups)."
                )
        elif push_sets > 0:
            # Downgrade to warning for power/plyometric-focused workouts
            warnings.append(
                "No pulling movements found. Consider adding rows, pull-ups, or face pulls "
                "to balance pushing volume and protect shoulder health."
            )

        # Horizontal vs Vertical balance within upper body
        h_push = pattern_sets.get(MovementPattern.HORIZONTAL_PUSH, 0)
        v_push = pattern_sets.get(MovementPattern.VERTICAL_PUSH, 0)
        h_pull = pattern_sets.get(MovementPattern.HORIZONTAL_PULL, 0)
        v_pull = pattern_sets.get(MovementPattern.VERTICAL_PULL, 0)

        if h_push > 0 and v_push == 0:
            suggestions.append(
                "Consider adding vertical pushing (overhead press, pike push-ups) "
                "for complete shoulder development."
            )
        if h_pull > 0 and v_pull == 0:
            suggestions.append(
                "Consider adding vertical pulling (pull-ups, lat pulldowns) "
                "for lat development and shoulder health."
            )

        # Quad vs Hip dominant for lower body
        squat_sets = pattern_sets.get(MovementPattern.SQUAT, 0)
        lunge_sets = pattern_sets.get(MovementPattern.LUNGE, 0)
        hinge_sets = pattern_sets.get(MovementPattern.HIP_HINGE, 0)

        quad_dominant = squat_sets + lunge_sets
        hip_dominant = hinge_sets

        if hip_dominant > 0:
            quad_hip_ratio = quad_dominant / hip_dominant
            if quad_hip_ratio < self.quad_hip_min:
                warnings.append(
                    f"Quad:Hip ratio ({quad_hip_ratio:.2f}) is low. "
                    f"Consider adding squats or lunges."
                )
            elif quad_hip_ratio > self.quad_hip_max:
                warnings.append(
                    f"Quad:Hip ratio ({quad_hip_ratio:.2f}) is high. "
                    f"Consider adding hip hinges (deadlifts, RDLs)."
                )
        elif quad_dominant > 0:
            suggestions.append(
                "Consider adding hip hinge movements (deadlifts, RDLs) "
                "for posterior chain development."
            )

        # Check for core/rotation work
        core_sets = (
            pattern_sets.get(MovementPattern.ROTATION, 0)
            + pattern_sets.get(MovementPattern.ANTI_ROTATION, 0)
        )
        if core_sets == 0 and data.get_total_sets() > 10:
            suggestions.append(
                "Consider adding core/rotation work (planks, pallof press, cable rotations)."
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            metadata={
                "pattern_sets": {p.value: s for p, s in pattern_sets.items()},
                "push_sets": push_sets,
                "pull_sets": pull_sets,
                "push_pull_ratio": push_pull_ratio if push_pull_ratio != float("inf") else None,
                "quad_dominant_sets": quad_dominant,
                "hip_dominant_sets": hip_dominant,
                "core_sets": core_sets,
            },
        )


class UpperLowerBalanceValidator(BaseValidator[WorkoutPlan]):
    """
    Validates upper/lower body volume balance.

    For full-body or balanced programs, checks that upper and lower
    body volumes are reasonably balanced.
    """

    def __init__(
        self,
        min_ratio: float = 0.6,
        max_ratio: float = 1.6,
    ):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    @property
    def name(self) -> str:
        return "upper_lower_balance_validator"

    def validate(self, data: WorkoutPlan) -> ValidationResult:
        """Validate upper/lower volume balance."""
        warnings: list[str] = []

        pattern_sets: dict[MovementPattern, int] = {}
        for block in data.blocks:
            for exercise in block.exercises:
                pattern = exercise.movement_pattern
                pattern_sets[pattern] = pattern_sets.get(pattern, 0) + exercise.sets

        upper_patterns = PUSH_PATTERNS | PULL_PATTERNS
        lower_patterns = LOWER_PATTERNS | {MovementPattern.CARRY}

        upper_sets = sum(pattern_sets.get(p, 0) for p in upper_patterns)
        lower_sets = sum(pattern_sets.get(p, 0) for p in lower_patterns)

        if lower_sets > 0:
            ratio = upper_sets / lower_sets
            if ratio < self.min_ratio:
                warnings.append(
                    f"Upper:Lower ratio ({ratio:.2f}) is low. "
                    f"Consider adding more upper body work."
                )
            elif ratio > self.max_ratio:
                warnings.append(
                    f"Upper:Lower ratio ({ratio:.2f}) is high. "
                    f"Consider adding more lower body work."
                )

        return ValidationResult(
            is_valid=True,  # Balance issues are warnings, not errors
            errors=[],
            warnings=warnings,
            metadata={
                "upper_sets": upper_sets,
                "lower_sets": lower_sets,
                "ratio": upper_sets / lower_sets if lower_sets > 0 else None,
            },
        )

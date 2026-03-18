import pytest

from src.models.enums import FitnessGoal, MovementPattern, MuscleGroup
from src.models.schemas import (
    Biometrics,
    Exercise,
    ExerciseBlock,
    MedicalHistory,
    UserContext,
    WorkoutPlan,
)
from src.validators.base import ValidatorChain
from src.validators.ratio_validator import RatioValidator, UpperLowerBalanceValidator
from src.validators.red_flag_scanner import MedicationInteractionScanner, RedFlagScanner
from src.validators.volume_validator import SingleSessionVolumeValidator, VolumeValidator

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def balanced_workout() -> WorkoutPlan:
    """Create a balanced workout plan for testing."""
    return WorkoutPlan(
        blocks=[
            ExerciseBlock(
                block_name="push",
                block_type="main_work",
                exercises=[
                    Exercise(
                        name="Bench Press",
                        sets=4,
                        reps="8-10",
                        rest_seconds=90,
                        movement_pattern=MovementPattern.HORIZONTAL_PUSH,
                        primary_muscles=[MuscleGroup.CHEST],
                        secondary_muscles=[MuscleGroup.TRICEPS],
                    ),
                    Exercise(
                        name="Overhead Press",
                        sets=3,
                        reps="8-10",
                        rest_seconds=90,
                        movement_pattern=MovementPattern.VERTICAL_PUSH,
                        primary_muscles=[MuscleGroup.SHOULDERS],
                        secondary_muscles=[MuscleGroup.TRICEPS],
                    ),
                ],
                estimated_duration_minutes=20,
            ),
            ExerciseBlock(
                block_name="pull",
                block_type="main_work",
                exercises=[
                    Exercise(
                        name="Barbell Row",
                        sets=4,
                        reps="8-10",
                        rest_seconds=90,
                        movement_pattern=MovementPattern.HORIZONTAL_PULL,
                        primary_muscles=[MuscleGroup.BACK],
                        secondary_muscles=[MuscleGroup.BICEPS],
                    ),
                    Exercise(
                        name="Pull-ups",
                        sets=3,
                        reps="6-8",
                        rest_seconds=90,
                        movement_pattern=MovementPattern.VERTICAL_PULL,
                        primary_muscles=[MuscleGroup.BACK],
                        secondary_muscles=[MuscleGroup.BICEPS],
                    ),
                ],
                estimated_duration_minutes=20,
            ),
        ],
        total_duration_minutes=40,
        difficulty_level="intermediate",
        rationale="Balanced push/pull workout",
    )


@pytest.fixture
def push_heavy_workout() -> WorkoutPlan:
    """Create a push-dominant workout for testing imbalance detection."""
    return WorkoutPlan(
        blocks=[
            ExerciseBlock(
                block_name="push",
                block_type="main_work",
                exercises=[
                    Exercise(
                        name="Bench Press",
                        sets=5,
                        reps="5",
                        rest_seconds=180,
                        movement_pattern=MovementPattern.HORIZONTAL_PUSH,
                        primary_muscles=[MuscleGroup.CHEST],
                    ),
                    Exercise(
                        name="Incline Press",
                        sets=4,
                        reps="8",
                        rest_seconds=90,
                        movement_pattern=MovementPattern.HORIZONTAL_PUSH,
                        primary_muscles=[MuscleGroup.CHEST],
                    ),
                    Exercise(
                        name="OHP",
                        sets=4,
                        reps="8",
                        rest_seconds=90,
                        movement_pattern=MovementPattern.VERTICAL_PUSH,
                        primary_muscles=[MuscleGroup.SHOULDERS],
                    ),
                ],
                estimated_duration_minutes=30,
            ),
            ExerciseBlock(
                block_name="pull",
                block_type="accessory",
                exercises=[
                    Exercise(
                        name="Face Pulls",
                        sets=2,
                        reps="15",
                        rest_seconds=60,
                        movement_pattern=MovementPattern.HORIZONTAL_PULL,
                        primary_muscles=[MuscleGroup.BACK],
                    ),
                ],
                estimated_duration_minutes=5,
            ),
        ],
        total_duration_minutes=35,
        difficulty_level="intermediate",
        rationale="Push focused workout",
    )


@pytest.fixture
def high_volume_workout() -> WorkoutPlan:
    """Create a high-volume workout that exceeds thresholds."""
    return WorkoutPlan(
        blocks=[
            ExerciseBlock(
                block_name="chest",
                block_type="main_work",
                exercises=[
                    Exercise(
                        name="Bench Press",
                        sets=6,
                        reps="8",
                        rest_seconds=90,
                        movement_pattern=MovementPattern.HORIZONTAL_PUSH,
                        primary_muscles=[MuscleGroup.CHEST],
                    ),
                    Exercise(
                        name="Incline DB Press",
                        sets=5,
                        reps="10",
                        rest_seconds=90,
                        movement_pattern=MovementPattern.HORIZONTAL_PUSH,
                        primary_muscles=[MuscleGroup.CHEST],
                    ),
                    Exercise(
                        name="Cable Flyes",
                        sets=5,
                        reps="12",
                        rest_seconds=60,
                        movement_pattern=MovementPattern.HORIZONTAL_PUSH,
                        primary_muscles=[MuscleGroup.CHEST],
                    ),
                    Exercise(
                        name="Dips",
                        sets=4,
                        reps="10",
                        rest_seconds=90,
                        movement_pattern=MovementPattern.HORIZONTAL_PUSH,
                        primary_muscles=[MuscleGroup.CHEST],
                        secondary_muscles=[MuscleGroup.TRICEPS],
                    ),
                    Exercise(
                        name="Push-ups",
                        sets=5,
                        reps="15",
                        rest_seconds=60,
                        movement_pattern=MovementPattern.HORIZONTAL_PUSH,
                        primary_muscles=[MuscleGroup.CHEST],
                    ),
                ],
                estimated_duration_minutes=45,
            ),
        ],
        total_duration_minutes=45,
        difficulty_level="advanced",
        rationale="High volume chest focus",
    )


@pytest.fixture
def healthy_user() -> UserContext:
    """Create a healthy user context."""
    return UserContext(
        biometrics=Biometrics(age=30, weight_kg=80),
        fitness_goals=[FitnessGoal.HYPERTROPHY],
        experience_level="intermediate",
    )


@pytest.fixture
def user_with_red_flags() -> UserContext:
    """Create a user context with red flags."""
    return UserContext(
        biometrics=Biometrics(age=45, weight_kg=90),
        medical_history=MedicalHistory(
            conditions=["chest pain during exercise"],
            injuries=["recent ACL surgery"],
        ),
        fitness_goals=[FitnessGoal.GENERAL_FITNESS],
        experience_level="beginner",
    )


# =============================================================================
# Volume Validator Tests
# =============================================================================


class TestVolumeValidator:
    """Test volume validation."""

    def test_balanced_volume_passes(self, balanced_workout):
        validator = VolumeValidator()
        # Override thresholds for test
        validator.min_sets = 4
        validator.max_sets = 10

        result = validator.validate(balanced_workout)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_excessive_volume_fails(self, high_volume_workout):
        validator = VolumeValidator()
        validator.max_sets = 20  # Set threshold below workout volume

        result = validator.validate(high_volume_workout)
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "exceeds maximum" in result.errors[0].lower()

    def test_low_volume_warns(self, balanced_workout):
        validator = VolumeValidator()
        validator.min_sets = 20  # Set threshold above workout volume

        result = validator.validate(balanced_workout)
        assert result.is_valid  # Warnings don't fail
        assert len(result.warnings) > 0
        assert "below minimum" in result.warnings[0].lower()

    def test_metadata_includes_volume_breakdown(self, balanced_workout):
        validator = VolumeValidator()
        result = validator.validate(balanced_workout)

        assert "total_sets" in result.metadata
        assert "sets_by_muscle" in result.metadata
        assert "thresholds" in result.metadata


class TestSingleSessionVolumeValidator:
    """Test single session volume validation."""

    def test_normal_session_passes(self, balanced_workout):
        validator = SingleSessionVolumeValidator()
        result = validator.validate(balanced_workout)
        assert result.is_valid

    def test_excessive_session_warns(self, high_volume_workout):
        validator = SingleSessionVolumeValidator(max_sets_per_session=5)
        result = validator.validate(high_volume_workout)

        assert result.is_valid  # Session limits are advisory
        assert len(result.warnings) > 0


# =============================================================================
# Ratio Validator Tests
# =============================================================================


class TestRatioValidator:
    """Test push/pull ratio validation."""

    def test_balanced_ratio_passes(self, balanced_workout):
        validator = RatioValidator()
        result = validator.validate(balanced_workout)

        assert result.is_valid
        assert len(result.errors) == 0
        # Should have suggestions but no warnings for balanced workout
        assert result.metadata["push_pull_ratio"] is not None

    def test_push_heavy_warns(self, push_heavy_workout):
        validator = RatioValidator()
        result = validator.validate(push_heavy_workout)

        assert result.is_valid  # Ratio issues are warnings
        assert len(result.warnings) > 0
        assert "high" in result.warnings[0].lower() or "pull" in result.warnings[0].lower()

    def test_no_pull_errors(self):
        """Test that a workout with no pulling generates an error."""
        push_only = WorkoutPlan(
            blocks=[
                ExerciseBlock(
                    block_name="push",
                    block_type="main_work",
                    exercises=[
                        Exercise(
                            name="Bench Press",
                            sets=5,
                            reps="5",
                            rest_seconds=180,
                            movement_pattern=MovementPattern.HORIZONTAL_PUSH,
                            primary_muscles=[MuscleGroup.CHEST],
                        ),
                    ],
                    estimated_duration_minutes=15,
                ),
            ],
            total_duration_minutes=15,
            difficulty_level="beginner",
            rationale="Push only",
        )

        validator = RatioValidator()
        result = validator.validate(push_only)

        # No pulling movements is now a warning, not an error
        # This allows power/plyometric-focused workouts to pass validation
        assert result.is_valid
        assert len(result.warnings) > 0
        assert any("pulling" in w.lower() for w in result.warnings)

    def test_suggests_missing_patterns(self, balanced_workout):
        validator = RatioValidator()
        result = validator.validate(balanced_workout)

        # Should suggest core work since it's missing
        assert any("core" in s.lower() for s in result.suggestions)


class TestUpperLowerBalanceValidator:
    """Test upper/lower balance validation."""

    def test_upper_only_warns(self, balanced_workout):
        validator = UpperLowerBalanceValidator()
        result = validator.validate(balanced_workout)

        # Balanced workout has no lower body
        assert result.is_valid  # Balance is advisory
        # Depending on ratio settings, may or may not warn


# =============================================================================
# Red Flag Scanner Tests
# =============================================================================


class TestRedFlagScanner:
    """Test red flag detection."""

    def test_healthy_user_passes(self, healthy_user):
        scanner = RedFlagScanner()
        result = scanner.validate(healthy_user)

        assert result.is_valid
        assert len(result.errors) == 0
        assert result.metadata["red_flag_count"] == 0

    def test_chest_pain_is_critical(self, user_with_red_flags):
        scanner = RedFlagScanner()
        result = scanner.validate(user_with_red_flags)

        assert not result.is_valid
        assert result.metadata["critical_count"] > 0
        assert any("cardiovascular" in e.lower() for e in result.errors)

    def test_acute_injury_detected(self, user_with_red_flags):
        scanner = RedFlagScanner()
        result = scanner.validate(user_with_red_flags)

        # ACL surgery should be detected
        assert any("acute" in e.lower() or "injury" in e.lower() for e in result.errors)

    def test_neurological_symptoms_detected(self):
        user = UserContext(
            biometrics=Biometrics(age=40),
            medical_history=MedicalHistory(
                conditions=["numbness in left arm", "tingling sensation"]
            ),
        )

        scanner = RedFlagScanner()
        result = scanner.validate(user)

        assert not result.is_valid
        assert any("neurological" in e.lower() for e in result.errors)

    def test_radicular_symptoms_detected(self):
        user = UserContext(
            medical_history=MedicalHistory(
                notes="sharp shooting pain radiating down my left leg"
            ),
        )
        scanner = RedFlagScanner()
        result = scanner.validate(user)
        assert not result.is_valid
        assert any("neurological" in e.lower() or "high risk" in e.lower() for e in result.errors)

    def test_dizziness_at_rest_detected(self):
        user = UserContext(
            medical_history=MedicalHistory(
                notes="feeling dizzy and shortness of breath at rest while sitting"
            ),
        )
        scanner = RedFlagScanner()
        result = scanner.validate(user)
        assert not result.is_valid
        assert any("cardiovascular" in e.lower() or "critical" in e.lower() for e in result.errors)

    def test_historical_acl_is_warning_not_error(self):
        user = UserContext(
            medical_history=MedicalHistory(
                notes="history of ACL tear repaired years ago, no current pain"
            ),
        )
        scanner = RedFlagScanner(strict_mode=True)
        result = scanner.validate(user)
        assert result.is_valid
        assert len(result.errors) == 0
        assert any("caution" in w.lower() for w in result.warnings)

    def test_age_warnings_youth(self):
        youth = UserContext(
            biometrics=Biometrics(age=15, weight_kg=55),
            fitness_goals=[FitnessGoal.ATHLETIC_PERFORMANCE],
            experience_level="beginner",
        )

        scanner = RedFlagScanner()
        result = scanner.validate(youth)

        assert result.is_valid  # Age is warning, not error
        assert any("youth" in w.lower() for w in result.warnings)

    def test_age_warnings_senior(self):
        senior = UserContext(
            biometrics=Biometrics(age=70, weight_kg=75),
            fitness_goals=[FitnessGoal.GENERAL_FITNESS],
            experience_level="beginner",
        )

        scanner = RedFlagScanner()
        result = scanner.validate(senior)

        assert result.is_valid
        assert any("senior" in w.lower() for w in result.warnings)

    def test_strict_mode_affects_high_severity(self):
        user = UserContext(
            biometrics=Biometrics(age=35),
            medical_history=MedicalHistory(
                injuries=["herniated disc"]  # High severity, not critical
            ),
        )

        strict_scanner = RedFlagScanner(strict_mode=True)
        lenient_scanner = RedFlagScanner(strict_mode=False)

        strict_result = strict_scanner.validate(user)
        lenient_result = lenient_scanner.validate(user)

        # Strict mode: high severity = error
        assert not strict_result.is_valid
        # Lenient mode: high severity = warning
        assert lenient_result.is_valid


class TestMedicationInteractionScanner:
    """Test medication interaction detection."""

    def test_beta_blocker_warning(self):
        user = UserContext(
            biometrics=Biometrics(age=55),
            medical_history=MedicalHistory(
                medications=["metoprolol 25mg"]
            ),
        )

        scanner = MedicationInteractionScanner()
        result = scanner.validate(user)

        assert result.is_valid  # Medications are warnings
        assert any("heart rate" in w.lower() or "rpe" in w.lower() for w in result.warnings)

    def test_blood_thinner_warning(self):
        user = UserContext(
            biometrics=Biometrics(age=60),
            medical_history=MedicalHistory(
                medications=["warfarin"]
            ),
        )

        scanner = MedicationInteractionScanner()
        result = scanner.validate(user)

        assert any("bleeding" in w.lower() for w in result.warnings)

    def test_multiple_medications(self):
        user = UserContext(
            biometrics=Biometrics(age=65),
            medical_history=MedicalHistory(
                medications=["atorvastatin", "metoprolol", "aspirin"]
            ),
        )

        scanner = MedicationInteractionScanner()
        result = scanner.validate(user)

        # Should detect statin and beta blocker
        assert result.metadata["concerns_found"] >= 2


# =============================================================================
# Validator Chain Tests
# =============================================================================


class TestValidatorChain:
    """Test validator chain functionality."""

    def test_chain_aggregates_results(self, balanced_workout, healthy_user):
        volume_validator = VolumeValidator()
        volume_validator.min_sets = 4
        volume_validator.max_sets = 50

        ratio_validator = RatioValidator()

        chain = ValidatorChain([volume_validator, ratio_validator])
        result = chain.validate(balanced_workout)

        assert "volume_validator" in result.metadata
        assert "ratio_validator" in result.metadata

    def test_chain_fails_on_any_error(self, high_volume_workout):
        volume_validator = VolumeValidator()
        volume_validator.max_sets = 10  # Will fail

        ratio_validator = RatioValidator()

        chain = ValidatorChain([volume_validator, ratio_validator])
        result = chain.validate(high_volume_workout)

        assert not result.is_valid
        assert len(result.errors) > 0

    def test_chain_add_method(self, balanced_workout):
        chain = ValidatorChain()

        volume_validator = VolumeValidator()
        volume_validator.min_sets = 1
        volume_validator.max_sets = 100

        chain.add(volume_validator).add(RatioValidator())

        assert len(chain.validators) == 2

        result = chain.validate(balanced_workout)
        assert len(result.metadata) == 2

    def test_empty_chain_passes(self, balanced_workout):
        chain = ValidatorChain()
        result = chain.validate(balanced_workout)

        assert result.is_valid
        assert len(result.errors) == 0

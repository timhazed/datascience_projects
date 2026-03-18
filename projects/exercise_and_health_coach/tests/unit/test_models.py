import pytest
from pydantic import ValidationError

from src.models.enums import (
    AuditStatus,
    Equipment,
    FitnessGoal,
    IntentType,
    MovementPattern,
    MuscleGroup,
    RecoveryModality,
    RedFlagCategory,
)
from src.models.schemas import (
    AuditLog,
    Biometrics,
    CoachOutput,
    Exercise,
    ExerciseBlock,
    MedicalHistory,
    RecoveryBlock,
    RecoveryExercise,
    RecoveryPlan,
    RedFlag,
    UserContext,
)


class TestEnums:
    """Test enum definitions and values."""

    def test_fitness_goal_values(self):
        assert FitnessGoal.HYPERTROPHY.value == "hypertrophy"
        assert FitnessGoal.STRENGTH.value == "strength"
        assert len(FitnessGoal) == 7

    def test_equipment_values(self):
        assert Equipment.BARBELL.value == "barbell"
        assert Equipment.FOAM_ROLLER.value == "foam_roller"
        assert len(Equipment) == 11

    def test_movement_pattern_values(self):
        assert MovementPattern.HORIZONTAL_PUSH.value == "horizontal_push"
        assert MovementPattern.HIP_HINGE.value == "hip_hinge"
        assert len(MovementPattern) == 11

    def test_muscle_group_values(self):
        assert MuscleGroup.CHEST.value == "chest"
        assert MuscleGroup.QUADRICEPS.value == "quadriceps"
        assert len(MuscleGroup) == 11

    def test_recovery_modality_order(self):
        modalities = list(RecoveryModality)
        assert modalities[0] == RecoveryModality.SMR
        assert modalities[1] == RecoveryModality.DYNAMIC_MOBILITY
        assert modalities[2] == RecoveryModality.STATIC_STRETCHING

    def test_audit_status_values(self):
        assert AuditStatus.APPROVED.value == "approved"
        assert AuditStatus.REJECTED.value == "rejected"
        assert len(AuditStatus) == 3

    def test_red_flag_category_values(self):
        assert RedFlagCategory.NEUROLOGICAL.value == "neurological"
        assert RedFlagCategory.CARDIOVASCULAR.value == "cardiovascular"
        assert len(RedFlagCategory) == 5

    def test_intent_type_values(self):
        assert IntentType.EXERCISE_REQUEST.value == "exercise_request"
        assert IntentType.RECOVERY_REQUEST.value == "recovery_request"
        assert len(IntentType) == 6


class TestBiometrics:
    """Test Biometrics model validation."""

    def test_valid_biometrics(self):
        bio = Biometrics(age=30, weight_kg=75.5, height_cm=180.0, sex="male")
        assert bio.age == 30
        assert bio.weight_kg == 75.5
        assert bio.is_complete()

    def test_empty_biometrics(self):
        bio = Biometrics()
        assert bio.age is None
        assert not bio.is_complete()

    def test_partial_biometrics(self):
        bio = Biometrics(age=25)
        assert not bio.is_complete()  # Missing weight

    def test_age_validation_min(self):
        with pytest.raises(ValidationError):
            Biometrics(age=12)  # Below 13

    def test_age_validation_max(self):
        with pytest.raises(ValidationError):
            Biometrics(age=101)  # Above 100

    def test_weight_validation(self):
        with pytest.raises(ValidationError):
            Biometrics(weight_kg=-5)

    def test_sex_validation(self):
        with pytest.raises(ValidationError):
            Biometrics(sex="invalid")


class TestMedicalHistory:
    """Test MedicalHistory model."""

    def test_empty_history(self):
        history = MedicalHistory()
        assert history.conditions == []
        assert not history.has_red_flags()

    def test_red_flag_detection(self):
        history = MedicalHistory(conditions=["chest pain when exercising"])
        assert history.has_red_flags()

    def test_red_flag_in_injuries(self):
        history = MedicalHistory(injuries=["acute shoulder tear"])
        assert history.has_red_flags()

    def test_red_flag_in_notes(self):
        history = MedicalHistory(notes="Patient has uncontrolled hypertension")
        assert history.has_red_flags()

    def test_no_red_flags(self):
        history = MedicalHistory(
            conditions=["mild asthma"],
            injuries=["old ankle sprain"],
        )
        assert not history.has_red_flags()


class TestUserContext:
    """Test UserContext model and validation."""

    def test_empty_context(self):
        ctx = UserContext()
        missing = ctx.get_missing_required_fields()
        assert "age" in missing
        assert "weight" in missing
        assert "fitness_goals" in missing
        assert "experience_level" in missing

    def test_complete_context(self):
        ctx = UserContext(
            biometrics=Biometrics(age=30, weight_kg=80),
            fitness_goals=[FitnessGoal.HYPERTROPHY],
            experience_level="intermediate",
            available_equipment=[Equipment.BODYWEIGHT],
        )
        assert ctx.is_complete_for_exercise()
        assert ctx.get_missing_required_fields() == []

    def test_recovery_requires_less(self):
        ctx = UserContext(biometrics=Biometrics(age=30))
        assert ctx.is_complete_for_recovery()

    def test_recovery_with_pain_areas(self):
        ctx = UserContext(pain_areas=["lower back"])
        assert ctx.is_complete_for_recovery()


class TestExercise:
    """Test Exercise model validation."""

    def test_valid_exercise(self):
        ex = Exercise(
            name="Bench Press",
            sets=4,
            reps="8-12",
            rest_seconds=90,
            rpe_target=8,
            movement_pattern=MovementPattern.HORIZONTAL_PUSH,
            primary_muscles=[MuscleGroup.CHEST],
            secondary_muscles=[MuscleGroup.TRICEPS, MuscleGroup.SHOULDERS],
            equipment=[Equipment.BARBELL],
        )
        assert ex.name == "Bench Press"
        assert ex.sets == 4

    def test_sets_validation_min(self):
        with pytest.raises(ValidationError):
            Exercise(
                name="Test",
                sets=0,
                reps="10",
                rest_seconds=60,
                movement_pattern=MovementPattern.SQUAT,
                primary_muscles=[MuscleGroup.QUADRICEPS],
            )

    def test_sets_validation_max(self):
        with pytest.raises(ValidationError):
            Exercise(
                name="Test",
                sets=11,
                reps="10",
                rest_seconds=60,
                movement_pattern=MovementPattern.SQUAT,
                primary_muscles=[MuscleGroup.QUADRICEPS],
            )

    def test_primary_muscles_required(self):
        with pytest.raises(ValidationError):
            Exercise(
                name="Test",
                sets=3,
                reps="10",
                rest_seconds=60,
                movement_pattern=MovementPattern.SQUAT,
                primary_muscles=[],
            )


class TestExerciseBlock:
    """Test ExerciseBlock model."""

    @pytest.fixture
    def sample_exercises(self):
        return [
            Exercise(
                name="Squat",
                sets=4,
                reps="5",
                rest_seconds=180,
                movement_pattern=MovementPattern.SQUAT,
                primary_muscles=[MuscleGroup.QUADRICEPS, MuscleGroup.GLUTES],
            ),
            Exercise(
                name="Romanian Deadlift",
                sets=3,
                reps="8-10",
                rest_seconds=120,
                movement_pattern=MovementPattern.HIP_HINGE,
                primary_muscles=[MuscleGroup.HAMSTRINGS],
                secondary_muscles=[MuscleGroup.GLUTES],
            ),
        ]

    def test_valid_block(self, sample_exercises):
        block = ExerciseBlock(
            block_name="main_work",
            block_type="main_work",
            exercises=sample_exercises,
            estimated_duration_minutes=30,
        )
        assert block.get_total_sets() == 7

    def test_sets_by_muscle(self, sample_exercises):
        block = ExerciseBlock(
            block_name="main_work",
            block_type="main_work",
            exercises=sample_exercises,
            estimated_duration_minutes=30,
        )
        muscle_sets = block.get_sets_by_muscle()
        assert muscle_sets[MuscleGroup.QUADRICEPS] == 4
        assert muscle_sets[MuscleGroup.HAMSTRINGS] == 3
        # Glutes: 4 primary + 1 secondary (3//2)
        assert muscle_sets[MuscleGroup.GLUTES] == 5

    def test_block_type_validation(self):
        with pytest.raises(ValidationError):
            ExerciseBlock(
                block_name="test",
                block_type="invalid_type",
                exercises=[
                    Exercise(
                        name="Test",
                        sets=3,
                        reps="10",
                        rest_seconds=60,
                        movement_pattern=MovementPattern.SQUAT,
                        primary_muscles=[MuscleGroup.QUADRICEPS],
                    )
                ],
                estimated_duration_minutes=10,
            )


class TestRecoveryModels:
    """Test recovery-related models."""

    def test_recovery_exercise(self):
        ex = RecoveryExercise(
            name="Foam Roll Quads",
            modality=RecoveryModality.SMR,
            duration_seconds=60,
            target_areas=["quadriceps"],
            equipment=[Equipment.FOAM_ROLLER],
        )
        assert ex.intensity == "moderate"

    def test_duration_validation(self):
        with pytest.raises(ValidationError):
            RecoveryExercise(
                name="Test",
                modality=RecoveryModality.STATIC_STRETCHING,
                duration_seconds=5,  # Below 10
                target_areas=["hamstrings"],
            )

    def test_recovery_block_modality_consistency(self):
        with pytest.raises(ValidationError):
            RecoveryBlock(
                modality=RecoveryModality.SMR,
                exercises=[
                    RecoveryExercise(
                        name="Test",
                        modality=RecoveryModality.STATIC_STRETCHING,  # Mismatch
                        duration_seconds=30,
                        target_areas=["back"],
                    )
                ],
                estimated_duration_minutes=5,
            )

    def test_recovery_plan_modality_order(self):
        # Should fail: static before SMR
        with pytest.raises(ValidationError):
            RecoveryPlan(
                blocks=[
                    RecoveryBlock(
                        modality=RecoveryModality.STATIC_STRETCHING,
                        exercises=[
                            RecoveryExercise(
                                name="Static Stretch",
                                modality=RecoveryModality.STATIC_STRETCHING,
                                duration_seconds=30,
                                target_areas=["hips"],
                            )
                        ],
                        estimated_duration_minutes=5,
                    ),
                    RecoveryBlock(
                        modality=RecoveryModality.SMR,  # Should come first
                        exercises=[
                            RecoveryExercise(
                                name="Foam Roll",
                                modality=RecoveryModality.SMR,
                                duration_seconds=60,
                                target_areas=["quads"],
                            )
                        ],
                        estimated_duration_minutes=5,
                    ),
                ],
                total_duration_minutes=10,
                rationale="Test plan",
            )

    def test_valid_recovery_plan_order(self):
        plan = RecoveryPlan(
            blocks=[
                RecoveryBlock(
                    modality=RecoveryModality.SMR,
                    exercises=[
                        RecoveryExercise(
                            name="Foam Roll",
                            modality=RecoveryModality.SMR,
                            duration_seconds=60,
                            target_areas=["quads"],
                        )
                    ],
                    estimated_duration_minutes=5,
                ),
                RecoveryBlock(
                    modality=RecoveryModality.DYNAMIC_MOBILITY,
                    exercises=[
                        RecoveryExercise(
                            name="Leg Swings",
                            modality=RecoveryModality.DYNAMIC_MOBILITY,
                            duration_seconds=30,
                            target_areas=["hips"],
                        )
                    ],
                    estimated_duration_minutes=3,
                ),
                RecoveryBlock(
                    modality=RecoveryModality.STATIC_STRETCHING,
                    exercises=[
                        RecoveryExercise(
                            name="Hip Flexor Stretch",
                            modality=RecoveryModality.STATIC_STRETCHING,
                            duration_seconds=45,
                            target_areas=["hip flexors"],
                        )
                    ],
                    estimated_duration_minutes=5,
                ),
            ],
            total_duration_minutes=13,
            rationale="Progressive recovery sequence",
        )
        assert len(plan.blocks) == 3


class TestAuditModels:
    """Test audit-related models."""

    def test_red_flag(self):
        flag = RedFlag(
            category=RedFlagCategory.CARDIOVASCULAR,
            description="User reported chest pain during exercise",
            severity="critical",
            source_field="medical_history.conditions",
            recommendation="Refer to physician before exercise program",
        )
        assert flag.severity == "critical"

    def test_audit_log_critical_flags(self):
        audit = AuditLog(
            status=AuditStatus.REJECTED,
            red_flags=[
                RedFlag(
                    category=RedFlagCategory.NEUROLOGICAL,
                    description="Numbness in extremities",
                    severity="critical",
                    recommendation="Medical clearance required",
                )
            ],
            rejection_reason="Critical red flags detected",
        )
        assert audit.has_critical_flags()

    def test_audit_log_no_critical_flags(self):
        audit = AuditLog(
            status=AuditStatus.MODIFIED,
            red_flags=[
                RedFlag(
                    category=RedFlagCategory.ACUTE_INJURY,
                    description="Minor muscle strain",
                    severity="low",
                    recommendation="Avoid direct loading",
                )
            ],
            modifications=["Removed overhead movements"],
        )
        assert not audit.has_critical_flags()


class TestCoachOutput:
    """Test CoachOutput model."""

    def test_approved_output(self):
        output = CoachOutput(
            session_id="test-session",
            audit=AuditLog(status=AuditStatus.APPROVED),
            user_message="Here is your workout plan.",
        )
        assert output.is_approved()

    def test_modified_output_is_approved(self):
        output = CoachOutput(
            session_id="test-session",
            audit=AuditLog(
                status=AuditStatus.MODIFIED,
                modifications=["Reduced volume"],
            ),
            user_message="Here is your modified workout plan.",
        )
        assert output.is_approved()

    def test_rejected_output(self):
        output = CoachOutput(
            session_id="test-session",
            audit=AuditLog(
                status=AuditStatus.REJECTED,
                rejection_reason="Critical red flags",
            ),
            user_message="I cannot provide a workout plan due to safety concerns.",
        )
        assert not output.is_approved()

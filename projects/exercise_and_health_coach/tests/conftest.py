"""Shared pytest fixtures for the Exercise and Health Coach tests."""

from unittest.mock import MagicMock

import pytest

from src.models.enums import (
    AuditStatus,
    Equipment,
    FitnessGoal,
    MovementPattern,
    MuscleGroup,
    RecoveryModality,
)
from src.models.schemas import (
    AuditLog,
    Biometrics,
    Exercise,
    ExerciseBlock,
    IntakeResponse,
    MedicalHistory,
    RecoveryBlock,
    RecoveryExercise,
    RecoveryPlan,
    UserContext,
    WorkoutPlan,
)
from src.state.conversation_state import create_session

# =============================================================================
# Mock LLM Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Create a basic mock LLM."""
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def mock_llm_with_response(mock_llm):
    """Factory fixture for mock LLM with specific response."""
    def _create(response):
        mock_llm.invoke = MagicMock(return_value=response)
        return mock_llm
    return _create


# =============================================================================
# User Context Fixtures
# =============================================================================


@pytest.fixture
def empty_user_context():
    """Empty user context (no information)."""
    return UserContext()


@pytest.fixture
def minimal_user_context():
    """Minimal user context (just age)."""
    return UserContext(
        biometrics=Biometrics(age=30),
    )


@pytest.fixture
def healthy_user_context():
    """Complete healthy user context."""
    return UserContext(
        biometrics=Biometrics(
            age=30,
            weight_kg=80,
            height_cm=180,
            sex="male",
        ),
        fitness_goals=[FitnessGoal.HYPERTROPHY],
        available_equipment=[
            Equipment.BARBELL,
            Equipment.DUMBBELL,
            Equipment.CABLE,
        ],
        experience_level="intermediate",
        training_days_per_week=4,
        session_duration_minutes=60,
    )


@pytest.fixture
def beginner_user_context():
    """Beginner user context."""
    return UserContext(
        biometrics=Biometrics(age=25, weight_kg=70),
        fitness_goals=[FitnessGoal.GENERAL_FITNESS],
        available_equipment=[Equipment.BODYWEIGHT, Equipment.DUMBBELL],
        experience_level="beginner",
        training_days_per_week=3,
    )


@pytest.fixture
def user_with_injuries():
    """User context with injuries."""
    return UserContext(
        biometrics=Biometrics(age=35, weight_kg=85),
        medical_history=MedicalHistory(
            injuries=["old shoulder injury", "mild knee pain"],
        ),
        fitness_goals=[FitnessGoal.GENERAL_FITNESS],
        experience_level="intermediate",
        pain_areas=["right shoulder", "left knee"],
    )


@pytest.fixture
def user_with_red_flags():
    """User context with red flags (should be rejected)."""
    return UserContext(
        biometrics=Biometrics(age=45, weight_kg=90),
        medical_history=MedicalHistory(
            conditions=["chest pain during exercise", "heart palpitations"],
        ),
        fitness_goals=[FitnessGoal.GENERAL_FITNESS],
        experience_level="beginner",
    )


@pytest.fixture
def recovery_user_context():
    """User context for recovery-only requests."""
    return UserContext(
        biometrics=Biometrics(age=28),
        pain_areas=["lower back", "hip flexors"],
        available_equipment=[Equipment.FOAM_ROLLER],
    )


# =============================================================================
# Exercise/Workout Fixtures
# =============================================================================


@pytest.fixture
def sample_exercise():
    """Single sample exercise."""
    return Exercise(
        name="Bench Press",
        sets=4,
        reps="8-10",
        rest_seconds=90,
        rpe_target=8,
        movement_pattern=MovementPattern.HORIZONTAL_PUSH,
        primary_muscles=[MuscleGroup.CHEST],
        secondary_muscles=[MuscleGroup.TRICEPS, MuscleGroup.SHOULDERS],
        equipment=[Equipment.BARBELL],
    )


@pytest.fixture
def sample_exercise_block(sample_exercise):
    """Sample exercise block."""
    return ExerciseBlock(
        block_name="main_work",
        block_type="main_work",
        exercises=[
            sample_exercise,
            Exercise(
                name="Barbell Row",
                sets=4,
                reps="8-10",
                rest_seconds=90,
                movement_pattern=MovementPattern.HORIZONTAL_PULL,
                primary_muscles=[MuscleGroup.BACK],
                secondary_muscles=[MuscleGroup.BICEPS],
            ),
        ],
        estimated_duration_minutes=25,
    )


@pytest.fixture
def sample_workout_plan(sample_exercise_block):
    """Complete sample workout plan."""
    return WorkoutPlan(
        blocks=[sample_exercise_block],
        total_duration_minutes=30,
        difficulty_level="intermediate",
        rationale="Balanced push/pull workout for hypertrophy",
    )


# =============================================================================
# Recovery Fixtures
# =============================================================================


@pytest.fixture
def sample_recovery_exercise():
    """Single sample recovery exercise."""
    return RecoveryExercise(
        name="Foam Roll Quads",
        modality=RecoveryModality.SMR,
        duration_seconds=60,
        target_areas=["quadriceps"],
        equipment=[Equipment.FOAM_ROLLER],
    )


@pytest.fixture
def sample_recovery_block():
    """Sample recovery block."""
    return RecoveryBlock(
        modality=RecoveryModality.SMR,
        exercises=[
            RecoveryExercise(
                name="Foam Roll Quads",
                modality=RecoveryModality.SMR,
                duration_seconds=60,
                target_areas=["quadriceps"],
            ),
            RecoveryExercise(
                name="Foam Roll IT Band",
                modality=RecoveryModality.SMR,
                duration_seconds=60,
                target_areas=["IT band"],
            ),
        ],
        estimated_duration_minutes=5,
    )


@pytest.fixture
def sample_recovery_plan():
    """Complete sample recovery plan."""
    return RecoveryPlan(
        blocks=[
            RecoveryBlock(
                modality=RecoveryModality.SMR,
                exercises=[
                    RecoveryExercise(
                        name="Foam Roll",
                        modality=RecoveryModality.SMR,
                        duration_seconds=60,
                        target_areas=["quads"],
                    ),
                ],
                estimated_duration_minutes=3,
            ),
            RecoveryBlock(
                modality=RecoveryModality.DYNAMIC_MOBILITY,
                exercises=[
                    RecoveryExercise(
                        name="Leg Swings",
                        modality=RecoveryModality.DYNAMIC_MOBILITY,
                        duration_seconds=30,
                        target_areas=["hips"],
                    ),
                ],
                estimated_duration_minutes=2,
            ),
            RecoveryBlock(
                modality=RecoveryModality.STATIC_STRETCHING,
                exercises=[
                    RecoveryExercise(
                        name="Hip Flexor Stretch",
                        modality=RecoveryModality.STATIC_STRETCHING,
                        duration_seconds=45,
                        target_areas=["hip flexors"],
                    ),
                ],
                estimated_duration_minutes=3,
            ),
        ],
        total_duration_minutes=8,
        rationale="Post-workout recovery focusing on lower body",
    )


# =============================================================================
# Audit Fixtures
# =============================================================================


@pytest.fixture
def approved_audit():
    """Approved audit log."""
    return AuditLog(
        status=AuditStatus.APPROVED,
        approval_notes="No safety concerns identified.",
    )


@pytest.fixture
def modified_audit():
    """Modified audit log."""
    return AuditLog(
        status=AuditStatus.MODIFIED,
        modifications=["Reduced overhead pressing volume due to shoulder history"],
        approval_notes="Safe to proceed with modifications.",
    )


@pytest.fixture
def rejected_audit():
    """Rejected audit log."""
    return AuditLog(
        status=AuditStatus.REJECTED,
        rejection_reason="Critical red flags detected requiring medical clearance.",
    )


# =============================================================================
# Intake Response Fixtures
# =============================================================================


@pytest.fixture
def complete_intake_response(healthy_user_context):
    """Complete intake response."""
    return IntakeResponse(
        extracted_context=healthy_user_context,
        clarification_needed=[],
        confidence_score=0.95,
    )


@pytest.fixture
def partial_intake_response():
    """Partial intake response needing more info."""
    return IntakeResponse(
        extracted_context=UserContext(
            biometrics=Biometrics(age=30),
        ),
        clarification_needed=["fitness_goals", "experience_level"],
        confidence_score=0.6,
    )


# =============================================================================
# Conversation State Fixtures
# =============================================================================


@pytest.fixture
def empty_session():
    """Empty conversation session."""
    return create_session()


@pytest.fixture
def session_with_context(healthy_user_context):
    """Session with complete user context."""
    session = create_session()
    session.user_context = healthy_user_context
    session.intake_complete = True
    return session


@pytest.fixture
def session_with_history(healthy_user_context):
    """Session with conversation history."""
    session = create_session()
    session.user_context = healthy_user_context
    session.intake_complete = True
    session.add_turn(
        user_message="I want to build muscle",
        assistant_response="Great! Let me gather some information.",
    )
    session.add_turn(
        user_message="I'm 30 years old, intermediate lifter",
        assistant_response="Perfect, I have what I need.",
    )
    return session

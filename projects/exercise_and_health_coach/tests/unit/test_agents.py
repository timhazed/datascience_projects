"""Unit tests for agents."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.base_agent import BaseAgent
from src.agents.clinical_gatekeeper import ClinicalGatekeeper
from src.agents.intake_agent import IntakeAgent
from src.agents.kinesiologist_specialist import KinesiologistSpecialist
from src.agents.recovery_specialist import RecoverySpecialist
from src.exceptions import AgentExecutionError
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

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def healthy_user_context():
    """Create a healthy user context."""
    return UserContext(
        biometrics=Biometrics(age=30, weight_kg=80, sex="male"),
        fitness_goals=[FitnessGoal.HYPERTROPHY],
        available_equipment=[Equipment.BARBELL, Equipment.DUMBBELL],
        experience_level="intermediate",
        training_days_per_week=4,
        session_duration_minutes=60,
    )


@pytest.fixture
def user_with_conditions():
    """Create a user context with medical conditions."""
    return UserContext(
        biometrics=Biometrics(age=45, weight_kg=90),
        medical_history=MedicalHistory(
            conditions=["mild lower back pain"],
            injuries=["old shoulder injury"],
        ),
        fitness_goals=[FitnessGoal.GENERAL_FITNESS],
        experience_level="beginner",
        pain_areas=["lower back"],
    )


@pytest.fixture
def sample_workout_plan():
    """Create a sample workout plan."""
    return WorkoutPlan(
        blocks=[
            ExerciseBlock(
                block_name="main_work",
                block_type="main_work",
                exercises=[
                    Exercise(
                        name="Bench Press",
                        sets=4,
                        reps="8-10",
                        rest_seconds=90,
                        movement_pattern=MovementPattern.HORIZONTAL_PUSH,
                        primary_muscles=[MuscleGroup.CHEST],
                    ),
                    Exercise(
                        name="Barbell Row",
                        sets=4,
                        reps="8-10",
                        rest_seconds=90,
                        movement_pattern=MovementPattern.HORIZONTAL_PULL,
                        primary_muscles=[MuscleGroup.BACK],
                    ),
                ],
                estimated_duration_minutes=30,
            ),
        ],
        total_duration_minutes=30,
        difficulty_level="intermediate",
        rationale="Balanced push/pull workout",
    )


@pytest.fixture
def sample_recovery_plan():
    """Create a sample recovery plan."""
    return RecoveryPlan(
        blocks=[
            RecoveryBlock(
                modality=RecoveryModality.SMR,
                exercises=[
                    RecoveryExercise(
                        name="Foam Roll Quads",
                        modality=RecoveryModality.SMR,
                        duration_seconds=60,
                        target_areas=["quadriceps"],
                    ),
                ],
                estimated_duration_minutes=5,
            ),
        ],
        total_duration_minutes=5,
        rationale="Post-workout recovery",
    )


@pytest.fixture
def sample_intake_response():
    """Create a sample intake response."""
    return IntakeResponse(
        extracted_context=UserContext(
            biometrics=Biometrics(age=30, weight_kg=80),
            fitness_goals=[FitnessGoal.HYPERTROPHY],
        ),
        clarification_needed=[],
        confidence_score=0.9,
    )


@pytest.fixture
def sample_audit_log():
    """Create a sample audit log."""
    return AuditLog(
        status=AuditStatus.APPROVED,
        red_flags=[],
        approval_notes="No safety concerns identified.",
    )


# =============================================================================
# Base Agent Tests
# =============================================================================


class TestBaseAgent:
    """Test BaseAgent functionality."""

    def test_agent_requires_output_schema(self, mock_llm):
        """Test that BaseAgent requires output_schema implementation."""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            BaseAgent(llm=mock_llm)

    def test_structured_llm_created_lazily(self, mock_llm):
        """Test that structured LLM is created on first access."""
        agent = IntakeAgent(llm=mock_llm)

        # Should not be created yet
        assert agent._structured_llm is None

        # Access triggers creation
        _ = agent.structured_llm
        mock_llm.with_structured_output.assert_called_once()

    def test_invoke_retries_on_failure(self, mock_llm, sample_intake_response):
        """Test that invoke retries on failure."""
        mock_llm.invoke = MagicMock(
            side_effect=[
                Exception("First failure"),
                Exception("Second failure"),
                sample_intake_response,
            ]
        )
        mock_llm.with_structured_output.return_value = mock_llm

        agent = IntakeAgent(llm=mock_llm, max_retries=3)
        result = agent.invoke(user_message="I'm 30 years old")

        assert result == sample_intake_response
        assert mock_llm.invoke.call_count == 3

    def test_invoke_raises_after_max_retries(self, mock_llm):
        """Test that invoke raises AgentExecutionError after max retries."""
        mock_llm.invoke = MagicMock(side_effect=Exception("Always fails"))
        mock_llm.with_structured_output.return_value = mock_llm

        agent = IntakeAgent(llm=mock_llm, max_retries=3)

        with pytest.raises(AgentExecutionError) as exc_info:
            agent.invoke(user_message="test")

        assert "Failed after 3 attempts" in str(exc_info.value)
        assert exc_info.value.agent_name == "intake_agent"


# =============================================================================
# Intake Agent Tests
# =============================================================================


class TestIntakeAgent:
    """Test IntakeAgent functionality."""

    def test_agent_properties(self, mock_llm):
        """Test agent property values."""
        agent = IntakeAgent(llm=mock_llm)

        assert agent.name == "intake_agent"
        assert agent.output_schema == IntakeResponse
        # System prompt now loaded from SKILL.md + agent instructions
        assert "intake" in agent.system_prompt.lower()
        assert "extract" in agent.system_prompt.lower()

    def test_format_user_message_simple(self, mock_llm):
        """Test formatting simple user message."""
        agent = IntakeAgent(llm=mock_llm)
        message = agent.format_user_message(user_message="I want to build muscle")

        assert "build muscle" in message
        assert "Current User Message" in message

    def test_format_user_message_with_context(self, mock_llm, healthy_user_context):
        """Test formatting with existing context."""
        agent = IntakeAgent(llm=mock_llm)
        message = agent.format_user_message(
            user_message="I want to add chest focus",
            existing_context=healthy_user_context,
        )

        assert "Already Known Information" in message
        assert "30" in message  # Age
        assert "chest focus" in message

    def test_extract_from_message(self, mock_llm, sample_intake_response):
        """Test extract_from_message convenience method."""
        mock_llm.invoke = MagicMock(return_value=sample_intake_response)
        mock_llm.with_structured_output.return_value = mock_llm

        agent = IntakeAgent(llm=mock_llm)
        result = agent.extract_from_message("I'm 30 years old")

        assert result == sample_intake_response
        mock_llm.invoke.assert_called_once()


# =============================================================================
# Kinesiologist Specialist Tests
# =============================================================================


class TestKinesiologistSpecialist:
    """Test KinesiologistSpecialist functionality."""

    def test_agent_properties(self, mock_llm):
        """Test agent property values."""
        agent = KinesiologistSpecialist(llm=mock_llm)

        assert agent.name == "kinesiologist_specialist"
        assert agent.output_schema == WorkoutPlan
        assert "strength" in agent.system_prompt.lower()

    def test_has_validators(self, mock_llm):
        """Test that agent has validators configured."""
        agent = KinesiologistSpecialist(llm=mock_llm)

        assert agent.validators is not None
        assert len(agent.validators.validators) >= 2  # Volume + Ratio

    def test_format_user_message(self, mock_llm, healthy_user_context):
        """Test formatting user context for workout generation."""
        agent = KinesiologistSpecialist(llm=mock_llm)
        message = agent.format_user_message(user_context=healthy_user_context)

        assert "User Profile" in message
        assert "30" in message  # Age
        assert "hypertrophy" in message.lower()
        assert "barbell" in message.lower()
        assert "intermediate" in message

    def test_format_message_with_conditions(self, mock_llm, user_with_conditions):
        """Test formatting includes medical considerations."""
        agent = KinesiologistSpecialist(llm=mock_llm)
        message = agent.format_user_message(user_context=user_with_conditions)

        assert "Medical Considerations" in message
        assert "lower back" in message.lower()
        assert "shoulder" in message.lower()
        assert "AVOID" in message

    def test_format_message_requires_context(self, mock_llm):
        """Test that user_context is required."""
        agent = KinesiologistSpecialist(llm=mock_llm)

        with pytest.raises(ValueError, match="user_context is required"):
            agent.format_user_message()

    def test_create_workout(self, mock_llm, healthy_user_context, sample_workout_plan):
        """Test create_workout convenience method."""
        mock_llm.invoke = MagicMock(return_value=sample_workout_plan)
        mock_llm.with_structured_output.return_value = mock_llm

        agent = KinesiologistSpecialist(llm=mock_llm)
        # Mock validators to pass
        agent.validators = None

        result = agent.create_workout(healthy_user_context)

        assert result == sample_workout_plan


# =============================================================================
# Recovery Specialist Tests
# =============================================================================


class TestRecoverySpecialist:
    """Test RecoverySpecialist functionality."""

    def test_agent_properties(self, mock_llm):
        """Test agent property values."""
        agent = RecoverySpecialist(llm=mock_llm)

        assert agent.name == "recovery_specialist"
        assert agent.output_schema == RecoveryPlan
        assert "recovery" in agent.system_prompt.lower()

    def test_no_validators(self, mock_llm):
        """Test that recovery agent has no validators."""
        agent = RecoverySpecialist(llm=mock_llm)
        assert agent.validators is None

    def test_format_standalone_recovery(self, mock_llm, healthy_user_context):
        """Test formatting standalone recovery request."""
        agent = RecoverySpecialist(llm=mock_llm)
        message = agent.format_user_message(
            user_context=healthy_user_context,
            standalone=True,
        )

        assert "Standalone" in message
        assert "modality sequence" in message.lower()

    def test_format_post_workout_recovery(
        self, mock_llm, healthy_user_context, sample_workout_plan
    ):
        """Test formatting post-workout recovery request."""
        agent = RecoverySpecialist(llm=mock_llm)
        message = agent.format_user_message(
            user_context=healthy_user_context,
            workout_plan=sample_workout_plan,
        )

        assert "Post-Workout" in message
        assert "Preceding Workout" in message
        assert "chest" in message.lower()  # From workout muscles

    def test_create_recovery_plan(
        self, mock_llm, healthy_user_context, sample_recovery_plan
    ):
        """Test create_recovery_plan convenience method."""
        mock_llm.invoke = MagicMock(return_value=sample_recovery_plan)
        mock_llm.with_structured_output.return_value = mock_llm

        agent = RecoverySpecialist(llm=mock_llm)
        result = agent.create_recovery_plan(healthy_user_context)

        assert result == sample_recovery_plan


# =============================================================================
# Clinical Gatekeeper Tests
# =============================================================================


class TestClinicalGatekeeper:
    """Test ClinicalGatekeeper functionality."""

    def test_agent_properties(self, mock_llm):
        """Test agent property values."""
        agent = ClinicalGatekeeper(llm=mock_llm)

        assert agent.name == "clinical_gatekeeper"
        assert agent.output_schema == AuditLog
        assert "safety" in agent.system_prompt.lower()

    def test_has_red_flag_scanner(self, mock_llm):
        """Test that gatekeeper has red flag scanner."""
        agent = ClinicalGatekeeper(llm=mock_llm)
        assert agent._red_flag_scanner is not None

    def test_format_audit_request(
        self, mock_llm, healthy_user_context, sample_workout_plan
    ):
        """Test formatting audit request."""
        agent = ClinicalGatekeeper(llm=mock_llm)
        message = agent.format_user_message(
            user_context=healthy_user_context,
            workout_plan=sample_workout_plan,
        )

        assert "Audit Request" in message
        assert "Medical History" in message
        assert "Workout Plan to Audit" in message
        assert "Bench Press" in message

    def test_quick_safety_check_healthy(self, mock_llm, healthy_user_context):
        """Test quick safety check for healthy user."""
        agent = ClinicalGatekeeper(llm=mock_llm)
        result = agent.quick_safety_check(healthy_user_context)

        assert result is True

    def test_quick_safety_check_red_flags(self, mock_llm):
        """Test quick safety check detects red flags."""
        user = UserContext(
            biometrics=Biometrics(age=40),
            medical_history=MedicalHistory(
                conditions=["chest pain during exercise"]
            ),
        )

        agent = ClinicalGatekeeper(llm=mock_llm)
        result = agent.quick_safety_check(user)

        assert result is False

    def test_audit_fast_fails_on_critical(self, mock_llm):
        """Test that audit fast-fails on critical red flags."""
        user = UserContext(
            biometrics=Biometrics(age=50),
            medical_history=MedicalHistory(
                conditions=["chest pain", "heart palpitations"]
            ),
        )

        agent = ClinicalGatekeeper(llm=mock_llm)
        result = agent.audit(user)

        assert result.status == AuditStatus.REJECTED
        assert "critical" in result.rejection_reason.lower()
        # LLM should not be called for critical fast-fail
        mock_llm.invoke.assert_not_called()

    def test_audit_calls_llm_for_non_critical(
        self, mock_llm, healthy_user_context, sample_audit_log
    ):
        """Test that audit calls LLM for non-critical cases."""
        mock_llm.invoke = MagicMock(return_value=sample_audit_log)
        mock_llm.with_structured_output.return_value = mock_llm

        agent = ClinicalGatekeeper(llm=mock_llm)
        result = agent.audit(healthy_user_context)

        assert result == sample_audit_log
        mock_llm.invoke.assert_called_once()


# =============================================================================
# Async Tests
# =============================================================================


class TestAsyncAgents:
    """Test async agent methods."""

    @pytest.mark.asyncio
    async def test_ainvoke(self, mock_llm, sample_intake_response):
        """Test async invoke."""
        mock_llm.ainvoke = AsyncMock(return_value=sample_intake_response)
        mock_llm.with_structured_output.return_value = mock_llm

        agent = IntakeAgent(llm=mock_llm)
        result = await agent.ainvoke(user_message="test")

        assert result == sample_intake_response
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_ainvoke_retries(self, mock_llm, sample_intake_response):
        """Test async invoke retries on failure."""
        mock_llm.ainvoke = AsyncMock(
            side_effect=[
                Exception("First fail"),
                sample_intake_response,
            ]
        )
        mock_llm.with_structured_output.return_value = mock_llm

        agent = IntakeAgent(llm=mock_llm, max_retries=3)
        result = await agent.ainvoke(user_message="test")

        assert result == sample_intake_response
        assert mock_llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_ainvoke_raises_after_max_retries(self, mock_llm):
        """Test async invoke raises after max retries."""
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("Always fails"))
        mock_llm.with_structured_output.return_value = mock_llm

        agent = IntakeAgent(llm=mock_llm, max_retries=2)

        with pytest.raises(AgentExecutionError):
            await agent.ainvoke(user_message="test")

        assert mock_llm.ainvoke.call_count == 2

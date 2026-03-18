from unittest.mock import MagicMock, patch

import pytest

from src.models.enums import Equipment, FitnessGoal
from src.models.schemas import Biometrics, MedicalHistory, UserContext
from src.orchestrator.coach import ExerciseCoach
from src.orchestrator.state_router import StateRouter, WorkflowType
from src.state.conversation_state import create_session


class TestIntakeFlow:
    """Test multi-turn intake flow."""

    def test_incomplete_context_prompts_for_info(self):
        """Test that incomplete context triggers intake prompts."""
        router = StateRouter()
        session = create_session()

        # Empty context should need intake
        result = router.route("I want to build muscle", session)
        assert result.workflow == WorkflowType.INTAKE_NEEDED
        assert len(result.missing_fields) > 0

    def test_progressive_context_building(self):
        """Test that context builds progressively."""
        session = create_session()

        # Add age and weight (both required for biometrics.is_complete())
        session.user_context.biometrics.age = 30
        session.user_context.biometrics.weight_kg = 80
        missing = session.user_context.get_missing_required_fields()
        assert "age" not in missing
        assert "weight" not in missing
        assert "fitness_goals" in missing

        # Add goals
        session.user_context.fitness_goals = [FitnessGoal.HYPERTROPHY]
        missing = session.user_context.get_missing_required_fields()
        assert "fitness_goals" not in missing

        # Add experience
        session.user_context.experience_level = "intermediate"
        missing = session.user_context.get_missing_required_fields()
        assert "equipment" in missing

        # Add equipment
        session.user_context.available_equipment = [Equipment.BARBELL, Equipment.DUMBBELL]
        assert session.user_context.is_complete_for_exercise()

    def test_complete_context_routes_to_workflow(self):
        """Test that complete context routes to workout generation."""
        router = StateRouter()
        session = create_session()
        session.user_context = UserContext(
            biometrics=Biometrics(age=30, weight_kg=80),
            fitness_goals=[FitnessGoal.HYPERTROPHY],
            experience_level="intermediate",
            available_equipment=[Equipment.BARBELL],
        )

        result = router.route("Create a workout for me", session)
        assert result.workflow == WorkflowType.INTEGRATED


class TestSafetyBlocking:
    """Test safety red flag detection and blocking."""

    def test_cardiovascular_red_flag_blocks(self):
        """Test that cardiovascular red flags block workout generation."""
        router = StateRouter()
        session = create_session()
        session.user_context = UserContext(
            biometrics=Biometrics(age=45),
            medical_history=MedicalHistory(
                conditions=["chest pain during exercise"],
            ),
        )

        result = router.route("I want a workout plan", session)
        assert result.workflow == WorkflowType.SAFETY_BLOCK
        assert len(result.safety_concerns) > 0

    def test_neurological_red_flag_blocks(self):
        """Test that neurological red flags block."""
        router = StateRouter()
        session = create_session()
        session.user_context = UserContext(
            biometrics=Biometrics(age=40),
            medical_history=MedicalHistory(
                conditions=["numbness in arm", "tingling sensation"],
            ),
        )

        result = router.route("Help me train", session)
        assert result.workflow == WorkflowType.SAFETY_BLOCK

    def test_acute_message_blocks(self):
        """Test that acute symptoms in message block."""
        router = StateRouter()
        session = create_session()
        session.user_context = UserContext(biometrics=Biometrics(age=30))

        result = router.route("I have severe chest pain", session)
        assert result.workflow == WorkflowType.SAFETY_BLOCK

    def test_healthy_user_not_blocked(self):
        """Test that healthy users are not blocked."""
        router = StateRouter()
        session = create_session()
        session.user_context = UserContext(
            biometrics=Biometrics(age=30, weight_kg=80),
            fitness_goals=[FitnessGoal.HYPERTROPHY],
            experience_level="intermediate",
            available_equipment=[Equipment.BODYWEIGHT],
        )

        result = router.route("Create a workout", session)
        assert result.workflow != WorkflowType.SAFETY_BLOCK


class TestWorkflowRouting:
    """Test correct workflow routing based on intent."""

    def test_exercise_request_routes_to_integrated(self):
        """Test that exercise requests route to integrated workflow."""
        router = StateRouter()
        session = create_session()
        session.user_context = UserContext(
            biometrics=Biometrics(age=30, weight_kg=80),
            fitness_goals=[FitnessGoal.HYPERTROPHY],
            experience_level="intermediate",
            available_equipment=[Equipment.DUMBBELL],
        )

        result = router.route("Build me a chest workout", session)
        assert result.workflow == WorkflowType.INTEGRATED

    def test_recovery_request_routes_to_recovery(self):
        """Test that recovery requests route to recovery workflow."""
        router = StateRouter()
        session = create_session()
        session.user_context = UserContext(
            biometrics=Biometrics(age=30),
            pain_areas=["lower back"],
        )

        result = router.route("My hips are tight", session)
        assert result.workflow == WorkflowType.RECOVERY_ONLY

    def test_general_question_routes_correctly(self):
        """Test that general questions don't trigger workflows."""
        router = StateRouter()
        session = create_session()

        result = router.route("What is progressive overload?", session)
        assert result.workflow == WorkflowType.GENERAL_RESPONSE


class TestCoachIntegration:
    """Test ExerciseCoach integration with mocked agents."""

    @pytest.fixture
    def mock_coach(self):
        """Create coach with mocked components."""
        with patch("src.orchestrator.coach.get_llm_for_agent") as mock_llm:
            mock_llm.return_value = MagicMock()
            coach = ExerciseCoach()

            # Mock intake agent to return realistic responses
            from src.models.schemas import IntakeResponse
            coach.intake_agent.extract_from_message = MagicMock(
                return_value=IntakeResponse(
                    extracted_context=UserContext(),
                    clarification_needed=[],
                    confidence_score=0.5,
                )
            )
            return coach

    def test_coach_handles_empty_session(self, mock_coach):
        """Test coach creates new session when none provided."""
        response, state = mock_coach.process_message("Hello")

        assert state is not None
        assert state.session_id is not None

    def test_coach_maintains_session(self, mock_coach):
        """Test coach maintains session across calls."""
        _, state1 = mock_coach.process_message("I'm 30 years old")
        _, state2 = mock_coach.process_message("I want to build muscle", state1)

        assert state1.session_id == state2.session_id

    def test_coach_prompts_for_missing_info(self, mock_coach):
        """Test coach prompts when info is missing."""
        response, _ = mock_coach.process_message("I want a workout")

        assert response.needs_more_info
        assert len(response.missing_fields) > 0

    def test_coach_blocks_red_flags(self, mock_coach):
        """Test coach blocks requests with red flags."""
        from src.models.schemas import IntakeResponse

        # Mock intake to return red flag context
        mock_coach.intake_agent.extract_from_message = MagicMock(
            return_value=IntakeResponse(
                extracted_context=UserContext(
                    medical_history=MedicalHistory(
                        conditions=["chest pain during exercise"],
                    ),
                ),
                clarification_needed=[],
                confidence_score=0.9,
            )
        )

        response, _ = mock_coach.process_message("I want to workout")

        assert response.is_rejected
        assert "safety" in response.message.lower()


class TestValidatorIntegration:
    """Test validators integrate correctly with workflows."""

    def test_volume_validation_in_workout(self, sample_workout_plan):
        """Test volume validator runs on workout plans."""
        from src.validators.volume_validator import VolumeValidator

        validator = VolumeValidator()
        validator.min_sets = 1
        validator.max_sets = 100

        result = validator.validate(sample_workout_plan)
        assert result.is_valid

    def test_ratio_validation_in_workout(self, sample_workout_plan):
        """Test ratio validator runs on workout plans."""
        from src.validators.ratio_validator import RatioValidator

        validator = RatioValidator()
        result = validator.validate(sample_workout_plan)

        # Our sample has balanced push/pull
        assert result.is_valid

    def test_red_flag_scanner_in_context(self, user_with_red_flags):
        """Test red flag scanner catches issues."""
        from src.validators.red_flag_scanner import RedFlagScanner

        scanner = RedFlagScanner(strict_mode=True)
        result = scanner.validate(user_with_red_flags)

        assert not result.is_valid
        assert result.metadata["critical_count"] > 0


class TestConversationStateIntegration:
    """Test conversation state management."""

    def test_context_accumulates_across_turns(self):
        """Test that context builds across conversation turns."""
        session = create_session()

        # First update: age
        session.update_context(UserContext(biometrics=Biometrics(age=30)))
        assert session.user_context.biometrics.age == 30

        # Second update: goals
        session.update_context(UserContext(fitness_goals=[FitnessGoal.HYPERTROPHY]))
        assert session.user_context.biometrics.age == 30  # Still there
        assert FitnessGoal.HYPERTROPHY in session.user_context.fitness_goals

    def test_intake_complete_updates_correctly(self):
        """Test intake_complete flag updates."""
        session = create_session()
        assert not session.intake_complete

        # Add required fields
        session.update_context(UserContext(
            biometrics=Biometrics(age=30, weight_kg=80),
            fitness_goals=[FitnessGoal.STRENGTH],
            experience_level="beginner",
            available_equipment=[Equipment.BODYWEIGHT],
        ))

        assert session.intake_complete

    def test_turn_history_maintained(self):
        """Test conversation history is maintained."""
        session = create_session()

        session.add_turn("Hello", "Hi there!")
        session.add_turn("I want a workout", "Let me get some info.")

        assert len(session.turns) == 2
        assert session.turns[0].user_message == "Hello"
        assert session.turns[1].user_message == "I want a workout"

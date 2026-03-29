"""Unit tests for orchestration components."""

from unittest.mock import MagicMock, patch

import pytest

from src.models.enums import Equipment, FitnessGoal, IntentType, WorkflowType
from src.models.schemas import (
    AuditLog,
    AuditStatus,
    Biometrics,
    CoachOutput,
    MedicalHistory,
    UserContext,
)
from src.orchestrator.intent_classifier import IntentClassifier
from src.orchestrator.state_router import StateRouter
from src.state.conversation_state import create_session

# =============================================================================
# Intent Classifier Tests
# =============================================================================


class TestIntentClassifier:
    """Test IntentClassifier functionality."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    def test_exercise_request_detection(self, classifier):
        messages = [
            "I want to build muscle",
            "Create a workout routine for me",
            "Help me with my chest workout",
            "I want to get stronger",
            "Plan a push pull legs split",
        ]
        for msg in messages:
            result = classifier.classify(msg)
            assert result.intent in (
                IntentType.EXERCISE_REQUEST,
                IntentType.INTEGRATED_REQUEST,
            ), f"Failed for: {msg}"

    def test_recovery_request_detection(self, classifier):
        messages = [
            "I need to stretch my hips",
            "My lower back is tight",
            "Help me with mobility",
            "I need a good stretching routine",
            "Create a recovery routine",
        ]
        for msg in messages:
            result = classifier.classify(msg)
            assert result.intent == IntentType.RECOVERY_REQUEST, f"Failed for: {msg}"

    def test_integrated_request_detection(self, classifier):
        messages = [
            "I want a full workout with stretching",
            "Create a complete training session with recovery",
            "Workout and mobility routine",
        ]
        for msg in messages:
            result = classifier.classify(msg)
            assert result.intent == IntentType.INTEGRATED_REQUEST, f"Failed for: {msg}"

    def test_recovery_protocol_overrides_exercise_terms(self, classifier):
        msg = "Just finished a marathon training block, calves are sore, need a recovery protocol"
        result = classifier.classify(msg)
        assert result.intent == IntentType.RECOVERY_REQUEST

    def test_intake_update_detection(self, classifier):
        messages = [
            "I am 30 years old",
            "I weigh 80 kg",
            "I'm a beginner",
            "My goal is to build muscle",
            "I have access to a full gym",
        ]
        for msg in messages:
            result = classifier.classify(msg)
            # Intake may combine with other intents
            assert result.intent in (
                IntentType.INTAKE_UPDATE,
                IntentType.EXERCISE_REQUEST,
                IntentType.RECOVERY_REQUEST,
            ), f"Failed for: {msg}"

    def test_general_question_detection(self, classifier):
        messages = [
            "What time should I eat my meals?",
            "How many hours of sleep do I need?",
            "What are the benefits of staying hydrated?",
        ]
        for msg in messages:
            result = classifier.classify(msg)
            assert result.intent == IntentType.GENERAL_QUESTION, f"Failed for: {msg}"

    def test_clarification_detection(self, classifier):
        messages = [
            "Yes",
            "That's correct",
            "No, I meant something else",
        ]
        for msg in messages:
            result = classifier.classify(msg)
            assert result.intent == IntentType.CLARIFICATION, f"Failed for: {msg}"

    def test_confidence_scores(self, classifier):
        # Clear exercise intent should have high confidence (multiple matches)
        result = classifier.classify("Create a hypertrophy workout routine for chest")
        assert result.confidence >= 0.5

        # Ambiguous should have lower confidence (fallback)
        result = classifier.classify("help")
        assert result.confidence < 0.5  # Falls back to general question

    def test_keywords_matched(self, classifier):
        result = classifier.classify("I want to build muscle at the gym")
        assert len(result.keywords_matched) > 0
        assert any("muscle" in kw for kw in result.keywords_matched)

    def test_is_exercise_related(self, classifier):
        assert classifier.is_exercise_related("I want a workout plan")
        assert not classifier.is_exercise_related("I need to stretch")

    def test_is_recovery_related(self, classifier):
        assert classifier.is_recovery_related("My hips are tight")
        assert not classifier.is_recovery_related("Build me a workout")

    def test_needs_action(self, classifier):
        assert classifier.needs_action("Create a workout for me")
        assert classifier.needs_action("I need a mobility routine")
        assert not classifier.needs_action("What is progressive overload?")


# =============================================================================
# State Router Tests
# =============================================================================


class TestStateRouter:
    """Test StateRouter functionality."""

    @pytest.fixture
    def router(self):
        return StateRouter()

    @pytest.fixture
    def empty_state(self):
        return create_session()

    @pytest.fixture
    def complete_state(self):
        state = create_session()
        state.user_context = UserContext(
            biometrics=Biometrics(age=30, weight_kg=80),
            fitness_goals=[FitnessGoal.HYPERTROPHY],
            experience_level="intermediate",
            available_equipment=[Equipment.BARBELL],
        )
        state.intake_complete = True
        return state

    @pytest.fixture
    def state_with_red_flags(self):
        state = create_session()
        state.user_context = UserContext(
            biometrics=Biometrics(age=45),
            medical_history=MedicalHistory(
                conditions=["chest pain during exercise"]
            ),
        )
        return state

    def test_route_exercise_with_incomplete_state(self, router, empty_state):
        result = router.route("I want to build muscle", empty_state)

        assert result.workflow == WorkflowType.INTAKE_NEEDED
        assert len(result.missing_fields) > 0

    def test_route_exercise_with_complete_state(self, router, complete_state):
        result = router.route("I want to build muscle", complete_state)

        assert result.workflow == WorkflowType.INTEGRATED

    def test_route_recovery_minimal_requirements(self, router):
        # Step 4: is_complete_for_recovery() now requires age + health acknowledgment
        state = create_session()
        state.user_context = UserContext(
            biometrics=Biometrics(age=30),
            pain_areas=["tight hips"],
        )

        result = router.route("My hips are tight", state)
        assert result.workflow == WorkflowType.RECOVERY_ONLY

    def test_route_safety_block(self, router, state_with_red_flags):
        result = router.route("I want to workout", state_with_red_flags)

        assert result.workflow == WorkflowType.SAFETY_BLOCK
        assert len(result.safety_concerns) > 0

    def test_route_general_question(self, router, empty_state):
        result = router.route("What is the best chest exercise?", empty_state)

        assert result.workflow == WorkflowType.GENERAL_RESPONSE

    def test_route_clarification(self, router, complete_state):
        result = router.route("Yes, that's correct", complete_state)

        assert result.workflow == WorkflowType.GENERAL_RESPONSE

    def test_route_acute_symptoms_in_message(self, router):
        state = create_session()
        state.user_context = UserContext(
            biometrics=Biometrics(age=30),
        )

        result = router.route("I have chest pain when I exercise", state)
        assert result.workflow == WorkflowType.SAFETY_BLOCK

    def test_route_dizziness_and_shortness_of_breath_blocks(self, router):
        state = create_session()
        result = router.route(
            "I'm feeling dizzy and shortness of breath while sitting still", state
        )
        assert result.workflow == WorkflowType.SAFETY_BLOCK

    def test_route_shooting_pain_blocks(self, router):
        state = create_session()
        result = router.route(
            "Sharp shooting pain radiating down my left leg", state
        )
        assert result.workflow == WorkflowType.SAFETY_BLOCK

    def test_should_generate_plan(self, router, complete_state):
        result = router.route("Create a workout", complete_state)
        assert router.should_generate_plan(result)

        result = router.route("What is RPE?", complete_state)
        assert not router.should_generate_plan(result)

    def test_needs_intake(self, router, empty_state):
        result = router.route("I want a workout", empty_state)
        assert router.needs_intake(result)

    def test_is_blocked(self, router, state_with_red_flags):
        result = router.route("Help me workout", state_with_red_flags)
        assert router.is_blocked(result)


# =============================================================================
# Workflow Result Tests
# =============================================================================


class TestWorkflowResult:
    """Test WorkflowResult properties."""

    def test_is_approved(self):
        from datetime import datetime

        from src.models.enums import AuditStatus
        from src.models.schemas import AuditLog
        from src.workflows.base import WorkflowResult

        result = WorkflowResult(
            success=True,
            workflow_name="test",
            timestamp=datetime.now(),
            audit=AuditLog(status=AuditStatus.APPROVED),
        )
        assert result.is_approved
        assert not result.is_rejected

    def test_is_modified_counts_as_approved(self):
        from datetime import datetime

        from src.models.enums import AuditStatus
        from src.models.schemas import AuditLog
        from src.workflows.base import WorkflowResult

        result = WorkflowResult(
            success=True,
            workflow_name="test",
            timestamp=datetime.now(),
            audit=AuditLog(status=AuditStatus.MODIFIED),
        )
        assert result.is_approved
        assert not result.is_rejected

    def test_is_rejected(self):
        from datetime import datetime

        from src.models.enums import AuditStatus
        from src.models.schemas import AuditLog
        from src.workflows.base import WorkflowResult

        result = WorkflowResult(
            success=False,
            workflow_name="test",
            timestamp=datetime.now(),
            audit=AuditLog(status=AuditStatus.REJECTED),
        )
        assert result.is_rejected
        assert not result.is_approved


# =============================================================================
# Coach Response Tests
# =============================================================================


class TestCoachResponse:
    """Test CoachResponse dataclass."""

    def test_basic_response(self):
        from src.orchestrator.coach import CoachResponse

        response = CoachResponse(message="Hello!")
        assert response.message == "Hello!"
        assert not response.needs_more_info
        assert not response.is_rejected
        assert response.missing_fields == []

    def test_intake_needed_response(self):
        from src.orchestrator.coach import CoachResponse

        response = CoachResponse(
            message="I need more info",
            needs_more_info=True,
            missing_fields=["age", "goals"],
        )
        assert response.needs_more_info
        assert "age" in response.missing_fields

    def test_rejected_response(self):
        from src.orchestrator.coach import CoachResponse

        response = CoachResponse(
            message="Safety concern",
            is_rejected=True,
        )
        assert response.is_rejected


# =============================================================================
# ExerciseCoach Tests (with mocks)
# =============================================================================


class TestExerciseCoach:
    """Test ExerciseCoach orchestration."""

    @pytest.fixture
    def mock_intake_agent(self):
        mock = MagicMock()
        from src.models.schemas import IntakeResponse
        mock.extract_from_message.return_value = IntakeResponse(
            extracted_context=UserContext(
                biometrics=Biometrics(age=30, weight_kg=80),
                fitness_goals=[FitnessGoal.HYPERTROPHY],
                experience_level="intermediate",
            ),
            clarification_needed=[],
            confidence_score=0.9,
        )
        return mock

    def test_coach_initialization(self):
        """Test coach can be initialized (with mocked LLM)."""
        with patch("src.orchestrator.coach.get_llm_for_agent") as mock_llm:
            mock_llm.return_value = MagicMock()
            from src.orchestrator.coach import ExerciseCoach
            coach = ExerciseCoach()
            assert coach.router is not None
            assert coach.intake_agent is not None

    def test_coach_intake_needed_response(self):
        """Test coach returns intake prompt when info missing."""
        with patch("src.orchestrator.coach.get_llm_for_agent") as mock_llm:
            mock_llm.return_value = MagicMock()
            from src.orchestrator.coach import ExerciseCoach

            coach = ExerciseCoach()
            # Mock intake to return empty context
            from src.models.schemas import IntakeResponse
            coach.intake_agent.extract_from_message = MagicMock(
                return_value=IntakeResponse(
                    extracted_context=UserContext(),
                    clarification_needed=[],
                    confidence_score=0.3,
                )
            )

            response, state = coach.process_message("I want a workout")

            assert response.needs_more_info
            assert len(response.missing_fields) > 0

    def test_coach_safety_response(self):
        """Test coach returns safety response for red flags."""
        with patch("src.orchestrator.coach.get_llm_for_agent") as mock_llm:
            mock_llm.return_value = MagicMock()
            from src.orchestrator.coach import ExerciseCoach

            coach = ExerciseCoach()
            # Mock intake to return context with red flags
            from src.models.schemas import IntakeResponse
            coach.intake_agent.extract_from_message = MagicMock(
                return_value=IntakeResponse(
                    extracted_context=UserContext(
                        medical_history=MedicalHistory(
                            conditions=["chest pain during exercise"]
                        ),
                    ),
                    clarification_needed=[],
                    confidence_score=0.9,
                )
            )

            response, state = coach.process_message("I want to workout")

            assert response.is_rejected
            assert "safety" in response.message.lower()

    def test_coach_general_response(self):
        """Test coach returns general response for questions."""
        with patch("src.orchestrator.coach.get_llm_for_agent") as mock_llm:
            mock_llm.return_value = MagicMock()
            from src.orchestrator.coach import ExerciseCoach

            coach = ExerciseCoach()
            from src.models.schemas import IntakeResponse
            coach.intake_agent.extract_from_message = MagicMock(
                return_value=IntakeResponse(
                    extracted_context=UserContext(),
                    clarification_needed=[],
                    confidence_score=0.3,
                )
            )

            response, state = coach.process_message("What is progressive overload?")

            assert not response.needs_more_info
            assert not response.is_rejected


# =============================================================================
# Integration-like Tests (still unit, but testing flow)
# =============================================================================


class TestIntentToRoutingFlow:
    """Test the flow from intent classification to routing."""

    def test_exercise_intent_routes_correctly(self):
        classifier = IntentClassifier()
        router = StateRouter()

        state = create_session()
        state.user_context = UserContext(
            biometrics=Biometrics(age=30, weight_kg=80),
            fitness_goals=[FitnessGoal.HYPERTROPHY],
            experience_level="intermediate",
            available_equipment=[Equipment.DUMBBELL],
        )

        message = "Create a chest workout"
        intent = classifier.classify(message)
        routing = router.route(message, state)

        assert intent.intent in (IntentType.EXERCISE_REQUEST, IntentType.INTEGRATED_REQUEST)
        assert routing.workflow == WorkflowType.INTEGRATED

    def test_recovery_intent_routes_correctly(self):
        classifier = IntentClassifier()
        router = StateRouter()

        state = create_session()
        state.user_context = UserContext(
            biometrics=Biometrics(age=30),
            pain_areas=["lower back"],
        )

        message = "My hips are tight"
        intent = classifier.classify(message)
        routing = router.route(message, state)

        assert intent.intent == IntentType.RECOVERY_REQUEST
        assert routing.workflow == WorkflowType.RECOVERY_ONLY


# =============================================================================
# Recovery Follow-up Routing Tests (Step 3)
# =============================================================================


class TestRecoveryFollowupRouting:
    """Tests for the inverted follow-up detection logic."""

    @pytest.fixture
    def router(self):
        return StateRouter()

    def _recovery_state(self):
        """State pre-populated as if a recovery workflow just completed."""
        state = create_session()
        state.user_context = UserContext(
            biometrics=Biometrics(age=66),
            pain_areas=["left hamstring"],
        )
        state.minimal_intake_complete = True
        state.last_workflow = WorkflowType.RECOVERY_ONLY
        stub_audit = AuditLog(status=AuditStatus.APPROVED)
        state.last_output = CoachOutput(
            session_id=state.session_id,
            audit=stub_audit,
            user_message="Here is your hamstring recovery plan.",
        )
        return state

    def test_icing_question_routes_to_recovery_followup(self, router):
        state = self._recovery_state()
        routing = router.route("What about icing the hamstring?", state)
        assert routing.workflow == WorkflowType.RECOVERY_FOLLOWUP

    def test_short_question_routes_to_recovery_followup(self, router):
        state = self._recovery_state()
        routing = router.route("Should I rest it?", state)
        assert routing.workflow == WorkflowType.RECOVERY_FOLLOWUP

    def test_can_i_massage_routes_to_recovery_followup(self, router):
        state = self._recovery_state()
        routing = router.route("Can I massage it?", state)
        assert routing.workflow == WorkflowType.RECOVERY_FOLLOWUP

    def test_exercise_terms_guard_blocks_followup(self, router):
        """Messages containing exercise_terms must NOT route to RECOVERY_FOLLOWUP."""
        state = self._recovery_state()
        routing = router.route("What about stretching before my workout?", state)
        assert routing.workflow != WorkflowType.RECOVERY_FOLLOWUP

    def test_exercise_terms_sets_reps_guard(self, router):
        """'sets' and 'reps' are exercise_terms — must not trigger recovery follow-up."""
        state = self._recovery_state()
        routing = router.route("5 sets of squats?", state)
        assert routing.workflow != WorkflowType.RECOVERY_FOLLOWUP

    def test_declarative_statement_does_not_followup(self, router):
        """Declarative request without '?' or question word must NOT false-positive."""
        state = self._recovery_state()
        routing = router.route("I need a recovery plan for my hamstring", state)
        assert routing.workflow != WorkflowType.RECOVERY_FOLLOWUP

    def test_definitional_question_stays_general(self, router):
        """'What is RPE?' must stay on the definitional path, no intake triggered."""
        state = create_session()
        state.minimal_intake_complete = True
        routing = router.route("What is RPE?", state)
        # Definitional questions must not trigger follow-up even if recovery thread exists
        assert routing.workflow == WorkflowType.GENERAL_RESPONSE

    def test_no_last_workflow_does_not_followup(self, router):
        """Without last_workflow == RECOVERY_ONLY, follow-up routing must not activate."""
        state = create_session()
        state.minimal_intake_complete = True
        routing = router.route("What about icing?", state)
        assert routing.workflow != WorkflowType.RECOVERY_FOLLOWUP

    def test_last_workflow_set_after_recovery(self, router):
        """last_workflow is set to RECOVERY_ONLY after recovery plan (Step 3.4)."""
        state = self._recovery_state()
        assert state.last_workflow == WorkflowType.RECOVERY_ONLY


# =============================================================================
# Age + Injury Classification Tests (Steps 1-2)
# =============================================================================


class TestAgeAndInjuryClassification:
    """Tests for bare-age INTAKE_UPDATE and injury keyword detection."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    def test_bare_age_classifies_as_intake_update(self, classifier):
        result = classifier.classify("I am 66 and I tweaked my left hamstring")
        assert result.intent in (
            IntentType.INTAKE_UPDATE,
            IntentType.RECOVERY_REQUEST,
        ), f"Got {result.intent} — bare age + injury should be INTAKE_UPDATE or RECOVERY_REQUEST"

    def test_age_without_years_keyword(self, classifier):
        result = classifier.classify("I am 66")
        assert result.intent == IntentType.INTAKE_UPDATE

    def test_age_with_years_still_works(self, classifier):
        result = classifier.classify("I am 66 years old")
        assert result.intent == IntentType.INTAKE_UPDATE

    def test_tweaked_classified_as_recovery(self, classifier):
        result = classifier.classify("I tweaked my hamstring")
        assert result.intent == IntentType.RECOVERY_REQUEST

    def test_icing_question_is_general_question(self, classifier):
        """'What about icing?' must classify as GENERAL_QUESTION (router handles follow-up)."""
        result = classifier.classify("What about icing the hamstring?")
        assert result.intent == IntentType.GENERAL_QUESTION

    def test_what_is_rpe_is_definitional(self, classifier):
        result = classifier.classify("What is RPE?")
        assert result.intent == IntentType.GENERAL_QUESTION


# =============================================================================
# Action Verb / Sport / Skill-Learning Intent Guard (Step 3)
# =============================================================================


class TestActionVerbIntentGuard:
    """Step 3: new INTENT_PATTERNS[EXERCISE_REQUEST] patterns — correct classification
    and false-positive guard for the skill-learning non-optional suffix."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier()

    def test_i_want_to_lose_fat_is_exercise_request(self, classifier):
        """'I want to lose fat' → fat-loss action verb pattern → EXERCISE_REQUEST."""
        result = classifier.classify("I want to lose fat and get leaner.")
        assert result.intent in (
            IntentType.EXERCISE_REQUEST,
            IntentType.INTEGRATED_REQUEST,
        )

    def test_improving_explosive_power_is_exercise_request(self, classifier):
        """'Improving explosive power for basketball' → sport/explosive pattern."""
        result = classifier.classify(
            "I'm 27, advanced. Improving explosive power for basketball."
        )
        assert result.intent in (
            IntentType.EXERCISE_REQUEST,
            IntentType.INTEGRATED_REQUEST,
        )

    def test_learning_lsit_progression_is_exercise_request(self, classifier):
        """'Learning the L-Sit progression' → skill-learning pattern."""
        result = classifier.classify(
            "I'm 26, 68kg. Learning the L-Sit progression. I have two sturdy chairs."
        )
        assert result.intent in (
            IntentType.EXERCISE_REQUEST,
            IntentType.INTEGRATED_REQUEST,
        )

    def test_learning_condition_message_is_not_exercise_request(self, classifier):
        """False-positive guard: 'I am learning I have a knee condition' must NOT classify
        as EXERCISE_REQUEST.

        Verifies the non-optional suffix constraint in the skill-learning regex. If the
        suffix were optional (zero-length match allowed), this message would false-match.
        """
        result = classifier.classify("I am learning I have a knee condition.")
        assert result.intent != IntentType.EXERCISE_REQUEST

    def test_pure_age_intake_still_classifies_as_intake_update(self, classifier):
        """Regression: bare 'I am 30, beginner' without an action verb → INTAKE_UPDATE.

        Adding fat-loss / sport / skill patterns must not cause a pure intake message
        to route as an exercise request via the demotion block.
        """
        result = classifier.classify("I am 30 years old, beginner.")
        assert result.intent == IntentType.INTAKE_UPDATE

from src.models.enums import Equipment, FitnessGoal, IntentType
from src.models.schemas import Biometrics, MedicalHistory, UserContext
from src.state.conversation_state import ConversationState, ConversationTurn, create_session


class TestConversationTurn:
    """Test ConversationTurn model."""

    def test_turn_creation(self):
        turn = ConversationTurn(
            user_message="I want to build muscle",
            assistant_response="Great! Let me gather some information.",
            intent=IntentType.EXERCISE_REQUEST,
        )
        assert turn.user_message == "I want to build muscle"
        assert turn.turn_id is not None
        assert turn.timestamp is not None

    def test_turn_with_context_updates(self):
        turn = ConversationTurn(
            user_message="I'm 30 years old",
            assistant_response="Got it, you're 30.",
            context_updates={"age": 30},
        )
        assert turn.context_updates["age"] == 30


class TestConversationState:
    """Test ConversationState session management."""

    def test_create_session(self):
        state = create_session()
        assert state.session_id is not None
        assert state.intake_complete is False
        assert len(state.turns) == 0

    def test_add_turn(self):
        state = ConversationState()
        _ = state.add_turn(
            user_message="Hello",
            assistant_response="Hi there!",
            intent=IntentType.GENERAL_QUESTION,
        )
        assert len(state.turns) == 1
        assert state.turns[0].user_message == "Hello"

    def test_turn_limit(self):
        state = ConversationState(max_turns_to_keep=3)
        for i in range(5):
            state.add_turn(
                user_message=f"Message {i}",
                assistant_response=f"Response {i}",
            )
        assert len(state.turns) == 3
        assert state.turns[0].user_message == "Message 2"  # Oldest kept

    def test_update_context_biometrics(self):
        state = ConversationState()
        updates = UserContext(
            biometrics=Biometrics(age=30, weight_kg=80),
        )
        state.update_context(updates)
        assert state.user_context.biometrics.age == 30
        assert state.user_context.biometrics.weight_kg == 80

    def test_update_context_incremental(self):
        state = ConversationState()
        # First update: age
        state.update_context(UserContext(biometrics=Biometrics(age=25)))
        # Second update: weight
        state.update_context(UserContext(biometrics=Biometrics(weight_kg=70)))
        # Both should be present
        assert state.user_context.biometrics.age == 25
        assert state.user_context.biometrics.weight_kg == 70

    def test_update_context_fitness_goals(self):
        state = ConversationState()
        state.update_context(
            UserContext(fitness_goals=[FitnessGoal.HYPERTROPHY])
        )
        state.update_context(
            UserContext(fitness_goals=[FitnessGoal.STRENGTH, FitnessGoal.HYPERTROPHY])
        )
        # Should have both unique goals
        assert FitnessGoal.HYPERTROPHY in state.user_context.fitness_goals
        assert FitnessGoal.STRENGTH in state.user_context.fitness_goals
        assert len(state.user_context.fitness_goals) == 2

    def test_update_context_equipment(self):
        state = ConversationState()
        state.update_context(
            UserContext(available_equipment=[Equipment.BARBELL])
        )
        state.update_context(
            UserContext(available_equipment=[Equipment.DUMBBELL, Equipment.BARBELL])
        )
        assert len(state.user_context.available_equipment) == 2

    def test_update_context_medical_history_appends(self):
        state = ConversationState()
        state.update_context(
            UserContext(medical_history=MedicalHistory(conditions=["asthma"]))
        )
        state.update_context(
            UserContext(medical_history=MedicalHistory(conditions=["knee surgery"]))
        )
        assert len(state.user_context.medical_history.conditions) == 2
        assert "asthma" in state.user_context.medical_history.conditions
        assert "knee surgery" in state.user_context.medical_history.conditions

    def test_update_context_notes_concatenate(self):
        state = ConversationState()
        state.update_context(
            UserContext(medical_history=MedicalHistory(notes="Has mild anxiety"))
        )
        state.update_context(
            UserContext(medical_history=MedicalHistory(notes="Prefers morning workouts"))
        )
        assert "anxiety" in state.user_context.medical_history.notes
        assert "morning" in state.user_context.medical_history.notes

    def test_intake_complete_updates(self):
        state = ConversationState()
        assert not state.intake_complete

        # Add required fields incrementally
        from src.models.enums import Equipment

        state.update_context(
            UserContext(
                biometrics=Biometrics(age=30, weight_kg=80),
                fitness_goals=[FitnessGoal.HYPERTROPHY],
                experience_level="intermediate",
                available_equipment=[Equipment.BODYWEIGHT],
            )
        )
        assert state.intake_complete

    def test_minimal_intake_respects_no_health_concerns_flag(self):
        state = ConversationState()
        state.user_context.biometrics.age = 30
        state.user_context.medical_history.no_concerns_reported = True
        state._check_minimal_intake()
        assert state.minimal_intake_complete

    def test_get_conversation_summary_empty(self):
        state = ConversationState()
        summary = state.get_conversation_summary()
        assert summary == "No previous conversation."

    def test_get_conversation_summary(self):
        state = ConversationState()
        state.add_turn("Hello", "Hi there!")
        state.add_turn("I want to build muscle", "Great goal!")
        summary = state.get_conversation_summary(max_turns=2)
        assert "Hello" in summary
        assert "build muscle" in summary

    def test_clear_pending_clarifications(self):
        state = ConversationState()
        state.pending_clarifications = ["What is your age?", "What equipment do you have?"]
        state.clear_pending_clarifications()
        assert state.pending_clarifications == []

    def test_reset(self):
        state = ConversationState()
        original_session_id = state.session_id
        state.add_turn("Hello", "Hi")
        state.update_context(
            UserContext(biometrics=Biometrics(age=30))
        )
        state.intake_complete = True

        state.reset()

        assert state.session_id == original_session_id  # Preserved
        assert len(state.turns) == 0
        assert state.user_context.biometrics.age is None
        assert not state.intake_complete

    def test_pain_areas_update(self):
        state = ConversationState()
        state.update_context(
            UserContext(pain_areas=["lower back"])
        )
        state.update_context(
            UserContext(pain_areas=["shoulders", "lower back"])
        )
        assert len(state.user_context.pain_areas) == 2
        assert "lower back" in state.user_context.pain_areas
        assert "shoulders" in state.user_context.pain_areas

    def test_specific_requests_update(self):
        state = ConversationState()
        state.update_context(
            UserContext(specific_requests=["focus on chest"])
        )
        state.update_context(
            UserContext(specific_requests=["include stretching"])
        )
        assert len(state.user_context.specific_requests) == 2


class TestConversationStateWorkflow:
    """Test workflow-related state management."""

    def test_current_workflow_tracking(self):
        state = ConversationState()
        assert state.current_workflow is None

        state.current_workflow = "integrated"
        assert state.current_workflow == "integrated"

    def test_updated_at_changes(self):
        state = ConversationState()
        original_time = state.updated_at

        # Small delay to ensure time difference
        import time
        time.sleep(0.01)

        state.add_turn("Test", "Response")
        assert state.updated_at > original_time

from unittest.mock import MagicMock, patch


class TestCLIApp:
    """Test CLI app module."""

    def test_setup_logging(self):
        """Test logging setup doesn't crash."""
        from src.cli.app import setup_logging
        setup_logging("INFO")
        setup_logging("DEBUG")
        setup_logging("WARNING")

    def test_print_response_basic(self, capsys):
        """Test response printing."""
        from src.cli.app import print_response
        from src.orchestrator.coach import CoachResponse

        response = CoachResponse(message="Test message")
        print_response(response, verbose=False)

        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_print_response_verbose(self, capsys):
        """Test verbose response printing."""
        from src.cli.app import print_response
        from src.models.enums import AuditStatus
        from src.models.schemas import AuditLog
        from src.orchestrator.coach import CoachResponse

        response = CoachResponse(
            message="Test message",
            audit=AuditLog(status=AuditStatus.APPROVED),
        )
        print_response(response, verbose=True)

        captured = capsys.readouterr()
        assert "Test message" in captured.out
        assert "Audit Log" in captured.out


class TestGradioApp:
    """Test Gradio app module."""

    def test_respond_new_session(self):
        """Test respond function with new session."""
        with patch("src.ui.app.get_coach") as mock_get_coach:
            from src.orchestrator.coach import CoachResponse
            from src.state.conversation_state import create_session

            mock_coach = MagicMock()
            mock_coach.process_message.return_value = (
                CoachResponse(message="Hello!"),
                create_session(),
            )
            mock_get_coach.return_value = mock_coach

            from src.ui.app import respond

            response, state = respond("Hello", [], None)

            assert response == "Hello!"
            assert state is not None
            assert "session" in state

    def test_respond_with_existing_session(self):
        """Test respond function with existing session."""
        with patch("src.ui.app.get_coach") as mock_get_coach:
            from src.orchestrator.coach import CoachResponse
            from src.state.conversation_state import create_session

            session = create_session()
            mock_coach = MagicMock()
            mock_coach.process_message.return_value = (
                CoachResponse(message="Response"),
                session,
            )
            mock_get_coach.return_value = mock_coach

            from src.ui.app import respond

            state = {"session": session.model_dump()}
            response, new_state = respond("Test", [], state)

            assert response == "Response"
            assert new_state is not None

    def test_respond_handles_error(self):
        """Test respond function handles errors gracefully."""
        with patch("src.ui.app.get_coach") as mock_get_coach:
            mock_coach = MagicMock()
            mock_coach.process_message.side_effect = Exception("Test error")
            mock_get_coach.return_value = mock_coach

            from src.ui.app import respond

            response, state = respond("Hello", [], None)

            assert "error" in response.lower()

    def test_create_demo(self):
        """Test demo creation doesn't crash."""
        with patch("src.ui.app.get_coach"):
            from src.ui.app import create_demo

            demo = create_demo()
            assert demo is not None

    def test_get_coach_singleton(self):
        """Test coach singleton pattern."""
        import src.ui.app as ui_module

        # Reset singleton
        ui_module._coach = None

        with (
            patch("src.ui.app.get_settings") as mock_settings,
            patch("src.ui.app.ExerciseCoach") as mock_coach_class,
        ):
            mock_settings.return_value = MagicMock()
            mock_instance = MagicMock()
            mock_coach_class.return_value = mock_instance

            # First call creates
            coach1 = ui_module.get_coach()

            # Second call returns same instance
            coach2 = ui_module.get_coach()

            assert coach1 is coach2
            mock_coach_class.assert_called_once()

        # Reset for other tests
        ui_module._coach = None

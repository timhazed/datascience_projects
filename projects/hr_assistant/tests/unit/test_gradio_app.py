import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from src.gradio.app import (
    _gradio_history_to_langchain,
    NestleApp,
)


class TestGradioHistoryToLangchain:
    """Tests for _gradio_history_to_langchain helper function."""

    def test_empty_history(self):
        """Empty history returns empty list."""
        result = _gradio_history_to_langchain([])
        assert result == []

    def test_single_exchange(self):
        """Single user-assistant exchange converts correctly."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        result = _gradio_history_to_langchain(history)

        assert len(result) == 2
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "Hello"
        assert isinstance(result[1], AIMessage)
        assert result[1].content == "Hi there"

    def test_multiple_exchanges(self):
        """Multiple exchanges convert correctly."""
        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"}
        ]
        result = _gradio_history_to_langchain(history)

        assert len(result) == 4
        assert result[0].content == "First question"
        assert result[1].content == "First answer"
        assert result[2].content == "Second question"
        assert result[3].content == "Second answer"

    def test_empty_user_message(self):
        """Empty user message is skipped."""
        history = [{"role": "user", "content": ""}, {"role": "assistant", "content": "Response"}]
        result = _gradio_history_to_langchain(history)

        assert len(result) == 1
        assert isinstance(result[0], AIMessage)

    def test_empty_assistant_message(self):
        """Empty assistant message is skipped."""
        history = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": ""}]
        result = _gradio_history_to_langchain(history)

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)

    def test_missing_content_skipped(self):
        """Messages without content key are skipped."""
        history = [{"role": "user"}, {"role": "assistant", "content": "Response"}]
        result = _gradio_history_to_langchain(history)

        assert len(result) == 1
        assert isinstance(result[0], AIMessage)


class TestNestleApp:
    """Tests for NestleApp class."""

    @pytest.fixture
    def mock_app_dependencies(self):
        """Mock all external dependencies for NestleApp."""
        with patch("src.gradio.app.load_dotenv"):
            with patch("src.gradio.app.load_settings") as mock_settings:
                mock_settings.return_value = MagicMock(
                    provider_name="openai",
                    provider_config=MagicMock(
                        model="gpt-3.5-turbo",
                        temperature=0.2,
                        max_tokens=512
                    ),
                    store_name="faiss",
                    store_config=MagicMock(persist_directory="data/faiss")
                )
                with patch("src.gradio.app.get_llm") as mock_get_llm:
                    with patch("src.gradio.app.OpenAIEmbeddings"):
                        with patch("src.gradio.app.get_vector_store") as mock_get_store:
                            mock_store = MagicMock()
                            mock_store.get_store.return_value = MagicMock()
                            mock_get_store.return_value = mock_store
                            with patch("src.gradio.app.Loader") as mock_loader:
                                mock_loader.return_value.load.return_value = []
                                yield {
                                    "settings": mock_settings,
                                    "get_llm": mock_get_llm,
                                    "get_store": mock_get_store,
                                    "loader": mock_loader
                                }

    def test_init_loads_settings(self, mock_app_dependencies):
        """__init__ loads settings from config."""
        NestleApp()
        mock_app_dependencies["settings"].assert_called_once()

    def test_init_creates_llm(self, mock_app_dependencies):
        """__init__ creates LLM with correct parameters."""
        NestleApp()
        mock_app_dependencies["get_llm"].assert_called_once()

    def test_init_creates_vector_store(self, mock_app_dependencies):
        """__init__ creates vector store."""
        NestleApp()
        mock_app_dependencies["get_store"].assert_called_once()

    def test_init_loads_documents(self, mock_app_dependencies):
        """__init__ loads documents via Loader."""
        NestleApp()
        mock_app_dependencies["loader"].assert_called_once()
        mock_app_dependencies["loader"].return_value.load.assert_called_once()

    def test_chat_empty_message_returns_empty(self, mock_app_dependencies):
        """chat returns empty string for whitespace-only message."""
        app = NestleApp()
        result = app.chat("   ", [], False)
        assert result == ""

    def test_chat_creates_assistant_per_call(self, mock_app_dependencies):
        """chat creates new HRAssistant for each call."""
        with patch("src.gradio.app.HRAssistant") as mock_assistant_cls:
            mock_assistant = MagicMock()
            mock_assistant.run.return_value = "response"
            mock_assistant_cls.return_value = mock_assistant

            app = NestleApp()
            app.chat("question", [], False)

            mock_assistant_cls.assert_called_once()
            call_kwargs = mock_assistant_cls.call_args[1]
            assert call_kwargs["evidence"] is False

    def test_chat_converts_history(self, mock_app_dependencies):
        """chat converts Gradio history to LangChain format."""
        with patch("src.gradio.app.HRAssistant") as mock_assistant_cls:
            with patch("src.gradio.app._gradio_history_to_langchain") as mock_convert:
                mock_convert.return_value = []
                mock_assistant = MagicMock()
                mock_assistant.run.return_value = "response"
                mock_assistant_cls.return_value = mock_assistant

                history = [{"role": "user", "content": "prev"}, {"role": "assistant", "content": "ans"}]
                app = NestleApp()
                app.chat("question", history, False)

                mock_convert.assert_called_once_with(history)

    def test_chat_returns_assistant_response(self, mock_app_dependencies):
        """chat returns the response from HRAssistant."""
        with patch("src.gradio.app.HRAssistant") as mock_assistant_cls:
            mock_assistant = MagicMock()
            mock_assistant.run.return_value = "test response"
            mock_assistant_cls.return_value = mock_assistant

            app = NestleApp()
            result = app.chat("question", [], False)

            assert result == "test response"

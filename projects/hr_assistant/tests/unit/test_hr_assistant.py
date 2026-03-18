import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.assistant.hr_assistant import HRAssistant, _CitationRetriever


class TestCitationRetriever:
    """Tests for _CitationRetriever wrapper class."""

    @pytest.fixture
    def mock_retriever(self):
        """Create mock base retriever."""
        mock = MagicMock()
        mock.invoke.return_value = [
            Document(
                page_content="PTO policy content",
                metadata={"document_name": "hr_policy.pdf", "chunk_number": 1}
            )
        ]
        return mock

    def test_invoke_with_citations_enabled(self, mock_retriever):
        """invoke prefixes content with source citation when enabled."""
        citation_retriever = _CitationRetriever(mock_retriever, include_citations=True)
        result = citation_retriever.invoke("PTO question")

        assert "[Source: hr_policy.pdf, Chunk 1]" in result[0].page_content
        assert "PTO policy content" in result[0].page_content

    def test_invoke_with_citations_disabled(self, mock_retriever):
        """invoke returns original content when citations disabled."""
        citation_retriever = _CitationRetriever(mock_retriever, include_citations=False)
        result = citation_retriever.invoke("PTO question")

        mock_retriever.invoke.assert_called_once_with("PTO question")
        assert result[0].page_content == "PTO policy content"

    def test_invoke_handles_missing_metadata(self, mock_retriever):
        """invoke handles documents with missing metadata gracefully."""
        mock_retriever.invoke.return_value = [
            Document(page_content="content", metadata={})
        ]

        citation_retriever = _CitationRetriever(mock_retriever, include_citations=True)
        result = citation_retriever.invoke("query")

        assert "[Source: unknown, Chunk ?]" in result[0].page_content

    def test_invoke_passes_args_to_base_retriever(self, mock_retriever):
        """invoke passes additional args/kwargs to base retriever."""
        citation_retriever = _CitationRetriever(mock_retriever, include_citations=True)
        citation_retriever.invoke("query", config={"test": "value"})

        mock_retriever.invoke.assert_called_once()

    def test_invoke_multiple_documents(self, mock_retriever):
        """invoke handles multiple documents correctly."""
        mock_retriever.invoke.return_value = [
            Document(page_content="Doc 1", metadata={"document_name": "a.pdf", "chunk_number": 1}),
            Document(page_content="Doc 2", metadata={"document_name": "b.pdf", "chunk_number": 2}),
        ]

        citation_retriever = _CitationRetriever(mock_retriever, include_citations=True)
        result = citation_retriever.invoke("query")

        assert len(result) == 2
        assert "[Source: a.pdf, Chunk 1]" in result[0].page_content
        assert "[Source: b.pdf, Chunk 2]" in result[1].page_content


class TestHRAssistantInit:
    """Tests for HRAssistant initialization."""

    @pytest.fixture
    def mock_dependencies(self):
        """Patch external dependencies for init."""
        with patch("src.assistant.hr_assistant.OpenAIEmbeddings"):
            with patch("src.assistant.hr_assistant.create_tool_calling_agent"):
                with patch("src.assistant.hr_assistant.AgentExecutor"):
                    with patch("src.assistant.hr_assistant.create_retriever_tool"):
                        yield

    def test_init_sets_attributes(self, mock_vector_store, mock_llm, mock_dependencies):
        """__init__ sets all attributes correctly."""
        assistant = HRAssistant(
            vector_store=mock_vector_store,
            llm=mock_llm,
            verbose=True,
            evidence=True
        )

        assert assistant.vectorstore is mock_vector_store
        assert assistant.llm is mock_llm
        assert assistant.verbose is True
        assert assistant.evidence is True

    def test_init_default_values(self, mock_vector_store, mock_llm, mock_dependencies):
        """__init__ uses correct default values."""
        assistant = HRAssistant(
            vector_store=mock_vector_store,
            llm=mock_llm
        )

        assert assistant.verbose is False
        assert assistant.evidence is False

    def test_init_creates_guard_chain(self, mock_vector_store, mock_llm, mock_dependencies):
        """__init__ creates intent guard chain."""
        assistant = HRAssistant(
            vector_store=mock_vector_store,
            llm=mock_llm
        )

        assert assistant.guard_chain is not None

    def test_init_creates_agent_executor(self, mock_vector_store, mock_llm, mock_dependencies):
        """__init__ creates agent executor."""
        assistant = HRAssistant(
            vector_store=mock_vector_store,
            llm=mock_llm
        )

        assert assistant.agent_executor is not None


class TestHRAssistantSetupAgent:
    """Tests for _setup_agent method."""

    def test_creates_retriever_tool(self, mock_vector_store, mock_llm):
        """_setup_agent creates hr_policy_search tool."""
        with patch("src.assistant.hr_assistant.OpenAIEmbeddings"):
            with patch("src.assistant.hr_assistant.create_retriever_tool") as mock_create_tool:
                with patch("src.assistant.hr_assistant.create_tool_calling_agent"):
                    with patch("src.assistant.hr_assistant.AgentExecutor"):
                        HRAssistant(mock_vector_store, mock_llm)

                        mock_create_tool.assert_called_once()
                        call_args = mock_create_tool.call_args
                        assert call_args[0][1] == "hr_policy_search"

    def test_evidence_mode_affects_retriever(self, mock_vector_store, mock_llm):
        """evidence=True creates CitationRetriever with citations enabled."""
        with patch("src.assistant.hr_assistant.OpenAIEmbeddings"):
            with patch("src.assistant.hr_assistant.create_retriever_tool"):
                with patch("src.assistant.hr_assistant.create_tool_calling_agent"):
                    with patch("src.assistant.hr_assistant.AgentExecutor"):
                        assistant = HRAssistant(mock_vector_store, mock_llm, evidence=True)
                        assert assistant.evidence is True


class TestHRAssistantRun:
    """Tests for run method."""

    @pytest.fixture
    def assistant_with_mocks(self, mock_vector_store, mock_llm):
        """Create assistant with mocked chains."""
        with patch("src.assistant.hr_assistant.OpenAIEmbeddings"):
            with patch("src.assistant.hr_assistant.create_tool_calling_agent"):
                with patch("src.assistant.hr_assistant.AgentExecutor") as mock_executor_cls:
                    with patch("src.assistant.hr_assistant.create_retriever_tool"):
                        mock_executor = MagicMock()
                        mock_executor.invoke.return_value = {"output": "HR policy response"}
                        mock_executor_cls.return_value = mock_executor

                        assistant = HRAssistant(mock_vector_store, mock_llm)
                        assistant.guard_chain = MagicMock()
                        assistant.guard_chain.invoke.return_value = "SAFE"

                        yield assistant

    def test_run_safe_query_returns_agent_response(self, assistant_with_mocks):
        """run returns agent response for safe HR queries."""
        response = assistant_with_mocks.run("What is the PTO policy?", [])
        assert response == "HR policy response"

    def test_run_unsafe_query_returns_rejection(self, assistant_with_mocks):
        """run returns rejection message for unsafe queries."""
        assistant_with_mocks.guard_chain.invoke.return_value = "UNSAFE - off topic"

        response = assistant_with_mocks.run("What is the weather today?", [])

        assert "I am sorry" in response

    def test_run_passes_user_input_to_guard(self, assistant_with_mocks):
        """run passes user input to guard chain."""
        assistant_with_mocks.run("PTO question", [])

        assistant_with_mocks.guard_chain.invoke.assert_called_once()
        call_args = assistant_with_mocks.guard_chain.invoke.call_args[0][0]
        assert call_args["user_input"] == "PTO question"

    def test_run_passes_history_to_agent(self, assistant_with_mocks):
        """run passes chat history to agent executor."""
        history = [{"role": "user", "content": "previous message"}]
        assistant_with_mocks.run("follow up", history)

        call_args = assistant_with_mocks.agent_executor.invoke.call_args[0][0]
        assert call_args["chat_history"] == history

    def test_run_handles_agent_exception(self, assistant_with_mocks):
        """run returns error message when agent raises exception."""
        assistant_with_mocks.agent_executor.invoke.side_effect = Exception("API error")

        response = assistant_with_mocks.run("valid question", [])

        assert "internal error occurred" in response
        assert "API error" in response

    def test_run_case_insensitive_unsafe_check(self, assistant_with_mocks):
        """run detects UNSAFE regardless of case."""
        assistant_with_mocks.guard_chain.invoke.return_value = "unsafe query detected"

        response = assistant_with_mocks.run("off topic question", [])

        assert "I am sorry" in response

    def test_run_partial_unsafe_string(self, assistant_with_mocks):
        """run detects UNSAFE within longer response."""
        assistant_with_mocks.guard_chain.invoke.return_value = "This is clearly UNSAFE content"

        response = assistant_with_mocks.run("bad question", [])

        assert "I am sorry" in response

    def test_run_safe_string_variations(self, assistant_with_mocks):
        """run allows queries when guard returns SAFE."""
        assistant_with_mocks.guard_chain.invoke.return_value = "SAFE"

        response = assistant_with_mocks.run("HR question", [])

        assert response == "HR policy response"

    def test_run_with_empty_history(self, assistant_with_mocks):
        """run works with empty history."""
        response = assistant_with_mocks.run("Question", [])

        call_args = assistant_with_mocks.agent_executor.invoke.call_args[0][0]
        assert call_args["chat_history"] == []
        assert response == "HR policy response"

from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from src.assistant import HRAssistant
from src.config import load_settings
from src.llm import get_llm
from src.stores import get_vector_store
from src.loader import Loader

load_dotenv()


def _content_to_str(content) -> str:
    """Extract text from ChatInterface content (str, list of dicts, or list of strings)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return " ".join(parts)
    return str(content) if content else ""


def _gradio_history_to_langchain(history: list) -> list:
    """Convert ChatInterface history to LangChain messages."""
    messages = []
    for msg in history or []:
        role = msg.get("role")
        content = _content_to_str(msg.get("content"))
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


class NestleApp:
    """Nestlé HR Policy chatbot."""

    def __init__(self):
        settings = load_settings()
        self.llm = get_llm(
            settings.provider_name,
            settings.provider_config.model,
            settings.provider_config.temperature,
            max_tokens=settings.provider_config.max_tokens,
        )
        embeddings = OpenAIEmbeddings()
        vector_store = get_vector_store(
            settings.store_name,
            settings.store_config.persist_directory,
            embeddings,
        )
        data_path = Path(__file__).resolve().parents[2] / "data"
        loader = Loader(file_path=str(data_path))
        documents = loader.load()
        self.store = vector_store.get_store(f"{settings.store_name}_hr_policies", documents)

    def chat(self, message: str, history: list, evidence: bool) -> str:
        """Handle chat messages."""
        message = _content_to_str(message) if not isinstance(message, str) else message
        if not message or not message.strip():
            return ""
        assistant = HRAssistant(self.store, self.llm, verbose=False, evidence=evidence)
        langchain_history = _gradio_history_to_langchain(history)
        return assistant.run(message, langchain_history)


def main():
    nestle_app = NestleApp()

    def respond(message, history, evidence):
        try:
            return nestle_app.chat(message, history, evidence)
        except Exception as e:
            return f"Error: {e}"

    # Setup the chat interface along with a button to show evidence
    demo = gr.ChatInterface(
        fn=respond,
        title="Nestlé HR Policy Assistant",
        description="Ask questions about Nestlé Human Resources policies.",
        additional_inputs=[
            gr.Checkbox(label="Show Evidence", value=False)
        ],
        additional_inputs_accordion="Advanced Settings",
        chatbot=gr.Chatbot(height=750),
        submit_btn="Submit",
    )

    demo.queue().launch()


def launch():
    """Entry point for poetry run gradio-app."""
    main()


if __name__ == "__main__":
    launch()

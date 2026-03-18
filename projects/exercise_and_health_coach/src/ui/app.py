import logging
import time

import gradio as gr

from src.config.settings import get_settings
from src.orchestrator.coach import ExerciseCoach
from src.state.conversation_state import ConversationState, create_session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize coach (singleton)
_coach: ExerciseCoach | None = None


def get_coach() -> ExerciseCoach:
    """Get or create the coach instance."""
    global _coach
    if _coach is None:
        settings = get_settings()
        _coach = ExerciseCoach(settings=settings)
    return _coach


def _normalize_message_content(content: str | list) -> str:
    """Extract plain text from Gradio message content (string or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else ""
    return str(content) if content is not None else ""


def respond(
    message: str | list,
    history: list[dict],
    state: dict | None,
) -> tuple[str, dict]:
    """
    Process user message and return response.

    Args:
        message: User's input message.
        history: Conversation history (not used, we maintain our own).
        state: Session state dictionary.

    Returns:
        Tuple of (response_message, updated_state).
    """
    # Initialize or restore session state
    if state is None or "session" not in state:
        session = create_session()
        state = {"session": session.model_dump()}
    else:
        session = ConversationState(**state["session"])

    # Process message (normalize in case Gradio passes content as list of blocks)
    message_text = _normalize_message_content(message)
    try:
        coach = get_coach()
        response, session = coach.process_message(message_text, session)

        # Format response (message already includes follow-up questions for intake prompts)
        response_text = response.message

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        response_text = (
            "I encountered an error processing your request. "
            "Please try again or rephrase your message."
        )

    # Persist updated state
    state["session"] = session.model_dump()

    return response_text, state


def create_demo() -> gr.Blocks:
    """Create the Gradio demo application."""
    with gr.Blocks(
        title="Exercise & Recovery Coach",
    ) as demo:
        gr.Markdown(
            """
            # Exercise & Recovery Coach

            Your AI-powered fitness coaching assistant. I can help you with:
            - **Workout Plans**: Customized exercise routines based on your goals
            - **Recovery Routines**: Mobility and stretching programs
            - **Safety Guidance**: Ensuring your training is safe and effective

            To get started, please share your **age** and any **health concerns**.
            """
        )

        # Session state
        state = gr.State(value=None)

        # Chat interface (Gradio 5+ expects [{"role": "user"|"assistant", "content": "..."}])
        chatbot = gr.Chatbot(
            label="Conversation",
            height=750,
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Your message",
                placeholder="Type your message here... (e.g., 'I want to build muscle')",
                lines=2,
                scale=4,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)

        # Example messages (starting with age/health as required for safety intake)
        gr.Examples(
            examples=[
                "I'm 30 years old with no health concerns.",
                "I'm 45, I have mild lower back pain from sitting at a desk.",
                "28 years old, no injuries or conditions.",
                "I'm 35, I have knee issues from an old injury.",
            ],
            inputs=msg,
            label="Start with your age and health info:",
        )

        # Clear and reset buttons
        with gr.Row():
            clear_btn = gr.Button("Clear Chat")
            reset_btn = gr.Button("New Session")

        # Latency display (always visible)
        latency_text = gr.Markdown("**Last response:** —")

        # Status display
        with gr.Accordion("Session Info", open=False):
            status_text = gr.Markdown("*No active session*")

        def handle_submit(user_msg: str, history: list, state_dict: dict | None):
            """
            Handle user message and generate bot response.

            Yields twice: first to show user message immediately, then with the
            assistant response when ready. This avoids the confusing delay where
            nothing appears until the full response is generated.
            """
            if not user_msg.strip():
                return "", history, state_dict, get_status_text(state_dict), gr.Skip()

            # Add user message (messages format: {role, content})
            history = history + [{"role": "user", "content": user_msg}]

            # Yield immediately so user sees their message right away
            yield "", history, state_dict, get_status_text(state_dict), "**Last response:** *Processing...*"

            # Process and get response (state must flow through in same call)
            start = time.perf_counter()
            user_text = _normalize_message_content(user_msg)
            response, new_state = respond(user_text, history[:-1], state_dict)
            elapsed = time.perf_counter() - start

            # Append assistant response and yield final result
            history = history + [{"role": "assistant", "content": response}]
            yield "", history, new_state, get_status_text(new_state), f"**Last response:** {elapsed:.1f}s"

        def clear_chat():
            """Clear chat history but keep session."""
            return [], None, "*Chat cleared*"

        def new_session():
            """Start a completely new session."""
            return [], None, "*New session started*"

        def get_status_text(state_dict: dict | None) -> str:
            """Get session status text."""
            if state_dict is None or "session" not in state_dict:
                return "*No active session*"

            try:
                session = ConversationState(**state_dict["session"])
                lines = [f"**Session:** `{session.session_id[:8]}...`"]
                safety_status = "Complete" if session.minimal_intake_complete else "Pending"
                lines.append(f"**Safety intake:** {safety_status}")
                full_status = "Complete" if session.intake_complete else "Pending"
                lines.append(f"**Full intake:** {full_status}")

                if session.user_context.biometrics.age:
                    lines.append(f"**Age:** {session.user_context.biometrics.age}")
                if session.user_context.fitness_goals:
                    goals = [g.value for g in session.user_context.fitness_goals]
                    lines.append(f"**Goals:** {', '.join(goals)}")
                if session.user_context.experience_level:
                    lines.append(f"**Experience:** {session.user_context.experience_level}")
                if session.user_context.medical_history.conditions:
                    conditions = ", ".join(session.user_context.medical_history.conditions)
                    lines.append(f"**Conditions:** {conditions}")

                return "\n".join(lines)
            except Exception:
                return "*Error reading session*"

        # Event handlers (single function ensures state persists between user msg and response)
        submit_outputs = [msg, chatbot, state, status_text, latency_text]
        msg.submit(handle_submit, [msg, chatbot, state], submit_outputs)
        submit_btn.click(handle_submit, [msg, chatbot, state], submit_outputs)

        def clear_chat_with_latency():
            result = clear_chat()
            return result[0], result[1], result[2], "**Last response:** —"

        def new_session_with_latency():
            result = new_session()
            return result[0], result[1], result[2], "**Last response:** —"

        clear_btn.click(
            clear_chat_with_latency,
            outputs=[chatbot, state, status_text, latency_text],
        )

        reset_btn.click(
            new_session_with_latency,
            outputs=[chatbot, state, status_text, latency_text],
        )

        gr.Markdown("---\n*Powered by Old Zin Software Corp*")

    return demo


def main() -> None:
    """Main entry point for Gradio app."""
    import os

    demo = create_demo()
    port_str = os.environ.get("GRADIO_SERVER_PORT")
    # Omit server_port to auto-find (7860-7959). Set GRADIO_SERVER_PORT for specific port.
    launch_kwargs: dict = {
        "server_name": "0.0.0.0",
        "share": False,
        "theme": gr.themes.Soft(),
    }
    if port_str:
        launch_kwargs["server_port"] = int(port_str)
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()

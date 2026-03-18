from dotenv import load_dotenv
import gradio as gr
from src.assistant.image_architect import ImageArchitectAgent


def main() -> None:
    architect = ImageArchitectAgent()

    with gr.Blocks(title="Image Designer Assistant", fill_height=True) as demo:
        gr.ChatInterface(
            fn=architect.generate_image,
            title="Image Designer Assistant",
            fill_height=True,
        )
        gr.Markdown("---\n*Powered by Old Zin Software Corp*")

    demo.launch()


def launch() -> None:
    # Load environment variables from .env file
    load_dotenv()

    main()


if __name__ == "__main__":
    launch()

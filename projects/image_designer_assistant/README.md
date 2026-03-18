# Image Designer Assistant

A Gradio-based conversational image generation application using LangChain agents and OpenAI DALL-E 3.

## Features

- **Conversational Image Generation**: Chat naturally to describe and refine images iteratively
- **High-Quality Output**: Generates 1024x1024 HD images using DALL-E 3
- **Design Refinement**: Maintains conversation history to refine designs across multiple turns
- **Native Image Display**: Images are validated with PIL, saved as PNG, and displayed directly in the chat via Gradio
- **Image Validation**: Uses PIL (Pillow) and BytesIO to validate image integrity before processing
- **Smart Response Handling**: Cleans up agent responses and extracts image URLs automatically
- **Copyright Awareness**: Gracefully handles content policy restrictions with alternative suggestions

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              User Browser                               │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Gradio ChatInterface                            │
│                           (src/gradio/app.py)                           │
│  • Web UI for chat interactions                                         │
│  • Manages message history                                              │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ImageArchitectAgent                             │
│                    (src/assistant/image_architect.py)                   │
│  • Converts Gradio messages to LangChain format                         │
│  • Cleans and formats responses                                         │
│  • Extracts image URLs, validates with PIL/BytesIO, saves as PNG        │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          LangChain Agent                                │
│  • Model: GPT-3.5 Turbo                                                 │
│  • Orchestrates conversation and tool usage                             │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       DALL-E 3 Image Generation                         │
│                    (OpenAIDALLEImageGenerationTool)                     │
│  • Generates 1024x1024 HD images                                        │
│  • Returns image URLs (fetched, validated via PIL, saved as PNG)        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
src/
├── assistant/
│   ├── __init__.py
│   └── image_architect.py
└── gradio/
    ├── __init__.py
    └── app.py
```

### Components

**ImageArchitectAgent** (`src/assistant/image_architect.py`)
- Wraps a LangChain agent configured with OpenAI GPT-3.5 Turbo
- Integrates DALL-E 3 as a tool for image generation
- Manages conversation history and message transformation
- Extracts image URLs from tool responses, downloads image data, then uses **PIL (Pillow)** and **BytesIO** to validate image integrity, convert to PNG, and return `gr.Image` for native display in the chat

**Gradio App** (`src/gradio/app.py`)
- Provides a web-based chat interface using Gradio's `ChatInterface`
- Handles environment configuration via `.env` file

## Prerequisites

- Python 3.13+
- [Poetry](https://python-poetry.org/) for dependency management
- OpenAI API key with DALL-E access

## Key Dependencies

- **Gradio** – Web chat interface
- **LangChain** – Agent orchestration and DALL-E tool integration
- **PIL (Pillow)** – Image validation and PNG conversion (via `Image.open(BytesIO(...))`)
- **BytesIO** – Wraps raw image bytes for PIL processing

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd image_designer_assistant
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY="your-api-key-here"
   ```

## Running the Application

Start the Gradio web interface:

```bash
poetry run gradio-app
```

Or run directly with Python:

```bash
poetry run python -m src.gradio.app
```

The application will launch and display a local URL (typically `http://127.0.0.1:7860`).

## Usage

1. Open the web interface in your browser
2. Describe the image you want to create (e.g., "Create a sunset over mountains with a cabin")
3. The assistant will generate the image, validate it with PIL, and display it directly in the chat
4. Continue the conversation to refine the design (e.g., "Add snow on the mountains")


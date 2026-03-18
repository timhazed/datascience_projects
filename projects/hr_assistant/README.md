# HR Assistant

A RAG based HR Assitant chatbot for answering questions about Nestlé Human Resources policies. Built with LangChain, supporting multiple LLM providers and vector stores.

## Features

- **RAG Architecture**: Loads HR policy documents, chunks them, and stores embeddings in a vector database for semantic search
- **Multiple LLM Providers**: Supports OpenAI and Groq
- **Multiple Vector Stores**: Supports FAISS and ChromaDB
- **Intent Guardrails**: LLM-as-a-Judge pattern to filter off-topic queries
- **Evidence Citations**: Optional source attribution showing which document chunks support the answer
- **Dual Interface**: CLI for testing and Gradio web UI for interactive use
- **Document Deduplication**: Automatic detection and skipping of already-indexed documents

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Intent Guard (LLM)                           │
│              Classifies query as SAFE or UNSAFE                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                 SAFE               UNSAFE
                    │                   │
                    ▼                   ▼
┌──────────────────────────┐   ┌──────────────────────┐
│   Vector Store Search    │   │   Rejection Message  │
│   (FAISS or Chroma)      │   └──────────────────────┘
└──────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Agent Executor                           │
│         Generates response with retrieved context               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Response (with optional citations)           │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
hr_assistant/
├── src/
│   ├── assistant/
│   │   └── hr_assistant.py      # Main HRAssistant class with guardrails
│   ├── config/
│   │   └── settings.py          # Pydantic configuration management
│   ├── llm/
│   │   └── llm_factory.py       # LLM provider factory (OpenAI/Groq)
│   ├── stores/
│   │   ├── vector_store_adapter.py  # Abstract base class
│   │   ├── faiss_vector_store.py    # FAISS implementation
│   │   ├── chroma_vector_store.py   # ChromaDB implementation
│   │   └── stores_factory.py        # Vector store factory
│   ├── loader/
│   │   └── loader.py            # Document loader (PDF/TXT)
│   ├── utils/
│   │   └── hash.py              # SHA-256 chunk ID generation
│   ├── cli/
│   │   └── app.py               # CLI entry point
│   └── gradio/
│       └── app.py               # Gradio web interface
├── tests/
│   ├── conftest.py              # Shared test fixtures
│   └── unit/                    # Unit tests
├── data/
│   ├── input/                   # Place HR policy documents here
│   ├── faiss/                   # FAISS index storage
│   └── chroma/                  # ChromaDB storage
├── config.yaml                  # LLM and store configuration
├── .env                         # API keys (not in git)
├── .env.example                 # Example environment file
└── pyproject.toml               # Project dependencies
```

## Installation

### Prerequisites

- Python 3.12+
- Poetry

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hr_assistant
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   GROQ_API_KEY=your_groq_key_here
   ```

4. **Configure LLM and vector store**

   Edit `config.yaml` to select your provider and store:
   ```yaml
   provider:
     # Uncomment ONE provider:

     # openai:
     #   model: "gpt-3.5-turbo"
     #   temperature: 0.1
     #   max_tokens: 512

     groq:
       model: "openai/gpt-oss-120b"
       temperature: 0.1
       max_tokens: 512

   store:
     # Uncomment ONE store:

     faiss:
       persist_directory: "data/faiss"

     # chroma:
     #   persist_directory: "data/chroma"
   ```

5. **Add HR policy documents**

   Place your PDF or TXT files in the `data/input/` directory.

## Usage

### Gradio Web Interface (Recommended)

Launch the interactive web UI:

```bash
poetry run gradio-app
```

Open your browser to `http://localhost:7860`

Features:
- Chat interface with conversation history
- Toggle "Evidence" checkbox to show source citations

## Examples chats ##

*Basic HR Question*

![Basic HR Question](images/HRAssistant_BasicQuestions.png)

*Basic HR Question with Evidence*

![Basic HR Question with Evidence](images/HRAssistant_QuestionWithEvidence.png)

*Invalid question not having to do with HR*

![Invalid question not having to do with HR](images/HRAssistant_QuestionActivatingGuard.png)


### CLI Application

Run a quick test query:

```bash
poetry run python -m src.cli.app
```

This runs a sample query and displays the response with evidence citations.

### Running Tests

Run all tests with coverage:

```bash
poetry run pytest --cov=src --cov-report=term-missing
```

Run specific test file:

```bash
poetry run pytest tests/unit/test_hr_assistant.py -v
```

Run with HTML coverage report:

```bash
poetry run pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## Configuration Options

### LLM Providers

| Provider | Models | Notes |
|----------|--------|-------|
| `openai` | gpt-3.5-turbo, gpt-4, gpt-4-turbo | Requires `OPENAI_API_KEY` |
| `groq` | llama-3.3-70b-versatile, mixtral-8x7b-32768 | Requires `GROQ_API_KEY` |

### Vector Stores

| Store | Description |
|-------|-------------|
| `faiss` | Facebook AI Similarity Search - fast, file-based |
| `chroma` | ChromaDB - persistent, supports metadata filtering |

### Provider Config Options

```yaml
provider:
  openai:
    model: "gpt-4"           # Model name
    temperature: 0.2         # 0.0-2.0, lower = more deterministic
    max_tokens: 512          # Maximum response length
```

## How It Works

### Document Processing

1. **Loading**: PDF and TXT files are loaded from `data/input/`
2. **Chunking**: Documents are split into chunks (default: 1000 chars, 200 overlap)
3. **Embedding**: Chunks are embedded using OpenAI embeddings
4. **Storage**: Embeddings are stored in the configured vector store
5. **Deduplication**: SHA-256 hashes prevent duplicate chunks

### Query Processing

1. **Intent Classification**: The guard chain classifies queries as SAFE or UNSAFE
2. **Retrieval**: For SAFE queries, relevant chunks are retrieved from the vector store
3. **Generation**: The LLM generates a response using retrieved context
4. **Citations**: If enabled, source documents and chunk numbers are included

### Intent Guardrails

The assistant only answers questions about:
- HR policies
- HR questions
- Nestlé company policies

Off-topic queries receive a polite rejection message.

## Example Queries

```
"What is the core spirit of the Nestlé Human Resources Policy?"
"What does Nestlé say about employee dialogue?"
"How does Nestlé approach collective bargaining?"
"What is the weather today?" (rejected - off-topic)
"Write me a poem" (rejected - off-topic)
```

## Development

### Code Quality

Format and lint with Ruff:

```bash
poetry run ruff check src/
poetry run ruff format src/
```

### Adding New LLM Providers

1. Add the provider client to `src/llm/llm_factory.py`
2. Add provider case to `get_llm()` function
3. Update `config.yaml` with provider options

### Adding New Vector Stores

1. Create new class extending `VectorStoreAdapter` in `src/stores/`
2. Implement `db_exists()`, `collection_exists()`, and `get_store()`
3. Add store case to `src/stores/stores_factory.py`


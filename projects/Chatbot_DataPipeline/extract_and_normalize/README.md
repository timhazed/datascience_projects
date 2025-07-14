# Document Processing and Metadata Extraction for Sustainable Agriculture

A robust Python application that extracts text and metadata from both PDF documents and HTML webpages, generating standardized payloads for Qdrant vector database indexing. Focused on sustainable agriculture documents, it combines text extraction, LLM-powered metadata generation, and consistent output formatting.

## Overview

This application processes documents from URLs through the following pipeline:

1. **Document Download & Text Extraction**: Downloads PDFs or scrapes HTML content
2. **Section Identification**: Intelligently identifies logical sections and titles within the document
3. **Text Cleaning & Normalization**: Standardizes text formatting for better processing
4. **Metadata Generation**: Uses LLMs to extract document metadata (topics, audience, etc.)
5. **Payload Construction**: Builds a standardized payload for Qdrant vector database integration
6. **Output Generation**: Saves processed text and metadata to the filesystem

## Components

### Core Modules

- **extract_and_normalize_pdf.py**: Processing pipeline for PDF documents
- **extract_and_normalize_html.py**: Processing pipeline for HTML webpages
- **config/config.yaml**: Configuration for LLM integration, output paths, and extraction parameters
- **orchestrator.py**: Coordinates the pipeline for integration with Prefect workflows

### Document Processing (`pdf/` and `html_text/`)

- **PDF Processing Modules**
  - PDF downloading and content extraction
  - Section and title detection
  - PDF-specific text normalization
  - Support for different sources (arXiv, ResearchGate, direct URLs)

- **HTML Processing Modules**
  - Web page scraping with JavaScript support
  - Content extraction from complex HTML structures
  - Section and title detection from HTML elements
  - Deduplication and cleaning of HTML content
  - Enhanced handling for academic and research sites

### Common Utilities (`common/`)

- **DocumentUtils**: Core document processing functionality
  - Text extraction and cleaning
  - Section identification
  - File and directory management
  - Metadata payload construction
  - Organization name inference from URLs

- **MetadataGenerator**: LLM-powered metadata extraction
  - Integration with multiple LLM providers (OpenAI, Groq)
  - Structured metadata generation with schema validation
  - Fallback mechanisms for missing information
  - Retry logic for API resilience
  - Configurable rate limiting to prevent API throttling

- **RateLimiter**: API call rate limiting
  - Thread-safe implementation for concurrent processing
  - Configurable calls-per-minute threshold
  - Token-based rate limiting to manage usage

- **ResultsManager**: Standardized result handling
  - Centralized output file generation
  - Consistent error reporting
  - Support for integration with workflow engines

## Inputs

- **Document URL**: The application takes a URL to a PDF document or HTML webpage as its primary input
- **Configuration**: Optional path to a custom configuration file
- **Result File**: Optional path for writing structured results for workflow integration

### Command-line Usage

For PDF documents:
```bash
python extract_and_normalize_pdf.py --url https://example.org/path/to/document.pdf --config config/config.yaml --result-file results/output.json
```

For HTML webpages:
```bash
python extract_and_normalize_html.py --url https://example.org/path/to/webpage --config config/config.yaml --result-file results/output.json
```

### Configuration Options

The application supports multiple LLM providers through configuration settings, with API keys sourced from environment variables for security:

#### OpenAI Configuration

```yaml
llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  temperature: 0.3
```

Required environment variable: `OPENAI_API_KEY`

#### Groq Configuration

```yaml
llm:
  provider: "groq"
  model: "llama3-70b-8192"  # Alternatives: "llama3-8b-8192", "mixtral-8x7b-32768"
  temperature: 0.1  # Lower temperature for more consistent metadata extraction
  rate_limit:
    calls_per_minute: 60      # Maximum API calls per minute (0 = no limit)
    tokens_per_minute: 100000 # Maximum tokens per minute (0 = no limit)
```

Required environment variable: `GROQ_API_KEY`

#### Environment Variables

API keys must be configured as environment variables for security. These can be set directly in your environment or through the `.env` file specified in the config:

```
# Example .env file
OPENAI_API_KEY=sk-...your-key-here...
GROQ_API_KEY=gsk_...your-key-here...
ELSEVIER_API_KEY=...your-key-here...
```

The path to this `.env` file is configured with:

```yaml
env_path: "~/src/python/.env"
```

#### Obtaining an Elsevier API Key

To use the Elsevier/ScienceDirect integration, you need to obtain an API key:

1. Visit the [Elsevier Developer Portal](https://dev.elsevier.com/apikey/manage)
2. Request an API key (select "I want an API Key")
3. Complete the registration form with your institutional information

Eligibility:
- **Non-Commercial Users**: Researchers in academic institutions, public sector, and non-profit organizations can access most APIs for free for non-commercial use
- **Commercial Users**: Researchers in private sector or commercial institutions need an API license and subscription

Once obtained, add the API key to your environment variables as `ELSEVIER_API_KEY`.

Note that full API access is only granted to researchers affiliated with organizations that have subscriptions to the corresponding Elsevier products.

#### Current Configuration (config.yaml)

The current configuration uses:

```yaml
env_path:  "~/src/datascience/omdena/.env"
llm_document_chunk_size: 4096

schema_path: "schemas/base_payload.schema.json"

llm:
  provider: groq
  model: llama3-70b-8192
  temperature: 0.1
  rate_limit:
    calls_per_minute: 60      # Maximum API calls per minute (0 = no limit)
    tokens_per_minute: 100000 # Maximum tokens per minute (0 = no limit)

section_title_filter:
  ignore_if_contains:
    - "table of contents"
    - "page"
    - "framework"
    - "figure"
    - "reference"

pdf_extractor:
  headless_domains:
    - "mdpi.com"

# HTML-specific configuration
html_parameters:
  topic_page:
    title_selector: "h1, .article-title, .title, header h1"
    content_selectors: 
      - "article, .content, .article-content, main"
    subtopics_selector: "h2, h3"
    block_elements: ["p", "div", "section", "article", "blockquote"]
    list_elements: ["ul", "ol"]
  rendering:
    wait_time: 3000
    scroll: true
    js_patterns: [".gov", "javascript", "dynamic"]
  rate_limit:
    enabled: true
    requests_per_second: 1
    delay: 2
  headers:
    User-Agent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
  scraping_rate_limit:
    requests_per_second: 1 # Maximum requests per second
    delay: 1.0            # Minimum seconds between requests
```

Additionally, detailed HTML parameters are configured for improved web page processing.

#### Elsevier API Integration

The application now supports downloading PDFs from Elsevier's ScienceDirect using their official API:

```yaml
# No specific config.yaml settings needed, just environment variable
```

Required environment variable: `ELSEVIER_API_KEY`

This integration:
- Automatically extracts PIIs from ScienceDirect URLs
- Uses the Elsevier Content API to fetch article PDFs
- Handles authentication via API key

## Outputs

The application generates several outputs in the configured output directory:

### Text Files

- **Page-level text files (PDF)**: `{base_filename}_page{page_number}.txt`
  - Contains the cleaned text content of each page
  - Includes the section title as a header

- **Content text files (HTML)**: `{base_filename}_page1.txt`
  - Contains the cleaned text content from the webpage
  - Includes the page title as a header

### Metadata

- **Base payload JSON**: `{base_filename}_base_payload.json`
  - Contains all extracted metadata including:
    - Document title, language, region/country
    - Publication date (extracted or inferred)
    - Document type (scientific, policy, etc.)
    - Sustainability dimensions (environmental, economic, social)
    - Key topics (agricultural innovation, soil health, etc.)
    - Intended audience (farmers, policymakers, researchers, NGOs)
    - Source organization
    - Document identifier (stable UUID derived from URL)

### Results File (for Workflow Integration)

- **JSON results file**: Path specified by `--result-file` parameter
  - Status of processing (success/failure)
  - Source URL
  - Output paths for generated files
  - Error information if processing failed

## Architecture Details

### Academic Document Sources

The application supports multiple academic document sources:
- **arXiv**: Direct PDF retrieval using the arXiv API
- **ResearchGate**: Intelligent extraction of PDFs from publication pages
- **ScienceDirect/Elsevier**: Direct PDF retrieval using the Elsevier Content API
- **MDPI**: Headless browser extraction for MDPI journal articles
- **General PDFs**: Direct download from any accessible URL

### Direct PDF URL Handling

The application includes special handling for direct PDF URLs:
- **Auto-detection**: Automatically identifies direct PDF links (URLs ending with .pdf)
- **Smart referer**: Infers and uses appropriate referer headers to bypass access restrictions
- **Content-type validation**: Verifies downloaded content is actually a PDF
- **Local file support**: Processes local PDF files with the same pipeline as remote URLs
- **Temporary file management**: Securely handles downloaded PDFs in temporary storage

### Error Handling

The application implements robust error handling throughout:
- Document processing errors are caught, allowing processing to continue
- LLM API calls include retry logic with exponential backoff
- HTML scraping includes fallbacks for JavaScript-rendered content
- All operations are wrapped in try/except blocks with detailed logging
- Standardized error reporting through the ResultsManager

### Metadata Schema Validation

All generated metadata is validated against a JSON schema to ensure consistency and completeness. The metadata generator includes:
- Structured prompt templates optimized for different LLM providers
- Explicit field validation and sanitization
- Temperature optimization for precise metadata extraction

### Text Processing

The text cleaning process:
1. Removes control characters and form feeds
2. Normalizes whitespace and newlines
3. Preserves paragraph structure with double line breaks
4. Handles Unicode and special characters appropriately
5. For HTML, provides deduplication of content to handle repeated elements

### HTML Processing Features

The HTML processor includes:
- Support for JavaScript-rendered websites using Playwright
- Intelligent content extraction with configurable selectors
- Enhanced academic site handling with specialized configurations
- Cookie consent dialog handling for improved access
- Handling of dynamic content loaded through scrolling
- Clean extraction of structured content (lists, headers, paragraphs)
- Rate limiting and polite scraping practices

### LLM Provider Abstraction

The application supports multiple LLM providers through a provider-agnostic interface:
- Provider-specific client initialization
- Consistent prompt templates across providers
- Unified error handling and retry logic
- **Rate limiting** to prevent API throttling and reduce costs
- Configuration-driven provider selection
- Currently supported providers:
  - OpenAI (GPT models)
  - Groq (Llama and Mixtral models)

## Dependencies

- **PDF Processing**: PyMuPDF (fitz), arxiv
- **HTML Processing**: BeautifulSoup4, Playwright
- **LLM Integration**: OpenAI and Groq API clients
- **Data Validation**: jsonschema
- **Resilience**: tenacity for retry logic
- **HTTP Operations**: requests
- **Configuration**: PyYAML, python-dotenv

## Environment Setup

The project supports both Conda and pip for dependency management:

### Option 1: Using Conda

The project includes an `environment.yaml` file for Conda-based setup:

1. Ensure you have [Conda](https://docs.conda.io/en/latest/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
2. Clone this repository
3. Create and activate the environment with:

```bash
conda env create -f environment.yaml
conda activate extract-and-normalize
```

### Option 2: Using pip

Alternatively, you can use pip with the provided `requirements.txt` file:

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Additional Setup

If processing HTML with JavaScript, initialize Playwright browsers:

```bash
playwright install chromium
```

These methods will install all required dependencies including:
- Python 3.10 (required version)
- Core packages: requests, urllib3, PyYAML, nltk
- HTML parsing: BeautifulSoup4, Playwright
- LLM clients: openai, groq
- Support libraries: python-dotenv, PyMuPDF, jsonschema, tenacity, arxiv, validators

## Docker Usage

The application can be containerized using Docker, providing a consistent and isolated runtime environment without the need for local dependency management.

### Building the Docker Container

To build the Docker container:

```bash
# Normal build
docker build -t extract-normalize .

# Build without using cache (for troubleshooting)
docker build --no-cache -t extract-normalize .
```

The Dockerfile includes:
- Python 3.10 base image
- All required dependencies from requirements.txt
- NLTK data packages (punkt, stopwords, etc.)
- Playwright with Chromium for HTML processing
- Volume configuration for input/output directories

### Running the Application with Docker

You can run the application using Docker, mounting host directories for input/output and providing environment variables for API keys:

```bash
# Processing a PDF from a URL
docker run \
  --env-file ~/path/to/.env \
  -v /path/to/host/output:/app/output \
  -e PYTHONUNBUFFERED=1 \
  extract-normalize \
  --url "https://example.org/document.pdf" \
  --output "/app/output" \
  --config "config/config.yaml"

# Processing a local PDF file
docker run \
  --env-file ~/path/to/.env \
  -v /path/to/host/input:/app/input \
  -v /path/to/host/output:/app/output \
  -e PYTHONUNBUFFERED=1 \
  extract-normalize \
  --url "/app/input/document.pdf" \
  --output "/app/output" \
  --config "config/config.yaml"
```

### Docker Command Explanation

- `--env-file`: Path to your .env file containing API keys
- `-v /path/to/host/input:/app/input`: Mount a local directory with input files
- `-v /path/to/host/output:/app/output`: Mount a local directory for output files
- `-e PYTHONUNBUFFERED=1`: Ensure Python doesn't buffer stdout/stderr (shows logs in real-time)
- `extract-normalize`: The Docker image name
- `--url`, `--output`, `--config`: Standard application parameters

### Environment Variables

When running with Docker, you need to provide environment variables for API keys. You can either:

1. Use an environment file with `--env-file`
2. Pass individual variables with `-e KEY=VALUE`

Example .env file content:
```
OPENAI_API_KEY=sk-...your-key-here...
GROQ_API_KEY=gsk_...your-key-here...
ELSEVIER_API_KEY=...your-key-here...
```

## Extending the Application

### Adding New LLM Providers

To add support for a new LLM provider:
1. Update the `_initialize_llm_client` method in `MetadataGenerator`
2. Add provider-specific logic to the `_call_llm` method
3. Update the configuration template with provider-specific options

### Supporting Different Document Types

To adapt the application for different document types:
1. Modify the schema in `schemas/` directory
2. Update the prompt template in `MetadataGenerator._build_prompt()`
3. Adjust the document processing parameters in `config.yaml`

### Adding Academic Document Sources

To add support for new academic sources:
1. Add a source-specific method in the `PdfManager` class
2. Update the `_get_url` method to detect and route to the new source handler
3. Implement appropriate authentication and download logic

### Adding New HTML Selectors

To improve HTML content extraction for specific websites:
1. Update the `content_selectors` list in the configuration
2. Add specific selectors for different website structures
3. Add site-specific configurations if needed for academic or complex websites
4. Test with a variety of target websites to ensure robust extraction

### Rate Limiting

The application implements two separate rate limiting systems:

1. **LLM API Rate Limiting**:
   - Prevents excessive calls to language model APIs
   - Controls both request frequency and token usage
   - Configured in the `llm.rate_limit` section:
     ```yaml
     rate_limit:
       calls_per_minute: 60      # Request-based limit (0 = no limit)
       tokens_per_minute: 100000 # Token-based limit (0 = no limit)
     ```

2. **Web Scraping Rate Limiting**:
   - Prevents blocking by websites when scraping content
   - Implements polite crawling practices
   - Configured in the `html_parameters.rate_limit` section:
     ```yaml
     rate_limit:
       enabled: true
       requests_per_second: 1 # Maximum requests per second
       delay: 1.0             # Minimum seconds between requests
     ```

Both systems use thread-safe implementations to ensure consistent behavior in concurrent processing environments.


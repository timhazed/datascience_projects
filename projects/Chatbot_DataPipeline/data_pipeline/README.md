# Sustainable Agriculture Chatbot - Data Pipeline

A robust ETL pipeline for processing documents from various sources for the Sustainable Agriculture Chatbot. This system extracts content from URLs or files, chunks the content into manageable pieces, and inserts the data into a vector database for retrieval.

## Overview

The pipeline is designed with a modular architecture that allows for flexible configuration and separate execution of each stage. It uses Docker containers to run specialized tools for extraction, chunking, and database insertion, ensuring clear separation of concerns and isolation.

Key features:
- Extracts and normalizes content from various sources (URLs, PDF files)
- Chunks extracted content into appropriate sizes for embedding/storage
- Stores chunks in a vector database for efficient retrieval
- Configurable workflow with selective enabling/disabling of pipeline stages
- Advanced error recovery with exponential backoff retries
- Docker-based execution for isolation and reproducibility
- Prefect for workflow orchestration

## Pipeline Flow

The pipeline consists of three main stages, each of which can be selectively enabled or disabled:

1. **Extraction** (`Extractor`): 
   - Inputs: URLs or file paths from a CSV file
   - Processing: Extracts and normalizes content using a Docker-based tool
   - Output: Structured documents in the `/extracted` directory

2. **Chunking** (`Chunker`):
   - Inputs: Extracted documents from the extraction phase
   - Processing: Splits content into smaller, manageable chunks for embedding
   - Output: Chunked documents in the `/chunked` directory 

3. **Database Insertion** (`InsertDB`):
   - Inputs: Chunked documents from the chunking phase
   - Processing: Inserts chunks into a vector database with metadata
   - Output: Stored vector embeddings in the database

Each stage passes its output to the next stage. If a stage is disabled, the pipeline will skip that stage and proceed to the next enabled stage (if any).

## Architecture

The application follows a pipeline architecture with these key components:

- **Orchestrator**: Controls the execution flow of the pipeline, manages component lifecycle
- **ConfigManager**: Centralizes configuration management with a clear API
- **ToolExecutor**: Executes Docker containers for various tools
- **Components**: Specialized classes for each pipeline stage (Extractor, Chunker, InsertDB)

All components use dependency injection for loggers and tool executors, making the code more testable and maintainable.

## Configuration

The pipeline is configured using YAML files (default: `config/pipeline_config.yaml`). The configuration includes:

### Tool Configuration

Each tool can be enabled/disabled and has its own Docker configuration:

```yaml
tools:
  extractor:
    enabled: true
    docker:
      image: "extract-normalize:latest"
      input_volume: "/path/to/input:/app/input"
      output_volume: "/path/to/output:/app/output"
      env_file: "/path/to/.env_file"
      config_path: "/app/config/config.yaml"
  
  chunker:
    enabled: false
    # ...configuration...
  
  insert_db:
    enabled: false
    # ...configuration...
```

### Timeouts

Configure timeout values for different operations:

```yaml
timeouts:
  extraction: 600  # 10 minutes
  chunking: 300    # 5 minutes
  db_insertion: 600  # 10 minutes
```

## Running the Pipeline

The pipeline can be executed via command line:

```bash
python orchestrator.py --csv input.csv --config config/pipeline_config.yaml
```

### Parameters:
- `--csv`: Path to CSV file containing URLs to process
- `--config`: (Optional) Path to configuration file (defaults to `config/pipeline_config.yaml`)

## Error Handling

The pipeline implements error handling through:

1. **Built-in Error Handling**: 
   - Components have error handling for their specific operations
   - Failures are properly logged with detailed error information
   - Each component gracefully handles failures and continues processing where possible
   
2. **Prefect Task Retry**:
   - The `ToolExecutor` implements basic retry for transient errors
   - Docker execution issues are retried automatically when possible
   - Prefect provides visibility into task success/failure

3. **Isolation through Docker**:
   - Each processing stage runs in an isolated Docker container
   - Failures in one container don't affect the stability of the pipeline
   - Resource limitations and timeouts are enforced by container configuration

## Logging

The pipeline implements a comprehensive logging system:

1. **Centralized Logger**: 
   - A single logger instance is created in the main flow and passed to all components
   - Uses dependency injection for consistent logging across the application
   - All components use the same logging format and level

2. **Tool Execution Logs**:
   - Each Docker tool execution produces a separate log file
   - Log files are stored in `logs/{tool_name}/` directories
   - Human-readable naming convention: `{tool_name}_{doc_id}_{YYYYMMDD-HHMMSS}.log`
   - Captures both stdout and stderr from Docker containers

3. **Prefect Flow Logs**:
   - The pipeline runs as a Prefect flow, providing additional logging
   - Task-level logging with status information
   - Can be viewed through Prefect UI when running with a Prefect server

4. **Logging Levels**:
   - INFO: Normal operation, pipeline progress
   - WARNING: Non-critical issues that don't stop processing
   - ERROR: Issues that prevent processing of specific documents
   - EXCEPTION: Unexpected errors with full stack traces

To troubleshoot issues:
1. Check Prefect flow logs for high-level pipeline execution
2. Examine tool-specific logs in the `logs/{tool_name}/` directories
3. Look for ERROR or WARNING level messages indicating specific issues

## Assumptions

1. **Docker Environment**:
   - Docker is installed and running on the host system
   - The required Docker images are available (`extract-normalize`, `document-chunker`, `db-inserter`)
   - Docker images have been pre-built and are ready to use
   - User has permissions to run Docker commands

2. **File Structure**:
   - Input CSV file contains valid URLs or file paths in the first column
   - Output directory structure follows the pattern established in configuration

3. **Configuration**:
   - Volume mappings in Docker configuration are correctly set up
   - Environment files contain necessary variables for each tool
   - Network connectivity is available for remote URLs

4. **Vector Database**:
   - The vector database API is accessible from the host machine
   - Proper credentials are configured in the environment files

## Docker Integration with Prefect

This pipeline implementation assumes Docker images are pre-built and available in the local Docker environment. It does not include any of the following image building capabilities:

- No dynamic image building
- No infrastructure as code integration
- No Docker image registry integration
- No docker-compose integration

The current implementation simply references the images by name in the configuration file:

```yaml
tools:
  extractor:
    docker:
      image: "extract-normalize:latest"
      # ...
```

If images are not pre-built, the pipeline will fail with Docker image not found errors. Building the necessary Docker images must be done manually before running the pipeline:

```bash
# Example commands to build required images
docker build -t extract-normalize:latest ./docker/extractor
docker build -t document-chunker:latest ./docker/chunker
docker build -t db-inserter:latest ./docker/db-inserter
```

For information on other approaches to Docker integration with Prefect (not implemented in this version), see the Prefect documentation on containerization.

## Development

For extending or modifying the pipeline:

1. Follow the established dependency injection pattern
2. Use the ConfigManager's facade methods for configuration access
3. Implement appropriate error handling with task retries
4. Maintain separation of concerns between components

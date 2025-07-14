# Sustainable Agriculture Chatbot - Backend

## Overview
This application provides a robust backend for a sustainable agriculture chatbot, leveraging Qdrant for vector storage and semantic search capabilities. The system processes agricultural documents, generates embeddings, and enables efficient querying of relevant information.

## Architecture

### Core Components

1. **QdrantManager**
   - Manages connection and operations with Qdrant vector database
   - Handles document ingestion and embedding generation
   - Implements batch processing for efficient data insertion
   - Provides error handling and logging

2. **Vector Database (Qdrant)**
   - Stores document embeddings and metadata
   - Enables semantic search capabilities
   - Supports batch operations for efficient data management

3. **Embedding Generation**
   - Uses SentenceTransformers for generating document embeddings
   - Configurable embedding model and dimensions
   - Efficient batch processing of text content

## Features

### Document Processing
- Batch processing of document chunks
- Automatic embedding generation
- Metadata preservation and management
- Configurable batch sizes for optimal performance

### Error Handling
- Comprehensive error logging
- Batch-level failure tracking
- Document and chunk-level error reporting
- Graceful failure handling with detailed error information

### Configuration
The application is configured through a dictionary containing:
```python
config = {
    "qdrant_collection": "collection_name",
    "embedding_model": "model_name",
    "embedding_dimension": 768,  # or appropriate dimension
    "qdrant_batch_size": 200,    # optional, defaults to 200
    "qdrant_url": "http://localhost:6333"
}
```

Environment Variables:
- `QDRANT_API_KEY`: API key for Qdrant authentication

## Data Flow

1. **Document Ingestion**
   ```
   Document → Chunk Processing → Embedding Generation → Batch Creation → Qdrant Storage
   ```

2. **Batch Processing**
   - Documents are split into chunks
   - Each chunk is processed and embedded
   - Chunks are batched according to configured size
   - Batches are sent to Qdrant for storage

3. **Error Handling Flow**
   ```
   Batch Failure → Error Logging → Process Termination → Error Reporting
   ```

## Usage

### Initialization
```python
from qdrant_manager import QdrantManager
import logging

logger = logging.getLogger(__name__)
config = {
    "qdrant_collection": "agriculture_data",
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dimension": 384
}

manager = QdrantManager(config, logger)
```

### Adding Documents
```python
documents = [
    {
        "doc_id": "doc1",
        "chunk_id": "chunk1",
        "text": "Document content...",
        "metadata": {...}
    }
]

metrics = manager.add_documents(documents)
```

## Dockerization

### Building the Docker Image
```bash
# Build with no cache for clean build
docker build --no-cache -t insert-db .

# Build with cache (faster subsequent builds)
docker build -t insert-db .
```

### Running the Container
```bash
docker run \
  --env-file ~/path/to/.env \
  -v /path/to/input:/app/input \
  -e PYTHONUNBUFFERED=1 \
  insert-db \
  --input "/app/input/chunked/doc_XXXX"
```

### Environment Variables
- Mount your environment file containing Qdrant credentials
- Set `PYTHONUNBUFFERED=1` for real-time logging
- Volume mount your input directory to `/app/input`

### Data Pipeline Integration
The application is designed to be part of a larger data pipeline:
1. Documents are pre-processed and chunked
2. Chunks are placed in the input directory
3. Docker container processes chunks and inserts into Qdrant
4. Metrics and logs are output for pipeline monitoring

Example pipeline structure:
```
input/
  chunked/
    doc_0001/
    doc_0002/
    ...
```

## Error Handling

The system implements a robust error handling strategy:
- Batch-level errors are logged with document and chunk IDs
- Processing stops on first failure to maintain data consistency
- Detailed error metrics are returned for debugging
- No partial document insertion is allowed

## Performance Considerations

1. **Batch Sizing**
   - Default batch size: 200
   - Configurable based on system resources
   - Optimized for memory usage and processing speed

2. **Embedding Generation**
   - Uses efficient sentence transformers
   - Batch processing for optimal performance
   - Configurable model selection

3. **Qdrant Operations**
   - Atomic batch operations
   - Wait for completion to ensure consistency
   - Progress tracking for monitoring

## Best Practices

1. **Configuration**
   - Set appropriate batch sizes based on system resources
   - Choose embedding model based on accuracy/performance needs
   - Configure logging level appropriately

2. **Error Handling**
   - Monitor error logs for batch failures
   - Use document and chunk IDs for debugging
   - Implement appropriate retry strategies at the application level

3. **Performance**
   - Monitor batch processing times
   - Adjust batch sizes based on system performance
   - Consider memory usage when processing large documents

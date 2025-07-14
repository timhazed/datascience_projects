FROM python:3.10-slim

RUN mkdir -p /app
WORKDIR /app
COPY . /app
COPY config/ ./config/
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/output

# Set volume for input
VOLUME ["/app/input"]

# Set volume for output
VOLUME ["/app/output"]

# Set entrypoint
ENTRYPOINT ["python", "create_chunks.py"]

# Default command (can be overridden)
CMD ["--chunk_size", "256", "--chunker_technique", "overlap"]

# Usage:
# docker build -t document-chunker .
# docker run --rm -v /[absolute_path_to]/input:/app/input -v /[absolute_path_to]/output:/app/output document-chunker
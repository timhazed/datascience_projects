FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python source files
COPY qdrant_insert.py .
COPY qdrant_manager.py .
COPY config/ ./config/

# Set volume for input
VOLUME ["/app/input"]

ENTRYPOINT ["python", "qdrant_insert.py"]

# Default command (can be overridden)
CMD ["--insert", "app/input/doc_0001"]
FROM python:3.10-slim

WORKDIR /app

# Copy application files
COPY requirements.txt .
COPY extract_and_normalize.py .
COPY extract_and_normalize_pdf.py .
COPY extract_and_normalize_html.py .
COPY common/ ./common/
COPY config/ ./config/
COPY schemas/ ./schemas/
COPY pdf/ ./pdf/
COPY html_text/ ./html_text/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data - add punkt_tab specifically
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# Install Playwright browsers
RUN pip install playwright && python -m playwright install --with-deps chromium

# Create output directory
RUN mkdir -p /app/output

# Set volume for output
VOLUME ["/app/output"]

# Set entrypoint
ENTRYPOINT ["python", "extract_and_normalize.py"]

# Default command (can be overridden)
CMD ["--url", "https://example.com", "--output", "/app/output"]

# Usage:
# docker build -t extract-normalize .
# docker run -v /Users/timhazed/tmp/output:/app/output extract-normalize --url "https://example.com" 
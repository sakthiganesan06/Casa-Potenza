# ============================================================
# Production Dockerfile for Voice RAG Web Application
# ============================================================
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install system audio and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create necessary runtime directories
RUN mkdir -p logs lance_db data_cache

# Expose ports (7860 for Hugging Face Spaces, 8000 for standard Docker)
EXPOSE 7860 8000

# Run uvicorn server binding to 0.0.0.0 and $PORT
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]

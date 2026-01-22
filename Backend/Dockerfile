FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY Backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir uvicorn[standard] fastapi && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY Backend/ ./Backend/

# Create necessary directories
RUN mkdir -p /app/uploads && \
    mkdir -p /app/Backend/models && \
    chmod -R 755 /app/uploads

# Set environment variables
ENV PYTHONPATH=/app/Backend/src:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check (inline, simple)
HEALTHCHECK --interval=10s --timeout=5s --retries=20 --start-period=180s \
    CMD curl -f http://localhost:8000/health || exit 1

# Start uvicorn directly - NO TEST SCRIPT!
CMD ["python", "-m", "uvicorn", "src.DietPlanRAG.DietGuide:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
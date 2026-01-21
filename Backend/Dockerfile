FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY Backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir uvicorn[standard] fastapi && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY Backend/ ./Backend/

# Create uploads directory with proper permissions
RUN mkdir -p /app/uploads && \
    chmod -R 755 /app/uploads

# Set Python path
ENV PYTHONPATH=/app/Backend/src:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.DietPlanRAG.DietGuide:app", "--host", "0.0.0.0", "--port", "8000"]
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

# Set Python path
ENV PYTHONPATH=/app/Backend/src:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Create a test script to verify imports
RUN echo 'import sys; sys.path.insert(0, "/app/Backend/src"); print("Python path:", sys.path); try: from DietPlanRAG.DietGuide import app; print("✅ Import successful"); except Exception as e: print(f"❌ Import failed: {e}"); import traceback; traceback.print_exc()' > /app/test_import.py

EXPOSE 8000

# Test imports before starting
CMD ["sh", "-c", "python /app/test_import.py && python -m uvicorn Backend.src.DietPlanRAG.DietGuide:app --host 0.0.0.0 --port 8000"]
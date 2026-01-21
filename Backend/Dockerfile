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
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir uvicorn[standard] fastapi && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set Python path AFTER copying files
ENV PYTHONPATH=/app:$PYTHONPATH

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.DietPlanRAG.DietGuide:app", "--host", "0.0.0.0", "--port", "8000"]
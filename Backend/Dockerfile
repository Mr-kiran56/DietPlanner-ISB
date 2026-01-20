FROM python:3.11-slim

# Set working directory to project root
WORKDIR /app

# System deps (minimal but safe)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Expose port
EXPOSE 8000

# BULLETPROOF START COMMAND
CMD ["python", "-m", "uvicorn", "src.DietPlanRAG.DietGuide:app", "--host", "0.0.0.0", "--port", "8000"]

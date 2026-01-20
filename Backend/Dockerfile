# Base image
FROM python:3.11-slim

WORKDIR /app/Backend

# Install system dependencies for ML, OpenCV, dlib, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libopenblas-dev \
    liblapack-dev \
    poppler-utils \
    python3-dev \
    git \
    wget \
    apt-transport-https \
    tesseract-ocr \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . ./

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "DietPlanRAG.DietGuide:app", "--host", "0.0.0.0", "--port", "8000"]
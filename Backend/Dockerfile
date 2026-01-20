# Base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app/Backend

# Install system dependencies needed for ML, OpenCV, dlib, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest
RUN pip install --upgrade pip

# Copy requirements.txt and backend code
COPY requirements.txt ./
COPY . ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "DietPlanRAG.DietGuide:app", "--host", "0.0.0.0", "--port", "8000"]

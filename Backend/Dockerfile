FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
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

# Upgrade pip
RUN pip install --upgrade pip

# Copy backend code and requirements
COPY requirements.txt .
COPY Backend/ ./Backend
COPY short_answer_rag ./short_answer_rag

# Replace editable path in requirements
RUN sed -i 's#-e b:\\users\\kiran\\downloads2\\short_answer_rag#-e ./short_answer_rag#g' requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory to backend
WORKDIR /app/Backend

EXPOSE 8000

# Start app
CMD ["uvicorn", "DietPlanRAG.DietGuide:app", "--host", "0.0.0.0", "--port", "8000"]

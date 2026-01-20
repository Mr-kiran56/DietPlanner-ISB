FROM python:3.11-slim

WORKDIR /app

# Set Python path to include current directory
ENV PYTHONPATH=/app:$PYTHONPATH

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Try with full path
CMD ["python", "-m", "uvicorn", "src.DietPlanRAG.DietGuide:app", "--host", "0.0.0.0", "--port", "8000"]
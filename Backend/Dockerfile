FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \ 
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY Backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir uvicorn[standard] fastapi && \
    pip install --no-cache-dir -r requirements.txt

COPY Backend/ ./Backend/

RUN mkdir -p /app/uploads && chmod -R 755 /app/uploads

ENV PYTHONPATH=/app/Backend/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=20 --start-period=180s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "DietPlanRAG.DietGuide:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]

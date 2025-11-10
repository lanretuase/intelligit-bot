# Dockerfile for INTELLIGIT BOT Dashboard + Bot runtime
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (optional: build tools for xgboost/lightgbm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

# Default command: start dashboard API
CMD ["uvicorn", "dashboard.server:app", "--host", "0.0.0.0", "--port", "8000"]

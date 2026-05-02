# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="your-email@example.com"
LABEL description="Supply Chain Demand Forecasting — FastAPI + Streamlit"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY src/        ./src/
COPY api/        ./api/
COPY dashboard/  ./dashboard/
COPY models/     ./models/
COPY data/       ./data/

# Create output directories
RUN mkdir -p reports

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# ── Expose ports ──────────────────────────────────────────────────────────────
# FastAPI
EXPOSE 8000
# Streamlit
EXPOSE 8501

# ── Default command: start FastAPI ────────────────────────────────────────────
# Override with docker-compose to run both services
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

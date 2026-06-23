# ============================================================
# AIC-26 Production Dockerfile
# Multi-stage build for optimized image size
# ============================================================

# ── Stage 1: Builder ───────────────────────────────────────
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml README.md ./
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# ── Stage 2: Production ────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY pyproject.toml README.md ./
COPY .env.example .env

# Cài đặt code thực tế của project để đè lên package rỗng và liên kết với thư mục /app/src
RUN pip install --no-cache-dir --no-deps -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash aic26 && \
    chown -R aic26:aic26 /app

# Create data directory
RUN mkdir -p /app/data && chown -R aic26:aic26 /app/data

# Switch to non-root user
USER aic26

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/health || exit 1

# Expose port
EXPOSE 5000

# Environment variables
ENV HOST=0.0.0.0
ENV PORT=5000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "-m", "aic26.cli.main", "serve", "api"]
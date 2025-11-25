# E2EControl - 智能水网控制系统
# Multi-stage Dockerfile for optimal image size

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        pyyaml \
        numpy \
        scipy \
        cvxpy \
        matplotlib \
        networkx

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.11-slim as runtime

# Labels
LABEL maintainer="E2EControl Team"
LABEL version="1.0.0"
LABEL description="智能水网自主控制系统 - Autonomous Water Network Control"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    E2E_ENV=production \
    E2E_LOG_LEVEL=INFO \
    E2E_WEB_HOST=0.0.0.0 \
    E2E_WEB_PORT=8080 \
    E2E_API_PORT=8000

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash e2euser

# Copy application code
COPY --chown=e2euser:e2euser . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/reports && \
    chown -R e2euser:e2euser /app

# Switch to non-root user
USER e2euser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${E2E_WEB_PORT}/api/status || exit 1

# Expose ports
EXPOSE ${E2E_WEB_PORT} ${E2E_API_PORT}

# Default command
CMD ["python", "-m", "phase5.monitoring.unified_system"]

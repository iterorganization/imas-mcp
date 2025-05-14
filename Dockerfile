# Start with a Python 3.12 slim image
FROM python:3.12-slim AS builder

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    POETRY_VERSION=1.7.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/

# Create and set working directory
WORKDIR /app

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./
COPY imas_mcp_server/ ./imas_mcp_server/

# Configure Poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-root --no-dev --no-interaction

# Create a slim production image
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages and application files
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app/imas_mcp_server /app/imas_mcp_server
COPY --from=builder /app/pyproject.toml /app/
COPY --from=builder /app/build_index.py /app/
COPY --from=builder /app/path_index.py /app/
COPY --from=builder /app/mcp_imas.py /app/

# Set the entrypoint
ENTRYPOINT ["python", "-m", "mcp_imas"]
# Start with a Python 3.12 slim image
FROM python:3.12-slim

# Install system dependencies including git for git dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy git directory and essential files for versioning with hatch-vcs
COPY .git/ ./.git/
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv (git context is now available for hatch-vcs)
# Force fresh install to ensure imas-data-dictionary is up to date
RUN uv sync --no-cache

# Copy source code, scripts, and documentation after dependency installation
COPY imas_mcp_server/ ./imas_mcp_server/
COPY scripts/ ./scripts/
COPY README.md ./

# Copy index files if present in build context, using a pattern that won't fail
COPY index ./index/
RUN echo "Building/verifying index" && \
    INDEX_NAME=$(uv run build-index) && \
    echo "Index built/verified: $INDEX_NAME" && \
    echo "Final index files:" && \
    ls -la ./index/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Expose port 
EXPOSE 8000

# Run the application using the script entrypoint with streamable-http transport
CMD ["uv", "run", "run-server", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]
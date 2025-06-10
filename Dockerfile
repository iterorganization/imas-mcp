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

# Copy uv configuration files first for better caching
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv
RUN uv sync --frozen --no-cache

# Copy only the source package and essential files
COPY imas_mcp_server/ ./imas_mcp_server/
COPY scripts/get_index_name.py ./scripts/

# Get the exact index name from LexicographicSearch class and copy only those files
RUN mkdir -p ./index/ && \
    INDEX_NAME=$(uv run python scripts/get_index_name.py) && \
    echo "Copying index: $INDEX_NAME" && \
    cp index/${INDEX_NAME}_*.* ./index/ 2>/dev/null || true && \
    cp index/_${INDEX_NAME}_*.toc ./index/ 2>/dev/null || true

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Expose port 
EXPOSE 8000

# Run the application using the script entrypoint
CMD ["uv", "run", "run-server"]
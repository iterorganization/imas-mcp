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
COPY README.md ./

# Copy source code and scripts needed for dependency resolution
COPY imas_mcp_server/ ./imas_mcp_server/
COPY scripts/ ./scripts/

# Install dependencies using uv (git context is now available for hatch-vcs)
# Force fresh install to ensure imas-data-dictionary is up to date
RUN uv sync --no-cache

# Build the index to ensure it exists for CI/CD deployments
# This will create the index if it doesn't exist or verify it exists
# Add build args to bust cache when source code or dependencies change
ARG IMAS_INFO
ARG SOURCE_HASH
ENV IMAS_INFO=${IMAS_INFO}
ENV SOURCE_HASH=${SOURCE_HASH}
RUN echo "Building/verifying index with IMAS_INFO: $IMAS_INFO, SOURCE_HASH: $SOURCE_HASH" && \
    INDEX_NAME=$(uv run build-index) && \
    echo "Index name: $INDEX_NAME"

# Copy existing index files from host if they exist, otherwise use the built index
COPY index/ ./index_temp/
RUN INDEX_NAME=$(uv run get-index-name) && \
    echo "Processing index files for: $INDEX_NAME" && \
    if [ "$(ls -A ./index_temp/${INDEX_NAME}* 2>/dev/null)" ]; then \
    echo "Using existing index files from host..." && \
    cp ./index_temp/${INDEX_NAME}*.seg ./index/ 2>/dev/null || true && \
    cp ./index_temp/_${INDEX_NAME}*.toc ./index/ 2>/dev/null || true && \
    cp ./index_temp/${INDEX_NAME}*WRITELOCK ./index/ 2>/dev/null || true; \
    else \
    echo "Using built index files..."; \
    fi && \
    rm -rf ./index_temp && \
    echo "Final index files:" && \
    ls -la ./index/

# Verify that imas-data-dictionary is properly installed with IDSDef.xml
RUN uv run python -c "import imas_data_dictionary; print('imas-data-dictionary installed successfully')"
RUN uv run python -c "from pathlib import Path; import imas_data_dictionary; xml_path = Path(imas_data_dictionary.__file__).parent / 'resources' / 'xml' / 'IDSDef.xml'; print(f'IDSDef.xml exists: {xml_path.exists()}')"

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Expose port 
EXPOSE 8000

# Run the application using the script entrypoint with streamable-http transport
CMD ["uv", "run", "run-server", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]
# Makefile for imas-mcp

.PHONY: install install-dev clean test run package docker-build docker-run

# Install dependencies
install:
	uv sync --no-dev

# Install with development dependencies
install-dev:
	uv sync

# Clean up build artifacts and cache
clean:
	@if exist imas_mcp\__pycache__ rmdir /s /q imas_mcp\__pycache__
	@if exist tests\__pycache__ rmdir /s /q tests\__pycache__
	@if exist scripts\__pycache__ rmdir /s /q scripts\__pycache__
	@if exist build rmdir /s /q build
	@if exist dist rmdir /s /q dist
	@if exist *.egg-info rmdir /s /q *.egg-info
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist .coverage del /q .coverage
	@if exist htmlcov rmdir /s /q htmlcov

# Run tests with coverage
test:
	uv run pytest --cov=imas_mcp --cov-report=html --cov-report=term

# Run tests without coverage
test-fast:
	uv run pytest

# Run the server (default STDIO transport)
run:
	uv run run-server

# Run the server with SSE transport
run-sse:
	uv run run-server --transport sse --host 0.0.0.0 --port 8000

# Run the server with streamable-http transport
run-http:
	uv run run-server --transport streamable-http --host 0.0.0.0 --port 8000

# Build the package
package:
	uv build

# Docker build
docker-build:
	docker build -t imas-mcp .

# Docker run
docker-run:
	docker run -p 8000:8000 imas-mcp

# Format code with black
format:
	uv run black imas_mcp tests scripts

# Lint with ruff
lint:
	uv run ruff check imas_mcp tests scripts

# Fix linting issues
lint-fix:
	uv run ruff check --fix imas_mcp tests scripts

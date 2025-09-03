# Makefile for imas-mcp

.PHONY: install install-dev clean test run package docker-build docker-run install-bench test-baseline performance-baseline test-current performance-current performance-compare test-and-performance

# Install dependencies
install:
	uv sync --no-dev

# Install with development dependencies
install-dev:
	uv sync

# Install benchmark dependencies
install-bench:
	@echo "Installing benchmark dependencies with uv..."
	uv sync --extra bench
	asv machine --yes

# Clean up build artifacts and cache
clean:
	@if exist imas_mcp\__pycache__ rmdir /s /q imas_mcp\__pycache__
	@if exist tests\__pycache__ rmdir /s /q tests\__pycache__
	@if exist scripts\__pycache__ rmdir /s /q scripts\__pycache__
	@if exist benchmarks\__pycache__ rmdir /s /q benchmarks\__pycache__
	@if exist build rmdir /s /q build
	@if exist dist rmdir /s /q dist
	@if exist *.egg-info rmdir /s /q *.egg-info
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist .coverage del /q .coverage
	@if exist htmlcov rmdir /s /q htmlcov
	@if exist .asv rmdir /s /q .asv

# Run tests with coverage
test:
	uv run pytest --cov=imas_mcp --cov-report=html --cov-report=term

# Run tests without coverage
test-fast:
	uv run pytest

# Run baseline tests for current tools
test-baseline:
	@echo "Running baseline tests for current tools..."
	uv run pytest tests/test_server_tools.py -v --tb=short

# Run all current tests
test-current: test-baseline
	@echo "Running all current tests..."
	uv run pytest tests/ -v --tb=short -m "not slow"

# Establish performance baseline
performance-baseline:
	@echo "Establishing performance baseline..."
	uv sync --extra bench
	uv run python scripts/run_performance_baseline.py

# Run current performance benchmarks
performance-current:
	@echo "Running current performance benchmarks..."
	asv run --python=3.12

# Compare performance against baseline
performance-compare:
	@echo "Comparing performance against baseline..."
	asv compare HEAD~1 HEAD

# Run tests and performance monitoring
test-and-performance: test-current performance-current
	@echo "Running tests and performance monitoring..."

# Run tests with coverage
test:
	uv run pytest --cov=imas_mcp --cov-report=html --cov-report=term

# Run tests without coverage
test-fast:
	uv run pytest

# Run the server (default STDIO transport)
run:
	uv run imas-mcp

# Run the server with SSE transport
run-sse:
	uv run imas-mcp --transport sse --host 0.0.0.0 --port 8000

# Run the server with streamable-http transport
run-http:
	uv run imas-mcp --transport streamable-http --host 0.0.0.0 --port 8000

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

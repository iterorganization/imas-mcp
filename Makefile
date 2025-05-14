# Makefile for imas-mcp-server

.PHONY: build-index install install-dev clean test run package publish

# Build the path index
build-index:
	python build_index.py

# Install in development mode
install:
	poetry install

# Install with development dependencies
install-dev:
	poetry install --with dev

# Clean up build artifacts
clean:
	@if exist imas_mcp_server\cache\*.pkl del /q imas_mcp_server\cache\*.pkl
	@if exist build rmdir /s /q build
	@if exist dist rmdir /s /q dist
	@if exist *.egg-info rmdir /s /q *.egg-info
	@if exist __pycache__ rmdir /s /q __pycache__
	poetry cache clear --all --no-interaction

# Run tests
test:
	poetry run pytest

# Run the server
run:
	poetry run python -m mcp_imas

# Build the package
package:
	poetry build

# Publish to PyPI (requires credentials)
publish:
	poetry publish --build

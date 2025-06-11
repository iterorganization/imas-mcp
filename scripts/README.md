# Build Index Script

This document explains the `build_index.py` script and its integration with the Docker build process.

## Overview

The `build_index.py` script is responsible for building the lexicographic search index for the IMAS Data Dictionary. It ensures that the index is available before deployment, especially in CI/CD pipelines where the index might not be pre-built.

## Script Location

- **Script**: `scripts/build_index.py`
- **Command**: `uv run build-index` (configured in `pyproject.toml`)

## How It Works

1. **Initialization**: Creates a `LexicographicSearch` instance with `build=True`
2. **Index Building**: The `LexicographicSearch` class automatically builds the index if it doesn't exist (`len(self) == 0`)
3. **Output**: Returns the exact index name that was built/verified

## Integration with Docker

The script is integrated into the Dockerfile build process:

```dockerfile
# Build the index to ensure it exists for CI/CD deployments
RUN echo "Building/verifying index..." && \
    INDEX_NAME=$(uv run build-index) && \
    echo "Index name: $INDEX_NAME"
```

This ensures that:

- In CI/CD environments where no pre-built index exists, the index is built during Docker image creation
- In development environments where an index already exists, it verifies the index and reports its name
- The same index name is used consistently throughout the build process

## Benefits

1. **CI/CD Compatibility**: Ensures deployments work even without pre-built indices
2. **Consistency**: Uses the same index name determination logic as the main application
3. **Efficiency**: Only builds if the index doesn't already exist
4. **Logging**: Provides clear feedback about the index building process

## Usage

### Manual Usage

```bash
# Build index and get the index name
uv run build-index

# Just get the index name (without building)
uv run get-index-name
```

### Docker Build

The script is automatically called during Docker image building, ensuring the index is ready for deployment.

## Testing

The script includes a test in `tests/test_build_index.py` that verifies:

- The script runs without errors
- It outputs a valid index name
- The index name follows the expected format (`lexicographic_*`)

Run the test with:

```bash
python tests/test_build_index.py
```

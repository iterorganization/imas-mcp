# IMAS MCP Server

A server providing Model Context Protocol (MCP) access to IMAS data structures through a fast, optimized path indexing system.

## Features

- Fast startup with pre-built path index
- Efficient path searching with regex support
- Natural language search for finding paths by description
- Comprehensive metadata access
- Optimized for production use cases

## Installation

```bash
# From PyPI (once published)
pip install imas-mcp-server

# Using Poetry (recommended)
poetry add imas-mcp-server

# From source using Poetry
git clone https://github.com/simon-mcintosh/imas-mcp-server.git
cd imas-mcp-server
poetry install
```

During installation, the package will automatically build a path index that speeds up imports. This one-time process takes 1-2 minutes but reduces future startup times to seconds.

## Usage

### Command Line

```bash
# Using the installed command
imas-mcp

# Or directly with Python
python -m imas-mcp-server
```

### Python API

```python
# Start the server
from imas_mcp_server import run_server
run_server()

# Or use the tools directly
from mcp_imas import find_paths_by_pattern

paths = find_paths_by_pattern("equilibrium/time_slice")
```

### Natural Language Search

You can now search for IMAS paths using natural language descriptions rather than knowing the exact path structure:

```bash
# Using the installed command
find-paths "electron temperature profile"

# Or directly with Python
python -m bin.find_imas_path "electron temperature profile"

# With verbose output to show documentation
find-paths -v "safety factor"
```

From Python code:

```python
from imas_mcp_server.path_index_cache import PathIndexCache

# Load the path index
path_index = PathIndexCache().path_index

# Search for paths by natural language description
results = path_index.search_by_keywords("plasma current measurement")

for result in results:
    print(f"Path: {result['path']}")
    print(f"Score: {result['score']}")
    print(f"Documentation: {result['doc'][:100]}...")
print(paths)
```

## Performance

The path index is built once during package installation and loaded on import, which significantly reduces the startup time of the server:

|              | Without Pre-built Index | With Pre-built Index |
| ------------ | ----------------------: | -------------------: |
| Startup Time |             ~80 seconds |            ~1 second |
| Memory Usage |                    Same |                 Same |
| First Query  |                    Slow |                 Fast |

The index includes:

- Full paths to all IDS elements (~60,000 paths)
- Path segments for hierarchical lookups
- Keywords for fuzzy matching
- Prefixes for autocomplete functionality

## Development

```bash
# Setup development environment
poetry install --with dev

# Manually rebuild the path index
poetry run python build_index.py

# Run tests
poetry run pytest

# Run linting
poetry run ruff check .
poetry run black .

# Build package
poetry build

# Publish to PyPI
poetry publish
```

## Environment Variables

- `IMAS_MCP_SILENT_INIT`: Set to "true" to suppress initialization messages

## How It Works

1. **Installation**: During package installation, Poetry's build hook runs `build_index.py`
2. **Build Process**: The script parses the IMAS data dictionary and creates a comprehensive path index
3. **Serialization**: The index is saved to disk in `cache/path_index.pkl`
4. **Import**: When importing the module, the pre-built index is loaded in ~1 second

This approach uses modern Python packaging tools (Poetry) and avoids the expensive index building process each time the module is imported. The build hook system ensures the index is created during installation without relying on deprecated setuptools mechanisms.

## Implementation Details

### Path Indexing System

The IMAS MCP Server uses a sophisticated path indexing system to efficiently navigate the IMAS data dictionary:

1. **Indexing**: During initialization, the system builds a comprehensive index of all IMAS paths
2. **Caching**: The index is serialized to disk for fast loading in subsequent runs
3. **Efficient Retrieval**: Various indexing strategies enable fast path lookups

### Keyword Search Implementation

The natural language search functionality works through the following mechanisms:

1. **Documentation Extraction**: During index building, documentation strings are extracted from the IMAS data dictionary
2. **Keyword Processing**: Documentation is tokenized, stop words are removed, and keywords are extracted
3. **Inverted Index**: An inverted index maps keywords to relevant paths
4. **Relevance Scoring**: Search results are ranked by relevance based on keyword matches

The implementation handles:

- Stop word filtering (common words like "the", "and", "of")
- Partial keyword matching
- Relevance scoring based on exact and partial matches

Future enhancements could include:

- Word embeddings for semantic search capabilities
- TF-IDF weighting for better keyword ranking
- Fuzzy matching for typo tolerance

## Docker Usage

The server is available as a pre-built Docker container with the index already built:

```bash
# Pull and run the latest container
docker run -d -p 8000:8000 ghcr.io/iterorganization/imas-mcp-server:latest

# Or use Docker Compose
docker-compose up -d
```

See [DOCKER.md](DOCKER.md) for detailed container usage, deployment options, and troubleshooting.

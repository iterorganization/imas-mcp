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
pip install imas-mcp

# Using uv (recommended)
uv add imas-mcp

# From source using uv
git clone https://github.com/iterorganization/imas-mcp.git
cd imas-mcp
uv sync
```

During installation, the package will automatically build a Whoosh index that speeds up imports. This one-time process takes ~8 minutes but reduces future startup times to seconds.

## Usage

### Command Line

```bash
# Using the installed command
run-server

# Or directly with Python
python -m imas_mcp
```

### Python API

```python
# Start the server
from imas_mcp import run_server
run_server()

# Or use the tools directly
from mcp_imas import find_paths_by_pattern

paths = find_paths_by_pattern("equilibrium/time_slice")
```

### Natural Language Search

You can now search for IMAS paths using natural language descriptions rather than knowing the exact path structure:

From Python code:

```python
from imas_mcp.lexicographic_search import LexicographicSearch

# Create search instance
search = LexicographicSearch()

# Search for paths by natural language description
results = search.search_by_keywords("plasma current measurement")

for result in results:
    print(f"Path: {result.path}")
    print(f"Documentation: {result.documentation[:100]}...")
```

## Development

```bash
# Setup development environment
uv sync --all-extras

# Run tests
uv run pytest

# Run linting and formatting
uv run ruff check .
uv run ruff format .

# Build package
uv build

# Publish to PyPI
uv publish
```

## Environment Variables

- `IMAS_MCP_SILENT_INIT`: Set to "true" to suppress initialization messages

## How It Works

1. **Installation**: During package installation, the index builds automatically when the module is first imported
2. **Build Process**: The system parses the IMAS data dictionary and creates a comprehensive path index
3. **Serialization**: The Whoosh index is serialized to the `index/` directory with persistent storage
4. **Import**: When importing the module, the pre-built index loads in ~1 second

This approach uses modern Python packaging tools (uv) and avoids the expensive index building process each time the module is imported. The automatic index building ensures the index gets created during installation without relying on deprecated setuptools mechanisms.

## Implementation Details

### Path Indexing System

The IMAS MCP Server uses a sophisticated path indexing system to efficiently navigate the IMAS data dictionary:

1. **Indexing**: During initialization, the system builds a comprehensive index of all IMAS paths
2. **Caching**: The index is serialized to disk for fast loading in subsequent runs
3. **Efficient Retrieval**: Various indexing strategies enable fast path lookups

### LexicographicSearch Class

The `LexicographicSearch` class is the core component that provides fast, flexible search capabilities over the IMAS Data Dictionary. It combines Whoosh full-text indexing with IMAS-specific data processing to enable multiple search modes:

#### Search Methods

1. **Keyword Search** (`search_by_keywords`):

   - Natural language queries with advanced syntax support
   - Field-specific searches (e.g., `documentation:plasma ids:core_profiles`)
   - Boolean operators (`AND`, `OR`, `NOT`)
   - Wildcards (`*` and `?` patterns)
   - Fuzzy matching for typo tolerance (using `~` operator)
   - Phrase matching with quotes
   - Relevance scoring and sorting

2. **Exact Path Lookup** (`search_by_exact_path`):

   - Direct retrieval by complete IDS path
   - Returns full documentation and metadata
   - Fastest lookup method for known paths

3. **Path Prefix Search** (`search_by_path_prefix`):

   - Hierarchical exploration of IDS structure
   - Find all sub-elements under a given path
   - Useful for browsing related data elements

4. **Filtered Search** (`filter_search_results`):
   - Apply regex filters to search results
   - Filter by specific fields (units, IDS name, etc.)
   - Combine with other search methods for precise results

#### Key Capabilities

- **Automatic Index Building**: Creates search index on first use
- **Persistent Caching**: Index stored on disk for fast subsequent loads
- **Advanced Query Parsing**: Supports complex search expressions
- **Relevance Ranking**: Results sorted by match quality
- **Pagination Support**: Handle large result sets efficiently
- **Field-Specific Boosts**: Weight certain fields higher in searches

## Future Work

### Semantic Search Enhancement

We are planning to enhance the current lexicographic search with semantic search capabilities using modern language models. This enhancement will provide:

#### Planned Features

- **Vector Embeddings**: Generate semantic embeddings for IMAS documentation using transformer models
- **Semantic Similarity**: Find conceptually related terms even when exact keywords don't match
- **Context-Aware Search**: Understand the scientific context and domain-specific terminology
- **Hybrid Search**: Combine lexicographic and semantic approaches for optimal results

#### Technical Approach

The semantic search will complement the existing fast lexicographic search:

1. **Embedding Generation**: Process IMAS documentation through scientific language models
2. **Vector Storage**: Store embeddings alongside the current Whoosh index
3. **Similarity Search**: Use cosine similarity or other distance metrics for semantic matching
4. **Result Fusion**: Combine lexicographic and semantic results with configurable weighting

#### Use Cases

This will enable searches like:

- "plasma confinement parameters" → finds relevant equilibrium and profiles data
- "fusion reactor diagnostics" → discovers measurement and sensor-related paths
- "energy transport coefficients" → locates thermal and particle transport data

The semantic layer will make the IMAS data dictionary more accessible to researchers who may not be familiar with the exact terminology or path structure.

## Docker Usage

The server is available as a pre-built Docker container with the index already built:

```bash
# Pull and run the latest container
docker run -d -p 8000:8000 ghcr.io/iterorganization/imas-mcp:latest

# Or use Docker Compose
docker-compose up -d
```

See [DOCKER.md](DOCKER.md) for detailed container usage, deployment options, and troubleshooting.

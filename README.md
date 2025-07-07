# IMAS MCP Server

A Model Context Protocol (MCP) server providing AI assistants with access to IMAS (Integrated Modelling & Analysis Suite) data structures through natural language search and optimized path indexing.

## Quick Start - Connect to Hosted Server

The easiest way to get started is connecting to our hosted IMAS MCP server. No installation required!

### VS Code Setup

#### Option 1: Interactive Setup (Recommended)

1. Open VS Code and press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "MCP: Add Server" and select it
3. Choose "HTTP Server"
4. Enter server name: `imas-mcp-hosted`
5. Enter server URL: `https://imas-mcp.example.com/mcp/`

#### Option 2: Manual Configuration

Choose one of these file locations:

- **Workspace Settings (Recommended)**: `.vscode/mcp.json` in your workspace (`Ctrl+Shift+P` → "Preferences: Open Workspace Settings (JSON)")
- **User Settings**: VS Code `settings.json` (`Ctrl+Shift+P` → "Preferences: Open User Settings (JSON)")

Then add this configuration:

```json
{
  "servers": {
    "imas-mcp-hosted": {
      "type": "http",
      "url": "https://imas-mcp.example.com/mcp/"
    }
  }
}
```

_Note: For user settings.json, wrap the above in `"mcp": { ... }`_

### Claude Desktop Setup

Add to your Claude Desktop config file:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`  
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Linux:** `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "imas-mcp-hosted": {
      "command": "npx",
      "args": ["mcp-remote", "https://imas-mcp.example.com/mcp/"]
    }
  }
}
```

## Quick Start - Local Docker Server

If you have Docker available, you can run a local IMAS MCP server:

### Start the Docker Server

```bash
# Pull and run the server
docker run -d \
  --name imas-mcp \
  -p 8000:8000 \
  ghcr.io/iterorganization/imas-mcp:latest

# Verify it's running
docker ps --filter name=imas-mcp --format "table {{.Names}}\t{{.Status}}"
```

### Configure Your Client

**VS Code** - Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "imas-mcp-docker": {
      "type": "http",
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

**Claude Desktop** - Add to your config file:

```json
{
  "mcpServers": {
    "imas-mcp-docker": {
      "command": "npx",
      "args": ["mcp-remote", "http://localhost:8000/mcp/"]
    }
  }
}
```

## Quick Start - Local UV Installation

If you have [uv](https://docs.astral.sh/uv/) installed, you can run the server directly:

### Install and Configure

```bash
# Install imas-mcp with uv
uv tool install imas-mcp

# Or add to a project
uv add imas-mcp
```

### UV Client Configuration

**VS Code** - Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "imas-mcp-uv": {
      "type": "stdio",
      "command": "uv",
      "args": ["tool", "run", "run-server", "--auto-build"]
    }
  }
}
```

**Claude Desktop** - Add to your config file:

```json
{
  "mcpServers": {
    "imas-mcp-uv": {
      "command": "uv",
      "args": ["tool", "run", "run-server", "--auto-build"]
    }
  }
}
```

## Development

For local development and customization:

### Setup

```bash
# Clone repository
git clone https://github.com/iterorganization/imas-mcp.git
cd imas-mcp

# Install development dependencies (search index build takes ~8 minutes first time)
uv sync --all-extras
```

### Build Dependencies

This project requires additional dependencies during the build process that are not part of the runtime dependencies. These include:

- **`imas-data-dictionary`** - Required for generating the search index during build
- **`rich`** - Used for enhanced console output during build processes

**For developers:** These build dependencies are included in the `dev` dependency group and can be installed with:

```bash
uv sync --group dev
```

**Location in configuration:**

- **Build-time dependencies**: Listed in `[build-system.requires]` in `pyproject.toml`
- **Development access**: Also available in `[dependency-groups.dev]` for local development

**Note:** Regular users installing the package don't need these dependencies - they're only required when building from source or working with the data dictionary directly.

### Development Commands

```bash
# Run tests
uv run pytest

# Run linting and formatting
uv run ruff check .
uv run ruff format .


# Run the server locally
uv run python -m imas_mcp --transport streamable-http --port 8000

# Run with stdio transport for MCP development
uv run python -m imas_mcp --transport stdio --auto-build
```

### Local Development MCP Configuration

**VS Code** - The repository includes a `.vscode/mcp.json` file with pre-configured development server options. Use the `imas-local-stdio` configuration for local development.

**Claude Desktop** - Add to your config file:

```json
{
  "mcpServers": {
    "imas-local-dev": {
      "command": "uv",
      "args": ["run", "python", "-m", "imas_mcp", "--auto-build"],
      "cwd": "/path/to/imas-mcp"
    }
  }
}
```

## How It Works

1. **Installation**: During package installation, the index builds automatically when the module first imports
2. **Build Process**: The system parses the IMAS data dictionary and creates a comprehensive path index
3. **Serialization**: The system stores indexes in organized subdirectories:
   - **Lexicographic index**: `imas_mcp/resources/index_data/` (Whoosh search index)
   - **JSON data**: `imas_mcp/resources/json_data/` (LLM-optimized structured data)
4. **Import**: When importing the module, the pre-built index loads in ~1 second

## Optional Dependencies and Runtime Requirements

The IMAS MCP server uses a composable pattern that allows it to work with or without the `imas-data-dictionary` package at runtime:

### Package Installation Options

- **Runtime only**: `pip install imas-mcp` - Uses pre-built indexes, stdio transport only
- **With HTTP support**: `pip install imas-mcp[http]` - Adds support for sse/streamable-http transports
- **With build support**: `pip install imas-mcp[build]` - Includes `imas-data-dictionary` for index building
- **Full installation**: `pip install imas-mcp[all]` - Includes all optional dependencies

### Data Dictionary Access

The system uses multiple fallback strategies to access IMAS Data Dictionary version and metadata:

1. **Environment Variable**: `IMAS_DD_VERSION` (highest priority)
2. **Metadata File**: JSON metadata stored alongside indexes
3. **Index Name Parsing**: Extracts version from index filename
4. **IMAS Package**: Direct access to `imas-data-dictionary` (if available)

This design ensures the server can:

- **Build indexes** when the IMAS package is available
- **Run with pre-built indexes** without requiring the IMAS package
- **Access version/metadata** through multiple reliable fallback mechanisms

### Index Building vs Runtime

- **Index Building**: Requires `imas-data-dictionary` package to parse XML and create indexes
- **Runtime Search**: Only requires pre-built indexes and metadata, no IMAS package dependency
- **Version Access**: Uses composable accessor pattern with multiple fallback strategies

## Implementation Details

### LexicographicSearch Class

The `LexicographicSearch` class is the core component that provides fast, flexible search capabilities over the IMAS Data Dictionary. It combines Whoosh full-text indexing with IMAS-specific data processing to enable different search modes:

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
- **Persistent Caching**: Index stored on disk for fast future loads
- **Advanced Query Parsing**: Supports complex search expressions
- **Relevance Ranking**: Results sorted by match quality
- **Pagination Support**: Handle large result sets efficiently
- **Field-Specific Boosts**: Weight certain fields higher in searches

## Future Work

### Semantic Search Enhancement

We plan to enhance the current lexicographic search with semantic search capabilities using modern language models. This enhancement will provide:

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

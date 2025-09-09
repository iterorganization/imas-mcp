# IMAS MCP Server

[![pre-commit][pre-commit-badge]][pre-commit-link]
[![Ruff][ruff-badge]][ruff-link]
[![Python versions][python-badge]][python-link]
[![CI/CD status][build-deploy-badge]][build-deploy-link]
[![Coverage status][codecov-badge]][codecov-link]
[![Documentation][docs-badge]][docs-link]
[![ASV][asv-badge]][asv-link]

A Model Context Protocol (MCP) server providing AI assistants with access to IMAS (Integrated Modelling & Analysis Suite) data structures through natural language search and optimized path indexing.

## Quick Start - Connect to Hosted Server

The easiest way to get started is connecting to our hosted IMAS MCP server. No installation required!

### VS Code Setup

#### Option 1: Interactive Setup (Recommended)

1. Open VS Code and press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "MCP: Add Server" and select it
3. Choose "HTTP Server"
4. Enter server name: `imas`
5. Enter server URL: `https://imas-dd.iter.org/mcp/`

#### Option 2: Manual Configuration

Choose one of these file locations:

- **Workspace Settings (Recommended)**: `.vscode/mcp.json` in your workspace (`Ctrl+Shift+P` → "Preferences: Open Workspace Settings (JSON)")
- **User Settings**: VS Code `settings.json` (`Ctrl+Shift+P` → "Preferences: Open User Settings (JSON)")

Then add this configuration:

```json
{
  "servers": {
    "imas": {
      "type": "http",
      "url": "https://imas-dd.iter.org/mcp/"
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
      "args": ["mcp-remote", "https://imas-dd.iter.org/mcp/"]
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
  ghcr.io/iterorganization/imas-mcp:latest-streamable-http

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
      "args": ["run", "python", "-m", "imas_mcp.server"]
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
      "args": ["run", "python", "-m", "imas_mcp.server"]
    }
  }
}
```

  ## Running Over Slurm (STDIO Transport)

  For HPC environments where direct TCP access from a login node to compute nodes is restricted or you prefer ephemeral allocations, you can run the server via Slurm using STDIO. A helper script is provided: `scripts/imas_mcp_sdcc.sh`.

  ### Add to VS Code

  Update (or create) `.vscode/mcp.json` (JSONC supported) with:

  ```jsonc
  {
    "servers": {
      "imas-slurm-stdio": {
        "type": "stdio",
        "command": "scripts/imas_mcp_slurm_stdio.sh",
      }
    }
  }
  ```

  When the MCP client starts this server it will:

  1. Detect if already inside an allocation (`SLURM_JOB_ID`). If yes, it starts immediately.
  2. Otherwise, it runs `srun --pty` to request a node and then launches the server with unbuffered STDIO.

  ### Customizing the Allocation

  Control Slurm resources via environment variables before the client spawns the server:

  | Variable | Purpose | Default |
  |----------|---------|---------|
  | `IMAS_MCP_SLURM_TIME` | Walltime | `08:00:00` |
  | `IMAS_MCP_SLURM_CPUS` | CPUs per task | `1` |
  | `IMAS_MCP_SLURM_MEM` | Memory (e.g. `4G`) | Slurm default |
  | `IMAS_MCP_SLURM_PARTITION` | Partition/queue | Cluster default |
  | `IMAS_MCP_SLURM_ACCOUNT` | Account/project | User default |
  | `IMAS_MCP_SLURM_EXTRA` | Extra raw `srun` flags | (empty) |
  | `IMAS_MCP_USE_ENTRYPOINT` | Use `imas-mcp` entrypoint instead of `python -m` | `0` |

  Example (from your shell before launching VS Code):

  ```bash
  export IMAS_MCP_SLURM_TIME=02:00:00
  export IMAS_MCP_SLURM_CPUS=4
  export IMAS_MCP_SLURM_MEM=8G
  export IMAS_MCP_SLURM_PARTITION=compute
  ```

  ### Direct CLI Usage

  You can also invoke the script directly:

  ```bash
  scripts/imas_mcp_slurm_stdio.sh --ids-filter "core_profiles equilibrium"
  ```

  ### Why STDIO for Slurm?

  Using STDIO avoids opening network ports, simplifying security on clusters that block ephemeral sockets or require extra approvals for services. Tools and responses stream through the existing `srun` pseudo-TTY.


## Example IMAS Queries

Once you have the IMAS MCP server configured, you can interact with it using natural language queries. Use the `@imas` prefix to direct queries to the IMAS server:

### Basic Search Examples

```text
@imas Find data paths related to plasma temperature
@imas Search for electron density measurements
@imas What data is available for magnetic field analysis?
@imas Show me core plasma profiles
```

### Physics Concept Exploration

```text
@imas Explain what equilibrium reconstruction means in plasma physics
@imas What is the relationship between pressure and magnetic fields?
@imas How do transport coefficients relate to plasma confinement?
@imas Describe the physics behind current drive mechanisms
```

### Data Structure Analysis

```text
@imas Analyze the structure of the core_profiles IDS
@imas What are the relationships between equilibrium and core_profiles?
@imas Show me identifier schemas for transport data
@imas Export bulk data for equilibrium, core_profiles, and transport IDS
```

### Advanced Queries

```text
@imas Find all paths containing temperature measurements across different IDS
@imas What physics domains are covered in the IMAS data dictionary?
@imas Show me measurement dependencies for fusion power calculations
@imas Explore cross-domain relationships between heating and confinement
```

### Workflow and Integration

```text
@imas How do I access electron temperature profiles from IMAS data?
@imas What's the recommended workflow for equilibrium analysis?
@imas Show me the branching logic for diagnostic identifier schemas
@imas Export physics domain data for comprehensive transport analysis
```

The IMAS MCP server provides 8 specialized tools for different types of queries:

- **Search**: Natural language and structured search across IMAS data paths
- **Explain**: Physics concepts with IMAS context and domain expertise
- **Overview**: General information about IMAS structure and available data
- **Analyze**: Detailed structural analysis of specific IDS
- **Explore**: Relationship discovery between data paths and physics domains
- **Identifiers**: Exploration of enumerated options and branching logic
- **Bulk Export**: Comprehensive export of multiple IDS with relationships
- **Domain Export**: Physics domain-specific data with measurement dependencies

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

# Build schema data structures from IMAS data dictionary
uv run build-schemas

# Build document store and semantic search embeddings
uv run build-embeddings

# Run the server locally
uv run python -m imas_mcp --transport streamable-http --port 8000

# Run with stdio transport for MCP development
uv run python -m imas_mcp --transport stdio --auto-build
```

### Build Scripts

The project includes two separate build scripts for creating the required data structures:

**`build-schemas`** - Creates schema data structures from IMAS XML data dictionary:

- Transforms XML data into optimized JSON format
- Creates catalog and relationship files
- Use `--ids-filter "core_profiles equilibrium"` to build specific IDS
- Use `--force` to rebuild even if files exist

**`build-embeddings`** - Creates document store and semantic search embeddings:

- Builds in-memory document store from JSON data
- Generates sentence transformer embeddings for semantic search
- Caches embeddings for fast loading
- Use `--model-name "all-mpnet-base-v2"` for different models
- Use `--force` to rebuild embeddings cache
- Use `--no-normalize` to disable embedding normalization
- Use `--half-precision` to reduce memory usage
- Use `--similarity-threshold 0.1` to set similarity score thresholds

**Note:** The build hook creates JSON data. Build embeddings separately using `build-embeddings` for better control and performance.

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
2. **Build Process**: The system parses the IMAS data dictionary and creates comprehensive JSON files with structured data
3. **Embedding Generation**: Creates semantic embeddings using sentence transformers for advanced search capabilities
4. **Serialization**: The system stores indexes in organized subdirectories:
   - **JSON data**: `imas_mcp/resources/schemas/` (LLM-optimized structured data)
   - **Embeddings cache**: Pre-computed sentence transformer embeddings for semantic search
5. **Import**: When importing the module, the pre-built index and embeddings load in ~1 second

## Optional Dependencies and Runtime Requirements

The IMAS MCP server uses a composable pattern that allows it to work with or without the `imas-data-dictionary` package at runtime:

### Package Installation Options

- **Runtime only**: `uv add imas-mcp` - Uses pre-built indexes, stdio transport only
- **With HTTP support**: `uv add imas-mcp[http]` - Adds support for sse/streamable-http transports
- **With build support**: `uv add imas-mcp[build]` - Includes `imas-data-dictionary` for index building
- **Full installation**: `uv add imas-mcp[all]` - Includes all optional dependencies

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

### Search Implementation

The search system is the core component that provides fast, flexible search capabilities over the IMAS Data Dictionary. It combines efficient indexing with IMAS-specific data processing and semantic search to enable different search modes:

#### Search Methods

1. **Semantic Search** (`SearchMode.SEMANTIC`):

   - AI-powered semantic understanding using sentence transformers
   - Natural language queries with physics context awareness
   - Finds conceptually related terms even without exact keyword matches
   - Best for exploratory research and concept discovery

2. **Lexical Search** (`SearchMode.LEXICAL`):

   - Fast text-based search with exact keyword matching
   - Boolean operators (`AND`, `OR`, `NOT`)
   - Wildcards (`*` and `?` patterns)
   - Field-specific searches (e.g., `documentation:plasma ids:core_profiles`)
   - Fastest performance for known terminology

3. **Hybrid Search** (`SearchMode.HYBRID`):

   - Combines semantic and lexical approaches
   - Provides both exact matches and conceptual relevance
   - Balanced performance and comprehensiveness

4. **Auto Search** (`SearchMode.AUTO`):
   - Intelligent search mode selection based on query characteristics
   - Automatically chooses optimal search strategy
   - Adaptive performance optimization

#### Key Capabilities

- **Search Mode Selection**: Choose between semantic, lexical, hybrid, or auto modes
- **Performance Caching**: TTL-based caching system with hit rate monitoring
- **Semantic Embeddings**: Pre-computed sentence transformer embeddings for fast semantic search
- **Physics Context**: Domain-aware search with IMAS-specific terminology
- **Advanced Query Parsing**: Supports complex search expressions and field filtering
- **Relevance Ranking**: Results sorted by match quality and physics relevance

## Future Work

### MCP Resources Implementation (Phase 2 - Planned)

We plan to implement MCP resources to provide efficient access to pre-computed IMAS data:

#### Planned Resource Features

- **Static JSON IDS Data**: Pre-computed IDS catalog and structure data served as MCP resources
- **Physics Measurement Data**: Domain-specific measurement data and relationships
- **Usage Examples**: Code examples and workflow patterns for common analysis tasks
- **Documentation Resources**: Interactive documentation and API references

#### Resource Types

- `ids://catalog` - Complete IDS catalog with metadata
- `ids://structure/{ids_name}` - Detailed structure for specific IDS
- `ids://physics-domains` - Physics domain mappings and relationships
- `examples://search-patterns` - Common search patterns and workflows

### MCP Prompts Implementation (Phase 3 - Planned)

Specialized prompts for physics analysis and workflow automation:

#### Planned Prompt Categories

- **Physics Analysis Prompts**: Specialized prompts for plasma physics analysis tasks
- **Code Generation Prompts**: Generate Python analysis code for IMAS data
- **Workflow Automation Prompts**: Automate complex multi-step analysis workflows
- **Data Validation Prompts**: Create validation approaches for IMAS measurements

#### Prompt Templates

- `physics-explain` - Generate comprehensive physics explanations
- `measurement-workflow` - Create measurement analysis workflows
- `cross-ids-analysis` - Analyze relationships between multiple IDS
- `imas-python-code` - Generate Python code for data analysis

### Performance Optimization (Phase 4 - In Progress)

Continued optimization of search and tool performance:

#### Current Optimizations (Implemented)

- ✅ **Search Mode Selection**: Multiple search modes (semantic, lexical, hybrid, auto)
- ✅ **Search Caching**: TTL-based caching with hit rate monitoring for search operations
- ✅ **Semantic Embeddings**: Pre-computed sentence transformer embeddings
- ✅ **ASV Benchmarking**: Automated performance monitoring and regression detection

#### Planned Optimizations

- **Advanced Caching Strategy**: Intelligent cache management for all MCP operations (beyond search)
- **Performance Monitoring**: Enhanced metrics tracking and analysis across all tools
- **Multi-Format Export**: Optimized export formats (raw, structured, enhanced)
- **Selective AI Enhancement**: Conditional AI enhancement based on request context

### Testing and Quality Assurance (Phase 5 - Planned)

Comprehensive testing strategy for all MCP components:

#### Test Implementation Goals

- **MCP Tool Testing**: Complete test coverage using FastMCP 2 testing framework
- **Resource Testing**: Validation of all MCP resources and data integrity
- **Prompt Testing**: Automated testing of prompt templates and responses
- **Performance Testing**: Benchmarking and regression detection for all tools

## Docker Usage

The server is available as a pre-built Docker container with the index already built:

```bash
# Pull and run the latest container
docker run -d -p 8000:8000 ghcr.io/iterorganization/imas-mcp:latest

# Or use Docker Compose
docker-compose up -d
```

See [DOCKER.md](DOCKER.md) for detailed container usage, deployment options, and troubleshooting.

[python-badge]: https://img.shields.io/badge/python-3.12-blue
[python-link]: https://www.python.org/downloads/
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json
[ruff-link]: https://docs.astral.sh/ruff/
[pre-commit-badge]: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
[pre-commit-link]: https://github.com/pre-commit/pre-commit
[build-deploy-badge]: https://img.shields.io/github/actions/workflow/status/simon-mcintosh/imas-mcp/test.yml?branch=main&color=brightgreen&label=CI%2FCD
[build-deploy-link]: https://github.com/iterorganization/imas-mcp/actions/workflows/test.yml
[codecov-badge]: https://codecov.io/gh/simon-mcintosh/imas-mcp/graph/badge.svg
[codecov-link]: https://codecov.io/gh/simon-mcintosh/imas-mcp
[docs-badge]: https://img.shields.io/badge/docs-online-brightgreen
[docs-link]: https://simon-mcintosh.github.io/imas-mcp/
[asv-badge]: https://img.shields.io/badge/ASV-Benchmarks-blue?style=flat&logo=speedtest&logoColor=white
[asv-link]: https://simon-mcintosh.github.io/imas-mcp/benchmarks/

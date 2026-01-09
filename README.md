# IMAS Codex Server

[![pre-commit][pre-commit-badge]][pre-commit-link]
[![Ruff][ruff-badge]][ruff-link]
[![Python versions][python-badge]][python-link]
[![CI/CD status][build-deploy-badge]][build-deploy-link]
[![Coverage status][codecov-badge]][codecov-link]
[![Documentation][docs-badge]][docs-link]
[![ASV][asv-badge]][asv-link]

A Model Context Protocol (MCP) server providing AI assistants with access to IMAS (Integrated Modelling & Analysis Suite) data structures through natural language search and optimized path indexing.

## MCP Servers

IMAS Codex provides two MCP servers:

| Server | Command | Purpose |
|--------|---------|---------|
| **IMAS DD** | `imas-codex serve imas` | IMAS Data Dictionary knowledge, semantic search |
| **Agents** | `imas-codex serve agents` | Remote facility exploration via subagents |

## Quick Start

Select the setup method that matches your environment:

- HTTP (Hosted): Zero install. Connect to the public endpoint running the latest tagged MCP server from the ITER Organization.
- UV (Local): Install and run in your own Python environment for editable development.
- Docker : Run an isolated container with pre-built indexes.
- Slurm / HPC (STDIO): Launch inside a cluster allocation without opening network ports.

Choose hosted for instant access; choose a local option for customization or controlled resources.

[HTTP](#http-remote-public-endpoint) | [UV](#uv-local-install) | [Docker](#docker-setup) | [Slurm / HPC](#slurm--hpc-stdio)

### HTTP (Remote Public Endpoint)

Connect to the public ITER Organization hosted server—no local install.

#### VS Code (Interactive)

1. `Ctrl+Shift+P` → "MCP: Add Server"
2. Select "HTTP Server"
3. Name: `imas`
4. URL: `https://imas-dd.iter.org/mcp`

#### VS Code (Manual JSON)

Workspace `.vscode/mcp.json` (or inside `"mcp"` in user settings):

```json
{
  "servers": {
    "imas": { "type": "http", "url": "https://imas-dd.iter.org/mcp" }
  }
}
```

#### Claude Desktop config

Pick path for your OS:

Windows: `%APPDATA%\\Claude\\claude_desktop_config.json`  
macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`  
Linux: `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "imas-codex-hosted": {
      "command": "npx",
      "args": ["mcp-remote", "https://imas-dd.iter.org/mcp"]
    }
  }
}
```

#### OP Client (Pending Clarification)

Placeholder: clarify what "op" refers to (e.g. OpenAI, Operator) to add tailored instructions.

### UV Local Install

Install with [uv](https://docs.astral.sh/uv/):

```bash
# Standard installation (includes sentence-transformers)
uv tool install imas-codex

# Add to a project env
uv add imas-codex
```

#### Data Dictionary Version

The IMAS Data Dictionary version determines which schema definitions are used. The version is resolved in priority order:

| Priority | Source | Description |
|----------|--------|-------------|
| 1 | `--dd-version` CLI option | Highest priority, explicit override |
| 2 | `IMAS_DD_VERSION` env var | Environment-based override |
| 3 | `pyproject.toml` default | Configured default from `[tool.imas-codex].default-dd-version` |

**Configuration:**

Set the default DD version in `pyproject.toml`:

```toml
[tool.imas-codex]
default-dd-version = "4.1.0"
```

**Runtime Override:**

```bash
# Via CLI option
imas-codex --dd-version 3.42.2

# Via environment variable
IMAS_DD_VERSION=3.42.2 imas-codex

# Docker build
docker build --build-arg IMAS_DD_VERSION=3.42.2 ...
```

**Version Validation:**

The server validates that the requested DD version does not exceed the maximum version available in the installed `imas-data-dictionaries` package. If you request a version that's not available, you'll see:

```
ValueError: Requested DD version 5.0.0 exceeds maximum available version 4.1.0.
Update imas-data-dictionaries dependency or use a lower version.
```

#### Embedding Configuration

The IMAS Codex server uses sentence-transformers for generating embeddings:

**Configuration:**

The default embedding model is configured in `pyproject.toml` under `[tool.imas-codex]`:

```toml
[tool.imas-codex]
imas-embedding-model = "all-MiniLM-L6-v2"  # For DD embeddings
```

Environment variables override pyproject.toml settings:

```bash
export IMAS_CODEX_EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

**Path Inclusion Settings:**

Control which IMAS paths are indexed and searchable. These settings affect schema generation, embeddings, and semantic search:

| Setting              | pyproject.toml         | Environment Variable              | Default | Description                                                             |
| -------------------- | ---------------------- | --------------------------------- | ------- | ----------------------------------------------------------------------- |
| Include GGD          | `include-ggd`          | `IMAS_CODEX_INCLUDE_GGD`          | `true`  | Include Grid Geometry Description paths                                 |
| Include Error Fields | `include-error-fields` | `IMAS_CODEX_INCLUDE_ERROR_FIELDS` | `false` | Include uncertainty bound fields (`_error_upper`, `_error_lower`, etc.) |

Example pyproject.toml configuration:

```toml
[tool.imas-codex]
include-ggd = true
include-error-fields = false
```

Environment variable overrides:

```bash
export IMAS_CODEX_INCLUDE_GGD=false     # Exclude GGD paths
export IMAS_CODEX_INCLUDE_ERROR_FIELDS=true  # Include error fields
```

**Error Handling:**

If model loading fails, the system will fall back to the default `all-MiniLM-L6-v2` model.

VS Code (`.vscode/mcp.json`):

```json
{
  "servers": {
    "imas-codex-uv": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "imas-codex", "serve", "imas", "--transport", "stdio"]
    }
  }
}
```

Claude Desktop:

```json
{
  "mcpServers": {
    "imas-codex-uv": {
      "command": "uv",
      "args": ["run", "imas-codex", "serve", "imas", "--transport", "stdio"]
    }
  }
}
```

#### Agents MCP Server (Local Facility Exploration)

For exploring remote fusion facilities, run the Agents server locally:

VS Code (`.vscode/mcp.json`):

```json
{
  "servers": {
    "imas-agents": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "imas-codex", "serve", "agents"]
    }
  }
}
```

This provides the `/explore` prompt for LLM-driven exploration of remote facilities via SSH.

#### SSH ControlMaster Setup (Recommended)

For fast repeated SSH connections during facility exploration, configure SSH ControlMaster. This keeps connections alive, reducing latency from ~1-2 seconds to ~100ms for subsequent commands.

```bash
# Create socket directory
mkdir -p ~/.ssh/sockets
chmod 700 ~/.ssh/sockets
```

Add to `~/.ssh/config`:

```
# EPFL / Swiss Plasma Center
Host epfl
    HostName spcepfl.epfl.ch
    User your_username
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600

# Add other facilities as needed
Host ipp
    HostName gateway.ipp.mpg.de
    User your_username
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
```

**How it works:**
- First connection: ~1-2 seconds (establishes master connection)
- Subsequent connections: ~100ms (reuses existing socket)
- `ControlPersist 600`: Keep connection alive for 10 minutes after last use

**Verify setup:**
```bash
# Check if master connection is active
ssh -O check epfl

# Manually close master connection
ssh -O exit epfl
```

#### Facility Exploration Commands

Once SSH is configured, explore facilities directly from the terminal:

```bash
# Execute commands on remote facility
uv run imas-codex epfl "python --version"
uv run imas-codex epfl "ls /common/tcv/codes"

# View session history
uv run imas-codex epfl --status

# Persist learnings when done
uv run imas-codex epfl --finish << 'EOF'
python:
  version: "3.9.21"
tools:
  rg: unavailable
paths:
  codes: /common/tcv/codes
EOF

# Or discard session
uv run imas-codex epfl --discard
```

#### ReAct Agents (Autonomous Discovery)

ReAct agents provide autonomous discovery and evaluation of remote resources using LlamaIndex and OpenRouter.

**Prerequisites:**
- SSH configured for the target facility (see above)
- OpenRouter API key in `.env`: `OPENROUTER_API_KEY=sk-or-...`
- Neo4j running: `imas-codex neo4j start`

**Wiki Discovery Pipeline:**

Discover and evaluate wiki pages in three phases:

```bash
# Full discovery (crawl + score in one command)
imas-codex wiki discover epfl

# Or run phases separately for more control:

# Phase 1: Fast link crawling (no LLM, builds graph structure)
imas-codex wiki crawl epfl --max-pages 500

# Phase 2: Agent-based scoring (evaluates pages using graph metrics)
imas-codex wiki score epfl -v  # -v for verbose agent reasoning

# Phase 3: Ingest high-score pages
imas-codex wiki ingest epfl --min-score 0.7

# Check progress
imas-codex wiki status epfl
```

**Model Configuration:**

Models are configured centrally in `pyproject.toml`:

```toml
[tool.imas-codex.models]
discovery = "anthropic/claude-haiku-4.5"    # Fast crawl/discover
scoring = "anthropic/claude-sonnet-4.5"     # Accurate evaluation
enrichment = "google/gemini-3-pro-preview"  # Physics understanding
```

**Cost Control:**

Discovery uses a cost budget (default $10) tracked via OpenRouter:

```bash
# Set lower cost limit for testing
imas-codex wiki discover epfl --cost-limit 2.0
```

**Graph-Driven Workflow:**

The pipeline is graph-driven - it persists progress to Neo4j so you can:
- Resume interrupted crawls
- Run scoring on pages crawled in previous sessions
- Track which pages are queued, scored, skipped, or ingested

### Docker Setup

Run locally in a container (pre-built indexes included):

```bash
docker run -d \
  --name imas-codex \
  -p 8000:8000 \
  ghcr.io/iterorganization/imas-codex:latest-streamable-http

# Optional: verify
docker ps --filter name=imas-codex --format "table {{.Names}}\t{{.Status}}"
```

VS Code (`.vscode/mcp.json`):

```json
{
  "servers": {
    "imas-codex-docker": { "type": "http", "url": "http://localhost:8000/mcp" }
  }
}
```

Claude Desktop:

```json
{
  "mcpServers": {
    "imas-codex-docker": {
      "command": "npx",
      "args": ["mcp-remote", "http://localhost:8000/mcp"]
    }
  }
}
```

### Slurm / HPC (STDIO)

Helper script: `scripts/imas_codex_slurm_stdio.sh`

VS Code (`.vscode/mcp.json`, JSONC ok):

```jsonc
{
  "servers": {
    "imas-slurm-stdio": {
      "type": "stdio",
      "command": "scripts/imas_codex_slurm_stdio.sh"
    }
  }
}
```

Launch behavior:

1. If `SLURM_JOB_ID` present → start inside current allocation.
2. Else requests node with `srun --pty` then starts server (unbuffered stdio).

Resource tuning (export before client starts):

| Variable                     | Purpose                                    | Default         |
| ---------------------------- | ------------------------------------------ | --------------- |
| `IMAS_CODEX_SLURM_TIME`      | Walltime                                   | `08:00:00`      |
| `IMAS_CODEX_SLURM_CPUS`      | CPUs per task                              | `1`             |
| `IMAS_CODEX_SLURM_MEM`       | Memory (e.g. `4G`)                         | Slurm default   |
| `IMAS_CODEX_SLURM_PARTITION` | Partition                                  | Cluster default |
| `IMAS_CODEX_SLURM_ACCOUNT`   | Account/project                            | User default    |
| `IMAS_CODEX_SLURM_EXTRA`     | Extra raw `srun` flags                     | (empty)         |
| `IMAS_CODEX_USE_ENTRYPOINT`  | Use `imas-codex` entrypoint vs `python -m` | `0`             |

Example:

```bash
export IMAS_CODEX_SLURM_TIME=02:00:00
export IMAS_CODEX_SLURM_CPUS=4
export IMAS_CODEX_SLURM_MEM=8G
export IMAS_CODEX_SLURM_PARTITION=compute
```

Direct CLI:

```bash
scripts/imas_codex_slurm_stdio.sh --ids-filter "core_profiles equilibrium"
```

Why STDIO? Avoids opening network ports; all traffic rides the existing `srun` pseudo-TTY.

---

## Example IMAS Queries

Once you have the IMAS Codex server configured, you can interact with it using natural language queries. Use the `@imas` prefix to direct queries to the IMAS server:

### Basic Search Examples

```text
Find data paths related to plasma temperature
Search for electron density measurements
What data is available for magnetic field analysis?
Show me core plasma profiles
```

### Physics Concept Exploration

```text
Explain what equilibrium reconstruction means in plasma physics
What is the relationship between pressure and magnetic fields?
How do transport coefficients relate to plasma confinement?
Describe the physics behind current drive mechanisms
```

### Data Structure Analysis

```text
Analyze the structure of the core_profiles IDS
What are the relationships between equilibrium and core_profiles?
Show me identifier schemas for transport data
Export bulk data for equilibrium, core_profiles, and transport IDS
```

### Advanced Queries

```text
Find all paths containing temperature measurements across different IDS
What physics domains are covered in the IMAS data dictionary?
Show me measurement dependencies for fusion power calculations
Explore cross-domain relationships between heating and confinement
```

### Workflow and Integration

```text
How do I access electron temperature profiles from IMAS data?
What's the recommended workflow for equilibrium analysis?
Show me the branching logic for diagnostic identifier schemas
Export physics domain data for comprehensive transport analysis
```

The IMAS Codex server provides 8 specialized tools for different types of queries:

- **Search**: Natural language and structured search across IMAS data paths
- **Explain**: Physics concepts with IMAS context and domain expertise
- **Overview**: General information about IMAS structure and available data
- **Analyze**: Detailed structural analysis of specific IDS
- **Explore**: Relationship discovery between data paths and physics domains
- **Identifiers**: Exploration of enumerated options and branching logic
- **Bulk Export**: Comprehensive export of multiple IDS with relationships
- **Domain Export**: Physics domain-specific data with measurement dependencies

## Documentation Search

The server includes integrated search for documentation libraries with IMAS-Python as the default indexed library. This feature enables AI assistants to search across documentation sources using natural language queries.

### Available MCP Tool Functions

- **`search_docs`**: Search any indexed documentation library

  - Parameters: `query` (required), `library` (optional), `limit` (optional, 1-20), `version` (optional)
  - Supports multiple documentation libraries
  - Returns comprehensive version and library information

- **`search_imas_python_docs`**: Search specifically in IMAS-Python documentation

  - Parameters: `query` (required), `limit` (optional), `version` (optional)
  - Automatically uses IMAS-Python library
  - IMAS-specific search optimizations

- **`list_docs`**: List all available documentation libraries or get versions for a specific library
  - Parameters: `library` (optional)
  - When no library specified: returns list of all available libraries
  - When library specified: returns versions for that specific library
  - Shows all indexed versions and latest

### CLI Commands

- **`add-docs`**: Add new documentation libraries via command line
  - Usage: `add-docs LIBRARY URL [OPTIONS]`
  - Requires: OpenRouter API key and embedding model configuration
  - Supports custom max-pages and max-depth settings
  - Includes `--ignore-errors` flag (enabled by default) to handle problematic pages gracefully
  - See examples below

### Documentation Search Examples

```text
# Search IMAS-Python documentation
search_imas_python_docs "equilibrium calculations"
search_imas_python_docs "IDS data structures" limit=5
search_imas_python_docs "magnetic field" version="2.0.1"

# Search any documentation library
search_docs "neural networks" library="numpy"
search_docs "data visualization" library="matplotlib"

# List all available libraries
list_docs

# Get versions for specific library
list_docs "imas-python"

# Add new documentation using CLI
add-docs udunits https://docs.unidata.ucar.edu/udunits/current/
add-docs pandas https://pandas.pydata.org/docs/ --version 2.0.1 --max-pages 500
add-docs imas-python https://imas-python.readthedocs.io/en/stable/ --no-ignore-errors
```

### Setup Instructions

#### Production (Docker)

```bash
docker-compose up --build
```

#### Local Development

```bash
# Start IMAS Codex server
python -m imas_codex
```

#### API Key Configuration

For embedding generation capabilities (during build), you'll need an OpenRouter API key:

**For Local Development:**

```bash
# Set up environment variables (create .env file from env.example)
cp env.example .env
# Edit .env with your OpenRouter API key
```

**For CI/CD (GitHub Actions):**

1. Go to your repository settings: `Settings` → `Secrets and variables` → `Actions`
2. Add a new repository secret:
   - **Name**: `OPENAI_API_KEY`
   - **Value**: Your OpenRouter API key

> 📖 **Detailed Setup Guide:** See [.github/SECRETS_SETUP.md](.github/SECRETS_SETUP.md) for complete instructions on configuring GitHub repository secrets and troubleshooting.

**Build Behavior:**

- **With OPENAI_API_KEY**: Full embedding generation during build
- **Without OPENAI_API_KEY**: Uses local embedding model (all-MiniLM-L6-v2)
- The container works normally in both cases

**Local Docker Build:**

```bash
# Build with API key (for API-based embeddings)
docker build --build-arg OPENAI_API_KEY=your_key_here .

# Build without API key (uses local model)
docker build .
```

## Development

For local development and customization:

### Setup

```bash
# Clone repository
git clone https://github.com/iterorganization/imas-codex.git
cd imas-codex

# Install development dependencies (search index build takes ~8 minutes first time)
uv sync --all-extras
```

### Build Dependencies

This project requires additional dependencies during the build process that are not part of the runtime dependencies:

- **`imas-data-dictionary`** - Git development package, required only during wheel building for parsing latest DD changes
- **`rich`** - Used for enhanced console output during build processes

**For runtime:** The `imas-data-dictionaries` PyPI package is now a core dependency and provides access to stable DD versions (e.g., 4.0.0). This eliminates the need for the git package at runtime and ensures reproducible builds.

**For developers:** Build-time dependencies are included in the `[build-system.requires]` section for wheel building. The git package is only needed when building wheels with latest DD changes.

```bash
# Regular development - uses imas-data-dictionaries (PyPI)
uv sync --all-extras

# Set DD version for building (defaults to 4.0.0)
export IMAS_DD_VERSION=4.0.0
uv run build-schemas
```

**Location in configuration:**

- **Build-time dependencies**: Listed in `[build-system.requires]` in `pyproject.toml`
- **Runtime dependencies**: `imas-data-dictionaries>=4.0.0` in `[project.dependencies]`

**Note:** The `IMAS_DD_VERSION` environment variable controls which DD version is used for building schemas and embeddings. Docker containers have this set to `4.0.0` by default.

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

# Run the IMAS DD server locally (default: streamable-http on port 8000)
uv run imas-codex serve imas

# Run with stdio transport for MCP clients
uv run imas-codex serve imas --transport stdio

# Run the Agents server for facility exploration
uv run imas-codex serve agents
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

#### VS Code

The repository includes a `.vscode/mcp.json` file with pre-configured development server options. Use the `imas-local-stdio` configuration for local development.

#### Claude Desktop

Add to your config file:

```json
{
  "mcpServers": {
    "imas-local-dev": {
      "command": "uv",
      "args": ["run", "imas-codex", "serve", "imas", "--transport", "stdio"],
      "cwd": "/path/to/imas-codex"
    },
    "imas-agents-dev": {
      "command": "uv",
      "args": ["run", "imas-codex", "serve", "agents"],
      "cwd": "/path/to/imas-codex"
    }
  }
}
```

## How It Works

1. **Installation**: During package installation, the index builds automatically when the module first imports
2. **Build Process**: The system parses the IMAS data dictionary and creates comprehensive JSON files with structured data
3. **Embedding Generation**: Creates semantic embeddings using sentence transformers for advanced search capabilities
4. **Serialization**: The system stores indexes in organized subdirectories:
   - **JSON data**: `imas_codex/resources/schemas/` (LLM-optimized structured data)
   - **Embeddings cache**: Pre-computed sentence transformer embeddings for semantic search
5. **Import**: When importing the module, the pre-built index and embeddings load in ~1 second

## Optional Dependencies and Runtime Requirements

The IMAS Codex server now includes `imas-data-dictionaries` as a core dependency, providing stable DD version access (default: 4.0.0). The git development package (`imas-data-dictionary`) is used during wheel building when parsing latest DD changes.

### Package Installation Options

- **Runtime**: `uv add imas-codex` - Includes all transports (stdio, sse, streamable-http)
- **Full installation**: `uv add imas-codex` - Recommended for all users

### Data Dictionary Access

The system uses composable accessors to access IMAS Data Dictionary version and metadata:

1. **Environment Variable**: `IMAS_DD_VERSION` (highest priority) - Set to specify DD version (e.g., "4.0.0")
2. **Metadata File**: JSON metadata stored alongside indexes
3. **Index Name Parsing**: Extracts version from index filename
4. **Package Default**: Falls back to `imas-data-dictionaries` package (4.0.0)

This design ensures the server can:

- **Build indexes** using the version specified by `IMAS_DD_VERSION`
- **Run with pre-built indexes** using version metadata
- **Access stable DD versions** through `imas-data-dictionaries` PyPI package

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
docker run -d -p 8000:8000 ghcr.io/iterorganization/imas-codex:latest

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
[build-deploy-badge]: https://img.shields.io/github/actions/workflow/status/simon-mcintosh/imas-codex/test.yml?branch=main&color=brightgreen&label=CI%2FCD
[build-deploy-link]: https://github.com/iterorganization/imas-codex/actions/workflows/test.yml
[codecov-badge]: https://codecov.io/gh/simon-mcintosh/imas-codex/graph/badge.svg
[codecov-link]: https://codecov.io/gh/simon-mcintosh/imas-codex
[docs-badge]: https://img.shields.io/badge/docs-online-brightgreen
[docs-link]: https://simon-mcintosh.github.io/imas-codex/
[asv-badge]: https://img.shields.io/badge/ASV-Benchmarks-blue?style=flat&logo=speedtest&logoColor=white
[asv-link]: https://simon-mcintosh.github.io/imas-codex/benchmarks/

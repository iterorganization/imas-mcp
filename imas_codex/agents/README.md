# Agents Module

The `imas_codex.agents` module provides two complementary agent systems:

1. **MCP Server** (`AgentsServer`) - Tool-based exploration via Model Context Protocol
2. **LlamaIndex ReActAgents** - Autonomous agents with tool-use capabilities

## Quick Start

### CLI Usage

```bash
# Run a quick agent task
imas-codex agent run "Describe what \RESULTS::ASTRA is used for"

# Use a specific agent type
imas-codex agent run "Find IMAS paths for electron temperature" --type mapping

# Batch enrich TreeNode paths
imas-codex agent enrich "\RESULTS::IBS" "\RESULTS::ASTRA"
```

### Python API

```python
from imas_codex.agents import (
    get_enrichment_agent,
    get_mapping_agent,
    get_exploration_agent,
    run_agent,
    quick_agent_task,
)

# Quick one-liner
result = await quick_agent_task(
    "Describe what \\RESULTS::ASTRA is used for",
    agent_type="enrichment"
)

# Full control
agent = get_enrichment_agent(verbose=True)
result = await run_agent(agent, "Analyze this path...")
```

## Agent Types

| Agent | Use Case | System Prompt Focus |
|-------|----------|---------------------|
| **Enrichment** | TreeNode metadata generation | Physics accuracy, TCV-specific knowledge |
| **Mapping** | IMAS ↔ MDSplus discovery | Semantic equivalence, unit compatibility |
| **Exploration** | Facility exploration | Systematic discovery, graph persistence |

## Available Tools

Agents use `get_exploration_tools()` by default for fast startup:

| Tool | Description |
|------|-------------|
| `query_neo4j` | Execute Cypher queries on the knowledge graph |
| `ssh_mdsplus_query` | Query MDSplus database via SSH for node metadata |
| `ssh_command` | Execute arbitrary SSH commands on remote facilities |
| `search_code_examples` | Find usage patterns in ingested code |
| `get_tree_structure` | View TreeNode hierarchy from the graph |

### IMAS DD Tools (Optional)

IMAS DD tools have ~30s startup cost for embedding model loading.
Add them explicitly if needed:

```python
from imas_codex.agents import get_exploration_tools, get_imas_tools

# Fast agents (default)
tools = get_exploration_tools()

# Add IMAS tools if needed (slow startup)
tools = get_exploration_tools() + get_imas_tools()
```

| IMAS Tool | Description |
|-----------|-------------|
| `search_imas_paths` | Search IMAS Data Dictionary semantically |
| `check_imas_paths` | Fast path validation |
| `fetch_imas_paths` | Full path documentation |
| `list_imas_paths` | Structure exploration |
| `get_imas_overview` | High-level DD summary |
| `search_imas_clusters` | Find related paths |

## Configuration

### LLM Setup

Agents use OpenRouter for LLM access. Configure via environment:

```bash
# Required: OpenRouter API key (in .env)
OPENAI_API_KEY=sk-or-v1-...
```

### Model Selection

Default: `google/gemini-3-flash-preview` (1M context, cost-effective)

```python
from imas_codex.agents import get_llm, MODELS

# Use preset
llm = get_llm(model=MODELS["gemini-pro"])

# Or specific model ID
llm = get_llm(model="anthropic/claude-sonnet-4")
```

Available presets in `MODELS`:
- `gemini-flash` - google/gemini-3-flash-preview (default)
- `gemini-2.5-flash` - google/gemini-2.5-flash
- `gemini-pro` - google/gemini-2.5-pro-preview
- `claude-sonnet` - anthropic/claude-sonnet-4
- `gpt-4o` - openai/gpt-4o

## Architecture

```
imas_codex/agents/
├── __init__.py      # Public API exports
├── llm.py           # OpenRouter LLM configuration
├── tools.py         # Reusable FunctionTools
├── react.py         # ReActAgent configurations
├── server.py        # MCP server (AgentsServer)
├── prompt_loader.py # Prompt template loading
└── prompts/         # System prompts (YAML)
```

### MCP Server vs ReActAgents

| Feature | MCP Server | ReActAgents |
|---------|------------|-------------|
| Orchestration | External (Claude, Copilot) | Self (LlamaIndex) |
| Tool calls | One at a time | Multi-step reasoning |
| State | Stateless | Conversation context |
| Use case | Interactive exploration | Batch processing |

## Examples

### Enrichment Agent

```python
from imas_codex.agents import get_enrichment_agent, run_agent

agent = get_enrichment_agent(verbose=True)

result = await run_agent(agent, """
Analyze the TreeNode path \\RESULTS::LIUQE.

1. Query the graph for existing metadata
2. Search code examples for usage patterns  
3. Query MDSplus via SSH if needed
4. Provide a physics description
""")
```

### Mapping Agent

```python
from imas_codex.agents import get_mapping_agent, run_agent

agent = get_mapping_agent()

result = await run_agent(agent, """
Find IMAS equivalents for these TCV TreeNodes:
- \\RESULTS::THOMSON:NE (electron density from Thomson)
- \\RESULTS::LIUQE:IP (plasma current)
- \\RESULTS::ASTRA:TE (electron temperature profile)

For each, provide:
- IMAS path
- Unit compatibility
- Required transformations
""")
```

### Exploration Agent

```python
from imas_codex.agents import get_exploration_agent, run_agent

agent = get_exploration_agent()

result = await run_agent(agent, """
Explore the CXRS diagnostic data in TCV:
1. Find TreeNodes containing 'CXRS' in the graph
2. List the structure using SSH if needed
3. Identify which IMAS IDS this maps to
4. Document key quantities and their units
""")
```

### Batch Enrichment

```python
from imas_codex.agents import batch_enrich_paths

paths = [
    "\\RESULTS::IBS",
    "\\RESULTS::ASTRA",
    "\\RESULTS::LIUQE",
]

results = await batch_enrich_paths(paths, tree_name="results", verbose=True)

for r in results:
    print(f"{r.path}: {r.description[:100]}...")
```

## Extending Agents

### Custom Tools

```python
from llama_index.core.tools import FunctionTool
from imas_codex.agents import create_agent, AgentConfig, get_exploration_tools

def my_custom_tool(query: str) -> str:
    """Custom analysis tool."""
    return f"Analyzed: {query}"

custom_tool = FunctionTool.from_defaults(
    fn=my_custom_tool,
    name="custom_analysis",
    description="Custom analysis for specific use case",
)

# Add to standard tools
tools = get_exploration_tools() + [custom_tool]

config = AgentConfig(
    name="custom-agent",
    system_prompt="You are a specialized agent...",
    tools=tools,
)

agent = create_agent(config)
```

### Custom System Prompts

```python
from imas_codex.agents import AgentConfig, create_agent

config = AgentConfig(
    name="transport-agent",
    system_prompt="""You are an expert in tokamak transport physics.
    Focus on heat and particle transport analysis.
    Use ASTRA, GENE, and TGLF references when relevant.""",
    model="google/gemini-3-flash-preview",
    temperature=0.2,  # Lower for more deterministic output
)

agent = create_agent(config)
```

## Related Documentation

- [AGENTS.md](../../../AGENTS.md) - Project-wide agent guidelines
- [plans/LLAMAINDEX_AGENTS.md](../../../plans/LLAMAINDEX_AGENTS.md) - Architecture and roadmap
- [config/README.md](../config/README.md) - Facility exploration guide
- [plans/MDSPLUS_INGESTION.md](../../../plans/MDSPLUS_INGESTION.md) - Tree ingestion workflow

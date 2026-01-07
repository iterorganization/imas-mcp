# LlamaIndex Agents Architecture

> **Module**: `imas_codex.agents`

## Overview

The agent system uses LlamaIndex ReActAgent for autonomous exploration, metadata enrichment, and IMAS mapping discovery.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     imas_codex.agents                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Enrichment  │  │   Mapping   │  │  Discovery  │  Agents     │
│  │   Agent     │  │    Agent    │  │    Agent    │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                      │
│                    ┌─────▼─────┐                                │
│                    │   Tools   │                                │
│                    └─────┬─────┘                                │
│         ┌────────────────┼────────────────┐                     │
│         │                │                │                     │
│   ┌─────▼─────┐   ┌──────▼──────┐  ┌──────▼──────┐             │
│   │  Neo4j    │   │    SSH      │  │   Search    │             │
│   │  Graph    │   │  Commands   │  │   (IMAS)    │             │
│   └───────────┘   └─────────────┘  └─────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## CLI Commands

```bash
# Ad-hoc agent task
uv run imas-codex agent run "describe path \\RESULTS::PSI"

# Batch enrichment
uv run imas-codex agent enrich --tree results --batch-size 50

# Mark nodes for re-enrichment
uv run imas-codex agent mark-stale "\\RESULTS::*"
```

## Tool Library

| Tool | Purpose |
|------|---------|
| `query_neo4j` | Execute Cypher queries |
| `ssh_command` | Run SSH commands on facility |
| `ssh_mdsplus_query` | Query MDSplus data |
| `search_code_examples` | Semantic code search |
| `search_imas_paths` | Search IMAS DD |
| `ingest_nodes` | Batch create graph nodes |

## ReAct Workflow

```python
from imas_codex.agents import create_enrichment_agent

agent = create_enrichment_agent(model="gemini-2.0-flash")
result = await agent.run("Enrich these TreeNodes: \\RESULTS::PSI, \\RESULTS::IP")
```

## Batch Processing

The enrichment agent uses smart batching:

1. **Group by subtree** - Nodes in same subtree share context
2. **Include code examples** - For nodes with DataReference links
3. **Confidence filtering** - Only persist high-confidence results

## MCP Server

The agents MCP server exposes tools for VS Code chat:

```bash
# Start agents server
uv run imas-codex serve agents

# In VS Code, tools available via MCP:
# - cypher(query)
# - ingest_nodes(type, data)
# - private(facility, data?)
# - get_graph_schema()
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | For ReAct agents |
| `GOOGLE_API_KEY` | For Gemini models |

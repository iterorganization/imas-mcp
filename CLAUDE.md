# Claude Code Guidelines

All core rules are in [AGENTS.md](AGENTS.md). This file covers Claude Code (CLI) specifics only.

## Environment Variables

The `.env` file contains secrets. **NEVER** expose or commit it.

## Project Structure

```
agents/                  # Domain-specific agent workflows
imas_codex/
├── agentic/             # LlamaIndex agents, MCP server
├── graph/               # Neo4j knowledge graph
├── schemas/             # LinkML schemas (source of truth)
├── code_examples/       # Code ingestion pipeline
├── remote/              # SSH execution, tool installation
└── ...
```

## Domain Workflows

| Workflow | Agent | Documentation |
|----------|-------|---------------|
| Facility exploration | Explore | [agents/explore.md](agents/explore.md) |
| Development | Develop | [agents/develop.md](agents/develop.md) |
| Code ingestion | Ingest | [agents/ingest.md](agents/ingest.md) |
| Graph operations | Graph | [agents/graph.md](agents/graph.md) |

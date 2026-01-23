# Custom Agents

This directory contains domain-specific agent instructions. These files are optional deep-dives - the main [AGENTS.md](../AGENTS.md) contains all critical rules.

## Architecture

```
AGENTS.md                        # Primary source of truth (all agents read this)
agents/                          # Domain-specific deep-dives (optional)
├── README.md                    # This file
├── explore.md                   # Remote facility exploration
├── develop.md                   # Development workflows
├── ingest.md                    # Code ingestion pipeline
├── graph.md                     # Knowledge graph operations
└── onboard.md                   # New facility onboarding guide

.github/agents/                  # VS Code shims (thin wrappers)
├── explore.agent.md
├── develop.agent.md
├── ingest.agent.md
└── graph.agent.md
```

## Agent Selection

| Agent | Purpose | Toolset |
|-------|---------|---------|
| Explore | Remote facility discovery | Read-only + MCP |
| Develop | Code development | Standard + MCP |
| Ingest | Code ingestion pipeline | Core + MCP |
| Graph | Knowledge graph operations | Core + MCP |

## Tool Sets

VS Code agents use custom tool sets defined in `.vscode/toolsets.jsonc`:

- **readonly**: Read-only workspace access (codebase, readFile, listDirectory, fileSearch, textSearch, fetch, problems, usages, changes, terminalLastCommand)
- **core**: readonly + terminal access (runInTerminal, getTerminalOutput)
- **standard**: core + file editing and tests (editFiles, createFile, createDirectory, runTests, testFailure)
- **full**: Everything except notebooks

## MCP Tools

All agents have access to Codex MCP tools:

| Tool | Purpose |
|------|---------|
| `python()` | Persistent Python REPL with pre-loaded utilities |
| `get_graph_schema()` | Neo4j schema for Cypher generation |
| `add_to_graph()` | Schema-validated batch node creation |
| `update_facility_infrastructure()` | Update private facility data |
| `get_facility_infrastructure()` | Read private facility data |
| `add_exploration_note()` | Add timestamped exploration note |

## Handoffs

Handoffs create clickable buttons after a chat response completes:

```
explore → ingest (Queue discovered files)
explore → graph (Persist to graph)
develop → graph (Update schema)
ingest → explore (Discover more files)
```

## Creating New Agents

1. Create `agents/your-agent.md` with domain-specific workflows
2. Create `.github/agents/your-agent.agent.md` with VS Code frontmatter:

```markdown
---
name: Your Agent
description: Brief description shown in dropdown
tools:
  - readonly
  - codex/*
handoffs:
  - label: Next Step
    agent: other-agent
---

# Your Agent

See [full instructions](../../agents/your-agent.md) for complete workflows.
```

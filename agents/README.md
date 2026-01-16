# Custom Agents

This directory contains portable agent instructions that work across IDEs. The `.github/agents/*.agent.md` files are VS Code shims that reference these files.

## Architecture

```
agents/                          # Portable content (source of truth)
├── README.md                    # This file - tool taxonomy, handoffs
├── explore.md                   # Remote facility exploration
├── develop.md                   # Development workflows
├── ingest.md                    # Code ingestion pipeline
└── graph.md                     # Knowledge graph operations

.github/agents/                  # VS Code shims (thin wrappers)
├── explore.agent.md             # Frontmatter + link to agents/explore.md
├── develop.agent.md
├── ingest.agent.md
└── graph.agent.md
```

## Tool Taxonomy

### VS Code Built-in Tool Sets

| Set | Tools | Use Case |
|-----|-------|----------|
| `#edit` | editFiles, createFile, createDirectory, editNotebook | File modifications |
| `#search` | codebase, fileSearch, textSearch, usages | Code search |
| `#runCommands` | runInTerminal, getTerminalOutput | Terminal execution |
| `#runTasks` | runTask, createAndRunTask, getTaskOutput | VS Code tasks |
| `#runNotebooks` | runCell, editNotebook, getNotebookSummary, readNotebookCellOutput | Jupyter |

### Custom Tool Sets (`.vscode/toolsets.jsonc`)

```yaml
readonly:   # Read-only workspace access
  - codebase, readFile, listDirectory, fileSearch, textSearch
  - fetch, problems, usages, changes, terminalLastCommand

core:       # + Terminal access, no file editing
  - readonly tools
  - runInTerminal, getTerminalOutput

standard:   # + File editing and tests
  - core tools
  - editFiles, createFile, createDirectory
  - runTests, testFailure

full:       # Everything except notebooks
  - standard tools
  - runTask, runVscodeCommand, githubRepo, extensions
```

### MCP Tools

Reference MCP server tools in agent frontmatter:

```yaml
tools:
  - codex/*           # All tools from codex server
  - codex/python      # Specific tool
  - imas/*            # All IMAS DD tools
```

**Codex MCP Tools:**
| Tool | Purpose |
|------|---------|
| `codex/python` | Persistent Python REPL with pre-loaded utilities |
| `codex/get_graph_schema` | Neo4j schema for Cypher generation |
| `codex/ingest_nodes` | Schema-validated batch node creation |
| `codex/private` | Read/update sensitive infrastructure data |

**IMAS MCP Tools:**
| Tool | Purpose |
|------|---------|
| `imas/search_imas` | Semantic search across Data Dictionary |
| `imas/fetch_imas` | Full documentation for paths |
| `imas/list_imas` | List IDS structure |
| `imas/check_imas` | Validate path existence |

## Agent Tool Assignments

| Agent | Tool Set | MCP | Purpose |
|-------|----------|-----|---------|
| explore | readonly | codex/* | Remote facility discovery |
| develop | standard | codex/* | Code development with guardrails |
| ingest | core | codex/* | File ingestion pipeline |
| graph | core | codex/* | Knowledge graph operations |

## Handoffs

Handoffs create clickable buttons after a chat response completes. They do not automatically delegate - the user must click to transition.

```yaml
handoffs:
  - label: "Button Text"      # Displayed on button
    agent: target-agent       # Agent to switch to
    prompt: "Pre-filled text" # Optional prompt
    send: false               # false = user reviews, true = auto-submit
```

**Handoff Flow:**
```
explore → ingest (Queue discovered files)
explore → graph (Persist to graph)
develop → graph (Update schema)
ingest → explore (Discover more files)
```

## Creating New Agents

### 1. Create Portable Content

Create `agents/your-agent.md` with:
- Role description
- MCP tool usage examples
- Workflow documentation
- Restrictions and guardrails

### 2. Create VS Code Shim

Create `.github/agents/your-agent.agent.md`:

```markdown
---
name: Your Agent
description: Brief description shown in dropdown
tools:
  - readonly           # Use custom tool set
  - codex/*            # Add MCP tools
handoffs:
  - label: Next Step
    agent: other-agent
    prompt: Continue with...
    send: false
---

# Your Agent

Brief role summary and restrictions.

See [full instructions](../../agents/your-agent.md) for complete workflows.
```

### 3. Test

1. Agent appears in VS Code dropdown
2. Tool restrictions are enforced (disabled tools not available)
3. Handoff buttons appear after response
4. MCP tools are accessible

## External CLI Tools

VS Code agents cannot directly specify external CLI tools (rg, fd, scc) in frontmatter. Instead:

1. Document preferred tools in agent instructions
2. Agent uses `runInTerminal` to invoke them
3. For remote facilities, use `codex/python` with `run()` (auto-detects local vs SSH):

```python
python("print(run('~/bin/rg -l \"pattern\" /path', facility='epfl'))")
```

## Globally Excluded Tools

These tools are excluded from all custom agents:
- `newJupyterNotebook` - Use standard workflows
- `editNotebook` - Notebooks not used in this project
- `runCell` - Notebooks not used in this project

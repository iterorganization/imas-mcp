# Agent Teams Infrastructure

Collaborative Claude Code agent teams for IMAS mapping discovery across fusion facilities (TCV, JET, JT60SA). Reusable infrastructure for future scientific agent teams.

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                   Lead Session                        │
│   (Claude Code in imas-codex workspace)               │
│                                                       │
│   ┌───────────┐  ┌───────────┐  ┌───────────┐       │
│   │ IMAS      │  │ Facility  │  │ Evidence   │  ...  │
│   │ Expert    │  │ Analyst   │  │ Researcher │       │
│   └─────┬─────┘  └─────┬─────┘  └─────┬─────┘       │
│         │              │              │               │
│         v              v              v               │
│   ┌─────────────────────────────────────────┐        │
│   │        LiteLLM Proxy (:4000)            │        │
│   │   Cost tracking · Model routing         │        │
│   └─────────────────┬───────────────────────┘        │
│                     │                                 │
│   ┌─────────────────v───────────────────────┐        │
│   │         OpenRouter → Anthropic           │        │
│   │   Opus 4 · Sonnet 4 · Haiku 4           │        │
│   └─────────────────┬───────────────────────┘        │
│                     │                                 │
│   ┌─────────────────v───────────────────────┐        │
│   │         Langfuse Observability           │        │
│   │   Per-facility cost · Prompt cache hits  │        │
│   └─────────────────────────────────────────┘        │
│                                                       │
│   ┌─────────────────────────────────────────┐        │
│   │          Neo4j Knowledge Graph           │        │
│   │   Shared state · Proposals · Evidence    │        │
│   └─────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Dynamic teams, not pre-defined agents.** Agent teams are launched via natural language — you tell Claude to "create a team with N teammates focused on X" and it spawns them. The `.claude/agents/` files define reusable subagent roles, not team members.

2. **Graph as shared state machine.** Agents write proposals, evidence, and results to Neo4j — not to files or context windows. Any team can resume from any previous team's work.

3. **Skills over CLAUDE.md.** Team members get domain knowledge through compact skill files (~2-4k tokens each) rather than the full AGENTS.md (15k+ tokens). The spawn prompt controls which skills are loaded.

4. **Codegen over exploration.** Agents write Python scripts for testing, not interactive exploration. This keeps tool call output manageable (200 tokens for a script result vs 10k for inline graph data).

---

## Setup

### Prerequisites

```bash
# Core install
uv sync

# Serve extra (LiteLLM proxy + embedding server)
pip install -e ".[serve]"
# Or for development:
uv sync  # dev group includes all extras
```

### Environment Variables

Add to `.env`:

```bash
# Required
OPENROUTER_API_KEY=your_openrouter_api_key

# LiteLLM Proxy
LITELLM_MASTER_KEY=sk-litellm-imas-codex
LITELLM_PROXY_URL=http://localhost:4000

# Langfuse (optional but recommended for cost tracking)
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Start the LiteLLM Proxy

```bash
# Foreground (development)
imas-codex serve llm start

# As systemd service (production)
imas-codex serve llm service install
imas-codex serve llm service start

# Verify
imas-codex serve llm status
```

The proxy routes all LLM calls through a single endpoint with:
- **Model aliases**: `opus`, `sonnet`, `haiku`, `scoring` → OpenRouter models
- **Credential management**: One API key, referenced by all models
- **Cost tracking**: Every call logged to Langfuse with facility metadata
- **Prompt caching**: Anthropic cache_control directives preserved through the chain

### Verify Prompt Caching

```bash
# Via proxy (preferred)
uv run python scripts/test_prompt_caching.py

# Direct comparison
uv run python scripts/test_prompt_caching.py --direct

# Pin provider if cache hit rates are low
uv run python scripts/test_prompt_caching.py --pin-provider
```

---

## File Layout

```
.claude/
├── settings.json              # Project settings (teams enabled, permissions)
├── agents/
│   ├── graph-querier.md       # Reusable: Cypher queries, graph writes
│   └── facility-explorer.md   # Reusable: SSH, data access, signal enumeration
└── skills/
    ├── graph-queries.md       # Cypher patterns for common operations
    ├── schema-summary.md      # Condensed graph schema (~2k tokens)
    ├── mapping-workflow.md    # Proposal → validation → persistence workflow
    └── facility-access.md     # SSH patterns, data system access per facility
```

### Subagents vs Team Members

| Concept | Definition | How Configured |
|---------|-----------|----------------|
| **Subagent** | Reusable role that Claude auto-delegates to during sessions | `.claude/agents/*.md` with YAML frontmatter |
| **Team member** | Independent session spawned by lead for a specific task | Dynamic — spawn prompt defines scope |
| **Skill** | Domain knowledge injected into subagents/teams | `.claude/skills/*.md` — compact, focused context |

---

## Graph Schema

The agent teams infrastructure extends the knowledge graph with these types:

### New Enums

| Enum | Values | Purpose |
|------|--------|---------|
| `MappingStatus` | proposed → endorsed/contested → validated/rejected | Mapping lifecycle |
| `EvidenceType` | wiki_documentation, code_reference, data_validation, unit_analysis, sign_convention, shape_analysis, expert_knowledge | Evidence categorization |
| `AgentSessionStatus` | running, completed, failed, budget_exhausted | Session tracking |

### New Node Types

| Type | Key Fields | Purpose |
|------|-----------|---------|
| `MappingProposal` | signal_id, imas_path_id, status, confidence | Agent-proposed mapping |
| `MappingEvidence` | evidence_type, content, supports_mapping | Evidence for/against mapping |
| `AgentSession` | facility_id, session_type, budget_used_usd | Team session tracking |

### Extended Types

- **`IMASMapping`**: Added `status` (MappingStatus), `proposed_by`, `evidence`, `rejection_reason`, `validated_by` fields for full audit trail.

### Relationships

```
(MappingProposal)-[:PROPOSES_SOURCE]->(FacilitySignal)
(MappingProposal)-[:PROPOSES_TARGET]->(IMASPath)
(MappingProposal)-[:HAS_EVIDENCE]->(MappingEvidence)
(MappingProposal)-[:PROPOSED_BY]->(AgentSession)
(MappingEvidence)-[:CREATED_BY]->(AgentSession)
(IMASMapping)-[:HAS_EVIDENCE]->(MappingEvidence)
(AgentSession)-[:AT_FACILITY]->(Facility)
```

---

## Mapping Workflow

### Lifecycle

```
1. Enumerate    → FacilitySignals (checked, target domain)
2. Discover     → Candidate IMASPaths via semantic search
3. Propose      → MappingProposal nodes with initial confidence
4. Evidence     → MappingEvidence from wiki, code, data tests
5. Endorse      → Raise status when evidence agrees
6. Validate     → Test against real shot data
7. Persist      → Promote to IMASMapping
```

### Team Composition

For IMAS mapping discovery, the recommended team structure:

| Role | Model | Capabilities |
|------|-------|-------------|
| **IMAS Expert** | Opus | DD lookups, target path identification, unit analysis |
| **Facility Analyst** | Opus | SSH to facility, signal enumeration, data access testing |
| **Evidence Researcher** | Sonnet | Wiki/code chunk semantic search, documentation mining |
| **Validation Tester** | Opus | Python scripts for data access, unit conversion, shape checks |

### Launching a Team

From a Claude Code session in the imas-codex workspace:

```
Create an agent team for TCV IMAS mapping discovery.

Teammates:
1. IMAS Expert — knows the Data Dictionary structure. Uses imas-ddv4 MCP
   server to look up IDS paths, types, units, and coordinates. Proposes
   target paths for facility signals.
2. Facility Analyst — knows TCV data systems. SSHs to TCV to enumerate
   signals, test data access, check units and shapes.
3. Evidence Researcher — searches wiki and code chunks in the knowledge
   graph for mapping documentation, tables, and existing code.
4. Validation Tester — writes Python scripts to test proposed mappings
   against real shot data. Checks unit conversions, sign conventions.

Use Opus for all teammates.
Load the mapping-workflow and graph-queries skills.
Start with equilibrium signals.
Each teammate should write code to test hypotheses — query via MCP and SSH only.
Require plan approval before persisting validated mappings.
Wait for all teammates to complete before synthesizing results.
```

**The spawn prompt IS the isolation mechanism.** Team members work through MCP tools and SSH — they don't read project source files.

### Resuming Interrupted Work

If a team is interrupted, query the graph for in-progress proposals:

```
There are 23 MappingProposals for TCV with status "proposed" that need validation.
Create a team to validate these proposals and either endorse or reject them.
```

The graph serves as the state machine — all progress is durable.

---

## Cost Management

### Budget Allocation ($1000 total)

| Item | Budget | Notes |
|------|--------|-------|
| TCV mapping team | $150 | First run, includes iteration |
| JET mapping team | $200 | Larger signal set |
| JT60SA mapping team | $120 | Benefit from refined approach |
| Development & debugging | $200 | Infrastructure setup, prompt refinement |
| Prompt caching overhead | $80 | Cache write costs (1.25x for first calls) |
| Buffer | $250 | Unexpected costs, retries, edge cases |

### Why $1000 is Practical

With prompt caching:
- Opus input: $15/M tokens, cache read: $1.50/M (90% savings)
- System prompts + skill content: ~4k tokens per agent, cached after first call
- With 80% cache hit rate: effective input cost drops to ~$3/M
- Per-facility mapping: ~$35-70 in LLM cost

If cost pressure builds, shift Evidence Researcher and Facility Analyst to Sonnet ($15/M output vs Opus $75/M).

### Enforcement Layers

| Layer | Mechanism | Granularity |
|-------|-----------|-------------|
| OpenRouter | Per-key spend limits | Total budget |
| LiteLLM proxy | `max_budget` per virtual key | Per-session |
| Neo4j graph | `AgentSession.budget_limit_usd` | Per-team-run |
| Langfuse | Real-time cost dashboard + alerts | Per-request |

### Monitoring

```bash
# Check proxy health and model availability
imas-codex serve llm status

# Query session cost from graph
python -c "
from imas_codex.agentic.tools import query
print(query('''
  MATCH (s:AgentSession {facility_id: 'tcv'})
  RETURN s.id, s.budget_used_usd, s.proposals_validated
  ORDER BY s.started_at DESC LIMIT 5
'''))
"
```

---

## Phased Facility Rollout

| Phase | Facility | Signal Count | Expected Cost | Duration |
|-------|----------|-------------|---------------|----------|
| 1 | TCV | ~300 | $100-150 | 1-2 days |
| 2 | JET | ~500 | $150-200 | 2-3 days |
| 3 | JT60SA | ~200 | $80-120 | 1-2 days |

Start with TCV — smallest signal set, best SSH access, most wiki documentation. Validate approach, refine prompts, then scale.

### Per-Phase Checklist

1. Start LiteLLM proxy: `imas-codex serve llm start`
2. Ensure Neo4j is running: `imas-codex graph db status -g <facility>`
3. Verify checked signals exist:
   ```cypher
   MATCH (s:FacilitySignal {facility_id: $f, status: 'checked'})
   RETURN s.physics_domain, count(s) ORDER BY count(s) DESC
   ```
4. Launch team with spawn prompt (see above)
5. Monitor cost in Langfuse dashboard
6. Review proposals: `MATCH (mp:MappingProposal {facility_id: $f}) RETURN mp.status, count(mp)`
7. Promote validated mappings to IMASMapping nodes

---

## Context Management

### Auto-Compaction

Set to 70% via `.claude/settings.json` (default 95%). Opus has strong online compaction — earlier triggering prevents context overflow during long team sessions.

### Token Efficiency Strategies

| Strategy | Token Impact |
|----------|-------------|
| Skills over CLAUDE.md | 2-4k per skill vs 15k+ for full AGENTS.md |
| Codegen over exploration | 200 tokens for script output vs 10k for inline data |
| Graph as context store | Proposals persisted in Neo4j, not context window |
| Projected Cypher | `RETURN n.id, n.name` not full node objects |

### Agent Memory

Subagents accumulate domain knowledge in `.claude/agent-memory/<name>/MEMORY.md` across sessions. The IMAS expert remembers pattern discoveries from previous facilities.

---

## Dependency Structure

The agent teams infrastructure uses a restructured dependency model:

### Core Dependencies
Required for MCP server, graph queries, search, CLI:
```
fastmcp, pydantic, pint, nest-asyncio, click, python-dotenv, PyYAML,
numpy, cachetools, imas-data-dictionaries, anyio, rapidfuzz, neo4j,
ruamel-yaml, rich, httpx, requests
```

### Dev Dependencies
Full development stack (installed via `uv sync`):
- LLM & Discovery: litellm, smolagents, llama-index-*
- Graph build: linkml, networkx, scikit-learn, hdbscan
- Wiki parsing: beautifulsoup4, Pillow, python-docx, openpyxl
- Auth & remote: keyring, fabric, pykakasi
- Testing: pytest, pytest-asyncio, pytest-cov

### Serve Extra
For proxy and embedding server (`pip install imas-codex[serve]`):
```
fastapi, uvicorn, litellm
```

---

## Future Applications

The graph-as-shared-state pattern supports teams beyond IMAS mapping:

- **Hypothesis testing teams**: Propose physics hypotheses, write analysis code, test against graph data
- **Cross-facility comparison teams**: Compare measurements across facilities, identify discrepancies
- **Data quality audit teams**: Systematically validate graph data against source
- **Documentation mining teams**: Extract signal tables and mapping hints from facility wikis

The key enabler: agents write proposals, evidence, and results to the graph, not to files or context. Any team can pick up from any previous team's work.

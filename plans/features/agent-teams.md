# Agent Teams Infrastructure

> **Goal**: Deploy collaborative Claude Code agent teams for IMAS mapping discovery across JET, TCV, and JT60SA. Build reusable infrastructure for future scientific teams.

## Table of Contents

1. [Dependency Restructuring](#phase-1-dependency-restructuring)
2. [LiteLLM Proxy Server](#phase-2-litellm-proxy-server)
3. [Prompt Caching Verification](#phase-3-prompt-caching-verification)
4. [Claude Code Agent Team Configuration](#phase-4-claude-code-agent-team-configuration)
5. [Graph Schema Extensions](#phase-5-graph-schema-extensions)
6. [IMAS Mapping Team Deployment](#phase-6-imas-mapping-team-deployment)
7. [Budget & Constraints](#phase-7-budget--constraints)

---

## Phase 1: Dependency Restructuring

### Rationale

72 packages in core deps is unsustainable. Many are only needed for graph construction, wiki scraping, or embedding. Non-dev users who just run the MCP server or query the graph shouldn't pull the entire discovery stack.

### New Structure

**Core dependencies** (`[project.dependencies]`) — required for MCP server, graph queries, search, CLI:

```
fastmcp, pydantic, pint, nest-asyncio, click, python-dotenv, PyYAML,
numpy, cachetools, imas-data-dictionaries, anyio, rapidfuzz, neo4j,
ruamel-yaml, rich, httpx, requests
```

**Dev group** (`[dependency-groups] dev`) — full development stack, installed automatically via `uv run`:

```python
[dependency-groups]
dev = [
    # --- Code quality ---
    "ruff>=0",
    "mypy>=1.15.0",
    "pre-commit>=4.2.0",
    "tqdm-stubs>=0.2.1",

    # --- Interactive dev ---
    "ipython>=9.2.0",
    "ipykernel>=6.29.5",

    # --- LLM & Discovery ---
    "litellm>=1.81.0",
    "smolagents>=1.24.0",
    "llama-index-core>=0.14",
    "llama-index-vector-stores-neo4jvector>=0.4",
    "llama-index-llms-openrouter>=0.4",
    "llama-index-readers-file>=0.5.6",

    # --- Graph build & schema ---
    "linkml>=1.9.3",
    "linkml-runtime>=1.9.5",
    "imas-python>=2.0.1",
    "networkx>=3.0,<4.0",
    "scikit-learn>=1.7.2",
    "hdbscan>=0.8.41",

    # --- Wiki & document parsing ---
    "beautifulsoup4>=4.14.3",
    "Pillow>=11.0.0",
    "python-docx>=1.1.2",
    "python-pptx>=1.0.2",
    "openpyxl>=3.1.5",
    "nbformat>=5.10.4",
    "xlrd>=2.0.2",

    # --- Auth & remote ---
    "keyring>=25.7.0",
    "secretstorage>=3.5.0",
    "fabric>=3.0.0",
    "pykakasi>=2.3.0",
    "jellyfish>=1.2.1",
    "huggingface_hub[hf_xet]>=0.20.0",

    # --- Tree-sitter ---
    "tree-sitter>=0.25.2",
    "tree-sitter-languages>=1.10.2",
    "tree-sitter-language-pack>=0.13.0",

    # --- Testing ---
    "pytest>=8.4.2",
    "pytest-asyncio>=0.26.0",
    "pytest-cov>=6.1.1",
    "pytest-timeout>=2.1.0",
    "xlwt>=1.3.0",
]
```

**Extras** (pip-installable via `pip install imas-codex[serve]`):

| Extra | Purpose | Packages |
|-------|---------|----------|
| `test` | CI testing | pytest, pytest-cov, pytest-asyncio, pytest-xdist, pytest-benchmark, pytest-timeout, coverage |
| `serve` | Local server hosting | fastapi, uvicorn, litellm |
| `cpu` | CPU embedding backend | torch, sentence-transformers |
| `gpu` | GPU embedding backend | torch==2.5.1, accelerate, sentence-transformers |

**Remove entirely** (unused — not imported anywhere in codebase):

- `llama-index-llms-openai`
- `llama-index-llms-openai-like`
- `llama-index-tools-mcp`
- `llama-index-readers-web`
- `html2text`
- `openai`
- `llama-index-embeddings-huggingface` (from cpu/gpu extras)

**Promote to core**: `rich` (imported by 20+ files, currently only in dev/build-system).

**Remove from dev** (unused): `black`, `matplotlib`, `seaborn`, `qrcode[pil]`, `pytest-httpserver`, `ipywidgets`, `anyio` (dupe of core).

### Files Changed

- `pyproject.toml` — restructure `[project.dependencies]`, `[dependency-groups]`, `[project.optional-dependencies]`

### Verification

```bash
# Core install works (MCP server, graph queries)
pip install -e .
uv run imas-codex serve imas --transport stdio

# Dev install works (full stack)
uv sync
uv run pytest

# Serve extra works
pip install -e ".[serve]"
uv run imas-codex serve embed start --dry-run
```

---

## Phase 2: LiteLLM Proxy Server

### Why a Proxy

1. **Single API key management** — one `credential_list` entry, referenced by all models
2. **Centralized cost tracking** — all LLM calls (discovery workers, MCP tools, agent teams) flow through one point with Langfuse callbacks
3. **Prompt caching configuration** — `cache_control_injection_points` set per-model in YAML, not scattered across Python code
4. **OpenAI-compatible endpoint** — Claude Code and any external tools can use `http://localhost:4000` as their API base
5. **Budget enforcement** — per-key and per-model spend limits at the proxy level
6. **Facility tagging** — pass `metadata.facility` on every call for per-facility cost breakdown in Langfuse

### LiteLLM Config

Create `imas_codex/config/litellm_config.yaml`:

```yaml
# imas-codex LiteLLM Proxy Configuration
# Start: imas-codex serve llm start
# Or:    litellm --config imas_codex/config/litellm_config.yaml

# --- Shared credentials ---
credential_list:
  - credential_name: openrouter
    credential_values:
      api_key: os.environ/OPENROUTER_API_KEY
    credential_info:
      description: "OpenRouter API key for all model routing"

# --- Model definitions ---
model_list:
  # Opus 4.6 — agent teams, complex reasoning
  - model_name: opus
    litellm_params:
      model: openrouter/anthropic/claude-opus-4-20250514
      litellm_credential_name: openrouter
      cache_control_injection_points:
        - {location: message, role: system}

  # Sonnet 4 — focused tasks, search, data gathering
  - model_name: sonnet
    litellm_params:
      model: openrouter/anthropic/claude-sonnet-4-20250514
      litellm_credential_name: openrouter
      cache_control_injection_points:
        - {location: message, role: system}

  # Haiku 4 — structured output, graph writes, low-reasoning
  - model_name: haiku
    litellm_params:
      model: openrouter/anthropic/claude-haiku-4-20250514
      litellm_credential_name: openrouter

  # Gemini Flash — scoring, batch classification (auto-caches)
  - model_name: scoring
    litellm_params:
      model: openrouter/google/gemini-3-flash-preview
      litellm_credential_name: openrouter

  # Passthrough — any model not explicitly listed
  - model_name: "*"
    litellm_params:
      model: "openrouter/*"
      litellm_credential_name: openrouter

litellm_settings:
  drop_params: true
  num_retries: 3
  request_timeout: 120
  success_callback: ["langfuse"]
  failure_callback: ["langfuse"]

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

**Key design**: `credential_list` defines the OpenRouter key once. All model entries reference it via `litellm_credential_name: openrouter`. No key duplication.

### CLI Integration

Add `serve llm` subcommand group to `imas_codex/cli/serve.py` (follows `serve embed` pattern):

```
imas-codex serve llm
├── start     # litellm --config ... (foreground, port 4000)
├── status    # Health check against proxy
└── service   # systemd install/start/stop/status
```

### Dual Access Pattern

| Consumer | Access Method | Notes |
|----------|---------------|-------|
| Discovery workers | `litellm.completion()` pointed at proxy (`LITELLM_PROXY_URL`) | Existing retry/cost logic in `discovery/base/llm.py` |
| MCP tools / smolagents | `LiteLLMModel(api_base=proxy_url)` | Replace direct OpenRouter with proxy |
| Claude Code agent teams | OpenAI-compatible endpoint | Configure in `.claude/settings.json` |

### Environment Variables

Add to `.env` and `env.example`:

```bash
# LiteLLM Proxy
LITELLM_MASTER_KEY=sk-litellm-imas-codex
LITELLM_PROXY_URL=http://localhost:4000

# Langfuse Observability
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Files Changed

- New: `imas_codex/config/litellm_config.yaml`
- `imas_codex/cli/serve.py` — add `llm` subcommand group
- `imas_codex/discovery/base/llm.py` — add `LITELLM_PROXY_URL` support
- `env.example` — add new env vars

---

## Phase 3: Prompt Caching Verification

### The Problem

Anthropic prompt caching requires explicit `cache_control` breakpoints in message content. When routing through OpenRouter → LiteLLM, we need to confirm this chain preserves the `cache_control` directives and that the response includes `cached_tokens` metrics.

**Additional concern — OpenRouter provider routing**: OpenRouter may route Anthropic requests to different backend providers (GCP Vertex, AWS Bedrock, Anthropic direct). Prompt caching requires hitting the **same provider instance** across requests. OpenRouter does "best-effort" routing to the same provider for cache reuse, but this is not guaranteed.

**Mitigation**: OpenRouter's `provider.order` parameter can pin to a specific provider:

```python
response = litellm.completion(
    model="openrouter/anthropic/claude-opus-4-20250514",
    messages=[...],
    extra_body={"provider": {"order": ["Anthropic"]}},
)
```

This can be set per-model in the LiteLLM config via `extra_body` in `litellm_params`, or tested first without pinning to see if default routing gives acceptable cache hit rates.

### Diagnostic Test

Create `scripts/test_prompt_caching.py`:

```python
"""Test prompt caching through LiteLLM proxy → OpenRouter → Anthropic.

Usage:
    # Via proxy (preferred):
    LITELLM_PROXY_URL=http://localhost:4000 uv run python scripts/test_prompt_caching.py

    # Direct (for comparison):
    uv run python scripts/test_prompt_caching.py --direct
"""
import litellm
import os
import time

LARGE_SYSTEM = "You are an IMAS expert. " * 500  # ~1500 tokens

def test_caching(model: str, via_proxy: bool = True):
    kwargs = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": LARGE_SYSTEM},
                    {
                        "type": "text",
                        "text": "Always respond concisely.",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
            {"role": "user", "content": "What is an IDS?"},
        ],
        "max_tokens": 50,
    }
    if via_proxy:
        kwargs["api_base"] = os.environ.get("LITELLM_PROXY_URL", "http://localhost:4000")
        kwargs["api_key"] = os.environ.get("LITELLM_MASTER_KEY", "sk-litellm-imas-codex")

    # First call — cache write
    r1 = litellm.completion(**kwargs)
    print(f"Call 1: {r1.usage}")
    time.sleep(2)

    # Second call — should be cache hit
    r2 = litellm.completion(**kwargs)
    print(f"Call 2: {r2.usage}")

    cached = getattr(r2.usage.prompt_tokens_details, "cached_tokens", 0)
    print(f"\nCached tokens on call 2: {cached}")
    if cached > 0:
        print("✅ Prompt caching WORKING")
    else:
        print("❌ Prompt caching NOT detected — check Langfuse/OpenRouter activity")
    return cached > 0

if __name__ == "__main__":
    import sys
    direct = "--direct" in sys.argv
    model = "openrouter/anthropic/claude-sonnet-4-20250514" if direct else "sonnet"
    test_caching(model, via_proxy=not direct)
```

### Verification Steps

1. Start proxy: `imas-codex serve llm start`
2. Run test: `uv run python scripts/test_prompt_caching.py`
3. Check Langfuse dashboard for `cached_tokens` in usage response
4. Check OpenRouter activity page for `cache_discount` field
5. If caching fails, try pinning provider: add `extra_body: {"provider": {"order": ["Anthropic"]}}` to litellm config
6. If that fails, verify the response includes `prompt_tokens_details` at all — some OpenRouter providers may not report it

### Langfuse Setup

1. Sign up at https://cloud.langfuse.com (free tier: 50k observations/month)
2. Create project "imas-codex"
3. Copy public/secret keys to `.env`
4. Configure OpenRouter observability at https://openrouter.ai/settings/observability → **Langfuse** (same platform for both sides, single dashboard)

**Why Langfuse**: Open source (MIT), first-class LiteLLM integration (2 env vars + `success_callback`), self-hostable, traces prompt cache hits with token-level detail, supports per-request metadata tagging for facility-level cost breakdown.

---

## Phase 4: Claude Code Agent Team Configuration

### Key Architecture Decision: Dynamic Teams, Not Pre-defined Agents

After full review of the Claude Code agent teams documentation, the correct pattern is:

**Agent teams are launched dynamically** via natural language. You tell Claude to "create a team with N teammates focused on X" and it spawns them. You do NOT pre-define every teammate as a separate `.md` file.

**Subagents** (`.claude/agents/*.md`) are for reusable roles that Claude auto-delegates to during any session. They're not the same as team members.

**The right approach for imas-codex**:

1. Define **2-3 reusable subagents** for common roles (graph-querier, facility-explorer)
2. Launch **agent teams dynamically** with focused spawn prompts that reference project context via skills
3. Use **skills** (`.claude/skills/`) for domain knowledge injection — compact, focused context that doesn't bloat agent windows

### File Layout

```
.claude/
├── settings.json              # Project settings (team mode, env, hooks)
├── agents/
│   ├── graph-querier.md       # Reusable: queries graph, persists results
│   └── facility-explorer.md   # Reusable: SSH, data access, signal enumeration
├── skills/
│   ├── graph-queries.md       # Cypher patterns for common operations
│   ├── schema-summary.md      # Condensed graph schema (~2k tokens)
│   ├── mapping-workflow.md    # How to propose/validate/persist mappings
│   └── facility-access.md    # SSH patterns, data system access per facility
└── hooks/
    └── task-completed.sh      # Quality gate: check evidence before completion
```

### Context Isolation: How to Prevent Agent Teams from Reading Project Files

This is critical. Agent teams by default load `CLAUDE.md` and have access to all tools. To prevent context bloat:

**1. Use a `.claude/CLAUDE.md` for team context** (separate from root `CLAUDE.md`):

According to Claude Code docs, teammates read `CLAUDE.md` from their working directory. If we set their working directory (or use the project-level `.claude/CLAUDE.md`), they get only the team-relevant context. However, Claude Code's CLAUDE.md resolution reads from the project root first.

**2. Restrict tools via subagent definitions**:

The `tools` field in subagent frontmatter is the primary isolation mechanism. If you restrict a subagent to `Bash, Read, Grep` and specific `mcpServers`, it cannot use `Edit` or `Write` on project files. But for agent teams (which are independent sessions, not subagents), isolation works differently.

**3. The practical solution — `.claudeignore` + targeted MCP + spawn prompts**:

Create `.claudeignore` in the project root to exclude files from Claude Code's file access. This is the most reliable isolation mechanism for agent teams since each teammate is a full session.

Agent team members get their context from:
- The **spawn prompt** (provided by the lead when creating the team)
- **CLAUDE.md** (loaded from working directory — we keep this minimal for teams)
- **Skills** (preloaded domain knowledge)
- **MCP servers** (graph queries, IMAS DD access)

For the **development pattern** (you working with Claude Code on the codebase), you continue using `AGENTS.md`, `CLAUDE.md`, and the full project.

For the **agent team pattern** (autonomous mapping teams), the lead session operates in the project but teammates are launched with specific skill injections and MCP-only tool access:

```
Create an agent team for TCV IMAS mapping with 4 teammates:
- IMAS expert (uses imas-ddv4 MCP server for DD lookups)
- Facility analyst (uses codex MCP for graph queries + Bash for SSH)
- Evidence researcher (uses codex MCP to search wiki/code chunks)
- Mapping validator (uses Bash to test data access remotely)

Use Opus for all teammates. Load the mapping-workflow and graph-queries skills.
Each teammate should write code to test their hypotheses — do not read project
source files, work through MCP tools and SSH only.
Require plan approval before any teammate persists mappings to the graph.
```

**The spawn prompt IS the isolation mechanism.** Teams don't need `.claudeignore` — the lead's instructions to teammates define their scope. Teammates inherently don't have incentive to read random project files if their task is "query the graph and SSH to TCV."

### `.claude/settings.json`

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1",
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "70"
  },
  "permissions": {
    "allow": [
      "Bash(ssh *)",
      "Bash(uv run *)",
      "Bash(curl *)",
      "mcp__codex__*",
      "mcp__imas-ddv4__*",
      "Read(.claude/**)"
    ]
  }
}
```

### Subagent Definitions

**`.claude/agents/graph-querier.md`**:

```yaml
---
name: graph-querier
description: Query and persist data to the Neo4j knowledge graph. Use for any operation requiring Cypher queries, semantic search, or node creation.
tools: Bash, Read, Grep
model: opus
permissionMode: acceptEdits
maxTurns: 30
memory: project
mcpServers:
  - codex
  - imas-ddv4
skills:
  - graph-queries
  - schema-summary
---

You are a knowledge graph specialist for the imas-codex fusion data graph.

Your capabilities:
- Execute Cypher queries via the codex MCP server
- Perform semantic search across vector indexes
- Create and update graph nodes (MappingProposal, IMASMapping, MappingEvidence)
- Look up IMAS Data Dictionary paths via the imas-ddv4 MCP server

Always project specific properties in Cypher (RETURN n.id, n.name), never full nodes.
Use UNWIND for batch operations.
Track your findings in your agent memory.
```

**`.claude/agents/facility-explorer.md`**:

```yaml
---
name: facility-explorer
description: Explore remote fusion facilities via SSH. Use for signal enumeration, data access testing, and infrastructure discovery.
tools: Bash, Read, Grep
model: opus
permissionMode: acceptEdits
maxTurns: 30
memory: project
mcpServers:
  - codex
skills:
  - facility-access
---

You are a facility data expert for fusion research facilities.

Your capabilities:
- SSH to remote facilities (tcv, jet, jt60sa, iter)
- Enumerate MDSplus trees and signals
- Test data access patterns (MDSplus, TDI, PPF)
- Check signal availability at specific shots

Write Python scripts to test data access hypotheses.
Never read project source code — work through SSH and MCP tools only.
Track discoveries in your agent memory.
```

### Prompt Management — Langfuse vs In-Project

**Recommendation: Keep prompts in-project, use Langfuse for tracking**.

Langfuse Prompt Management is designed for versioned prompt A/B testing in production systems. For imas-codex, the `.claude/skills/` and subagent `.md` files serve as the prompt source of truth — they're version-controlled in git, reviewable in PRs, and loaded directly by Claude Code.

**What Langfuse provides**: trace which prompt version was active when a generation occurred, compare cost/quality across prompt versions. This is valuable for the discovery pipeline's LLM calls (scoring, enrichment) but not for Claude Code agent system prompts — those are managed by Claude Code itself.

**Pattern**: Prompts in git (`.claude/skills/`, `imas_codex/agentic/prompts/`). Langfuse for observability of LLM call quality and cost. If we later want to A/B test scoring prompts, we can version them in Langfuse at that point.

### Context Management & Token Efficiency

**Auto-compaction**: Set `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=70` to trigger compaction at 70% context capacity (default is 95%). Opus 4.6 has strong online compaction — this is where it excels.

**Codegen approach**: Agents write Python scripts for testing, not interactive exploration. This keeps tool call output manageable:

```
Bad: agent reads 50 graph nodes inline → 10k tokens of context
Good: agent writes script.py, runs it, captures 5-line summary → 200 tokens
```

**Skills over CLAUDE.md**: Skills are injected at subagent creation. They're compact (2-4k tokens each) vs AGENTS.md (15k+ tokens). Agent team members get only their relevant skills.

**Graph as context store**: Instead of holding mapping state in context, agents write proposals to the graph and query it. The graph is the shared memory, not the context window.

---

## Phase 5: Graph Schema Extensions

### Design Principle: Proposal → Consensus → Mapping

Every team-generated mapping follows a lifecycle with full audit trail. Rejected mappings are persisted with rejection reasons — they prevent future agents from re-proposing the same bad mapping.

The `IMASMapping` class already exists in the schema with full pump context (driver, units, scale, transformation). We extend it with a status lifecycle and add supporting types for the proposal/evidence workflow.

### New Schema Types

Add to `imas_codex/schemas/facility.yaml`:

```yaml
# =============================================================================
# Agent Team Schema Extensions
# =============================================================================

  MappingStatus:
    description: Status lifecycle for IMAS mappings
    permissible_values:
      proposed:
        description: Initial proposal from an agent, awaiting team review
      endorsed:
        description: Supported by multiple evidence sources and team consensus
      contested:
        description: Conflicting evidence, requires further investigation
      validated:
        description: Tested against real data with passing result
      rejected:
        description: Rejected with documented reason (persisted for future reference)
      published:
        description: Final, released mapping ready for Ambix recipe generation

  MappingEvidence:
    description: >-
      A piece of evidence supporting or contradicting a mapping proposal.
      Evidence is never deleted — even for rejected mappings, to prevent
      re-investigation of known-bad mappings.
    class_uri: facility:MappingEvidence
    attributes:
      id:
        identifier: true
        required: true
      evidence_type:
        description: Category of evidence
        range: EvidenceType
        required: true
      source_url:
        description: URL or path where evidence was found
      source_chunk_id:
        description: Link to WikiChunk or CodeChunk if from ingested content
      excerpt:
        description: Relevant text excerpt (max 500 chars)
      supports:
        description: Whether this evidence supports (true) or contradicts (false) the mapping
        range: boolean
        required: true
      weight:
        description: Relative importance of this evidence (0.0-1.0)
        range: float
      contributed_by:
        description: Agent name or session that contributed this evidence
      facility_id:
        description: Parent facility
        required: true
        range: Facility
        annotations:
          relationship_type: AT_FACILITY

  EvidenceType:
    description: Categories of mapping evidence
    permissible_values:
      wiki_reference:
        description: Documentation from facility wiki
      code_analysis:
        description: Evidence from source code analysis
      data_test:
        description: Result of testing data access
      unit_match:
        description: Unit compatibility analysis
      dimension_match:
        description: Dimensionality/shape analysis
      semantic_similarity:
        description: Vector similarity between signal and IDS path
      sign_convention:
        description: COCOS or sign convention analysis
      expert_assertion:
        description: Domain expert assertion with reasoning

  AgentSession:
    description: >-
      Tracks a team run for cost accounting and progress monitoring.
      Links to all proposals and evidence generated during the session.
    class_uri: facility:AgentSession
    attributes:
      id:
        identifier: true
        required: true
      team_name:
        description: Name of the agent team
      facility_id:
        description: Target facility for this session
        required: true
        range: Facility
        annotations:
          relationship_type: AT_FACILITY
      started_at:
        range: datetime
      completed_at:
        range: datetime
      status:
        range: AgentSessionStatus
      budget_limit_usd:
        description: Maximum spend allowed for this session
        range: float
      budget_used_usd:
        description: Actual spend recorded (updated from Langfuse)
        range: float
      proposals_created:
        description: Count of MappingProposals created
        range: integer
      proposals_validated:
        description: Count of proposals that reached validated status
        range: integer

  AgentSessionStatus:
    permissible_values:
      running:
        description: Team is actively working
      completed:
        description: Team finished within budget
      budget_exceeded:
        description: Budget limit reached, team stopped
      failed:
        description: Team encountered unrecoverable error
```

### Extend Existing `IMASMapping`

Add `status` field and `evidence` relationship:

```yaml
  IMASMapping:
    # ... existing fields ...
    attributes:
      # Add:
      status:
        description: Lifecycle status of this mapping
        range: MappingStatus
      rejection_reason:
        description: Why this mapping was rejected (persisted to prevent re-proposal)
      session_id:
        description: AgentSession that created this mapping
        range: AgentSession
        annotations:
          relationship_type: PROPOSED_IN
      evidence:
        description: Evidence nodes supporting/contradicting this mapping
        multivalued: true
        range: MappingEvidence
        annotations:
          relationship_type: HAS_EVIDENCE
```

### Cost Tracking: Graph vs Observability

**Do we need cost in the graph?** Partially. Langfuse gives us per-request cost breakdowns for free. But the graph needs a lightweight `AgentSession` node for:

1. **Correlating proposals to cost** — "this session spent $45 and produced 80 validated mappings"
2. **Budget enforcement** — workers check `budget_used_usd < budget_limit_usd` before claiming new work
3. **Per-facility accounting** — `AgentSession.facility_id` lets us query "total spend on TCV mappings"

**Implementation**: `AgentSession.budget_used_usd` is periodically synced from Langfuse by querying traces tagged with the session ID. This is a lightweight cron update, not real-time — Langfuse is the source of truth for cost.

**Distinguishing cost per facility**: Every LLM call through the proxy includes `metadata: {facility: "tcv", session_id: "..."}` in the request. Langfuse traces this and we can filter/aggregate by facility in the dashboard. The proxy config does NOT need per-facility API keys.

### Files Changed

- `imas_codex/schemas/facility.yaml` — add `MappingStatus`, `MappingEvidence`, `EvidenceType`, `AgentSession`, `AgentSessionStatus`; extend `IMASMapping`
- `imas_codex/schemas/common.yaml` — potentially move enums if shared
- Run: `uv run build-models --force` after changes

---

## Phase 6: IMAS Mapping Team Deployment

### Team Composition

Agent teams are launched dynamically. The lead creates teammates based on the task. For IMAS mapping, the recommended team prompt:

```
Create an agent team for [FACILITY] IMAS mapping discovery.

Teammates:
1. IMAS Expert — knows the Data Dictionary structure. Uses imas-ddv4 MCP server
   to look up IDS paths, types, units, and coordinates. Proposes target paths
   for facility signals.
2. Facility Analyst — knows [FACILITY] data systems. SSHs to the facility to
   enumerate signals, test data access, check units and shapes.
3. Evidence Researcher — searches wiki and code chunks in the knowledge graph
   for mapping documentation, tables, and existing code that reads/writes IMAS.
4. Validation Tester — writes Python scripts to test proposed mappings against
   real shot data. Checks unit conversions, sign conventions, and array shapes.

Use Opus 4.6 for all teammates.
Load the mapping-workflow and graph-queries skills.
Start with [PHYSICS_DOMAIN] signals (equilibrium, magnetics, etc.).
Each teammate should write code to solve problems — query via MCP and SSH only.
Require plan approval before persisting validated mappings.
Wait for all teammates to complete before synthesizing results.
```

### Workflow

```
1. Lead creates task list:
   - Task 1: Enumerate checked FacilitySignals for [domain]
   - Task 2: Find candidate IMASPath matches via semantic search
   - Task 3: Search wiki/code for mapping documentation
   - Task 4: Cross-reference and propose mappings with evidence
   - Task 5: Validate top candidates against shot data
   - Task 6: Persist validated mappings to graph

2. Facility Analyst claims Task 1:
   - Queries graph: MATCH (s:FacilitySignal {facility_id: $f, status: "checked"})
   - SSHs to facility to verify signal accessibility
   - Reports back: "Found 45 equilibrium signals on TCV"

3. IMAS Expert claims Task 2:
   - semantic_search("plasma current", "imas_path_embedding", 10)
   - Identifies candidate paths: equilibrium.time_slice[:].global_quantities.ip
   - Creates MappingProposal nodes

4. Evidence Researcher claims Task 3:
   - semantic_search("equilibrium mapping", "wiki_chunk_embedding", 10)
   - semantic_search("liuqe IMAS", "code_chunk_embedding", 10)
   - Creates MappingEvidence nodes linked to proposals

5. All teammates communicate findings → Lead synthesizes → contested items highlighted

6. Validation Tester claims Task 5:
   - Writes Python: SSH to TCV, read MDSplus signal, check units/shape
   - Updates proposal status: proposed → endorsed or contested

7. Lead resolves contested items, promotes endorsed → validated

8. Graph Writer persists: validated → IMASMapping nodes with evidence chain
```

### Phased Facility Rollout

| Phase | Facility | Signal Count | Expected Cost | Duration |
|-------|----------|-------------|---------------|----------|
| 1 | TCV | ~300 | $100-150 | 1-2 days |
| 2 | JET | ~500 | $150-200 | 2-3 days |
| 3 | JT60SA | ~200 | $80-120 | 1-2 days |

Start with TCV — smallest signal set, best SSH access, most wiki documentation. Validate approach, refine prompts, then scale.

---

## Phase 7: Budget & Constraints

### $1000 Budget Allocation

| Item | Budget | Notes |
|------|--------|-------|
| TCV mapping team | $150 | First run, includes iteration |
| JET mapping team | $200 | Larger signal set |
| JT60SA mapping team | $120 | Benefit from refined approach |
| Development & debugging | $200 | Infrastructure setup, prompt refinement |
| Prompt caching overhead | $80 | Cache write costs (1.25x for first calls) |
| Buffer | $250 | Unexpected costs, retries, edge cases |
| **Total** | **$1,000** | |

### Is $1000 Practical?

**Yes, with prompt caching.** Key assumptions:

- Opus 4.6 input: $15/M tokens, cache read: $1.50/M (90% savings)
- System prompts + skill content: ~4k tokens per agent, cached after first call
- Per-signal mapping: ~5 agent interactions, ~2k new tokens each
- With 80% cache hit rate: effective input cost drops to ~$3/M tokens
- 1000 signals × 5 interactions × 2k tokens = 10M tokens ≈ $30 input + $75 output (1M output tokens)
- Per-facility: ~$35-70 in LLM cost (well within allocation)

Main cost driver is Opus output tokens ($75/M). If cost pressure builds, shift Facility Analyst and Evidence Researcher to Sonnet 4 ($15/M output) — only IMAS Expert and Validation Tester need Opus-grade reasoning.

### Enforcement Mechanisms

| Layer | Mechanism | Granularity |
|-------|-----------|-------------|
| **OpenRouter** | Per-key spend limits | Total budget |
| **LiteLLM proxy** | `max_budget` per virtual key | Per-session |
| **Graph** | `AgentSession.budget_limit_usd` | Per-team-run |
| **Claude Code** | `--max-budget-usd` (print mode) | Per-session |
| **Langfuse** | Real-time cost dashboard + alerts | Per-request |
| **Hooks** | `TaskCompleted` hook checks cumulative spend | Per-task |

### Long-Running Team Management

1. **Graph as state machine**: All mapping state in Neo4j. Teams can be stopped mid-run and new teams resume from unclaimed signals. Claimed signals auto-release after 5-minute timeout.

2. **Agent memory** (`.claude/agent-memory/<name>/MEMORY.md`): Subagents accumulate domain knowledge across sessions. The IMAS expert remembers pattern discoveries from previous facilities.

3. **Phased execution**: Don't run all 3 facilities simultaneously. TCV → validate → JET → JT60SA. Each phase is a separate team session with its own `AgentSession` node.

4. **Time constraints**: Set `maxTurns: 50` on subagents. Use OpenRouter per-key rate limits (RPM/TPM) to prevent runaway. `TaskCompleted` hook can check elapsed time.

5. **Resumability**: If a team is interrupted, the lead can query the graph for in-progress proposals and spawn a new team to continue:

```
There are 23 MappingProposals for TCV with status "proposed" that need validation.
Create a team to validate these proposals and either endorse or reject them.
```

---

## Implementation Order

```
Phase 1: pyproject.toml dependency restructuring
  └─ Verify: uv sync, uv run pytest, pip install -e .

Phase 2: LiteLLM proxy server
  ├─ litellm_config.yaml
  ├─ CLI serve llm subcommand
  └─ Langfuse integration (env vars + success_callback)

Phase 3: Prompt caching verification
  ├─ test_prompt_caching.py script
  ├─ Run via proxy, check Langfuse
  └─ Provider pinning if needed

Phase 4: .claude/ agent team configuration
  ├─ .claude/settings.json
  ├─ .claude/agents/ (2 subagents)
  ├─ .claude/skills/ (4 skills)
  └─ .claude/hooks/task-completed.sh

Phase 5: Graph schema extensions
  ├─ MappingStatus, MappingEvidence, AgentSession
  ├─ Extend IMASMapping
  └─ uv run build-models --force

Phase 6: TCV mapping team (first deployment)
  ├─ Start with equilibrium physics domain
  ├─ Validate approach, refine spawn prompts
  └─ Iterate on evidence thresholds

Phase 7: JET, JT60SA mapping teams
  └─ Scale validated approach
```

---

## Future: Scientific Agent Teams

This infrastructure is designed to be general. The mapping team is the first application, but the same patterns support:

- **Hypothesis testing teams**: propose physics hypotheses, write analysis code, test against graph data
- **Cross-facility comparison teams**: compare measurements across facilities, identify discrepancies
- **Data quality audit teams**: systematically validate graph data against source

The key enabler is the **graph as shared state machine** — agents write proposals, evidence, and results to the graph, not to files or context. Any team can pick up from any previous team's work.

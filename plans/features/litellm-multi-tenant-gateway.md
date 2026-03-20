# LiteLLM Multi-Tenant Gateway

Transform the single-key LiteLLM proxy into a multi-tenant LLM gateway with per-client
API key isolation, Claude Code integration, spend tracking, security hardening, and a
test deployment of a local open-weight model on the Titan cluster.

---

## Current State

The LiteLLM proxy runs on the ITER login node as a systemd user service (port 18400),
routing all LLM calls through a single OpenRouter API key. All consumers — imas-codex
discovery workers, MCP server agents, and any future clients — share one master key for
authentication and one OpenRouter key for billing.

**Key files:**
- `imas_codex/config/litellm_config.yaml` — proxy config
- `imas_codex/config/prompt_caching.yaml` — cache control patterns
- `imas_codex/cli/llm_cli.py` — CLI management commands
- `imas_codex/discovery/base/llm.py` — call routing (`_build_kwargs`, `ensure_openrouter_prefix`)
- `imas_codex/settings.py` — port/URL resolution (`get_llm_proxy_url`, `get_llm_location`)

**Limitations:**
- No database → no virtual keys, teams, or per-client spend tracking
- Single OpenRouter key → all usage aggregates to one bill
- No per-client rate/budget limits → runaway client can exhaust budget
- `0.0.0.0` binding with no TLS → accessible to all facility network users
- Master key shared with workers → no admin/user key separation
- No Claude Code or external client configuration

---

## Target Architecture

```
┌─ Login Node ─────────────────────────────────────────────────────────────┐
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  LiteLLM Proxy (:18400) + SQLite DB                                │ │
│  │                                                                     │ │
│  │  Virtual Keys ────────────────────────── Credentials                │ │
│  │  ├─ vk-codex-xxx   → team:codex       → openrouter-codex key       │ │
│  │  ├─ vk-claude-xxx  → team:claude-code → openrouter-claude-code key │ │
│  │  └─ vk-copilot-xxx → team:copilot-cli → openrouter-copilot-cli key │ │
│  │                                                                     │ │
│  │  Model Router ──────────────────────────────────────────            │ │
│  │  ├─ claude/*  ───────→ OpenRouter (Anthropic)                       │ │
│  │  ├─ gemini/*  ───────→ OpenRouter (Google)                          │ │
│  │  ├─ gpt/*     ───────→ OpenRouter (OpenAI)                          │ │
│  │  └─ local/*   ───────→ Ollama/llama.cpp on Titan (future)           │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│         ▲              ▲              ▲              ▲                    │
│         │              │              │              │                    │
│   imas-codex     Claude Code    Copilot CLI    Other clients             │
│   workers        (WSL/ITER)     (WSL/ITER)                               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Credential Routing Design

### How Virtual Key → API Key Routing Works

The LiteLLM proxy identifies callers by their **virtual key** (passed as
the `Authorization: Bearer <key>` header). Each virtual key belongs to a
**team**, and each team has **router settings** that include a
`model_group_alias` mapping. This chain provides transparent credential
routing:

```
1. Client sends request:
     POST /v1/chat/completions
     Authorization: Bearer sk-claude-xxxxxxxx   ← virtual key
     {"model": "opus", ...}

2. LiteLLM looks up virtual key → finds team: claude-code

3. Team claude-code has router_settings.model_group_alias:
     {"opus": "opus-claude-code", "sonnet": "sonnet-claude-code", ...}

4. LiteLLM resolves "opus" → "opus-claude-code" (internal model name)

5. Model entry "opus-claude-code" has:
     litellm_credential_name: openrouter-claude-code
     model: openrouter/anthropic/claude-opus-4.6

6. Credential "openrouter-claude-code" resolves to:
     api_key: $OPENROUTER_API_KEY_CLAUDE_CODE

7. LiteLLM calls OpenRouter with the claude-code API key
   → spend tracked to claude-code team + separate OpenRouter bill
```

### Practical Example

Two users both call `model: "opus"` but get billed to different accounts:

```bash
# ── Simon using Claude Code (team: claude-code) ─────────
curl http://localhost:18400/v1/chat/completions \
  -H "Authorization: Bearer sk-claude-xxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{"model": "opus", "messages": [{"role": "user", "content": "hello"}]}'
# → virtual key sk-claude-xxxxxxxx belongs to team claude-code
# → opus resolves to opus-claude-code → openrouter-claude-code credential
# → billed to OPENROUTER_API_KEY_CLAUDE_CODE on OpenRouter

# ── imas-codex discovery worker (team: codex) ───────────
curl http://localhost:18400/v1/chat/completions \
  -H "Authorization: Bearer sk-codex-yyyyyyyy" \
  -H "Content-Type: application/json" \
  -d '{"model": "opus", "messages": [{"role": "user", "content": "classify this signal"}]}'
# → virtual key sk-codex-yyyyyyyy belongs to team codex
# → opus resolves to opus-codex → openrouter-codex credential
# → billed to OPENROUTER_API_KEY_CODEX on OpenRouter
```

Both calls hit the same underlying model (`anthropic/claude-opus-4.6`)
but are billed to separate OpenRouter API keys.

### Why This Approach

LiteLLM's `litellm_credential_name` is a 1:1 binding — each model entry
gets exactly one credential. You cannot attach multiple credentials to a
single model entry. Three approaches were evaluated:

| Approach | Mechanism | Clients call | Verdict |
|---|---|---|---|
| **A. Team `model_group_alias`** | Duplicate models as `opus-codex` / `opus-claude-code`; teams remap via router settings | `model: opus` | **Selected** — transparent, scalable |
| B. Access Groups | Two `opus` entries in different access groups | `model: opus` | Rejected — LiteLLM may not filter deployments by access group |
| C. Explicit model names | Clients call `opus-codex` or `opus-claude-code` directly | `model: opus-codex` | Rejected — bad UX, leaks internals |

---

## Config Design

### Model Registry

The model registry defines each model alias once. The config uses YAML
anchors so that the OpenRouter model ID is written exactly once, then
referenced in each team's model entries.

**OpenRouter model IDs — note: dots in version numbers:**

| Alias | OpenRouter model ID | Notes |
|-------|---------------------|-------|
| `opus` | `anthropic/claude-opus-4.6` | Strongest coding/agentic |
| `sonnet` | `anthropic/claude-sonnet-4.6` | Best price/performance |
| `haiku` | `anthropic/claude-haiku-4.5-20251001` | Fast/cheap |
| `gemini-flash` | `google/gemini-3-flash-preview` | Fast structured output |
| `gemini-pro` | `google/gemini-3-pro-preview` | Complex structured output |
| `gpt` | `openai/gpt-5.4` | Alternative agentic |
| `gpt-mini` | `openai/gpt-5.4-mini` | Fast/cheap alternative |

**Important:** OpenRouter uses dots in version numbers
(`anthropic/claude-opus-4.6`), while Anthropic's native API uses hyphens
(`claude-opus-4-6`). Always use the OpenRouter format in `litellm_config.yaml`.
The `pyproject.toml` also stores the dotted form; `ensure_openrouter_prefix()`
adds the `openrouter/` prefix at call time.

### `litellm_config.yaml` (target)

```yaml
# imas-codex LiteLLM Proxy — Multi-Tenant Configuration
#
# Start: imas-codex llm start
#
# CREDENTIAL ROUTING:
# Each model is defined once per team/credential. Model names follow
# the pattern: {alias}-{team}  (e.g., opus-codex, opus-claude-code).
# Teams use model_group_alias in router settings to map clean client
# names (opus, sonnet) to their team-specific model entries.
#
# YAML ANCHORS:
# OpenRouter model IDs are defined once as anchors and reused across
# all team entries. To add a new model, define the anchor and add
# one entry per team.

# ── OpenRouter Model ID Anchors ──────────────────────────
# Define once, reference everywhere. When a model is updated,
# change it in one place.
x-models:
  opus:         &opus         openrouter/anthropic/claude-opus-4.6
  sonnet:       &sonnet       openrouter/anthropic/claude-sonnet-4.6
  haiku:        &haiku        openrouter/anthropic/claude-haiku-4.5-20251001
  gemini-flash: &gemini-flash openrouter/google/gemini-3-flash-preview
  gemini-pro:   &gemini-pro   openrouter/google/gemini-3-pro-preview
  gpt:          &gpt          openrouter/openai/gpt-5.4
  gpt-mini:     &gpt-mini     openrouter/openai/gpt-5.4-mini

# ── Credentials ──────────────────────────────────────────
# Each credential maps to a separate OpenRouter API key for
# independent billing. Create keys at https://openrouter.ai/settings/keys
#
# The credential name matches the team name for clarity.
credential_list:
  - credential_name: openrouter-codex
    credential_values:
      api_key: os.environ/OPENROUTER_API_KEY_CODEX
    credential_info:
      description: "imas-codex internal use (discovery, agents, MCP)"

  - credential_name: openrouter-claude-code
    credential_values:
      api_key: os.environ/OPENROUTER_API_KEY_CLAUDE_CODE
    credential_info:
      description: "Claude Code users"

  - credential_name: openrouter-copilot-cli
    credential_values:
      api_key: os.environ/OPENROUTER_API_KEY_COPILOT_CLI
    credential_info:
      description: "Copilot CLI users"

# ── Model Definitions ───────────────────────────────────
# Pattern: {alias}-{team} → same OpenRouter model ID, different credential.
# Clients never see these names — teams remap via model_group_alias.

model_list:
  # ── Team: codex (internal workers) ─────────────────────
  - model_name: opus-codex
    litellm_params:
      model: *opus
      litellm_credential_name: openrouter-codex

  - model_name: sonnet-codex
    litellm_params:
      model: *sonnet
      litellm_credential_name: openrouter-codex

  - model_name: haiku-codex
    litellm_params:
      model: *haiku
      litellm_credential_name: openrouter-codex

  - model_name: gemini-flash-codex
    litellm_params:
      model: *gemini-flash
      litellm_credential_name: openrouter-codex

  - model_name: gemini-pro-codex
    litellm_params:
      model: *gemini-pro
      litellm_credential_name: openrouter-codex

  - model_name: gpt-codex
    litellm_params:
      model: *gpt
      litellm_credential_name: openrouter-codex

  - model_name: gpt-mini-codex
    litellm_params:
      model: *gpt-mini
      litellm_credential_name: openrouter-codex

  # Codex-only aliases (internal use)
  - model_name: reasoning-codex
    litellm_params:
      model: *sonnet
      litellm_credential_name: openrouter-codex

  - model_name: scoring-codex
    litellm_params:
      model: *gemini-flash
      litellm_credential_name: openrouter-codex

  # ── Team: claude-code ──────────────────────────────────
  - model_name: opus-claude-code
    litellm_params:
      model: *opus
      litellm_credential_name: openrouter-claude-code

  - model_name: sonnet-claude-code
    litellm_params:
      model: *sonnet
      litellm_credential_name: openrouter-claude-code

  - model_name: haiku-claude-code
    litellm_params:
      model: *haiku
      litellm_credential_name: openrouter-claude-code

  - model_name: gemini-flash-claude-code
    litellm_params:
      model: *gemini-flash
      litellm_credential_name: openrouter-claude-code

  - model_name: gemini-pro-claude-code
    litellm_params:
      model: *gemini-pro
      litellm_credential_name: openrouter-claude-code

  - model_name: gpt-claude-code
    litellm_params:
      model: *gpt
      litellm_credential_name: openrouter-claude-code

  - model_name: gpt-mini-claude-code
    litellm_params:
      model: *gpt-mini
      litellm_credential_name: openrouter-claude-code

  # ── Team: copilot-cli ─────────────────────────────────
  - model_name: opus-copilot-cli
    litellm_params:
      model: *opus
      litellm_credential_name: openrouter-copilot-cli

  - model_name: sonnet-copilot-cli
    litellm_params:
      model: *sonnet
      litellm_credential_name: openrouter-copilot-cli

  - model_name: haiku-copilot-cli
    litellm_params:
      model: *haiku
      litellm_credential_name: openrouter-copilot-cli

  - model_name: gemini-flash-copilot-cli
    litellm_params:
      model: *gemini-flash
      litellm_credential_name: openrouter-copilot-cli

  - model_name: gemini-pro-copilot-cli
    litellm_params:
      model: *gemini-pro
      litellm_credential_name: openrouter-copilot-cli

  - model_name: gpt-copilot-cli
    litellm_params:
      model: *gpt
      litellm_credential_name: openrouter-copilot-cli

  - model_name: gpt-mini-copilot-cli
    litellm_params:
      model: *gpt-mini
      litellm_credential_name: openrouter-copilot-cli

  # ── Local models (Phase 6 — Titan cluster) ────────────
  # Local models don't need per-team duplication — no billing.
  # All teams can access local models directly.
  # Uncomment when Ollama is deployed on Titan:
  # - model_name: qwen3-14b
  #   litellm_params:
  #     model: ollama/qwen3:14b-q4_K_M
  #     api_base: http://98dci4-gpu-0002:11434

litellm_settings:
  drop_params: true
  num_retries: 3
  request_timeout: 120

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/LITELLM_DATABASE_URL
```

### Model Group Alias Mappings

Each team's `model_group_alias` maps clean client-facing names to
team-specific internal names. The suffix always matches the team name:

| Client calls | Team codex | Team claude-code | Team copilot-cli |
|---|---|---|---|
| `opus` | `opus-codex` | `opus-claude-code` | `opus-copilot-cli` |
| `sonnet` | `sonnet-codex` | `sonnet-claude-code` | `sonnet-copilot-cli` |
| `haiku` | `haiku-codex` | `haiku-claude-code` | `haiku-copilot-cli` |
| `gemini-flash` | `gemini-flash-codex` | `gemini-flash-claude-code` | `gemini-flash-copilot-cli` |
| `gemini-pro` | `gemini-pro-codex` | `gemini-pro-claude-code` | `gemini-pro-copilot-cli` |
| `gpt` | `gpt-codex` | `gpt-claude-code` | `gpt-copilot-cli` |
| `gpt-mini` | `gpt-mini-codex` | `gpt-mini-claude-code` | `gpt-mini-copilot-cli` |
| `reasoning` | `reasoning-codex` | *(not available)* | *(not available)* |
| `scoring` | `scoring-codex` | *(not available)* | *(not available)* |

### Example Teams & Keys

```bash
# ── Create teams with model_group_alias ──────────────────
# The model_group_alias in router_settings maps clean model names
# to team-specific internal names: {alias}-{team}.

# Codex team (internal — uses openrouter-codex credential)
curl -X POST http://localhost:18400/team/new \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "team_alias": "codex",
    "max_budget": 500,
    "budget_duration": "30d",
    "models": ["opus-codex", "sonnet-codex", "haiku-codex",
               "gemini-flash-codex", "gemini-pro-codex",
               "gpt-codex", "gpt-mini-codex",
               "reasoning-codex", "scoring-codex"],
    "router_settings": {
      "model_group_alias": {
        "opus": "opus-codex",
        "sonnet": "sonnet-codex",
        "haiku": "haiku-codex",
        "gemini-flash": "gemini-flash-codex",
        "gemini-pro": "gemini-pro-codex",
        "gpt": "gpt-codex",
        "gpt-mini": "gpt-mini-codex",
        "reasoning": "reasoning-codex",
        "scoring": "scoring-codex"
      }
    }
  }'

# Claude Code team (uses openrouter-claude-code credential)
curl -X POST http://localhost:18400/team/new \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "team_alias": "claude-code",
    "max_budget": 200,
    "budget_duration": "30d",
    "models": ["opus-claude-code", "sonnet-claude-code", "haiku-claude-code",
               "gemini-flash-claude-code", "gemini-pro-claude-code",
               "gpt-claude-code", "gpt-mini-claude-code"],
    "router_settings": {
      "model_group_alias": {
        "opus": "opus-claude-code",
        "sonnet": "sonnet-claude-code",
        "haiku": "haiku-claude-code",
        "gemini-flash": "gemini-flash-claude-code",
        "gemini-pro": "gemini-pro-claude-code",
        "gpt": "gpt-claude-code",
        "gpt-mini": "gpt-mini-claude-code"
      }
    }
  }'

# Copilot CLI team (uses openrouter-copilot-cli credential)
curl -X POST http://localhost:18400/team/new \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "team_alias": "copilot-cli",
    "max_budget": 100,
    "budget_duration": "30d",
    "models": ["opus-copilot-cli", "sonnet-copilot-cli", "haiku-copilot-cli",
               "gemini-flash-copilot-cli", "gemini-pro-copilot-cli",
               "gpt-copilot-cli", "gpt-mini-copilot-cli"],
    "router_settings": {
      "model_group_alias": {
        "opus": "opus-copilot-cli",
        "sonnet": "sonnet-copilot-cli",
        "haiku": "haiku-copilot-cli",
        "gemini-flash": "gemini-flash-copilot-cli",
        "gemini-pro": "gemini-pro-copilot-cli",
        "gpt": "gpt-copilot-cli",
        "gpt-mini": "gpt-mini-copilot-cli"
      }
    }
  }'

# ── Generate virtual keys ───────────────────────────────
# Codex workers key
curl -X POST http://localhost:18400/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"key_alias": "codex-workers", "team_id": "<codex-team-id>"}'
# → {"key": "sk-codex-xxxxxxxx"}
# → model="opus" → opus-codex → openrouter-codex → $OPENROUTER_API_KEY_CODEX

# Claude Code key
curl -X POST http://localhost:18400/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"key_alias": "claude-code-simon", "team_id": "<claude-code-team-id>",
       "max_budget": 200, "budget_duration": "30d"}'
# → {"key": "sk-claude-xxxxxxxx"}
# → model="opus" → opus-claude-code → openrouter-claude-code → $OPENROUTER_API_KEY_CLAUDE_CODE

# Copilot CLI key
curl -X POST http://localhost:18400/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"key_alias": "copilot-cli-simon", "team_id": "<copilot-cli-team-id>",
       "max_budget": 100, "budget_duration": "30d"}'
# → {"key": "sk-copilot-xxxxxxxx"}
# → model="opus" → opus-copilot-cli → openrouter-copilot-cli → $OPENROUTER_API_KEY_COPILOT_CLI
```

### Adding a New Team

To add a team (e.g., `collaborator`) with its own OpenRouter billing:

1. Create an OpenRouter API key at openrouter.ai/settings/keys
2. Add `OPENROUTER_API_KEY_COLLABORATOR` to `.env`
3. Add a credential entry in `litellm_config.yaml`:
   ```yaml
   - credential_name: openrouter-collaborator
     credential_values:
       api_key: os.environ/OPENROUTER_API_KEY_COLLABORATOR
   ```
4. Add model entries using anchors (one per alias):
   ```yaml
   - model_name: opus-collaborator
     litellm_params:
       model: *opus
       litellm_credential_name: openrouter-collaborator
   # ... repeat for sonnet, haiku, etc.
   ```
5. Create team via `/team/new` with `model_group_alias` mapping
6. Generate virtual key via `/key/generate`

No changes needed for existing teams or clients.

---

## Implementation Phases

### Phase Overview & Parallelism

```
                    ┌──────────────────────────────────┐
                    │  Phase 1: Database & Virtual Keys │
                    │  (foundation — must be first)     │
                    └──────────┬───────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Phase 2:        │ │  Phase 3:        │ │  Phase 4:        │
│  Multi-Credential│ │  Claude Code     │ │  Security        │
│  Routing         │ │  Setup           │ │  Hardening       │
│                  │ │                  │ │                  │
│  (needs teams)   │ │  (needs virtual  │ │  (needs DB)      │
│                  │ │   key from Ph 1) │ │                  │
└──────────────────┘ └──────────────────┘ └──────────────────┘

┌──────────────────────────────────────────────────────────┐
│  These phases are INDEPENDENT — can run at any time,     │
│  in parallel with each other and with Phases 1-4:        │
│                                                          │
│  Phase 5: Model Config Update                            │
│  Phase 6: Local Model Deployment (Titan)                 │
│  Phase 7: Documentation                                  │
└──────────────────────────────────────────────────────────┘

Phase 8: Operational Tooling (stretch — requires Phases 1-4)
```

**Parallel work opportunities:**

| Work stream | Can start | Prerequisites |
|---|---|---|
| Phase 1 (DB + Keys) | Immediately | None |
| Phase 5 (Model Config) | Immediately | None |
| Phase 6 (Titan local model) | Immediately | Titan access |
| Phase 7 (Documentation) | Immediately | None |
| Phase 2 (Multi-Credential) | After Phase 1 | Phase 1 complete |
| Phase 3 (Claude Code) | After Phase 1 | Phase 1 virtual key |
| Phase 4 (Security) | After Phase 1 | Phase 1 DB exists |
| Phases 2, 3, 4 | In parallel | Each only needs Phase 1 |
| Phase 8 (Stretch) | After Phases 1-4 | All core phases |

**Recommended parallel assignments (3 agents):**

| Agent | Stream 1 | Stream 2 |
|---|---|---|
| Agent A | Phase 1 → Phase 2 | — |
| Agent B | Phase 5 → Phase 7 | Phase 3 (after Phase 1) |
| Agent C | Phase 6 (Titan) | Phase 4 (after Phase 1) |

---

### Phase 1: Database & Virtual Keys

Add a SQLite database to the proxy and create the first virtual key for
imas-codex workers. No behavior change for existing consumers.

**Tasks:**
1. Add `LITELLM_DATABASE_URL` to `.env` and `env.example`
   - Default: `sqlite:///home/ITER/mcintos/.local/share/imas-codex/services/litellm.db`
2. Add `database_url` to `general_settings` in `litellm_config.yaml`
3. Restart proxy — LiteLLM auto-creates tables on first start with DB
4. Generate a virtual key for codex workers via `/key/generate`
5. Add `LITELLM_API_KEY` to `.env` (the virtual key for workers)
6. Update `_build_kwargs()` in `discovery/base/llm.py`:
   - Use `LITELLM_API_KEY` when routing through the proxy
   - Fall back to `LITELLM_MASTER_KEY` for backwards compatibility
7. Add `llm keys` CLI subgroup to `llm_cli.py`:
   - `imas-codex llm keys list` — list all virtual keys
   - `imas-codex llm keys create --team <name> --budget <usd>` — create key
   - `imas-codex llm keys revoke <key-id>` — revoke key
   - `imas-codex llm spend [--team <name>]` — per-team spend summary

**Verification:**
- Existing workers continue to function
- Prompt caching still works (test with `scripts/test_prompt_caching.py`)
- `imas-codex llm status` shows virtual key count

### Phase 2: Multi-Credential Routing

*Depends on: Phase 1*

Add separate OpenRouter API keys per team for independent spend tracking
at the OpenRouter billing level. Uses the **team `model_group_alias`
approach** (see "Credential Routing Design" above).

**Tasks:**
1. Create OpenRouter API keys on openrouter.ai/settings/keys:
   - `OPENROUTER_API_KEY_CODEX` — imas-codex internal ($500/month limit)
   - `OPENROUTER_API_KEY_CLAUDE_CODE` — Claude Code users ($200/month limit)
   - `OPENROUTER_API_KEY_COPILOT_CLI` — Copilot CLI users ($100/month limit)
2. Add all keys to `.env` and `env.example`
3. Update `litellm_config.yaml`:
   - Add YAML anchors in `x-models` for DRY model ID definitions
   - Add `credential_list` with one credential per team
   - Add model entries: `{alias}-{team}` for each model × team combination
4. Rename existing `OPENROUTER_API_KEY` to `OPENROUTER_API_KEY_CODEX`
   - Keep `OPENROUTER_API_KEY` as a backwards-compat fallback in the CLI
5. Create teams via `/team/new` API **with `model_group_alias`** in
   `router_settings` (see "Example Teams & Keys" above)
6. Generate team-scoped virtual keys via `/key/generate`
7. Test credential isolation:
   - Using a codex virtual key: `model: opus` → `opus-codex`
     → `openrouter-codex` → appears on codex OpenRouter dashboard
   - Using a claude-code virtual key: `model: opus` → `opus-claude-code`
     → `openrouter-claude-code` → appears on claude-code OpenRouter dashboard

**Verification:**
- `imas-codex llm status` shows all credentials
- OpenRouter dashboard shows spend split by key
- Same `model: opus` call routes to different credentials based on caller's team

### Phase 3: Claude Code Setup

*Depends on: Phase 1 (needs a virtual key). Can run in parallel with Phases 2, 4.*

Configure Claude Code to route through the LiteLLM proxy on both WSL
(local development) and the ITER remote (facility operations).

**Tasks:**

#### 3a. ITER Remote Setup

Claude Code runs directly on the ITER login node (same machine as proxy):

1. Set environment variables in `~/.bashrc` or `~/.claude.env`:
   ```bash
   export ANTHROPIC_BASE_URL="http://localhost:18400"
   export ANTHROPIC_AUTH_TOKEN="sk-claude-xxxxxxxx"  # virtual key
   ```
2. Alternatively, configure in `~/.claude/settings.json`:
   ```json
   {
     "env": {
       "ANTHROPIC_BASE_URL": "http://localhost:18400",
       "ANTHROPIC_AUTH_TOKEN": "sk-claude-xxxxxxxx"
     }
   }
   ```
3. Update project `.claude/settings.json` to add the LiteLLM bash permission:
   ```json
   "Bash(curl http://localhost:18400/*)"
   ```

#### 3b. WSL Remote Setup

Claude Code on WSL reaches the proxy via SSH tunnel:

1. Start the SSH tunnel:
   ```bash
   ssh -f -N -L 18400:localhost:18400 iter
   ```
2. Set environment variables:
   ```bash
   export ANTHROPIC_BASE_URL="http://localhost:18400"
   export ANTHROPIC_AUTH_TOKEN="sk-claude-xxxxxxxx"
   ```
3. Add a convenience alias to `~/.bashrc`:
   ```bash
   alias llm-tunnel='ssh -f -N -L 18400:localhost:18400 iter'
   ```

**Verification:**
- `claude --model claude-sonnet-4.6` works on both WSL and ITER
- Requests appear under the `claude-code` team in spend tracking
- Spend tracked on the claude-code OpenRouter key

### Phase 4: Security Hardening

*Depends on: Phase 1 (needs DB). Can run in parallel with Phases 2, 3.*

Harden the proxy for production use on a shared-compute login node.

**Tasks:**
1. **Master key isolation**: Ensure master key is only in the systemd
   `EnvironmentFile` and never distributed to clients. All client access
   via virtual keys only.
2. **Admin endpoint restriction**: Configure LiteLLM to disable the
   admin UI in production, or restrict `/key/`, `/team/`, `/user/`
   endpoints to localhost-only requests.
3. **File permissions**:
   - `.env`: `chmod 600`
   - SQLite DB: `chmod 600`
   - Log files: `chmod 600`
4. **Budget alerts**: Configure Langfuse or LiteLLM webhooks to send
   alerts when a team reaches 80% of budget.
5. **Auth verification**: Extend `_show_llm_auth_status()` to verify
   that unauthenticated requests to `/v1/chat/completions` are rejected
   (already partially implemented).
6. **Log sanitization**: Ensure LiteLLM does not log full request/response
   content (only metadata — model, tokens, cost).

**Stretch:**
- TLS via Nginx reverse proxy (self-signed cert from ITER IT)
- IP allowlisting at the proxy level

### Phase 5: Model Config Update

*Independent — can run any time, in parallel with all other phases.*

Update `pyproject.toml` model sections and `litellm_config.yaml` model
list with correct OpenRouter model IDs.

**Current → Target model IDs:**

| Section | Current pyproject.toml | Target pyproject.toml | OpenRouter model ID |
|---------|----------------------|----------------------|---------------------|
| language | `google/gemini-3-flash-preview` | `google/gemini-3-flash-preview` | `openrouter/google/gemini-3-flash-preview` |
| vision | `google/gemini-3-flash-preview` | `google/gemini-3-flash-preview` | `openrouter/google/gemini-3-flash-preview` |
| agent | `anthropic/claude-opus-4.6` | `anthropic/claude-opus-4.6` | `openrouter/anthropic/claude-opus-4.6` |
| reasoning | `anthropic/claude-sonnet-4.6` | `anthropic/claude-sonnet-4.6` | `openrouter/anthropic/claude-sonnet-4.6` |
| compaction | `anthropic/claude-sonnet-4.6` | `anthropic/claude-sonnet-4.6` | `openrouter/anthropic/claude-sonnet-4.6` |

**Note:** OpenRouter model IDs use dots in version numbers
(`anthropic/claude-opus-4.6`), matching the pyproject.toml format.
The `ensure_openrouter_prefix()` function adds the `openrouter/` prefix
at call time.

**Tasks:**
1. Verify pyproject.toml model IDs match current OpenRouter format
2. Add new model aliases to `litellm_config.yaml` (with YAML anchors)
3. Update `prompt_caching.yaml` if new providers need cache control
4. Test all models via `scripts/test_prompt_caching.py --all`

### Phase 6: Local Model Deployment on Titan

*Independent — can run any time, requires Titan cluster access.*

Deploy an open-weight agentic model on the Titan cluster as a proof of
concept for local LLM inference, routed through the LiteLLM proxy.

#### Titan Cluster Hardware

| Component | Specification |
|-----------|---------------|
| **Node** | 1× `98dci4-gpu-0002` |
| **GPUs** | 8× NVIDIA Tesla P100-PCIE-16GB |
| **GPU VRAM** | 16 GB HBM2 per GPU, 128 GB total |
| **Compute Capability** | 6.0 (Pascal) — **no tensor cores** |
| **GPU Interconnect** | PCIe only (no NVLink) |
| **CPU** | 2× Intel Xeon E5-2689 v4, 20 cores total |
| **RAM** | 252 GB |
| **CUDA** | 12.2 (driver 535.86.10) |
| **Network** | No internet from compute nodes |
| **Current load** | 6 GPUs running embedding service, 2 GPUs idle |

#### Critical Hardware Constraints

1. **vLLM is not compatible** — requires compute capability ≥7.0;
   P100 is 6.0. Not usable.
2. **Ollama IS compatible** — supports compute capability ≥5.0.
   This is the recommended inference runtime for Titan.
3. **llama.cpp IS compatible** — uses CUDA directly with GGUF models.
   Alternative to Ollama with more control.
4. **No internet** — all model weights and binaries must be pre-staged
   from the login node.
5. **PCIe-only** — multi-GPU tensor parallelism will be bandwidth-limited.
   Prefer models that fit on 1-2 GPUs.
6. **No bfloat16** — Pascal only supports FP16 and FP32.
7. **2 GPUs currently available** (6 used by embedding service).

#### Open-Weight Agentic Model Candidates

Given the P100 constraints (16 GB per GPU, no tensor cores, FP16 only),
the best candidates are quantized models ≤14B parameters (dense) or
small MoE models:

| Model | Params | Type | VRAM (Q4) | Fits P100? | Agentic Quality | Notes |
|-------|--------|------|-----------|------------|-----------------|-------|
| **Qwen3 14B** | 14B | Dense | ~9 GB | ✅ 1 GPU | Very good | Best balance of quality/size; strong tool use |
| **Qwen3 30B-A3B** | 30B/3B active | MoE | ~8 GB | ✅ 1 GPU | Good | MoE keeps VRAM low; fast inference |
| **Devstral Small 2** | 24B | Dense | ~14 GB | ⚠️ Tight | Excellent | Mistral's coding agent; may need 2 GPUs |
| **DeepSeek-Coder V2 Lite** | 16B/2B active | MoE | ~10 GB | ✅ 1 GPU | Good | Code-focused |
| **Gemma 3 12B** | 12B | Dense | ~8 GB | ✅ 1 GPU | Moderate | Google's small model; less agentic |
| **Llama 3.3 8B** | 8B | Dense | ~5 GB | ✅ 1 GPU | Moderate | Fast; good for simple tasks |
| **GLM-4 9B** | 9B | Dense | ~6 GB | ✅ 1 GPU | Good | Zhipu's smaller model |

**Recommendation: Qwen3 14B (Q4_K_M quantization)**
- Best agentic capability in the size class
- 9 GB VRAM at Q4 — fits comfortably on 1× P100 16GB
- Strong tool use, coding, and reasoning benchmarks
- Well-supported by Ollama and llama.cpp
- Can run alongside the existing embedding service

#### Deployment Plan

**Pre-staging (login node with internet):**
```bash
# Install Ollama on login node (for model download)
curl -fsSL https://ollama.com/install.sh | sh

# Pull model to local cache
ollama pull qwen3:14b-q4_K_M

# Package for transfer to compute node
tar -czf qwen3-14b.tar.gz ~/.ollama/models/
scp qwen3-14b.tar.gz 98dci4-gpu-0002:/tmp/
```

**SLURM deployment (compute node):**
```bash
# SLURM job script: slurm/ollama-llm.sh
#!/bin/bash
#SBATCH --job-name=codex-llm
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=$HOME/.local/share/imas-codex/services/llm-local.log

# Unpack model if not already done
if [ ! -d "$HOME/.ollama/models" ]; then
    tar -xzf /tmp/qwen3-14b.tar.gz -C $HOME/
fi

# Start Ollama server
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_MODELS=$HOME/.ollama/models
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

ollama serve
```

**LiteLLM integration:**
```yaml
# Add to litellm_config.yaml when Ollama is running
- model_name: qwen3-14b
  litellm_params:
    model: ollama/qwen3:14b-q4_K_M
    api_base: http://98dci4-gpu-0002:11434
```

**Tasks:**
1. Install Ollama binary on login node and compute node
   (pre-download since compute has no internet)
2. Download Qwen3 14B Q4_K_M model on login node
3. Transfer model files to compute node
4. Create SLURM job script for Ollama server
5. Test Ollama serves the model correctly
6. Add local model entry to `litellm_config.yaml`
7. Test end-to-end: proxy → Ollama → Qwen3 14B
8. Add `llm local` CLI subgroup:
   - `imas-codex llm local start` — submit SLURM job
   - `imas-codex llm local stop` — cancel SLURM job
   - `imas-codex llm local status` — check job + model health
9. Update `ensure_openrouter_prefix()` → `ensure_model_prefix()`:
   skip `openrouter/` prefix for `ollama/` and `hosted_vllm/` models

**Performance expectations (Qwen3 14B Q4 on 1× P100):**
- Prompt processing: ~200-400 tokens/sec
- Generation: ~15-30 tokens/sec
- Suitable for batch processing, not real-time multi-agent

### Phase 7: Documentation

*Independent — can run any time, in parallel with all other phases.
Best started early so docs evolve alongside implementation.*

Create operator and user documentation for the multi-tenant gateway.

**Tasks:**
1. **`docs/llm-gateway.md`** — Operator guide:
   - Architecture diagram and credential routing explanation
   - How to add a new team/credential (step-by-step)
   - How to generate and distribute virtual keys
   - Budget monitoring and spend tracking via CLI
   - Troubleshooting (key rotation, budget exhaustion, DB issues)
2. **`docs/client-setup.md`** — LLM client setup guide (Claude Code, Copilot, OpenAI):
   - ITER remote setup (direct localhost)
   - WSL setup (SSH tunnel)
   - Verifying the connection works
   - Model selection and aliases
3. **Update `CLAUDE.md`** — Add proxy configuration section:
   - `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` usage
   - Which models are available via the proxy
4. **Update `env.example`** — Add all new environment variables with
   comments explaining each:
   - `OPENROUTER_API_KEY_CODEX`, `OPENROUTER_API_KEY_CLAUDE_CODE`,
     `OPENROUTER_API_KEY_COPILOT_CLI`
   - `LITELLM_DATABASE_URL`, `LITELLM_API_KEY`
   - `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`
5. **Update `AGENTS.md`** — Add LLM gateway section to Model & Tool
   Configuration covering:
   - Virtual key usage for proxy auth
   - Team/credential routing explanation
   - `imas-codex llm keys` and `imas-codex llm spend` commands

**Verification:**
- All env vars in `env.example` have comments
- `docs/llm-gateway.md` covers the full team onboarding workflow
- Claude Code setup instructions tested on both WSL and ITER

### Phase 8: Operational Tooling (Stretch)

*Requires: Phases 1-4 complete.*

1. **`imas-codex llm spend`** — query per-team spend from the SQLite DB
2. **`imas-codex llm keys rotate`** — automated key rotation
3. **Budget alerts** — webhook/email when team approaches limit
4. **Failover routing** — if local model is down, fall back to cloud model
5. **SSO/OAuth** — replace virtual keys with OIDC tokens from ITER IdP
6. **Request guardrails** — PII detection, content filtering

---

## Environment Variables (Target State)

```bash
# ── Existing (unchanged) ────────────────────────────────
LITELLM_MASTER_KEY=sk-litellm-...          # Admin only — never distribute
LITELLM_PROXY_URL=http://localhost:18400   # Override for proxy endpoint

# ── Renamed ─────────────────────────────────────────────
OPENROUTER_API_KEY_CODEX=sk-or-v1-...     # imas-codex internal use
# (OPENROUTER_API_KEY kept as backwards-compat alias)

# ── New (per-team OpenRouter keys) ──────────────────────
OPENROUTER_API_KEY_CLAUDE_CODE=sk-or-v1-... # Claude Code users
OPENROUTER_API_KEY_COPILOT_CLI=sk-or-v1-... # Copilot CLI users
LITELLM_DATABASE_URL=sqlite:///path/to/litellm.db
LITELLM_API_KEY=sk-codex-...              # Virtual key for workers

# ── Claude Code (user-level, not in .env) ───────────────
ANTHROPIC_BASE_URL=http://localhost:18400
ANTHROPIC_AUTH_TOKEN=sk-claude-...         # Virtual key for Claude Code
```

---

## Security Considerations

### Shared Login Node Threats

| Threat | Mitigation | Phase |
|--------|-----------|-------|
| Unauthorized proxy access | Virtual keys required; master key admin-only | 1 |
| API key exposure in `.env` | `chmod 600`; separate keys per team | 2, 4 |
| Network eavesdropping | TLS via Nginx (stretch goal) | 4 |
| Budget exhaustion by client | Per-team budget caps at LiteLLM + OpenRouter level | 2 |
| Process args visible in `ps` | systemd EnvironmentFile (not CLI args) | Already done |
| Admin endpoint abuse | Restrict `/key/`, `/team/` to localhost | 4 |
| Log file exposure | `chmod 600` on all log files | 4 |
| DB file exposure | `chmod 600` on SQLite file | 1 |
| `/proc/PID/environ` readable | Modern Linux default `hidepid=2` on `/proc` | Verify |

### Key Principle

**Defense in depth:** Budget limits at three layers:
1. **OpenRouter dashboard** — hard spending cap per API key
2. **LiteLLM virtual key** — `max_budget` + `budget_duration` per team
3. **Application code** — `ProviderBudgetExhausted` exception halts workers

---

## Code Changes Summary

| File | Change | Phase |
|------|--------|-------|
| `imas_codex/config/litellm_config.yaml` | YAML anchors, per-team model entries, credential list, DB URL | 1, 2 |
| `imas_codex/discovery/base/llm.py` | Use `LITELLM_API_KEY` for proxy auth; rename `ensure_openrouter_prefix` | 1, 6 |
| `imas_codex/cli/llm_cli.py` | Add `keys`, `spend`, `local` subcommands; team creation with `model_group_alias` | 1, 2, 6 |
| `imas_codex/settings.py` | Add `get_litellm_api_key()` accessor | 1 |
| `env.example` | Add new env vars with documentation | 2, 7 |
| `.claude/settings.json` | Add Claude Code proxy env vars | 3 |
| `pyproject.toml` | Verify model ID formats (dots for OpenRouter) | 5 |
| `slurm/ollama-llm.sh` | SLURM job script for local model | 6 |
| `docs/llm-gateway.md` | Operator guide for multi-tenant gateway | 7 |
| `docs/client-setup.md` | LLM client setup guide (Claude Code, Copilot, OpenAI) | 7 |
| `CLAUDE.md` | Proxy configuration section | 7 |
| `AGENTS.md` | LLM gateway section in Model & Tool Configuration | 7 |

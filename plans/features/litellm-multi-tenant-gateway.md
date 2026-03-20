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
┌─ Login Node ─────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  LiteLLM Proxy (:18400) + SQLite DB                            │ │
│  │                                                                 │ │
│  │  Virtual Keys ──────────────────────── Credentials              │ │
│  │  ├─ vk-codex-xxx   → team:codex       → openrouter-codex key   │ │
│  │  ├─ vk-claude-xxx  → team:claude-code → openrouter-external key│ │
│  │  └─ vk-copilot-xxx → team:copilot-cli → openrouter-external key│ │
│  │                                                                 │ │
│  │  Model Router ──────────────────────────────────────────        │ │
│  │  ├─ claude/*  ───────→ OpenRouter (Anthropic)                   │ │
│  │  ├─ gemini/*  ───────→ OpenRouter (Google)                      │ │
│  │  ├─ gpt/*     ───────→ OpenRouter (OpenAI)                      │ │
│  │  └─ local/*   ───────→ Ollama/llama.cpp on Titan (future)       │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│         ▲              ▲              ▲              ▲                │
│         │              │              │              │                │
│   imas-codex     Claude Code    Copilot CLI    Other clients         │
│   workers        (WSL/ITER)     (WSL/ITER)                           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Credential Routing Design

### Problem

LiteLLM's `litellm_credential_name` is a 1:1 binding — each model entry
gets exactly one credential. You cannot attach multiple credentials to a
single model entry and have LiteLLM pick one based on the caller's team.
The goal is: different users call `model: opus` and get routed to different
OpenRouter API keys (and thus different billing accounts) based on their
team identity.

### Options Evaluated

| Approach | Mechanism | Clients call | Credential selection | Verdict |
|---|---|---|---|---|
| **A. Team `model_group_alias`** | Duplicate models as `opus-codex` / `opus-external`; teams set `model_group_alias` in per-team router settings | `model: opus` | Automatic via team router settings | **Selected** |
| B. Access Groups | Two `opus` entries in different access groups | `model: opus` | Uncertain — LiteLLM may not filter within a model group by access group | Rejected (ambiguous) |
| C. Explicit model names | Define `opus-codex`, `opus-external`; restrict team access | `model: opus-codex` | Explicit — client must know their model name | Rejected (bad UX) |

### Selected Approach: Team `model_group_alias`

LiteLLM supports per-team router settings (docs: "Router Settings for
Keys and Teams"). The `model_group_alias` router setting remaps model
names at the team level, allowing transparent credential routing:

```
Client calls:  model: "opus"
  ↓
Team router settings resolve alias:
  Team codex:       opus → opus-codex       (uses openrouter-codex credential)
  Team claude-code: opus → opus-external    (uses openrouter-external credential)
  ↓
LiteLLM routes to the resolved model group with the correct credential
```

**How it works:**
1. Each model is defined twice in `litellm_config.yaml` with credential-
   suffixed internal names (`opus-codex`, `opus-external`)
2. Each credential-suffixed model points to the same OpenRouter model ID
   but uses a different `litellm_credential_name`
3. When creating teams via the API, `model_group_alias` maps the clean
   user-facing name to the credential-specific internal name
4. Clients use clean names (`opus`, `sonnet`, etc.) — the team's router
   settings transparently resolve to the correct backend

**Advantages:**
- Clients never need to know about credentials or internal model names
- Adding a new credential/team just requires a new set of model entries +
  team config — no client-side changes
- Spend tracking is automatic: each credential maps to a separate
  OpenRouter API key, so billing is isolated at the provider level
- LiteLLM's native credential usage tracking tags requests by credential

---

## Config Design

### `litellm_config.yaml` (target)

```yaml
# imas-codex LiteLLM Proxy — Multi-Tenant Configuration
#
# Start: imas-codex llm start
#
# Credentials are env-sourced. Virtual keys (per-team) are managed
# via the /key/generate API and stored in the SQLite database.
#
# CREDENTIAL ROUTING:
# Each model is defined once per credential. Teams use
# model_group_alias in their router settings to map clean names
# (opus, sonnet) to credential-specific names (opus-codex, opus-external).
# See "Example Teams & Keys" below.

# ── Credentials ──────────────────────────────────────────────
# Each credential maps to a separate OpenRouter API key for
# independent billing. Create keys at https://openrouter.ai/settings/keys
credential_list:
  - credential_name: openrouter-codex
    credential_values:
      api_key: os.environ/OPENROUTER_API_KEY_CODEX
    credential_info:
      description: "imas-codex internal use (discovery, agents, MCP)"

  - credential_name: openrouter-external
    credential_values:
      api_key: os.environ/OPENROUTER_API_KEY_EXTERNAL
    credential_info:
      description: "External clients (Claude Code, Copilot CLI, etc.)"

# ── Model Definitions ───────────────────────────────────────
# Each model appears twice: once per credential. The -codex and
# -external suffixes are internal names — clients never see them.
# Teams map clean aliases via model_group_alias router settings.
#
# OpenRouter model IDs (March 2026):
#   anthropic/claude-opus-4-6
#   anthropic/claude-sonnet-4-6
#   anthropic/claude-haiku-4-5-20251001
#   google/gemini-3-flash-preview
#   google/gemini-3-pro-preview
#   openai/gpt-5.4
#   openai/gpt-5.4-mini

model_list:
  # ── Anthropic Claude (codex credential) ────────────────
  - model_name: opus-codex
    litellm_params:
      model: openrouter/anthropic/claude-opus-4-6
      litellm_credential_name: openrouter-codex

  - model_name: sonnet-codex
    litellm_params:
      model: openrouter/anthropic/claude-sonnet-4-6
      litellm_credential_name: openrouter-codex

  - model_name: haiku-codex
    litellm_params:
      model: openrouter/anthropic/claude-haiku-4-5-20251001
      litellm_credential_name: openrouter-codex

  # ── Anthropic Claude (external credential) ─────────────
  - model_name: opus-external
    litellm_params:
      model: openrouter/anthropic/claude-opus-4-6
      litellm_credential_name: openrouter-external

  - model_name: sonnet-external
    litellm_params:
      model: openrouter/anthropic/claude-sonnet-4-6
      litellm_credential_name: openrouter-external

  - model_name: haiku-external
    litellm_params:
      model: openrouter/anthropic/claude-haiku-4-5-20251001
      litellm_credential_name: openrouter-external

  # ── Google Gemini (codex credential) ───────────────────
  - model_name: gemini-flash-codex
    litellm_params:
      model: openrouter/google/gemini-3-flash-preview
      litellm_credential_name: openrouter-codex

  - model_name: gemini-pro-codex
    litellm_params:
      model: openrouter/google/gemini-3-pro-preview
      litellm_credential_name: openrouter-codex

  # ── Google Gemini (external credential) ────────────────
  - model_name: gemini-flash-external
    litellm_params:
      model: openrouter/google/gemini-3-flash-preview
      litellm_credential_name: openrouter-external

  - model_name: gemini-pro-external
    litellm_params:
      model: openrouter/google/gemini-3-pro-preview
      litellm_credential_name: openrouter-external

  # ── OpenAI GPT (codex credential) ──────────────────────
  - model_name: gpt-codex
    litellm_params:
      model: openrouter/openai/gpt-5.4
      litellm_credential_name: openrouter-codex

  - model_name: gpt-mini-codex
    litellm_params:
      model: openrouter/openai/gpt-5.4-mini
      litellm_credential_name: openrouter-codex

  # ── OpenAI GPT (external credential) ───────────────────
  - model_name: gpt-external
    litellm_params:
      model: openrouter/openai/gpt-5.4
      litellm_credential_name: openrouter-external

  - model_name: gpt-mini-external
    litellm_params:
      model: openrouter/openai/gpt-5.4-mini
      litellm_credential_name: openrouter-external

  # ── Reasoning / Scoring (codex only — internal aliases) ─
  - model_name: reasoning-codex
    litellm_params:
      model: openrouter/anthropic/claude-sonnet-4-6
      litellm_credential_name: openrouter-codex

  - model_name: scoring-codex
    litellm_params:
      model: openrouter/google/gemini-3-flash-preview
      litellm_credential_name: openrouter-codex

  # ── Passthrough wildcard (codex credential) ────────────
  - model_name: "openrouter-codex-wildcard"
    litellm_params:
      model: "openrouter/*"
      litellm_credential_name: openrouter-codex

  - model_name: "openrouter-external-wildcard"
    litellm_params:
      model: "openrouter/*"
      litellm_credential_name: openrouter-external

  # ── Local models (Phase 6 — Titan cluster) ────────────
  # Uncomment when Ollama/llama.cpp is deployed on Titan.
  # Local models don't need credential duplication — no billing.
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

Each team's `model_group_alias` maps clean client-facing names to the
credential-specific internal names:

| Client calls | Team codex resolves to | Team claude-code resolves to |
|---|---|---|
| `opus` | `opus-codex` | `opus-external` |
| `sonnet` | `sonnet-codex` | `sonnet-external` |
| `haiku` | `haiku-codex` | `haiku-external` |
| `gemini-flash` | `gemini-flash-codex` | `gemini-flash-external` |
| `gemini-pro` | `gemini-pro-codex` | `gemini-pro-external` |
| `gpt` | `gpt-codex` | `gpt-external` |
| `gpt-mini` | `gpt-mini-codex` | `gpt-mini-external` |
| `reasoning` | `reasoning-codex` | *(not available)* |
| `scoring` | `scoring-codex` | *(not available)* |
| `*` | `openrouter-codex-wildcard` | `openrouter-external-wildcard` |

### Example Teams & Keys

```bash
# ── Create teams with model_group_alias ──────────────────
# The model_group_alias in router_settings maps clean model names
# to credential-specific internal names.

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
               "reasoning-codex", "scoring-codex",
               "openrouter-codex-wildcard"],
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

# Claude Code team (external — uses openrouter-external credential)
curl -X POST http://localhost:18400/team/new \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "team_alias": "claude-code",
    "max_budget": 200,
    "budget_duration": "30d",
    "models": ["opus-external", "sonnet-external", "haiku-external",
               "gemini-flash-external", "gemini-pro-external",
               "gpt-external", "gpt-mini-external",
               "openrouter-external-wildcard"],
    "router_settings": {
      "model_group_alias": {
        "opus": "opus-external",
        "sonnet": "sonnet-external",
        "haiku": "haiku-external",
        "gemini-flash": "gemini-flash-external",
        "gemini-pro": "gemini-pro-external",
        "gpt": "gpt-external",
        "gpt-mini": "gpt-mini-external"
      }
    }
  }'

# Copilot CLI team (external — same credential, separate budget)
curl -X POST http://localhost:18400/team/new \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "team_alias": "copilot-cli",
    "max_budget": 100,
    "budget_duration": "30d",
    "models": ["opus-external", "sonnet-external", "haiku-external",
               "gemini-flash-external", "gemini-pro-external",
               "gpt-external", "gpt-mini-external",
               "openrouter-external-wildcard"],
    "router_settings": {
      "model_group_alias": {
        "opus": "opus-external",
        "sonnet": "sonnet-external",
        "haiku": "haiku-external",
        "gemini-flash": "gemini-flash-external",
        "gemini-pro": "gemini-pro-external",
        "gpt": "gpt-external",
        "gpt-mini": "gpt-mini-external"
      }
    }
  }'

# ── Generate virtual keys ───────────────────────────────
# Codex workers key
curl -X POST http://localhost:18400/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_alias": "codex-workers",
    "team_id": "<codex-team-id>"
  }'
# → returns: {"key": "sk-codex-xxxxxxxx"}
# → client calls: model="opus" → resolved to opus-codex → openrouter-codex key

# Claude Code key
curl -X POST http://localhost:18400/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_alias": "claude-code-simon",
    "team_id": "<claude-code-team-id>",
    "max_budget": 200,
    "budget_duration": "30d"
  }'
# → returns: {"key": "sk-claude-xxxxxxxx"}
# → client calls: model="opus" → resolved to opus-external → openrouter-external key

# Copilot CLI key
curl -X POST http://localhost:18400/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_alias": "copilot-cli-simon",
    "team_id": "<copilot-cli-team-id>",
    "max_budget": 100,
    "budget_duration": "30d"
  }'
# → returns: {"key": "sk-copilot-xxxxxxxx"}
```

### Adding a New Credential / Team

To add a third OpenRouter key (e.g., for a collaborator's personal billing):

1. Add a credential to `credential_list`:
   ```yaml
   - credential_name: openrouter-collaborator
     credential_values:
       api_key: os.environ/OPENROUTER_API_KEY_COLLABORATOR
   ```
2. Duplicate the model entries with `-collaborator` suffix
3. Create a team with `model_group_alias` mapping clean names → `-collaborator` names
4. Generate a virtual key for the team

No client-side changes needed — the collaborator calls `model: opus` and
gets routed to their own OpenRouter key.

---

## Implementation Phases

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

Add separate OpenRouter API keys per team for independent spend tracking
at the OpenRouter billing level. Uses the **team `model_group_alias`
approach** (see "Credential Routing Design" above).

**Tasks:**
1. Create 2 OpenRouter API keys on openrouter.ai/settings/keys:
   - `OPENROUTER_API_KEY_CODEX` — imas-codex internal ($500/month limit)
   - `OPENROUTER_API_KEY_EXTERNAL` — external clients ($200/month limit)
2. Add both keys to `.env` and `env.example`
3. Update `litellm_config.yaml`:
   - Add `credential_list` with `openrouter-codex` and `openrouter-external`
   - Duplicate each model entry with `-codex` and `-external` suffixes,
     each pointing to its respective credential
4. Rename existing `OPENROUTER_API_KEY` to `OPENROUTER_API_KEY_CODEX`
   - Keep `OPENROUTER_API_KEY` as a backwards-compat fallback in the CLI
5. Create teams via `/team/new` API **with `model_group_alias`** in
   `router_settings` to map clean model names → credential-specific names
   (see "Example Teams & Keys" above for full curl commands)
6. Generate team-scoped virtual keys via `/key/generate`
7. Test credential isolation:
   - Using a codex virtual key: `model: opus` → resolves to `opus-codex`
     → uses `openrouter-codex` → appears on codex OpenRouter dashboard
   - Using an external virtual key: `model: opus` → resolves to
     `opus-external` → uses `openrouter-external` → appears on external
     OpenRouter dashboard

**Verification:**
- `imas-codex llm status` shows 2 credentials
- Langfuse shows team attribution per request
- OpenRouter dashboard shows spend split by key
- Same `model: opus` call routes to different credentials based on caller's team

### Phase 3: Claude Code Setup

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

#### 3c. Documentation

1. Add a `docs/claude-code-setup.md` guide covering both environments
2. Update `CLAUDE.md` with proxy configuration section
3. Add `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` to `env.example`

**Verification:**
- `claude --model claude-sonnet-4-6` works on both WSL and ITER
- Requests appear in Langfuse under the `claude-code` team
- Spend tracked separately on the external OpenRouter key

### Phase 4: Security Hardening

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

Update `pyproject.toml` model sections and `litellm_config.yaml` model
list with correct OpenRouter model IDs.

**Current → Target model IDs:**

| Section | Current pyproject.toml | Target pyproject.toml | OpenRouter model ID |
|---------|----------------------|----------------------|---------------------|
| language | `google/gemini-3-flash-preview` | `google/gemini-3-flash-preview` | `openrouter/google/gemini-3-flash-preview` |
| vision | `google/gemini-3-flash-preview` | `google/gemini-3-flash-preview` | `openrouter/google/gemini-3-flash-preview` |
| agent | `anthropic/claude-opus-4.6` | `anthropic/claude-opus-4-6` | `openrouter/anthropic/claude-opus-4-6` |
| reasoning | `anthropic/claude-sonnet-4.6` | `anthropic/claude-sonnet-4-6` | `openrouter/anthropic/claude-sonnet-4-6` |
| compaction | `anthropic/claude-sonnet-4.6` | `anthropic/claude-sonnet-4-6` | `openrouter/anthropic/claude-sonnet-4-6` |

**Note:** The model IDs in pyproject.toml use the bare provider format
(`anthropic/claude-opus-4-6`); the `ensure_openrouter_prefix()` function
adds the `openrouter/` prefix at call time.

**New models to add to litellm_config.yaml:**

| Alias | OpenRouter ID | Use case |
|-------|---------------|----------|
| `gemini-pro` | `openrouter/google/gemini-3-pro-preview` | Complex structured output |
| `gpt` | `openrouter/openai/gpt-5.4` | Alternative agentic model |
| `gpt-mini` | `openrouter/openai/gpt-5.4-mini` | Fast/cheap alternative |

**Note:** Each new model alias needs both `-codex` and `-external`
entries in `litellm_config.yaml` (per the credential routing design).
Teams' `model_group_alias` settings must also be updated to include the
new model mappings.

**Tasks:**
1. Fix model ID format in `pyproject.toml` (dots → hyphens where needed)
2. Add new model aliases to `litellm_config.yaml`
3. Update `prompt_caching.yaml` if new providers need cache control
4. Test all models via `scripts/test_prompt_caching.py --all`

### Phase 6: Local Model Deployment on Titan

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

### Phase 7: Operational Tooling (Stretch)

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

# ── New ─────────────────────────────────────────────────
OPENROUTER_API_KEY_EXTERNAL=sk-or-v1-...  # External clients
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
| `imas_codex/config/litellm_config.yaml` | Multi-credential, model duplication per credential, DB URL | 1, 2, 5, 6 |
| `imas_codex/discovery/base/llm.py` | Use `LITELLM_API_KEY` for proxy auth; rename `ensure_openrouter_prefix` | 1, 6 |
| `imas_codex/cli/llm_cli.py` | Add `keys`, `spend`, `local` subcommands; team creation with `model_group_alias` | 1, 2, 6 |
| `imas_codex/settings.py` | Add `get_litellm_api_key()` accessor | 1 |
| `env.example` | Add new env vars | 1, 2, 3 |
| `.claude/settings.json` | Add Claude Code proxy env vars | 3 |
| `pyproject.toml` | Fix model ID formats | 5 |
| `slurm/ollama-llm.sh` | SLURM job script for local model | 6 |
| `docs/claude-code-setup.md` | Claude Code configuration guide | 3 |

---

## Dependencies Between Phases

```
Phase 1 (DB + Keys) ──────┬──▶ Phase 2 (Multi-Credential)
                           │
                           ├──▶ Phase 3 (Claude Code Setup)
                           │
                           └──▶ Phase 4 (Security Hardening)

Phase 5 (Model Config) ────── Independent (can run any time)

Phase 6 (Local Models) ────── Independent (requires Titan access)

Phase 7 (Stretch) ─────────── Requires Phases 1-4 complete
```

Phases 1-4 are sequential. Phases 5 and 6 are independent and can
proceed in parallel with the others.

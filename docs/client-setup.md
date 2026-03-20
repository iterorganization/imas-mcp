# LLM Client Setup

Connect LLM clients to the imas-codex LiteLLM proxy for centralized
routing, cost tracking, and budget management.

## Supported Clients

| Client | Config Method | Env Vars |
|--------|--------------|----------|
| Claude Code | `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` | Shell / settings.json |
| VS Code Copilot Chat | `debug.overrideProxyUrl` | VS Code settings.json |
| Generic OpenAI client | `OPENAI_API_BASE` + `OPENAI_API_KEY` | Shell |

## Prerequisites

1. LiteLLM proxy running: `imas-codex llm status`
2. A team for your client: `imas-codex llm teams create --alias <team-name>`
3. A virtual key: `imas-codex llm keys create --team <team-id> --alias <key-name>`

---

## Claude Code

### ITER Remote (Direct)

Claude Code on the ITER login node connects directly:

```bash
# Add to ~/.bashrc or ~/.claude.env
export ANTHROPIC_BASE_URL="http://localhost:18400"
export ANTHROPIC_AUTH_TOKEN="sk-your-virtual-key"
```

Or in `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:18400",
    "ANTHROPIC_AUTH_TOKEN": "sk-your-virtual-key"
  }
}
```

### WSL / Off-Site (SSH Tunnel)

```bash
# Start tunnel
ssh -f -N -L 18400:localhost:18400 iter

# Set env vars (same as above)
export ANTHROPIC_BASE_URL="http://localhost:18400"
export ANTHROPIC_AUTH_TOKEN="sk-your-virtual-key"
```

Convenience alias:

```bash
alias llm-tunnel='ssh -f -N -L 18400:localhost:18400 iter'
```

---

## GitHub Copilot (VS Code)

> **Note:** The standalone `gh copilot` CLI cannot be redirected
> to a custom proxy. Only VS Code Copilot Chat supports proxy
> override via `debug.overrideProxyUrl`.

### VS Code Copilot Chat → LiteLLM Proxy

Add to your VS Code `settings.json` (User or Workspace):

```json
{
  "github.copilot.advanced": {
    "debug.overrideProxyUrl": "http://localhost:18400",
    "debug.testOverrideProxyUrl": "http://localhost:18400"
  }
}
```

When accessing via SSH tunnel from WSL:

```bash
ssh -f -N -L 18400:localhost:18400 iter
```

Then the same `localhost:18400` address works in VS Code settings.

### Alternative: LiteLLM Connector Extension

The [LiteLLM Connector for Copilot](https://marketplace.visualstudio.com/items?itemName=Gethnet.litellm-connector-copilot)
VS Code extension provides interactive proxy setup from within VS Code.

---

## Generic OpenAI-Compatible Clients

Any client supporting the OpenAI API can connect:

```bash
export OPENAI_API_BASE="http://localhost:18400/v1"
export OPENAI_API_KEY="sk-your-virtual-key"
```

Python example:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:18400/v1",
    api_key="sk-your-virtual-key",
)
response = client.chat.completions.create(
    model="sonnet",  # Uses the proxy's model alias
    messages=[{"role": "user", "content": "Hello!"}],
)
```

---

## Other Facilities (TCV, JT-60SA, etc.)

The proxy is designed for multi-facility deployment. Adjust the
tunnel target to match where the proxy runs:

```bash
# TCV
ssh -f -N -L 18400:localhost:18400 tcv

# JT-60SA
ssh -f -N -L 18400:localhost:18400 jt-60sa
```

The port is configurable via `IMAS_CODEX_LLM_PORT` or the
`[tool.imas-codex.llm]` section in `pyproject.toml`.

---

## Verification

```bash
# Check proxy health
curl -s http://localhost:18400/

# List available models
curl -s http://localhost:18400/v1/models \
  -H "Authorization: Bearer sk-your-virtual-key" | python3 -m json.tool

# Test a completion
curl -s http://localhost:18400/v1/chat/completions \
  -H "Authorization: Bearer sk-your-virtual-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Say hello"}]}' \
  | python3 -m json.tool
```

---

## Team & Key Management

```bash
# Create teams
imas-codex llm teams create --alias imas-codex --budget 500 --duration 30d
imas-codex llm teams create --alias claude-code --budget 200 --duration 30d

# Generate keys
imas-codex llm keys create --team <team-id> --alias worker-key
imas-codex llm keys create --team <team-id> --alias claude-code-simon

# Monitor spend
imas-codex llm spend
imas-codex llm spend --team claude-code

# Rotate a key
imas-codex llm keys rotate --key sk-old-key

# Revoke a key
imas-codex llm keys revoke sk-compromised-key
```

## Security

- **Never share the master key** (`LITELLM_MASTER_KEY`)
- Virtual keys are scoped to teams with budget limits
- Budget enforcement at three levels: OpenRouter, LiteLLM, application
- Keys can be revoked instantly
- File permissions: `.env` and `litellm.db` should be `chmod 600`
- For external users: create separate teams with restricted budgets

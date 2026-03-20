# Claude Code + LiteLLM Proxy Setup

Route Claude Code through the imas-codex LiteLLM proxy for centralized
cost tracking and budget management.

## Prerequisites

- LiteLLM proxy running: `imas-codex llm status`
- A virtual key for Claude Code: `imas-codex llm keys create --team <team-id> --alias claude-code`

## ITER Remote (Direct Access)

Claude Code runs on the same login node as the proxy:

```bash
# Add to ~/.bashrc or ~/.claude.env
export ANTHROPIC_BASE_URL="http://localhost:18400"
export ANTHROPIC_AUTH_TOKEN="sk-your-virtual-key-here"
```

Or configure in `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:18400",
    "ANTHROPIC_AUTH_TOKEN": "sk-your-virtual-key-here"
  }
}
```

## WSL / Off-Site (SSH Tunnel)

Claude Code on WSL reaches the proxy via SSH tunnel:

```bash
# Start the tunnel (add to ~/.bashrc or create an alias)
ssh -f -N -L 18400:localhost:18400 iter

# Then set env vars as above — same localhost:18400 address works
export ANTHROPIC_BASE_URL="http://localhost:18400"
export ANTHROPIC_AUTH_TOKEN="sk-your-virtual-key-here"
```

Convenience alias for `~/.bashrc`:

```bash
alias llm-tunnel='ssh -f -N -L 18400:localhost:18400 iter'
```

## Other Remotes (TCV, etc.)

The setup works on any facility by adjusting the tunnel target:

```bash
# If proxy runs at TCV instead of ITER:
ssh -f -N -L 18400:localhost:18400 tcv
```

The proxy port is configurable via `IMAS_CODEX_LLM_PORT` or the
`[tool.imas-codex.llm]` section in `pyproject.toml`.

## Verification

```bash
# Test the connection
curl -s http://localhost:18400/v1/models \
  -H "Authorization: Bearer $ANTHROPIC_AUTH_TOKEN" | python3 -m json.tool

# Test with Claude Code
claude "say hello"
```

## Team & Key Management

```bash
# Create a team for Claude Code users
imas-codex llm teams create --alias claude-code --budget 200 --duration 30d

# Generate a key for the team
imas-codex llm keys create --team <team-id> --alias claude-code-simon

# Check spend
imas-codex llm spend --team claude-code

# Rotate a compromised key
imas-codex llm keys rotate --key sk-old-key-here
```

## Security Notes

- **Never share the master key** (`LITELLM_MASTER_KEY`) — it has admin access
- Virtual keys are scoped to a team with budget limits
- Each team maps to a separate OpenRouter API key for billing isolation
- Keys can be revoked instantly: `imas-codex llm keys revoke <key>`

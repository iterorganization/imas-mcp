# GitHub Copilot CLI — Advanced User Guide

> Generated 2026-03-20. For your environment: SSH remote host, zellij tabs,
> VS Code Remote-SSH sessions, imas-codex project.

---

## Table of Contents

1. [Config File Reference](#1-config-file-reference)
2. [Shell Aliases & Launch Patterns](#2-shell-aliases--launch-patterns)
3. [Modes: Interactive → Plan → Autopilot](#3-modes-interactive--plan--autopilot)
4. [/fleet — Parallel Subagent Execution](#4-fleet--parallel-subagent-execution)
5. [/fleet vs Claude Code Agent Teams](#5-fleet-vs-claude-code-agent-teams)
6. [/ide — VS Code Integration Over SSH](#6-ide--vs-code-integration-over-ssh)
7. [ACP Server & Agent-to-Agent Communication](#7-acp-server--agent-to-agent-communication)
8. [LiteLLM Proxy vs Business Subscription](#8-litellm-proxy-vs-business-subscription)
9. [Zellij Multi-Tab Workflow](#9-zellij-multi-tab-workflow)
10. [Recommended Settings Analysis](#10-recommended-settings-analysis)
11. [Quick Reference Card](#11-quick-reference-card)

---

## 1. Config File Reference

**Location:** `~/.copilot/config.json` (override with `COPILOT_HOME` env var)

JSON does not support comments. A fully annotated `.jsonc` reference has been
saved alongside this guide at `/tmp/copilot-config-annotated.jsonc`.

### Installed Config

```json
{
  "model": "claude-opus-4.6",
  "experimental": true,
  "banner": "never",
  "alt_screen": true,
  "mouse": false,
  "theme": "dark",
  "render_markdown": true,
  "update_terminal_title": true,
  "compact_paste": true,
  "beep": true,
  "stream": true,
  "ide.auto_connect": true,
  "ide.open_diff_on_edit": true,
  "allowed_urls": [
    "https://docs.github.com",
    "*.github.com",
    "https://api.github.com"
  ],
  "trusted_folders": [
    "/home/ITER/mcintos/Code/imas-codex",
    "/home/ITER/mcintos/Code"
  ],
  "include_coauthor": true,
  "respectGitignore": true,
  "log_level": "default",
  "auto_update": true,
  "bash_env": false,
  "custom_agents.default_local_only": false
}
```

### Key Design Decisions

| Setting | Value | Rationale |
|---------|-------|-----------|
| `mouse: false` | Disabled | **Critical for zellij.** Mouse capture conflicts with zellij's own mouse handling (pane resize, tab click, text selection). Disable in Copilot, let zellij own the mouse. |
| `alt_screen: true` | Enabled | Clean terminal buffer. Works well in zellij panes. Set `false` if you see rendering artifacts. |
| `beep: true` | Enabled | Hear when tasks complete in other zellij tabs. Essential for multi-tab workflows. |
| `update_terminal_title: true` | Enabled | Zellij shows tab titles — you'll see what each Copilot instance is doing. |
| `compact_paste: true` | Enabled | Saves context window when pasting large blocks. |
| `experimental: true` | Enabled | Unlocks autopilot mode and upcoming features. Persisted. |
| `banner: "never"` | Suppressed | Skip animation for fast startup. |

---

## 2. Shell Aliases & Launch Patterns

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# ── Copilot CLI Aliases ──────────────────────────────────────────────
# Full autopilot with all permissions (the "just do it" mode)
alias cpy='copilot --yolo'

# Autopilot mode for scripting
alias cpa='copilot --autopilot --yolo'

# Quick non-interactive prompt (exits after completion)
alias cpq='copilot --yolo -p'

# Resume last session
alias cpc='copilot --continue'

# Resume with full permissions
alias cpr='copilot --yolo --resume'

# Specific model shortcuts
alias cp-opus='copilot --yolo --model claude-opus-4.6'
alias cp-sonnet='copilot --yolo --model claude-sonnet-4.6'
alias cp-haiku='copilot --yolo --model claude-haiku-4.5'
alias cp-codex='copilot --yolo --model gpt-5.3-codex'

# Non-interactive with max autopilot (CI-safe)
alias cp-ci='copilot --autopilot --yolo --max-autopilot-continues 15 -p'

# Start with extra directories
alias cp-multi='copilot --yolo --add-dir ~/Code'
```

### Environment Variables

```bash
# Force all permissions without flags
export COPILOT_ALLOW_ALL=true

# Override model via environment
export COPILOT_MODEL=claude-opus-4.6

# Custom config directory (if needed)
# export COPILOT_HOME=~/.copilot

# Custom instructions from additional directories
export COPILOT_CUSTOM_INSTRUCTIONS_DIRS=/home/ITER/mcintos/Code/shared-instructions

# Auth via PAT (alternative to OAuth device flow)
# export GH_TOKEN=ghp_xxxxx
```

---

## 3. Modes: Interactive → Plan → Autopilot

Press **Shift+Tab** to cycle through modes. The current mode shows in the status bar.

### Interactive Mode (default)
- You prompt, Copilot responds, you prompt again.
- Full control at every step.
- Best for: exploration, debugging, learning a codebase.

### Plan Mode
- Copilot creates a structured implementation plan before writing code.
- Asks clarifying questions to align on requirements.
- Saves plan to `plan.md` in the session folder.
- Best for: complex features, multi-file refactors.
- Trigger: `Shift+Tab` to plan mode, or `/plan <prompt>` from interactive.

### Autopilot Mode (experimental)
- Copilot works autonomously until the task is complete.
- No human input needed after the initial prompt.
- Stops when: task complete, error, Ctrl+C, or max continues reached.
- **Requires `--allow-all` or `/yolo` for full effectiveness.**
- Best for: well-defined tasks, test writing, CI fixes, batch operations.

### Recommended Workflow

```
1. Plan Mode:   /plan Add OAuth2 with Google and GitHub providers
2. Review:      Read and edit the plan, ask clarifying questions
3. Autopilot:   Accept plan → "Accept plan and build on autopilot + /fleet"
4. Monitor:     Watch progress, Ctrl+C if it goes off track
5. Review:      /diff to review all changes
6. Commit:      Commit with descriptive message
```

---

## 4. /fleet — Parallel Subagent Execution

### What It Does

`/fleet` tells Copilot to decompose a task into independent subtasks and
run them **in parallel** using subagents. Each subagent gets its own context
window and can use different models or custom agents.

### How To Use It

```
/fleet Implement the following features:
1. Add user authentication with JWT tokens
2. Create CRUD endpoints for the blog posts API
3. Write comprehensive unit tests for both features
4. Update the README with API documentation
```

### Key Behaviors

- The **main agent acts as orchestrator** — it plans the decomposition,
  dispatches subagents, and merges results.
- Subagents run in parallel when subtasks are independent.
- Sequential dependencies are respected (e.g., "write tests" waits for
  "implement feature").
- **Subagents use a low-cost model by default** to save premium requests.
- Override per-subtask: `Use GPT-5.3-Codex to create the API endpoints`
- Reference custom agents: `Use @test-writer to create unit tests`

### When To Use /fleet

| Good For | Not Good For |
|----------|-------------|
| Multi-file feature implementation | Sequential debugging |
| Test suite creation (one test per subagent) | Single-file changes |
| Documentation across multiple files | Tasks with tight coupling |
| Batch refactoring (rename pattern across files) | Exploratory coding |
| Coordinated API + client changes | Tasks requiring human judgment |

### Monitoring Fleet Tasks

```
/tasks          # View all background subagents and their status
```

### Fleet + Autopilot Combo

The most powerful pattern. In plan mode:
1. Create a detailed plan
2. When plan completes, select **"Accept plan and build on autopilot + /fleet"**
3. Copilot decomposes the plan and runs subtasks in parallel on autopilot

### Premium Request Cost

Fleet uses **more premium requests** than sequential work because each
subagent interacts with the LLM independently. The tradeoff is wall-clock
time: a 30-minute sequential task might complete in 10 minutes with fleet.

---

## 5. /fleet vs Claude Code Agent Teams

### Architecture Comparison

| Dimension | Copilot `/fleet` | Claude Code `task` tool |
|-----------|-----------------|------------------------|
| **Pattern** | **Orchestrator + Workers** | **Manager + Specialists** |
| **Decomposition** | Automatic by the main agent | Manual by the developer (or main agent) |
| **Parallelism** | True parallel — multiple subagents run simultaneously | True parallel — background agents run concurrently |
| **Context** | Each subagent has **isolated context** | Each agent has **isolated context** |
| **Custom Agents** | Can use custom agent profiles per subtask | Uses typed agents (explore, task, general-purpose, code-review, + custom) |
| **Model Control** | Subagents default to low-cost model; override per-subtask | Explicit model per agent type; override with `model` param |
| **Communication** | One-way: orchestrator → subagent → result | Bidirectional: `write_agent` for follow-ups, multi-turn |
| **State** | Stateless subagents — no follow-up after completion | Stateful — agents stay alive for `write_agent` refinement |
| **Scope** | Single CLI session | Single CLI session |

### Which Is Which?

**Copilot `/fleet` is a parallelization pattern.** The main agent acts as a
foreman, splitting work across disposable workers. Workers complete their
subtask and report back. No inter-worker communication.

**Claude Code's `task` tool is closer to a team model.** Agents are typed
specialists with different capabilities. Background agents persist and
accept follow-up messages (`write_agent`). The main agent acts as a manager
who can refine work iteratively. Multiple explore agents can safely run in
parallel, while task/general-purpose agents are serialized for safety.

### Practical Difference

- Use **`/fleet`** when you have a large task that decomposes into
  independent units of work and you want fast wall-clock completion.
- Use **Claude Code agents** when you need specialist reasoning (explore
  for codebase questions, code-review for diffs), iterative refinement
  (multi-turn conversations with agents), or when subtasks have complex
  dependencies.

### Can They Interoperate?

Not directly. They are features of different products (GitHub Copilot CLI
vs Claude Code / Copilot CLI's built-in agent system). However, within
Copilot CLI you get both: `/fleet` for parallelization and the built-in
agent system (explore, task, general-purpose, code-review) for specialist
delegation. The main agent decides when to spawn subagents vs handle work
directly.

---

## 6. /ide — VS Code Integration Over SSH

### How /ide Works

When VS Code is running, it creates a **lock file** in `~/.copilot/ide/`
that advertises its workspace. Copilot CLI reads these lock files to
discover and connect to IDE sessions.

### The SSH Problem (Known Limitation)

**When you SSH into a remote machine and run Copilot CLI, `/ide` often
fails to find VS Code Remote-SSH sessions.** This is because:

1. VS Code Remote-SSH runs a **server process** on the remote host, but
   it may not create the IDE lock file in the same `~/.copilot/ide/`
   directory that the CLI looks in.
2. The lock file mechanism relies on both processes sharing the same
   filesystem view of `~/.copilot/ide/`.
3. VS Code's Copilot extension and the standalone Copilot CLI use
   **different discovery mechanisms** — the extension communicates via
   the VS Code extension host, not filesystem lock files.

### Current Status

This is a **known limitation** rather than a bug. The IDE integration was
designed primarily for local co-located scenarios (CLI + VS Code on the
same machine). Remote-SSH adds a layer of indirection that the lock file
mechanism doesn't handle well.

### Workarounds

**1. Check for lock files manually:**
```bash
ls -la ~/.copilot/ide/
# If empty, VS Code Server isn't creating them
```

**2. Use /ide to manually connect:**
```
/ide
# If it shows "No IDE workspaces found", the lock files aren't present
```

**3. Separate workflows (pragmatic approach):**
Rather than trying to bridge CLI ↔ VS Code over SSH, use each tool for
what it does best:
- **Copilot CLI in zellij tab:** Agentic coding, autopilot, fleet
- **VS Code + Copilot extension:** Visual editing, inline completions,
  chat sidebar, diff review
- **Shared state:** Git is your synchronization layer. Commit from CLI,
  VS Code auto-refreshes.

**4. Use /diff for review instead of IDE diffs:**
```
/diff           # Review changes in terminal (no IDE needed)
```
Set `"ide.open_diff_on_edit": false` in config if you primarily work
from the CLI and don't want it trying to open diffs in a disconnected IDE.

**5. Forward the IDE connection (experimental):**
If both VS Code Server and Copilot CLI run on the same remote host,
they should share `~/.copilot/ide/`. If the lock files aren't appearing,
check that VS Code's Copilot extension is installed and active in the
remote session (not just locally).

### Switching Between CLI and VS Code

The smoothest workflow on a remote SSH host:

```
┌─ zellij ──────────────────────────────────┐
│ Tab 1: copilot --yolo    (agentic work)   │
│ Tab 2: copilot --yolo    (parallel task)  │
│ Tab 3: shell             (git, tests)     │
│ Tab 4: copilot --continue (resume)        │
└───────────────────────────────────────────┘
         ↕ git commit/push ↕
┌─ VS Code Remote-SSH ─────────────────────┐
│ Visual editing, inline Copilot, diffs     │
└───────────────────────────────────────────┘
```

---

## 7. ACP Server & Agent-to-Agent Communication

### What Is ACP?

The **Agent Client Protocol** is a standard protocol for communication
between coding agents and their clients (IDEs, other agents, CI systems).
Think of it as "LSP for AI agents."

### Does Copilot CLI Support Agent Teams via ACP?

**Not yet as a first-class feature.** Currently:

- Copilot CLI can **act as an ACP server** (`copilot --acp`)
- This allows external clients to send prompts and receive responses
- It enables **multi-agent coordination** where another orchestrator
  can treat Copilot CLI as one of its agents

### Running an ACP Server

```bash
# stdio mode (for IDE/tool integration)
copilot --acp --stdio

# TCP mode (for network access)
copilot --acp --port 3000
```

### Use Cases

1. **Build Copilot into custom IDEs** — any editor can be an ACP client
2. **CI/CD pipelines** — orchestrate agentic tasks in automated workflows
3. **Multi-agent systems** — coordinate Copilot with other AI agents
4. **Custom frontends** — specialized interfaces for specific workflows

### Agent Teams Today

Within Copilot CLI, you already have a form of agent teams:

| Agent | Purpose | Invocation |
|-------|---------|------------|
| **Explore** | Codebase analysis without polluting main context | Auto-delegated or via `/agent` |
| **Task** | Execute commands (tests, builds) with brief summaries | Auto-delegated |
| **General-purpose** | Complex multi-step tasks in separate context | Auto-delegated |
| **Code-review** | High signal-to-noise code review | `/review` |
| **Custom agents** | Your own specialists | `/agent` or `@agent-name` |
| **Fleet subagents** | Parallel execution workers | `/fleet` |

The main agent automatically delegates to these specialists when appropriate.
You can force delegation with `/agent` or `@agent-name` in prompts.

### Creating Custom Agents for Team-Like Behavior

Define agents in `~/.copilot/agents/` or `.github/agents/`:

```markdown
<!-- ~/.copilot/agents/test-writer.md -->
---
name: test-writer
description: Specialized test writing agent
model: claude-sonnet-4.6
tools:
  - shell
  - write
  - read
---

# Test Writer Agent

You are a testing specialist. When given code, you:
1. Analyze the code for testable behaviors
2. Write comprehensive unit tests
3. Run the tests to verify they pass
4. Fix any failures

Always use pytest. Always aim for >90% coverage.
```

Then reference in `/fleet` prompts:
```
/fleet Use @test-writer to create tests for the auth module,
       and @doc-writer to update the API documentation
```

---

## 8. LiteLLM Proxy vs Business Subscription

### Can Copilot CLI Use a LiteLLM Proxy?

**No.** Copilot CLI does **not** support custom LLM endpoints or proxies
like LiteLLM. Unlike Claude Code (which can be configured with
`ANTHROPIC_BASE_URL` or OpenRouter API keys), Copilot CLI is a
**closed system** that routes all requests through GitHub's infrastructure.

The models available are those GitHub provides through its Copilot service.
You cannot bring your own API keys or route through OpenRouter/LiteLLM.

### Cost Comparison: Subscription vs Token-Based

| Dimension | Copilot Business/Enterprise | OpenRouter + LiteLLM (Claude Code) |
|-----------|---------------------------|--------------------------------------|
| **Pricing Model** | Fixed monthly ($19-39/user/month) + premium request quotas | Pay-per-token (varies by model) |
| **Included Requests** | Base quota included in subscription | None — pure usage-based |
| **Premium Requests** | Pooled across org; multiplier per model | N/A — you pay exact token cost |
| **Model Access** | Curated set (Claude, GPT, Gemini) | Any model on OpenRouter (100+) |
| **Opus-class Cost** | 50x multiplier on premium requests | ~$15/M input, ~$75/M output tokens |
| **Sonnet-class Cost** | 1x multiplier (baseline) | ~$3/M input, ~$15/M output tokens |
| **Predictability** | Predictable monthly cost with quota | Variable — heavy use = high bills |
| **Organization** | Central billing, admin controls | Self-managed |

### Recommendation

**For your use case (heavy agentic development, autopilot, fleet), the
Copilot Business subscription is almost certainly more cost-effective.**

Here's why:

1. **Autopilot + Fleet = Many Premium Requests.** A single `/fleet`
   session can consume 50-200+ premium requests. At token-based pricing
   with Opus-class models, this could cost $5-50+ per session. With a
   business subscription, it's covered by your quota.

2. **Premium Request Pooling.** Enterprise/Business plans pool premium
   requests across the organization. If your team isn't using their full
   quota, you benefit from the surplus.

3. **No Token Accounting Overhead.** You don't need to manage API keys,
   monitor spend, set up proxies, or worry about runaway costs from
   autopilot sessions.

4. **Model Multipliers Favor Lighter Models.** Copilot's multiplier
   system (1x for Sonnet, 50x for Opus) incentivizes using the right
   model for the job. Fleet subagents default to low-cost models,
   keeping costs manageable.

5. **Integrated GitHub Features.** `/delegate`, `/pr`, `/review`,
   GitHub MCP server — these only work with the Copilot subscription.

### When Token-Based (OpenRouter) Makes Sense

- You need models not available in Copilot (specialized fine-tunes,
  open-source models, DeepSeek, etc.)
- You have very low usage (< $19/month worth of tokens)
- You need full control over routing, caching, and model selection
- You're using Claude Code as your primary tool and want one billing
  relationship

### Hybrid Approach

You can run **both** — use Copilot CLI (subscription) for GitHub-integrated
agentic work, and Claude Code (OpenRouter) for specialized tasks that need
custom models or the MCP tools specific to your project (like imas-codex).

---

## 9. Zellij Multi-Tab Workflow

### Recommended Layout

```
┌─────────────────────────────────────────────────────────┐
│ Tab 1: COPILOT-MAIN                                     │
│   copilot --yolo                                        │
│   Primary development session. Autopilot + fleet here.  │
│                                                         │
│ Tab 2: COPILOT-AUX                                      │
│   copilot --yolo --model claude-sonnet-4.6              │
│   Secondary tasks, quick questions, code review.        │
│                                                         │
│ Tab 3: SHELL                                            │
│   Git operations, test runs, file browsing.             │
│   Manual verification of agent work.                    │
│                                                         │
│ Tab 4: COPILOT-RESUME                                   │
│   copilot --continue                                    │
│   Resume previous sessions when needed.                 │
└─────────────────────────────────────────────────────────┘
```

### Zellij-Specific Tips

1. **Disable mouse in Copilot** (`"mouse": false` in config) — let
   zellij handle mouse events for pane resize and text selection.

2. **Use `beep: true`** — you'll hear when a long autopilot task
   finishes in another tab.

3. **Terminal title updates** — with `update_terminal_title: true`,
   zellij tab names show what each Copilot instance is working on.

4. **Multiple sessions are independent** — each `copilot` invocation
   is a separate session with its own context. They don't interfere.

5. **Share work via git** — if Tab 1's Copilot makes changes, Tab 3's
   shell can immediately `git diff`, `git add`, `git commit`.

### Zellij Keybinding Conflicts

| Zellij | Copilot | Conflict? | Resolution |
|--------|---------|-----------|------------|
| `Ctrl+T` (tab mode) | `Ctrl+T` (toggle reasoning) | Yes | Use zellij's `Ctrl+T` first, then Copilot's works inside the pane |
| `Ctrl+O` (session mode) | `Ctrl+O` (expand timeline) | Yes | Same — zellij mode takes precedence; use when prompt is empty |
| `Ctrl+S` (scroll mode) | `Ctrl+S` (run preserving input) | Yes | Be aware of zellij mode state |

Zellij's modal keybinding system means conflicts are usually resolved by
which "mode" zellij is in. In **normal mode**, keys pass through to the
active pane (Copilot).

---

## 10. Recommended Settings Analysis

### Every Config Setting Evaluated

| Setting | Default | Recommended | Impact on Velocity |
|---------|---------|-------------|-------------------|
| `model` | claude-sonnet-4.5 | **claude-opus-4.6** | ⬆⬆⬆ Higher quality code, fewer iterations |
| `experimental` | false | **true** | ⬆⬆⬆ Unlocks autopilot mode |
| `banner` | "once" | **"never"** | ⬆ Saves 2-3 seconds per launch |
| `alt_screen` | true | **true** | — Clean terminal buffer |
| `mouse` | true | **false** | ⬆ Fixes zellij mouse conflicts |
| `theme` | "auto" | **"dark"** | — Explicit, no detection lag |
| `render_markdown` | true | **true** | — Better readability |
| `update_terminal_title` | true | **true** | ⬆ Tab awareness in zellij |
| `compact_paste` | true | **true** | ⬆ Saves context on large pastes |
| `beep` | true | **true** | ⬆ Multi-tab notification |
| `stream` | true | **true** | ⬆ Real-time feedback |
| `ide.auto_connect` | true | **true** | — Worth trying, may not work over SSH |
| `ide.open_diff_on_edit` | true | **true** | — Nice when IDE is connected |
| `include_coauthor` | true | **true** | — Attribution tracking |
| `respectGitignore` | true | **true** | — Cleaner @ picker |
| `auto_update` | true | **true** | ⬆ Always get new features |
| `bash_env` | false | **false** | — Only enable if needed for PATH |
| `log_level` | "default" | **"default"** | — Bump to "debug" when troubleshooting |
| `copy_on_select` | varies | (default) | — Platform-dependent |
| `custom_agents.default_local_only` | false | **false** | — Keep org agents available |

### CLI Flags to Use Regularly

| Flag | What It Does | When To Use |
|------|-------------|-------------|
| `--yolo` / `--allow-all` | Enable all permissions | Every session (alias it) |
| `--autopilot` | Enable autopilot for `-p` mode | Scripting, CI |
| `--max-autopilot-continues N` | Cap autopilot steps | Safety net for autopilot |
| `--continue` | Resume last session | Pick up where you left off |
| `--resume` | Session picker | Find a specific past session |
| `--model MODEL` | Override model | Per-task model selection |
| `--add-dir PATH` | Add trusted directory | Multi-repo work |
| `--no-ask-user` | Suppress clarifying questions | When instructions are clear |
| `--reasoning-effort high` | Increase reasoning depth | Complex problems |
| `--agent AGENT` | Use specific custom agent | Specialist tasks |
| `--share` / `--share-gist` | Export session | Documentation, sharing |

### Slash Commands Power Users Should Know

| Command | Purpose | Pro Tip |
|---------|---------|---------|
| `/fleet PROMPT` | Parallel subagents | Combine with autopilot for max speed |
| `/plan PROMPT` | Structured planning | Always plan before large changes |
| `/delegate PROMPT` | Push to cloud agent | Fire-and-forget for tangential work |
| `/review` | AI code review | Use after `/fleet` to review all changes |
| `/diff` | Review changes | Before committing |
| `/compact` | Free context space | Rarely needed (auto-compaction at 95%) |
| `/context` | Token usage visualization | Monitor context health |
| `/usage` | Session statistics | Track premium request consumption |
| `/tasks` | View background subagents | Monitor fleet progress |
| `/research PROMPT` | Deep research | Uses GitHub search + web sources |
| `/share gist` | Share session to gist | Great for team knowledge sharing |
| `Ctrl+X then /` | Slash command mid-prompt | Change model without retyping |
| `Ctrl+G` | Edit in $EDITOR | Write complex prompts in vim/nvim |

---

## 11. Quick Reference Card

```
LAUNCH
  copilot --yolo                    Full permissions, interactive
  copilot --yolo --continue         Resume last session
  copilot --yolo -p "PROMPT"        One-shot (exits after)
  copilot --autopilot --yolo -p ... Autonomous one-shot

MODES (Shift+Tab to cycle)
  Interactive  →  Plan  →  Autopilot

KEY SHORTCUTS
  Shift+Tab    Cycle modes
  Ctrl+T       Toggle reasoning display
  Ctrl+O       Expand recent timeline (no input)
  Ctrl+E       Expand all timeline (no input)
  Ctrl+G       Edit prompt in $EDITOR
  Ctrl+S       Run command, keep input
  Ctrl+C       Cancel / clear / exit (x2)
  Esc          Cancel current operation
  @file        Include file in context

FLEET
  /fleet PROMPT                     Parallel subagent execution
  /tasks                            Monitor background tasks

DELEGATION
  /delegate PROMPT                  Push to Copilot cloud agent
  & PROMPT                          Shorthand for /delegate

SESSION
  /session                          Session info
  /session checkpoints              View compaction history
  /context                          Token usage
  /usage                            Premium request stats
  /compact                          Manual context compaction
  /clear                            Fresh conversation

REVIEW
  /diff                             Review all changes
  /review                           AI code review
  /pr                               PR operations

CONFIG
  /model                            Switch model mid-session
  /agent                            Browse custom agents
  /mcp                              Manage MCP servers
  /instructions                     View/toggle instruction files
  /allow-all                        Grant all permissions
  /reset-allowed-tools              Reset tool approvals

FILES
  ~/.copilot/config.json            Main configuration
  ~/.copilot/mcp-config.json        MCP server config
  ~/.copilot/lsp-config.json        LSP server config
  ~/.copilot/agents/                User-level custom agents
  ~/.copilot/ide/                   IDE lock files
  ~/.copilot/logs/                  CLI logs
  ~/.copilot/session-state/         Session data
  .github/agents/                   Repo-level custom agents
  .github/copilot-instructions.md   Repo-level instructions
```

---

*This guide is a session artifact. It lives in your session files directory
and is not committed to the repository.*

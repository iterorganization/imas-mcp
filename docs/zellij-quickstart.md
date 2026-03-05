# Zellij Session Quickstart

Organize `imas-codex` work across facilities using **one zellij session per facility**, with discovery tabs inside each session. Connect from WSL via the `cx` command.

## Architecture

```
WSL Client
 └─ cx iter          → SSH → iter login node → zellij session "codex"
 └─ cx iter tcv      → SSH → iter login node → zellij session "tcv"
 └─ cx iter jet      → SSH → iter login node → zellij session "jet"
 └─ cx iter jt-60sa  → SSH → iter login node → zellij session "jt-60sa"
```

All sessions run on the **same iter login node**. The `cx` command handles SSH + PTY + focus-reporting fixes for Windows Terminal/WSL.

## Session Layout

| Session    | Purpose                        | Tabs                                            |
|------------|--------------------------------|-------------------------------------------------|
| `codex`    | Core infrastructure & graph    | `graph`, `embed`, `tunnel`, `shell`             |
| `tcv`      | TCV facility discovery         | `paths`, `wiki`, `code`, `signals`, `docs`, `monitor` |
| `jet`      | JET facility discovery         | `paths`, `wiki`, `code`, `signals`, `docs`, `monitor` |
| `jt-60sa`  | JT-60SA facility discovery     | `paths`, `wiki`, `code`, `signals`, `docs`, `monitor` |
| `iter`     | ITER facility discovery        | `paths`, `wiki`, `code`, `signals`, `docs`, `monitor` |

## Install Layouts

Copy the layout files to the **remote** (iter) zellij config:

```bash
# From your WSL client
scp ~/.config/zellij/layouts/codex.kdl iter:~/.config/zellij/layouts/codex.kdl
scp ~/.config/zellij/layouts/facility.kdl iter:~/.config/zellij/layouts/facility.kdl
```

Or create them directly on iter:

```bash
ssh iter 'cat > ~/.config/zellij/layouts/codex.kdl' << 'LAYOUT'
layout {
    pane size=1 borderless=true {
        plugin location="tab-bar"
    }
    pane
    pane size=2 borderless=true {
        plugin location="status-bar"
    }

    tab name="graph" {
        pane
    }
    tab name="embed" {
        pane
    }
    tab name="tunnel" {
        pane
    }
    tab name="shell" {
        pane
    }
}
LAYOUT

ssh iter 'cat > ~/.config/zellij/layouts/facility.kdl' << 'LAYOUT'
layout {
    pane size=1 borderless=true {
        plugin location="tab-bar"
    }
    pane
    pane size=2 borderless=true {
        plugin location="status-bar"
    }

    tab name="paths" {
        pane
    }
    tab name="wiki" {
        pane
    }
    tab name="code" {
        pane
    }
    tab name="signals" {
        pane
    }
    tab name="docs" {
        pane
    }
    tab name="monitor" {
        pane
    }
}
LAYOUT
```

## Update `cx` for Facility Sessions

The `cx` script needs to use the `facility` layout when creating facility sessions. Add this to the remote `cx` on iter (`~/.local/bin/cx`), replacing the layout logic:

```bash
# Near the top, after SESSION is set:
FACILITY_SESSIONS="tcv jet jt-60sa iter"

# Choose layout based on session name
LAYOUT="codex"
for fac in $FACILITY_SESSIONS; do
    if [ "$SESSION" = "$fac" ]; then
        LAYOUT="facility"
        break
    fi
done
```

## Daily Workflow

### 1. Start infrastructure (once)

```bash
cx                          # Attach to "codex" session on iter
```

In the `codex` session tabs:

| Tab       | Command                                              |
|-----------|------------------------------------------------------|
| `graph`   | `uv run imas-codex graph start`                      |
| `embed`   | `uv run imas-codex embed start`                      |
| `tunnel`  | `uv run imas-codex tunnel start iter`                |
| `shell`   | general purpose shell / `uv run imas-codex graph shell` |

### 2. Open a facility session

```bash
# From WSL — each command opens a separate SSH + zellij session
cx iter tcv
cx iter jet
cx iter jt-60sa
```

### 3. Run discovery in facility tabs

In the `tcv` session (for example):

| Tab        | Command                                         |
|------------|--------------------------------------------------|
| `paths`    | `uv run imas-codex discover paths tcv`           |
| `wiki`     | `uv run imas-codex discover wiki tcv`            |
| `code`     | `uv run imas-codex discover code tcv`            |
| `signals`  | `uv run imas-codex discover signals tcv`         |
| `docs`     | `uv run imas-codex discover documents tcv`       |
| `monitor`  | `uv run imas-codex discover status tcv`          |

### 4. Monitor progress

```bash
# In the monitor tab — watch live stats
watch -n 10 uv run imas-codex discover status tcv

# Or check logs
tail -f ~/.local/share/imas-codex/logs/paths_tcv.log
tail -f ~/.local/share/imas-codex/logs/wiki_tcv.log
```

## Session Navigation

### Switching Between Sessions

From within any zellij session:

| Action                          | Key / Command                             |
|---------------------------------|-------------------------------------------|
| Detach (keeps session alive)    | `Ctrl+q`                                  |
| Then reattach to another        | `cx iter jet` (from WSL)                  |
| List sessions (from iter shell) | `zellij list-sessions`                    |
| Attach from iter shell          | `zellij attach tcv`                       |
| Kill a session                  | `zellij kill-session tcv`                 |

### Within a Session

| Action              | Key                        |
|---------------------|----------------------------|
| Switch tab          | `Ctrl+t` then tab name/num |
| Next/prev tab       | `Alt+←` / `Alt+→` (tab mode) |
| Move tab position   | Tab mode → `←` / `→`      |
| New tab             | `Ctrl+t` → `n`            |
| Rename tab          | `Ctrl+t` → `r`            |
| Move focus (panes)  | `Alt+h/j/k/l`             |
| Scroll mode         | `Ctrl+s`                   |
| Search in scroll    | `Ctrl+s` → `s`            |
| Lock (pass-through) | `Ctrl+g`                   |

### Quick Session Switching (No Detach)

Open multiple WSL terminal tabs, one per facility:

```
WSL Tab 1:  cx              → codex session (infra)
WSL Tab 2:  cx iter tcv     → tcv session
WSL Tab 3:  cx iter jet     → jet session
WSL Tab 4:  cx iter jt-60sa → jt-60sa session
```

Each WSL tab maps to one remote zellij session. The zellij sessions persist independently — close and reopen any WSL tab without losing state.

## Tips

- **Sessions are persistent.** SSH drops, laptop sleep, network changes — just `cx iter tcv` again to reattach.
- **Don't nest zellij.** If you're inside zellij and want a different session, detach first (`Ctrl+q`) or use a separate WSL terminal tab.
- **Logs over pipes.** Never pipe `imas-codex` CLI output. Check `~/.local/share/imas-codex/logs/` instead.
- **Rename tabs on the fly.** If you repurpose a tab, `Ctrl+t` → `r` to rename it.
- **Kill stale sessions.** `zellij kill-session <name>` or `zellij delete-all-sessions` for a clean slate.

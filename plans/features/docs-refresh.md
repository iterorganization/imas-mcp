# Documentation Refresh

> Priority: P4 — Stale docs mislead agents. Fix after code changes land.

## Problem

Architecture review (2026-04-14) found docs/README.md lists 6 of 17+ docs,
several docs have stale information, and one referenced file doesn't exist.
Fixing docs after Waves 1-3 avoids documenting features that are about to change.

## Verified Issues

### Critical (fix immediately)

| Doc | Issue |
|-----|-------|
| `docs/README.md` | Only lists 6 of 17+ architecture docs. Missing: graph-profiles, facility-access, services, keyring, client-setup, standard-names, standard-names-decisions, cocos-validation, copilot-cli-guide, zellij-quickstart, api/REPL_API |
| `docs/README.md` | Lists `west-access-justification.md` which does NOT exist on disk |
| `graph.md` | Documents `graph backup` and `graph restore` CLI commands that don't exist (internal helpers only) |
| `services.md` | LLM Proxy port listed as 18790, actual is 18400 (`LLM_BASE_PORT` in settings.py) |
| `standard-names.md` | References non-existent `docs/architecture/boundary.md` |
| `standard-names.md` | Phase table includes REVIEW as part of main pipeline (it's standalone/optional) |
| `api/REPL_API.md` | Documents `get_dd_overview()` which doesn't exist in server.py |

### Medium (fix during docs pass)

| Doc | Issue |
|-----|-------|
| `imas_dd.md` | CLI commands outdated (`uv run build-dd-graph` → `imas-codex imas dd build`) |
| `imas_dd.md` | README description says "MCP server" but doc covers graph schema & build |
| `cocos.md` | Documents COCOSMixin in schema but not found in `common.yaml` |
| `llamaindex-agents.md` | Documents `create_enrichment_agent()` which doesn't exist — likely superseded |
| `signals.md` | Incorrect relationship: says `TDIFunction -[:RESOLVES_TO_NODE]->` but actual is `DataReference -[:RESOLVES_TO_NODE]->` |
| `wiki.md` | Wrong CLI order: `uv run imas-codex wiki discover` → `imas-codex discover wiki` |
| `mdsplus.md` | 3 stale CLI commands: `discover-mdsplus`, `ingest-mdsplus`, `agent enrich` don't exist |
| `ids-mapping.md` | Confusing dual "IMASMapping" headings; stale `imas seed` command |

### Low (informational)

| Doc | Issue |
|-----|-------|
| `copilot-cli-guide.md` | Date says "2026-03-20" — user environment guide, not architecture |

## Implementation

Single agent task — fix all docs in one pass:

1. **Rewrite `docs/README.md`** — add all 17+ docs with correct descriptions
2. **Fix `graph.md`** — remove fake backup/restore CLI section, replace with note about internal helpers
3. **Fix `services.md`** — correct port 18790 → 18400
4. **Fix `standard-names.md`** — remove boundary.md reference, clarify review is standalone
5. **Fix `imas_dd.md`** — update CLI commands to current `imas-codex imas dd build` syntax
6. **Fix `api/REPL_API.md`** — remove `get_dd_overview()` or replace with current function
7. **Remove `west-access-justification.md` from README** (file doesn't exist)
8. **Assess `llamaindex-agents.md`** — mark as historical or update for current MCP architecture

## Phase Dependencies

```
This plan → after Waves 1-3 (so docs reflect latest code)
```

## Documentation Updates

| Target | Update |
|--------|--------|
| `docs/README.md` | Complete rewrite (this IS the plan) |
| All docs listed above | Individual fixes |

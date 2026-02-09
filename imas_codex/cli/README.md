# CLI Architecture

The IMAS Codex CLI follows a hierarchical structure with domain-specific subgroups.

## Discovery Commands

The `discover` group provides facility exploration with three domain subgroups:

```
imas-codex discover
├── status <facility>           # All domains
├── clear <facility>            # All domains (nuclear reset)
├── seed <facility>             # Seed root paths
├── inspect <facility>          # Debug view
│
├── paths <facility>            # Directory structure discovery
│   ├── status <facility>
│   └── clear <facility>
│
├── wiki <facility>             # Wiki page discovery
│   ├── status <facility>
│   └── clear <facility>
│
└── signals <facility>          # Facility signal discovery
    ├── status <facility>
    └── clear <facility>
```

## Design Principles

1. **Top-level aggregates**: `discover status` and `discover clear` operate on ALL domains
2. **Domain subgroups**: Each domain (paths, wiki, signals) runs discovery directly, with status/clear subcommands
3. **Consistent naming**: `<domain> <facility>` executes discovery, `<domain> status` shows stats
4. **Single source of truth**: Commands under one group, not duplicated at main level

## Examples

```bash
# Show status for all domains
imas-codex discover status tcv

# Show status for specific domain
imas-codex discover wiki status tcv

# Clear all discovery data
imas-codex discover clear tcv

# Clear only wiki data
imas-codex discover wiki clear tcv

# Run wiki discovery
imas-codex discover wiki tcv --cost-limit 5.0
```

## Status Output

`discover status <facility>` shows stats for all domains:
- **Paths**: discovered/scanned/scored counts, purpose distribution, high-value paths
- **Wiki**: pages/chunks/artifacts counts, accumulated LLM cost
- **Signals**: signal/enrichment counts

Use `-d/--domain` to filter: `discover status tcv -d wiki`

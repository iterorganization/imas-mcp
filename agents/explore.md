# Exploration Workflows

Remote facility exploration for discovering source files, MDSplus trees, and analysis codes.

## Setup: Ensure Fast Tools Are Available

Before exploring a new facility, ensure fast CLI tools are installed:

```python
# Check tool availability
python("print(check_tools('epfl'))")

# If tools are missing, install them (uses ReAct agent)
python("result = setup_tools('epfl'); print(result.summary)")

# Or quick install without agent
python("print(quick_setup('epfl', required_only=True))")
```

Tool definitions are in [`imas_codex/config/fast_tools.yaml`](../imas_codex/config/fast_tools.yaml).

## MCP Tools (Primary Interface)

When the Codex MCP server is running, use `python()` REPL:

```python
# Run command on facility (auto-detects local vs SSH)
python("print(run('ls /home/codes', facility='epfl'))")

# Get facility info and actionable paths
python("info = get_facility('epfl'); print(info['actionable_paths'][:5])")

# Search code examples
python("print(search_code('equilibrium reconstruction'))")

# Persist discoveries
python("""
ingest_nodes("FacilityPath", [
    {"id": "epfl:/home/codes/liuqe", "path": "/home/codes/liuqe",
     "facility_id": "epfl", "path_type": "code_directory", "status": "discovered"}
])
""")
```

## Fast CLI Tools

Tools are defined in [`imas_codex/config/fast_tools.yaml`](../imas_codex/config/fast_tools.yaml).

| Tool | Purpose | Fallback |
|------|---------|----------|
| `rg` | Fast pattern search | `grep -r` |
| `fd` | Fast file finder | `find . -name` |
| `tokei` | LOC by language | `wc -l` |
| `scc` | Code complexity | - |
| `dust` | Disk usage | `du -h` |

Use `run()` for unified local/remote execution:

```python
# Count Python files
python("print(run('fd -e py /home/codes | wc -l', facility='epfl'))")

# Find IMAS-related files
python("print(run('rg -l \"write_ids|read_ids\" /home/codes -g \"*.py\" | head -20', facility='epfl'))")

# Get complexity metrics
python("print(run('scc /home/codes/liuqe --format json', facility='epfl'))")

# Local facility (ITER) - no SSH needed
python("print(run('rg pattern /work/imas', facility='iter'))")
```

## FacilityPath Workflow

Track exploration progress with status transitions:

```
discovered → explored | skipped | stale
```

| Status | Meaning |
|--------|---------|
| `discovered` | Path found, awaiting exploration |
| `explored` | Fully examined, files queued for ingestion |
| `skipped` | Intentionally not exploring (low value) |
| `stale` | Path may no longer exist |

Interest score guidelines:

| Score | Use Case |
|-------|----------|
| 0.9+ | IMAS integration, IDS read/write |
| 0.7+ | MDSplus access, equilibrium codes |
| 0.5+ | General analysis codes |
| 0.3+ | Utilities, helpers |
| <0.3 | Config files, documentation |

## Persistence Checklist

After every exploration session, persist all discoveries:

| Discovery Type | Tool |
|----------------|------|
| Analysis codes | `ingest_nodes("AnalysisCode", [...])` |
| MDSplus trees | `ingest_nodes("MDSplusTree", [...])` |
| Directory paths | `ingest_nodes("FacilityPath", [...])` |
| Source files | `ingest_nodes("SourceFile", [...])` |
| Sensitive data (OS, tools) | `private(facility, {...})` |

Example persistence:

```python
python("""
# Persist discovered source files
ingest_nodes("SourceFile", [
    {"id": "epfl:/home/codes/liuqe/liuqe.py", 
     "path": "/home/codes/liuqe/liuqe.py",
     "facility_id": "epfl", "status": "discovered", 
     "interest_score": 0.8}
])
""")

python("""
# Update private infrastructure data
private("epfl", {"tools": {"rg": "14.1.1", "fd": "10.2.0"}})
""")
```

## Data Classification

Graph (public) - data access semantics:
- MDSplus tree names, diagnostic names
- Analysis code names, versions, paths
- TDI function names

Infrastructure (private) - security-sensitive:
- Hostnames, IPs, NFS mounts
- OS/kernel versions
- Tool availability
- User directories

## EPFL Wiki Access

```python
# Wiki requires -k flag (SSL cert issue)
python("print(ssh('curl -skL \"https://spcwiki.epfl.ch/wiki/PageName\"'))")
```

## Handoff to Ingestion

After discovering files, hand off to the `ingest` agent:
1. Queue files: `ingest_nodes("SourceFile", [...])`
2. Use handoff button → "Queue for Ingestion"
3. Ingest agent runs: `uv run imas-codex ingest run epfl`

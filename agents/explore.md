# Exploration Workflows

Facility exploration for discovering source files, MDSplus trees, and analysis codes.

**Architecture Documentation:** [docs/architecture/discovery.md](../docs/architecture/discovery.md)

## Critical: Check Constraints First

**Before ANY disk-intensive operation, check facility excludes:**

```python
python("""
info = get_facility('tcv')
excludes = info.get('excludes', {})
print(f"Large dirs: {excludes.get('large_dirs', [])}")
print(f"Depth limits: {excludes.get('depth_limits', {})}")
print(f"Recent notes: {info.get('exploration_notes', [])[-3:]}")
""")
```

This prevents repeating known timeouts. **Always use depth limits on large directories.**

## Critical: Check Locality First

**Always determine if you're on the target facility before choosing execution method:**

```bash
# Quick check
hostname
pwd
```

**Command Execution Decision Tree:**

1. **Single command on local facility?** → Use terminal directly (`rg`, `fd`, `dust`)
2. **Single command on remote facility?** → Use direct SSH (`ssh facility "command"`)
3. **Chained processing with logic?** → Use `python()` with `run()` (auto-detects local/remote)
4. **Graph queries or MCP functions?** → Use `python()` with `query()`, `ingest_nodes()`, etc.

## Setup: Ensure Fast Tools Are Available

Before exploring a new facility, ensure fast CLI tools are installed:

```python
# Check tool availability
python("print(check_tools('tcv'))")

# If tools are missing, install them (uses ReAct agent)
python("result = setup_tools('tcv'); print(result.summary)")

# Or quick install without agent
python("print(quick_setup('tcv', required_only=True))")
```

Tool definitions are in [`imas_codex/config/fast_tools.yaml`](../imas_codex/config/fast_tools.yaml).

## Execution Patterns

### Local Facility (You're On The Target System)

Use **terminal directly** for single commands:

```bash
# Check you're local
hostname  # e.g., 98dci4-srv-1003.iter.org

# Direct terminal commands
rg -l "IMAS" /work/imas
fd -e py /home/codes
dust -d 2 /work
tokei /home/codes/liuqe
```

### Remote Facility (Accessing Different System)

Use **direct SSH** for single commands:

```bash
# Direct SSH
ssh tcv "rg -l 'IMAS' /home/codes"
ssh tcv "fd -e py /home/codes | head -20"
ssh tcv "dust -d 2 /home/codes"
```

### Chained Processing & Graph Operations

Use `python()` only when you need:
- Multiple operations with intermediate processing
- Graph queries and data manipulation
- MCP functions (`add_to_graph`, `update_infrastructure`, `get_facility`)

```python
# Chained processing (run() auto-detects local/remote)
python("""
files = run('fd -e py /home/codes', facility='tcv').strip().split('\\n')
for f in files[:10]:
    content = run(f'head -20 {f}', facility='tcv')
    if 'write_ids' in content:
        print(f'IDS writer: {f}')
""")

# Graph operations
python("info = get_facility('tcv'); print(info['actionable_paths'][:5])")
python("print(search_code('equilibrium reconstruction'))")

# Persist discoveries
python("""
add_to_graph("FacilityPath", [
    {"id": "tcv:/home/codes/liuqe", "path": "/home/codes/liuqe",
     "facility_id": "tcv", "path_type": "code_directory", "status": "discovered"}
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

**Examples:**

```bash
# Local facility - direct terminal
rg -l "write_ids|read_ids" /home/codes -g "*.py" | head -20
fd -e py /home/codes | wc -l
scc /home/codes/liuqe --format json

# Remote facility - direct SSH
ssh tcv "rg -l 'write_ids|read_ids' /home/codes -g '*.py' | head -20"
ssh tcv "fd -e py /home/codes | wc -l"

# Chained processing - use python() with run()
python("""
files = run('fd -e py /home/codes', facility='tcv').strip().split('\\n')
for f in files[:10]:
    content = run(f'head -20 {f}', facility='tcv')
    if 'write_ids' in content:
        print(f'IDS writer: {f}')
""")
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
| Analysis codes | `add_to_graph("AnalysisCode", [...])` |
| MDSplus trees | `add_to_graph("MDSplusTree", [...])` |
| Directory paths | `add_to_graph("FacilityPath", [...])` |
| Source files | `add_to_graph("SourceFile", [...])` |
| Infrastructure (OS, tools, paths) | `update_infrastructure(facility, {...})` |

Example persistence:

```python
python("""
# Persist discovered source files
add_to_graph("SourceFile", [
    {"id": "tcv:/home/codes/liuqe/liuqe.py", 
     "path": "/home/codes/liuqe/liuqe.py",
     "facility_id": "tcv", "status": "discovered", 
     "interest_score": 0.8}
])
""")

python("""
# Update private infrastructure data
update_infrastructure("tcv", {"tools": {"rg": "14.1.1", "fd": "10.2.0"}})
""")
```

## Timeout Handling

When a command times out or hangs, **persist the constraint immediately**:

```python
python("""
# Get current excludes and merge
info = get_facility('tcv')
excludes = info.get('excludes', {})
large_dirs = excludes.get('large_dirs', [])
if '/work' not in large_dirs:
    large_dirs.append('/work')

depth_limits = excludes.get('depth_limits', {})
depth_limits['/work'] = 2  # Set safe depth limit

update_infrastructure('tcv', {
    'excludes': {
        'large_dirs': large_dirs,
        'depth_limits': depth_limits
    }
})

# Add human-readable context
add_exploration_note('tcv', '/work causes timeouts - use --max-depth 2 or target /work/imas')
""")
```

**Rules for timeout persistence:**
- `excludes.large_dirs`: Paths to avoid entirely with unbounded scans
- `excludes.depth_limits`: Max `--max-depth` value for each path
- `exploration_notes`: Human context explaining *why* (for future sessions)

**Never repeat a timeout** - always persist the learning before continuing.

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
python("print(run('curl -skL \"https://spcwiki.epfl.ch/wiki/PageName\"', facility='tcv'))")
```

## Handoff to Ingestion

After discovering files, hand off to the `ingest` agent:
1. Queue files: `add_to_graph("SourceFile", [...])`
2. Use handoff button → "Queue for Ingestion"
3. Ingest agent runs: `uv run imas-codex ingest run tcv`

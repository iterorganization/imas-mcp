## Tool Usage Guidelines

### Remote CLI Tools (Prefer Over Standard Unix)

| Tool | Instead Of | Speed | Example |
|-----------|------------|-------|---------|
| `rg` | `grep -r` | 10x faster | `rg 'IMAS' /work/projects -g '*.py'` |
| `fd` | `find` | 5x faster | `fd -e py /work/projects` |
| `tokei` | `wc -l` | Better | `tokei /path` |
| `dust` | `du -h` | Visual | `dust -d 2 /work` |

**Critical: fd requires path as trailing argument:**
```bash
# CORRECT - path is required
fd -e py /work/projects
fd 'pattern' /path

# WRONG - will hang or search cwd unexpectedly
fd -e py  # Missing path!
```

### Graph Queries

Use `query_neo4j` to check existing data before exploring:
```cypher
MATCH (f:Facility {id: $fid})
RETURN f.name, f.description
```

Always project specific properties - never `RETURN n`:
```cypher
-- Good
RETURN n.id, n.name, n.path

-- Bad (returns all properties including embeddings)
RETURN n
```

### IMAS Search

Use `search_imas_paths` for semantic search over the Data Dictionary:
```
search_imas("electron temperature")
search_imas("magnetic field boundary", ids_filter="equilibrium")
```

### Persisting Discoveries

- **Infrastructure** (tools, OS, paths) → `update_infrastructure()`
- **Source files** → `queue_source_files()` or `add_to_graph("SourceFile", [...])`
- **Exploration notes** → `add_exploration_note()`

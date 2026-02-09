# IMAS Codex REPL API

Public API for the MCP Python REPL (`python()` tool).

## Discovery

```python
# List all available functions
dir()

# Get help for a function
help(function_name)

# Get function signature
import inspect
inspect.signature(function_name)
```

## Graph Operations

### query()
```python
query(cypher: str, **params) -> list[dict[str, Any]]
```
Execute Cypher query and return results.

**Examples:**
```python
query('MATCH (f:Facility) RETURN f.id, f.name')
query('MATCH (t:TreeNode {tree_name: $tree}) RETURN t.path LIMIT 10', tree='results')
```

### semantic_search()
```python
semantic_search(text: str, index: str = "imas_path_embedding", k: int = 5) -> list[dict[str, Any]]
```
Vector similarity search on graph embeddings.

**Available indexes:**
- `imas_path_embedding`: IMAS Data Dictionary paths (61k)
- `code_chunk_embedding`: Code examples (8.5k chunks)
- `wiki_chunk_embedding`: Wiki documentation (25k chunks)
- `cluster_centroid`: Semantic clusters

**Examples:**
```python
semantic_search('plasma current', 'code_chunk_embedding', 3)
```

### embed()
```python
embed(text: str) -> list[float]
```
Get 256-dim embedding vector for text.

## Facility Configuration

### get_facility()
```python
get_facility(facility_id: str) -> dict[str, Any]
```
Load complete facility configuration (public + private merged).

Use this when you need full context for exploration.

**Examples:**
```python
config = get_facility('iter')
print(config['paths'])  # Has both public and private paths
print(config['tools'])  # Tool availability (private)
```

### get_facility_infrastructure()
```python
get_facility_infrastructure(facility_id: str) -> dict[str, Any]
```
Load only private facility infrastructure data.

Returns sensitive infrastructure data: hostnames, paths, tools, OS versions.

**Examples:**
```python
infra = get_facility_infrastructure('iter')
print(infra['tools'])  # Tool versions
print(infra['paths'])  # File system paths
```

### update_infrastructure()
```python
update_infrastructure(facility_id: str, data: dict[str, Any]) -> None
```
Update private facility infrastructure data.

Use this for sensitive infrastructure data:
- Tool versions and availability
- File system paths
- Hostnames and network info
- OS and environment details
- Exploration notes

**Examples:**
```python
# Update tool availability
update_infrastructure('iter', {
    'tools': {'rg': '14.1.1', 'fd': '10.2.0'}
})

# Add exploration notes
update_infrastructure('iter', {
    'exploration_notes': ['Found IMAS modules at /work/imas']
})
```

### update_metadata()
```python
update_metadata(facility_id: str, data: dict[str, Any]) -> None
```
Update public facility metadata.

Use this for public metadata:
- Facility name and description
- Machine name
- Data system types
- Wiki site URLs (if public)

**Examples:**
```python
# Update facility description
update_metadata('iter', {
    'description': 'ITER SDCC - Updated description'
})
```

### get_exploration_targets()
```python
get_exploration_targets(facility: str, limit: int = 10) -> list[dict[str, Any]]
```
Get prioritized exploration targets for a facility.

### get_tree_structure()
```python
get_tree_structure(tree_name: str, path_prefix: str = "", limit: int = 50) -> list[dict[str, Any]]
```
Get TreeNode structure from the graph.

## Remote Execution

### run()
```python
run(cmd: str, facility: str | None = None, timeout: int = 60) -> str
```
Execute command locally or via SSH (auto-detects based on facility).

**Examples:**
```python
run('rg pattern', facility='iter')  # Local (ITER is local)
run('rg pattern', facility='tcv')  # SSH to EPFL
run('rg pattern')                   # Local (no facility)
```

### check_tools()
```python
check_tools(facility: str | None = None) -> dict[str, Any]
```
Check availability of all fast CLI tools.

**Examples:**
```python
check_tools('tcv')
check_tools('iter')  # Local check
check_tools()        # Local check
```

## Code Search

### search_code()
```python
search_code(query_text: str, top_k: int = 5, facility: str | None = None, min_score: float = 0.5) -> str
```
Semantic code search over ingested code examples.

## IMAS Data Dictionary

### search_imas()
```python
search_imas(query_text: str, ids_filter: str | None = None, max_results: int = 10) -> str
```
Search IMAS Data Dictionary using semantic search.

**Examples:**
```python
search_imas('electron temperature profile')
search_imas('plasma current', ids_filter='core_profiles equilibrium')
```

### fetch_imas()
```python
fetch_imas(paths: str) -> str
```
Get full documentation for IMAS paths.

**Examples:**
```python
fetch_imas('equilibrium/time_slice/global_quantities/ip')
```

### list_imas()
```python
list_imas(paths: str, leaf_only: bool = True, max_paths: int = 100) -> str
```
List data paths in IDS with minimal overhead.

### check_imas()
```python
check_imas(paths: str) -> str
```
Validate IMAS paths for existence in the Data Dictionary.

### get_imas_overview()
```python
get_imas_overview(query: str | None = None) -> str
```
Get high-level overview of IMAS Data Dictionary structure.

## COCOS

### validate_cocos()
```python
validate_cocos(cocos: int) -> dict[str, Any]
```
Validate COCOS value (1-8 or 11-18).

### determine_cocos()
```python
determine_cocos(psi_axis: float, psi_edge: float, ip: float, b0: float, q: float | None = None, dp_dpsi: float | None = None) -> dict[str, Any]
```
Infer COCOS from equilibrium data using Sauter & Medvedev paper.

### cocos_sign_flip_paths()
```python
cocos_sign_flip_paths(ids_name: str | None = None) -> dict[str, Any]
```
Get paths requiring COCOS sign flip between DD3/DD4.

### cocos_info()
```python
cocos_info(cocos_value: int) -> dict[str, Any]
```
Get COCOS parameters for a given value.

## Schema Utilities

### get_schema()
```python
get_schema() -> Schema
```
Get the graph schema for introspection.

## MCP Tools

In addition to the REPL functions above, the MCP server provides these tools:

### add_to_graph
Create nodes in the knowledge graph with schema validation.

Use this for semantic data (files, codes, nodes).
For infrastructure data (paths, tools, OS), use `update_infrastructure()` instead.

**Examples:**
```python
# Via MCP tool
add_to_graph("SourceFile", [
    {"id": "tcv:/home/codes/file.py", "path": "/home/codes/file.py",
     "facility_id": "tcv", "status": "discovered"}
])

# Via python() REPL - not available as function
# Use the MCP tool instead
```

### update_facility_config
Read or update facility configuration (public or private).

**Examples:**
```python
# Via MCP tool
update_facility_config("iter", {"tools": {"rg": "14.1.1"}}, private=True)

# Via python() REPL - use the functions instead
update_infrastructure("iter", {"tools": {"rg": "14.1.1"}})
update_metadata("iter", {"description": "Updated"})
```

### get_graph_schema
Get complete graph schema for Cypher query generation.

## Data Classification

**Graph (semantic data):**
- Use: `add_to_graph()`
- What: SourceFile, FacilityPath, TreeNode, CodeExample, AnalysisCode
- Why: Searchable, relational, public

**Private Config (infrastructure):**
- Use: `update_infrastructure()`
- What: hostnames, paths, tools, OS versions, exploration notes
- Why: Sensitive, not searchable, gitignored

**Public Config (metadata):**
- Use: `update_metadata()`
- What: facility name, machine name, data systems
- Why: Safe for version control and graph

## Decision Tree

```
New information discovered
    │
    ├─ Is it sensitive? (paths, IPs, versions)
    │   ├─ YES → update_infrastructure()
    │   └─ NO → Continue
    │
    ├─ Is it infrastructure? (tools, OS, mounts)
    │   ├─ YES → update_infrastructure()
    │   └─ NO → Continue
    │
    ├─ Is it data semantics? (files, codes, nodes)
    │   ├─ YES → add_to_graph()
    │   └─ NO → update_metadata()
```

## Examples

### Exploration Workflow

```python
# 1. Check locality
import socket
print(socket.gethostname())

# 2. Get facility info
info = get_facility('iter')
print(info['paths'])

# 3. Run commands (auto-detects local/SSH)
result = run('fd -e py /work/imas/core', facility='iter')
files = result.strip().split('\n')

# 4. Add files to graph
add_to_graph('SourceFile', [
    {'id': f'iter:{f}', 'path': f, 'facility_id': 'iter', 'status': 'discovered'}
    for f in files[:10]
])

# 5. Update infrastructure
update_infrastructure('iter', {
    'exploration_notes': [f'Found {len(files)} Python files in /work/imas/core']
})
```

### IMAS Search Workflow

```python
# 1. Search for relevant paths
results = search_imas('electron temperature profile')
print(results)

# 2. Get detailed documentation
docs = fetch_imas('core_profiles/profiles_1d/electrons/temperature')
print(docs)

# 3. Check path existence
check = check_imas('core_profiles/profiles_1d/electrons/temperature')
print(check)
```

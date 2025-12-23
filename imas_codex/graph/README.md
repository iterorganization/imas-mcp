# Knowledge Graph Module

This module provides Neo4j-based storage for facility knowledge discovered through exploration.

## Architecture Strategy

### Single Source of Truth: LinkML

The **LinkML schema** (`imas_codex/schemas/facility.yaml`) is the authoritative source for:
- Class definitions → Neo4j **node labels**
- Slots with class ranges → Neo4j **relationships**
- Enumerations → Constrained property values
- Constraints → Database validation rules

### What Gets Generated

From the LinkML schema, we auto-generate:
- **Pydantic models** (`models.py`) via `gen-pydantic`
- **Node labels** derived from class names
- **Relationship types** derived from slot names with class ranges

### Current State (Transitional)

Currently `cypher.py` contains hard-coded enums for Neo4j labels and relationships.
This is **temporary** until we implement schema-driven generation.

**Current files**:
```
graph/
├── __init__.py      # Public API exports
├── client.py        # Neo4j client for CRUD operations
├── cypher.py        # ⚠️ Hard-coded labels (to be replaced)
└── models.py        # ✅ Auto-generated from LinkML
```

### Target State

Replace hard-coded cypher.py with runtime introspection of LinkML schema:

```python
from linkml_runtime.utils.schemaview import SchemaView

# Derive node labels from class names
sv = SchemaView("schemas/facility.yaml")
node_labels = [c.name for c in sv.all_classes().values()]

# Derive relationships from slots with class ranges
relationships = [
    (slot.name, slot.range)
    for slot in sv.all_slots().values()
    if slot.range in node_labels
]
```

## Recommended Approach

### Option A: linkml-store (Preferred)

[linkml-store](https://github.com/linkml/linkml-store) provides a unified abstraction
layer for multiple backends including **Neo4j**. It handles:

- Schema-to-graph mapping automatically
- CRUD operations with validation
- Semantic search with embeddings
- Import/export across backends

```bash
# Install
pip install linkml-store[neo4j]

# Use from CLI
linkml-store -d neo4j://localhost:7687 -c facilities insert data.json

# Use from Python
from linkml_store import Client
client = Client()
db = client.attach_database("neo4j://localhost:7687", schema="facility.yaml")
collection = db.get_collection("Facility")
collection.insert({"id": "epfl", "name": "EPFL/TCV"})
```

**Benefits**:
- Schema is the single source of truth
- Automatic label/relationship mapping
- Built-in validation
- Backend-agnostic (swap Neo4j for DuckDB in development)

### Option B: Custom Generator

Write a LinkML generator that produces cypher.py from the schema:

```bash
# Custom generator
gen-cypher schemas/facility.yaml > graph/cypher.py
```

The generator would:
1. Read LinkML classes → generate `NodeLabel` enum
2. Read slots with class ranges → generate `RelationType` enum
3. Extract `inverse` annotations for bidirectional relationships
4. Generate constraint queries from `identifier` and `required` slots

### Option C: Runtime Introspection (Current Direction)

Use `linkml_runtime.utils.schemaview.SchemaView` at runtime:

```python
from linkml_runtime.utils.schemaview import SchemaView
from dataclasses import dataclass

@dataclass
class GraphSchema:
    """Schema-derived graph metadata."""
    
    schema_view: SchemaView
    
    @classmethod
    def from_yaml(cls, path: str) -> "GraphSchema":
        return cls(SchemaView(path))
    
    @property
    def node_labels(self) -> list[str]:
        """Class names become Neo4j node labels."""
        return list(self.schema_view.all_classes().keys())
    
    @property
    def relationships(self) -> list[tuple[str, str, str]]:
        """Slots with class ranges become relationships.
        
        Returns: List of (slot_name, domain_class, range_class)
        """
        result = []
        classes = set(self.schema_view.all_classes().keys())
        for slot in self.schema_view.all_slots().values():
            if slot.range in classes:
                # This slot creates a relationship
                for domain in slot.domain_of or []:
                    result.append((slot.name, domain, slot.range))
        return result
```

## Schema Patterns for Neo4j

### Defining Relationships in LinkML

Use slots with class ranges. The slot name becomes the relationship type:

```yaml
classes:
  MDSplusServer:
    attributes:
      facility_id:            # Creates FACILITY_ID relationship
        range: Facility       # Target node type
        required: true
```

For more semantic relationship names, use `slot_uri`:

```yaml
slots:
  facility_id:
    range: Facility
    slot_uri: facility:HOSTED_BY  # Neo4j uses: -[:HOSTED_BY]->
    inverse: hosts               # Optional: for bidirectional queries
```

### Annotations for Neo4j-Specific Behavior

Use LinkML annotations for database hints:

```yaml
classes:
  Facility:
    annotations:
      neo4j:index: [ssh_host, machine]  # Create indexes
      neo4j:constraint: id              # Unique constraint
    attributes:
      id:
        identifier: true  # → UNIQUE constraint
        required: true    # → NOT NULL
```

## Migration Path

1. **Phase 1** (Current): Hard-coded cypher.py + generated models.py
2. **Phase 2**: Add `GraphSchema` class with runtime introspection
3. **Phase 3**: Remove cypher.py enums, derive from schema
4. **Phase 4**: Consider linkml-store for full abstraction

## Testing

```bash
# Start Neo4j
docker-compose up neo4j -d

# Run graph tests
uv run pytest tests/graph/ -v
```

## References

- [LinkML Schema](../schemas/facility.yaml) - Source of truth
- [linkml-store docs](https://linkml.io/linkml-store/)
- [LinkML Runtime SchemaView](https://linkml.io/linkml/developers/schemaview.html)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)

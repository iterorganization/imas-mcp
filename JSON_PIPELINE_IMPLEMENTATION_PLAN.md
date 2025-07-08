# JSON Pipeline Enhancement Implementation Plan

## Overview

Transform the current basic JSON accessor into a relationship-aware, physics-domain organized pipeline that enhances MCP tool responses with comprehensive context.

## Phase 1: Relationship Mapping Foundation (Week 1-2)

### 1.1 Enhanced Graph Analysis

Extend existing `IMASGraphAnalyzer` with physics relationship detection.

```python
# File: imas_mcp/graph_analyzer.py
class IMASGraphAnalyzer:
    # ...existing code...

    def find_physics_relationships(self, path: str) -> Dict[str, List[str]]:
        """Find related paths by physics domain."""
        return {
            "same_quantity": self._find_same_physics_quantity(path),
            "coordinate_shared": self._find_coordinate_relationships(path),
            "workflow_related": self._find_workflow_relationships(path)
        }

    def _find_same_physics_quantity(self, path: str) -> List[str]:
        """Find same physical quantity in different IDS."""
        quantity = path.split('/')[-1]  # e.g., 'density', 'temperature'
        related = []
        for other_path in self.all_paths:
            if other_path.endswith(quantity) and other_path != path:
                related.append(other_path)
        return related
```

### 1.2 Relationship-Aware JSON Accessor

Enhance `JsonDataDictionaryAccessor` with relationship context.

```python
# File: imas_mcp/json_data_accessor.py
class JsonDataDictionaryAccessor:
    # ...existing code...

    def get_with_relationships(self, path: str) -> Dict[str, Any]:
        """Get path data with full relationship context."""
        base_data = self.get_path_info(path)

        if not self.graph_analyzer:
            self.graph_analyzer = IMASGraphAnalyzer()

        relationships = self.graph_analyzer.find_physics_relationships(path)

        return {
            "path_info": base_data,
            "relationships": relationships,
            "physics_context": self._get_physics_context(path),
            "usage_patterns": self._get_usage_patterns(path)
        }
```

## Phase 2: Enhanced MCP Tools (Week 2-3)

### 2.1 Relationship-Enhanced Search

Upgrade search tool with relationship context.

```python
# File: imas_mcp/mcp_server_ai.py
async def search_imas(
    query: str,
    include_relationships: bool = True,
    # ...existing parameters...
) -> Dict[str, Any]:
    # ...existing search logic...

    if include_relationships:
        for result in results:
            path = result["path"]
            relationships = data_accessor.get_with_relationships(path)
            result.update({
                "related_quantities": relationships["relationships"]["same_quantity"][:3],
                "coordinate_connections": relationships["relationships"]["coordinate_shared"][:3],
                "physics_context": relationships["physics_context"]
            })

    return {
        "results": results,
        "workflow_suggestions": await _generate_workflow_suggestions(results, ctx),
        # ...existing fields...
    }
```

### 2.2 Bulk Export Tool

Add comprehensive IDS export capability.

```python
# File: imas_mcp/mcp_server_ai.py
async def export_ids_complete(
    ids_name: str,
    include_examples: bool = True,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """Export complete IDS with all relationships and context."""

    ids_data = data_accessor.get_ids_data(ids_name)
    relationships = {}

    # Get relationships for key paths
    key_paths = data_accessor.get_key_paths(ids_name, limit=20)
    for path in key_paths:
        relationships[path] = data_accessor.get_with_relationships(path)

    export_data = {
        "ids_overview": {
            "name": ids_name,
            "structure": data_accessor.get_ids_structure(ids_name),
            "key_quantities": key_paths
        },
        "relationships": relationships,
        "physics_domains": data_accessor.get_physics_domains(ids_name)
    }

    if include_examples and ctx:
        examples_prompt = f"""Generate practical Python code examples for working with {ids_name} IDS.
        Include: data access patterns, typical workflows, coordinate handling.
        Key paths: {key_paths[:5]}"""

        ai_response = await ctx.sample(examples_prompt, temperature=0.2)
        export_data["code_examples"] = ai_response

    return export_data
```

## Phase 3: Physics Domain Organization (Week 3-4)

### 3.1 Domain Catalog Generator

Create physics-domain organized data catalogs.

```python
# File: imas_mcp/domain_catalog.py
class PhysicsDomainCatalog:
    PHYSICS_DOMAINS = {
        "equilibrium": ["equilibrium", "profiles_1d", "profiles_2d"],
        "transport": ["core_profiles", "transport", "core_transport"],
        "heating": ["ec_launchers", "ic_antennas", "nb_injection"],
        "mhd": ["mhd", "disruptions", "sawteeth"],
        "diagnostics": ["magnetics", "interferometer", "thomson_scattering"]
    }

    def generate_domain_export(self, domain: str) -> Dict[str, Any]:
        """Generate comprehensive domain-specific export."""
        domain_ids = self.PHYSICS_DOMAINS.get(domain, [])

        catalog = {
            "domain_info": {
                "name": domain,
                "description": self._get_domain_description(domain),
                "included_ids": domain_ids
            },
            "cross_connections": self._map_cross_ids_relationships(domain_ids),
            "typical_workflows": self._extract_workflows(domain_ids),
            "key_quantities": self._identify_key_quantities(domain_ids)
        }

        return catalog
```

### 3.2 Domain-Aware MCP Tool

Add physics domain export tool.

```python
# File: imas_mcp/mcp_server_ai.py
async def export_physics_domain(
    domain: str,
    include_workflows: bool = True,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """Export complete physics domain with cross-IDS relationships."""

    catalog = PhysicsDomainCatalog()
    domain_export = catalog.generate_domain_export(domain)

    if include_workflows and ctx:
        workflow_prompt = f"""Describe typical research workflows for {domain} physics domain in IMAS.
        Include: data flow between IDS, common analysis patterns, coordinate transformations.
        Available IDS: {domain_export['domain_info']['included_ids']}"""

        workflows = await ctx.sample(workflow_prompt, temperature=0.3)
        domain_export["ai_workflow_guide"] = workflows

    return domain_export
```

## Phase 4: Validation & Optimization (Week 4-5)

### 4.1 Data Validation Pipeline

Add validation for exported data integrity.

```python
# File: imas_mcp/validation.py
from pydantic import BaseModel, validator
from typing import List, Dict, Any

class PhysicsQuantity(BaseModel):
    path: str
    units: str
    coordinates: List[str]
    physics_domain: str

    @validator('units')
    def units_not_empty(cls, v):
        if not v or v == 'undefined':
            raise ValueError('Units must be defined')
        return v

class RelationshipExport(BaseModel):
    primary_path: str
    related_quantities: List[str]
    relationship_types: List[str]
    physics_context: Dict[str, Any]

    def validate_export(self) -> bool:
        """Validate relationship export consistency."""
        return len(self.related_quantities) > 0
```

### 4.2 Caching Layer

Add intelligent caching for relationship data.

```python
# File: imas_mcp/cache_manager.py
import hashlib
import json
from pathlib import Path

class RelationshipCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def get_cached_relationships(self, path: str) -> Optional[Dict]:
        """Get cached relationship data."""
        cache_key = hashlib.md5(path.encode()).hexdigest()
        cache_file = self.cache_dir / f"rel_{cache_key}.json"

        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None

    def cache_relationships(self, path: str, data: Dict):
        """Cache relationship data."""
        cache_key = hashlib.md5(path.encode()).hexdigest()
        cache_file = self.cache_dir / f"rel_{cache_key}.json"

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
```

## Implementation Schedule

| Phase       | Duration | Deliverables                                         | Dependencies                 |
| ----------- | -------- | ---------------------------------------------------- | ---------------------------- |
| **Phase 1** | Week 1-2 | Enhanced graph analysis, relationship-aware accessor | Existing `IMASGraphAnalyzer` |
| **Phase 2** | Week 2-3 | Enhanced search tool, bulk export tool               | Phase 1 complete             |
| **Phase 3** | Week 3-4 | Domain catalogs, domain export tool                  | Phase 2 complete             |
| **Phase 4** | Week 4-5 | Validation pipeline, caching layer                   | Phase 3 complete             |

## Success Metrics

1. **Relationship Coverage**: 80% of search results include relevant relationships
2. **Bulk Efficiency**: Single domain export replaces 10+ individual queries
3. **Response Quality**: AI-enhanced exports provide actionable workflows
4. **Performance**: <500ms response time for bulk exports
5. **Validation**: 100% exported data passes validation checks

## Integration Points

- **Extends existing**: `IMASGraphAnalyzer`, `JsonDataDictionaryAccessor`
- **Enhances current**: MCP tools gain relationship context
- **Maintains compatibility**: All existing tools continue working
- **Adds capabilities**: New bulk export and domain tools

This plan builds incrementally on existing infrastructure while adding the relationship-aware capabilities needed for comprehensive MCP responses.

## Implementation Notes

### Current State vs. Target

- **Current**: Basic JSON file access with simple path lookups
- **Target**: Relationship-aware, physics-domain organized pipeline
- **Approach**: Evolutionary enhancement of existing components

### Key Benefits

1. **Reduced MCP Requests**: Bulk operations replace multiple individual queries
2. **Enhanced Context**: Physics relationships provide deeper understanding
3. **Better AI Responses**: Rich context enables more accurate AI enhancements
4. **Domain Organization**: Physics-aware data organization matches research workflows

### Risk Mitigation

- **Backward Compatibility**: All existing functionality preserved
- **Incremental Rollout**: Each phase adds capabilities without breaking changes
- **Performance**: Caching layer ensures response times remain acceptable
- **Validation**: Data integrity checks prevent inconsistent exports

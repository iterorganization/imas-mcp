# IMAS MCP Server - Short Term Development Plan

## Executive Summary

This plan outlines a complete redesign of the IMAS MCP server to address current limitations:

- Too many granular requests slowing LLM interaction
- Insufficient contextual information in responses
- Need for richer, more comprehensive data access

**Goal**: Create a high-performance MCP server that provides comprehensive, contextual responses through bulk data access and intelligent templating.

## Overall Strategy: Multi-Modal Knowledge Graph Approach

### Core Principles

1. **Bulk over granular**: Return comprehensive datasets in single requests
2. **Context-rich responses**: Include relationships, examples, and documentation
3. **Template-driven formatting**: Structured, consistent outputs for LLMs
4. **Semantic understanding**: Physics-aware search and discovery
5. **Best practices**: Use established libraries and patterns

---

## Phase 1: Data Model Transformation (Weeks 1-2)

### 1.1 Multi-Level JSON Export Pipeline

#### Primary Objective

Transform the monolithic IDSDef.xml into a layered JSON architecture that supports both fast bulk access and granular queries.

#### 1.1.1 High-Level Catalog Structure

```python
# File: imas_mcp/exports/ids_catalog.json
{
    "metadata": {
        "version": "4.0.0",
        "export_timestamp": "2025-06-25T12:00:00Z",
        "total_ids": 83,
        "total_leaf_nodes": 48000
    },
    "ids_catalog": {
        "core_profiles": {
            "description": "Plasma core profiles and transport data",
            "leaf_count": 847,
            "max_depth": 6,
            "primary_coordinates": ["rho_tor_norm", "time"],
            "physics_domain": "transport",
            "documentation_coverage": 0.95,  # Fraction of documented fields
            "related_ids": ["equilibrium", "transport", "mhd"],
            "common_use_cases": [
                "plasma_analysis",
                "transport_modeling",
                "profile_fitting"
            ]
        },
        "equilibrium": {
            "description": "Magnetohydrodynamic equilibrium data",
            "leaf_count": 1203,
            "max_depth": 7,
            "primary_coordinates": ["psi", "time"],
            "physics_domain": "equilibrium",
            "documentation_coverage": 0.88,
            "related_ids": ["core_profiles", "mhd", "wall"],
            "common_use_cases": [
                "equilibrium_reconstruction",
                "stability_analysis",
                "geometry_definition"
            ]
        }
        // ... 81 more IDS entries
    }
}
```

#### 1.1.2 Detailed IDS Structure Files

```python
# File: imas_mcp/exports/detailed/core_profiles.json
{
    "ids_info": {
        "name": "core_profiles",
        "description": "Plasma core profiles and transport data",
        "version": "4.0.0"
    },
    "coordinate_systems": {
        "rho_tor_norm": {
            "description": "Normalized toroidal flux coordinate",
            "units": "-",
            "range": [0.0, 1.0],
            "usage": "Primary radial coordinate for profiles"
        },
        "time": {
            "description": "Time coordinate",
            "units": "s",
            "usage": "Temporal evolution of profiles"
        }
    },
    "paths": {
        "core_profiles/profiles_1d/electrons/density": {
            "documentation": "Electron density profile",
            "units": "m^-3",
            "coordinates": ["time", "rho_tor_norm"],
            "lifecycle": "active",
            "data_type": "FLT_1D",
            "physics_context": {
                "domain": "plasma_physics",
                "phenomena": ["transport", "confinement"],
                "typical_values": {
                    "core": "1e20 m^-3",
                    "edge": "1e19 m^-3"
                }
            },
            "related_paths": [
                "core_profiles/profiles_1d/electrons/temperature",
                "core_profiles/profiles_1d/electrons/pressure",
                "equilibrium/time_slice/profiles_1d/pressure"
            ],
            "usage_examples": [
                {
                    "scenario": "Basic density profile access",
                    "code": "density = ids.core_profiles.profiles_1d[time_idx].electrons.density",
                    "notes": "Access density at specific time slice"
                }
            ],
            "validation_rules": {
                "min_value": 0.0,
                "units_required": true,
                "coordinate_check": "must_match_coordinate_arrays"
            }
        }
        // ... all other paths
    },
    "semantic_groups": {
        "electron_properties": [
            "core_profiles/profiles_1d/electrons/density",
            "core_profiles/profiles_1d/electrons/temperature",
            "core_profiles/profiles_1d/electrons/pressure"
        ],
        "transport_coefficients": [
            "core_profiles/profiles_1d/electrons/d_eff",
            "core_profiles/profiles_1d/electrons/v_eff"
        ]
    }
}
```

#### 1.1.3 Relationship Graph

```python
# File: imas_mcp/exports/relationships.json
{
    "cross_ids_relationships": {
        "equilibrium_to_core_profiles": {
            "type": "coordinate_mapping",
            "relationships": [
                {
                    "equilibrium_path": "equilibrium/time_slice/profiles_1d/psi",
                    "core_profiles_path": "core_profiles/profiles_1d/grid/psi",
                    "relationship_type": "same_coordinate",
                    "usage": "flux_surface_mapping"
                }
            ]
        }
    },
    "physics_concepts": {
        "confinement": {
            "description": "Plasma confinement physics",
            "relevant_paths": [
                "core_profiles/profiles_1d/electrons/density",
                "equilibrium/time_slice/global_quantities/beta_pol",
                "mhd/time_slice/linear/growth_rate"
            ],
            "key_relationships": [
                "pressure_balance",
                "stability_limits",
                "transport_barriers"
            ]
        }
    },
    "unit_families": {
        "density_units": {
            "base_unit": "m^-3",
            "paths_using": [
                "core_profiles/profiles_1d/electrons/density",
                "core_profiles/profiles_1d/ion/density"
            ],
            "conversion_factors": {
                "cm^-3": 1e-6
            }
        }
    }
}
```

#### 1.1.4 Implementation Code Structure

```python
# File: imas_mcp/core/xml_parser_enhanced.py
"""Enhanced XML parser for IMAS Data Dictionary with relationship extraction."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from pydantic import BaseModel, Field


class CoordinateSystem(BaseModel):
    """Coordinate system definition."""
    description: str
    units: str
    range: Optional[List[float]] = None
    usage: str


class PhysicsContext(BaseModel):
    """Physics context for a data field."""
    domain: str
    phenomena: List[str]
    typical_values: Dict[str, str] = Field(default_factory=dict)


class ValidationRules(BaseModel):
    """Validation rules for data fields."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    units_required: bool = True
    coordinate_check: Optional[str] = None


class UsageExample(BaseModel):
    """Code usage example."""
    scenario: str
    code: str
    notes: str


class DataPath(BaseModel):
    """Complete data path information."""
    documentation: str
    units: str
    coordinates: List[str]
    lifecycle: str
    data_type: str
    physics_context: PhysicsContext
    related_paths: List[str] = Field(default_factory=list)
    usage_examples: List[UsageExample] = Field(default_factory=list)
    validation_rules: ValidationRules = Field(default_factory=ValidationRules)


class IdsDetailed(BaseModel):
    """Detailed IDS information."""
    ids_info: Dict[str, str]
    coordinate_systems: Dict[str, CoordinateSystem]
    paths: Dict[str, DataPath]
    semantic_groups: Dict[str, List[str]] = Field(default_factory=dict)


class DataDictionaryTransformer:
    """Transform IDSDef.xml into layered JSON structure."""

    def __init__(self, xml_path: Path, output_dir: Path):
        self.xml_path = xml_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def transform_complete(self) -> Dict[str, Path]:
        """Transform XML to complete JSON structure."""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # Extract all IDS information
        ids_data = self._extract_ids_data(root)

        # Generate outputs
        outputs = {}
        outputs['catalog'] = self._generate_catalog(ids_data)
        outputs['detailed'] = self._generate_detailed_files(ids_data)
        outputs['relationships'] = self._generate_relationships(ids_data)

        return outputs

    def _extract_ids_data(self, root: ET.Element) -> Dict[str, Any]:
        """Extract structured data from XML root."""
        # Implementation details...
        pass

    def _generate_catalog(self, ids_data: Dict[str, Any]) -> Path:
        """Generate high-level catalog file."""
        catalog_path = self.output_dir / "ids_catalog.json"
        # Implementation details...
        return catalog_path

    def _generate_detailed_files(self, ids_data: Dict[str, Any]) -> List[Path]:
        """Generate detailed IDS files."""
        detailed_dir = self.output_dir / "detailed"
        detailed_dir.mkdir(exist_ok=True)

        paths = []
        for ids_name, ids_info in ids_data.items():
            detailed_path = detailed_dir / f"{ids_name}.json"
            # Generate detailed JSON...
            paths.append(detailed_path)

        return paths

    def _generate_relationships(self, ids_data: Dict[str, Any]) -> Path:
        """Generate relationship graph."""
        rel_path = self.output_dir / "relationships.json"
        # Implementation details...
        return rel_path
```

### 1.2 Leverage Existing Tools Enhancement

Build on your current implementation while adding new capabilities:

```python
# File: imas_mcp/core/data_model.py
"""Enhanced data models using Pydantic for validation."""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Literal
from enum import Enum


class PhysicsDomain(str, Enum):
    """Physics domains in IMAS."""
    TRANSPORT = "transport"
    EQUILIBRIUM = "equilibrium"
    MHD = "mhd"
    HEATING = "heating"
    DIAGNOSTICS = "diagnostics"
    WALL = "wall"


class DataLifecycle(str, Enum):
    """Data lifecycle status."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    OBSOLETE = "obsolete"


class IdsMetadata(BaseModel):
    """IDS metadata structure."""
    name: str
    description: str
    leaf_count: int
    max_depth: int
    primary_coordinates: List[str]
    physics_domain: PhysicsDomain
    documentation_coverage: float = Field(ge=0.0, le=1.0)
    related_ids: List[str] = Field(default_factory=list)
    common_use_cases: List[str] = Field(default_factory=list)

    @validator('documentation_coverage')
    def validate_coverage(cls, v):
        """Ensure coverage is between 0 and 1."""
        return max(0.0, min(1.0, v))
```

---

## Phase 2: Enhanced MCP Tools with Template System (Weeks 3-4)

### 2.1 Bulk Data Tools Architecture

#### 2.1.1 Core Tool Design Philosophy

Replace many small, targeted tools with fewer, comprehensive tools that return rich, structured data. Each tool should provide sufficient context to minimize subsequent requests.

#### 2.1.2 Primary Bulk Tools Implementation

```python
# File: imas_mcp/tools/bulk_tools.py
"""Comprehensive MCP tools for bulk data access."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal

from mcp.server.fastmcp import FastMCP

from ..core.data_model import IdsMetadata, PhysicsDomain
from ..templates.response_formatter import ResponseFormatter
from ..search.hybrid_search import HybridSearchEngine


class BulkDataTools:
    """High-level MCP tools for comprehensive data access."""

    def __init__(self, data_dir: Path, search_engine: HybridSearchEngine):
        self.data_dir = data_dir
        self.search_engine = search_engine
        self.formatter = ResponseFormatter()

        # Load cached data structures
        self._load_cached_data()

    def _load_cached_data(self):
        """Load pre-processed JSON data."""
        with open(self.data_dir / "ids_catalog.json") as f:
            self.catalog = json.load(f)
        with open(self.data_dir / "relationships.json") as f:
            self.relationships = json.load(f)


# MCP Tool Implementations
mcp = FastMCP("IMAS-Enhanced")
bulk_tools = BulkDataTools(Path("exports"), search_engine)


@mcp.tool()
def get_ids_complete_context(
    ids_name: str,
    include_related: bool = True,
    max_depth: int = 3,
    response_format: Literal["detailed", "summary", "code_examples"] = "detailed"
) -> str:
    """
    Return comprehensive IDS context including structure, documentation,
    relationships, and examples.

    Args:
        ids_name: Name of the IDS (e.g., 'core_profiles', 'equilibrium')
        include_related: Whether to include related IDS information
        max_depth: Maximum depth for hierarchical structure
        response_format: Format of the response (detailed/summary/code_examples)

    Returns:
        Formatted comprehensive context about the IDS

    Example Usage:
        get_ids_complete_context(
            ids_name="core_profiles",
            include_related=True,
            response_format="detailed"
        )
    """
    # Load detailed IDS data
    ids_file = bulk_tools.data_dir / "detailed" / f"{ids_name}.json"
    if not ids_file.exists():
        available_ids = list(bulk_tools.catalog["ids_catalog"].keys())
        return f"IDS '{ids_name}' not found. Available IDS: {', '.join(available_ids)}"

    with open(ids_file) as f:
        ids_data = json.load(f)

    # Get related IDS data if requested
    related_data = {}
    if include_related:
        related_ids = bulk_tools.catalog["ids_catalog"][ids_name]["related_ids"]
        for related_id in related_ids[:3]:  # Limit to 3 most related
            related_file = bulk_tools.data_dir / "detailed" / f"{related_id}.json"
            if related_file.exists():
                with open(related_file) as f:
                    related_data[related_id] = json.load(f)

    # Format response using templates
    context_data = {
        "ids_data": ids_data,
        "related_data": related_data,
        "max_depth": max_depth,
        "relationships": bulk_tools.relationships.get(f"{ids_name}_relationships", {})
    }

    return bulk_tools.formatter.format_ids_context(
        context_data,
        format_type=response_format
    )


@mcp.tool()
def search_semantic_concepts(
    query: str,
    context_type: Literal["physics", "diagnostic", "geometry", "transport"] = "physics",
    max_results: int = 10,
    include_examples: bool = True
) -> str:
    """
    Find conceptually related elements using semantic search across all IDS.

    Args:
        query: Natural language query (e.g., "electron temperature profile")
        context_type: Type of physics context to emphasize
        max_results: Maximum number of results to return
        include_examples: Whether to include code examples in results

    Returns:
        Formatted search results with semantic relationships

    Example Usage:
        search_semantic_concepts(
            query="plasma confinement and pressure profiles",
            context_type="physics",
            include_examples=True
        )
    """
    # Perform hybrid search (semantic + lexical)
    search_results = bulk_tools.search_engine.search_hybrid(
        query=query,
        context_filter=context_type,
        max_results=max_results
    )

    # Enhance results with relationship data
    enhanced_results = []
    for result in search_results:
        enhanced_result = result.copy()

        # Add related concepts
        path = result["path"]
        enhanced_result["related_concepts"] = bulk_tools._find_related_concepts(
            path, context_type
        )

        # Add usage examples if requested
        if include_examples:
            enhanced_result["examples"] = bulk_tools._get_usage_examples(path)

        enhanced_results.append(enhanced_result)

    # Format using templates
    return bulk_tools.formatter.format_search_results(
        enhanced_results,
        query=query,
        context_type=context_type
    )


@mcp.tool()
def get_development_context(
    path_or_concept: str,
    include_history: bool = True,
    response_format: Literal["full", "implementation", "reference"] = "full"
) -> str:
    """
    Get comprehensive development context for implementing or using IMAS data.

    Args:
        path_or_concept: Specific path or concept name
        include_history: Include lifecycle and version information
        response_format: Type of development context to provide

    Returns:
        Formatted development guidance with examples and best practices

    Example Usage:
        get_development_context(
            path_or_concept="core_profiles/profiles_1d/electrons/density",
            response_format="implementation"
        )
    """
    # Determine if input is a path or concept
    if "/" in path_or_concept:
        # It's a specific path
        context_data = bulk_tools._get_path_development_context(
            path_or_concept, include_history
        )
    else:
        # It's a concept - find related paths
        concept_results = bulk_tools.search_engine.search_concept(path_or_concept)
        context_data = bulk_tools._get_concept_development_context(
            path_or_concept, concept_results, include_history
        )

    return bulk_tools.formatter.format_development_context(
        context_data,
        format_type=response_format
    )
```

### 2.2 Template System Implementation

#### 2.2.1 Template Architecture

The template system provides consistent, rich formatting for MCP responses, making them more useful for LLMs while maintaining readability.

```python
# File: imas_mcp/templates/response_formatter.py
"""Template-based response formatting for MCP tools."""

from typing import Dict, Any, List, Literal
from jinja2 import Environment, DictLoader
from pathlib import Path


class ResponseFormatter:
    """Format MCP tool responses using Jinja2 templates."""

    def __init__(self):
        self.templates = self._load_templates()
        self.env = Environment(loader=DictLoader(self.templates))

    def _load_templates(self) -> Dict[str, str]:
        """Load all response templates."""        return {
            "ids_context_detailed": self._ids_context_detailed_template(),
            "ids_context_summary": self._ids_context_summary_template(),
            "search_results": self._search_results_template(),
            "development_context": self._development_context_template(),
            "code_examples": self._code_examples_template()
        }

    def format_ids_context(
        self,
        context_data: Dict[str, Any],
        format_type: Literal["detailed", "summary", "code_examples"] = "detailed"
    ) -> str:
        """Format IDS context using appropriate template."""
        template_name = f"ids_context_{format_type}"
        template = self.env.get_template(template_name)
        return template.render(**context_data)

    def format_search_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        context_type: str
    ) -> str:
        """Format search results using template."""
        template = self.env.get_template("search_results")
        return template.render(
            results=results,
            query=query,
            context_type=context_type,
            result_count=len(results)
        )

    def format_development_context(
        self,
        context_data: Dict[str, Any],
        format_type: Literal["full", "implementation", "reference"] = "full"
    ) -> str:
        """Format development context using template."""
        template = self.env.get_template("development_context")
        return template.render(context_data=context_data, format_type=format_type)

    # Template Definitions
    def _ids_context_detailed_template(self) -> str:
        """Detailed IDS context template."""
        return """
# {{ ids_data.ids_info.name | title }} IDS - Complete Context

## Overview
**Description**: {{ ids_data.ids_info.description }}
**Physics Domain**: {{ ids_data.ids_info.physics_domain | default("General") }}
**Total Fields**: {{ ids_data.paths | length }}
**Version**: {{ ids_data.ids_info.version }}

## Primary Coordinate Systems
{% for coord_name, coord_info in ids_data.coordinate_systems.items() %}
- **{{ coord_name }}**: {{ coord_info.description }}
  - Units: `{{ coord_info.units }}`
  - Usage: {{ coord_info.usage }}
  {% if coord_info.range -%}
  - Range: {{ coord_info.range[0] }} to {{ coord_info.range[1] }}
  {% endif %}
{% endfor %}

## Key Data Paths
### High-Priority Fields
{% for path, info in ids_data.paths.items() if info.physics_context.domain == "primary" %}
**`{{ path }}`**
- **Description**: {{ info.documentation }}
- **Units**: `{{ info.units }}`
- **Coordinates**: {{ info.coordinates | join(", ") }}
- **Physics Context**: {{ info.physics_context.phenomena | join(", ") }}
{% if info.physics_context.typical_values %}
- **Typical Values**:
  {% for context, value in info.physics_context.typical_values.items() -%}
  - {{ context }}: {{ value }}
  {% endfor %}
{% endif %}
{% if info.usage_examples %}
**Usage Example**:

    {{ info.usage_examples[0].code }}

_{{ info.usage_examples[0].notes }}_
{% endif %}
{% endfor %}

## Related IDS Connections
{% if related_data %}
{% for related_name, related_info in related_data.items() %}
### {{ related_name | title }}
- **Connection Type**: {{ relationships.get(related_name + "_connection", {}).get("type", "Coordinate sharing") }}
- **Key Shared Fields**:
  {% for connection in relationships.get(related_name + "_connections", [])[:3] %}
  - `{{ connection.this_path }}` ↔ `{{ connection.related_path }}`
  {% endfor %}
{% endfor %}
{% endif %}

## Semantic Field Groups
{% for group_name, paths in ids_data.semantic_groups.items() %}
### {{ group_name | replace("_", " ") | title }}
{% for path in paths[:5] %}
- `{{ path }}`
{% endfor %}
{% if paths | length > 5 %}
- _... and {{ paths | length - 5 }} more fields_
{% endif %}
{% endfor %}

## Implementation Notes
- **Data Access Pattern**: Time-sliced profiles with coordinate arrays
- **Validation**: All fields require units and coordinate consistency
- **Performance**: Use vectorized access for profile data
- **Common Pitfalls**: Ensure coordinate array alignment before calculations

---
_Generated for {{ ids_data.ids_info.name }} IDS - Comprehensive development context_
"""

    def _search_results_template(self) -> str:
        """Search results template."""
        return """
# Search Results: "{{ query }}"

**Context**: {{ context_type | title }} Physics
**Found**: {{ result_count }} relevant entries

{% for result in results %}

## {{ loop.index }}. `{{ result.path }}`

**Documentation**: {{ result.documentation }}
**Units**: `{{ result.units }}`
**IDS**: {{ result.ids_name }}

{% if result.physics_context %}
**Physics Context**: {{ result.physics_context.phenomena | join(", ") }}
{% endif %}

{% if result.related_concepts %}
**Related Concepts**:
{% for concept in result.related_concepts[:3] %}

- {{ concept }}
  {% endfor %}
  {% endif %}

{% if result.examples %}
**Code Example**:

    {{ result.examples[0].code }}

{% endif %}

---

{% endfor %}

## Summary

The search found {{ result_count }} entries related to "{{ query }}" in the {{ context_type }} domain.
These results show the interconnected nature of {{ context_type }} data in IMAS, with multiple IDS
providing complementary information for comprehensive analysis.
"""

    def _development_context_template(self) -> str:
        """Development context template."""
        return """
# Development Context{% if context_data.path %}: {{ context_data.path }}{% endif %}

## Implementation Guide

{% if format_type == "implementation" or format_type == "full" %}

### Data Access Pattern

{% if context_data.access_patterns %}
{% for pattern in context_data.access_patterns %}
**{{ pattern.name }}**:

    {{ pattern.code }}

_{{ pattern.description }}_

{% endfor %}
{% endif %}

### Validation Requirements

{% if context_data.validation_rules %}
{% for rule_type, rule_info in context_data.validation_rules.items() %}

- **{{ rule_type | replace("_", " ") | title }}**: {{ rule_info }}
  {% endfor %}
  {% endif %}
  {% endif %}

{% if format_type == "full" %}

### Best Practices

{% if context_data.best_practices %}
{% for practice in context_data.best_practices %}

- **{{ practice.category }}**: {{ practice.description }}
  {% if practice.example -%}

      {{ practice.example }}

  {% endif %}
  {% endfor %}
  {% endif %}

### Common Issues and Solutions

{% if context_data.common_issues %}
{% for issue in context_data.common_issues %}
**Problem**: {{ issue.problem }}
**Solution**: {{ issue.solution }}
{% if issue.code_fix %}

    {{ issue.code_fix }}

{% endif %}

{% endfor %}
{% endif %}
{% endif %}

---

_Development context generated for IMAS {{ context_data.version }}_
""" 

    def _ids_context_summary_template(self) -> str:
        """Summary IDS context template."""
        raise NotImplementedError("Summary template to be implemented in development phase")

    def _code_examples_template(self) -> str:
        """Code examples template."""
        raise NotImplementedError("Code examples template to be implemented in development phase")

```

#### 2.2.2 Template Usage Examples

Here's how the template system works in practice:

**Example 1: Complete IDS Context Request**

```python
# MCP Client Request:
{
    "method": "tools/call",
    "params": {
        "name": "get_ids_complete_context",
        "arguments": {
            "ids_name": "core_profiles",
            "include_related": true,
            "response_format": "detailed"
        }    }
}
```

Generated Response (truncated):

```markdown
# Core Profiles IDS - Complete Context

## Overview

**Description**: Plasma core profiles and transport data
**Physics Domain**: Transport
**Total Fields**: 847
**Version**: 4.0.0

## Primary Coordinate Systems

- **rho_tor_norm**: Normalized toroidal flux coordinate
  - Units: `-`
  - Usage: Primary radial coordinate for profiles
  - Range: 0.0 to 1.0
- **time**: Time coordinate
  - Units: `s`
  - Usage: Temporal evolution of profiles

## Key Data Paths

### High-Priority Fields

**`core_profiles/profiles_1d/electrons/density`**

- **Description**: Electron density profile
- **Units**: `m^-3`
- **Coordinates**: time, rho_tor_norm
- **Physics Context**: transport, confinement
- **Typical Values**:
  - core: 1e20 m^-3
  - edge: 1e19 m^-3

**Usage Example**:
`python
    density = ids.core_profiles.profiles_1d[time_idx].electrons.density
    `
_Access density at specific time slice_

## Related IDS Connections

### Equilibrium

- **Connection Type**: Coordinate sharing
- **Key Shared Fields**:
  - `core_profiles/profiles_1d/grid/psi` ↔ `equilibrium/time_slice/profiles_1d/psi`

## Implementation Notes

- **Data Access Pattern**: Time-sliced profiles with coordinate arrays
- **Validation**: All fields require units and coordinate consistency

---
```

**Example 2: Semantic Search with Examples**

````python
# MCP Client Request:
{
    "method": "tools/call",
    "params": {
        "name": "search_semantic_concepts",
        "arguments": {
            "query": "plasma confinement and pressure profiles",
            "context_type": "physics",
            "include_examples": true
        }
    }
}

# Generated Response:
"""
# Search Results: "plasma confinement and pressure profiles"

**Context**: Physics
**Found**: 8 relevant entries

## 1. `core_profiles/profiles_1d/electrons/pressure`

**Documentation**: Electron pressure profile
**Units**: `Pa`
**IDS**: core_profiles

**Physics Context**: transport, confinement, pressure_balance

**Related Concepts**:
- pressure_gradient_driven_transport
- confinement_scaling_laws
- pressure_balance_equilibrium

**Code Example**:
    ```python
    pe = ids.core_profiles.profiles_1d[0].electrons.pressure
    pressure_gradient = np.gradient(pe, rho_tor_norm)
    ```

---

## Summary
The search found 8 entries related to "plasma confinement and pressure profiles" in the physics domain.
These results show the interconnected nature of physics data in IMAS, with multiple IDS
providing complementary information for comprehensive analysis.
````

### 2.3 Template System Benefits

1. **Consistency**: All responses follow the same structured format
2. **Completeness**: Templates ensure all relevant information is included
3. **Readability**: Well-formatted responses that LLMs can easily parse
4. **Maintainability**: Easy to update response formats centrally
5. **Flexibility**: Different templates for different use cases

### 2.4 Integration with Existing FastMCP

The template system integrates seamlessly with your existing FastMCP server:

```python
# File: imas_mcp/server.py - Enhanced integration
"""Enhanced IMAS MCP server with template system."""

from mcp.server.fastmcp import FastMCP
from .tools.bulk_tools import BulkDataTools
from .search.hybrid_search import HybridSearchEngine

app = FastMCP("IMAS-Enhanced")

# Initialize components
search_engine = HybridSearchEngine()
bulk_tools = BulkDataTools(Path("exports"), search_engine)

# Register bulk tools
app.add_tools(bulk_tools.get_mcp_tools())

if __name__ == "__main__":
    app.run()
```

This enhanced system provides rich, contextual responses that significantly reduce the number of MCP requests needed while providing comprehensive information for effective LLM interaction.

---

## Phase 3: Semantic Search Integration (Weeks 5-6)

Brief overview of semantic capabilities to be detailed in implementation phase.

## Phase 4: Template System Enhancement (Weeks 7-8)

Advanced templating features and response optimization.

## Phase 5: Polish and Advanced Features (Weeks 9+)

Performance optimization, caching, and advanced discovery features.

---

## Success Metrics

1. **Reduced Request Count**: Target 70% reduction in MCP requests per task
2. **Response Completeness**: 90% of queries answered in single request
3. **Context Quality**: Rich documentation and examples in every response
4. **Performance**: Sub-200ms response time for bulk operations
5. **Developer Experience**: Clear, actionable responses with code examples

## Timeline Summary

- **Week 1-2**: JSON export pipeline and data model
- **Week 3-4**: Bulk MCP tools and template system
- **Week 5-6**: Semantic search integration
- **Week 7-8**: Advanced templates and optimization
- **Week 9+**: Polish and advanced features

This plan addresses the core issues of granular requests and insufficient context while building on your solid foundation and following modern best practices.

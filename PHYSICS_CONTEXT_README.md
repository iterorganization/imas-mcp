# Physics Context Module for IMAS MCP

This module provides a comprehensive physics context engine that maps physics concepts to IMAS attributes, enabling semantic understanding of plasma physics quantities within the IMAS data dictionary.

## Overview

The Physics Context Module addresses the core challenge of bridging the gap between physics concepts and IMAS data structure. Users can now query with physics terminology (e.g., "poloidal flux", "electron temperature") and get direct mappings to IMAS attributes, units, and symbols.

## Key Features

### 🎯 **Concept-to-IMAS Mapping**

- Direct mapping from physics concepts to IMAS paths
- Symbol and unit lookup (e.g., "poloidal flux" → "psi", "Wb")
- Alternative names and terminology support

### 🔍 **Enhanced Search**

- Physics-aware search with relevance scoring
- Concept suggestions and auto-completion
- Unit-based and domain-based queries

### 📚 **Comprehensive Explanations**

- Detailed physics concept explanations
- Measurement context and typical ranges
- Related quantities and cross-references

### 🏗️ **Domain Organization**

- Physics domains: equilibrium, profiles, transport, heating, etc.
- Domain-specific quantity organization
- Statistical overview and coverage analysis

## Core Components

### 1. Physics Context (`physics_context.py`)

**Core Classes:**

- `PhysicsQuantity`: Represents a physics quantity with IMAS attributes
- `PhysicsContext`: Container for all physics mappings
- `PhysicsContextEngine`: Query and search engine

**Key Functions:**

```python
# Basic concept mapping
concept_to_imas_paths("poloidal flux")  # → IMAS paths for psi
concept_to_units("electron temperature")  # → "eV"
concept_to_symbol("plasma current")      # → "Ip"

# Search and discovery
search_physics_concepts("temperature")   # → all temperature concepts
get_quantities_by_units("Pa")           # → all pressure quantities
get_quantities_by_domain(PhysicsDomain.EQUILIBRIUM)  # → equilibrium quantities
```

### 2. Physics Integration (`physics_integration.py`)

**Enhanced Capabilities:**

- `PhysicsEnhancedSearch`: Advanced search with physics context
- `PhysicsConceptExplainer`: Detailed concept explanations
- `IMASPhysicsIntegration`: Main integration class

**Key Functions:**

```python
# Enhanced search with physics context
physics_enhanced_search("electron temperature")

# Comprehensive concept explanation
explain_physics_concept("safety factor", detail_level="advanced")

# Complete concept-to-IMAS mapping
get_concept_imas_mapping("poloidal flux")
```

## Physics Quantities Covered

### **Magnetic Quantities**

| Concept                  | Symbol       | Units | Key IMAS Paths                                |
| ------------------------ | ------------ | ----- | --------------------------------------------- |
| Poloidal flux            | psi          | Wb    | `equilibrium/time_slice/profiles_1d/psi`      |
| Safety factor            | q            | 1     | `equilibrium/time_slice/profiles_1d/q`        |
| Normalized toroidal flux | rho_tor_norm | 1     | `core_profiles/profiles_1d/grid/rho_tor_norm` |

### **Plasma Parameters**

| Concept              | Symbol | Units | Key IMAS Paths                                    |
| -------------------- | ------ | ----- | ------------------------------------------------- |
| Electron temperature | Te     | eV    | `core_profiles/profiles_1d/electrons/temperature` |
| Ion temperature      | Ti     | eV    | `core_profiles/profiles_1d/ion/temperature`       |
| Electron density     | ne     | m^-3  | `core_profiles/profiles_1d/electrons/density`     |
| Ion density          | ni     | m^-3  | `core_profiles/profiles_1d/ion/density`           |
| Electron pressure    | pe     | Pa    | `core_profiles/profiles_1d/electrons/pressure`    |
| Ion pressure         | pi     | Pa    | `core_profiles/profiles_1d/ion/pressure`          |

### **Currents & Beta**

| Concept         | Symbol   | Units  | Key IMAS Paths                             |
| --------------- | -------- | ------ | ------------------------------------------ |
| Plasma current  | Ip       | A      | `summary/global_quantities/ip`             |
| Current density | j        | A.m^-2 | `equilibrium/time_slice/profiles_1d/j_tor` |
| Toroidal beta   | beta_tor | 1      | `summary/global_quantities/beta_tor`       |
| Poloidal beta   | beta_pol | 1      | `summary/global_quantities/beta_pol`       |
| Normalized beta | beta_N   | 1      | `summary/global_quantities/beta_tor_norm`  |

### **Geometry & Coordinates**

| Concept           | Symbol | Units | Key IMAS Paths                                             |
| ----------------- | ------ | ----- | ---------------------------------------------------------- |
| Major radius      | R      | m     | `equilibrium/time_slice/global_quantities/magnetic_axis/r` |
| Minor radius      | a      | m     | `equilibrium/time_slice/global_quantities/a_minor`         |
| Vertical position | Z      | m     | `equilibrium/time_slice/global_quantities/magnetic_axis/z` |
| Time              | t      | s     | `equilibrium/time`, `core_profiles/time`                   |

### **Heating & Transport**

| Concept                 | Symbol | Units | Key IMAS Paths                                     |
| ----------------------- | ------ | ----- | -------------------------------------------------- |
| NBI power               | P_NBI  | W     | `summary/heating_current_drive/nbi/power_launched` |
| EC power                | P_EC   | W     | `summary/heating_current_drive/ec/power_launched`  |
| IC power                | P_IC   | W     | `summary/heating_current_drive/ic/power_launched`  |
| Energy confinement time | tau_E  | s     | `summary/global_quantities/tau_energy`             |

## Usage Examples

### Example 1: Basic Concept Lookup

```python
from imas_mcp.physics_context import concept_to_imas_paths, concept_to_units, concept_to_symbol

# Find where poloidal flux is stored
paths = concept_to_imas_paths("poloidal flux")
# Result: ['equilibrium/time_slice/profiles_1d/psi', 'core_profiles/profiles_1d/grid/psi', ...]

units = concept_to_units("poloidal flux")  # Result: "Wb"
symbol = concept_to_symbol("poloidal flux")  # Result: "psi"
```

### Example 2: Enhanced Search

```python
from imas_mcp.physics_integration import physics_enhanced_search

# Search for temperature-related quantities
result = physics_enhanced_search("temperature")

for match in result['physics_matches']:
    print(f"{match['concept']}: {match['symbol']} [{match['units']}]")
    print(f"  IMAS paths: {match['imas_paths']}")
```

### Example 3: Concept Explanation

```python
from imas_mcp.physics_integration import explain_physics_concept

# Get detailed explanation of safety factor
explanation = explain_physics_concept("safety factor", "advanced")

print(f"Description: {explanation['quantity']['description']}")
print(f"Physics significance: {explanation['physics_context']['significance']}")
print(f"Mathematical definition: {explanation['physics_context']['mathematical_definition']}")
```

### Example 4: Domain Exploration

```python
from imas_mcp.physics_context import get_quantities_by_domain, PhysicsDomain

# Get all equilibrium quantities
eq_quantities = get_quantities_by_domain(PhysicsDomain.EQUILIBRIUM)

for q in eq_quantities:
    print(f"{q.concept}: {q.symbol} [{q.units}] - {len(q.imas_paths)} paths")
```

## MCP Server Integration

The physics context is integrated into the IMAS MCP server with new tools:

### New MCP Tools

1. **`search_physics_concepts`**

   - Enhanced physics-aware search
   - Returns concept matches with IMAS integration

2. **`explain_physics_concept`**

   - Comprehensive concept explanations
   - Multiple detail levels (basic, intermediate, advanced)

3. **`map_concept_to_imas`**

   - Direct concept-to-IMAS mapping
   - Usage examples and code snippets

4. **`get_physics_domain_overview`**

   - Domain-specific overviews
   - Coverage statistics and key quantities

5. **`validate_physics_query`**
   - Query validation and suggestions
   - Helps users formulate better queries

### Example MCP Usage

```python
# Using the MCP tools
await server.call_tool("search_physics_concepts", {"query": "electron temperature"})
await server.call_tool("explain_physics_concept", {"concept": "safety factor", "detail_level": "advanced"})
await server.call_tool("map_concept_to_imas", {"concept": "poloidal flux"})
```

## Benefits for IMAS Users

### 🎯 **Improved Discoverability**

- Users can search with physics terminology instead of IMAS path names
- Natural language queries: "temperature", "magnetic flux", "pressure"
- Automatic suggestion of related concepts

### 🔗 **Seamless Integration**

- Direct mapping from concepts to IMAS implementation
- Ready-to-use code examples for data access
- Cross-references between related quantities

### 📖 **Educational Value**

- Comprehensive physics explanations
- Measurement context and typical ranges
- Links physics theory to IMAS implementation

### 🚀 **Enhanced Productivity**

- Faster data discovery and access
- Reduced learning curve for new users
- Standardized physics terminology

## Testing and Validation

The module includes comprehensive tests (`test_physics_context.py`) covering:

- ✅ Basic concept mapping functionality
- ✅ Enhanced search capabilities
- ✅ Concept explanations with multiple detail levels
- ✅ Domain organization and statistics
- ✅ Real-world usage scenarios
- ✅ Query validation and suggestions

Run tests with:

```bash
python test_physics_context.py
```

## Extension and Customization

### Adding New Physics Quantities

To add new physics quantities, extend the `_build_comprehensive_context()` method in `PhysicsContextBuilder`:

```python
PhysicsQuantity(
    name="new_quantity",
    concept="new physics concept",
    description="Detailed description",
    units="appropriate_units",
    symbol="symbol",
    imas_paths=["path1", "path2"],
    alternative_names=["alt1", "alt2"],
    physics_domain=PhysicsDomain.APPROPRIATE_DOMAIN,
    coordinate_type="flux|spatial|time|global",
    typical_ranges={"tokamak": "range_info"}
)
```

### Extending Physics Domains

Add new domains by extending the `PhysicsDomain` enum and updating related methods.

## Future Enhancements

### Planned Features

- **Machine Learning Integration**: Automatic concept extraction from IMAS documentation
- **Multi-language Support**: Physics concepts in multiple languages
- **Interactive Visualization**: Graphical representation of concept relationships
- **Advanced Analytics**: Statistical analysis of IMAS usage patterns
- **Custom Ontologies**: User-defined physics concept hierarchies

### Integration Opportunities

- **OMAS Integration**: Bridge with OMAS for enhanced data access
- **ITER IMAS**: Specialized mappings for ITER-specific quantities
- **Experimental Databases**: Integration with experimental data repositories
- **Simulation Workflows**: Enhanced integration with physics simulation codes

## Conclusion

The Physics Context Module transforms the IMAS MCP experience by providing a semantic bridge between physics concepts and IMAS data structures. Users can now work with familiar physics terminology while seamlessly accessing the underlying IMAS implementation.

This enhancement significantly reduces the learning curve for new users, improves productivity for experienced users, and promotes standardized physics terminology across the fusion community.

For detailed implementation examples and advanced usage, see the test file and integration examples provided with the module.

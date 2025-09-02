# IMAS MCP Definitions

This directory contains data definitions and configurations used by the IMAS MCP server.

## Structure

### Physics Definitions (`physics/`)

- `domains/` - Physics domain definitions with characteristics, relationships, and IDS mappings
- `units/` - Unit contexts, categories, and domain hints
- `constants/` - Physics constants (future expansion)

### IMAS Definitions (`imas/`)

- `data_dictionary/` - IMAS data dictionary schema mappings and validation rules
- `workflows/` - Standard IMAS workflows
- `metadata/` - IMAS metadata definitions

### Validation (`validation/`)

- JSON schemas for validating YAML definitions
- Separate schemas for physics and IMAS definitions

### Templates (`templates/`)

- Template files for creating new definitions
- Documentation for definition formats

## Data Format

All definition files use YAML format for human readability and maintainability.
JSON schemas in the `validation/` directory ensure data integrity.

## Design Principles

1. **Physics-Based Categorization**: Domains are organized by physics phenomena rather than generic categories
2. **Option 1 Diagnostic Structure**: Implements physics-based diagnostic categorization (particle, electromagnetic, radiation, magnetic, mechanical) instead of a generic "diagnostics" category
3. **AI-Assisted Generation**: Definitions were generated with AI assistance but are maintained as static files for consistency
4. **Version Control**: All definitions are checked into the repository for change tracking
5. **Maintainability**: YAML format allows easy editing without code changes

## Domain Categories

### Core Plasma Physics (9 domains)

- `equilibrium` - MHD equilibrium and magnetic field configuration
- `transport` - Particle, energy, and momentum transport
- `mhd` - Magnetohydrodynamic instabilities and modes
- `turbulence` - Microscopic turbulence and transport
- `heating` - Auxiliary heating systems
- `current_drive` - Non-inductive current drive methods
- `wall` - Plasma-wall interactions
- `divertor` - Divertor physics and heat exhaust
- `edge_physics` - Edge and scrape-off layer physics

### Diagnostics (5 domains) - Option 1 Physics-Based

- `particle_diagnostics` - Particle measurement and analysis systems
- `electromagnetic_diagnostics` - Electromagnetic wave and field diagnostics
- `radiation_diagnostics` - Radiation-based diagnostic systems
- `magnetic_diagnostics` - Magnetic field measurement systems
- `mechanical_diagnostics` - Mechanical and pressure measurement systems

### Engineering & Control (4 domains)

- `control` - Plasma control and feedback systems
- `operational` - Operational parameters and machine status
- `coils` - Magnetic coil systems and field generation
- `structure` - Structural components and mechanical systems
- `systems` - Engineering systems and plant components

### Data & Workflow (2 domains)

- `data_management` - Data organization, metadata, and information management
- `workflow` - Computational workflows and process management

### General (1 domain)

- `general` - General purpose or uncategorized data structures

## Usage

The definitions are loaded automatically by the respective loader modules in `imas_mcp/core/`.

### Physics Domains

```python
from imas_mcp.core.domain_loader import load_physics_domains_from_yaml

# Load all definitions
definitions = load_physics_domains_from_yaml()

# Access individual components
characteristics = definitions["characteristics"]
ids_mapping = definitions["ids_mapping"]
relationships = definitions["relationships"]
validation = definitions["validation"]
```

### Unit Contexts

```python
from imas_mcp.core.unit_loader import load_unit_contexts, get_enhanced_unit_context

# Load unit contexts
unit_contexts = load_unit_contexts()

# Get enhanced context for a specific unit
enhanced_context = get_enhanced_unit_context("T")  # Returns: "magnetic_field_strength magnetic_flux_density electromagnetic magnetics current_drive heating"

# Unit contexts are automatically loaded by DocumentStore for semantic search
```

## Validation

The domain loader includes validation to ensure consistency between:

- Domain names across all files
- IDS mappings reference valid domains
- Relationships reference valid domains
- Complete coverage of all IDS

## Maintenance

When adding new IDS or modifying categorization:

1. Update the appropriate YAML file(s)
2. Run validation to check consistency
3. Test with the analysis script: `python scripts/analyze_domain_categorization.py`
4. Commit changes to version control

The definitions are designed to be generated once with AI assistance and then maintained manually for consistency and auditability.

# Physics Domain Definitions

This folder contains YAML-based definitions for the IMAS Data Dictionary physics domain categorization system. These definitions were generated with AI assistance and are designed to be version-controlled, maintainable, and reusable.

## Structure

### `physics_domains/`

Contains physics domain categorization definitions:

- **`domain_characteristics.yaml`** - Comprehensive characteristics for each physics domain including:
  - Description and primary phenomena
  - Typical units and measurement methods
  - Related domains and complexity levels
- **`ids_mapping.yaml`** - Mapping of IMAS IDS names to their primary physics domains
- **`domain_relationships.yaml`** - Relationships and connections between different physics domains

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

The definitions are loaded automatically by the `domain_loader.py` module in `imas_mcp/core/`. To use:

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

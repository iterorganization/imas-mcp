# Physics Extraction System

AI-assisted extraction of physics quantities from IMAS Data Dictionary JSON data with Pydantic validation, batch processing, and conflict resolution.

## Overview

This system provides:

- **AI-powered extraction**: Uses AI models to identify physics quantities from IMAS data
- **Pydantic validation**: Ensures data quality and type safety
- **Batch processing**: Handles large datasets efficiently with configurable batch sizes
- **Incremental updates**: Resumable processing with progress tracking
- **Conflict resolution**: Intelligent merging of quantities found in multiple IDS
- **Race condition protection**: File-based locking for safe concurrent access

## Architecture

```
physics_extraction/
├── __init__.py          # Package exports
├── __main__.py          # CLI entry point
├── models.py            # Pydantic data models
├── extractors.py        # AI extraction logic
├── storage.py           # Persistence and file management
└── coordination.py      # High-level orchestration
```

### Core Components

1. **Models** (`models.py`):

   - `PhysicsQuantity`: Individual physics quantity with metadata
   - `PhysicsDatabase`: Complete database of extracted quantities
   - `ExtractionProgress`: Progress tracking across sessions
   - `ConflictResolution`: Conflict detection and resolution
   - `ExtractionResult`: Results from processing individual IDS

2. **Extractors** (`extractors.py`):

   - `AIPhysicsExtractor`: AI-powered extraction from data paths
   - `BatchProcessor`: Manages processing of multiple IDS

3. **Storage** (`storage.py`):

   - `PhysicsStorage`: Database persistence with backups
   - `ProgressTracker`: Session progress tracking
   - `ConflictManager`: Conflict detection and resolution

4. **Coordination** (`coordination.py`):
   - `ExtractionCoordinator`: Main orchestration class
   - `LockManager`: Concurrent access protection

## Usage

### Command Line Interface

```bash
# Show current status
python -m imas_mcp.physics_extraction status

# Extract from 5 IDS with 20 paths each
python -m imas_mcp.physics_extraction extract --max-ids 5 --paths-per-ids 20

# Extract specific IDS
python -m imas_mcp.physics_extraction extract --ids equilibrium core_profiles

# List conflicts requiring review
python -m imas_mcp.physics_extraction conflicts

# Export database
python -m imas_mcp.physics_extraction export --output physics_quantities.json
```

### Programmatic Interface

```python
from pathlib import Path
from imas_mcp.physics_extraction import setup_extraction_environment, run_extraction_session

# Set up extraction environment
coordinator = setup_extraction_environment(
    json_data_dir=Path("imas_mcp/resources/json_data/detailed"),
    storage_dir=Path("storage/physics_extraction")
)

# Run extraction for a few IDS
session_id = run_extraction_session(
    coordinator=coordinator,
    max_ids=3,
    paths_per_ids=15
)

# Check results
status = coordinator.get_extraction_status()
print(f"Found {status['database_stats']['total_quantities']} physics quantities")
```

## Data Flow

1. **Input**: IMAS Data Dictionary JSON files from `resources/json_data/detailed/`
2. **Processing**:
   - Load IDS JSON data
   - Extract path information
   - Apply AI analysis to identify physics quantities
   - Validate with Pydantic models
   - Handle conflicts between IDS
3. **Output**:
   - Physics database in `storage/physics_extraction/physics_database.json`
   - Progress tracking in `storage/physics_extraction/extraction_progress.json`
   - Conflict logs in `storage/physics_extraction/conflicts.json`

## Configuration

### AI Model Configuration

The system uses configurable AI models for extraction:

```python
coordinator = ExtractionCoordinator(
    json_data_dir=json_data_dir,
    storage_dir=storage_dir,
    ai_model="gpt-4",  # AI model to use
    confidence_threshold=0.5  # Minimum confidence for extraction
)
```

### Batch Processing

Control processing granularity:

- `paths_per_ids`: Number of data paths to process per IDS (default: 10)
- `max_ids`: Maximum number of IDS to process in one session
- `confidence_threshold`: Minimum AI confidence score (0.0-1.0)

### Conflict Resolution

Automatic resolution strategies:

- `MERGE`: Combine information from multiple sources (default)
- `REPLACE`: Use newer version
- `SKIP`: Skip conflicting entries
- `MANUAL`: Require human review

## Progress Tracking

The system provides detailed progress tracking:

```python
status = coordinator.get_extraction_status()
print(f"Completion: {status['extraction_progress']['completion_percentage']:.1f}%")
print(f"Remaining IDS: {status['remaining_ids']}")
print(f"Conflicts: {status['conflicts']['total_unresolved']}")
```

## Storage and Persistence

### Database Format

The physics database is stored as JSON with the following structure:

```json
{
  "version": "1.0.0",
  "created_at": "2025-01-09T12:00:00Z",
  "last_updated": "2025-01-09T12:30:00Z",
  "quantities": {
    "quantity_id": {
      "name": "electron_temperature",
      "symbol": "Te",
      "description": "Electron temperature in the plasma",
      "unit": "eV",
      "imas_paths": ["core_profiles/profiles_1d/electrons/temperature"],
      "ids_sources": ["core_profiles"],
      "extraction_confidence": 0.95,
      "human_verified": false
    }
  },
  "quantities_by_name": { "electron_temperature": "quantity_id" },
  "quantities_by_ids": { "core_profiles": ["quantity_id"] },
  "total_quantities": 1,
  "verified_quantities": 0
}
```

### Backup and Recovery

- Automatic backups before database updates
- Up to 10 timestamped backups retained
- Automatic recovery from backup on corruption
- Atomic writes with temporary files

## Integration with Existing MCP Tools

The extraction system integrates with existing IMAS MCP tools:

```python
# Use with existing physics context
from imas_mcp.physics_context import get_physics_engine
from imas_mcp.physics_extraction import setup_extraction_environment

# Extract new quantities
coordinator = setup_extraction_environment()
run_extraction_session(coordinator, max_ids=5)

# Compare with existing context
engine = get_physics_engine()
existing_concepts = engine.get_all_concepts()
new_quantities = coordinator.database.quantities
```

## Error Handling and Logging

- Comprehensive logging at INFO/DEBUG levels
- Graceful error handling with session recovery
- Warning collection for data quality issues
- Failed IDS tracking with retry capability

## Development and Testing

### Adding New Extraction Logic

1. Extend `AIPhysicsExtractor._analyze_path_with_ai()` for custom extraction
2. Add new fields to `PhysicsQuantity` model
3. Update conflict detection in `ConflictManager`
4. Add validation rules in Pydantic models

### Testing

```bash
# Run extraction system tests
python -m pytest tests/test_physics_extraction/

# Test with sample data
python -m imas_mcp.physics_extraction extract --max-ids 1 --paths-per-ids 5
```

## Future Enhancements

- Integration with actual AI/LLM APIs
- Web interface for manual conflict resolution
- Export to different formats (YAML, XML, SQL)
- Real-time progress monitoring
- Distributed processing across multiple machines
- Integration with physics simulation tools

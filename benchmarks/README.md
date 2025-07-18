# IMAS MCP Performance Benchmarks

This directory contains the performance benchmarking setup for the IMAS MCP server using [ASV (airspeed velocity)](https://asv.readthedocs.io/).

## Key Design Principles

### Consistent Embedding Usage

- **All benchmarks use the same IDS pair**: `["core_profiles", "equilibrium"]`
- **No single-IDS benchmarks** to avoid loading additional embeddings
- **Comprehensive warmup** ensures embeddings are pre-loaded before timing
- **Embedding generation excluded from benchmark timing** through proper warmup

This ensures benchmarks measure tool performance, not embedding generation overhead.

## Setup

1. **Install benchmark dependencies:**

   ```bash
   make install-bench
   # or manually:
   uv sync --extra bench
   asv machine --yes
   ```

2. **Run performance baseline:**
   ```bash
   make performance-baseline
   ```

## Files

- **`asv.conf.json`** - ASV configuration with uv integration
- **`benchmarks.py`** - Main benchmark suite with all MCP tool benchmarks
- **`benchmark_runner.py`** - Utility class for running and managing ASV benchmarks
- **`performance_targets.py`** - Performance targets and validation functions
- **`__init__.py`** - Package initialization

## Usage

### Run all benchmarks

```bash
make performance-current
# or:
asv run --python=3.12
```

### Run specific benchmarks

```bash
asv run --python=3.12 -b SearchBenchmarks.time_search_imas_basic
```

### Compare performance

```bash
make performance-compare
# or:
asv compare HEAD~1 HEAD
```

### Generate HTML report

```bash
asv publish
# Results will be in .asv/html/
```

## Benchmark Suites

### SearchBenchmarks

- `time_search_imas_basic` - Basic search performance
- `time_search_imas_with_ai` - Search with AI enhancement
- `time_search_imas_complex_query` - Complex query performance
- `time_search_imas_ids_filter` - Search with IDS filtering
- `peakmem_search_imas_basic` - Memory usage for basic search

### ExplainConceptBenchmarks

- `time_explain_concept_basic` - Basic concept explanation
- `time_explain_concept_advanced` - Advanced concept explanation

### StructureAnalysisBenchmarks

- `time_analyze_ids_structure_small` - Small IDS structure analysis
- `time_analyze_ids_structure_large` - Large IDS structure analysis

### BulkExportBenchmarks

- `time_export_ids_bulk_single` - Single IDS export
- `time_export_ids_bulk_multiple` - Multiple IDS export
- `time_export_ids_bulk_with_relationships` - Export with relationships
- `time_export_physics_domain` - Physics domain export
- `peakmem_export_ids_bulk_large` - Memory usage for large export

### RelationshipBenchmarks

- `time_explore_relationships_depth_1` - Depth 1 relationship exploration
- `time_explore_relationships_depth_2` - Depth 2 relationship exploration
- `time_explore_relationships_depth_3` - Depth 3 relationship exploration

## Performance Targets

Current baseline targets are defined in `performance_targets.py`:

- **search_imas_basic**: <2.0s target, <5.0s max
- **search_imas_with_ai**: <3.0s target, <8.0s max
- **explain_concept_basic**: <1.5s target, <4.0s max
- **analyze_ids_structure**: <2.5s target, <6.0s max
- **export_ids_bulk_single**: <1.0s target, <3.0s max
- **export_ids_bulk_multiple**: <3.0s target, <8.0s max
- **explore_relationships**: <2.0s target, <5.0s max

## Integration with CI/CD

The benchmarks can be integrated into GitHub Actions or other CI systems:

```yaml
- name: Run performance benchmarks
  run: |
    uv sync --extra bench
    asv machine --yes
    asv run --python=3.12
    asv publish
```

## Tips

- Use `asv run --quick` for faster development iterations
- Use `asv run --bench <pattern>` to run specific benchmark patterns
- Use `asv show <commit>` to view results for a specific commit
- Use `asv find` to find performance regressions between commits

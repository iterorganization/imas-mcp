# Feature 05: SN Build Pipeline

**Repository:** imas-codex  
**Complexity:** High  
**Depends on:** 01 (Grammar API Exports), 04 (JSON Schema Contract)  
**Enables:** 06 (Cross-Model Review), 07 (Benchmarking), 08 (Publish)  
**Wave:** 3

---

## Overview

Multi-source standard name generation pipeline. Extracts candidate quantities from graph entities (DD paths, facility signals), composes grammatically valid standard names via LLM, reviews via cross-model validation, and publishes to the catalog.

## Module Structure

```
imas_codex/standard_names/
├── __init__.py
├── pipeline.py      # Orchestrator: run_sn_build()
├── workers.py       # WorkerSpec definitions for each phase
├── state.py         # SNBuildState(DiscoveryStateBase)
├── graph_ops.py     # StandardName node CRUD, claim/release
├── progress.py      # DataDrivenProgressDisplay subclass
└── sources/
    ├── __init__.py
    ├── base.py       # Abstract source protocol
    ├── dd.py         # Extract from IMASNode paths
    └── signals.py    # Extract from FacilitySignal nodes
```

## Pipeline Phases

### EXTRACT
- Source-specific: walks graph entities matching filters
- DD source: query IMASNode paths by IDS, domain, lifecycle
  - Group by semantic cluster for batch composition
  - Gather parent node context (structure, documentation)
  - Use `search_dd_paths` / `list_dd_paths` backing functions
- Signals source: query FacilitySignal nodes by facility, domain
  - Group by physics domain
  - Gather signal descriptions, units, MEASURES relationships
- Output: extraction batches with context for LLM composition

### COMPOSE
- LLM generates candidate standard names from extraction batches
- Jinja2 templates in `llm/prompts/sn/` inject:
  - Grammar rules (from imas-standard-names export)
  - Source context (DD path docs, signal descriptions)
  - Existing catalog names (dedup awareness)
  - Cluster context (related paths/signals)
- Grammar validation: compose → parse round-trip
- Structured output via `acall_llm_structured()` with Pydantic models

### REVIEW (Feature 06)
- Cross-model review using different LLM family
- Scoring: grammar correctness, semantic accuracy, description quality

### VALIDATE
- Import `validate_models()` from imas-standard-names
- Full semantic + structural + quality checks
- JSON schema contract validation

### PUBLISH (Feature 08)
- Convert validated candidates to catalog YAML
- Batched PRs to imas-standard-names-catalog

## State Management

```python
@dataclass
class SNBuildState(DiscoveryStateBase):
    source: str              # "dd" | "signals"
    ids_filter: str | None   # For DD source
    domain_filter: str | None
    facility_filter: str | None  # For signals source
    # Inherited: cost_limit, deadline, service_monitor
```

## CLI Interface

```
imas-codex sn build --source dd [--ids NAME] [--domain NAME] [--cost-limit N]
imas-codex sn build --source signals [--facility NAME] [--cost-limit N]
imas-codex sn status
```

## Progress Display

Uses `DataDrivenProgressDisplay` with 5 `StageDisplaySpec` entries:
- Extract, Compose, Review, Validate, Publish
- Shows per-phase bars, rates, ETAs
- Resource gauges for LLM cost and token budget
- Worker status (idle/running/crashed) per phase

## Schema Changes (Graph)

Extend `StandardName` node in `schemas/facility.yaml`:
- Add `source` attribute (dd | signals | manual)
- Add `DERIVED_FROM` relationship to `IMASNode` (for DD-sourced names)
- `FacilitySignal -[:MEASURES]-> StandardName` already exists

## Deliverables

- [ ] `imas_codex/standard_names/` module with all pipeline components
- [ ] `imas_codex/cli/sn.py` — top-level CLI group registered in `cli/__init__.py`
- [ ] `llm/prompts/sn/` — Jinja2 templates for compose and review
- [ ] `sn/progress.py` — rich progress display
- [ ] Graph schema updates in `schemas/facility.yaml`
- [ ] Tests with >95% coverage
- [ ] DD source working end-to-end for at least one IDS

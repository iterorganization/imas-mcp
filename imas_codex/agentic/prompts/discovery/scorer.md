---
name: discovery/scorer
description: Directory scoring and enrichment for graph-led discovery
used_by: imas_codex.discovery.scorer.DirectoryScorer
task: score
---

You are analyzing directories at a fusion research facility to enrich the knowledge graph with structured metadata. Your output directly populates graph node properties.

## Task

For each directory, collect evidence about its contents and purpose, then provide a structured assessment that will enrich the facility path nodes in our knowledge graph.

## Output Schema

Each directory response MUST include these fields:

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `path` | string | The directory path (echo from input) |
| `path_purpose` | enum | Classification (see values below) |
| `description` | string | Concise description of directory contents (1-2 sentences) |
| `score_code` | float | Code discovery value (0.0-1.0) |
| `score_data` | float | Data discovery value (0.0-1.0) |
| `score_imas` | float | IMAS relevance (0.0-1.0) |
| `should_expand` | bool | Whether to explore children |
| `evidence` | object | Structured evidence (see below) |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `keywords` | string[] | Relevant keywords for search (max 5) |
| `physics_domain` | string | Primary physics domain if applicable |
| `expansion_reason` | string | Why to expand (if should_expand=true) |
| `skip_reason` | string | Why to skip (if should_expand=false) |

### path_purpose Values

- `physics_code`: Simulation or analysis code (equilibrium, transport, MHD, heating, etc.)
- `data_files`: Scientific data storage (HDF5, NetCDF, MDSplus trees)
- `documentation`: Docs, wikis, READMEs, manuals
- `configuration`: Config files, settings, environment scripts
- `build_artifacts`: Compiled outputs, caches, __pycache__, .o files
- `test_files`: Test suites, pytest directories, test data
- `user_home`: Personal directories (usually low value)
- `system`: OS or infrastructure (/usr, /lib, /etc)
- `unknown`: Cannot determine from available evidence

### Evidence Object

```json
{
  "code_indicators": ["py", "f90", "cpp"],
  "data_indicators": ["nc", "h5", "mat"],
  "imas_indicators": ["put_slice", "ids_properties"],
  "physics_indicators": ["equilibrium", "transport"],
  "quality_indicators": ["has_readme", "has_git"]
}
```

## Scoring Guidelines

### score_code (0.0-1.0)
- **0.9-1.0**: Core physics simulation code, IMAS actors, actively maintained
- **0.7-0.8**: Analysis tools, data processing scripts, utilities
- **0.4-0.6**: Mixed content, some code present
- **0.1-0.3**: Primarily documentation or configuration
- **0.0**: No code content

### score_data (0.0-1.0)
- **0.9-1.0**: Scientific data archives, shot databases
- **0.7-0.8**: Data directories with structured files
- **0.4-0.6**: Mixed content, some data present
- **0.1-0.3**: Configuration data, templates
- **0.0**: No data content

### score_imas (0.0-1.0)
- **0.9-1.0**: Direct IMAS integration (put_slice, get_slice, IDS names)
- **0.7-0.8**: Uses physics quantities that map to IMAS
- **0.4-0.6**: Related to fusion physics but no direct IMAS use
- **0.1-0.3**: Potentially relevant but no clear connection
- **0.0**: No IMAS relevance

## Expansion Decision

**Expand** (should_expand=true) when:
- Combined score >= 0.7 AND directory likely contains valuable children
- Contains subdirectories with promising names
- Is a project root (has .git, Makefile, setup.py)

**Don't expand** (should_expand=false) when:
- path_purpose is: user_home, system, build_artifacts
- Low scores across all dimensions (< 0.3)
- Leaf directory with only files

{% if focus %}
## Focus Area

Prioritize paths related to: **{{ focus }}**

Boost scores for directories matching this focus. Add focus-related keywords.
{% endif %}

## Response Format

Return a valid JSON array with one object per directory, maintaining input order:

```json
[
  {
    "path": "/home/codes/liuqe",
    "path_purpose": "physics_code",
    "description": "LIUQE equilibrium reconstruction code with Fortran core and Python bindings",
    "keywords": ["equilibrium", "reconstruction", "liuqe", "tokamak"],
    "physics_domain": "equilibrium",
    "evidence": {
      "code_indicators": ["f90", "py", "c"],
      "data_indicators": [],
      "imas_indicators": ["put_slice", "equilibrium IDS"],
      "physics_indicators": ["equilibrium", "flux surfaces"],
      "quality_indicators": ["has_readme", "has_makefile", "has_git"]
    },
    "score_code": 0.95,
    "score_data": 0.15,
    "score_imas": 0.85,
    "should_expand": true,
    "expansion_reason": "High-value equilibrium code with active IMAS integration"
  }
]
```

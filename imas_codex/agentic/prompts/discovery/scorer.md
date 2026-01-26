---
name: discovery/scorer
description: Directory scoring and enrichment for graph-led discovery
used_by: imas_codex.discovery.scorer.DirectoryScorer
task: score
---

You are analyzing directories at a fusion research facility to enrich the knowledge graph with structured metadata. Your output directly populates graph node properties.

## Task

For each directory, analyze the path and available metadata to provide structured scores:

1. **Classify** the directory purpose from the path name and contents
2. **Score** four dimensions: code value, data value, docs value, IMAS relevance (0.0-1.0)
3. **Decide** whether to expand into child directories
4. **Extract** keywords and physics domain if applicable

## path_purpose Values (CRITICAL - use exactly these values)

### Code Categories (score = ingestion priority)
- `modeling_code`: Physics simulation/modeling code (CHEASE, ASTRA, JOREK, JINTRAC, equilibrium solvers)
- `diagnostic_code`: Diagnostic analysis pipelines (Thomson, bolometry, interferometry, SXR)
- `data_interface`: Data access/conversion tools (IMAS wrappers, MDSplus readers, EQDSK tools)
- `workflow`: Orchestration, batch processing, shot review scripts
- `visualization`: Plotting and rendering tools

### Data Categories
- `simulation_data`: Outputs from modeling codes (HDF5 runs, NetCDF outputs)
- `diagnostic_data`: Experimental measurements (raw/calibrated diagnostic data)

### Support Categories
- `documentation`: Docs, papers, tutorials, READMEs, teaching materials
- `configuration`: Config files, settings, module files
- `test_suite`: Unit/integration tests (pytest directories, test fixtures)

### Structural Category (score = exploration potential)
- `container`: Organizational directory with varied content. Examples: `/home`, `/work`, `/work/imas`, `/work/projects`
  - **Score meaning for container**: How likely are children to contain valuable content?
  - High score (0.7-1.0): Explore children (e.g., `/work/imas` likely has valuable subdirs)
  - Low score (0.0-0.3): Skip subtree (e.g., `/home/user/Downloads`)

### Skip Categories (always low score, skip subtree)
- `archive`: Old/backup content (backup/, old_projects/, deprecated/, 2019/, archive/)
- `build_artifact`: Generated/cached files (__pycache__, .venv, node_modules, .o, .pyc)
- `system`: OS/infrastructure directories (/var, /tmp, /opt/modules, /usr, /lib)

## Path-Based Scoring Heuristics

**Use path names to infer purpose and score:**

### High exploration potential containers (container + high score):
- `/work/imas`, `/imas`, `*imas*` → container, score_imas ≥ 0.8, expand=true
- `/work/projects/*`, `/work/codes/*` → container, score_code ≥ 0.7, expand=true
- `/home/codes/*` → container, score_code ≥ 0.6, expand=true

### Modeling code indicators:
- `*equilibrium*`, `*efit*`, `*chease*`, `*helena*` → modeling_code, equilibrium domain
- `*transport*`, `*astra*`, `*jetto*`, `*jintrac*` → modeling_code, transport domain
- `*mhd*`, `*jorek*`, `*nimrod*` → modeling_code, MHD domain
- `*stability*` → modeling_code, stability domain

### Diagnostic code indicators:
- `*thomson*`, `*ece*`, `*interferom*` → diagnostic_code
- `*bolom*`, `*sxr*`, `*soft_xray*` → diagnostic_code
- `*diagnostic*` (in code context) → diagnostic_code

### Data interface indicators:
- `*mdsplus*`, `*mds_*`, `*tdi*` → data_interface
- `*imas_*`, `*ids_*`, `*eqdsk*` → data_interface
- `*reader*`, `*writer*`, `*interface*` → data_interface

### Skip patterns (classify as archive/build_artifact/system):
- `backup`, `old_*`, `deprecated`, `archive`, `2019`, `2020` → archive
- `__pycache__`, `.venv`, `venv`, `node_modules`, `build`, `dist` → build_artifact
- `/var/*`, `/tmp/*`, `/opt/modules/*`, `/usr/*`, `/lib/*` → system

## Scoring Guidelines

### For code/data categories (score = ingestion priority):

**score_code (0.0-1.0)**
- **0.9-1.0**: Core physics simulation code, IMAS actors, actively maintained
- **0.7-0.8**: Analysis tools, data processing scripts, utilities
- **0.4-0.6**: Mixed content, some code present
- **0.1-0.3**: Primarily documentation or configuration
- **0.0**: No code content

**score_data (0.0-1.0)**
- **0.9-1.0**: Scientific data archives, shot databases
- **0.7-0.8**: Data directories with structured files
- **0.4-0.6**: Mixed content, some data present
- **0.0**: No data content

**score_docs (0.0-1.0)**
- **0.9-1.0**: Dedicated documentation directories (docs/, tutorials/, papers/)
- **0.7-0.8**: Contains README, guides, or scientific papers
- **0.4-0.6**: Some documentation present alongside code
- **0.1-0.3**: Minimal docs (just comments in code)
- **0.0**: No documentation content

**score_imas (0.0-1.0)**
- **0.9-1.0**: Direct IMAS integration (put_slice, get_slice, IDS names)
- **0.7-0.8**: Path contains "imas" or known physics code names
- **0.4-0.6**: Fusion physics but no direct IMAS use
- **0.0**: No IMAS relevance

### For container category (score = exploration potential):

**Score based on how valuable children are likely to be:**
- **0.9-1.0**: `/work/imas`, `/imas` - almost certainly valuable children
- **0.7-0.8**: `/work/projects`, `/home/codes` - likely valuable
- **0.4-0.6**: Generic `/work/*`, research directories
- **0.1-0.3**: User home with no code indicators
- **0.0**: Downloads, temp directories

## Expansion Decision

**Expand** (should_expand=true) when:
- Purpose is `container` AND combined score >= 0.4
- Purpose is code/data category AND combined score >= 0.5 AND has subdirectories
- Is a project root (has .git, Makefile, setup.py)

**Don't expand** (should_expand=false) when:
- Purpose is: `system`, `build_artifact`, `archive`
- Combined score < 0.3 for any purpose
- Leaf directory with only files

## Evidence Collection

For each directory, collect evidence in these categories:
- **code_indicators**: Programming file extensions (py, f90, cpp, c, jl)
- **data_indicators**: Data file extensions (nc, h5, mat, csv, json)
- **doc_indicators**: Documentation signals (README, docs/, pdf, tutorial, paper, guide)
- **imas_indicators**: IMAS-specific patterns (put_slice, get_slice, IDS names, "imas" in path)
- **physics_indicators**: Physics domains (equilibrium, transport, MHD)
- **quality_indicators**: Project quality signals (has_readme, has_makefile, has_git)

{% if focus %}
## Focus Area

Prioritize paths related to: **{{ focus }}**

For paths matching this focus:
- Boost all scores by 0.2
- Add focus-related keywords
- Set should_expand=true if any dimension >= 0.4
{% endif %}

{% if is_rescore %}
## Second-Pass Rescoring (Enriched Paths)

This is a RESCORE pass for paths with enrichment data. You have additional context from:
- **Pattern matches**: Code pattern search results
- **Format conversion**: Whether path contains multi-format conversion code
- **Lines of code**: Total LOC and language breakdown
- **Storage size**: Directory size in bytes

### Rescoring Guidelines

**Multi-format detection** (score_multiformat):
- If path has both READ and WRITE format patterns, set score_multiformat = 1.0
- This indicates data conversion/mapping utilities - HIGH VALUE
- Examples: load EQDSK + write IMAS, read MDSplus + save HDF5

**Adjust scores based on enrichment**:
- LOC > 10,000 lines in physics language (Fortran, Python) → boost score_code by 0.1
- Total bytes > 1GB → boost score_data by 0.1
- Has multiple format reads/writes → this is a data interface, boost score_imas by 0.2

**Provided enrichment data**:
{{ enrichment_data }}
{% endif %}

## Score Precision (CRITICAL)

- Use exactly 2 decimal places (e.g., 0.85, 0.72, 0.31)
- **Maximum allowed score is 0.95** - NEVER use 1.0
- Scores of 1.0 are FORBIDDEN and will cause batch rejection
- Minimum non-zero score is 0.05
- Reserve scores above 0.90 for truly exceptional directories only

## Score Distribution Guidelines

Aim for a natural distribution across facilities:
- **0.00-0.25 (Low)**: ~25% of paths - build artifacts, archives, system dirs
- **0.25-0.50 (Medium)**: ~35% of paths - generic containers, mixed content
- **0.50-0.75 (High)**: ~30% of paths - valuable code, data, documentation
- **0.75-0.95 (Very High)**: ~10% of paths - core physics codes, IMAS integration

{% if example_paths %}
## Calibration Examples from This Facility

Use these previously scored paths to calibrate your decisions:

**Low (0.0-0.25):**
{% for p in example_paths.low %}
- `{{ p.path }}` → {{ p.score }} ({{ p.purpose }})
{% endfor %}

**Medium (0.25-0.5):**
{% for p in example_paths.medium %}
- `{{ p.path }}` → {{ p.score }} ({{ p.purpose }})
{% endfor %}

**High (0.5-0.75):**
{% for p in example_paths.high %}
- `{{ p.path }}` → {{ p.score }} ({{ p.purpose }})
{% endfor %}

**Very High (0.75-0.95):**
{% for p in example_paths.very_high %}
- `{{ p.path }}` → {{ p.score }} ({{ p.purpose }})
{% endfor %}
{% endif %}

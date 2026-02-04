---
name: discovery/scorer
description: Directory scoring and enrichment for graph-led discovery
used_by: imas_codex.discovery.scorer.DirectoryScorer
task: score
dynamic: true
---

You are analyzing directories at a fusion research facility to enrich the knowledge graph with structured metadata. Your output directly populates graph node properties.

## Task

For each directory, analyze the path and available metadata to provide structured scores:

1. **Classify** the directory purpose from the path name and contents
2. **Score** per-purpose dimensions (0.0-1.0) for each content type
3. **Decide** whether to expand into child directories
4. **Extract** keywords and physics domain if applicable

{% include "schema/path-purposes.md" %}

## Path-Based Scoring Heuristics

**Use the FULL PATH to infer context.**

The full path reveals the directory's place in the filesystem hierarchy. Consider where in the tree this directory sits and what the path components suggest about its purpose and current relevance.

Contents are shown sorted by modification time (most recent first), so you can infer activity level from the ordering.

### High exploration potential containers (container + high score):
- `/home`, `/home/*`, `/home/ITER/*`, `/solhome` → container, score 0.05-0.1, **expand=true ALWAYS**
  - User home directories contain researcher code
  - No matter how many subdirectories, ALWAYS expand
- `/work/imas`, `/imas`, `*imas*` → container, score_imas ≥ 0.8, expand=true
- `/work/projects/*`, `/work/codes/*` → container, score_modeling_code ≥ 0.7, expand=true
- `/home/codes/*` → container, score_modeling_code ≥ 0.6, expand=true

### Modeling code indicators:
- `*equilibrium*`, `*efit*`, `*chease*`, `*helena*` → modeling_code, equilibrium domain
- `*transport*`, `*astra*` → modeling_code, transport domain
- `*mhd*`, `*jorek*`, `*nimrod*` → modeling_code, MHD domain
- `*stability*` → modeling_code, stability domain

### Simulation run data (CRITICAL - NOT code):
Simulation workflow tools like JETTO, JINTRAC, ETS produce run directories containing OUTPUT DATA, not code.
- `*/jetto/runs`, `*/jintrac/runs`, `*/ets/runs` → modeling_data, expand=false
- `*/jetto/runs/*`, `*/jintrac/runs/*` → modeling_data, expand=false (individual runs)
- `*/run[0-9]*` (under simulation tool dirs) → modeling_data (numbered runs)
- `*_scan*`, `*_sweep*`, `*_baseline*` (in run context) → modeling_data (parameter scans)

**IMPORTANT**: These directories often contain helper scripts (plot.sh, analyze.m, startimes.py) for post-processing the simulation output. The presence of scripts does NOT make these code directories - they're data directories with convenience scripts. Score them HIGH on score_modeling_data but set path_purpose=modeling_data and should_expand=false.

### Simulation code vs simulation runs:
- `*/jetto` (no /runs suffix) → container, may expand if has code subdirs
- `*/jetto/src`, `*/jetto/source` → modeling_code (actual source)
- `*/jetto/runs`, `*/jintrac/runs` → modeling_data, NEVER expand (run outputs)

### Analysis code indicators:
- `*thomson*`, `*ece*`, `*interferom*` → analysis_code, diagnostic processing
- `*bolom*`, `*sxr*`, `*soft_xray*` → analysis_code, diagnostic processing
- `*diagnostic*` (in code context) → analysis_code
- `*liuqe*`, `*efit*` (processing context) → analysis_code, equilibrium reconstruction
- `*intershot*`, `*shot_review*`, `*campaign*` → analysis_code
- `*pulse*`, `*shot_*` (in code context) → analysis_code

### Operations code indicators:
- `*realtime*`, `*real-time*`, `*rt_*`, `*_rt*` → operations_code
- `*controller*`, `*feedback*`, `*actuator*` → operations_code
- `*daq*`, `*acquisition*`, `*pxi*`, `*fpga*` → operations_code

### Data access indicators (NATIVE facility data - NOT IMAS):
- `*mdsplus*`, `*mds_*`, `*tdi*` → data_access (score_data_access HIGH)
- `*eqdsk*`, `*gfile*`, `*aeqdsk*` → data_access (format-specific)
- `*reader*`, `*writer*`, `*interface*` → data_access (general I/O)
- `*shotfile*`, `*pulse*`, `*shot_*` → data_access (shot data handling)

### IMAS-specific indicators (score_imas, NOT score_data_access):
- `*imas_*`, `*ids_*`, `*imas2*` → score_imas HIGH, path_purpose varies
- `*put_slice*`, `*get_slice*` → score_imas HIGH (API usage)

### Skip patterns (classify as archive/build_artifact/system):
- `__pycache__`, `.venv`, `venv`, `node_modules`, `build`, `dist` → build_artifact
- `/var/*`, `/tmp/*`, `/opt/modules/*`, `/usr/*`, `/lib/*` → system
- Use your judgment for archive classification based on path naming and context

## Scoring Guidelines

Each directory is scored on 10 per-purpose dimensions (0.0-1.0 each), aligned with the DiscoveryRootCategory taxonomy. Score ONLY the dimensions relevant to the directory's content.

### Code Dimensions

**score_modeling_code (0.0-1.0)** - Forward modeling/simulation code
- **0.9-0.95**: Core physics simulation (CHEASE, ASTRA, JOREK, MHD solvers)
- **0.7-0.8**: Secondary modeling tools, scenario builders
- **0.4-0.6**: Mixed code with some modeling components
- **0.0**: No modeling code

**score_analysis_code (0.0-1.0)** - Experimental analysis code
- **0.9-0.95**: Diagnostic processing (Thomson, ECE, interferometry)
- **0.7-0.8**: Equilibrium reconstruction (LIUQE, EFIT processing)
- **0.4-0.6**: Intershot tools, campaign analysis scripts
- **0.0**: No analysis code

**score_operations_code (0.0-1.0)** - Real-time operations code
- **0.9-0.95**: Control systems, feedback algorithms (RAPTOR-RT)
- **0.7-0.8**: DAQ, FPGA/PXI code, hardware interfaces
- **0.4-0.6**: Timing code, actuator wrappers
- **0.0**: No operations code

### Data Dimensions

**score_modeling_data (0.0-1.0)** - Modeling outputs
- **0.9-0.95**: Large parameter scan databases, scenario libraries
- **0.7-0.8**: Simulation output directories (HDF5, NetCDF)
- **0.4-0.6**: Mixed data with some modeling outputs
- **0.0**: No modeling data

**score_experimental_data (0.0-1.0)** - Experimental shot data
- **0.9-0.95**: MDSplus shot trees, pulse file archives
- **0.7-0.8**: Diagnostic data stores, raw measurements
- **0.4-0.6**: Mixed data with some experimental content
- **0.0**: No experimental data

### Infrastructure Dimensions

**score_data_access (0.0-1.0)** - Native facility data access (MDSplus, TDI, shotfiles)
- **0.9-0.95**: Core MDSplus/TDI libraries, native data readers for the facility
- **0.7-0.8**: EQDSK/gfile converters, format-specific loaders
- **0.4-0.6**: Helper utilities for data I/O, generic readers
- **0.0**: No data access code
- **NOTE**: IMAS wrappers score on score_imas, NOT here

**score_workflow (0.0-1.0)** - Orchestration tools
- **0.9-0.95**: Shot review pipelines, batch processing frameworks
- **0.7-0.8**: Automation scripts, job schedulers
- **0.4-0.6**: Utility scripts with workflow aspects
- **0.0**: No workflow code

**score_visualization (0.0-1.0)** - Plotting tools
- **0.9-0.95**: Dedicated visualization packages, dashboards
- **0.7-0.8**: Plotting utilities, rendering tools
- **0.4-0.6**: Mixed code with some plotting
- **0.0**: No visualization code

### Support Dimensions

**score_documentation (0.0-1.0)** - Documentation value
- **0.9-0.95**: Comprehensive docs directories, tutorials, papers
- **0.7-0.8**: Good READMEs, guides, API docs
- **0.4-0.6**: Some documentation alongside code
- **0.1-0.3**: Minimal docs (just comments)
- **0.0**: No documentation

### Cross-Cutting Dimension

**score_imas (0.0-1.0)** - IMAS relevance (orthogonal to other dimensions)
- **0.9-0.95**: Direct IMAS integration (put_slice, get_slice, IDS names)
- **0.7-0.8**: Path contains "imas" or known IMAS-related code
- **0.4-0.6**: Fusion physics content that could map to IMAS
- **0.0**: No IMAS relevance

**IMPORTANT**: score_imas is INDEPENDENT of score_data_access. A directory with IMAS wrappers should have:
- HIGH score_imas (0.8+) because it uses IMAS
- LOW score_data_access (0.0-0.3) unless it ALSO handles native formats (MDSplus, TDI)

This prevents IMAS infrastructure from dominating over native facility data access code, which is our primary discovery target.

### Purpose-to-Score Mapping

When classifying `path_purpose`, set the corresponding dimension HIGH:

| path_purpose | Primary dimension to set high |
|--------------|-------------------------------|
| modeling_code | score_modeling_code |
| analysis_code | score_analysis_code |
| operations_code | score_operations_code |
| modeling_data | score_modeling_data |
| experimental_data | score_experimental_data |
| data_access | score_data_access |
| workflow | score_workflow |
| visualization | score_visualization |
| documentation | score_documentation |
| container | Use max of expected child dimensions |
| test_suite, configuration | Low across all (< 0.3) |
| archive, build_artifact, system | Zero or near-zero across all |

## Expansion Decision

**NEVER expand** (should_expand=false, CRITICAL):
- **Git repositories** (has .git): The code can be fetched from the remote instead
  - Even if highly scored, do NOT expand
  - We want to know the repo exists at this location, but we don't need to scan every file
- **Data containers** (modeling_data, experimental_data): Too many files
  - Simulation output directories (HDF5, NetCDF files) should NOT be expanded
  - Experimental shot archives and MDSplus trees should NOT be expanded
  - We only need to know high-value data exists here
- **High subdirectory count containers (>100 subdirs) of DATA**: These are almost always data containers
  - If you see `Dirs: 500`, `Dirs: 1000+ ` with similarly-named children (run*, scan*, shot*), this is data
  - Even with helper scripts mixed in, classify as modeling_data, expand=false
  - The scripts are for post-processing the data, not standalone code
  - **EXCEPTION: User home directories** (see below)
- **Numeric directory warning (⚠️ DATA CONTAINER)**: Input may include a warning like:
  `⚠️ DATA CONTAINER: 85% of subdirs are numeric (shot IDs/runs). Set should_expand=false.`
  - This is calculated automatically from directory names
  - Trust this signal - numeric subdirectories almost always contain shot/run data
  - Set should_expand=false and purpose=modeling_data or experimental_data
- **Purpose is: `system`, `build_artifact`, `archive`**
- **Combined score < 0.3 for any purpose**
- **Leaf directory with only files (no subdirectories)**

**ALWAYS expand** (should_expand=true, CRITICAL):
- **User home directories**: `/home`, `/home/*`, `/home/ITER/*`, `/solhome`
  - These contain researcher code and MUST be expanded regardless of subdirectory count
  - Children are username directories (e.g., `/home/jsmith`) containing personal code
  - Score as `container`, set score low (0.05-0.1), but ALWAYS expand=true
  - The subdirectories are NOT data files, they are user directories with potential code
- **Shared home directories**: `/work`, `/work/*`
  - These organize project/user work and should be expanded to find code
  - Score as `container`, expand=true

**Expand** (should_expand=true) when:
- **User home directories** (see ALWAYS expand above) - `/home`, `/home/*`, `/solhome`
- Purpose is `container` AND combined score >= 0.4 AND NOT a git repo
- Purpose is code category (modeling_code, analysis_code, operations_code, data_access, visualization, workflow) AND NO .git folder
- Purpose is documentation AND has subdirectories to explore

**Git repo handling**:
- If has_git=true: Score the directory based on its content quality
- Set should_expand=false regardless of score (code is available via git clone)
- High scores (0.7+) are still valuable - they indicate important code repos

**Data container handling**:
- Score data directories based on their scientific value
- Set should_expand=false for modeling_data and experimental_data
- We want to catalog where data exists, not enumerate every file

## Evidence Collection

For each directory, collect evidence in these categories:
- **code_indicators**: Programming file extensions (py, f90, cpp, c, jl)
- **data_indicators**: Data file extensions (nc, h5, mat, csv, json)
- **doc_indicators**: Documentation signals (README, docs/, pdf, tutorial, paper, guide)
- **imas_indicators**: IMAS-specific patterns (put_slice, get_slice, IDS names, "imas" in path)
- **physics_indicators**: Physics domains (equilibrium, transport, MHD)
- **quality_indicators**: Project quality signals (has_readme, has_makefile, has_git)

## Enrichment Decision (should_enrich)

Enrichment runs deep analysis: `dust` (file sizes), `tokei` (LOC), pattern matching.
This can be SLOW or HANG for very large directories.

**NEVER enrich** (should_enrich=false, CRITICAL):
- **Root containers**: `/work`, `/home`, `/opt`, `/common` - too large, would hang
- **Depth ≤ 1 containers**: `/work/*`, `/home/*` - still potentially huge
- **Data containers with many files**: modeling_data, experimental_data with total_files > 1000
- **Archive/system directories**: No value in deep analysis
- **Directories with no code indicators**: Nothing to count with tokei

**Enrich** (should_enrich=true) when:
- **Code directories**: Modeling code, analysis code, operations code, data access
- **Total files < 5000**: Reasonable size for analysis
- **Depth >= 2**: Not a top-level container
- **Has code indicators**: Python, Fortran, C, etc.

When setting **should_enrich=false**, set **enrich_skip_reason** to explain:
- "root container - too large"
- "data container - too many files"
- "no code files detected"
- "archive - no value"

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
- LOC > 10,000 lines in physics language (Fortran, Python) → boost score_modeling_code by 0.1
- Total bytes > 1GB → boost score_modeling_data by 0.1
- Has multiple format reads/writes → this is a data interface, boost score_data_access by 0.2

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
## Calibration Examples by Score Range

Use these previously scored paths to calibrate your decisions (combined score):

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

{% if dimension_examples %}
## Cross-Facility Dimension Calibration

These examples from across facilities show what HIGH scores look like per dimension.
Use them to calibrate your dimension-specific scoring:

{% if dimension_examples.score_modeling_code %}
**score_modeling_code (High 0.6+):**
{% for p in dimension_examples.score_modeling_code %}
- `{{ p.path }}` [{{ p.facility }}] → {{ p.dimension_score }} - {{ p.description }}
{% endfor %}
{% endif %}

{% if dimension_examples.score_analysis_code %}
**score_analysis_code (High 0.6+):**
{% for p in dimension_examples.score_analysis_code %}
- `{{ p.path }}` [{{ p.facility }}] → {{ p.dimension_score }} - {{ p.description }}
{% endfor %}
{% endif %}

{% if dimension_examples.score_operations_code %}
**score_operations_code (High 0.6+):**
{% for p in dimension_examples.score_operations_code %}
- `{{ p.path }}` [{{ p.facility }}] → {{ p.dimension_score }} - {{ p.description }}
{% endfor %}
{% endif %}

{% if dimension_examples.score_data_access %}
**score_data_access (High 0.6+):**
{% for p in dimension_examples.score_data_access %}
- `{{ p.path }}` [{{ p.facility }}] → {{ p.dimension_score }} - {{ p.description }}
{% endfor %}
{% endif %}

{% if dimension_examples.score_imas %}
**score_imas (High 0.6+):**
{% for p in dimension_examples.score_imas %}
- `{{ p.path }}` [{{ p.facility }}] → {{ p.dimension_score }} - {{ p.description }}
{% endfor %}
{% endif %}
{% endif %}
{% include "schema/scoring-output.md" %}
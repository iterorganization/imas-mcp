"""
Path enrichment pipeline for high-value directories.

Runs deep analysis on scored paths to collect additional metadata:
- Pattern search: Code patterns via rg (imports, format conversions)
- Storage analysis: Directory size breakdown via du
- Lines of code: Language breakdown via tokei

Enrichment is triggered for paths scoring above a threshold (0.75 by default).
Multi-format detection looks for code that loads one format and writes another,
indicating data conversion/mapping utilities.

Example format conversion patterns:
- Load EQDSK + write IMAS
- Read MDSplus + save HDF5
- Load NetCDF + put IDS
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from imas_codex.discovery.base.facility import get_facility
from imas_codex.remote.executor import run_python_script

logger = logging.getLogger(__name__)

# ============================================================================
# Pattern Definitions - Mapped to Score Dimensions
# ============================================================================
#
# These patterns are searched by `rg` during enrichment. Results inform rescoring.
# Patterns are grouped by the score dimension they primarily inform.

# -- Data Access Patterns (score_data_access) --
# Facility-native data systems: MDSplus, PPF, UFile, shotfiles
DATA_ACCESS_PATTERNS = {
    "mdsplus": r"(mdsconnect|mdsopen|mdsvalue|MDSplus|TdiExecute|connection\.openTree|TreeNode)",
    "ppf": r"(ppf\.read|ppfget|jet\.ppf|ppfuid|ppfgo|ppfdat)",
    "ufile": r"(ufile\.read|read_ufile|ufiles|UFILE)",
    "shotfile": r"(shotfile\.open|sfread|dd\.shotfile|kk\()",
    "hdf5": r"(h5py\.File|hdf5\.open|\.h5\"|\.hdf5\"|HDFStore)",
    "netcdf": r"(xr\.open|xarray\.open|netCDF4|\.nc\"|Dataset)",
}

# -- IMAS Patterns (score_imas) --
# IMAS data dictionary, IDS access, Access Layer
IMAS_PATTERNS = {
    "imas_read": r"(imas\.database|ids_get|get_ids|imas\.open|\.get_slice|DBEntry)",
    "imas_write": r"(put_slice|ids\.put|imas\.create|ids_put|partial_get)",
    "ids_struct": r"(ids_properties|homogeneous_time|ids\.[a-z_]+\.[a-z_]+)",
    "al_core": r"(imas_core|imas\.imasdef|uda_imas|al_|acces_layer)",
}

# -- Modeling Code Patterns (score_modeling_code) --
# Physics simulation codes and equilibrium tools
MODELING_CODE_PATTERNS = {
    "equilibrium": r"(EFIT|LIUQE|CLISTE|CREATE|HELENA|CHEASE|equilibrium|flux_surface)",
    "eqdsk": r"(read_eqdsk|load_eqdsk|write_eqdsk|geqdsk|from_eqdsk)",
    "transport": r"(JETTO|ASTRA|TRANSP|CRONOS|ETS|transport\.solve)",
    "mhd": r"(JOREK|MARS|KINX|MISHKA|stability|tearing|mhd)",
    "heating": r"(NUBEAM|RABBIT|NEMO|PENCIL|TORIC|heating|nbi|icrf|ecrh)",
    "core_profiles": r"(core_profiles|profiles_1d|electron_density|electron_temperature)",
}

# -- Analysis Code Patterns (score_analysis_code) --
# Data analysis, signal processing, fitting
ANALYSIS_PATTERNS = {
    "fitting": r"(curve_fit|lmfit|scipy\.optimize|least_squares|minimize)",
    "signal": r"(fft|spectral|bandpass|lowpass|savgol|butterworth|filtering)",
    "statistics": r"(bootstrap|monte_carlo|bayesian|mcmc|uncertainty)",
    "diagnostics": r"(thomson|ece|interferometer|mse|cxrs|bolo|bolometer)",
}

# -- Operations Code Patterns (score_operations_code) --
# Real-time plasma control systems
OPERATIONS_PATTERNS = {
    "control": r"(controller|pid|feedback|setpoint|actuator|plasma_control)",
    "realtime": r"(real_time|real-time|rtc|realtime|pcs|dcs)",
    "interlock": r"(interlock|safety|limit|protection|alarm|watchdog)",
}

# -- Workflow Patterns (score_workflow) --
# Orchestration, job submission, pipelines
WORKFLOW_PATTERNS = {
    "orchestration": r"(airflow|luigi|snakemake|nextflow|dask\.delayed)",
    "batch": r"(sbatch|slurm|pbs|qsub|job_submit|htcondor)",
    "pipeline": r"(pipeline|workflow|dag|task_graph|kepler)",
}

# -- Visualization Patterns (score_visualization) --
# Plotting and GUI tools
VISUALIZATION_PATTERNS = {
    "plotting": r"(matplotlib|plotly|bokeh|seaborn|plt\.plot|ax\.)",
    "gui": r"(tkinter|PyQt|PySide|wxPython|gui|widget)",
    "interactive": r"(jupyter|ipywidgets|panel|dash|streamlit)",
}

# -- Documentation Patterns (score_documentation) --
# Documentation tools and references
DOCUMENTATION_PATTERNS = {
    "doctools": r"(sphinx|mkdocs|doxygen|docstring|restructuredtext)",
    "readme": r"(README|CONTRIBUTING|CHANGELOG|LICENSE|AUTHORS)",
    "tutorial": r"(tutorial|example|demo|notebook|getting_started)",
}

# Combined read/write patterns for legacy compatibility
FORMAT_READ_PATTERNS = {
    **DATA_ACCESS_PATTERNS,
    **{k: v for k, v in IMAS_PATTERNS.items() if "read" in k},
    "eqdsk": MODELING_CODE_PATTERNS["eqdsk"],
    "mat": r"(scipy\.io\.loadmat|loadmat|sio\.loadmat|\.mat\")",
    "json": r"(json\.load|json\.loads)",
    "csv": r"(pd\.read_csv|csv\.reader)",
    "pickle": r"(pickle\.load|\.pkl|\.pickle)",
}

FORMAT_WRITE_PATTERNS = {
    **{k: v for k, v in IMAS_PATTERNS.items() if "write" in k},
    "hdf5_write": r"(h5py.*create|\.to_hdf|hdf5\.write|\.h5\")",
    "netcdf_write": r"(\.to_netcdf|xr\.to_netcdf|netcdf\.create)",
    "mat_write": r"(scipy\.io\.savemat|savemat|sio\.savemat)",
    "json_write": r"(json\.dump|\.to_json)",
    "csv_write": r"(\.to_csv|csv\.writer)",
    "pickle_write": r"(pickle\.dump)",
    "eqdsk_write": r"(write_eqdsk|to_eqdsk|save_eqdsk)",
}

# Master pattern registry: category -> (patterns_dict, primary_score_dimension)
PATTERN_REGISTRY = {
    "data_access": (DATA_ACCESS_PATTERNS, "score_data_access"),
    "imas": (IMAS_PATTERNS, "score_imas"),
    "modeling": (MODELING_CODE_PATTERNS, "score_modeling_code"),
    "analysis": (ANALYSIS_PATTERNS, "score_analysis_code"),
    "operations": (OPERATIONS_PATTERNS, "score_operations_code"),
    "workflow": (WORKFLOW_PATTERNS, "score_workflow"),
    "visualization": (VISUALIZATION_PATTERNS, "score_visualization"),
    "documentation": (DOCUMENTATION_PATTERNS, "score_documentation"),
}


@dataclass
class EnrichmentResult:
    """Result of enriching a single path with deep analysis."""

    path: str
    # Pattern matches
    read_formats_found: list[str] = field(default_factory=list)
    write_formats_found: list[str] = field(default_factory=list)
    read_matches: int = 0  # Total read pattern match count
    write_matches: int = 0  # Total write pattern match count
    is_multiformat: bool = False
    conversion_pairs: list[str] = field(default_factory=list)  # e.g., ["eqdsk->imas"]
    # Pattern categories (mdsplus, hdf5, imas, etc.) with per-category match counts
    pattern_categories: dict[str, int] = field(default_factory=dict)
    # Storage analysis
    total_bytes: int | None = None
    largest_dirs: dict[str, int] = field(default_factory=dict)
    # Lines of code
    total_lines: int | None = None
    language_breakdown: dict[str, int] = field(default_factory=dict)
    # Errors
    error: str | None = None

    def to_graph_dict(self) -> dict[str, Any]:
        """Convert to dict for graph update."""
        return {
            "is_enriched": True,
            "enriched_at": datetime.now(UTC).isoformat(),
            "read_formats_found": self.read_formats_found,
            "write_formats_found": self.write_formats_found,
            "is_multiformat": self.is_multiformat,
            "conversion_pairs": self.conversion_pairs,
            "total_bytes": self.total_bytes,
            "total_lines": self.total_lines,
            "language_breakdown": json.dumps(self.language_breakdown)
            if self.language_breakdown
            else None,
        }


def enrich_paths(
    facility: str,
    paths: list[str],
    timeout: int = 300,
    path_purposes: dict[str, str | None] | None = None,
) -> list[EnrichmentResult]:
    """Enrich multiple paths with deep analysis.

    Uses enrich_directories.py Python script via run_python_script() for
    reliable JSON parsing. Paths are passed as JSON input, avoiding shell
    quoting issues.

    Args:
        facility: Facility identifier
        paths: List of paths to enrich
        timeout: SSH timeout in seconds
        path_purposes: Optional mapping of path -> purpose category for
            targeted pattern matching. If a path is classified as
            documentation, skip code patterns. If classified as data,
            focus on data format patterns.

    Returns:
        List of EnrichmentResult objects
    """
    if not paths:
        return []

    # Resolve SSH host
    try:
        config = get_facility(facility)
        ssh_host = config.get("ssh_host", facility)
    except ValueError:
        ssh_host = facility

    # Build input data for the remote script
    # Include path_purposes so the remote script can select targeted patterns
    input_data = {
        "paths": paths,
        "path_purposes": path_purposes or {},
    }

    try:
        output = run_python_script(
            "enrich_directories.py",
            input_data=input_data,
            ssh_host=ssh_host,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"Enrichment timed out for {facility}")
        return [EnrichmentResult(path=p, error="timeout") for p in paths]
    except Exception as e:
        logger.warning(f"Enrichment failed for {facility}: {e}")
        return [EnrichmentResult(path=p, error=str(e)[:100]) for p in paths]

    # Parse results - handle stderr mixed in
    try:
        if "[stderr]:" in output:
            output = output.split("[stderr]:")[0].strip()
        results_data = json.loads(output)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse enrichment output: {e}")
        return [EnrichmentResult(path=p, error="parse error") for p in paths]

    results = []
    for data in results_data:
        path = data.get("path", "")
        result = EnrichmentResult(path=path)

        if data.get("error"):
            result.error = data.get("error")
            results.append(result)
            continue

        # Extract pattern matches
        read_matches = data.get("read_matches", 0)
        write_matches = data.get("write_matches", 0)

        # Store raw counts for display
        result.read_matches = read_matches
        result.write_matches = write_matches

        # Determine if multi-format (has both reads and writes)
        result.is_multiformat = read_matches > 0 and write_matches > 0

        # Storage
        result.total_bytes = data.get("total_bytes")

        # Lines of code
        result.total_lines = data.get("total_lines")
        lang_breakdown = data.get("language_breakdown", {})
        if lang_breakdown:
            result.language_breakdown = lang_breakdown

        # Pattern categories (mdsplus, hdf5, imas, etc.)
        pattern_cats = data.get("pattern_categories", {})
        if pattern_cats:
            result.pattern_categories = pattern_cats

        results.append(result)

    # Fill missing paths
    result_paths = {r.path for r in results}
    for p in paths:
        if p not in result_paths:
            results.append(EnrichmentResult(path=p, error="missing"))

    return results


def persist_enrichment(facility: str, results: list[EnrichmentResult]) -> int:
    """Persist enrichment results to graph.

    Args:
        facility: Facility ID
        results: List of EnrichmentResult objects

    Returns:
        Number of paths updated
    """
    from imas_codex.graph import GraphClient

    updates = []
    for r in results:
        if r.error:
            continue
        data = r.to_graph_dict()
        data["id"] = f"{facility}:{r.path}"
        updates.append(data)

    if not updates:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $updates AS u
            MATCH (p:FacilityPath {id: u.id})
            SET p.is_enriched = u.is_enriched,
                p.enriched_at = u.enriched_at,
                p.is_multiformat = u.is_multiformat,
                p.total_bytes = u.total_bytes,
                p.total_lines = u.total_lines,
                p.language_breakdown = u.language_breakdown
            """,
            updates=updates,
        )

    return len(updates)


def get_paths_pending_enrichment(facility: str, threshold: float = 0.75) -> list[str]:
    """Get paths marked for enrichment that haven't been enriched yet.

    Uses should_enrich flag set by LLM during scoring. The threshold is used
    as a fallback for paths scored before should_enrich was added.

    Args:
        facility: Facility ID
        threshold: Fallback minimum score for legacy paths without should_enrich

    Returns:
        List of paths ready for enrichment
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath {facility_id: $facility})
            WHERE p.status = 'scored'
                AND (p.is_enriched IS NULL OR p.is_enriched = false)
                AND (
                    p.should_enrich = true
                    OR (p.should_enrich IS NULL AND p.score >= $threshold)
                )
            RETURN p.path AS path
            ORDER BY p.score DESC
            """,
            facility=facility,
            threshold=threshold,
        )

    return [r["path"] for r in result]


def run_enrichment_pipeline(
    facility: str,
    threshold: float = 0.75,
    batch_size: int = 50,
    limit: int | None = None,
) -> dict[str, int]:
    """Run enrichment pipeline for a facility.

    Args:
        facility: Facility ID
        threshold: Score threshold for enrichment
        batch_size: Paths per SSH call
        limit: Maximum paths to process

    Returns:
        Dict with counts
    """
    paths = get_paths_pending_enrichment(facility, threshold)

    if limit:
        paths = paths[:limit]

    if not paths:
        logger.info(f"No paths pending enrichment for {facility}")
        return {"processed": 0, "enriched": 0}

    logger.info(f"Enriching {len(paths)} paths for {facility}")

    total_enriched = 0
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        results = enrich_paths(facility, batch)
        enriched = persist_enrichment(facility, results)
        total_enriched += enriched
        logger.debug(f"Batch {i // batch_size + 1}: enriched {enriched}/{len(batch)}")

    return {"processed": len(paths), "enriched": total_enriched}

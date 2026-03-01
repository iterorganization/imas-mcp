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

from imas_codex.config.discovery_config import get_discovery_config
from imas_codex.discovery.base.facility import get_facility
from imas_codex.remote.executor import async_run_python_script, run_python_script

logger = logging.getLogger(__name__)

# ============================================================================
# Pattern Definitions - Mapped to Score Dimensions
# ============================================================================
#
# These patterns are searched by `rg` during enrichment. Results inform rescoring.
# Patterns are grouped by the score dimension they primarily inform.

# -- Data Access Patterns (score_data_access) --
# Facility-native data systems: MDSplus, PPF, UFile, shotfiles, UDA, EDAS
DATA_ACCESS_PATTERNS = {
    "mdsplus": r"(mdsconnect|mdsopen|mdsvalue|MDSplus|TdiExecute|connection\.openTree|TreeNode)",
    "ppf": r"(ppf\.read|ppfget|jet\.ppf|ppfuid|ppfgo|ppfdat)",
    "ufile": r"(ufile\.read|read_ufile|ufiles|UFILE)",
    "shotfile": r"(shotfile\.open|sfread|dd\.shotfile|kk\()",
    "hdf5": r"(h5py\.File|hdf5\.open|\.h5\"|\.hdf5\"|HDFStore)",
    "netcdf": r"(xr\.open|xarray\.open|netCDF4|\.nc\"|Dataset)",
    # UDA - Universal Data Access (JET, MAST, JT-60SA)
    "uda": r"(uda\.|pyuda|UDA_client|getdata|getData|getIdamData|Client\(\)\.get)",
    # EDAS - JT-60SA Experimental Data Acquisition System
    "edas": r"(eddb\.|eGIS|eSLICE|eSURF|EDASDB|edas_read|edas\.get|labcom)",
}

# -- IMAS Patterns (score_imas) --
# IMAS data dictionary, IDS access, Access Layer
IMAS_PATTERNS = {
    "imas_read": r"(imas\.database|ids_get|get_ids|imas\.open|\.get_slice|DBEntry)",
    "imas_write": r"(put_slice|ids\.put|imas\.create|ids_put|partial_get)",
    "ids_struct": r"(ids_properties|homogeneous_time|ids\.[a-z_]+\.[a-z_]+)",
    "al_core": r"(imas_core|imas\.imasdef|uda_imas|al_|acces_layer)",
    "ids_path": r"(equilibrium[./]|core_profiles[./]|magnetics[./]|summary[./]|pf_active[./]|wall[./]|nbi[./]|ece[./]|thomson_scattering[./]|interferometer[./]|bolometer[./])",
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

# -- Convention Patterns (score_convention) --
# Sign conventions, coordinate conventions, COCOS, unit systems
CONVENTION_PATTERNS = {
    "cocos": r"(COCOS|cocos_[0-9]+|cocos_transform|cocos_identify|cocosify|set_cocos|get_cocos)",
    "sign_convention": r"(sign_convention|ip_sign|bt_sign|sign_bp|sign_b0|sigma_ip|sigma_b0|sigma_rphiz|sign_q)",
    "coord_convention": r"(coordinate_convention|coord_system|cylindrical|toroidal_angle|poloidal_angle|phi_convention|theta_convention)",
    "handedness": r"(right_hand|left_hand|clockwise|counter_clockwise|countercockwise|ccw_|cw_|rhs_|lhs_)",
    "flux_convention": r"(psi_norm|psi_boundary|psi_axis|rho_tor|rho_pol|flux_convention|psi_sign)",
    "unit_conversion": r"(unit_convert|units_to|to_si|from_si|pint\.Unit|ureg\.|astropy\.units|convert_units)",
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
    "convention": (CONVENTION_PATTERNS, "score_convention"),
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
    # Errors and warnings
    error: str | None = None
    warnings: list[str] = field(default_factory=list)  # e.g., ["tokei_timeout(61GB)"]

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
            "language_breakdown": self.language_breakdown or None,
            "pattern_categories": self.pattern_categories or None,
            "read_matches": self.read_matches,
            "write_matches": self.write_matches,
            "warnings": self.warnings,
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
        "pattern_categories": _build_enrich_patterns(),
    }

    try:
        output = run_python_script(
            "enrich_directories.py",
            input_data=input_data,
            ssh_host=ssh_host,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        # Recover partial results from JSONL lines that completed before timeout
        partial_output = getattr(e, "output", None) or ""
        if partial_output:
            logger.warning(
                f"Enrichment SSH timed out after {timeout}s for {facility}, "
                f"recovering partial output ({len(partial_output)} bytes)"
            )
            return _parse_enrich_output(
                partial_output,
                paths,
                fill_missing_error=f"ssh_timeout({timeout}s)",
            )
        logger.warning(f"Enrichment SSH timed out after {timeout}s for {facility}")
        return [
            EnrichmentResult(path=p, error=f"ssh_timeout({timeout}s)") for p in paths
        ]
    except Exception as e:
        logger.warning(f"Enrichment failed for {facility}: {e}")
        return [EnrichmentResult(path=p, error=str(e)[:100]) for p in paths]

    return _parse_enrich_output(output, paths)


def _build_enrich_patterns() -> dict[str, str]:
    """Build pattern categories for remote enrichment.

    Merges YAML-config patterns (data systems + physics domains) with
    enrichment-specific patterns not covered by config YAML.
    Returns category -> rg regex pattern string.
    """
    patterns: dict[str, str] = {}

    # Load canonical patterns from YAML config (same source as scanner)
    config = get_discovery_config()
    for name, ds in config.scoring.data_systems.items():
        if ds.patterns:
            patterns[name] = "|".join(p.pattern for p in ds.patterns)
    for name, pd in config.scoring.physics_domains.items():
        if pd.patterns:
            patterns[name] = "|".join(p.pattern for p in pd.patterns)

    # Add enrichment-specific patterns not in YAML config
    for category, (cat_patterns, _score_dim) in PATTERN_REGISTRY.items():
        for subcat, pattern in cat_patterns.items():
            key = f"{category}_{subcat}" if subcat != category else subcat
            if key not in patterns:
                patterns[key] = pattern

    return patterns


def _build_enrich_input(
    facility: str,
    paths: list[str],
    path_purposes: dict[str, str | None] | None = None,
) -> tuple[str, dict]:
    """Build SSH host and input data for enrichment.

    Returns:
        Tuple of (ssh_host, input_data)
    """
    try:
        config = get_facility(facility)
        ssh_host = config.get("ssh_host", facility)
    except ValueError:
        ssh_host = facility

    input_data = {
        "paths": paths,
        "path_purposes": path_purposes or {},
        "pattern_categories": _build_enrich_patterns(),
    }
    return ssh_host, input_data


def _parse_single_enrich_result(data: dict) -> EnrichmentResult:
    """Parse a single enrichment result dict into an EnrichmentResult."""
    path = data.get("path", "")
    result = EnrichmentResult(path=path)

    if data.get("error"):
        result.error = data.get("error")
        return result

    read_matches = data.get("read_matches", 0)
    write_matches = data.get("write_matches", 0)
    result.read_matches = read_matches
    result.write_matches = write_matches
    result.is_multiformat = read_matches > 0 and write_matches > 0
    result.total_bytes = data.get("total_bytes")
    result.total_lines = data.get("total_lines")
    lang_breakdown = data.get("language_breakdown", {})
    if lang_breakdown:
        result.language_breakdown = lang_breakdown
    pattern_cats = data.get("pattern_categories", {})
    if pattern_cats:
        result.pattern_categories = pattern_cats
    warnings = data.get("warnings", [])
    if warnings:
        result.warnings = warnings

    return result


def _parse_enrich_output(
    output: str,
    paths: list[str],
    fill_missing_error: str = "missing",
) -> list[EnrichmentResult]:
    """Parse JSON/JSONL output from enrich_directories.py.

    Handles two formats:
    - JSONL: One JSON object per line (streaming format, preferred)
    - JSON array: Single JSON array (legacy format)

    Gracefully handles partial output (e.g., from SSH timeout) by parsing
    whatever complete JSONL lines are available.

    Shared between sync and async enrich_paths.
    """
    if "[stderr]:" in output:
        output = output.split("[stderr]:")[0].strip()

    if not output.strip():
        return [EnrichmentResult(path=p, error=fill_missing_error) for p in paths]

    results: list[EnrichmentResult] = []

    # Try JSON array first (single-line legacy format)
    stripped = output.strip()
    if stripped.startswith("["):
        try:
            results_data = json.loads(stripped)
            results = [_parse_single_enrich_result(d) for d in results_data]
            result_paths = {r.path for r in results}
            for p in paths:
                if p not in result_paths:
                    results.append(EnrichmentResult(path=p, error=fill_missing_error))
            return results
        except json.JSONDecodeError:
            pass  # Fall through to JSONL parsing

    # Parse as JSONL (one JSON object per line, streaming format)
    for line in stripped.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            results.append(_parse_single_enrich_result(data))
        except json.JSONDecodeError:
            continue  # Skip incomplete/corrupt lines (e.g., truncated by timeout)

    # Fill missing paths
    result_paths = {r.path for r in results}
    for p in paths:
        if p not in result_paths:
            results.append(EnrichmentResult(path=p, error=fill_missing_error))

    return results


async def async_enrich_paths(
    facility: str,
    paths: list[str],
    timeout: int = 300,
    path_purposes: dict[str, str | None] | None = None,
) -> list[EnrichmentResult]:
    """Async version of enrich_paths using asyncio subprocesses.

    Fully cancellable — SSH subprocess is killed on task cancellation.
    Same parsing logic and error handling as sync version.
    """
    import asyncio

    if not paths:
        return []

    ssh_host, input_data = _build_enrich_input(facility, paths, path_purposes)

    try:
        output = await async_run_python_script(
            "enrich_directories.py",
            input_data=input_data,
            ssh_host=ssh_host,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        # Recover partial results from JSONL lines that completed before timeout
        partial_output = getattr(e, "output", None) or ""
        if partial_output:
            logger.warning(
                f"Enrichment SSH timed out after {timeout}s for {facility}, "
                f"recovering partial output ({len(partial_output)} bytes)"
            )
            return _parse_enrich_output(
                partial_output,
                paths,
                fill_missing_error=f"ssh_timeout({timeout}s)",
            )
        logger.warning(f"Enrichment SSH timed out after {timeout}s for {facility}")
        return [
            EnrichmentResult(path=p, error=f"ssh_timeout({timeout}s)") for p in paths
        ]
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"Enrichment failed for {facility}: {e}")
        return [EnrichmentResult(path=p, error=str(e)[:100]) for p in paths]

    return _parse_enrich_output(output, paths)


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

    # Serialize dicts/lists for Neo4j storage
    for u in updates:
        if isinstance(u.get("language_breakdown"), dict):
            u["language_breakdown"] = (
                json.dumps(u["language_breakdown"]) if u["language_breakdown"] else None
            )
        if isinstance(u.get("pattern_categories"), dict):
            u["pattern_categories"] = (
                json.dumps(u["pattern_categories"]) if u["pattern_categories"] else None
            )
        if isinstance(u.get("warnings"), list):
            u["enrich_warnings"] = (
                ", ".join(u.pop("warnings")) if u["warnings"] else None
            )

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
                p.language_breakdown = u.language_breakdown,
                p.pattern_categories = u.pattern_categories,
                p.read_matches = u.read_matches,
                p.write_matches = u.write_matches,
                p.enrich_warnings = u.enrich_warnings,
                p.claimed_at = null
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

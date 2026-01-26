"""
Path enrichment pipeline for high-value directories.

Runs deep analysis on scored paths to collect additional metadata:
- Pattern search: Code patterns via rg (imports, format conversions)
- Storage analysis: Directory size breakdown via dust
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

from imas_codex.discovery.facility import get_facility
from imas_codex.remote.executor import run_script_via_stdin as run_remote

logger = logging.getLogger(__name__)

# Common format loading/reading patterns by data system
FORMAT_READ_PATTERNS = {
    "eqdsk": r"(read_eqdsk|load_eqdsk|eqdsk\.read|from_eqdsk|ReadEQDSK)",
    "geqdsk": r"(read_geqdsk|load_geqdsk|geqdsk\.read|from_geqdsk)",
    "mdsplus": r"(mdsconnect|mdsopen|mdsvalue|MDSplus|TdiExecute|connection\.openTree)",
    "hdf5": r"(h5py\.File|hdf5\.open|\.h5|netCDF4\.Dataset)",
    "netcdf": r"(xr\.open|xarray\.open|netcdf\.open|\.nc)",
    "mat": r"(scipy\.io\.loadmat|loadmat|sio\.loadmat|\.mat)",
    "json": r"(json\.load|json\.loads|\.json)",
    "csv": r"(pd\.read_csv|csv\.reader|\.csv)",
    "pickle": r"(pickle\.load|\.pkl|\.pickle)",
    "imas_read": r"(imas\.database|ids_get|get_ids|imas\.open|\.get_slice)",
    "ppf": r"(ppf\.read|ppfget|jet\.ppf)",
    "ufile": r"(ufile\.read|read_ufile|ufiles)",
}

# Common format writing patterns by data system
FORMAT_WRITE_PATTERNS = {
    "imas_write": r"(put_slice|ids\.put|imas\.create|ids_put|\.close\(\))",
    "hdf5_write": r"(h5py.*create|\.to_hdf|hdf5\.write|\.h5)",
    "netcdf_write": r"(\.to_netcdf|xr\.to_netcdf|netcdf\.create)",
    "mat_write": r"(scipy\.io\.savemat|savemat|sio\.savemat)",
    "json_write": r"(json\.dump|\.to_json)",
    "csv_write": r"(\.to_csv|csv\.writer)",
    "pickle_write": r"(pickle\.dump)",
    "eqdsk_write": r"(write_eqdsk|to_eqdsk|save_eqdsk)",
}


@dataclass
class EnrichmentResult:
    """Result of enriching a single path with deep analysis."""

    path: str
    # Pattern matches
    read_formats_found: list[str] = field(default_factory=list)
    write_formats_found: list[str] = field(default_factory=list)
    is_multiformat: bool = False
    conversion_pairs: list[str] = field(default_factory=list)  # e.g., ["eqdsk->imas"]
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


def _build_enrichment_script(paths: list[str]) -> str:
    """Build a bash script for enrichment data collection.

    Collects:
    - rg pattern matches for format conversion detection (simplified patterns)
    - dust for storage (faster than du on large dirs), falls back to du
    - tokei for lines of code

    Returns JSON array with results per path.
    """
    # Simplified patterns - shorter to avoid arg length issues
    read_patterns = [
        "read_eqdsk|load_eqdsk|from_eqdsk",
        "mdsconnect|mdsopen|MDSplus",
        "h5py\\.File|hdf5|netCDF4",
        "xr\\.open|xarray\\.open",
        "json\\.load|pickle\\.load",
        "imas\\.database|get_ids|get_slice",
        "ppf\\.read|ppfget",
    ]
    write_patterns = [
        "put_slice|ids\\.put|imas\\.create",
        "to_hdf|hdf5\\.write",
        "to_netcdf|netcdf\\.create",
        "json\\.dump|pickle\\.dump",
        "write_eqdsk|to_eqdsk",
    ]

    read_pattern = "|".join(read_patterns)
    write_pattern = "|".join(write_patterns)

    # Generate path list as bash array entries
    path_entries = "\n".join(f'  "{p}"' for p in paths)

    script = f"""#!/bin/bash

PATHS=(
{path_entries}
)

echo "["
first=true

for path in "${{PATHS[@]}}"; do
    if [ ! -d "$path" ]; then
        continue
    fi

    if [ "$first" = true ]; then
        first=false
    else
        echo ","
    fi

    # Initialize counts
    read_count=0
    write_count=0
    total_bytes=0
    total_lines=0

    # Pattern search with rg (if available)
    if command -v rg &> /dev/null; then
        read_count=$(rg -c --no-messages --max-depth 3 -e '{read_pattern}' "$path" 2>/dev/null | awk -F: '{{sum+=$2}} END {{print sum+0}}' || echo 0)
        write_count=$(rg -c --no-messages --max-depth 3 -e '{write_pattern}' "$path" 2>/dev/null | awk -F: '{{sum+=$2}} END {{print sum+0}}' || echo 0)
    fi

    # Storage - prefer dust (faster on large dirs), fallback to du
    if command -v dust &> /dev/null; then
        total_bytes=$(dust -sb "$path" 2>/dev/null | awk '{{print $1}}' | head -1 || echo 0)
    else
        total_bytes=$(du -sb "$path" 2>/dev/null | cut -f1 || echo 0)
    fi

    # Lines of code with tokei (if available)
    if command -v tokei &> /dev/null; then
        total_lines=$(tokei "$path" -o json 2>/dev/null | grep -o '"code":[0-9]*' | head -1 | grep -o '[0-9]*' || echo 0)
    fi

    # Output JSON (no jq required)
    echo "{{\\"path\\":\\"$path\\",\\"read_matches\\":$read_count,\\"write_matches\\":$write_count,\\"total_bytes\\":$total_bytes,\\"total_lines\\":$total_lines}}"
done

echo "]"
"""
    return script


def enrich_paths(
    facility: str,
    paths: list[str],
    timeout: int = 300,
) -> list[EnrichmentResult]:
    """Enrich multiple paths with deep analysis.

    Args:
        facility: Facility identifier
        paths: List of paths to enrich
        timeout: SSH timeout in seconds

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

    script = _build_enrichment_script(paths)

    try:
        output = run_remote(
            script,
            ssh_host=ssh_host,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"Enrichment timed out for {facility}")
        return [EnrichmentResult(path=p, error="timeout") for p in paths]
    except Exception as e:
        logger.warning(f"Enrichment failed for {facility}: {e}")
        return [EnrichmentResult(path=p, error=str(e)[:100]) for p in paths]

    # Parse results
    try:
        results_data = json.loads(output)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse enrichment output: {e}")
        return [EnrichmentResult(path=p, error="parse error") for p in paths]

    results = []
    for data in results_data:
        path = data.get("path", "")
        result = EnrichmentResult(path=path)

        # Extract pattern matches
        read_matches = data.get("read_matches", 0)
        write_matches = data.get("write_matches", 0)

        # Determine if multi-format (has both reads and writes)
        result.is_multiformat = read_matches > 0 and write_matches > 0

        # Storage
        result.total_bytes = data.get("total_bytes")

        # Lines of code
        result.total_lines = data.get("total_lines")
        tokei = data.get("tokei", {})
        if tokei:
            # Extract language breakdown
            for lang, stats in tokei.items():
                if lang != "Total" and isinstance(stats, dict):
                    code = stats.get("code", 0)
                    if code > 0:
                        result.language_breakdown[lang] = code

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
    """Get high-scoring paths that haven't been enriched.

    Args:
        facility: Facility ID
        threshold: Minimum score for enrichment

    Returns:
        List of paths above threshold without enrichment
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath {facility_id: $facility})
            WHERE p.status = 'scored'
                AND p.score >= $threshold
                AND (p.is_enriched IS NULL OR p.is_enriched = false)
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

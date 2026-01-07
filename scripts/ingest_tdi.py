#!/usr/bin/env python3
"""Parse TDI function files and ingest into knowledge graph.

This script extracts quantity information from TCV TDI function files
and creates/updates TDIFunction nodes with supported_quantities lists.

Usage:
    uv run python scripts/ingest_tdi.py epfl [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TDIFunctionInfo:
    """Parsed information from a TDI .fun file."""

    name: str
    path: str
    quantities: list[str]
    signature: str | None = None
    description: str | None = None
    physics_domain: str | None = None


def parse_tdi_quantities(content: str) -> list[str]:
    """Extract case statement quantities from TDI function content.

    Patterns matched:
    - case("QUANTITY_NAME")
    - case ("QUANTITY_NAME")
    - CASE("quantity_name")

    Args:
        content: TDI function file content

    Returns:
        Sorted list of unique quantity names (uppercase)
    """
    # Match case statements with flexible spacing
    pattern = r'case\s*\(\s*["\']([A-Z_0-9]+)["\']\s*\)'
    matches = re.findall(pattern, content, re.IGNORECASE)
    return sorted({q.upper() for q in matches})


def parse_tdi_signature(content: str, name: str) -> str | None:
    """Extract function signature from TDI file.

    Looks for the PUBLIC FUN line that declares the function.

    Args:
        content: TDI function file content
        name: Function name (e.g., "tcv_eq")

    Returns:
        Signature string or None if not found
    """
    # Look for: PUBLIC FUN tcv_eq(optional in _signal = "", ...)
    pattern = rf"PUBLIC\s+FUN\s+{re.escape(name)}\s*\([^)]+\)"
    match = re.search(pattern, content, re.IGNORECASE)
    return match.group(0) if match else None


def parse_tdi_description(content: str) -> str | None:
    """Extract function description from comment header.

    Looks for /* ... */ style comments at the start of the file.

    Args:
        content: TDI function file content

    Returns:
        Description text or None
    """
    # Look for opening comment block
    match = re.search(r"/\*\s*\n(.*?)\*/", content, re.DOTALL)
    if match:
        lines = match.group(1).strip().split("\n")
        # Take first non-empty line as description
        for line in lines:
            line = line.strip().lstrip("* \t")
            if line and not line.startswith("$"):
                return line
    return None


def classify_physics_domain(quantities: list[str]) -> str | None:
    """Infer physics domain from quantity names.

    Args:
        quantities: List of quantity names

    Returns:
        Physics domain string or None
    """
    eq_keywords = {"PSI", "R_AXIS", "Z_AXIS", "KAPPA", "DELTA", "Q_95", "I_P", "B_PHI"}
    profile_keywords = {"TE", "TI", "NE", "NI", "ZEFF", "PRESSURE"}
    mhd_keywords = {"W_MHD", "BETA", "BETA_POL", "BETA_TOR", "LI"}

    qty_set = set(quantities)

    if qty_set & eq_keywords:
        return "equilibrium"
    if qty_set & profile_keywords:
        return "profiles"
    if qty_set & mhd_keywords:
        return "mhd"
    return None


def fetch_tdi_files(host: str, tdi_path: str) -> list[tuple[str, str]]:
    """Fetch TDI file list and contents from remote host.

    Args:
        host: SSH host alias
        tdi_path: Path to TDI directory

    Returns:
        List of (filename, content) tuples
    """
    # List .fun files
    result = subprocess.run(
        ["ssh", host, f"find {tdi_path} -name '*.fun' -type f 2>/dev/null"],
        capture_output=True,
        check=True,
    )
    files = [
        p.strip()
        for p in result.stdout.decode("utf-8", errors="replace").strip().split("\n")
        if p.strip()
    ]

    # Fetch content of each file
    results = []
    for filepath in files:
        try:
            result = subprocess.run(
                ["ssh", host, f"cat '{filepath}'"],
                capture_output=True,
                check=True,
            )
            # Use latin-1 for TDI files which may have special chars
            content = result.stdout.decode("latin-1", errors="replace")
            results.append((filepath, content))
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to read %s: %s", filepath, e)

    return results


def parse_tdi_file(filepath: str, content: str) -> TDIFunctionInfo | None:
    """Parse a TDI file and extract function information.

    Args:
        filepath: Path to the .fun file
        content: File content

    Returns:
        TDIFunctionInfo or None if no quantities found
    """
    name = Path(filepath).stem
    quantities = parse_tdi_quantities(content)

    if not quantities:
        return None

    return TDIFunctionInfo(
        name=name,
        path=filepath,
        quantities=quantities,
        signature=parse_tdi_signature(content, name),
        description=parse_tdi_description(content),
        physics_domain=classify_physics_domain(quantities),
    )


def ingest_tdi_functions(
    facility_id: str, functions: list[TDIFunctionInfo], dry_run: bool = False
) -> dict:
    """Ingest TDI functions into the knowledge graph.

    Args:
        facility_id: Facility identifier (e.g., "epfl")
        functions: List of parsed TDI function info
        dry_run: If True, don't actually write to graph

    Returns:
        Summary statistics
    """
    from imas_codex.graph import GraphClient

    stats = {"created": 0, "updated": 0, "skipped": 0}

    if dry_run:
        for func in functions:
            print(f"  {func.name}: {len(func.quantities)} quantities")
            if func.quantities:
                print(f"    {', '.join(func.quantities[:10])}...")
        return stats

    with GraphClient() as client:
        for func in functions:
            # Check if TDIFunction already exists
            result = client.query(
                "MATCH (t:TDIFunction {name: $name}) RETURN t",
                name=func.name,
            )

            if result:
                # Update existing
                client.query(
                    """
                    MATCH (t:TDIFunction {name: $name})
                    SET t.supported_quantities = $quantities,
                        t.signature = $signature,
                        t.description = $description,
                        t.physics_domain = $physics_domain,
                        t.source_file = $source_file
                    """,
                    name=func.name,
                    quantities=func.quantities,
                    signature=func.signature,
                    description=func.description,
                    physics_domain=func.physics_domain,
                    source_file=func.path,
                )
                stats["updated"] += 1
            else:
                # Create new
                client.query(
                    """
                    MERGE (f:Facility {id: $facility_id})
                    CREATE (t:TDIFunction {
                        name: $name,
                        facility_id: $facility_id,
                        supported_quantities: $quantities,
                        signature: $signature,
                        description: $description,
                        physics_domain: $physics_domain,
                        source_file: $source_file
                    })
                    CREATE (t)-[:FACILITY_ID]->(f)
                    """,
                    facility_id=facility_id,
                    name=func.name,
                    quantities=func.quantities,
                    signature=func.signature,
                    description=func.description,
                    physics_domain=func.physics_domain,
                    source_file=func.path,
                )
                stats["created"] += 1

    return stats


def main():
    """Parse TDI functions and ingest into graph."""
    parser = argparse.ArgumentParser(description="Ingest TDI functions into graph")
    parser.add_argument("facility", help="Facility ID (e.g., epfl)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to graph")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # TDI paths by facility
    tdi_paths = {
        "epfl": "/usr/local/CRPP/tdi/tcv",
    }

    if args.facility not in tdi_paths:
        logger.error("Unknown facility: %s (known: %s)", args.facility, list(tdi_paths))
        return 1

    tdi_path = tdi_paths[args.facility]
    logger.info("Fetching TDI files from %s:%s", args.facility, tdi_path)

    # Fetch and parse files
    files = fetch_tdi_files(args.facility, tdi_path)
    logger.info("Found %d .fun files", len(files))

    functions = []
    for filepath, content in files:
        info = parse_tdi_file(filepath, content)
        if info and info.quantities:
            functions.append(info)
            logger.debug("  %s: %d quantities", info.name, len(info.quantities))

    logger.info("Parsed %d functions with quantities", len(functions))

    # Ingest
    stats = ingest_tdi_functions(args.facility, functions, dry_run=args.dry_run)
    logger.info("Stats: %s", stats)

    return 0


if __name__ == "__main__":
    exit(main())

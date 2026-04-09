"""Export pipeline for creating filtered graph archives.

Queries the live production graph via Cypher, exports nodes and relationships
to CSV, and captures index DDL statements. The output is a directory of CSVs
+ DDL that can be archived and distributed via GHCR.

On the **load side** (``graph load`` / Docker entrypoint), the CSVs are
imported via ``neo4j-admin import`` and indexes created on first start.
This eliminates the fragile dump/load cycle entirely.

Usage::

    from imas_codex.graph.export_rebuild import export_dd_only_csv

    csv_dir = export_dd_only_csv(Path("/output/archive_dir"))
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import click

logger = logging.getLogger(__name__)


# ============================================================================
# DD subgraph specification
# ============================================================================

DD_LABELS: list[str] = [
    "COCOS",
    "DDVersion",
    "GraphMeta",
    "IDS",
    "IdentifierSchema",
    "IMASCoordinateSpec",
    "IMASNode",
    "IMASNodeChange",
    "IMASSemanticCluster",
    "Unit",
]

# Labels with integer IDs (all others use string IDs)
INTEGER_ID_LABELS: set[str] = {"COCOS"}


class RelSpec(NamedTuple):
    """Specification for a relationship type to export."""

    rel_type: str
    start_label: str
    end_label: str
    props: list[str] = []


DD_RELATIONSHIPS: list[RelSpec] = [
    RelSpec("IN_VERSION", "IMASNodeChange", "DDVersion"),
    RelSpec("FOR_IMAS_PATH", "IMASNodeChange", "IMASNode"),
    RelSpec("INTRODUCED_IN", "IMASNode", "DDVersion"),
    RelSpec("INTRODUCED_IN", "IDS", "DDVersion"),
    RelSpec("IN_IDS", "IMASNode", "IDS"),
    RelSpec("HAS_PARENT", "IMASNode", "IMASNode"),
    RelSpec("IN_CLUSTER", "IMASNode", "IMASSemanticCluster"),
    RelSpec("HAS_ERROR", "IMASNode", "IMASNode", ["error_type"]),
    RelSpec("HAS_UNIT", "IMASNode", "Unit"),
    RelSpec("HAS_COORDINATE", "IMASNode", "IMASCoordinateSpec", ["dimension"]),
    RelSpec("HAS_COORDINATE", "IMASNode", "IMASNode", ["dimension"]),
    RelSpec("DEPRECATED_IN", "IMASNode", "DDVersion"),
    RelSpec("COORDINATE_SAME_AS", "IMASNode", "IMASNode", ["dimension"]),
    RelSpec("RENAMED_TO", "IMASNode", "IMASNode"),
    RelSpec("HAS_IDENTIFIER_SCHEMA", "IMASNode", "IdentifierSchema"),
    RelSpec("HAS_PREDECESSOR", "DDVersion", "DDVersion"),
    RelSpec("HAS_SUCCESSOR", "DDVersion", "DDVersion"),
    RelSpec("HAS_COCOS", "DDVersion", "COCOS"),
]


# ============================================================================
# Export configuration
# ============================================================================


@dataclass
class ExportConfig:
    """Configuration for an export+rebuild run."""

    labels: list[str] = field(default_factory=lambda: list(DD_LABELS))
    relationships: list[RelSpec] = field(default_factory=lambda: list(DD_RELATIONSHIPS))
    batch_size: int = 5000
    facility: str | None = None


# ============================================================================
# CSV export
# ============================================================================


def _serialize_value(value: object) -> str:
    """Serialize a Neo4j property value for CSV.

    Strips newlines from string values to prevent multi-line CSV fields
    which neo4j-admin import rejects by default.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        if value and isinstance(value[0], float | int):
            # Float array (embedding) — semicolon-delimited
            return ";".join(f"{v:.8g}" for v in value)
        # String array — semicolon-delimited, newlines stripped
        return ";".join(str(v).replace("\n", " ").replace("\r", "") for v in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value.replace("\n", " ").replace("\r", "")
    return str(value)


def _neo4j_type(value: object) -> str:
    """Infer neo4j-admin CSV type annotation from a Python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    if isinstance(value, list):
        if value and isinstance(value[0], float | int):
            return "float[]"
        return "string[]"
    return "string"


def export_nodes_csv(
    gc: object,
    label: str,
    csv_dir: Path,
    batch_size: int = 5000,
) -> tuple[Path, int]:
    """Export all nodes of a label to CSV using keyset pagination.

    Returns (csv_path, row_count).
    """
    # Discover properties from first batch
    is_integer_id = label in INTEGER_ID_LABELS

    first_batch = gc.query(
        f"MATCH (n:{label}) RETURN n ORDER BY n.id ASC LIMIT $limit",
        limit=batch_size,
    )

    if not first_batch:
        return csv_dir / f"nodes_{label}.csv", 0

    # Extract property keys from first node (stable across nodes of same label)
    sample_node = first_batch[0]["n"]
    prop_keys = sorted(k for k in sample_node.keys() if k != "id")

    # Build header with type annotations
    id_group = f"ID({label})"
    header_parts = [f"id:{id_group}"]

    # Infer types from the sample
    type_map: dict[str, str] = {}
    for key in prop_keys:
        val = sample_node.get(key)
        if val is not None:
            type_map[key] = _neo4j_type(val)
        else:
            type_map[key] = "string"

    for key in prop_keys:
        t = type_map[key]
        header_parts.append(f"{key}:{t}")

    csv_path = csv_dir / f"nodes_{label}.csv"
    row_count = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_parts)

        # Write first batch
        for record in first_batch:
            node = record["n"]
            node_id = str(node["id"]) if not is_integer_id else node["id"]
            row = [node_id] + [_serialize_value(node.get(k)) for k in prop_keys]
            writer.writerow(row)
            row_count += 1

        # Keyset pagination for remaining batches
        last_id = first_batch[-1]["n"]["id"]
        while True:
            batch = gc.query(
                f"MATCH (n:{label}) WHERE n.id > $last_id "
                f"RETURN n ORDER BY n.id ASC LIMIT $limit",
                last_id=last_id,
                limit=batch_size,
            )
            if not batch:
                break
            for record in batch:
                node = record["n"]
                node_id = str(node["id"]) if not is_integer_id else node["id"]
                row = [node_id] + [_serialize_value(node.get(k)) for k in prop_keys]
                writer.writerow(row)
                row_count += 1
            last_id = batch[-1]["n"]["id"]

    logger.info("Exported %d %s nodes to %s", row_count, label, csv_path.name)
    return csv_path, row_count


def export_relationships_csv(
    gc: object,
    spec: RelSpec,
    csv_dir: Path,
    batch_size: int = 10000,
) -> tuple[Path, int]:
    """Export relationships of a specific type to CSV.

    Returns (csv_path, row_count).
    """
    start_group = f"START_ID({spec.start_label})"
    end_group = f"END_ID({spec.end_label})"

    header_parts = [f":{start_group}", f":{end_group}"]
    for prop in spec.props:
        header_parts.append(prop)

    fname = f"rels_{spec.rel_type}_{spec.start_label}_{spec.end_label}.csv"
    csv_path = csv_dir / fname
    row_count = 0

    # Build property return clause
    prop_return = ""
    if spec.props:
        prop_return = ", " + ", ".join(f"r.{p} AS {p}" for p in spec.props)

    # Export with SKIP/LIMIT — relationships don't have stable IDs for keyset
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_parts)

        offset = 0
        while True:
            cypher = (
                f"MATCH (a:{spec.start_label})-[r:{spec.rel_type}]->(b:{spec.end_label}) "
                f"RETURN a.id AS start_id, b.id AS end_id{prop_return} "
                f"SKIP $offset LIMIT $limit"
            )
            batch = gc.query(cypher, offset=offset, limit=batch_size)
            if not batch:
                break
            for record in batch:
                start_id = str(record["start_id"])
                end_id = str(record["end_id"])
                row = [start_id, end_id] + [str(record.get(p, "")) for p in spec.props]
                writer.writerow(row)
                row_count += 1
            offset += len(batch)
            if len(batch) < batch_size:
                break

    logger.info(
        "Exported %d %s rels (%s→%s) to %s",
        row_count,
        spec.rel_type,
        spec.start_label,
        spec.end_label,
        csv_path.name,
    )
    return csv_path, row_count


# ============================================================================
# Index DDL capture
# ============================================================================


def capture_index_ddl(gc: object, labels: list[str]) -> list[str]:
    """Capture CREATE INDEX/CONSTRAINT statements for the given labels.

    Queries the live graph and reconstructs DDL statements.
    Returns a list of Cypher CREATE statements.
    """
    label_set = set(labels)
    statements: list[str] = []

    # Constraints
    constraints = gc.query(
        "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties"
    )
    for c in constraints:
        c_labels = c.get("labelsOrTypes") or []
        if not any(lbl in label_set for lbl in c_labels):
            continue
        lbl = c_labels[0]
        props = c["properties"]
        name = c["name"]
        if c["type"] == "UNIQUENESS":
            prop_str = ", ".join(f"n.{p}" for p in props)
            statements.append(
                f"CREATE CONSTRAINT {name} IF NOT EXISTS "
                f"FOR (n:{lbl}) REQUIRE ({prop_str}) IS UNIQUE"
            )

    # Indexes (non-constraint)
    indexes = gc.query(
        "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, "
        "owningConstraint, options"
    )
    for idx in indexes:
        if idx.get("owningConstraint"):
            continue  # Skip constraint-backed indexes
        idx_labels = idx.get("labelsOrTypes") or []
        if not any(lbl in label_set for lbl in idx_labels):
            continue

        name = idx["name"]
        lbl = idx_labels[0]
        props = idx["properties"]
        idx_type = idx["type"]

        if idx_type == "RANGE":
            prop_str = ", ".join(f"n.{p}" for p in props)
            statements.append(
                f"CREATE INDEX {name} IF NOT EXISTS FOR (n:{lbl}) ON ({prop_str})"
            )
        elif idx_type == "VECTOR":
            options = idx.get("options", {})
            config = options.get("indexConfig", {})
            dim = config.get("vector.dimensions", 256)
            sim = config.get("vector.similarity_function", "COSINE")
            quant = config.get("vector.quantization.enabled", True)
            prop = props[0]
            statements.append(
                f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
                f"FOR (n:{lbl}) ON (n.{prop}) "
                f"OPTIONS {{indexConfig: {{"
                f"`vector.dimensions`: {dim}, "
                f"`vector.similarity_function`: '{sim}', "
                f"`vector.quantization.enabled`: {'true' if quant else 'false'}"
                f"}}}}"
            )
        elif idx_type == "FULLTEXT":
            prop_str = ", ".join(f"n.{p}" for p in props)
            statements.append(
                f"CREATE FULLTEXT INDEX {name} IF NOT EXISTS "
                f"FOR (n:{lbl}) ON EACH [{prop_str}]"
            )

    return statements


# ============================================================================
# neo4j-admin import
# ============================================================================


def _neo4j_image() -> Path:
    """Resolve the Neo4j Apptainer SIF image path."""
    from imas_codex.settings import get_neo4j_image_path

    return get_neo4j_image_path()


def run_import(
    csv_dir: Path,
    data_dir: Path,
    node_files: list[tuple[str, Path]],
    rel_files: list[tuple[RelSpec, Path]],
) -> None:
    """Run neo4j-admin database import full with the exported CSVs."""
    image = _neo4j_image()

    # Build the import command
    cmd = [
        "apptainer",
        "exec",
        "--bind",
        f"{csv_dir}:/import",
        "--bind",
        f"{data_dir}:/data",
        "--writable-tmpfs",
        str(image),
        "neo4j-admin",
        "database",
        "import",
        "full",
        "neo4j",
        "--overwrite-destination=true",
        "--id-type=string",
        "--array-delimiter=;",
        "--multiline-fields=true",
        "--skip-bad-relationships=true",
        "--verbose",
    ]

    # Add node files with label annotations
    for label, csv_path in node_files:
        cmd.append(f"--nodes={label}=/import/{csv_path.name}")

    # Add relationship files
    for spec, csv_path in rel_files:
        cmd.append(f"--relationships={spec.rel_type}=/import/{csv_path.name}")

    click.echo("  Running neo4j-admin import...")
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        logger.error("Import stderr: %s", result.stderr[-2000:])
        raise click.ClickException(
            f"neo4j-admin import failed (rc={result.returncode}):\n"
            f"{result.stderr[-1000:]}"
        )

    click.echo(f"  ✓ Import completed in {elapsed:.1f}s")
    logger.info("Import stdout: %s", result.stdout[-500:])


# ============================================================================
# Post-import index creation
# ============================================================================


def create_indexes_post_import(
    data_dir: Path,
    ddl_statements: list[str],
    bolt_port: int = 7690,
    http_port: int = 7480,
) -> None:
    """Start a temp Neo4j on imported data, create indexes, wait for ONLINE, stop.

    Unlike ``start_temp_neo4j`` which loads a dump first, this starts Neo4j
    directly on the data directory produced by ``neo4j-admin import``.
    """
    import urllib.request

    from imas_codex.graph.temp_neo4j import (
        _cleanup_stale_temp_neo4j,
        stop_temp_neo4j,
        write_temp_neo4j_conf,
    )

    temp_dir = data_dir.parent
    for subdir in ("conf", "logs", "run", "tmp"):
        (temp_dir / subdir).mkdir(exist_ok=True)

    write_temp_neo4j_conf(temp_dir / "conf", bolt_port, http_port)
    _cleanup_stale_temp_neo4j(bolt_port, http_port)

    image = _neo4j_image()

    # Start Neo4j directly on the imported data (no dump load needed)
    click.echo("  Starting temp Neo4j on imported data...")
    neo4j_log = temp_dir / "logs" / "neo4j-index.log"
    log_fh = open(neo4j_log, "w")  # noqa: SIM115
    start_cmd = [
        "apptainer",
        "exec",
        "--bind",
        f"{data_dir}:/data",
        "--bind",
        f"{temp_dir}/logs:/logs",
        "--bind",
        f"{temp_dir}/conf:/var/lib/neo4j/conf",
        "--bind",
        f"{temp_dir}/run:/var/lib/neo4j/run",
        "--bind",
        f"{temp_dir}/tmp:/tmp",
        "--writable-tmpfs",
        str(image),
        "neo4j",
        "console",
    ]
    proc = subprocess.Popen(
        start_cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    # Wait for readiness
    ready = False
    for _ in range(120):
        if proc.poll() is not None:
            log_fh.flush()
            tail = neo4j_log.read_text()[-500:] if neo4j_log.exists() else ""
            raise click.ClickException(
                f"Temp Neo4j exited prematurely (rc={proc.returncode})\n{tail}"
            )
        try:
            urllib.request.urlopen(f"http://localhost:{http_port}/", timeout=2)
            ready = True
            break
        except Exception:
            time.sleep(1)

    if not ready:
        log_fh.flush()
        stop_temp_neo4j(proc)
        log_fh.close()
        tail = neo4j_log.read_text()[-500:] if neo4j_log.exists() else ""
        raise click.ClickException(f"Temp Neo4j did not start within 120s\n{tail}")

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(f"bolt://localhost:{bolt_port}")
        try:
            with driver.session() as session:
                for stmt in ddl_statements:
                    try:
                        session.run(stmt).consume()
                        logger.debug("DDL: %s", stmt[:80])
                    except Exception as e:
                        logger.warning(
                            "DDL failed (may already exist): %s — %s", stmt[:60], e
                        )

                # Wait for all indexes to become ONLINE
                click.echo("  Waiting for indexes to go ONLINE...")
                for _ in range(120):
                    result = list(session.run("SHOW INDEXES YIELD state RETURN state"))
                    states = [r["state"] for r in result]
                    if states and all(s == "ONLINE" for s in states):
                        click.echo(f"  ✓ {len(states)} indexes ONLINE")
                        break
                    time.sleep(1)
                else:
                    not_online = [s for s in states if s != "ONLINE"]
                    logger.warning(
                        "%d indexes not ONLINE after 120s: %s",
                        len(not_online),
                        not_online[:5],
                    )
        finally:
            # Force a checkpoint before closing the driver so all index data
            # and transaction logs are flushed to store files. Without this,
            # neo4j-admin dump refuses to run ("active logical log detected").
            try:
                with driver.session() as session:
                    session.run("CALL db.checkpoint()").consume()
                    logger.debug("Checkpoint completed")
            except Exception as e:
                logger.warning("Checkpoint call failed: %s", e)
            driver.close()

        # Clean shutdown: send SIGTERM to just the Apptainer process (not the
        # whole process group). Apptainer forwards SIGTERM to the JVM which
        # runs its shutdown hook (final checkpoint + close store). Killing the
        # process group with os.killpg can race — Apptainer children may die
        # before the JVM completes its shutdown hook, leaving active txn logs.
        click.echo("  Stopping temp Neo4j (clean shutdown)...")
        proc.terminate()  # SIGTERM to Apptainer only
        try:
            proc.wait(timeout=30)
            logger.debug("Temp Neo4j exited cleanly (rc=%d)", proc.returncode)
        except subprocess.TimeoutExpired:
            logger.warning("Clean shutdown timed out, falling back to SIGKILL")
            stop_temp_neo4j(proc)
    except Exception:
        stop_temp_neo4j(proc)
        raise


# ============================================================================
# Export metadata helpers
# ============================================================================


def _write_export_metadata(
    output_dir: Path,
    node_files: list[tuple[str, Path]],
    rel_files: list[tuple[RelSpec, Path]],
    ddl_statements: list[str],
) -> None:
    """Write import.json, ddl.cypher, and import.sh into the output directory."""
    # import.json — structured metadata for programmatic loaders
    import_manifest = {
        "nodes": [
            {"label": label, "file": csv_path.name} for label, csv_path in node_files
        ],
        "relationships": [
            {"type": spec.rel_type, "file": csv_path.name}
            for spec, csv_path in rel_files
        ],
    }
    (output_dir / "import.json").write_text(json.dumps(import_manifest, indent=2))

    # ddl.cypher — index/constraint DDL for post-import execution
    (output_dir / "ddl.cypher").write_text("\n".join(ddl_statements))

    # import.sh — self-contained import script for Docker/shell use
    # Expects CSV_DIR env var pointing to the csv/ directory and
    # DATA_DIR env var pointing to the target data directory.
    lines = [
        "#!/bin/bash",
        "# Auto-generated neo4j-admin import command",
        "set -e",
        'CSV_DIR="${CSV_DIR:-.}"',
        "",
        "neo4j-admin database import full neo4j \\",
        "  --overwrite-destination=true \\",
        "  --id-type=string \\",
        '  --array-delimiter=";" \\',
        "  --multiline-fields=true \\",
        "  --skip-bad-relationships=true \\",
    ]
    for label, csv_path in node_files:
        lines.append(f'  --nodes={label}="$CSV_DIR/{csv_path.name}" \\')
    for spec, csv_path in rel_files:
        lines.append(f'  --relationships={spec.rel_type}="$CSV_DIR/{csv_path.name}" \\')
    # Remove trailing backslash from last line
    lines[-1] = lines[-1].rstrip(" \\")
    (output_dir / "import.sh").write_text("\n".join(lines) + "\n")


# ============================================================================
# Load-side: import from CSV archive
# ============================================================================


def import_from_csv(
    archive_dir: Path,
    data_dir: Path,
) -> None:
    """Import a CSV-based archive into a Neo4j data directory.

    This is the load-side counterpart to the export functions. It reads
    ``import.json`` for file→label mappings, runs ``neo4j-admin import``,
    then creates indexes from ``ddl.cypher`` via a temp Neo4j instance.

    Args:
        archive_dir: Extracted archive directory containing csv/, import.json,
            and ddl.cypher.
        data_dir: Target Neo4j data directory (e.g. ``profile.data_dir/data``).
    """
    csv_dir = archive_dir / "csv"
    meta_file = archive_dir / "import.json"
    ddl_file = archive_dir / "ddl.cypher"

    if not csv_dir.is_dir() or not meta_file.exists():
        raise click.ClickException(
            f"Not a CSV archive: expected csv/ and import.json in {archive_dir}"
        )

    meta = json.loads(meta_file.read_text())

    node_files = [(e["label"], csv_dir / e["file"]) for e in meta["nodes"]]
    rel_files = [
        (RelSpec(e["type"], "", ""), csv_dir / e["file"]) for e in meta["relationships"]
    ]

    click.echo("  Importing from CSV...")
    run_import(csv_dir, data_dir, node_files, rel_files)

    if ddl_file.exists():
        ddl_statements = [
            s.strip() for s in ddl_file.read_text().splitlines() if s.strip()
        ]
        if ddl_statements:
            click.echo("  Creating indexes...")
            create_indexes_post_import(data_dir, ddl_statements)


# ============================================================================
# Top-level orchestrators
# ============================================================================


def export_dd_only_csv(output_dir: Path) -> Path:
    """Export DD-only subgraph from live graph to CSVs + DDL.

    Queries live Neo4j for all DD labels and relationships, writes CSVs
    and a ``ddl.cypher`` file into *output_dir*. Zero production downtime.

    The output directory is ready to be archived and distributed. The
    load side runs ``import_from_csv()`` to build the database.

    Returns the output directory path.
    """
    from imas_codex.graph.client import GraphClient

    config = ExportConfig()

    click.echo("Export: DD-only variant")
    t_start = time.monotonic()

    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    click.echo("\n  Exporting from live graph...")
    t0 = time.monotonic()

    with GraphClient() as gc:
        node_files: list[tuple[str, Path]] = []
        total_nodes = 0

        for label in config.labels:
            csv_path, count = export_nodes_csv(gc, label, csv_dir, config.batch_size)
            if count > 0:
                node_files.append((label, csv_path))
                total_nodes += count
                click.echo(f"    {label}: {count:,} nodes")

        rel_files: list[tuple[RelSpec, Path]] = []
        total_rels = 0

        for spec in config.relationships:
            csv_path, count = export_relationships_csv(
                gc, spec, csv_dir, batch_size=10000
            )
            if count > 0:
                rel_files.append((spec, csv_path))
                total_rels += count

        # Capture index DDL while still connected
        ddl_statements = capture_index_ddl(gc, config.labels)

    export_time = time.monotonic() - t0
    click.echo(
        f"  ✓ Exported {total_nodes:,} nodes, {total_rels:,} rels in {export_time:.1f}s"
    )

    # Write DDL and import metadata
    _write_export_metadata(output_dir, node_files, rel_files, ddl_statements)
    click.echo(f"    {len(ddl_statements)} DDL statements → ddl.cypher")

    total_time = time.monotonic() - t_start
    csv_size = sum(f.stat().st_size for f in csv_dir.iterdir()) / 1024 / 1024
    click.echo(f"\n  ✓ Export complete: {csv_size:.1f} MB CSV in {total_time:.1f}s")

    return output_dir


def export_facility_csv(facility: str, output_dir: Path) -> Path:
    """Export facility+DD subgraph from live graph to CSVs + DDL.

    Exports all DD nodes plus nodes with ``facility_id = facility`` and
    all relationships between the exported node set.

    Returns the output directory path.
    """
    from imas_codex.graph.client import GraphClient

    click.echo(f"Export: {facility} + DD variant")
    t_start = time.monotonic()

    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    with GraphClient() as gc:
        # Export DD nodes
        click.echo("\n  Exporting DD nodes...")
        config = ExportConfig()
        node_files: list[tuple[str, Path]] = []
        total_nodes = 0

        for label in config.labels:
            csv_path, count = export_nodes_csv(gc, label, csv_dir, config.batch_size)
            if count > 0:
                node_files.append((label, csv_path))
                total_nodes += count

        click.echo(f"    DD: {total_nodes:,} nodes")

        # Export facility nodes
        click.echo(f"\n  Exporting {facility} nodes...")
        facility_labels = gc.query(
            "MATCH (n) WHERE n.facility_id = $facility "
            "WITH labels(n) AS lbls UNWIND lbls AS lbl "
            "RETURN DISTINCT lbl AS label, count(*) AS cnt "
            "ORDER BY cnt DESC",
            facility=facility,
        )

        dd_label_set = set(config.labels)
        for row in facility_labels:
            lbl = row["label"]
            if lbl in dd_label_set:
                continue

            csv_path, count = _export_facility_nodes_csv(
                gc, lbl, facility, csv_dir, config.batch_size
            )
            if count > 0:
                node_files.append((lbl, csv_path))
                total_nodes += count
                click.echo(f"    {lbl}: {count:,} nodes")

        # Export relationships
        click.echo("\n  Exporting relationships...")
        rel_files: list[tuple[RelSpec, Path]] = []
        total_rels = 0

        # DD relationships
        for spec in config.relationships:
            csv_path, count = export_relationships_csv(
                gc, spec, csv_dir, batch_size=10000
            )
            if count > 0:
                rel_files.append((spec, csv_path))
                total_rels += count

        # Facility relationships — discover and export
        fac_rels = gc.query(
            "MATCH (a)-[r]->(b) "
            "WHERE a.facility_id = $facility OR b.facility_id = $facility "
            "WITH type(r) AS rel_type, labels(a)[0] AS start_lbl, "
            "labels(b)[0] AS end_lbl, count(*) AS cnt "
            "RETURN rel_type, start_lbl, end_lbl, cnt "
            "ORDER BY cnt DESC",
            facility=facility,
        )

        exported_rels = {
            (s.rel_type, s.start_label, s.end_label) for s in config.relationships
        }
        for row in fac_rels:
            key = (row["rel_type"], row["start_lbl"], row["end_lbl"])
            if key in exported_rels:
                continue

            spec = RelSpec(row["rel_type"], row["start_lbl"], row["end_lbl"])
            csv_path, count = _export_facility_rels_csv(gc, spec, facility, csv_dir)
            if count > 0:
                rel_files.append((spec, csv_path))
                total_rels += count
                exported_rels.add(key)

        ddl_statements = capture_index_ddl(
            gc,
            config.labels + [r["label"] for r in facility_labels],
        )

    click.echo(f"  ✓ Exported {total_nodes:,} nodes, {total_rels:,} rels")

    # Write DDL and import manifest
    _write_export_metadata(output_dir, node_files, rel_files, ddl_statements)

    total_time = time.monotonic() - t_start
    csv_size = sum(f.stat().st_size for f in csv_dir.iterdir()) / 1024 / 1024
    click.echo(f"\n  ✓ Export complete: {csv_size:.1f} MB CSV in {total_time:.1f}s")

    return output_dir


# ============================================================================
# Facility-specific export helpers
# ============================================================================


def _export_facility_nodes_csv(
    gc: object,
    label: str,
    facility: str,
    csv_dir: Path,
    batch_size: int = 5000,
) -> tuple[Path, int]:
    """Export nodes of a label filtered by facility_id."""
    first_batch = gc.query(
        f"MATCH (n:{label}) WHERE n.facility_id = $facility "
        f"RETURN n ORDER BY n.id ASC LIMIT $limit",
        facility=facility,
        limit=batch_size,
    )

    if not first_batch:
        return csv_dir / f"nodes_{label}_{facility}.csv", 0

    sample_node = first_batch[0]["n"]
    prop_keys = sorted(k for k in sample_node.keys() if k != "id")

    id_group = f"ID({label})"
    header_parts = [f"id:{id_group}"]
    for key in prop_keys:
        val = sample_node.get(key)
        t = _neo4j_type(val) if val is not None else "string"
        header_parts.append(f"{key}:{t}")

    csv_path = csv_dir / f"nodes_{label}_{facility}.csv"
    row_count = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_parts)

        for record in first_batch:
            node = record["n"]
            row = [str(node["id"])] + [_serialize_value(node.get(k)) for k in prop_keys]
            writer.writerow(row)
            row_count += 1

        last_id = first_batch[-1]["n"]["id"]
        while True:
            batch = gc.query(
                f"MATCH (n:{label}) WHERE n.facility_id = $facility "
                f"AND n.id > $last_id "
                f"RETURN n ORDER BY n.id ASC LIMIT $limit",
                facility=facility,
                last_id=last_id,
                limit=batch_size,
            )
            if not batch:
                break
            for record in batch:
                node = record["n"]
                row = [str(node["id"])] + [
                    _serialize_value(node.get(k)) for k in prop_keys
                ]
                writer.writerow(row)
                row_count += 1
            last_id = batch[-1]["n"]["id"]

    return csv_path, row_count


def _export_facility_rels_csv(
    gc: object,
    spec: RelSpec,
    facility: str,
    csv_dir: Path,
    batch_size: int = 10000,
) -> tuple[Path, int]:
    """Export facility-specific relationships."""
    start_group = f"START_ID({spec.start_label})"
    end_group = f"END_ID({spec.end_label})"
    header_parts = [f":{start_group}", f":{end_group}"]

    fname = f"rels_{spec.rel_type}_{spec.start_label}_{spec.end_label}_{facility}.csv"
    csv_path = csv_dir / fname
    row_count = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header_parts)

        offset = 0
        while True:
            batch = gc.query(
                f"MATCH (a:{spec.start_label})-[r:{spec.rel_type}]->(b:{spec.end_label}) "
                f"WHERE a.facility_id = $facility OR b.facility_id = $facility "
                f"RETURN a.id AS start_id, b.id AS end_id "
                f"SKIP $offset LIMIT $limit",
                facility=facility,
                offset=offset,
                limit=batch_size,
            )
            if not batch:
                break
            for record in batch:
                writer.writerow([str(record["start_id"]), str(record["end_id"])])
                row_count += 1
            offset += len(batch)
            if len(batch) < batch_size:
                break

    return csv_path, row_count

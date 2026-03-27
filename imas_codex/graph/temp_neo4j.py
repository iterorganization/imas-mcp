"""Temporary Neo4j instance management for graph filtering.

Provides functions to:
- Start an ephemeral Neo4j instance for dump filtering
- Create per-facility or IMAS-only filtered dumps
- Write temporary Neo4j configuration
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import click

# ============================================================================
# Constants
# ============================================================================

# DD node labels to keep for --dd-only exports
IMAS_DD_LABELS = [
    "COCOS",
    "DDVersion",
    "IDS",
    "IMASNode",
    "IMASCoordinateSpec",
    "IMASSemanticCluster",
    "IdentifierSchema",
    "IMASNodeChange",
    "CoordinateRelationship",
    "ClusterMembership",
    "EmbeddingChange",
    "Unit",
    "PhysicsDomain",
    "SignConvention",
]


# ============================================================================
# Temp Neo4j helpers
# ============================================================================


def _neo4j_image() -> Path:
    """Resolve the Neo4j Apptainer SIF image path."""
    from imas_codex.settings import get_neo4j_image_path

    return get_neo4j_image_path()


def write_temp_neo4j_conf(conf_dir: Path, bolt_port: int, http_port: int) -> Path:
    """Write a neo4j.conf for a temporary filtering instance.

    Disables authentication and binds to non-standard ports to avoid
    conflicts with the production instance.  Memory settings sized for
    DD-only export which runs large DETACH DELETE batches.
    """
    conf_file = conf_dir / "neo4j.conf"
    conf_file.write_text(
        f"""\
dbms.security.auth_enabled=false
server.bolt.listen_address=127.0.0.1:{bolt_port}
server.http.listen_address=127.0.0.1:{http_port}
server.memory.heap.initial_size=1g
server.memory.heap.max_size=4g
server.memory.pagecache.size=512m
dbms.memory.transaction.total.max=4g
"""
    )
    return conf_file


def start_temp_neo4j(
    temp_dir: Path,
    bolt_port: int,
    http_port: int,
) -> tuple[subprocess.Popen, Path]:
    """Load a dump and start a temporary Neo4j instance for filtering.

    Returns the process handle and log file path.  The caller is
    responsible for terminating the process.
    """
    import time
    import urllib.request

    neo4j_image = _neo4j_image()

    # Load dump into temp data dir
    click.echo("  Loading dump into temp instance...")
    load_cmd = [
        "apptainer",
        "exec",
        "--bind",
        f"{temp_dir}/data:/data",
        "--bind",
        f"{temp_dir}/dumps:/dumps",
        "--writable-tmpfs",
        str(neo4j_image),
        "neo4j-admin",
        "database",
        "load",
        "neo4j",
        "--from-path=/dumps",
        "--overwrite-destination=true",
    ]
    result = subprocess.run(load_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(
            f"Failed to load dump into temp instance: {result.stderr}"
        )

    # Write config file — neo4j console does not process NEO4J_ env vars
    conf_dir = temp_dir / "conf"
    conf_dir.mkdir(exist_ok=True)
    write_temp_neo4j_conf(conf_dir, bolt_port, http_port)

    click.echo("  Starting temp Neo4j instance...")
    start_cmd = [
        "apptainer",
        "exec",
        "--bind",
        f"{temp_dir}/data:/data",
        "--bind",
        f"{temp_dir}/logs:/logs",
        "--bind",
        f"{temp_dir}/conf:/var/lib/neo4j/conf",
        "--writable-tmpfs",
        str(neo4j_image),
        "neo4j",
        "console",
    ]
    neo4j_log = temp_dir / "logs" / "neo4j-temp.log"
    log_fh = open(neo4j_log, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        start_cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    ready = False
    for _ in range(180):
        if proc.poll() is not None:
            log_fh.flush()
            tail = neo4j_log.read_text()[-500:] if neo4j_log.exists() else ""
            raise click.ClickException(
                f"Temp Neo4j exited prematurely (rc={proc.returncode})\n"
                f"Last log output:\n{tail}"
            )
        try:
            urllib.request.urlopen(f"http://localhost:{http_port}/", timeout=2)
            ready = True
            break
        except Exception:
            time.sleep(1)

    if not ready:
        log_fh.flush()
        proc.terminate()
        proc.wait(timeout=15)
        log_fh.close()
        tail = neo4j_log.read_text()[-500:] if neo4j_log.exists() else ""
        raise click.ClickException(
            f"Temp Neo4j instance did not start within 180 seconds\n"
            f"Last log output:\n{tail}"
        )

    return proc, neo4j_log


def stop_temp_neo4j(proc: subprocess.Popen) -> None:
    """Terminate a temporary Neo4j process."""
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def dump_temp_neo4j(temp_dir: Path, output_path: Path) -> None:
    """Dump the filtered temp database and move to output path."""
    neo4j_image = _neo4j_image()

    (temp_dir / "dumps" / "neo4j.dump").unlink(missing_ok=True)

    dump_cmd = [
        "apptainer",
        "exec",
        "--bind",
        f"{temp_dir}/data:/data",
        "--bind",
        f"{temp_dir}/dumps:/dumps",
        "--writable-tmpfs",
        str(neo4j_image),
        "neo4j-admin",
        "database",
        "dump",
        "neo4j",
        "--to-path=/dumps",
        "--overwrite-destination=true",
    ]
    result = subprocess.run(dump_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise click.ClickException(f"Failed to dump filtered graph: {result.stderr}")

    filtered_dump = temp_dir / "dumps" / "neo4j.dump"
    if not filtered_dump.exists():
        raise click.ClickException("Filtered dump file not created")

    shutil.move(str(filtered_dump), str(output_path))
    size_mb = output_path.stat().st_size / 1024 / 1024
    click.echo(f"    Filtered dump: {size_mb:.1f} MB")


def create_dd_only_dump(source_dump_path: Path, output_path: Path) -> None:
    """Create an IMAS-only dump by filtering out facility nodes.

    Loads the full dump into a temporary Neo4j instance, deletes all
    nodes that are not IMAS Data Dictionary types, then dumps the
    filtered graph.
    """
    temp_bolt_port = 27687
    temp_http_port = 27474

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir) / "neo4j-temp"
        for subdir in ("data", "logs", "dumps"):
            (temp_dir / subdir).mkdir(parents=True)

        shutil.copy(source_dump_path, temp_dir / "dumps" / "neo4j.dump")

        proc, neo4j_log = start_temp_neo4j(temp_dir, temp_bolt_port, temp_http_port)

        try:
            click.echo("  Filtering graph: keeping only IMAS DD nodes...")

            from neo4j import GraphDatabase

            label_check = " AND ".join(f"NOT n:{label}" for label in IMAS_DD_LABELS)
            driver = GraphDatabase.driver(
                f"bolt://localhost:{temp_bolt_port}",
            )
            with driver.session() as session:
                total_deleted = 0
                while True:
                    result = session.run(
                        f"MATCH (n) WHERE {label_check} "
                        "AND NOT n:GraphMeta "
                        "WITH n LIMIT 5000 "
                        "DETACH DELETE n "
                        "RETURN count(*) AS deleted"
                    )
                    batch_deleted = result.single()["deleted"]
                    if batch_deleted == 0:
                        break
                    total_deleted += batch_deleted
                click.echo(f"    Removed {total_deleted} non-DD nodes")

                # Clean up orphaned Unit nodes left after facility node removal
                orphan_result = session.run(
                    "MATCH (u:Unit) WHERE NOT (u)<-[:HAS_UNIT]-() "
                    "WITH u LIMIT 5000 DETACH DELETE u "
                    "RETURN count(*) AS deleted"
                )
                orphan_deleted = orphan_result.single()["deleted"]
                if orphan_deleted > 0:
                    click.echo(f"    Removed {orphan_deleted} orphaned Unit nodes")

                # Update GraphMeta to reflect dd-only content
                session.run(
                    'MATCH (m:GraphMeta {id: "meta"}) '
                    "SET m.facilities = [], m.imas = true, "
                    "    m.updated_at = datetime().epochMillis"
                )

            driver.close()
        finally:
            stop_temp_neo4j(proc)

        click.echo("  Dumping filtered graph...")
        dump_temp_neo4j(temp_dir, output_path)


def create_facility_dump(
    source_dump_path: Path, facility: str, output_path: Path
) -> None:
    """Create a per-facility dump by filtering a full graph dump.

    Loads the full dump into a temporary Neo4j instance, deletes nodes
    belonging to other facilities and orphaned non-DD nodes, then dumps
    the filtered graph.

    Args:
        source_dump_path: Path to the full ``neo4j.dump`` file.
        facility: Facility ID to keep (e.g. ``"tcv"``).
        output_path: Where to write the filtered dump file.
    """
    temp_bolt_port = 27687
    temp_http_port = 27474

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir) / "neo4j-temp"
        for subdir in ("data", "logs", "dumps"):
            (temp_dir / subdir).mkdir(parents=True)

        shutil.copy(source_dump_path, temp_dir / "dumps" / "neo4j.dump")

        proc, neo4j_log = start_temp_neo4j(temp_dir, temp_bolt_port, temp_http_port)

        try:
            click.echo(f"  Filtering graph: keeping facility={facility}...")

            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                f"bolt://localhost:{temp_bolt_port}",
            )
            with driver.session() as session:
                total_deleted = 0
                while True:
                    result = session.run(
                        "MATCH (n) "
                        "WHERE n.facility_id IS NOT NULL "
                        "AND n.facility_id <> $facility "
                        "WITH n LIMIT 10000 "
                        "DETACH DELETE n "
                        "RETURN count(*) AS deleted",
                        facility=facility,
                    )
                    batch_deleted = result.single()["deleted"]
                    if batch_deleted == 0:
                        break
                    total_deleted += batch_deleted
                click.echo(f"    Removed {total_deleted} non-{facility} nodes")

                result = session.run(
                    "MATCH (n) WHERE NOT (n)--() "
                    "AND NOT n:IMASNode AND NOT n:DDVersion AND NOT n:Unit "
                    "AND NOT n:IMASCoordinateSpec AND NOT n:PhysicsDomain "
                    "AND NOT n:IMASSemanticCluster "
                    "AND NOT n:GraphMeta "
                    "DELETE n "
                    "RETURN count(*) AS deleted"
                )
                deleted_orphans = result.single()["deleted"]
                click.echo(f"    Removed {deleted_orphans} orphan nodes")

                # Update GraphMeta to reflect the kept facility
                session.run(
                    'MATCH (m:GraphMeta {id: "meta"}) '
                    "SET m.facilities = [$facility], "
                    "    m.updated_at = datetime().epochMillis",
                    facility=facility,
                )

            driver.close()
        finally:
            stop_temp_neo4j(proc)

        click.echo("  Dumping filtered graph...")
        dump_temp_neo4j(temp_dir, output_path)

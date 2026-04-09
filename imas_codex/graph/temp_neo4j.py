"""Temporary Neo4j instance management for graph filtering.

Provides functions to:
- Start an ephemeral Neo4j instance for dump filtering
- Create per-facility or IMAS-only filtered dumps
- Write temporary Neo4j configuration

When running on a service/login node with cgroup memory limits,
the temp Neo4j is dispatched to a SLURM compute node via ``srun``
to avoid OOM kills from per-user memory constraints.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path

import click

logger = logging.getLogger(__name__)

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
    conflicts with the production instance.  Memory is kept very low
    because the temp instance runs inside a per-user cgroup with
    limited headroom.  Filtering uses ``CALL {} IN TRANSACTIONS``
    which commits in server-side batches and needs minimal heap.
    """
    conf_file = conf_dir / "neo4j.conf"
    conf_file.write_text(
        f"""\
dbms.security.auth_enabled=false
server.bolt.listen_address=127.0.0.1:{bolt_port}
server.http.listen_address=127.0.0.1:{http_port}
server.memory.heap.initial_size=128m
server.memory.heap.max_size=512m
server.memory.pagecache.size=128m
dbms.memory.transaction.total.max=512m
server.jvm.additional=-XX:MaxDirectMemorySize=256m
server.jvm.additional=-XX:MaxMetaspaceSize=128m
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

    All Neo4j write paths are bind-mounted to host directories so that
    the Apptainer ``--writable-tmpfs`` overlay (often capped at 64 MB
    by ``sessiondir max size``) is only used for trivial metadata.
    """
    import urllib.request

    neo4j_image = _neo4j_image()

    # Ensure all Neo4j write-target directories exist on host
    for subdir in ("run", "tmp"):
        (temp_dir / subdir).mkdir(exist_ok=True)

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
        "--bind",
        f"{temp_dir}/run:/var/lib/neo4j/run",
        "--bind",
        f"{temp_dir}/tmp:/tmp",
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
        stop_temp_neo4j(proc)
        log_fh.close()
        tail = neo4j_log.read_text()[-500:] if neo4j_log.exists() else ""
        raise click.ClickException(
            f"Temp Neo4j instance did not start within 180 seconds\n"
            f"Last log output:\n{tail}"
        )

    return proc, neo4j_log


def _cleanup_stale_temp_neo4j(bolt_port: int, http_port: int) -> None:
    """Kill any stale processes bound to the temp Neo4j ports.

    Previous temp instances may leave orphaned Java processes when
    apptainer exits but the JVM keeps running in the background.
    """
    for port in (bolt_port, http_port):
        try:
            result = subprocess.run(
                ["ss", "-tlnp", f"sport = :{port}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "pid=" in line:
                    pid_str = line.split("pid=")[1].split(",")[0]
                    pid = int(pid_str)
                    click.echo(f"  Killing stale process on port {port} (PID {pid})")
                    os.kill(pid, signal.SIGKILL)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            pass

    # Brief wait for port release
    time.sleep(2)


def stop_temp_neo4j(proc: subprocess.Popen) -> None:
    """Terminate a temporary Neo4j process and its entire process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait(timeout=10)


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


def _filter_to_dd_only(bolt_port: int, neo4j_log: Path) -> None:
    """Delete all non-DD nodes from the temp Neo4j instance.

    Uses ``CALL {} IN TRANSACTIONS`` for server-side batched commits
    so the JVM never accumulates a large transaction in memory.
    """
    from neo4j import GraphDatabase

    click.echo("  Filtering graph: keeping only IMAS DD nodes...")

    label_check = " AND ".join(f"NOT n:{label}" for label in IMAS_DD_LABELS)
    driver = GraphDatabase.driver(f"bolt://localhost:{bolt_port}")

    try:
        with driver.session() as session:
            # Count nodes to delete for progress reporting
            result = session.run(
                f"MATCH (n) WHERE {label_check} AND NOT n:GraphMeta "
                "RETURN count(n) AS total"
            )
            total = result.single()["total"]
            click.echo(f"    {total:,} non-DD nodes to remove")

            # Server-side batched deletion — single query, Neo4j manages commits
            result = session.run(
                f"MATCH (n) WHERE {label_check} AND NOT n:GraphMeta "
                "CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 1000 ROWS "
                "RETURN count(*) AS deleted"
            )
            deleted = result.single()["deleted"]
            click.echo(f"    Removed {deleted:,} non-DD nodes")

            # Clean up orphaned Unit nodes left after facility node removal
            result = session.run(
                "MATCH (u:Unit) WHERE NOT (u)<-[:HAS_UNIT]-() "
                "CALL { WITH u DETACH DELETE u } IN TRANSACTIONS OF 1000 ROWS "
                "RETURN count(*) AS deleted"
            )
            orphan_deleted = result.single()["deleted"]
            if orphan_deleted > 0:
                click.echo(f"    Removed {orphan_deleted:,} orphaned Unit nodes")

            # Update GraphMeta to reflect dd-only content
            session.run(
                'MATCH (m:GraphMeta {id: "meta"}) '
                "SET m.facilities = [], m.imas = true, "
                "    m.updated_at = datetime().epochMillis"
            )
    except Exception:
        # Capture temp Neo4j logs on failure for diagnostics
        if neo4j_log.exists():
            tail = neo4j_log.read_text()[-1000:]
            click.echo(f"  Temp Neo4j log tail:\n{tail}")
        raise
    finally:
        driver.close()


def _filter_to_facility(bolt_port: int, facility: str, neo4j_log: Path) -> None:
    """Delete nodes not belonging to *facility* from the temp instance.

    Uses ``CALL {} IN TRANSACTIONS`` for server-side batched commits.
    """
    from neo4j import GraphDatabase

    click.echo(f"  Filtering graph: keeping facility={facility}...")

    driver = GraphDatabase.driver(f"bolt://localhost:{bolt_port}")

    try:
        with driver.session() as session:
            # Remove nodes belonging to other facilities
            result = session.run(
                "MATCH (n) "
                "WHERE n.facility_id IS NOT NULL "
                "AND n.facility_id <> $facility "
                "CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 1000 ROWS "
                "RETURN count(*) AS deleted",
                facility=facility,
            )
            deleted = result.single()["deleted"]
            click.echo(f"    Removed {deleted:,} non-{facility} nodes")

            # Remove orphaned nodes (no relationships, not core DD types)
            result = session.run(
                "MATCH (n) WHERE NOT (n)--() "
                "AND NOT n:IMASNode AND NOT n:DDVersion AND NOT n:Unit "
                "AND NOT n:IMASCoordinateSpec AND NOT n:PhysicsDomain "
                "AND NOT n:IMASSemanticCluster "
                "AND NOT n:GraphMeta "
                "CALL { WITH n DELETE n } IN TRANSACTIONS OF 1000 ROWS "
                "RETURN count(*) AS deleted"
            )
            orphans = result.single()["deleted"]
            click.echo(f"    Removed {orphans:,} orphan nodes")

            # Update GraphMeta to reflect the kept facility
            session.run(
                'MATCH (m:GraphMeta {id: "meta"}) '
                "SET m.facilities = [$facility], "
                "    m.updated_at = datetime().epochMillis",
                facility=facility,
            )
    except Exception:
        if neo4j_log.exists():
            tail = neo4j_log.read_text()[-1000:]
            click.echo(f"  Temp Neo4j log tail:\n{tail}")
        raise
    finally:
        driver.close()


def _should_use_slurm() -> bool:
    """Return True if temp Neo4j should run via SLURM.

    On HPC service/login nodes, per-user cgroup memory limits can OOM-kill
    a temp Neo4j loading a multi-GB dump.  SLURM compute nodes have
    dedicated memory allocations that avoid this.
    """
    if os.environ.get("SLURM_JOB_ID"):
        return False  # Already on a compute node
    if not shutil.which("srun"):
        return False  # Not an HPC environment
    return True


def _slurm_partition() -> str:
    """Resolve the SLURM partition to use for temp Neo4j jobs.

    Uses the general (non-GPU) partition from compute config, falling
    back to ``"rigel"`` which is the standard CPU partition on ITER HPC.
    """
    try:
        from imas_codex.cli.services import _general_partition_name

        return _general_partition_name()
    except Exception:
        return "rigel"


def _build_filter_script(
    *,
    neo4j_image: str,
    source_dump: str,
    output_dump: str,
    mode: str,
    facility: str = "",
) -> str:
    """Build a bash script for the temp Neo4j filter lifecycle.

    The script runs entirely on a compute node: load → start →
    filter via cypher-shell → stop → dump.

    Args:
        neo4j_image: Path to the Neo4j Apptainer SIF image.
        source_dump: Path to the full neo4j.dump on shared GPFS.
        output_dump: Where to write the filtered dump on shared GPFS.
        mode: Either ``"dd-only"`` or ``"facility"``.
        facility: Facility ID (required when mode is ``"facility"``).
    """
    # Build the Cypher filter commands based on mode
    if mode == "dd-only":
        # Property-based deletion — self-maintaining.  Every facility
        # node has ``facility_id`` per schema, so new labels are
        # automatically covered without updating a hard-coded list.
        dd_keep_labels = " OR ".join(f"n:{lbl}" for lbl in IMAS_DD_LABELS)
        filter_cypher = textwrap.dedent(f"""\
            # Delete all facility-owned nodes (every facility node has facility_id)
            echo "    Deleting facility nodes..."
            $CYPHER "MATCH (n) WHERE n.facility_id IS NOT NULL CALL {{ WITH n DETACH DELETE n }} IN TRANSACTIONS OF 50000 ROWS RETURN count(*) AS deleted;"

            # Delete Facility nodes themselves (they use id, not facility_id)
            echo "    Deleting Facility nodes..."
            $CYPHER "MATCH (n:Facility) CALL {{ WITH n DETACH DELETE n }} IN TRANSACTIONS RETURN count(*) AS deleted;"

            # Clean orphan nodes that lost all relationships and aren't DD types
            echo "    Cleaning orphan nodes..."
            $CYPHER "MATCH (n) WHERE NOT (n)--() AND NOT ({dd_keep_labels}) AND NOT n:GraphMeta CALL {{ WITH n DELETE n }} IN TRANSACTIONS OF 50000 ROWS RETURN count(*) AS deleted;"

            # Orphan Unit cleanup
            echo "    Cleaning orphaned Unit nodes..."
            $CYPHER "MATCH (u:Unit) WHERE NOT (u)<-[:HAS_UNIT]-() CALL {{ WITH u DETACH DELETE u }} IN TRANSACTIONS OF 10000 ROWS RETURN count(*) AS deleted;"

            # Update GraphMeta
            $CYPHER "MATCH (m:GraphMeta {{id: 'meta'}}) SET m.facilities = [], m.imas = true, m.updated_at = datetime().epochMillis;"
        """)
    else:
        filter_cypher = textwrap.dedent(f"""\
            # Delete nodes from other facilities
            echo "    Deleting non-{facility} nodes..."
            $CYPHER "MATCH (n) WHERE n.facility_id IS NOT NULL AND n.facility_id <> '{facility}' CALL {{ WITH n DETACH DELETE n }} IN TRANSACTIONS OF 50000 ROWS RETURN count(*) AS deleted;"

            # Remove orphaned nodes (no relationships, not DD types)
            echo "    Cleaning orphaned nodes..."
            $CYPHER "MATCH (n) WHERE NOT (n)--() AND NOT n:IMASNode AND NOT n:DDVersion AND NOT n:Unit AND NOT n:IMASCoordinateSpec AND NOT n:PhysicsDomain AND NOT n:IMASSemanticCluster AND NOT n:GraphMeta CALL {{ WITH n DELETE n }} IN TRANSACTIONS OF 50000 ROWS RETURN count(*) AS deleted;"

            # Update GraphMeta
            $CYPHER "MATCH (m:GraphMeta {{id: 'meta'}}) SET m.facilities = ['{facility}'], m.updated_at = datetime().epochMillis;"
        """)

    return textwrap.dedent(f"""\
        #!/bin/bash
        set -euo pipefail

        NEO4J_IMAGE="{neo4j_image}"
        SOURCE_DUMP="{source_dump}"
        OUTPUT_DUMP="{output_dump}"
        BOLT_PORT=27687
        HTTP_PORT=27474
        NEO4J_PID=""

        # Use GPFS temp dir (accessible across nodes, no size cap)
        TEMP_DIR=$(mktemp -d "${{HOME}}/.local/share/imas-codex/.neo4j-filter-XXXXXX")

        cleanup() {{
            if [ -n "$NEO4J_PID" ]; then
                kill "$NEO4J_PID" 2>/dev/null || true
                wait "$NEO4J_PID" 2>/dev/null || true
            fi
            rm -rf "$TEMP_DIR"
        }}
        trap cleanup EXIT

        mkdir -p "$TEMP_DIR"/{{data,logs,dumps,conf,run,tmp}}

        echo "  Loading dump into temp instance..."
        # Bind-mount source dump directly into container (symlinks don't
        # resolve inside Apptainer when the target is outside bind paths).
        apptainer exec \\
            --bind "$TEMP_DIR/data:/data" \\
            --bind "$SOURCE_DUMP:/dumps/neo4j.dump:ro" \\
            --writable-tmpfs \\
            "$NEO4J_IMAGE" \\
            neo4j-admin database load neo4j --from-path=/dumps --overwrite-destination=true

        # Write minimal config
        cat > "$TEMP_DIR/conf/neo4j.conf" <<'CONF'
dbms.security.auth_enabled=false
server.bolt.listen_address=127.0.0.1:27687
server.http.listen_address=127.0.0.1:27474
server.memory.heap.initial_size=1g
server.memory.heap.max_size=4g
server.memory.pagecache.size=2g
dbms.memory.transaction.total.max=4g
CONF

        echo "  Starting temp Neo4j instance..."
        apptainer exec \\
            --bind "$TEMP_DIR/data:/data" \\
            --bind "$TEMP_DIR/logs:/logs" \\
            --bind "$TEMP_DIR/conf:/var/lib/neo4j/conf" \\
            --bind "$TEMP_DIR/run:/var/lib/neo4j/run" \\
            --bind "$TEMP_DIR/tmp:/tmp" \\
            --writable-tmpfs \\
            "$NEO4J_IMAGE" \\
            neo4j console > "$TEMP_DIR/logs/neo4j.log" 2>&1 &
        NEO4J_PID=$!

        # Wait for Bolt readiness (not just HTTP) via cypher-shell probe
        cypher_probe() {{
            apptainer exec --writable-tmpfs "$NEO4J_IMAGE" \\
                cypher-shell -a "bolt://localhost:$BOLT_PORT" "RETURN 1;" \\
                > /dev/null 2>&1
        }}

        READY=0
        for i in $(seq 1 180); do
            if ! kill -0 $NEO4J_PID 2>/dev/null; then
                echo "ERROR: Temp Neo4j exited prematurely"
                tail -20 "$TEMP_DIR/logs/neo4j.log" || true
                exit 1
            fi
            if cypher_probe; then
                READY=1
                break
            fi
            sleep 1
        done
        if [ $READY -eq 0 ]; then
            echo "ERROR: Temp Neo4j did not start in 180s"
            tail -20 "$TEMP_DIR/logs/neo4j.log" || true
            exit 1
        fi

        echo "  Filtering graph ({mode})..."
        CYPHER="apptainer exec --writable-tmpfs $NEO4J_IMAGE cypher-shell -a bolt://localhost:$BOLT_PORT"

        {filter_cypher}

        # Stop temp Neo4j gracefully
        kill $NEO4J_PID 2>/dev/null || true
        wait $NEO4J_PID 2>/dev/null || true
        NEO4J_PID=""

        # Dump filtered graph
        echo "  Dumping filtered graph..."
        apptainer exec \\
            --bind "$TEMP_DIR/data:/data" \\
            --bind "$TEMP_DIR/dumps:/dumps" \\
            --writable-tmpfs \\
            "$NEO4J_IMAGE" \\
            neo4j-admin database dump neo4j --to-path=/dumps --overwrite-destination=true

        mv "$TEMP_DIR/dumps/neo4j.dump" "$OUTPUT_DUMP"
        SIZE=$(du -h "$OUTPUT_DUMP" | cut -f1)
        echo "    Filtered dump: $SIZE"
        echo "FILTER_SUCCESS"
    """)


def _run_filter_via_slurm(
    source_dump_path: Path,
    output_path: Path,
    *,
    mode: str,
    facility: str = "",
) -> None:
    """Run the temp Neo4j filter lifecycle on a SLURM compute node.

    Submits the entire load → start → filter → dump pipeline via
    ``srun`` with dedicated memory, avoiding per-user cgroup limits
    on service/login nodes.

    The script is passed on stdin (``bash -s``) to avoid temp-file
    visibility issues between service and compute nodes.
    """
    neo4j_image = str(_neo4j_image())
    partition = _slurm_partition()

    script = _build_filter_script(
        neo4j_image=neo4j_image,
        source_dump=str(source_dump_path),
        output_dump=str(output_path),
        mode=mode,
        facility=facility,
    )

    click.echo(f"  Submitting filter job to SLURM ({partition})...")
    result = subprocess.run(
        [
            "srun",
            f"--partition={partition}",
            "--mem=16G",
            "--time=01:00:00",
            "--job-name=neo4j-filter",
            "bash",
            "-s",
        ],
        input=script,
        capture_output=True,
        text=True,
        timeout=4200,  # headroom over SLURM 1h walltime
    )
    if result.returncode != 0 or "FILTER_SUCCESS" not in result.stdout:
        # Print both stdout and stderr for diagnostics
        click.echo(result.stdout[-2000:] if result.stdout else "")
        raise click.ClickException(
            f"SLURM filter job failed (rc={result.returncode})\n"
            f"{result.stderr[-1000:] if result.stderr else ''}"
        )
    # Echo the output (progress messages from the script)
    for line in result.stdout.splitlines():
        if line.startswith("  ") or line.startswith("    "):
            click.echo(line)

    if not output_path.exists():
        raise click.ClickException(
            f"Filter script succeeded but output not found: {output_path}"
        )
    size_mb = output_path.stat().st_size / 1024 / 1024
    click.echo(f"    Filtered dump: {size_mb:.1f} MB")


def create_dd_only_dump(source_dump_path: Path, output_path: Path) -> None:
    """Create an IMAS-only dump by filtering out facility nodes.

    On HPC systems with SLURM, the filtering runs on a compute node
    to avoid per-user cgroup memory limits.  Otherwise falls back to
    a local temporary Neo4j instance.
    """
    if _should_use_slurm():
        _run_filter_via_slurm(source_dump_path, output_path, mode="dd-only")
        return

    # Local fallback (compute nodes, CI, non-HPC)
    temp_bolt_port = 27687
    temp_http_port = 27474

    _cleanup_stale_temp_neo4j(temp_bolt_port, temp_http_port)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir) / "neo4j-temp"
        for subdir in ("data", "logs", "dumps"):
            (temp_dir / subdir).mkdir(parents=True)

        shutil.copy(source_dump_path, temp_dir / "dumps" / "neo4j.dump")

        proc, neo4j_log = start_temp_neo4j(temp_dir, temp_bolt_port, temp_http_port)

        try:
            _filter_to_dd_only(temp_bolt_port, neo4j_log)
        finally:
            stop_temp_neo4j(proc)

        click.echo("  Dumping filtered graph...")
        dump_temp_neo4j(temp_dir, output_path)


def create_facility_dump(
    source_dump_path: Path, facility: str, output_path: Path
) -> None:
    """Create a per-facility dump by filtering a full graph dump.

    On HPC systems with SLURM, the filtering runs on a compute node.
    Otherwise falls back to a local temporary Neo4j instance.

    Args:
        source_dump_path: Path to the full ``neo4j.dump`` file.
        facility: Facility ID to keep (e.g. ``"tcv"``).
        output_path: Where to write the filtered dump file.
    """
    if _should_use_slurm():
        _run_filter_via_slurm(
            source_dump_path, output_path, mode="facility", facility=facility
        )
        return

    # Local fallback
    temp_bolt_port = 27687
    temp_http_port = 27474

    _cleanup_stale_temp_neo4j(temp_bolt_port, temp_http_port)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir) / "neo4j-temp"
        for subdir in ("data", "logs", "dumps"):
            (temp_dir / subdir).mkdir(parents=True)

        shutil.copy(source_dump_path, temp_dir / "dumps" / "neo4j.dump")

        proc, neo4j_log = start_temp_neo4j(temp_dir, temp_bolt_port, temp_http_port)

        try:
            _filter_to_facility(temp_bolt_port, facility, neo4j_log)
        finally:
            stop_temp_neo4j(proc)

        click.echo("  Dumping filtered graph...")
        dump_temp_neo4j(temp_dir, output_path)

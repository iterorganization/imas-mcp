"""Shared SLURM allocation and service management infrastructure.

Extracted from ``serve.py`` so that both ``graph_cli.py`` (Neo4j
management) and ``serve.py`` (embed/LLM deployment) can share
the same service lifecycle primitives without cross-module
private-function imports.

Sections:
    - Constants
    - Port helpers
    - Scheduler detection
    - Facility compute config
    - SSH helpers
    - SLURM allocation management
    - Service management (start/stop/check on compute nodes)
    - Neo4j-specific service helpers
    - Health checks
    - Log tailing
"""

from __future__ import annotations

import logging
import os
import subprocess
import time

import click

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────

_ALLOC_JOB = "imas-codex-services"
_SERVICES_DIR = "$HOME/.local/share/imas-codex/services"
_DEFAULT_GPUS = 4
_PROJECT = "$HOME/Code/imas-codex"


# ── Port helpers ─────────────────────────────────────────────────────────


def _embed_port() -> int:
    """Return the configured embed server port."""
    from imas_codex.settings import get_embed_server_port

    return get_embed_server_port()


def _graph_port() -> int:
    """Return the configured Neo4j bolt port."""
    from imas_codex.graph.profiles import resolve_neo4j

    return resolve_neo4j(auto_tunnel=False).bolt_port


def _graph_http_port() -> int:
    """Return the configured Neo4j HTTP port."""
    from imas_codex.graph.profiles import resolve_neo4j

    return resolve_neo4j(auto_tunnel=False).http_port


def _llm_port() -> int:
    """Return the configured LLM proxy port."""
    from imas_codex.settings import get_llm_proxy_port

    return get_llm_proxy_port()


# ── Scheduler detection ─────────────────────────────────────────────────


def _is_compute_target() -> bool:
    """True when embed scheduler is 'slurm' (deploy via SLURM)."""
    from imas_codex.settings import get_embed_scheduler

    return get_embed_scheduler() == "slurm"


def _is_graph_compute_target() -> bool:
    """True when graph runs on a SLURM compute node."""
    from imas_codex.graph.profiles import get_graph_location
    from imas_codex.remote.locations import resolve_location

    location = get_graph_location()
    if location == "local":
        return False
    return resolve_location(location).is_compute


# ── Facility compute config ─────────────────────────────────────────────


def _compute_config() -> dict:
    """Load compute config from the facility's private YAML.

    Resolves the facility from ``[embedding].location`` (e.g.
    ``"titan"`` → ``"iter"``).
    """
    from imas_codex.discovery.base.facility import get_facility_infrastructure
    from imas_codex.remote.locations import resolve_location
    from imas_codex.settings import get_embedding_location

    location = get_embedding_location()
    if location == "local":
        raise click.ClickException(
            "Embedding location is 'local' — no compute config available."
        )
    info = resolve_location(location)
    infra = get_facility_infrastructure(info.facility)
    compute = infra.get("compute", {})
    if not compute:
        raise click.ClickException(
            f"No compute config in {info.facility}_private.yaml.\n"
            "Add via: update_facility_infrastructure()"
        )
    return compute


def _gpu_entry() -> dict:
    """Get the GPU resource entry marked for embed_server."""
    compute = _compute_config()
    for gpu in compute.get("gpus", []):
        if gpu.get("current_use") == "embed_server":
            return gpu
    raise click.ClickException(
        "No GPU with current_use=embed_server in facility compute config."
    )


def _gpu_partition() -> dict:
    """Get the first scheduler partition with GPUs."""
    compute = _compute_config()
    for p in compute.get("scheduler", {}).get("partitions", []):
        if p.get("gpus_per_node") or p.get("gpu_type"):
            return p
    raise click.ClickException("No GPU partition found in facility compute config.")


# ── SSH helpers ──────────────────────────────────────────────────────────


def _facility_ssh() -> str | None:
    """SSH host for reaching the facility, or None if local."""
    from imas_codex.remote.locations import is_location_local, resolve_location
    from imas_codex.settings import get_embedding_location

    location = get_embedding_location()
    if location == "local":
        return None
    if is_location_local(location):
        return None
    return resolve_location(location).ssh_host


def _run_remote(cmd: str, timeout: int = 30, check: bool = False) -> str:
    """Run a command on the facility login node (locally if already there)."""
    from imas_codex.remote.executor import run_command

    return run_command(cmd, ssh_host=_facility_ssh(), timeout=timeout, check=check)


def _run_on_node(node: str, cmd: str, timeout: int = 30) -> str:
    """Run a command on a compute node (via SSH from login node).

    Uses base64 encoding to avoid nested shell quoting issues.
    """
    import base64

    cmd_b64 = base64.b64encode(cmd.encode()).decode()
    return _run_remote(
        f'ssh -o StrictHostKeyChecking=no {node} "echo {cmd_b64} | base64 -d | bash"',
        timeout=timeout,
    )


# ── SLURM allocation management ─────────────────────────────────────────


def _get_allocation() -> dict | None:
    """Get the active imas-codex-services SLURM allocation.

    Returns dict with job_id, state, node, gres, time — or None.
    """
    try:
        out = _run_remote(
            f'squeue -n {_ALLOC_JOB} -u "$USER" --format="%A|%T|%M|%N|%b" --noheader'
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    for line in out.strip().split("\n"):
        line = line.strip()
        if not line or line == "(no output)":
            continue
        parts = line.split("|")
        if len(parts) >= 5:
            return {
                "job_id": parts[0].strip(),
                "state": parts[1].strip(),
                "time": parts[2].strip(),
                "node": parts[3].strip(),
                "gres": parts[4].strip(),
            }
    return None


def _get_allocation_fallback() -> dict | None:
    """Fallback allocation from allocation.log when SLURM is unavailable.

    Returns a synthetic allocation dict if the last-known compute node
    is still reachable, or None.
    """
    from imas_codex.remote.locations import _host_reachable, _read_allocation_host

    host = _read_allocation_host()
    if not host:
        return None
    if _host_reachable(host):
        return {
            "job_id": "unknown",
            "state": "RUNNING",
            "time": "unknown",
            "node": host,
            "gres": "unknown",
            "_fallback": True,
        }
    return None


def _ensure_allocation(gpus: int = _DEFAULT_GPUS) -> dict:
    """Ensure a SLURM allocation exists, creating one if needed.

    The allocation is a ``sleep infinity`` batch job that reserves
    compute resources.  Services are managed separately via SSH.

    When SLURM is unavailable, falls back to the last-known compute
    node from allocation.log if it is still reachable.

    Returns the allocation dict (job_id, node, etc.).
    """
    alloc = _get_allocation()
    if alloc and alloc["state"] == "RUNNING":
        return alloc

    # SLURM returned no running allocation — check if the last-known
    # compute node is still reachable (services outlive SLURM outages).
    fallback = _get_allocation_fallback()
    if fallback:
        click.echo(
            f"SLURM unavailable but compute node {fallback['node']} is reachable"
        )
        return fallback

    # Cancel any PENDING allocation (stuck in queue)
    if alloc:
        _run_remote(f"scancel {alloc['job_id']}", check=False)
        time.sleep(1)

    _submit_allocation(gpus)

    # Wait for RUNNING state
    click.echo("Waiting for allocation to start...")
    deadline = time.time() + 120
    while time.time() < deadline:
        time.sleep(3)
        alloc = _get_allocation()
        if alloc and alloc["state"] == "RUNNING":
            click.echo(
                f"  Allocation ready: job {alloc['job_id']} on {alloc['node']} "
                f"({alloc['gres']})"
            )
            return alloc
        remaining = int(deadline - time.time())
        if remaining > 0 and remaining % 15 < 5:
            state = alloc["state"] if alloc else "UNKNOWN"
            click.echo(f"  State: {state} ({remaining}s remaining)")

    raise click.ClickException(
        "Allocation did not start within 120s. Check: squeue -u $USER"
    )


def _submit_allocation(gpus: int) -> None:
    """Submit a new allocation job (sleep infinity with cleanup trap)."""
    import base64

    gpu_e = _gpu_entry()
    partition = _gpu_partition()
    host = gpu_e["location"]
    partition_name = partition["name"]
    cpus = gpus * 4

    # Resolve remote $HOME for SBATCH directives (SLURM does NOT expand
    # shell variables in #SBATCH lines — only inside the script body).
    remote_home = _run_remote("echo $HOME", timeout=10).strip()
    services_dir_abs = f"{remote_home}/.local/share/imas-codex/services"

    script = (
        "#!/bin/bash\n"
        f"#SBATCH --partition={partition_name}\n"
        f"#SBATCH --gres=gpu:{gpus}\n"
        f"#SBATCH --cpus-per-task={cpus}\n"
        "#SBATCH --mem=64G\n"
        "#SBATCH --time=UNLIMITED\n"
        f"#SBATCH --job-name={_ALLOC_JOB}\n"
        f"#SBATCH --nodelist={host}\n"
        f"#SBATCH --output={services_dir_abs}/allocation.log\n"
        "\n"
        f"SERVICES_DIR={_SERVICES_DIR}\n"
        'mkdir -p "$SERVICES_DIR"\n'
        "\n"
        "cleanup() {\n"
        '    echo "Releasing allocation at $(date), stopping services..."\n'
        '    for pidfile in "$SERVICES_DIR"/*.pid; do\n'
        '        [ -f "$pidfile" ] || continue\n'
        '        pid=$(cat "$pidfile")\n'
        '        name=$(basename "$pidfile" .pid)\n'
        '        if kill -0 "$pid" 2>/dev/null; then\n'
        '            echo "Stopping $name (PID $pid)..."\n'
        '            pkill -TERM -P "$pid" 2>/dev/null || true\n'
        '            kill -TERM "$pid" 2>/dev/null || true\n'
        "            sleep 2\n"
        '            if kill -0 "$pid" 2>/dev/null; then\n'
        '                pkill -KILL -P "$pid" 2>/dev/null || true\n'
        '                kill -KILL "$pid" 2>/dev/null || true\n'
        "            fi\n"
        "        fi\n"
        '        rm -f "$pidfile"\n'
        "    done\n"
        '    echo "All services stopped"\n'
        "}\n"
        "trap cleanup EXIT TERM INT\n"
        "\n"
        'echo "Allocation ready on $(hostname) at $(date)"\n'
        'echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"\n'
        "\n"
        "sleep infinity\n"
    )

    # Cancel conflicting legacy jobs (standalone embed, etc.)
    _cancel_legacy_jobs()

    # Stop conflicting login-node services
    _stop_login_services()

    script_b64 = base64.b64encode(script.encode()).decode()
    submit_cmd = (
        f"mkdir -p {_SERVICES_DIR} && "
        f'echo "{script_b64}" | base64 -d > /tmp/codex-alloc.sh && '
        "sbatch /tmp/codex-alloc.sh && "
        "rm -f /tmp/codex-alloc.sh"
    )
    output = _run_remote(submit_cmd, timeout=30, check=True)
    click.echo(output.strip().split("\n")[0])


def _cancel_allocation() -> list[str]:
    """Cancel the allocation job (stopping all services first)."""
    alloc = _get_allocation()
    if not alloc:
        return []

    # Stop services explicitly before canceling the allocation,
    # since nohup'd processes survive SLURM job termination.
    if alloc["state"] == "RUNNING":
        node = alloc["node"]
        for svc in ("neo4j", "embed", "llm"):
            if _service_running(node, svc):
                _stop_service(node, svc)

    try:
        _run_remote(f"scancel {alloc['job_id']}", check=True)
        return [alloc["job_id"]]
    except subprocess.CalledProcessError:
        return []


def _cancel_legacy_jobs() -> None:
    """Cancel any legacy standalone SLURM jobs (codex-embed, etc.)."""
    for name in ("codex-embed", "codex-services"):
        try:
            out = _run_remote(f'squeue -n {name} -u "$USER" --format="%A" --noheader')
            for line in out.strip().split("\n"):
                jid = line.strip()
                if jid and jid != "(no output)" and jid.isdigit():
                    _run_remote(f"scancel {jid}", check=False)
                    click.echo(f"Cancelled legacy job {jid} ({name})")
        except subprocess.CalledProcessError:
            pass


def _stop_login_services() -> None:
    """Stop any conflicting login-node services (systemd, apptainer)."""
    # Embed systemd service
    try:
        result = _run_remote(
            "systemctl --user is-active imas-codex-embed 2>/dev/null || true",
            timeout=10,
        )
        if "active" in result:
            _run_remote(
                "systemctl --user stop imas-codex-embed 2>/dev/null || true",
                timeout=15,
            )
            click.echo("Stopped login embed service")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    # Neo4j systemd service
    try:
        result = _run_remote(
            "systemctl --user list-units 'imas-codex-neo4j-*' --no-pager "
            "--plain --no-legend 2>/dev/null | awk '{print $1}' | head -1",
            timeout=10,
        )
        service = result.strip()
        if service and service != "(no output)":
            _run_remote(
                f"systemctl --user stop {service} 2>/dev/null || true", timeout=15
            )
            click.echo(f"Stopped login service ({service})")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    # Kill any direct Neo4j apptainer processes (manual starts)
    try:
        result = _run_remote(
            "pgrep -u $USER -f 'neo4j_.*community.*\\.sif' 2>/dev/null | head -1",
            timeout=10,
        )
        pid = result.strip()
        if pid and pid != "(no output)" and pid.isdigit():
            _run_remote(f"kill {pid} 2>/dev/null || true", timeout=10)
            click.echo(f"Killed login Neo4j process (PID {pid})")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    # LLM proxy systemd service
    try:
        result = _run_remote(
            "systemctl --user is-active imas-codex-llm 2>/dev/null || true",
            timeout=10,
        )
        if "active" in result:
            _run_remote(
                "systemctl --user stop imas-codex-llm 2>/dev/null || true",
                timeout=15,
            )
            click.echo("Stopped login LLM proxy service")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass


# ── Service management on allocated node ─────────────────────────────────


def _start_service(node: str, name: str, command: str) -> None:
    """Start a named service on the compute node.

    Writes the command to a launcher script on the shared filesystem,
    then runs ``nohup`` on the compute node.  PID tracked in
    ``$SERVICES_DIR/<name>.pid``, logs in ``$SERVICES_DIR/<name>.log``.

    Idempotent: no-op if already running.
    """
    import base64 as b64

    if _service_running(node, name):
        click.echo(f"  {name} already running on {node}")
        return

    # Write service command as a launcher script on shared GPFS.
    # Source .env first for API keys, then exec the service command
    # so the script's PID becomes the actual process PID (not a
    # dead bash wrapper).
    svc_script = f"{_SERVICES_DIR}/{name}.sh"
    script_content = (
        f"#!/bin/bash\n"
        f"cd {_PROJECT}\n"
        f"source {_PROJECT}/.env 2>/dev/null || true\n"
        f"exec {command}\n"
    )
    script_b64 = b64.b64encode(script_content.encode()).decode()
    _run_remote(
        f"mkdir -p {_SERVICES_DIR} && "
        f'echo "{script_b64}" | base64 -d > {svc_script} && '
        f"chmod +x {svc_script}",
        timeout=10,
    )

    # Launch on compute node via nohup (script is on shared GPFS)
    launch = (
        f"nohup {svc_script} > {_SERVICES_DIR}/{name}.log 2>&1 &\n"
        f"echo $! > {_SERVICES_DIR}/{name}.pid\n"
        f'echo "Started {name} (PID $!)"\n'
    )
    result = _run_on_node(node, launch, timeout=30)
    click.echo(f"  {result.strip()}")


def _stop_service(node: str, name: str) -> bool:
    """Stop a named service on the compute node. Returns True if stopped.

    Sends SIGTERM to the top-level process first (Apptainer for Neo4j),
    which propagates it to the JVM for a graceful checkpoint/shutdown.
    Waits up to 90 seconds — Neo4j with a multi-GB database on GPFS
    can need 60+ seconds to checkpoint.

    Only escalates to SIGKILL as a last resort if the graceful window
    expires.  SIGKILL skips checkpointing and leaves stale POSIX locks
    on GPFS that require inode replacement (``_clean_neo4j_locks``).
    """
    stop = (
        f"pid=$(cat {_SERVICES_DIR}/{name}.pid 2>/dev/null) || exit 0\n"
        f'if kill -0 "$pid" 2>/dev/null; then\n'
        f"    # SIGTERM to top-level process — Apptainer forwards to JVM\n"
        f'    kill -TERM "$pid" 2>/dev/null || true\n'
        f"    # Wait up to 90s for graceful checkpoint + shutdown\n"
        f"    for i in $(seq 1 90); do\n"
        f'        kill -0 "$pid" 2>/dev/null || break\n'
        f"        sleep 1\n"
        f"    done\n"
        f'    if kill -0 "$pid" 2>/dev/null; then\n'
        f'        echo "Graceful shutdown timed out — escalating to SIGKILL"\n'
        f'        pkill -KILL -P "$pid" 2>/dev/null || true\n'
        f'        kill -KILL "$pid" 2>/dev/null || true\n'
        f"        sleep 2\n"
        f"    fi\n"
        f'    echo "Stopped {name} (PID $pid)"\n'
        f"else\n"
        f'    echo "{name} not running"\n'
        f"fi\n"
        f"rm -f {_SERVICES_DIR}/{name}.pid {_SERVICES_DIR}/{name}.sh\n"
    )
    result = _run_on_node(node, stop, timeout=120)
    click.echo(f"  {result.strip()}")
    return "Stopped" in result


_SERVICE_PORTS: dict[str, str] = {
    "neo4j": "_graph_port",
    "embed": "_embed_port",
}


def _service_running(node: str, name: str) -> bool:
    """Check if a named service is running on the compute node.

    First checks the PID file.  If the tracked PID is dead, falls back
    to a port-liveness check — this catches orphaned services that were
    started outside the deploy workflow (or whose PID file is stale).
    """
    check = (
        f"pid=$(cat {_SERVICES_DIR}/{name}.pid 2>/dev/null) || exit 1\n"
        f'kill -0 "$pid" 2>/dev/null || exit 1\n'
        f'echo "running"\n'
    )
    try:
        result = _run_on_node(node, check, timeout=10)
        if "running" in result:
            return True
    except subprocess.CalledProcessError:
        pass

    # Fallback: check if the service port is responding
    port_fn_name = _SERVICE_PORTS.get(name)
    if port_fn_name:
        port = globals()[port_fn_name]()
        try:
            result = _run_on_node(
                node,
                f"ss -tlnp sport = :{port} 2>/dev/null | grep -q LISTEN && echo listening",
                timeout=10,
            )
            if "listening" in result:
                # Port is bound — adopt the process into the PID file
                try:
                    pid_result = _run_on_node(
                        node,
                        f"ss -tlnp sport = :{port} 2>/dev/null"
                        f" | grep -oP 'pid=\\K[0-9]+' | head -1",
                        timeout=10,
                    )
                    pid = pid_result.strip()
                    if pid:
                        _run_remote(
                            f"echo {pid} > {_SERVICES_DIR}/{name}.pid",
                            timeout=5,
                        )
                except (subprocess.CalledProcessError, ValueError):
                    pass
                return True
        except subprocess.CalledProcessError:
            pass

    return False


# ── Neo4j-specific service helpers ───────────────────────────────────────


def _clean_neo4j_locks(node: str) -> None:
    """Remove Neo4j coordination lock files on the compute node.

    GPFS can retain stale ``store_lock`` and ``database_lock`` files
    after process death, preventing startup.

    Additionally, Java ``FileChannel.lock()`` POSIX locks on
    ``neostore*`` files can become stale on GPFS after SIGKILL.
    Neo4j locks all neostore files (``neostore``,
    ``neostore.nodestore.db``, ``neostore.propertystore.db``, etc.)
    — around 30 files per database.  Stale locks are cleared via
    inode replacement: read with ``dd`` (which avoids lock syscalls
    that ``cp`` uses and would hang on GPFS stale locks), write to
    a ``.unlock`` file, then ``mv`` to atomically replace the inode.

    **CRITICAL**: Only remove coordination locks (``store_lock``,
    ``database_lock``) and replace inodes on ``neostore*`` files.
    Never delete Lucene ``write.lock`` files (inside ``schema/index/``
    directories) — doing so corrupts vector indexes and can cause
    total data loss via checkpoint failure.
    """
    clean = (
        "DATA=$HOME/.local/share/imas-codex/neo4j/data\n"
        "# Remove coordination lock files\n"
        'rm -f "$DATA"/databases/store_lock "$DATA"/databases/*/database_lock '
        "2>/dev/null\n"
        "# Clear stale POSIX locks on neostore* via inode replacement.\n"
        "# Use dd instead of cp — cp uses fstat/fadvise which interact\n"
        "# with GPFS's distributed lock manager and hang on stale locks.\n"
        "# Each dd is wrapped in a 5s timeout — if GPFS blocks on a\n"
        "# stale lock the file is skipped (Neo4j recovery handles it).\n"
        'for ns in "$DATA"/databases/*/neostore*; do\n'
        '    [ -f "$ns" ] || continue\n'
        '    timeout 5 dd if="$ns" of="$ns.unlock" bs=4k 2>/dev/null '
        '&& mv -f "$ns.unlock" "$ns" '
        '|| rm -f "$ns.unlock" 2>/dev/null\n'
        "done\n"
        "echo locks_cleaned\n"
    )
    try:
        _run_on_node(node, clean, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass


def _kill_neo4j_on_node(node: str) -> None:
    """Kill any orphaned Neo4j processes on the compute node.

    Handles cases where PID file is missing but process is still running
    (e.g. after a failed deploy with health check timeout).
    """
    kill_cmd = (
        'pids=$(pgrep -u $USER -f "neo4j.*console|Neo4jCommunity" 2>/dev/null)\n'
        'if [ -n "$pids" ]; then\n'
        '    echo "Killing orphaned neo4j: $pids"\n'
        '    echo "$pids" | xargs kill -9 2>/dev/null || true\n'
        "    sleep 2\n"
        "fi\n"
    )
    try:
        result = _run_on_node(node, kill_cmd, timeout=15)
        if "Killing" in result:
            click.echo(f"  {result.strip()}")
    except subprocess.CalledProcessError:
        pass


def _neo4j_service_command() -> str:
    """Build the shell command to start Neo4j on a compute node.

    Binds to ``0.0.0.0`` on the exclusively-allocated SLURM compute
    node.  Port collisions between users are impossible because each
    user's allocation runs on a different node.

    Uses ``neo4j console`` directly with a host-side ``conf/`` bind
    mount for configuration.  **Never** use the Docker entrypoint —
    it calls ``set-initial-password`` and ``rm -rf conf/*`` on every
    start, which can reinitialize an existing database.

    The returned command is used inside ``_start_service()`` which wraps
    it with ``exec``, so the command must be a single executable call
    (no ``&&`` chains).  All setup (directory creation, config writing)
    is done in ``_neo4j_pre_launch()`` before this command runs.
    """
    from imas_codex.settings import get_neo4j_image_shell

    image = get_neo4j_image_shell()
    # Use $HOME so the path resolves on the compute node
    base = '"$HOME/.local/share/imas-codex/neo4j"'

    return (
        "apptainer exec "
        f"--bind {base}/data:/data "
        f"--bind {base}/logs:/logs "
        f"--bind {base}/import:/import "
        f"--bind {base}/conf:/var/lib/neo4j/conf "
        "--writable-tmpfs "
        '--env "NEO4J_server_jvm_additional=-Dfile.encoding=UTF-8 '
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        '--add-opens=java.base/java.lang=ALL-UNNAMED" '
        f"{image} "
        "neo4j console"
    )


def _neo4j_pre_launch(node: str) -> None:
    """Create directories and ensure neo4j.conf exists on the compute node.

    Must be called before ``_start_service()`` with the neo4j command.
    Separated from the service command because ``_start_service()``
    wraps the command with ``exec`` (replacing the shell), so
    ``&&``-chained setup steps would not work.

    Only writes neo4j.conf if it does not already exist — preserving
    any manual memory tuning.
    """
    from imas_codex.graph.dirs import DEFAULT_NEO4J_CONF
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j(auto_tunnel=False)
    conf_content = DEFAULT_NEO4J_CONF.format(
        listen_address="0.0.0.0",
        bolt_port=profile.bolt_port,
        http_port=profile.http_port,
    )

    setup_script = (
        'NEO4J_BASE="$HOME/.local/share/imas-codex/neo4j"\n'
        'mkdir -p "$NEO4J_BASE"/{data,logs,import,conf}\n'
        'if [ ! -f "$NEO4J_BASE/conf/neo4j.conf" ]; then\n'
        f"cat > \"$NEO4J_BASE/conf/neo4j.conf\" << 'CONFEOF'\n"
        f"{conf_content}\n"
        "CONFEOF\n"
        "fi\n"
    )
    _run_on_node(node, setup_script, timeout=15)


# ── Embed-specific service helpers ───────────────────────────────────────


def _embed_service_command(gpus: int, workers: int) -> str:
    """Build the shell command to start the embed server on a compute node.

    Binds to ``0.0.0.0`` on the exclusively-allocated SLURM compute
    node.  Accessible from the login node via the compute node's
    hostname on the cluster network.
    """
    port = _embed_port()
    gpu_ids = ",".join(str(i) for i in range(gpus))
    partition = _gpu_partition()
    partition_name = partition["name"]

    return (
        f"CUDA_VISIBLE_DEVICES={gpu_ids} "
        "uv run --offline --extra gpu imas-codex serve embed start "
        f"--host 0.0.0.0 --port {port} "
        f"--gpus {gpu_ids} --workers {workers} --deploy-label {partition_name}"
    )


# ── Health checks ────────────────────────────────────────────────────────


def _wait_for_health(
    label: str,
    check_cmd: str,
    timeout_s: int = 120,
    success_test: str | None = None,
) -> bool:
    """Wait for a service health check to pass. Returns True on success."""
    click.echo(f"  Waiting for {label}...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(5)
        try:
            result = _run_remote(check_cmd, timeout=10)
            if result and result != "(no output)":
                if success_test is None or success_test in result:
                    click.echo(f"  {label} healthy")
                    return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        remaining = int(deadline - time.time())
        if remaining > 0 and remaining % 15 < 5:
            click.echo(f"    {remaining}s remaining...")
    click.echo(f"  Warning: {label} not healthy after {timeout_s}s")
    return False


# ── Login-node deploy fallback (systemd) ─────────────────────────────────


def _deploy_login_embed() -> None:
    """Deploy embed server to login node via systemd."""
    click.echo("Deploying to login node via systemd...")
    try:
        _run_remote(
            "systemctl --user restart imas-codex-embed 2>/dev/null || "
            "systemctl --user start imas-codex-embed",
            timeout=15,
            check=True,
        )
        click.echo("  Service started")
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            "Service not installed. Run: imas-codex serve embed service install"
        ) from exc


def _deploy_login_neo4j() -> None:
    """Deploy Neo4j to login node via systemd."""
    from imas_codex.graph.profiles import resolve_neo4j

    profile = resolve_neo4j(auto_tunnel=False)
    service_name = f"imas-codex-neo4j-{profile.name}"
    click.echo(f"Deploying Neo4j [{profile.name}] to login node via systemd...")
    try:
        _run_remote(
            f"systemctl --user restart {service_name} 2>/dev/null || "
            f"systemctl --user start {service_name}",
            timeout=30,
            check=True,
        )
        click.echo("  Service started")
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            "Neo4j systemd service not installed.\n"
            "Install a systemd unit manually or use: imas-codex graph start"
        ) from exc


def _show_embed_info() -> None:
    """Print embed server info from /info endpoint."""
    import json

    from imas_codex.settings import get_embed_host

    host = get_embed_host() or "localhost"
    port = _embed_port()
    try:
        result = _run_remote(f"curl -sf http://{host}:{port}/info", timeout=10)
        if result and result != "(no output)":
            info = json.loads(result)
            model = info.get("model", {}).get("name", "unknown")
            device = info.get("model", {}).get("device", "unknown")
            click.echo(f"  Model: {model}")
            click.echo(f"  Device: {device}")
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        pass


def _show_embed_info_on_node(node: str, port: int) -> None:
    """Print embed server info by curling /info on the compute node."""
    import json

    try:
        result = _run_on_node(
            node, f"curl -sf http://localhost:{port}/info", timeout=10
        )
        if result and result != "(no output)":
            info = json.loads(result)
            model = info.get("model", {}).get("name", "unknown")
            device = info.get("model", {}).get("device", "unknown")
            click.echo(f"  Model: {model}")
            click.echo(f"  Device: {device}")
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        pass


# ── Log tailing ──────────────────────────────────────────────────────────


def _tail_log(log_file: str, follow: bool, lines: int) -> None:
    """Tail a log file on the remote host."""
    # Check if file exists
    try:
        result = _run_remote(f"test -f {log_file} && echo exists", timeout=5)
        if "exists" not in result:
            raise click.ClickException(f"Log file not found: {log_file}")
    except subprocess.CalledProcessError:
        raise click.ClickException(f"Log file not found: {log_file}") from None

    if follow:
        ssh = _facility_ssh()
        if ssh:
            os.execvp("ssh", ["ssh", ssh, f"tail -f {log_file}"])
        else:
            os.execvp("tail", ["tail", "-f", log_file])
    else:
        output = _run_remote(f"tail -n {lines} {log_file}", timeout=10)
        click.echo(output)

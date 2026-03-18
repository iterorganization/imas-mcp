"""Shared SLURM service management infrastructure.

Each service (Neo4j, embed) runs as its own SLURM job.  The job
script runs the service process directly — no ``sleep infinity``
allocation, no ``nohup`` wrappers.  SLURM manages the lifecycle:
``scancel`` stops the service, cgroup enforcement is automatic,
and ``squeue`` shows accurate resource accounting.

Extracted from ``serve.py`` so that both ``graph_cli.py`` (Neo4j
management) and ``serve.py`` (embed/LLM deployment) can share
the same service lifecycle primitives.

Sections:
    - Constants
    - Port helpers
    - Scheduler detection
    - Facility compute config
    - SSH helpers
    - Per-service SLURM job management
    - Neo4j-specific service helpers
    - Embed-specific service helpers
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

# Per-service SLURM job names
_NEO4J_JOB = "codex-neo4j"
_EMBED_JOB = "codex-embed"

_SERVICES_DIR = "$HOME/.local/share/imas-codex/services"
_DEFAULT_GPUS = 8
_PROJECT = "$HOME/Code/imas-codex"

# Per-service resource defaults
_NEO4J_CPUS = 4
_NEO4J_MEM = "32G"


# ── Color-coded status helpers ───────────────────────────────────────────


def _partition_limits() -> dict:
    """Return the partition resource limits for color coding.

    Loads from facility compute config; falls back to titan defaults.
    """
    try:
        partition = _gpu_partition()
        return {
            "cpus": partition.get("cpus_per_node", 20),
            "gpus": partition.get("gpus_per_node", 8),
            "mem_gb": partition.get("mem_per_node_gb", 250),
        }
    except click.ClickException:
        return {"cpus": 20, "gpus": 8, "mem_gb": 250}


def _usage_color(used: float, limit: float) -> str:
    """Return a color name based on proximity to limit.

    Green: <50%, yellow: 50-80%, red: >80%.
    """
    if limit <= 0:
        return "white"
    ratio = used / limit
    if ratio > 0.8:
        return "red"
    if ratio > 0.5:
        return "yellow"
    return "green"


def _colored_bar(used: float, limit: float, width: int = 20) -> str:
    """Render a colored usage bar: [████████░░░░] 40%."""
    if limit <= 0:
        return ""
    ratio = min(used / limit, 1.0)
    filled = int(ratio * width)
    empty = width - filled
    color = _usage_color(used, limit)
    bar = click.style("█" * filled, fg=color) + click.style("░" * empty, dim=True)
    pct = click.style(f"{ratio * 100:.0f}%", fg=color)
    return f"[{bar}] {pct}"


def _format_service_status(job: dict | None, svc_name: str) -> list[str]:
    """Format a service status block with color-coded resource usage.

    Returns a list of pre-formatted lines ready for click.echo().
    """
    lines: list[str] = []
    limits = _partition_limits()

    if not job:
        lines.append(click.style(f"  {svc_name}: not running", dim=True))
        return lines

    state = job["state"]
    if state == "RUNNING":
        state_color = "green"
    elif state == "PENDING":
        state_color = "yellow"
    else:
        state_color = "red"

    header = (
        f"  {svc_name}: job {job['job_id']} "
        f"{click.style(state, fg=state_color)} "
        f"on {job['node']}"
    )
    lines.append(header)

    if state != "RUNNING":
        return lines

    # Parse allocated resources from job info
    cpus = int(job.get("cpus", 0) or 0)
    gres = job.get("gres", "")
    gpus = 0
    if "gpu:" in gres:
        try:
            gpus = int(gres.split("gpu:")[-1].split(",")[0].split("(")[0])
        except (ValueError, IndexError):
            pass

    alloc_parts = [
        f"CPUs: {cpus}/{limits['cpus']}",
        f"Time: {job['time']}",
    ]
    if gpus > 0:
        alloc_parts.insert(1, f"GPUs: {gpus}/{limits['gpus']}")
    lines.append(f"    Allocated: {', '.join(alloc_parts)}")

    # Real-time load
    load = _get_service_load(job["node"], svc_name)
    if load:
        cpu_val = float(load.get("cpu", 0))
        mem_mb = float(load.get("mem_mb", 0))
        mem_limit_mb = limits["mem_gb"] * 1024

        cpu_bar = _colored_bar(cpu_val / 100, cpus)
        mem_bar = _colored_bar(mem_mb, mem_limit_mb)

        lines.append(f"    CPU:  {cpu_bar}  {cpu_val:.0f}% of {cpus} cores")
        lines.append(f"    Mem:  {mem_bar}  {mem_mb:.0f} MB / {limits['mem_gb']} GB")

        if "gpu_mem_mb" in load:
            gpu_mem = float(load.get("gpu_mem_mb", 0))
            gpu_limit = gpus * 16384  # P100 = 16GB
            gpu_bar = _colored_bar(gpu_mem, gpu_limit)
            lines.append(f"    GPU:  {gpu_bar}  {gpu_mem:.0f} MB / {gpu_limit} MB VRAM")

    return lines


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


def _llm_ssh() -> str | None:
    """SSH host for the LLM proxy node, or None if local.

    The LLM proxy runs on the default facility SSH target — no
    separate alias.  Returns the facility SSH host or None if local.
    """
    from imas_codex.remote.locations import is_location_local, resolve_location
    from imas_codex.settings import get_llm_location

    location = get_llm_location()
    if location == "local":
        return None
    if is_location_local(location):
        return None

    info = resolve_location(location)
    return info.ssh_host


def _run_llm_remote(cmd: str, timeout: int = 30, check: bool = False) -> str:
    """Run a command on the LLM proxy node (locally if already there)."""
    from imas_codex.remote.executor import run_command

    return run_command(cmd, ssh_host=_llm_ssh(), timeout=timeout, check=check)


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


# ── Per-service SLURM job management ─────────────────────────────────────


def _get_service_job(job_name: str) -> dict | None:
    """Get the active SLURM job for a named service.

    Returns dict with job_id, state, node, gres, time, cpus — or None.
    """
    try:
        out = _run_remote(
            f'squeue -n {job_name} -u "$USER" --format="%A|%T|%M|%N|%b|%C" --noheader'
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
                "job_name": job_name,
                "job_id": parts[0].strip(),
                "state": parts[1].strip(),
                "time": parts[2].strip(),
                "node": parts[3].strip(),
                "gres": parts[4].strip(),
                "cpus": parts[5].strip() if len(parts) > 5 else "",
            }
    return None


def _get_neo4j_job() -> dict | None:
    """Get the active Neo4j SLURM job."""
    return _get_service_job(_NEO4J_JOB)


def _get_embed_job() -> dict | None:
    """Get the active embed server SLURM job."""
    return _get_service_job(_EMBED_JOB)


def _get_all_service_jobs() -> dict[str, dict | None]:
    """Get status of all service jobs.

    Returns a dict mapping service name to job info (or None).
    """
    return {
        "neo4j": _get_neo4j_job(),
        "embed": _get_embed_job(),
    }


def _get_node_state(node: str) -> tuple[str, str]:
    """Get SLURM node state and reason.

    Returns:
        (state, reason) — e.g. ("idle", "none"), ("draining", "Duplicate jobid")
    """
    try:
        out = _run_remote(
            f'sinfo -n {node} -o "%T|%E" --noheader 2>/dev/null | head -1',
            timeout=10,
        )
        line = out.strip()
        if "|" in line:
            parts = line.split("|", 1)
            return parts[0].strip().lower(), parts[1].strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return "unknown", "unable to query"


def _submit_service_job(
    job_name: str,
    service_command: str,
    *,
    cpus: int,
    mem: str,
    gpus: int = 0,
    pre_launch: str = "",
) -> None:
    """Submit a SLURM job that runs a service directly.

    The job script sources ``.env``, runs optional pre-launch setup,
    then ``exec``s the service command.  SLURM manages the process
    lifecycle — ``scancel`` stops it, cgroup enforcement is automatic.

    Checks node state before submission — if the target node is
    draining/drained, raises a clear error instead of submitting
    a job that will never start.
    """
    import base64

    partition = _gpu_partition()
    host = _gpu_entry()["location"]
    partition_name = partition["name"]

    # Check node state before submitting
    node_state, reason = _get_node_state(host)
    if node_state in ("drained", "draining", "down", "down*", "drain", "drng"):
        raise click.ClickException(
            f"Node {host} is {node_state} (reason: {reason}).\n"
            f"SLURM will not schedule jobs on this node.\n"
            f"Ask an admin to resume: scontrol update NodeName={host} State=RESUME\n"
            f"Or bypass SLURM: imas-codex embed start --no-slurm"
        )

    remote_home = _run_remote("echo $HOME", timeout=10).strip()
    services_dir_abs = f"{remote_home}/.local/share/imas-codex/services"

    gres_line = f"#SBATCH --gres=gpu:{gpus}\n" if gpus > 0 else ""

    script = (
        "#!/bin/bash\n"
        f"#SBATCH --partition={partition_name}\n"
        f"{gres_line}"
        f"#SBATCH --cpus-per-task={cpus}\n"
        f"#SBATCH --mem={mem}\n"
        "#SBATCH --time=UNLIMITED\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --nodelist={host}\n"
        f"#SBATCH --output={services_dir_abs}/{job_name}.log\n"
        "\n"
        f"cd {_PROJECT}\n"
        f"source {_PROJECT}/.env 2>/dev/null || true\n"
        "\n"
        f'echo "{job_name} started on $(hostname) at $(date)"\n'
    )

    if pre_launch:
        script += f"\n# Pre-launch setup\n{pre_launch}\n"

    script += f"\n{service_command}\n"

    # Stop conflicting login-node services (systemd, apptainer)
    _stop_login_services(job_name)

    script_b64 = base64.b64encode(script.encode()).decode()
    submit_cmd = (
        f"mkdir -p {services_dir_abs} && "
        f'echo "{script_b64}" | base64 -d > /tmp/{job_name}.sh && '
        f"sbatch /tmp/{job_name}.sh && "
        f"rm -f /tmp/{job_name}.sh"
    )
    output = _run_remote(submit_cmd, timeout=30, check=True)
    click.echo(f"  {output.strip().split(chr(10))[0]}")


def _ensure_service_job(
    job_name: str,
    service_command: str,
    *,
    cpus: int,
    mem: str,
    gpus: int = 0,
    pre_launch: str = "",
    health_cmd: str | None = None,
    health_test: str | None = None,
) -> dict:
    """Ensure a service SLURM job is running, creating one if needed.

    Returns the job dict (job_id, node, etc.).
    """
    job = _get_service_job(job_name)
    if job and job["state"] == "RUNNING":
        return job

    # Cancel any existing job (PENDING, etc.) and wait for cleanup
    if job:
        _run_remote(f"scancel {job['job_id']}", check=False)
        # Wait for SLURM to fully process the cancellation before
        # submitting a new job — avoids "Duplicate jobid" race
        for _ in range(10):
            time.sleep(1)
            stale = _get_service_job(job_name)
            if not stale:
                break
        else:
            logger.warning(
                "Stale %s job %s still visible after cancel — proceeding",
                job_name,
                job["job_id"],
            )

    _submit_service_job(
        job_name,
        service_command,
        cpus=cpus,
        mem=mem,
        gpus=gpus,
        pre_launch=pre_launch,
    )

    # Wait for RUNNING state + health check with rich spinner.
    # 8 GPU workers loading Qwen3-0.6B models can take ~3 min on P100s.
    timeout_s = 240
    job = _wait_for_job(
        job_name, timeout_s=timeout_s, health_cmd=health_cmd, health_test=health_test
    )
    if job is None:
        raise click.ClickException(
            f"{job_name} did not start within {timeout_s}s. Check: squeue -u $USER"
        )
    return job


def _cancel_service_job(job_name: str) -> bool:
    """Cancel a service's SLURM job. Returns True if cancelled."""
    job = _get_service_job(job_name)
    if not job:
        return False
    try:
        _run_remote(f"scancel {job['job_id']}", check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def _stop_service(node: str, name: str) -> bool:
    """Stop a named service by canceling its SLURM job.

    For Neo4j, sends SIGTERM first for graceful shutdown before
    canceling the SLURM job, allowing checkpoint/flush.
    """
    # Map service name to SLURM job name
    job_names = {"neo4j": _NEO4J_JOB, "embed": _EMBED_JOB}
    job_name = job_names.get(name, name)
    job = _get_service_job(job_name)

    if not job or job["state"] != "RUNNING":
        click.echo(f"  {name} not running")
        return False

    # For Neo4j, send graceful shutdown signal before scancel
    if name == "neo4j":
        try:
            _run_on_node(
                node,
                'pid=$(pgrep -u $USER -f "Neo4jCommunity" 2>/dev/null | head -1)\n'
                'if [ -n "$pid" ]; then\n'
                '    kill -TERM "$pid" 2>/dev/null || true\n'
                "    for i in $(seq 1 60); do\n"
                '        kill -0 "$pid" 2>/dev/null || break\n'
                "        sleep 1\n"
                "    done\n"
                "fi\n",
                timeout=90,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

    try:
        _run_remote(f"scancel {job['job_id']}", check=True)
        click.echo(f"  Stopped {name} (job {job['job_id']})")
        return True
    except subprocess.CalledProcessError:
        click.echo(f"  Failed to stop {name}")
        return False


def _service_running(node: str, name: str) -> bool:
    """Check if a named service is running on a compute node."""
    job_names = {"neo4j": _NEO4J_JOB, "embed": _EMBED_JOB}
    job_name = job_names.get(name, name)
    job = _get_service_job(job_name)
    if job and job["state"] == "RUNNING":
        return True

    # Direct port check for orphaned processes
    port = _get_service_port(name)
    if port:
        return _port_listening(node, port)

    return False


def _get_service_port(name: str) -> int | None:
    """Get the port for a named service."""
    try:
        if name == "neo4j":
            return _graph_http_port()
        if name == "embed":
            return _embed_port()
    except Exception:
        pass
    return None


def _port_listening(node: str, port: int) -> bool:
    """Check if a port is listening on a compute node."""
    try:
        result = _run_on_node(
            node,
            f"ss -tlnp sport = :{port} 2>/dev/null | grep -q LISTEN && echo listening",
            timeout=10,
        )
        return "listening" in result
    except subprocess.CalledProcessError:
        return False


def _stop_login_services(job_name: str = "") -> None:
    """Stop conflicting login-node services (systemd, apptainer)."""
    if job_name in ("", _NEO4J_JOB):
        # Neo4j systemd
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
                click.echo(f"  Stopped login service ({service})")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

        # Kill direct Neo4j apptainer processes
        try:
            result = _run_remote(
                "pgrep -u $USER -f 'neo4j_.*community.*\\.sif' 2>/dev/null | head -1",
                timeout=10,
            )
            pid = result.strip()
            if pid and pid != "(no output)" and pid.isdigit():
                _run_remote(f"kill {pid} 2>/dev/null || true", timeout=10)
                click.echo(f"  Killed login Neo4j process (PID {pid})")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

    if job_name in ("", _EMBED_JOB):
        # Embed systemd
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
                click.echo("  Stopped login embed service")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

    # LLM proxy (always check)
    if job_name == "":
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
                click.echo("  Stopped login LLM proxy service")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass


def _get_service_load(node: str, name: str) -> dict | None:
    """Get real-time resource usage for a service on a compute node.

    Returns dict with cpu_percent, mem_mb, gpu_util (if applicable).
    """
    import json as _json

    if name == "neo4j":
        process_pattern = "Neo4jCommunity"
    elif name == "embed":
        process_pattern = "imas-codex embed"
    else:
        return None

    script = (
        f'pids=$(pgrep -u $USER -f "{process_pattern}" 2>/dev/null)\n'
        'if [ -z "$pids" ]; then echo "{{}}"; exit 0; fi\n'
        "cpu=0; mem=0\n"
        "for pid in $pids; do\n"
        "    vals=$(ps -p $pid -o %cpu=,%mem=,rss= 2>/dev/null | tail -1)\n"
        "    c=$(echo \"$vals\" | awk '{print $1}')\n"
        "    m=$(echo \"$vals\" | awk '{print $3}')\n"  # RSS in KB
        '    cpu=$(echo "$cpu + $c" | bc 2>/dev/null || echo "$cpu")\n'
        '    mem=$(echo "$mem + $m" | bc 2>/dev/null || echo "$mem")\n'
        "done\n"
        "mem_mb=$((mem / 1024))\n"
    )

    if name == "embed":
        # Include GPU utilization
        script += (
            "gpu_util=$(nvidia-smi --query-compute-apps=pid,used_memory "
            "--format=csv,noheader,nounits 2>/dev/null | "
            "awk -F', ' '{sum+=$2} END {print sum+0}')\n"
            "gpu_count=$(nvidia-smi --query-compute-apps=pid "
            "--format=csv,noheader 2>/dev/null | wc -l)\n"
            'printf \'{"cpu": %s, "mem_mb": %s, "gpu_mem_mb": %s, '
            '"gpu_count": %s}\\n\' "$cpu" "$mem_mb" "$gpu_util" "$gpu_count"\n'
        )
    else:
        script += 'printf \'{"cpu": %s, "mem_mb": %s}\\n\' "$cpu" "$mem_mb"\n'

    try:
        result = _run_on_node(node, script, timeout=15)
        for line in result.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                return _json.loads(line)
    except (subprocess.CalledProcessError, _json.JSONDecodeError, ValueError):
        pass
    return None


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


def _kill_embed_orphans(node: str) -> None:
    """Kill any orphaned embed server processes on the compute node.

    After scancel, some processes survive because:
    - SLURM cgroup cleanup races with process exit
    - ``uv run`` parent isn't always in the SLURM process group
    - torch/uvicorn workers trap SIGTERM and delay exit

    Uses SIGKILL for immediate cleanup, with a verify-and-retry loop.
    Matches both ``imas-codex embed`` and ``uv run.*imas-codex embed``
    to catch the full process tree.
    """
    kill_cmd = (
        'pids=$(pgrep -u $USER -f "imas-codex embed" 2>/dev/null)\n'
        'if [ -n "$pids" ]; then\n'
        '    echo "$pids" | xargs kill -9 2>/dev/null || true\n'
        '    sleep 1\n'
        '    # Verify\n'
        '    survivors=$(pgrep -u $USER -f "imas-codex embed" 2>/dev/null)\n'
        '    if [ -n "$survivors" ]; then\n'
        '        echo "$survivors" | xargs kill -9 2>/dev/null || true\n'
        '    fi\n'
        'fi\n'
    )
    try:
        _run_on_node(node, kill_cmd, timeout=15)
    except subprocess.TimeoutExpired:
        click.echo(
            f"Warning: could not reach {node} to kill orphans "
            f"(SSH timed out after 15s — node may be down)",
            err=True,
        )
    except subprocess.CalledProcessError:
        pass


def _neo4j_service_command() -> str:
    """Build the shell command to start Neo4j on a compute node.

    Binds to ``0.0.0.0`` inside the SLURM job's cgroup.  Port
    collisions between users are impossible because each user's
    job runs on a different node (or exclusive allocation).

    Uses ``neo4j console`` directly with a host-side ``conf/`` bind
    mount for configuration.  **Never** use the Docker entrypoint —
    it calls ``set-initial-password`` and ``rm -rf conf/*`` on every
    start, which can reinitialize an existing database.
    """
    from imas_codex.settings import get_neo4j_image_shell

    image = get_neo4j_image_shell()
    # Use $HOME so the path resolves on the compute node
    base = '"$HOME/.local/share/imas-codex/neo4j"'

    return (
        "exec apptainer exec "
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


def _neo4j_pre_launch(node: str | None = None) -> str:
    """Build pre-launch setup script for Neo4j.

    Creates directories and ensures neo4j.conf exists.  Only writes
    the config if it does not already exist — preserving any manual
    memory tuning.

    When ``node`` is provided, also runs the script on that node
    (backward compat for graph_cli).  Always returns the script text
    for embedding in ``_submit_service_job(..., pre_launch=...)``.
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

    if node is not None:
        _run_on_node(node, setup_script, timeout=15)

    return setup_script


# ── Embed-specific service helpers ───────────────────────────────────────


def _embed_service_command(gpus: int, workers: int) -> str:
    """Build the shell command to start the embed server on a compute node.

    Binds to ``0.0.0.0`` inside the SLURM job's cgroup.  Accessible
    from the login node via the compute node's hostname.
    """
    port = _embed_port()
    gpu_ids = ",".join(str(i) for i in range(gpus))
    partition = _gpu_partition()
    partition_name = partition["name"]

    return (
        f"export CUDA_VISIBLE_DEVICES={gpu_ids}\n"
        "exec uv run --offline --extra gpu imas-codex embed start -f "
        f"--host 0.0.0.0 --port {port} "
        f"--gpus {gpu_ids} --workers {workers} --deploy-label {partition_name}"
    )


# ── High-level deploy functions ──────────────────────────────────────────


def deploy_neo4j() -> dict:
    """Deploy Neo4j as its own SLURM job (no GPUs).

    Returns the job dict.
    """
    job = _get_neo4j_job()
    if job and job["state"] == "RUNNING":
        click.echo(
            f"Neo4j already running: job {job['job_id']} on {job['node']} "
            f"({job['cpus']} CPUs, {job['time']})"
        )
        return job

    # Kill orphaned Neo4j processes on the target node
    host = _gpu_entry()["location"]
    _kill_neo4j_on_node(host)
    _clean_neo4j_locks(host)

    pre_launch = _neo4j_pre_launch()
    http_port = _graph_http_port()

    click.echo("Deploying Neo4j...")
    return _ensure_service_job(
        _NEO4J_JOB,
        _neo4j_service_command(),
        cpus=_NEO4J_CPUS,
        mem=_NEO4J_MEM,
        gpus=0,
        pre_launch=pre_launch,
        health_cmd=f"curl -sf http://{host}:{http_port}/",
    )


def deploy_embed(gpus: int = _DEFAULT_GPUS, workers: int | None = None) -> dict:
    """Deploy the embed server as its own SLURM job.

    Returns the job dict.
    """
    if workers is None:
        workers = gpus

    job = _get_embed_job()
    if job and job["state"] == "RUNNING":
        click.echo(
            f"Embed already running: job {job['job_id']} on {job['node']} "
            f"({job['gres']}, {job['cpus']} CPUs, {job['time']})"
        )
        return job

    # Kill any orphaned embed processes before deploying a new job.
    # These survive scancel when SLURM doesn't fully clean cgroups.
    host = _gpu_entry()["location"]
    _kill_embed_orphans(host)

    port = _embed_port()

    click.echo(f"Deploying embed server ({gpus} GPUs, {workers} workers)...")
    # Scale CPUs and memory with worker count.
    # Inference is GPU-bound; 1 CPU per worker + 1 for the uvicorn
    # parent handles steady-state.  Thread-level parallelism (PyTorch
    # intra-op threads) doesn't need dedicated CPUs — they share the
    # allocation.  Cap at available CPUs minus neo4j headroom.
    embed_cpus = min(workers + 1, 15)
    embed_mem = f"{max(workers * 4, 16)}G"
    return _ensure_service_job(
        _EMBED_JOB,
        _embed_service_command(gpus, workers),
        cpus=embed_cpus,
        mem=embed_mem,
        gpus=gpus,
        health_cmd=f"curl -sf http://{host}:{port}/health",
        health_test='"status"',
    )


def deploy_embed_noslurm(gpus: int = _DEFAULT_GPUS, workers: int | None = None) -> None:
    """Deploy the embed server via SSH + nohup, bypassing SLURM.

    Use when the compute node is in draining/down state and SLURM
    won't schedule new jobs, but the GPUs are still accessible via SSH.

    Automatically detects a deadlocked CUDA driver (from D-state
    processes) and falls back to CPU mode when GPUs are unusable.
    """

    if workers is None:
        workers = gpus

    host = _gpu_entry()["location"]
    port = _embed_port()

    # Kill any existing embed processes on the node
    _kill_embed_orphans(host)

    gpu_ids = ",".join(str(i) for i in range(gpus))
    partition = _gpu_partition()
    partition_name = partition["name"]
    services_dir = "$HOME/.local/share/imas-codex/services"
    log_file = f"{services_dir}/codex-embed.log"

    # Check if CUDA is usable by running a quick probe.
    # D-state processes from previous crashes can deadlock the NVIDIA
    # kernel module, making torch.cuda.is_available() hang indefinitely.
    force_cpu = False
    try:
        probe_script = (
            "cd $HOME/Code/imas-codex && "
            "timeout 10 .venv/bin/python3 -c "
            "'import torch; print(torch.cuda.device_count())' 2>&1"
        )
        result = _run_on_node(host, probe_script, timeout=20)
        gpu_count = int(result.strip().split("\n")[-1])
        if gpu_count == 0:
            force_cpu = True
    except Exception:
        force_cpu = True

    if force_cpu:
        click.echo(
            click.style(
                "  ⚠ CUDA driver unresponsive — deploying in CPU mode",
                fg="yellow",
            )
        )
        workers = min(workers, 4)  # CPU doesn't need 8 workers

    # Build the nohup command that runs on the compute node
    env_lines = (
        f"mkdir -p {services_dir}\n"
        f"cd $HOME/Code/imas-codex\n"
        f"source $HOME/Code/imas-codex/.env 2>/dev/null || true\n"
        f'echo "codex-embed started on $(hostname) at $(date)" > {log_file}\n'
    )
    if force_cpu:
        env_lines += "export IMAS_CODEX_FORCE_CPU=1\n"
        env_lines += "export CUDA_VISIBLE_DEVICES=\n"

    else:
        env_lines += f"export CUDA_VISIBLE_DEVICES={gpu_ids}\n"
        env_lines += "export IMAS_CODEX_GPU_MEMORY_FRACTION=0.95\n"

    # In CPU mode, skip --gpus to avoid GPU claiming logic
    gpus_flag = "" if force_cpu else f"--gpus {gpu_ids} "

    script = (
        env_lines
        + f"nohup uv run --offline --extra gpu imas-codex embed start -f "
        f"--host 0.0.0.0 --port {port} "
        f"{gpus_flag}--workers {workers} "
        f"--deploy-label {partition_name} "
        f">> {log_file} 2>&1 &\n"
        f"echo $!\n"
    )

    click.echo(
        f"Deploying embed server via SSH ({gpus} GPUs, {workers} workers)..."
    )
    click.echo(
        click.style("  ⚠ Running outside SLURM — not managed by scheduler", fg="yellow")
    )

    try:
        result = _run_on_node(host, script, timeout=15)
        pid = result.strip().split("\n")[-1].strip()
        click.echo(f"  Started on {host} (PID: {pid})")
        click.echo(f"  Log: {log_file}")
    except Exception as e:
        raise click.ClickException(
            f"Failed to start embed server on {host}: {e}"
        ) from e

    # Wait for health check
    click.echo("  Waiting for health check...")
    # Sequential startup with startup lock: each worker loads the model
    # one at a time (~30s each).  8 workers × 30s + overhead ≈ 5 minutes.
    health_timeout_s = max(90, workers * 45)
    health_attempts = health_timeout_s // 3
    health_cmd = f"curl -sf http://{host}:{port}/health"
    for attempt in range(health_attempts):
        time.sleep(3)
        try:
            result = _run_remote(health_cmd, timeout=5)
            if '"status"' in result:
                click.echo(click.style("  ✓ Healthy", fg="green"))
                return
        except Exception:
            pass
        if attempt % 5 == 4:
            click.echo(f"  Still waiting... ({(attempt + 1) * 3}s)")

    click.echo(
        click.style(
            f"  ✗ Health check timed out after {health_timeout_s}s", fg="red"
        )
    )
    click.echo(f"  Check logs: ssh {host} 'tail -50 {log_file}'")


# ── Health checks ────────────────────────────────────────────────────────


def _wait_for_job(
    job_name: str,
    *,
    timeout_s: int = 120,
    health_cmd: str | None = None,
    health_test: str | None = None,
) -> dict | None:
    """Wait for a SLURM job to reach RUNNING and optionally pass health check.

    Displays a rich spinner with countdown when the terminal supports it,
    otherwise falls back to plain text.  Returns the job dict on success,
    or ``None`` on timeout.
    """
    from imas_codex.cli.rich_output import should_use_rich

    use_rich = should_use_rich()

    if use_rich:
        return _wait_rich(
            job_name,
            timeout_s=timeout_s,
            health_cmd=health_cmd,
            health_test=health_test,
        )
    return _wait_plain(
        job_name, timeout_s=timeout_s, health_cmd=health_cmd, health_test=health_test
    )


def _wait_rich(
    job_name: str,
    *,
    timeout_s: int,
    health_cmd: str | None,
    health_test: str | None,
) -> dict | None:
    """Rich spinner + countdown progress for SLURM job startup."""
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    console = Console()
    deadline = time.time() + timeout_s
    phase = "job"  # "job" → "health"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Starting {job_name}… [dim]PENDING[/dim]", total=None)

        while time.time() < deadline:
            time.sleep(3)
            remaining = max(0, int(deadline - time.time()))

            if phase == "job":
                job = _get_service_job(job_name)
                state = job["state"] if job else "PENDING"
                if job and state == "RUNNING":
                    progress.update(
                        task,
                        description=(
                            f"[green]✓[/green] {job_name} running — "
                            f"job {job['job_id']} on {job['node']}"
                        ),
                    )
                    if health_cmd:
                        phase = "health"
                        progress.update(
                            task,
                            description=(
                                f"Waiting for {job_name} health check… "
                                f"[dim]{remaining}s[/dim]"
                            ),
                        )
                        continue
                    # No health check — done
                    progress.stop()
                    console.print(
                        f"  [green]✓[/green] {job_name} running: "
                        f"job {job['job_id']} on {job['node']}"
                    )
                    return job
                progress.update(
                    task,
                    description=(
                        f"Starting {job_name}… [dim]{state} · {remaining}s[/dim]"
                    ),
                )
            else:
                # Health check phase
                try:
                    result = _run_remote(health_cmd, timeout=10)
                    if result and result != "(no output)":
                        if health_test is None or health_test in result:
                            progress.stop()
                            console.print(
                                f"  [green]✓[/green] {job_name} running: "
                                f"job {job['job_id']} on {job['node']}"
                            )
                            console.print(f"  [green]✓[/green] {job_name} healthy")
                            return job
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    pass
                progress.update(
                    task,
                    description=(
                        f"Waiting for {job_name} health check… [dim]{remaining}s[/dim]"
                    ),
                )

    return None


def _wait_plain(
    job_name: str,
    *,
    timeout_s: int,
    health_cmd: str | None,
    health_test: str | None,
) -> dict | None:
    """Plain text fallback for non-TTY environments."""
    deadline = time.time() + timeout_s
    click.echo(f"  Waiting for {job_name}...")

    # Phase 1: wait for RUNNING
    while time.time() < deadline:
        time.sleep(3)
        job = _get_service_job(job_name)
        if job and job["state"] == "RUNNING":
            click.echo(f"  {job_name} running: job {job['job_id']} on {job['node']}")
            break
    else:
        return None

    # Phase 2: health check
    if health_cmd and job:
        click.echo(f"  Checking {job_name} health...")
        while time.time() < deadline:
            time.sleep(5)
            try:
                result = _run_remote(health_cmd, timeout=10)
                if result and result != "(no output)":
                    if health_test is None or health_test in result:
                        click.echo(f"  {job_name} healthy")
                        return job
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass

    return job  # type: ignore[return-value]


def _wait_for_health(
    label: str,
    check_cmd: str,
    timeout_s: int = 120,
    success_test: str | None = None,
    ssh_host: str | None = ...,
) -> bool:
    """Wait for a service health check to pass. Returns True on success.

    Uses rich spinner when available, plain text otherwise.

    Args:
        ssh_host: SSH host override. Default (``...``) uses ``_facility_ssh()``.
    """
    from imas_codex.cli.rich_output import should_use_rich

    if should_use_rich():
        return _wait_for_health_rich(
            label, check_cmd, timeout_s, success_test, ssh_host
        )
    return _wait_for_health_plain(
        label, check_cmd, timeout_s, success_test, ssh_host
    )


def _wait_for_health_rich(
    label: str,
    check_cmd: str,
    timeout_s: int,
    success_test: str | None,
    ssh_host: str | None = ...,
) -> bool:
    """Rich spinner health check."""
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from imas_codex.remote.executor import run_command

    if ssh_host is ...:
        ssh_host = _facility_ssh()

    console = Console()
    deadline = time.time() + timeout_s

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Waiting for {label}…", total=None)
        while time.time() < deadline:
            time.sleep(5)
            remaining = max(0, int(deadline - time.time()))
            try:
                result = run_command(check_cmd, ssh_host=ssh_host, timeout=10)
                if result and result != "(no output)":
                    if success_test is None or success_test in result:
                        progress.stop()
                        console.print(f"  [green]✓[/green] {label} healthy")
                        return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
            progress.update(
                task,
                description=f"Waiting for {label}… [dim]{remaining}s[/dim]",
            )

    click.echo(f"  Warning: {label} not healthy after {timeout_s}s")
    return False


def _wait_for_health_plain(
    label: str,
    check_cmd: str,
    timeout_s: int,
    success_test: str | None,
    ssh_host: str | None = ...,
) -> bool:
    """Plain text health check fallback."""
    from imas_codex.remote.executor import run_command

    if ssh_host is ...:
        ssh_host = _facility_ssh()

    click.echo(f"  Waiting for {label}...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(5)
        try:
            result = run_command(check_cmd, ssh_host=ssh_host, timeout=10)
            if result and result != "(no output)":
                if success_test is None or success_test in result:
                    click.echo(f"  {label} healthy")
                    return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    click.echo(f"  Warning: {label} not healthy after {timeout_s}s")
    return False


# ── Login-node deploy fallback (systemd) ─────────────────────────────────


def _deploy_login_embed() -> None:
    """Deploy embed server to login node via systemd or nohup fallback.

    Tries systemd first.  Falls back to nohup when systemd --user is
    unavailable (common on HPC nodes without a D-Bus session bus).
    """
    # Try systemd first
    try:
        _run_remote(
            "systemctl --user status imas-codex-embed >/dev/null 2>&1 && "
            "systemctl --user restart imas-codex-embed 2>/dev/null || "
            "systemctl --user start imas-codex-embed 2>/dev/null",
            timeout=15,
            check=True,
        )
        click.echo("Deployed to login node via systemd")
        return
    except subprocess.CalledProcessError:
        pass

    # systemd unavailable — fall back to nohup
    click.echo("Deploying to login node via nohup (no systemd session bus)...")
    _deploy_login_embed_nohup()


def _deploy_login_embed_nohup() -> None:
    """Deploy embed server on the login node via nohup.

    Used when systemd --user is unavailable (no D-Bus session bus).
    """
    from imas_codex.settings import get_embed_server_port

    port = get_embed_server_port()
    services_dir = _SERVICES_DIR
    log_file = f"{services_dir}/codex-embed.log"

    # Kill any existing embed processes on this node
    _kill_login_embed()

    script = (
        f"mkdir -p {services_dir}\n"
        f"cd {_PROJECT}\n"
        f"source {_PROJECT}/.env 2>/dev/null || true\n"
        f'echo "codex-embed started on $(hostname) at $(date)" > {log_file}\n'
        f"export CUDA_VISIBLE_DEVICES=1\n"
        f"nohup uv run --offline --extra gpu imas-codex embed start -f "
        f"--host 0.0.0.0 --port {port} "
        f"--deploy-label login "
        f">> {log_file} 2>&1 &\n"
        f"echo $!\n"
    )

    try:
        result = _run_remote(script, timeout=15, check=True)
        pid = result.strip().split("\n")[-1].strip()
        click.echo(f"  Started (PID: {pid})")
        click.echo(f"  Log: {log_file}")
    except Exception as e:
        raise click.ClickException(
            f"Failed to start embed server: {e}"
        ) from e

    # Wait for health check
    click.echo("  Waiting for health check...")
    health_timeout_s = 90
    health_attempts = health_timeout_s // 3
    health_cmd = f"curl -sf http://localhost:{port}/health"
    for _attempt in range(health_attempts):
        time.sleep(3)
        try:
            result = _run_remote(health_cmd, timeout=5)
            if "ok" in result.lower() or "healthy" in result.lower():
                click.echo("  ✓ Healthy")
                _show_embed_info()
                return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    click.echo(f"  Warning: not healthy after {health_timeout_s}s — check logs")


def _kill_login_embed() -> None:
    """Kill any running embed server processes on the login node.

    Uses ``imas-codex embed start`` as the pgrep pattern to match only
    running server processes — not CLI management commands like
    ``embed stop`` or ``embed restart`` (which would self-kill).
    """
    kill_cmd = (
        'pids=$(pgrep -u $USER -f "imas-codex embed start" 2>/dev/null)\n'
        'if [ -n "$pids" ]; then\n'
        '    echo "$pids" | xargs kill -9 2>/dev/null || true\n'
        '    sleep 1\n'
        'fi\n'
    )
    try:
        _run_remote(kill_cmd, timeout=10)
    except subprocess.CalledProcessError:
        pass


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


def _tail_log(
    log_file: str, follow: bool, lines: int, ssh_host: str | None = ...
) -> None:
    """Tail a log file on the remote host.

    Args:
        log_file: Remote log file path.
        follow: Whether to follow the log (tail -f).
        lines: Number of lines to show.
        ssh_host: SSH host override. Default (``...``) uses ``_facility_ssh()``.
    """
    from imas_codex.remote.executor import run_command

    if ssh_host is ...:
        ssh_host = _facility_ssh()

    # Check if file exists
    try:
        result = run_command(
            f"test -f {log_file} && echo exists",
            ssh_host=ssh_host,
            timeout=5,
        )
        if "exists" not in result:
            raise click.ClickException(f"Log file not found: {log_file}")
    except subprocess.CalledProcessError:
        raise click.ClickException(f"Log file not found: {log_file}") from None

    if follow:
        if ssh_host:
            os.execvp("ssh", ["ssh", ssh_host, f"tail -f {log_file}"])
        else:
            os.execvp("tail", ["tail", "-f", log_file])
    else:
        output = run_command(
            f"tail -n {lines} {log_file}",
            ssh_host=ssh_host,
            timeout=10,
        )
        click.echo(output)

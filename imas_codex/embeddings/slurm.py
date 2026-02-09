"""SLURM-based embedding server lifecycle management.

Automatically submits, monitors, port-forwards, and shuts down a GPU embedding
server on the SLURM titan partition. The user never needs to interact with
SLURM directly - the system handles the full lifecycle:

1. ``ensure_server()`` checks for a running SLURM job
2. If none, submits one via ``sbatch`` and waits for it to start
3. Sets up port forwarding from login node to compute node
4. Returns when ``/health`` is reachable on ``localhost:PORT``
5. Server auto-stops after ``--idle-timeout`` of inactivity

State is persisted in ``~/.imas-codex/slurm-embed.json`` so multiple
processes can share the same SLURM job without redundant submissions.

Architecture::

    workstation ---SSH tunnel---> login node ---TCP proxy---> GPU node (SLURM)
    localhost:18765             localhost:18765              0.0.0.0:18765

The TCP proxy is used instead of SSH port forwarding because ITER compute
nodes restrict ``AllowTcpForwarding`` in sshd.  Direct IP connections from
the login node to compute nodes work fine, so the proxy relays TCP traffic.

Usage:
    from imas_codex.embeddings.slurm import ensure_server, get_job_status

    # Automatically start if needed, block until ready
    ensure_server()

    # Check status without starting
    info = get_job_status()
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import textwrap
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_PARTITION = "titan"
DEFAULT_GPU_COUNT = 4
DEFAULT_WALLTIME = "08:00:00"
DEFAULT_IDLE_TIMEOUT = 1800  # 30 minutes
DEFAULT_PORT = 18765

# State directory (works both locally and on ITER)
STATE_DIR = Path.home() / ".imas-codex"
STATE_FILE = STATE_DIR / "slurm-embed.json"

# Remote state dir - uses $HOME for shell expansion in sbatch scripts
REMOTE_STATE_DIR = "$HOME/.imas-codex"

# How long to wait for job to start and server to become ready
JOB_START_TIMEOUT = 300  # 5 minutes for SLURM scheduling
SERVER_READY_TIMEOUT = 120  # 2 minutes for model loading
POLL_INTERVAL = 3.0  # seconds between status checks


@lru_cache(maxsize=1)
def _is_on_iter() -> bool:
    """Detect if we're running on an ITER cluster node (login or compute).

    Checks for ITER-specific hostname pattern (98dci4-*) or SLURM availability.
    """
    hostname = os.uname().nodename
    if hostname.startswith("98dci4-"):
        return True
    return shutil.which("sbatch") is not None


@dataclass
class SlurmJobState:
    """Persisted state for the SLURM embedding server."""

    job_id: str | None = None
    node: str | None = None
    port: int = DEFAULT_PORT
    partition: str = DEFAULT_PARTITION
    submitted_at: float = 0.0
    started_at: float = 0.0
    pid_on_node: int | None = None

    # Port forwarding state
    forward_pid: int | None = None

    def save(self) -> None:
        """Persist state to disk."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(asdict(self), indent=2))
        logger.debug("Saved SLURM state: job_id=%s node=%s", self.job_id, self.node)

    @classmethod
    def load(cls) -> SlurmJobState:
        """Load state from disk, returning empty state if not found."""
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                return cls(
                    **{k: v for k, v in data.items() if k in cls.__dataclass_fields__}
                )
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Corrupt state file, resetting: %s", e)
        return cls()

    def clear(self) -> None:
        """Remove persisted state."""
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        self.job_id = None
        self.node = None
        self.forward_pid = None


def _run_cmd(cmd: str, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    """Run a command, locally if on ITER or via SSH otherwise."""
    if _is_on_iter():
        return subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    return subprocess.run(
        ["ssh", "iter", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _resolve_remote_home() -> str:
    """Resolve the remote home directory path.

    On ITER login nodes, $HOME expands to /home/ITER/<user>.
    When running locally (on ITER), Path.home() gives the right answer.
    When running via SSH, we ask the remote shell.
    """
    if _is_on_iter():
        return str(Path.home())
    result = _run_cmd("echo $HOME", timeout=10)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return str(Path.home())


def _generate_sbatch_script(
    port: int = DEFAULT_PORT,
    partition: str = DEFAULT_PARTITION,
    gpu_count: int = DEFAULT_GPU_COUNT,
    walltime: str = DEFAULT_WALLTIME,
    idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
    model_name: str | None = None,
) -> str:
    """Generate the SLURM batch script for the embedding server.

    Args:
        port: Port for the embedding server
        partition: SLURM partition
        gpu_count: Number of GPUs to request
        walltime: Maximum wall time
        idle_timeout: Seconds of inactivity before auto-shutdown
        model_name: Embedding model to use (overrides remote pyproject.toml)
    """
    # Scale CPUs and memory with GPU count
    cpus = max(4, gpu_count * 2)
    mem_gb = max(16, gpu_count * 8)

    # Resolve absolute home path for SLURM directives.
    # SLURM #SBATCH does NOT expand $HOME or %h on all versions.
    home = _resolve_remote_home()
    state_dir = f"{home}/.imas-codex"

    # Pin model via env var so the remote server uses the same model
    # as the client, regardless of the remote pyproject.toml config.
    model_env = ""
    if model_name:
        model_env = f'export IMAS_CODEX_EMBEDDING_MODEL="{model_name}"'

    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=imas-codex-embed
        #SBATCH --partition={partition}
        #SBATCH --gres=gpu:{gpu_count}
        #SBATCH --cpus-per-task={cpus}
        #SBATCH --mem={mem_gb}G
        #SBATCH --time={walltime}
        #SBATCH --output={state_dir}/slurm-embed-%j.log
        #SBATCH --error={state_dir}/slurm-embed-%j.log

        # Write node info for port forwarding
        echo "${{SLURM_JOB_NODELIST}}" > {state_dir}/slurm-embed-node
        echo "${{SLURM_JOB_ID}}" > {state_dir}/slurm-embed-jobid

        # GPU nodes have no internet - use HuggingFace cache from NFS
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1

        # Expose all allocated GPUs to the server
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(({gpu_count} - 1)))
        export PATH="{home}/.local/bin:${{PATH}}"
        {model_env}

        cd {home}/Code/imas-codex

        # Start embedding server with idle timeout
        exec {home}/.local/bin/uv run --extra gpu imas-codex serve embed start \\
            --host 0.0.0.0 \\
            --port {port} \\
            --idle-timeout {idle_timeout}
    """)


def submit_job(
    port: int = DEFAULT_PORT,
    partition: str = DEFAULT_PARTITION,
    gpu_count: int = DEFAULT_GPU_COUNT,
    walltime: str = DEFAULT_WALLTIME,
    idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
    model_name: str | None = None,
) -> SlurmJobState:
    """Submit a SLURM job to run the embedding server.

    Args:
        port: Port for the embedding server
        partition: SLURM partition (default: titan)
        gpu_count: Number of GPUs to request
        walltime: Maximum wall time
        idle_timeout: Seconds of inactivity before auto-shutdown
        model_name: Embedding model to use (overrides remote pyproject.toml)

    Returns:
        SlurmJobState with job_id set

    Raises:
        RuntimeError: If job submission fails
    """
    # Resolve model name from settings if not provided
    if model_name is None:
        from imas_codex.settings import get_imas_embedding_model

        model_name = get_imas_embedding_model()

    script = _generate_sbatch_script(
        port=port,
        partition=partition,
        gpu_count=gpu_count,
        walltime=walltime,
        idle_timeout=idle_timeout,
        model_name=model_name,
    )

    # Write script to remote, then submit (separate commands to avoid
    # heredoc + '&&' syntax error in bash -c / SSH contexts).
    script_path = f"{REMOTE_STATE_DIR}/slurm-embed.sh"
    write_result = _run_cmd(
        f"mkdir -p {REMOTE_STATE_DIR} && cat > {script_path} << 'SLURM_SCRIPT_EOF'\n{script}SLURM_SCRIPT_EOF"
    )
    if write_result.returncode != 0:
        raise RuntimeError(
            f"Failed to write sbatch script: {write_result.stderr.strip()}"
        )

    result = _run_cmd(f"sbatch {script_path}")

    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")

    # Parse job ID from "Submitted batch job 12345"
    output = result.stdout.strip()
    if "Submitted batch job" not in output:
        raise RuntimeError(f"Unexpected sbatch output: {output}")

    job_id = output.split()[-1]
    logger.info("Submitted SLURM job %s on partition %s", job_id, partition)

    state = SlurmJobState(
        job_id=job_id,
        port=port,
        partition=partition,
        submitted_at=time.time(),
    )
    state.save()
    return state


def get_job_status(state: SlurmJobState | None = None) -> SlurmJobState | None:
    """Check the status of a SLURM embedding server job.

    Returns updated state with node info if running, or None if no job exists.
    """
    if state is None:
        state = SlurmJobState.load()

    if not state.job_id:
        return None

    result = _run_cmd(
        f"squeue -j {state.job_id} --noheader --format='%T %N %S' 2>/dev/null"
    )

    if result.returncode != 0 or not result.stdout.strip():
        # Job no longer in queue (completed/cancelled/failed)
        logger.info("SLURM job %s no longer in queue", state.job_id)
        state.clear()
        return None

    parts = result.stdout.strip().split()
    job_state = parts[0] if parts else "UNKNOWN"

    if job_state == "RUNNING":
        node = parts[1] if len(parts) > 1 else None
        if node and node != state.node:
            state.node = node
            state.started_at = time.time()
            state.save()
            logger.info("Job %s running on node %s", state.job_id, node)
        return state

    if job_state in ("PENDING", "CONFIGURING"):
        logger.debug("Job %s is %s", state.job_id, job_state)
        return state

    # Failed/cancelled/etc
    logger.warning("Job %s in unexpected state: %s", state.job_id, job_state)
    state.clear()
    return None


def _free_port(port: int) -> None:
    """Free the embedding server port on the login node.

    Stops any non-SLURM process bound to the port (e.g., systemd user
    service) so that SLURM port forwarding can bind successfully.
    """
    # Stop systemd user service if running (most common scenario)
    result = _run_cmd("systemctl --user stop imas-codex-embed 2>/dev/null", timeout=10)
    if result.returncode == 0:
        logger.info("Stopped systemd embed service to free port %d", port)
        time.sleep(0.5)
        return

    # Kill any remaining (non-SSH) process on the port
    _run_cmd(
        f"fuser {port}/tcp 2>/dev/null | tr -s ' ' '\\n' | "
        f"while read pid; do "
        f"  ps -p $pid -o args= 2>/dev/null | grep -q 'ssh.*-L' || kill $pid 2>/dev/null; "
        f"done",
        timeout=10,
    )
    time.sleep(0.5)


def _is_slurm_server(port: int = DEFAULT_PORT) -> bool:
    """Check if the server on the given port is SLURM-managed.

    Queries the /info endpoint for a SLURM job ID. Servers launched via
    SLURM have SLURM_JOB_ID in their environment, while systemd or
    manual servers do not.
    """
    try:
        import httpx

        with httpx.Client(timeout=httpx.Timeout(5.0, connect=3.0)) as client:
            response = client.get(f"http://localhost:{port}/info")
            if response.status_code == 200:
                data = response.json()
                return data.get("server", {}).get("slurm_job_id") is not None
    except Exception:
        pass
    return False


def _setup_port_forward(state: SlurmJobState) -> bool:
    """Set up port forwarding from login node to compute node.

    ITER compute nodes restrict SSH TCP forwarding (AllowTcpForwarding),
    so we use a Python TCP proxy instead of ``ssh -L``.  The proxy
    listens on ``localhost:PORT`` on the login node and relays TCP
    connections to ``COMPUTE_NODE:PORT`` via direct IP.

    The workstation's SSH tunnel (workstation → login) chains through
    the proxy to reach the GPU server.

    Before binding, frees the port from any non-SLURM process (e.g.,
    systemd user service) that might be occupying it.

    Returns True if forwarding is established.
    """
    if not state.node or not state.port:
        return False

    # Kill any existing forward
    _kill_port_forward(state)

    # Free port from non-SLURM processes (e.g., systemd service)
    _free_port(state.port)

    # Use a Python TCP proxy instead of SSH -L (compute nodes block
    # SSH TCP forwarding but allow direct IP connections from login).
    proxy_script = textwrap.dedent(f"""\
        import asyncio, sys, os

        LOCAL_PORT = {state.port}
        REMOTE_HOST = "{state.node}"
        REMOTE_PORT = {state.port}

        async def relay(reader, writer):
            try:
                while True:
                    data = await reader.read(65536)
                    if not data:
                        break
                    writer.write(data)
                    await writer.drain()
            except Exception:
                pass
            finally:
                writer.close()

        async def handle(local_r, local_w):
            try:
                remote_r, remote_w = await asyncio.open_connection(
                    REMOTE_HOST, REMOTE_PORT
                )
            except Exception:
                local_w.close()
                return
            await asyncio.gather(
                relay(local_r, remote_w), relay(remote_r, local_w)
            )

        async def main():
            srv = await asyncio.start_server(handle, "127.0.0.1", LOCAL_PORT)
            # Write PID file for lifecycle management
            pid_path = os.path.expanduser(
                "~/.imas-codex/tcp-proxy-{state.port}.pid"
            )
            with open(pid_path, "w") as f:
                f.write(str(os.getpid()))
            async with srv:
                await srv.serve_forever()

        asyncio.run(main())
    """)

    # Write proxy script to file, then launch with nohup and full fd
    # isolation so subprocess.run() returns immediately.
    script_path = f"{REMOTE_STATE_DIR}/tcp-proxy-{state.port}.py"
    pid_path = f"{REMOTE_STATE_DIR}/tcp-proxy-{state.port}.pid"
    log_path = f"{REMOTE_STATE_DIR}/tcp-proxy-{state.port}.log"

    write_result = _run_cmd(
        f"mkdir -p {REMOTE_STATE_DIR}"
        f" && cat > {script_path} << 'PROXY_SCRIPT_EOF'\n{proxy_script}PROXY_SCRIPT_EOF"
    )
    if write_result.returncode != 0:
        logger.warning("Failed to write proxy script: %s", write_result.stderr.strip())
        return False

    # Launch proxy: nohup + redirect all fds so bash returns immediately.
    # The script writes its own PID file on startup.
    _run_cmd(
        f"nohup python3 {script_path} > {log_path} 2>&1 </dev/null &",
        timeout=5,
    )

    # Give the proxy a moment to start and write its PID file
    time.sleep(1.0)

    # Read PID from file
    result = _run_cmd(f"cat {pid_path} 2>/dev/null", timeout=5)
    output = result.stdout.strip()
    if not output:
        # Check if the proxy is listening despite no PID file
        check = _run_cmd(
            f"fuser {state.port}/tcp 2>/dev/null | tr -s ' ' '\\n' | head -1",
            timeout=5,
        )
        output = check.stdout.strip()

    if not output:
        logger.warning(
            "TCP proxy failed to start. Log: %s",
            _run_cmd(f"cat {log_path} 2>/dev/null", timeout=5).stdout.strip(),
        )
        return False

    # Parse PID
    try:
        pid = int(output.strip().split("\n")[-1])
        state.forward_pid = pid
        state.save()
    except (ValueError, IndexError):
        logger.debug("Could not parse proxy PID from: %s", output)

    logger.info(
        "TCP proxy established: login:%d → %s:%d (pid=%s)",
        state.port,
        state.node,
        state.port,
        state.forward_pid,
    )
    return True


def _kill_port_forward(state: SlurmJobState) -> None:
    """Kill any existing port forward (SSH tunnel or TCP proxy) for this job."""
    if state.forward_pid:
        _run_cmd(f"kill {state.forward_pid} 2>/dev/null")
        state.forward_pid = None

    # Also kill by pattern match (catch orphans — both SSH tunnels and TCP proxies)
    if state.node and state.port:
        _run_cmd(f"pkill -f 'ssh.*-L.*{state.port}.*{state.node}' 2>/dev/null")
        _run_cmd(f"pkill -f 'tcp-proxy-{state.port}.py' 2>/dev/null")
        # Clean up PID file
        _run_cmd(f"rm -f {REMOTE_STATE_DIR}/tcp-proxy-{state.port}.pid 2>/dev/null")


def _wait_for_job_start(
    state: SlurmJobState, timeout: float = JOB_START_TIMEOUT
) -> bool:
    """Wait for a pending SLURM job to start running.

    Returns True if job is running, False if timeout or failure.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        updated = get_job_status(state)
        if updated is None:
            logger.error("Job disappeared from queue")
            return False
        if updated.node:
            return True
        time.sleep(POLL_INTERVAL)

    logger.error("Timed out waiting for job %s to start (%ds)", state.job_id, timeout)
    return False


def _wait_for_server_ready(
    port: int = DEFAULT_PORT,
    timeout: float = SERVER_READY_TIMEOUT,
) -> bool:
    """Wait for the embedding server to respond to health checks.

    This checks localhost:PORT which is forwarded through to the compute node.
    """
    import httpx

    deadline = time.time() + timeout
    url = f"http://localhost:{port}/health"

    while time.time() < deadline:
        try:
            with httpx.Client(timeout=httpx.Timeout(5.0, connect=3.0)) as client:
                response = client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        logger.info("Embedding server ready on localhost:%d", port)
                        return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        except Exception as e:
            logger.debug("Health check error: %s", e)

        time.sleep(POLL_INTERVAL)

    logger.error("Timed out waiting for server to be ready (%ds)", timeout)
    return False


def ensure_server(
    port: int = DEFAULT_PORT,
    partition: str = DEFAULT_PARTITION,
    gpu_count: int = DEFAULT_GPU_COUNT,
    walltime: str = DEFAULT_WALLTIME,
    idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
    model_name: str | None = None,
) -> bool:
    """Ensure the SLURM embedding server is running and reachable.

    This is the main entry point for automatic lifecycle management.
    It will:
    1. Check for an existing running job and reuse it
    2. If no job, submit one and wait for it to start
    3. Set up port forwarding from login node to compute node
    4. Wait for the server to respond to health checks

    Args:
        port: Port for the embedding server
        partition: SLURM partition
        gpu_count: Number of GPUs
        walltime: Maximum wall time
        idle_timeout: Seconds of inactivity before auto-shutdown
        model_name: Embedding model to use (overrides remote pyproject.toml)

    Returns:
        True if server is ready, False on failure
    """
    logger.info("Ensuring SLURM embedding server is available...")

    # Step 1: Check for existing job
    state = SlurmJobState.load()
    if state.job_id:
        updated = get_job_status(state)
        if updated and updated.node:
            logger.info("Found running job %s on %s", updated.job_id, updated.node)
            # Verify existing server is actually our SLURM server
            if (
                _check_port_forward(updated)
                and _is_slurm_server(port)
                and _wait_for_server_ready(port, timeout=10)
            ):
                return True
            # Port forward missing or wrong server on port - refresh
            _setup_port_forward(updated)
            return _wait_for_server_ready(port, timeout=30)
        elif updated:
            # Job pending, wait for it
            logger.info("Job %s is pending, waiting...", state.job_id)
            if not _wait_for_job_start(updated):
                # Timed out, cancel and resubmit
                cancel_job(updated)
                state = SlurmJobState()
            else:
                _setup_port_forward(updated)
                return _wait_for_server_ready(port)
        # Job gone, need new one
        state = SlurmJobState()

    # Step 2: Submit new job
    logger.info("Submitting new SLURM job...")
    state = submit_job(
        port=port,
        partition=partition,
        gpu_count=gpu_count,
        walltime=walltime,
        idle_timeout=idle_timeout,
        model_name=model_name,
    )

    # Step 3: Wait for job to start
    if not _wait_for_job_start(state):
        cancel_job(state)
        return False

    # Step 4: Set up port forwarding
    if not _setup_port_forward(state):
        logger.error("Failed to set up port forwarding")
        return False

    # Step 5: Wait for server to be ready
    return _wait_for_server_ready(port)


def _check_port_forward(state: SlurmJobState) -> bool:
    """Check if port forwarding is still active."""
    if not state.forward_pid:
        return False

    result = _run_cmd(f"kill -0 {state.forward_pid} 2>/dev/null")
    return result.returncode == 0


def cancel_job(state: SlurmJobState | None = None) -> bool:
    """Cancel the SLURM embedding server job.

    Args:
        state: Job state (loads from disk if None)

    Returns:
        True if job was cancelled
    """
    if state is None:
        state = SlurmJobState.load()

    if not state.job_id:
        logger.info("No active SLURM job to cancel")
        return False

    _kill_port_forward(state)

    result = _run_cmd(f"scancel {state.job_id}")
    if result.returncode == 0:
        logger.info("Cancelled SLURM job %s", state.job_id)
    else:
        logger.warning(
            "Failed to cancel job %s: %s", state.job_id, result.stderr.strip()
        )

    state.clear()
    return True


def get_logs(state: SlurmJobState | None = None, tail: int = 50) -> str:
    """Get logs from the SLURM embedding server.

    Args:
        state: Job state (loads from disk if None)
        tail: Number of lines from the end

    Returns:
        Log content as string
    """
    if state is None:
        state = SlurmJobState.load()

    if not state.job_id:
        return "No active SLURM job"

    log_path = f"{REMOTE_STATE_DIR}/slurm-embed-{state.job_id}.log"
    result = _run_cmd(f"tail -n {tail} {log_path} 2>/dev/null")
    return result.stdout if result.returncode == 0 else f"No log file: {log_path}"


__all__ = [
    "SlurmJobState",
    "_is_slurm_server",
    "cancel_job",
    "ensure_server",
    "get_job_status",
    "get_logs",
    "submit_job",
]

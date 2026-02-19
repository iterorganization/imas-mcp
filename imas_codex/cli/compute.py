"""CLI commands for managing SLURM HPC jobs.

Provides interactive and batch job management to offload heavy work
(imas build, ingest, discover) from the shared login node to dedicated
SLURM compute nodes.

Architecture:
    Titan node (98dci4-gpu-0002): embedding server (P100 GPU)
    Login node: Neo4j (Apptainer), zellij sessions
    Compute nodes: imas build, ingest, discover, wiki scraping

Compute nodes access services over the network:
    - Embedding server: http://<embed-host>:18765 (Titan or login)
    - Neo4j: bolt://<login-hostname>:7687
    - Home directory: shared GPFS mount
"""

import logging
import os
import shutil
import socket
import subprocess

import click

logger = logging.getLogger(__name__)

# Default SLURM configuration for imas-codex workloads.
# Our tasks are overwhelmingly IO-bound (network calls to embedding server,
# Neo4j writes, SSH to remote facilities, HTTP scraping).  CPU profiling
# shows <25% user CPU even during peak `imas build`.  4 CPUs is plenty
# for the Python process + a few async workers.
DEFAULT_PARTITION = "rigel"
DEFAULT_CPUS = 4
DEFAULT_MEM_GB = 32
DEFAULT_JOB_NAME = "codex"


def _get_login_hostname() -> str:
    """Get the login node hostname for service URLs.

    When running on the login node, returns the current hostname.
    When running on a compute node (SLURM job), uses SLURM_SUBMIT_HOST
    which is the hostname from which the job was submitted.
    """
    submit_host = os.environ.get("SLURM_SUBMIT_HOST")
    if submit_host:
        return submit_host
    return socket.gethostname()


def _is_on_login_node() -> bool:
    """Check if we're running on the login node (not in a SLURM job)."""
    return not bool(os.environ.get("SLURM_JOB_ID"))


def _slurm_available() -> bool:
    """Check if SLURM commands are available."""
    return shutil.which("srun") is not None


def _get_active_jobs(job_name: str | None = None) -> list[dict]:
    """Get active SLURM jobs for current user.

    Args:
        job_name: Filter by job name (optional)

    Returns:
        List of dicts with job info
    """
    cmd = [
        "squeue",
        "-u",
        os.environ.get("USER", ""),
        "--format=%A|%j|%P|%T|%M|%N|%C|%m",
        "--noheader",
    ]
    if job_name:
        cmd.extend(["--name", job_name])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        jobs = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split("|")
            if len(parts) >= 8:
                jobs.append(
                    {
                        "job_id": parts[0].strip(),
                        "name": parts[1].strip(),
                        "partition": parts[2].strip(),
                        "state": parts[3].strip(),
                        "time": parts[4].strip(),
                        "node": parts[5].strip(),
                        "cpus": parts[6].strip(),
                        "memory": parts[7].strip(),
                    }
                )
        return jobs
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def _build_env_exports(login_hostname: str) -> str:
    """Build environment variable exports for SLURM jobs.

    Sets up the compute node environment so imas-codex commands can
    reach the embedding server and Neo4j instance.
    Uses the active graph profile for port and password resolution.
    The embed server may run on a different host (e.g. Titan).
    """
    from imas_codex.graph.profiles import resolve_neo4j
    from imas_codex.settings import get_embed_host

    profile = resolve_neo4j()
    # Embed server may be on a dedicated GPU node (e.g. Titan)
    embed_host = get_embed_host() or login_hostname
    return (
        f'export IMAS_CODEX_EMBED_REMOTE_URL="http://{embed_host}:18765"\n'
        f'export NEO4J_URI="bolt://{login_hostname}:{profile.bolt_port}"\n'
        f'export NEO4J_PASSWORD="{profile.password}"\n'
        f'export IMAS_CODEX_GRAPH="{profile.name}"\n'
        'export IMAS_CODEX_EMBEDDING_BACKEND="remote"\n'
    )


@click.group()
def hpc():
    """Manage SLURM HPC jobs for heavy workloads.

    Offloads CPU-intensive tasks (imas build, ingest, discover) to
    dedicated SLURM compute nodes, keeping the shared login node
    responsive.

    \b
    Examples:
        imas-codex hpc status           Show active HPC jobs
        imas-codex hpc run -- imas-codex imas build
        imas-codex hpc shell            Interactive shell on compute node
        imas-codex hpc attach           Attach to running compute job
    """
    pass


@hpc.command("status")
def hpc_status():
    """Show active SLURM HPC jobs.

    Lists all running and pending jobs for the current user,
    highlighting imas-codex related jobs.
    """
    if not _slurm_available():
        raise click.ClickException(
            "SLURM not available. Are you on the ITER HPC login node?"
        )

    jobs = _get_active_jobs()
    if not jobs:
        click.echo("No active SLURM jobs.")
        click.echo()
        click.echo("Start one with:")
        click.echo("  imas-codex hpc shell            # Interactive shell")
        click.echo("  imas-codex hpc run -- <command>  # Run a command")
        return

    click.echo(
        f"{'JOBID':<12} {'NAME':<15} {'PARTITION':<12} {'STATE':<10} "
        f"{'TIME':<12} {'NODE':<25} {'CPUS':<6} {'MEM'}"
    )
    click.echo("-" * 100)
    for job in jobs:
        marker = " *" if DEFAULT_JOB_NAME in job["name"] else ""
        click.echo(
            f"{job['job_id']:<12} {job['name']:<15} {job['partition']:<12} "
            f"{job['state']:<10} {job['time']:<12} {job['node']:<25} "
            f"{job['cpus']:<6} {job['memory']}{marker}"
        )


@hpc.command("shell")
@click.option(
    "--partition",
    "-p",
    default=DEFAULT_PARTITION,
    help=f"SLURM partition (default: {DEFAULT_PARTITION})",
)
@click.option(
    "--cpus",
    "-c",
    default=DEFAULT_CPUS,
    type=int,
    help=f"Number of CPUs (default: {DEFAULT_CPUS})",
)
@click.option(
    "--mem",
    "-m",
    default=DEFAULT_MEM_GB,
    type=int,
    help=f"Memory in GB (default: {DEFAULT_MEM_GB})",
)
@click.option(
    "--name",
    "-n",
    default=DEFAULT_JOB_NAME,
    help=f"Job name (default: {DEFAULT_JOB_NAME})",
)
@click.option(
    "--exclusive/--shared",
    default=False,
    help="Reserve an entire node exclusively, or share (default)",
)
def hpc_shell(
    partition: str,
    cpus: int,
    mem: int,
    name: str,
    exclusive: bool,
) -> None:
    """Start an interactive shell on a SLURM compute node.

    Allocates a compute node and drops you into an interactive bash
    shell with environment configured for imas-codex services.

    The shell has access to:
    - Shared GPFS home directory (same files as login node)
    - Login node's embedding server (via network)
    - Login node's Neo4j instance (via network)
    - Internet access (for wiki scraping, pip, etc.)

    SLURM cgroups enforce resource isolation per-job.  Other users on
    the same node cannot access your processes or memory.  Use
    --exclusive only if you need the full 28 CPUs / 128GB.

    \b
    Examples:
        imas-codex hpc shell                    # 4 CPUs, 32GB (default)
        imas-codex hpc shell -c 8 -m 64         # More resources
        imas-codex hpc shell --exclusive         # Full node
        imas-codex hpc shell -p rigel_debug      # Debug partition (1h max)
    """
    if not _slurm_available():
        raise click.ClickException(
            "SLURM not available. Are you on the ITER HPC login node?"
        )

    login_hostname = _get_login_hostname()

    cmd = [
        "srun",
        f"--partition={partition}",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}G",
        f"--job-name={name}",
        "--pty",
    ]
    if exclusive:
        cmd.append("--exclusive")

    # Build init script that sets up environment and drops into bash
    init_script = (
        "bash --init-file <(echo '"
        f"{_build_env_exports(login_hostname)}"
        'echo "Compute node: $(hostname) | '
        f"CPUs: {cpus} | Mem: {mem}GB | "
        f'Login: {login_hostname}"\n'
        f'echo "Embed server: $IMAS_CODEX_EMBED_REMOTE_URL"\n'
        f'echo "Neo4j: $NEO4J_URI"\n'
        "cd ~/Code/imas-codex\n"
        "source ~/.bashrc 2>/dev/null\n')"
    )

    cmd.extend(["bash", "-c", init_script])

    click.echo(
        f"Requesting {cpus} CPUs, {mem}GB RAM on {partition}"
        f"{' (exclusive)' if exclusive else ''}..."
    )
    from imas_codex.graph.profiles import BOLT_BASE_PORT
    from imas_codex.settings import get_embed_host, get_embed_server_port

    embed_port = get_embed_server_port()
    embed_host = get_embed_host() or login_hostname
    click.echo(
        f"Services: embed={embed_host}:{embed_port}, neo4j={login_hostname}:{BOLT_BASE_PORT}"
    )

    os.execvp("srun", cmd)


@hpc.command("run")
@click.option(
    "--partition",
    "-p",
    default=DEFAULT_PARTITION,
    help=f"SLURM partition (default: {DEFAULT_PARTITION})",
)
@click.option(
    "--cpus",
    "-c",
    default=DEFAULT_CPUS,
    type=int,
    help=f"Number of CPUs (default: {DEFAULT_CPUS})",
)
@click.option(
    "--mem",
    "-m",
    default=DEFAULT_MEM_GB,
    type=int,
    help=f"Memory in GB (default: {DEFAULT_MEM_GB})",
)
@click.option(
    "--name",
    "-n",
    default=DEFAULT_JOB_NAME,
    help=f"Job name (default: {DEFAULT_JOB_NAME})",
)
@click.argument("command", nargs=-1, required=True)
def hpc_run(
    partition: str,
    cpus: int,
    mem: int,
    name: str,
    command: tuple[str, ...],
) -> None:
    """Run a command on a SLURM compute node.

    Submits a foreground job that runs the specified command with
    environment configured for imas-codex services.  Output streams
    directly to your terminal.

    \b
    Examples:
        imas-codex hpc run -- imas-codex imas build
        imas-codex hpc run -- imas-codex ingest run tcv
        imas-codex hpc run -- imas-codex discover wiki jt-60sa -c 25
        imas-codex hpc run -c 2 -m 16 -- uv run pytest tests/
    """
    if not _slurm_available():
        raise click.ClickException(
            "SLURM not available. Are you on the ITER HPC login node?"
        )

    login_hostname = _get_login_hostname()
    cmd_str = " ".join(command)

    click.echo(f"Submitting to {partition} ({cpus} CPUs, {mem}GB): {cmd_str}")

    srun_cmd = [
        "srun",
        f"--partition={partition}",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}G",
        f"--job-name={name}",
        "bash",
        "-c",
        (f"{_build_env_exports(login_hostname)}cd ~/Code/imas-codex && {cmd_str}"),
    ]

    os.execvp("srun", srun_cmd)


@hpc.command("submit")
@click.option(
    "--partition",
    "-p",
    default=DEFAULT_PARTITION,
    help=f"SLURM partition (default: {DEFAULT_PARTITION})",
)
@click.option(
    "--cpus",
    "-c",
    default=DEFAULT_CPUS,
    type=int,
    help=f"Number of CPUs (default: {DEFAULT_CPUS})",
)
@click.option(
    "--mem",
    "-m",
    default=DEFAULT_MEM_GB,
    type=int,
    help=f"Memory in GB (default: {DEFAULT_MEM_GB})",
)
@click.option(
    "--name",
    "-n",
    default=DEFAULT_JOB_NAME,
    help=f"Job name (default: {DEFAULT_JOB_NAME})",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output log file (default: codex-<jobid>.log)",
)
@click.argument("command", nargs=-1, required=True)
def hpc_submit(
    partition: str,
    cpus: int,
    mem: int,
    name: str,
    output: str | None,
    command: tuple[str, ...],
) -> None:
    """Submit a background batch job to SLURM.

    Unlike 'run', this returns immediately and the job runs in the
    background.  Output is written to a log file.

    \b
    Examples:
        imas-codex hpc submit -- imas-codex imas build
        imas-codex hpc submit -o build.log -- imas-codex imas build --force
    """
    if not _slurm_available():
        raise click.ClickException(
            "SLURM not available. Are you on the ITER HPC login node?"
        )

    login_hostname = _get_login_hostname()
    cmd_str = " ".join(command)
    output_file = output or f"{name}-%j.log"

    script = (
        "#!/bin/bash\n"
        f"#SBATCH --partition={partition}\n"
        f"#SBATCH --cpus-per-task={cpus}\n"
        f"#SBATCH --mem={mem}G\n"
        f"#SBATCH --job-name={name}\n"
        f"#SBATCH --output={output_file}\n"
        "\n"
        f"{_build_env_exports(login_hostname)}"
        "cd ~/Code/imas-codex\n"
        f"{cmd_str}\n"
    )

    result = subprocess.run(
        ["sbatch"],
        input=script,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        raise click.ClickException(f"sbatch failed: {result.stderr.strip()}")

    click.echo(result.stdout.strip())
    click.echo(f"Output: {output_file}")


@hpc.command("attach")
@click.argument("job_id", required=False)
def hpc_attach(job_id: str | None) -> None:
    """Attach to a running interactive SLURM job.

    Without a job ID, attaches to the most recent 'codex' job.
    Uses sattach for running jobs.

    \b
    Examples:
        imas-codex hpc attach           # Most recent codex job
        imas-codex hpc attach 893042    # Specific job
    """
    if not _slurm_available():
        raise click.ClickException(
            "SLURM not available. Are you on the ITER HPC login node?"
        )

    if not job_id:
        # Find the most recent codex job
        jobs = _get_active_jobs(DEFAULT_JOB_NAME)
        running = [j for j in jobs if j["state"] == "RUNNING"]
        if not running:
            raise click.ClickException(
                f"No running '{DEFAULT_JOB_NAME}' jobs found.\n"
                "Start one with: imas-codex hpc shell"
            )
        job_id = running[0]["job_id"]
        click.echo(f"Attaching to job {job_id} on {running[0]['node']}...")

    os.execvp("sattach", ["sattach", f"{job_id}.0"])


@hpc.command("cancel")
@click.argument("job_id", required=False)
@click.option("--all", "cancel_all", is_flag=True, help="Cancel all codex jobs")
def hpc_cancel(job_id: str | None, cancel_all: bool) -> None:
    """Cancel SLURM HPC jobs.

    \b
    Examples:
        imas-codex hpc cancel           # Cancel most recent codex job
        imas-codex hpc cancel 893042    # Cancel specific job
        imas-codex hpc cancel --all     # Cancel all codex jobs
    """
    if not _slurm_available():
        raise click.ClickException(
            "SLURM not available. Are you on the ITER HPC login node?"
        )

    if cancel_all:
        jobs = _get_active_jobs(DEFAULT_JOB_NAME)
        if not jobs:
            click.echo("No active codex jobs to cancel.")
            return
        for job in jobs:
            subprocess.run(["scancel", job["job_id"]], check=True)
            click.echo(f"Cancelled job {job['job_id']} ({job['state']})")
        return

    if not job_id:
        jobs = _get_active_jobs(DEFAULT_JOB_NAME)
        if not jobs:
            raise click.ClickException("No active codex jobs found.")
        job_id = jobs[0]["job_id"]

    subprocess.run(["scancel", job_id], check=True)
    click.echo(f"Cancelled job {job_id}")


@hpc.command("info")
def hpc_info() -> None:
    """Show SLURM cluster information relevant to imas-codex.

    Displays partition availability, idle nodes, and recommended
    configurations for different workload types.
    """
    if not _slurm_available():
        raise click.ClickException(
            "SLURM not available. Are you on the ITER HPC login node?"
        )

    click.echo("SLURM Cluster Overview")
    click.echo("=" * 60)

    # Get partition info
    result = subprocess.run(
        [
            "sinfo",
            "--format=%P|%a|%l|%D|%T|%c|%m",
            "--noheader",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )

    # Parse and display key partitions
    partitions: dict[str, dict] = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.strip().split("|")
        if len(parts) >= 7:
            name = parts[0].strip().rstrip("*")
            state = parts[4].strip()
            count = int(parts[3].strip()) if parts[3].strip().isdigit() else 0
            if name not in partitions:
                partitions[name] = {
                    "avail": parts[1].strip(),
                    "timelimit": parts[2].strip(),
                    "cpus": parts[5].strip(),
                    "mem": parts[6].strip(),
                    "idle": 0,
                    "mix": 0,
                    "alloc": 0,
                    "total": 0,
                }
            partitions[name]["total"] += count
            if "idle" in state:
                partitions[name]["idle"] += count
            elif "mix" in state:
                partitions[name]["mix"] += count
            elif "alloc" in state:
                partitions[name]["alloc"] += count

    click.echo(
        f"\n{'PARTITION':<15} {'AVAIL':<8} {'TIME LIMIT':<12} "
        f"{'IDLE':<6} {'MIX':<6} {'ALLOC':<6} {'TOTAL':<6}"
    )
    click.echo("-" * 65)
    for name, info in sorted(partitions.items()):
        click.echo(
            f"{name:<15} {info['avail']:<8} {info['timelimit']:<12} "
            f"{info['idle']:<6} {info['mix']:<6} {info['alloc']:<6} "
            f"{info['total']:<6}"
        )

    from imas_codex.graph.profiles import BOLT_BASE_PORT
    from imas_codex.settings import get_embed_host, get_embed_server_port

    embed_port = get_embed_server_port()
    embed_host = get_embed_host()
    login_hn = _get_login_hostname()
    click.echo(f"\nLogin node: {login_hn}")
    embed_display = embed_host or login_hn
    click.echo(f"Embed server: http://{embed_display}:{embed_port}")
    click.echo(f"Neo4j: bolt://{login_hn}:{BOLT_BASE_PORT}")

    click.echo("\nRecommended configurations:")
    click.echo(
        "  imas build:      -c 4  -m 32  (IO-bound: embed server + Neo4j writes)"
    )
    click.echo("  ingest run:      -c 4  -m 32  (IO-bound: SSH + embed + Neo4j)")
    click.echo("  discover paths:  -c 4  -m 32  (--scan-workers 2 --score-workers 4)")
    click.echo("  discover wiki:   -c 4  -m 32  (--score-workers 6 --ingest-workers 8)")
    click.echo("  discover signal: -c 4  -m 32  (--enrich-workers 4 --check-workers 2)")
    click.echo("  cx session:      -c 4  -m 32  (interactive development)")
    click.echo("  pytest:          -c 8  -m 32  (CPU-bound test suite)")

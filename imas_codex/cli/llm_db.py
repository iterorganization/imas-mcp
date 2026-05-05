"""PostgreSQL database management for LiteLLM proxy.

Uses pgserver to provide user-space PostgreSQL without root access.
Data stored at ``~/.local/share/imas-codex/services/pgdata/``.

All commands run on the LLM host node (remote via SSH or local).

pgserver bundles pre-compiled PostgreSQL binaries — no system
installation needed.  First run installs via ``uv run --with pgserver``
(~30 seconds), subsequent runs use the uv cache.

Usage::

    imas-codex llm db init      # One-time: install pgserver, create database
    imas-codex llm db start     # Start PostgreSQL server
    imas-codex llm db stop      # Stop PostgreSQL server
    imas-codex llm db status    # Check server status
    imas-codex llm db uri       # Print connection URI
"""

import logging
import os
import secrets
import subprocess

import click

logger = logging.getLogger(__name__)

_PGDATA_REL = "pgdata"

# Script to locate pg_ctl from pgserver's bundled PostgreSQL.
# Output: the directory containing pg_ctl, initdb, createdb, psql.
_FIND_PG_BIN = (
    "import pathlib, pgserver; "
    "p = pathlib.Path(pgserver.__file__).parent; "
    "bins = list(p.rglob('pg_ctl')); "
    "print(str(bins[0].parent) if bins else 'NOT_FOUND')"
)


def _pgdata() -> str:
    """Remote pgdata path (shell-expandable)."""
    from imas_codex.cli.services import _SERVICES_DIR

    return f"{_SERVICES_DIR}/{_PGDATA_REL}"


# Socket directory under /run/user/<uid> — survives reboots but not
# /tmp cleanup cron (which purges /tmp every 10 days on ITER).
_PG_SOCKET_DIR = "/run/user/$(id -u)/imas-codex"


def _pg_port() -> int:
    from imas_codex.settings import get_postgres_port

    return get_postgres_port()


def _run(cmd: str, timeout: int = 30, check: bool = False) -> str:
    """Run command on the LLM host (SSH or local)."""
    from imas_codex.cli.services import _run_llm_remote

    return _run_llm_remote(cmd, timeout=timeout, check=check)


def _ensure_pg_bin() -> str:
    """Get path to PostgreSQL binaries on the LLM host.

    Installs pgserver on first run (provides bundled PostgreSQL) and
    caches the binary path at ``~/.local/share/imas-codex/services/.pg_bin``.
    """
    from imas_codex.cli.services import _SERVICES_DIR

    bin_cache = f"{_SERVICES_DIR}/.pg_bin"

    # Check cached path
    try:
        result = _run(f"cat {bin_cache} 2>/dev/null", timeout=5)
        path = result.strip()
        if path:
            _run(f"test -x {path}/pg_ctl", timeout=5)
            return path
    except subprocess.CalledProcessError:
        pass

    # Install pgserver and locate binaries
    click.echo("  Installing PostgreSQL binaries (pgserver)...")
    from imas_codex.cli.services import _llm_ssh
    from imas_codex.remote.executor import run_script_via_stdin

    wrapper = (
        "uv run --python 3.12 --with pgserver python3 - << 'PYEOF'\n"
        f"{_FIND_PG_BIN}\n"
        "PYEOF\n"
    )
    result = run_script_via_stdin(
        wrapper,
        ssh_host=_llm_ssh(),
        timeout=300,
        lang="bash",
    )
    # Last non-empty line has the path (uv may print progress before it)
    pg_bin = ""
    for line in result.strip().split("\n"):
        line = line.strip()
        if line and "/" in line and "NOT_FOUND" not in line:
            pg_bin = line

    if not pg_bin:
        raise click.ClickException(
            "Could not find PostgreSQL binaries in pgserver.\n"
            "Install manually: uv run --with pgserver python3 -c "
            '"import pgserver; print(pgserver.__file__)"'
        )

    # Cache for subsequent calls
    _run(
        f'mkdir -p "$(dirname {bin_cache})" && echo "{pg_bin}" > {bin_cache}',
        timeout=5,
    )
    click.echo(f"  Binaries: {pg_bin}")
    return pg_bin


def _pg_running() -> bool:
    """Check if PostgreSQL postmaster PID file exists and process is live."""
    pgdata = _pgdata()
    try:
        result = _run(
            f"test -f {pgdata}/postmaster.pid && "
            f"kill -0 $(head -1 {pgdata}/postmaster.pid) 2>/dev/null && "
            f"echo running",
            timeout=5,
        )
        return "running" in result
    except subprocess.CalledProcessError:
        return False


def start_db(quiet: bool = False) -> bool:
    """Start PostgreSQL if not running.

    Returns True if started (or already running), False on failure.
    Called internally by ``llm start`` before launching the proxy.
    """
    if _pg_running():
        if not quiet:
            click.echo("  PostgreSQL: running")
        return True

    pgdata = _pgdata()
    try:
        _run(f"test -f {pgdata}/PG_VERSION", timeout=5)
    except subprocess.CalledProcessError:
        if not quiet:
            click.echo("  PostgreSQL: not initialized")
            click.echo("    Run: imas-codex llm db init")
        return False

    pg_bin = _ensure_pg_bin()
    port = _pg_port()
    if not quiet:
        click.echo(f"  Starting PostgreSQL (port {port})...")

    try:
        # Clean stale PID file if process is dead
        _run(
            f"test -f {pgdata}/postmaster.pid && "
            f"! kill -0 $(head -1 {pgdata}/postmaster.pid) 2>/dev/null && "
            f"rm -f {pgdata}/postmaster.pid; true",
            timeout=5,
        )
        # Ensure socket dir exists (tmpfs, survives reboot but not /tmp purge)
        _run(f"mkdir -p {_PG_SOCKET_DIR}", timeout=5)
        _run(
            f"mkdir -p {pgdata}/log && "
            f"{pg_bin}/pg_ctl start -D {pgdata} -w "
            f"-l {pgdata}/log/postgresql.log",
            timeout=30,
            check=True,
        )
        if not quiet:
            click.echo("  PostgreSQL: started")
        return True
    except subprocess.CalledProcessError as e:
        if not quiet:
            click.echo(f"  PostgreSQL: failed to start — {e}")
        return False


def stop_db(quiet: bool = False) -> bool:
    """Stop PostgreSQL if running.

    Returns True if stopped (or already stopped), False on failure.
    """
    if not _pg_running():
        if not quiet:
            click.echo("  PostgreSQL: not running")
        return True

    pg_bin = _ensure_pg_bin()
    pgdata = _pgdata()
    if not quiet:
        click.echo("  Stopping PostgreSQL...")

    try:
        _run(
            f"{pg_bin}/pg_ctl stop -D {pgdata} -w -m fast",
            timeout=30,
            check=True,
        )
        if not quiet:
            click.echo("  PostgreSQL: stopped")
        return True
    except subprocess.CalledProcessError as e:
        if not quiet:
            click.echo(f"  PostgreSQL: failed to stop — {e}")
        return False


# ── Click command group ──────────────────────────────────────────────────


@click.group("db")
def llm_db() -> None:
    """Manage PostgreSQL database for LiteLLM proxy.

    Uses pgserver for user-space PostgreSQL — no root access needed.
    Data stored at ~/.local/share/imas-codex/services/pgdata/.

    \b
      imas-codex llm db init     Install pgserver + create database
      imas-codex llm db start    Start PostgreSQL server
      imas-codex llm db stop     Stop PostgreSQL server
      imas-codex llm db status   Check PostgreSQL server status
      imas-codex llm db uri      Print database connection URI
    """


@llm_db.command("init")
@click.option("--port", type=int, default=None, help="PostgreSQL port (default: 18450)")
@click.option("--user", default=None, help="Database user (default: OS user)")
@click.option("--password", default=None, help="User password (default: generated)")
@click.option("--database", default="litellm", help="Database name")
def db_init(
    port: int | None,
    user: str | None,
    password: str | None,
    database: str,
) -> None:
    """Initialize PostgreSQL: install pgserver binaries, create database.

    Run once per machine.  After init, use ``llm db start`` to start the
    server and ``llm start`` to bring up the full proxy stack.

    \b
    Examples:
        imas-codex llm db init
        imas-codex llm db init --port 18450 --database litellm
    """
    import getpass

    if port is None:
        port = _pg_port()
    if user is None:
        user = getpass.getuser()
    if password is None:
        password = secrets.token_hex(16)

    pgdata = _pgdata()

    # Already initialized?
    try:
        result = _run(f"test -f {pgdata}/PG_VERSION && echo exists", timeout=5)
        if "exists" in result:
            click.echo(f"  Already initialized: {pgdata}")
            click.echo("  Use: imas-codex llm db start")
            return
    except subprocess.CalledProcessError:
        pass

    click.echo("Initializing PostgreSQL database...")
    click.echo(f"  Data dir:  {pgdata}")
    click.echo(f"  Port:      {port}")
    click.echo(f"  Database:  {database}")
    click.echo(f"  User:      {user}")

    # Get PostgreSQL binaries from pgserver
    pg_bin = _ensure_pg_bin()

    # initdb
    click.echo("  Running initdb...")
    _run(
        f"mkdir -p {pgdata} && "
        f"{pg_bin}/initdb -D {pgdata} --auth=trust --encoding=UTF8 --no-locale",
        timeout=60,
        check=True,
    )

    # Configure postgresql.conf — append our settings
    click.echo("  Configuring...")
    socket_dir = _PG_SOCKET_DIR
    conf_script = (
        f"cat >> {pgdata}/postgresql.conf << 'PGEOF'\n"
        f"# imas-codex managed settings\n"
        f"port = {port}\n"
        f"listen_addresses = 'localhost'\n"
        f"unix_socket_directories = '{socket_dir}'\n"
        f"logging_collector = on\n"
        f"log_directory = 'log'\n"
        f"PGEOF"
    )
    _run(conf_script, timeout=5, check=True)

    # Configure pg_hba.conf — local trust + network md5
    hba_script = (
        f"cat > {pgdata}/pg_hba.conf << 'PGEOF'\n"
        f"# TYPE  DATABASE  USER  ADDRESS       METHOD\n"
        f"local   all       all                 trust\n"
        f"host    all       all   127.0.0.1/32  md5\n"
        f"host    all       all   ::1/128       md5\n"
        f"PGEOF"
    )
    _run(hba_script, timeout=5, check=True)

    # Start temporarily for setup
    click.echo("  Creating database and user...")
    _run(f"mkdir -p {_PG_SOCKET_DIR}", timeout=5)
    _run(
        f"mkdir -p {pgdata}/log && "
        f"{pg_bin}/pg_ctl start -D {pgdata} -w "
        f"-l {pgdata}/log/postgresql.log",
        timeout=30,
        check=True,
    )

    # Create database (may already exist from initdb template)
    # Connect via unix socket (trust auth) — network auth not set up yet
    _run(
        f"{pg_bin}/createdb -p {port} -h {_PG_SOCKET_DIR} {database} 2>/dev/null || true",
        timeout=10,
    )

    # Set user password for network connections
    _run(
        f"{pg_bin}/psql -p {port} -h {_PG_SOCKET_DIR} -c "
        f""""ALTER USER \\"{user}\\" WITH PASSWORD '{password}';" """,
        timeout=10,
    )

    # Stop server (user starts it with `llm db start` or `llm start`)
    _run(f"{pg_bin}/pg_ctl stop -D {pgdata} -w", timeout=15)

    uri = f"postgresql://{user}:{password}@localhost:{port}/{database}"
    click.echo()
    click.echo(click.style("  ✓ Database initialized", fg="green"))
    click.echo(f"  URI: {uri}")
    click.echo()
    click.echo("  Add to .env on the LLM host:")
    click.echo(f"    LITELLM_DATABASE_URL={uri}")
    click.echo()
    click.echo("  Then: imas-codex llm start")


@llm_db.command("start")
def db_start_cmd() -> None:
    """Start PostgreSQL server.

    Automatically called by ``llm start`` when LITELLM_DATABASE_URL is set.

    \b
    Examples:
        imas-codex llm db start
    """
    start_db()


@llm_db.command("stop")
def db_stop_cmd() -> None:
    """Stop PostgreSQL server.

    \b
    Examples:
        imas-codex llm db stop
    """
    stop_db()


@llm_db.command("status")
def db_status_cmd() -> None:
    """Check PostgreSQL server status.

    \b
    Examples:
        imas-codex llm db status
    """
    pgdata = _pgdata()
    port = _pg_port()

    # Check if initialized
    try:
        _run(f"test -f {pgdata}/PG_VERSION", timeout=5)
    except subprocess.CalledProcessError:
        click.echo(click.style("  ✗ Not initialized", fg="red"))
        click.echo("    Run: imas-codex llm db init")
        return

    if _pg_running():
        click.echo(click.style(f"  ✓ Running (port {port})", fg="green"))
        # Show PID
        try:
            pid = _run(f"head -1 {pgdata}/postmaster.pid", timeout=5).strip()
            click.echo(f"  PID: {pid}")
        except subprocess.CalledProcessError:
            pass
        # Show database size
        try:
            pg_bin = _ensure_pg_bin()
            result = _run(
                f"{pg_bin}/psql -p {port} -h {_PG_SOCKET_DIR} -t -c "
                f"\"SELECT pg_database_size('litellm')\" 2>/dev/null",
                timeout=10,
            )
            size_bytes = int(result.strip())
            if size_bytes > 1024 * 1024:
                click.echo(f"  Size: {size_bytes / 1024 / 1024:.1f} MB")
            else:
                click.echo(f"  Size: {size_bytes / 1024:.0f} KB")
        except Exception:
            pass
    else:
        click.echo(click.style("  ✗ Not running", fg="yellow"))
        click.echo("    Start: imas-codex llm db start")


@llm_db.command("uri")
def db_uri_cmd() -> None:
    """Print the database connection URI from environment.

    \b
    Examples:
        imas-codex llm db uri
    """
    url = os.environ.get("LITELLM_DATABASE_URL", "")
    if url:
        click.echo(url)
    else:
        # Try reading from remote .env
        from imas_codex.cli.services import _PROJECT

        try:
            result = _run(
                f"grep LITELLM_DATABASE_URL {_PROJECT}/.env 2>/dev/null | head -1",
                timeout=5,
            )
            if "=" in result:
                val = result.split("=", 1)[1].strip().strip('"')
                click.echo(val)
                return
        except subprocess.CalledProcessError:
            pass
        click.echo("LITELLM_DATABASE_URL not set")

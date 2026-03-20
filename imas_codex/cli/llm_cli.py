"""LLM proxy server management commands."""

import logging
import os
import subprocess

import click

logger = logging.getLogger(__name__)


def _check_file_permissions() -> None:
    """Warn about insecure file permissions on sensitive files."""
    import stat
    from pathlib import Path

    sensitive_files = [
        Path.cwd() / ".env",
        Path.home() / ".local" / "share" / "imas-codex" / "services" / "litellm.db",
    ]
    for path in sensitive_files:
        if not path.exists():
            continue
        mode = path.stat().st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
            click.echo(
                click.style(
                    f"  ⚠ {path.name} is world/group readable — run: chmod 600 {path}",
                    fg="yellow",
                )
            )


@click.group()
def llm() -> None:
    """Manage LiteLLM proxy server for centralized LLM routing.

    Routes all LLM calls through a single proxy with cost tracking via Langfuse.
    Provides OpenAI-compatible endpoint for agent teams and discovery workers.

    Runs on the login node via systemd (needs outbound HTTPS for API providers).

    \b
      imas-codex llm start              Start the proxy (systemd or foreground)
      imas-codex llm stop               Stop the proxy
      imas-codex llm restart            Restart (stop + start)
      imas-codex llm status             Check proxy health
      imas-codex llm logs               View service logs
      imas-codex llm service            Manage systemd service
      imas-codex llm setup              Initial team/key provisioning
      imas-codex llm keys list          List virtual API keys
      imas-codex llm keys create        Generate a new virtual key
      imas-codex llm keys revoke        Revoke (delete) a key
      imas-codex llm keys rotate        Rotate a key
      imas-codex llm teams list         List teams
      imas-codex llm teams create       Create a new team
      imas-codex llm teams info         Show team details
      imas-codex llm spend              View per-team spend
      imas-codex llm security audit     Audit security posture
      imas-codex llm security harden    Apply security fixes
      imas-codex llm local start        Start local LLM on Titan
      imas-codex llm local stop         Stop local LLM
      imas-codex llm local status       Check local LLM health
      imas-codex llm local models       List local models
    """
    pass


@llm.command("start")
@click.option(
    "--foreground",
    "-f",
    is_flag=True,
    help="Run proxy in foreground (for debugging)",
)
@click.option(
    "--host",
    envvar="LITELLM_HOST",
    default="0.0.0.0",
    help="Host to bind [foreground only] (default: 0.0.0.0)",
)
@click.option(
    "--port",
    envvar="LITELLM_PORT",
    default=None,
    type=int,
    help="Port to bind (default: from pyproject.toml [llm].port)",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Logging level [foreground only]",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to LiteLLM config YAML [foreground only] (default: bundled config)",
)
def llm_start(
    foreground: bool,
    host: str,
    port: int | None,
    log_level: str,
    config_path: str | None,
) -> None:
    """Start the LiteLLM proxy server.

    By default, deploys via systemd on the login node and waits for
    the health check to pass.  Use ``--foreground`` to run the proxy
    directly (for debugging).

    Requires OPENROUTER_API_KEY and LITELLM_MASTER_KEY environment variables.

    \b
    Examples:
        imas-codex llm start             # Deploy via systemd
        imas-codex llm start -f          # Run in foreground
        imas-codex llm start --port 19000  # Custom port
    """
    if foreground:
        _start_llm_foreground(host, port, log_level, config_path)
    else:
        _start_llm_deploy()


def _start_llm_foreground(
    host: str,
    port: int | None,
    log_level: str,
    config_path: str | None,
) -> None:
    """Run the LLM proxy in the foreground."""
    import shutil
    import subprocess
    from pathlib import Path

    from imas_codex.settings import get_llm_proxy_port

    if port is None:
        port = get_llm_proxy_port()

    uv_path = shutil.which("uv")
    if not uv_path:
        raise click.ClickException(
            "uv CLI not found. Install from: https://docs.astral.sh/uv/"
        )

    if config_path is None:
        config_path = str(
            Path(__file__).parent.parent / "config" / "litellm_config.yaml"
        )
        if not Path(config_path).exists():
            raise click.ClickException(f"Bundled config not found: {config_path}")

    if not os.environ.get("OPENROUTER_API_KEY_IMAS_CODEX") and not os.environ.get(
        "OPENROUTER_API_KEY"
    ):
        raise click.ClickException(
            "OPENROUTER_API_KEY_IMAS_CODEX (or OPENROUTER_API_KEY) not set. "
            "Add to .env or export in shell."
        )

    master_key = os.environ.get("LITELLM_MASTER_KEY")
    if not master_key:
        raise click.ClickException(
            "LITELLM_MASTER_KEY not set. Add to .env or export in shell."
        )

    # Security: check file permissions on sensitive files
    _check_file_permissions()

    click.echo(f"Starting LiteLLM proxy on {host}:{port}")
    click.echo(f"Config: {config_path}")
    click.echo(f"Master key: {master_key[:10]}...")

    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        click.echo("Langfuse: configured (cost tracking enabled)")
        os.environ.setdefault("LITELLM_CALLBACKS", "langfuse")
    else:
        click.echo(
            "Langfuse: not configured (set LANGFUSE_PUBLIC_KEY for cost tracking)"
        )

    cmd = [
        uv_path,
        "tool",
        "run",
        "--with",
        "litellm[proxy]>=1.81.0",
        "--with",
        "langfuse>=2.0.0",
        "--with",
        "prisma>=0.15.0",
        "--",
        "litellm",
        "--config",
        config_path,
        "--host",
        host,
        "--port",
        str(port),
        "--detailed_debug" if log_level == "DEBUG" else "--drop_params",
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        click.echo("\nProxy stopped")


def _start_llm_deploy() -> None:
    """Deploy LLM proxy via systemd and wait for health."""
    from imas_codex.cli.services import _llm_port, _llm_ssh, _wait_for_health

    _deploy_login_llm()
    port = _llm_port()
    _wait_for_health(
        "LLM proxy",
        f"curl -sf http://localhost:{port}/",
        timeout_s=60,
        ssh_host=_llm_ssh(),
    )
    click.echo(f"  URL: http://localhost:{port}")


@llm.command("status")
@click.option(
    "--url",
    default=None,
    help="Proxy URL (default: from pyproject.toml [llm].port)",
)
def llm_status(url: str | None) -> None:
    """Check LiteLLM proxy health.

    Shows proxy health, systemd service state, process resource usage,
    model availability, and authentication status.

    Examples:
        imas-codex llm status
        imas-codex llm status --url http://remote:18400
    """
    import httpx

    from imas_codex.settings import get_llm_proxy_url

    if url is None:
        url = get_llm_proxy_url()

    # Systemd service status
    _show_llm_service_status()

    click.echo(f"\nLiteLLM Proxy ({url}):")

    try:
        # Use root endpoint for quick alive check (no auth required)
        resp = httpx.get(f"{url}/", timeout=5.0)
        if resp.status_code == 200:
            click.echo("  ✓ Healthy")

            # Auth check — verify that unauthenticated requests are rejected
            _show_llm_auth_status(url)

            # Show model availability (requires auth)
            master_key = os.environ.get("LITELLM_MASTER_KEY", "")
            headers = {"Authorization": f"Bearer {master_key}"}
            models_resp = httpx.get(
                f"{url}/v1/models",
                timeout=10.0,
                headers=headers,
            )
            if models_resp.status_code == 200:
                models = models_resp.json().get("data", [])
                click.echo(f"  Models: {len(models)} available")
                for m in models[:10]:
                    click.echo(f"    - {m.get('id', 'unknown')}")
            elif models_resp.status_code == 401:
                click.echo("  Models: auth required (LITELLM_MASTER_KEY not set)")

            # Credential info
            _show_credential_info()

            # Database and team/key info
            _show_gateway_info(url)
        else:
            click.echo(f"  ✗ Unhealthy (HTTP {resp.status_code})")
    except httpx.ConnectError:
        click.echo("  ✗ Not running")
        click.echo("  Start with: imas-codex llm start")
    except httpx.RemoteProtocolError:
        click.echo("  ✗ Not responding (port in use by a non-HTTP process)")
        click.echo("  A stale process may be holding the port.")
        # Extract port from URL for diagnostic hint
        from urllib.parse import urlparse

        from imas_codex.settings import get_llm_proxy_port

        port = urlparse(url).port or get_llm_proxy_port()
        click.echo(f"  Check with: ss -tlnp sport = {port}")
        click.echo("  Then kill the process and restart:")
        click.echo("    imas-codex llm service start")
    except Exception as e:
        click.echo(f"  ✗ Error: {e}")


def _show_llm_service_status() -> None:
    """Show systemd service status with resource usage.

    Falls back to process detection when the systemd service is not
    installed (e.g. proxy started manually or via ``llm start``).
    """
    try:
        from imas_codex.cli.services import _colored_bar, _run_llm_remote
    except ImportError:
        return

    try:
        # Get systemd service properties in one call
        props = _run_llm_remote(
            "systemctl --user show imas-codex-llm "
            "--property=ActiveState,SubState,MainPID,MemoryCurrent,"
            "ActiveEnterTimestamp,NRestarts 2>/dev/null || true",
            timeout=10,
        )
    except Exception:
        click.echo("Service: unavailable (cannot reach login node)")
        return

    parsed = {}
    for line in props.strip().split("\n"):
        if "=" in line:
            key, _, val = line.partition("=")
            parsed[key.strip()] = val.strip()

    active_state = parsed.get("ActiveState", "unknown")
    sub_state = parsed.get("SubState", "")
    pid = parsed.get("MainPID", "0")

    # Fallback: if systemd doesn't know about the service, check for a
    # running litellm process directly.
    if active_state in ("unknown", "inactive") and (pid == "0" or not pid):
        try:
            proc_info = _run_llm_remote(
                "pgrep -af 'litellm.*config' 2>/dev/null | head -1",
                timeout=5,
            )
            if proc_info.strip():
                # Extract PID from pgrep output (format: "PID command...")
                proc_pid = proc_info.strip().split()[0]
                click.echo(f"Service: {click.style('running', fg='green')} (manual)")
                click.echo(f"  PID: {proc_pid}")
                # Get uptime via /proc
                try:
                    uptime_out = _run_llm_remote(
                        f"ps -p {proc_pid} -o etimes= 2>/dev/null",
                        timeout=5,
                    )
                    uptime_s = int(uptime_out.strip())
                    if uptime_s > 86400:
                        click.echo(f"  Uptime: {uptime_s / 86400:.1f}d")
                    elif uptime_s > 3600:
                        click.echo(f"  Uptime: {uptime_s / 3600:.1f}h")
                    else:
                        click.echo(f"  Uptime: {uptime_s / 60:.0f}m")
                except Exception:
                    pass
                return
        except Exception:
            pass
        click.echo(f"Service: {click.style('stopped', fg='red')}")
        return

    if active_state == "active":
        state_str = click.style("active", fg="green")
    elif active_state == "activating":
        state_str = click.style("starting", fg="yellow")
    else:
        state_str = click.style(active_state, fg="red")

    click.echo(f"Service: {state_str} ({sub_state})")

    if active_state != "active" or pid == "0":
        return

    click.echo(f"  PID: {pid}")

    # Uptime from ActiveEnterTimestamp
    timestamp = parsed.get("ActiveEnterTimestamp", "")
    if timestamp and timestamp != "[not set]":
        try:
            uptime_out = _run_llm_remote(
                f'echo $(( $(date +%s) - $(date -d "{timestamp}" +%s) ))',
                timeout=5,
            )
            uptime_s = int(uptime_out.strip())
            if uptime_s > 86400:
                click.echo(f"  Uptime: {uptime_s / 86400:.1f}d")
            elif uptime_s > 3600:
                click.echo(f"  Uptime: {uptime_s / 3600:.1f}h")
            else:
                click.echo(f"  Uptime: {uptime_s / 60:.0f}m")
        except Exception:
            pass

    # Memory from systemd cgroup (more reliable than ps)
    mem_current = parsed.get("MemoryCurrent", "")
    if mem_current and mem_current not in ("[not set]", "infinity"):
        try:
            mem_bytes = int(mem_current)
            mem_mb = mem_bytes / (1024 * 1024)
            # LLM proxy is lightweight — 2GB is generous for a Python proxy
            mem_limit_mb = 2048
            mem_bar = _colored_bar(mem_mb, mem_limit_mb)
            click.echo(f"  Mem:  {mem_bar}  {mem_mb:.0f} MB")
        except ValueError:
            pass

    # Process CPU via ps (systemd doesn't track cumulative CPU)
    try:
        cpu_out = _run_llm_remote(
            f"ps -p {pid} -o %cpu= 2>/dev/null || true",
            timeout=5,
        )
        cpu_val = float(cpu_out.strip()) if cpu_out.strip() else 0
        if cpu_val > 0:
            # Login node typically has many cores; show relative to 1 core
            cpu_bar = _colored_bar(cpu_val, 100)
            click.echo(f"  CPU:  {cpu_bar}  {cpu_val:.1f}%")
    except Exception:
        pass

    # Restart count
    restarts = parsed.get("NRestarts", "0")
    if restarts != "0":
        click.echo(f"  Restarts: {restarts}")


def _show_llm_auth_status(url: str) -> None:
    """Verify that the LLM proxy requires authentication."""
    import httpx

    try:
        # Try accessing models endpoint without auth
        resp = httpx.get(f"{url}/v1/models", timeout=5.0)
        if resp.status_code == 401:
            click.echo(f"  Auth: {click.style('✓ API key required', fg='green')}")
        elif resp.status_code == 200:
            click.echo(
                f"  Auth: {click.style('✗ UNPROTECTED — no API key required!', fg='red', bold=True)}"
            )
        else:
            click.echo(f"  Auth: HTTP {resp.status_code}")
    except Exception:
        pass


def _show_credential_info() -> None:
    """Show configured credential names from litellm_config.yaml."""
    from pathlib import Path

    import yaml

    config_path = Path(__file__).parent.parent / "config" / "litellm_config.yaml"
    if not config_path.exists():
        return

    with config_path.open() as f:
        config = yaml.safe_load(f)

    credentials = config.get("credential_list", [])
    if credentials:
        click.echo(f"  Credentials: {len(credentials)} configured")
        for cred in credentials:
            name = cred.get("credential_name", "unknown")
            desc = cred.get("credential_info", {}).get("description", "")
            # Check if the env var is set
            env_key = cred.get("credential_values", {}).get("api_key", "")
            env_name = (
                env_key.replace("os.environ/", "")
                if env_key.startswith("os.environ/")
                else ""
            )
            env_set = bool(os.environ.get(env_name)) if env_name else False
            status = (
                click.style("✓", fg="green") if env_set else click.style("✗", fg="red")
            )
            click.echo(f"    {status} {name}: {desc}")


def _show_gateway_info(url: str) -> None:
    """Show multi-tenant gateway info (DB, teams, keys)."""
    import httpx

    master_key = os.environ.get("LITELLM_MASTER_KEY", "")
    headers = {
        "Authorization": f"Bearer {master_key}",
        "Content-Type": "application/json",
    }

    # Database status — detect by attempting to list teams
    # (the /health endpoint doesn't report DB connection status)
    db_connected = False
    try:
        resp = httpx.get(f"{url}/team/list", timeout=5.0, headers=headers)
        if resp.status_code == 200:
            db_connected = True
            teams = resp.json()
            if isinstance(teams, list):
                click.echo(f"  Database: {click.style('✓ connected', fg='green')}")
                click.echo(f"  Teams: {len(teams)}")
                for team in teams[:5]:
                    alias = team.get("team_alias", team.get("team_id", "?")[:12])
                    budget = team.get("max_budget")
                    spend = team.get("spend", 0)
                    budget_str = f"${budget:.0f}" if budget else "unlimited"
                    click.echo(
                        f"    - {alias} (budget: {budget_str}, spent: ${spend:.2f})"
                    )
        else:
            click.echo(
                f"  Database: {click.style('✗ not connected', fg='yellow')} "
                "(set LITELLM_DATABASE_URL)"
            )
    except Exception:
        click.echo(
            f"  Database: {click.style('✗ not connected', fg='yellow')} "
            "(set LITELLM_DATABASE_URL)"
        )

    if not db_connected:
        return

    # Key count
    try:
        resp = httpx.get(
            f"{url}/key/list",
            timeout=5.0,
            headers=headers,
            params={"return_full_object": "true", "page_size": "100"},
        )
        if resp.status_code == 200:
            data = resp.json()
            keys = data if isinstance(data, list) else data.get("keys", [])
            active = [k for k in keys if isinstance(k, dict) and k.get("key_alias")]
            click.echo(f"  Virtual Keys: {len(active)}")
    except Exception:
        pass


@llm.command("service")
@click.argument(
    "action", type=click.Choice(["install", "uninstall", "status", "start", "stop"])
)
@click.option(
    "--port",
    default=None,
    type=int,
    help="Port for the proxy (default: from pyproject.toml)",
)
def llm_service(action: str, port: int | None) -> None:
    """Manage LiteLLM proxy as systemd user service.

    Examples:
        imas-codex llm service install
        imas-codex llm service start
        imas-codex llm service status
    """
    import platform
    import shutil
    import subprocess
    from pathlib import Path

    if platform.system() != "Linux":
        raise click.ClickException("systemd services only supported on Linux")

    if not shutil.which("systemctl"):
        raise click.ClickException("systemctl not found")

    from imas_codex.settings import get_llm_proxy_port

    if port is None:
        port = get_llm_proxy_port()

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_file = service_dir / "imas-codex-llm.service"

    if action == "install":
        import socket

        service_dir.mkdir(parents=True, exist_ok=True)

        uv_path = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")

        project_dir = Path.home() / "imas-codex"
        if not project_dir.exists():
            project_dir = Path.cwd()
            if not (project_dir / "pyproject.toml").exists():
                raise click.ClickException(
                    f"Project not found at {Path.home() / 'imas-codex'} or current directory"
                )

        config_path = project_dir / "imas_codex" / "config" / "litellm_config.yaml"

        env_file = project_dir / ".env"

        log_dir = Path.home() / ".local" / "share" / "imas-codex" / "services"
        log_file = log_dir / "llm.log"

        # ConditionHost= pins the service to this node — prevents
        # NFS-shared service files from spawning redundant proxies
        # on every login node that shares the same home directory.
        # Use FQDN to match systemd's static hostname.
        local_hostname = socket.getfqdn()

        service_content = f"""[Unit]
Description=IMAS Codex LiteLLM Proxy
After=network.target
ConditionHost={local_hostname}

[Service]
Type=simple
WorkingDirectory={project_dir}
EnvironmentFile=-{env_file}
Environment="PATH={Path.home()}/.local/bin:/usr/local/bin:/usr/bin"
Environment="LITELLM_CALLBACKS=langfuse"
ExecStartPre=/bin/mkdir -p {log_dir}
ExecStart={uv_path} tool run --with 'litellm[proxy]>=1.81.0' --with 'langfuse>=2.0.0' -- litellm --config {config_path} --host 0.0.0.0 --port {port} --drop_params
ExecStop=/bin/kill -15 $MAINPID
StandardOutput=append:{log_file}
StandardError=append:{log_file}
TimeoutStopSec=30
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"""
        service_file.write_text(service_content)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "imas-codex-llm"], check=True)
        click.echo(f"✓ LLM proxy service installed (pinned to {local_hostname})")
        click.echo(f"  Port: {port}")
        click.echo("  Start: imas-codex llm service start")

    elif action == "uninstall":
        if not service_file.exists():
            click.echo("Service not installed")
            return
        subprocess.run(
            ["systemctl", "--user", "stop", "imas-codex-llm"], capture_output=True
        )
        subprocess.run(
            ["systemctl", "--user", "disable", "imas-codex-llm"], capture_output=True
        )
        service_file.unlink()
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        click.echo("Service uninstalled")

    elif action == "status":
        if not service_file.exists():
            click.echo("Service not installed")
            return
        result = subprocess.run(
            ["systemctl", "--user", "status", "imas-codex-llm"],
            capture_output=True,
            text=True,
        )
        click.echo(result.stdout)

    elif action == "start":
        if not service_file.exists():
            raise click.ClickException(
                "Service not installed. Run: imas-codex llm service install"
            )
        result = subprocess.run(
            ["systemctl", "--user", "start", "imas-codex-llm"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # Read recent log lines for diagnostics
            log_file = (
                Path.home() / ".local" / "share" / "imas-codex" / "services" / "llm.log"
            )
            log_tail = ""
            if log_file.exists():
                log_tail = log_file.read_text()[-2000:]
            raise click.ClickException(
                "Failed to start LLM proxy service.\n"
                + log_tail
                + "\nCheck full logs: imas-codex llm logs"
            )
        click.echo("LLM proxy service started")

    elif action == "stop":
        subprocess.run(["systemctl", "--user", "stop", "imas-codex-llm"], check=True)
        click.echo("LLM proxy service stopped")


# ── LLM stop/restart/logs commands ───────────────────────────────────────

from imas_codex.cli.services import (  # noqa: E402
    _PROJECT,
    _SERVICES_DIR,
    _llm_port,
    _llm_ssh,
    _run_llm_remote,
    _tail_log,
    _wait_for_health,
)


def _systemd_user_available() -> bool:
    """Check if systemd user bus is available on the LLM target node.

    Returns False when the user manager is stuck (e.g. ``closing``
    state after a node switch) or the D-Bus session bus cannot be
    reached.  This typically happens on RHEL login nodes with shared
    NFS home directories where ``pam_systemd`` couldn't start a
    fresh ``user@.service`` for the user.
    """
    try:
        result = _run_llm_remote(
            "bash -c 'systemctl --user status 2>&1; echo EXIT:$?'",
            timeout=10,
        )
        return "EXIT:0" in result
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _deploy_login_llm_direct() -> None:
    """Deploy LLM proxy as a direct background process (systemd fallback).

    Used when the systemd user manager isn't available.  Launches the
    proxy via ``nohup`` and writes a PID file for later management.
    Also kills any existing proxy process to avoid port conflicts.
    """
    port = _llm_port()

    # Kill any existing proxy process on this port
    _run_llm_remote(
        f"bash -c 'pid=$(lsof -ti:{port} 2>/dev/null || true); "
        f'[ -n "$pid" ] && kill $pid 2>/dev/null; sleep 1; '
        f"pid=$(lsof -ti:{port} 2>/dev/null || true); "
        f'[ -n "$pid" ] && kill -9 $pid 2>/dev/null; true\'',
        timeout=15,
    )

    # Write a launcher script to the remote, then execute it via
    # setsid + nohup to fully detach from the SSH session.
    # We bypass run_command() because its `timeout` wrapper prevents
    # the SSH session from closing when backgrounding long-lived processes.
    import base64

    launcher = (
        f"#!/bin/bash\n"
        f"cd {_PROJECT}\n"
        f"set -a\n"
        f"source {_PROJECT}/.env 2>/dev/null || true\n"
        f"set +a\n"
        f"export LITELLM_CALLBACKS=langfuse\n"
        f"mkdir -p {_SERVICES_DIR}\n"
        f"exec $HOME/.local/bin/uv tool run "
        f"  --with 'litellm[proxy]>=1.81.0' --with 'langfuse>=2.0.0' "
        f"  --with 'prisma>=0.15.0' "
        f"  -- litellm "
        f"  --config {_PROJECT}/imas_codex/config/litellm_config.yaml "
        f"  --host 0.0.0.0 --port {port} --drop_params "
        f"  >> {_SERVICES_DIR}/llm.log 2>&1\n"
    )
    launcher_b64 = base64.b64encode(launcher.encode()).decode()
    launcher_path = f"{_SERVICES_DIR}/llm-launcher.sh"

    click.echo(f"  Launching LLM proxy directly (port {port})...")

    # Step 1: write launcher script (uses run_command — single command, fine)
    _run_llm_remote(
        f"bash -c 'mkdir -p {_SERVICES_DIR} && "
        f"echo {launcher_b64} | base64 -d > {launcher_path} && "
        f"chmod +x {launcher_path}'",
        timeout=15,
    )

    # Step 2: launch via setsid+nohup — bypass run_command to avoid
    # the timeout wrapper holding the SSH session open.
    ssh_host = _llm_ssh()
    subprocess.run(
        [
            "ssh",
            "-T",
            ssh_host,
            f"setsid nohup {launcher_path} </dev/null >/dev/null 2>&1 &",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
    )

    _wait_for_health(
        "LLM proxy",
        f"curl -sf http://localhost:{port}/",
        timeout_s=120,
        ssh_host=_llm_ssh(),
    )
    click.echo(f"  URL: http://localhost:{port}")

    # Secure database file permissions
    _run_llm_remote(
        f"chmod 600 {_SERVICES_DIR}/litellm.db 2>/dev/null || true",
        timeout=5,
    )

    _stop_stale_llm_instances()


def _deploy_login_llm() -> None:
    """Deploy LLM proxy to login node via systemd (with direct fallback).

    Installs the systemd user service if not present, then starts it.
    The LLM proxy needs outbound HTTPS to reach API providers
    (OpenRouter, Anthropic, Google), so it must run on the login
    node which has internet access, not on a compute node.

    The service unit includes ``ConditionHost=`` to pin it to a
    single node.  On every deploy the unit file is re-written with
    the current target hostname so the proxy follows SSH target
    changes.  Stale instances on sibling nodes are stopped.

    Falls back to direct ``nohup`` launch when the systemd user
    manager isn't available (e.g. stale ``closing`` state after a
    node switch where ``pam_systemd`` couldn't start a fresh user
    manager).

    Idempotent: no-op if already running on the correct node.
    """
    # Check if systemd user bus is available
    if not _systemd_user_available():
        click.echo("  systemd user session unavailable — using direct launch")
        _deploy_login_llm_direct()
        return

    # Always re-install to update ConditionHost= to the current target.
    # This is cheap (one SSH round-trip) and ensures the unit file
    # always matches the node we're deploying to.
    click.echo("  Installing systemd service...")
    _install_llm_service_remote()

    # Check if already running on this node
    try:
        result = _run_llm_remote(
            "systemctl --user is-active imas-codex-llm 2>/dev/null || true",
            timeout=10,
        )
        if "active" in result and "inactive" not in result:
            click.echo("  LLM proxy already running on login node")
            _stop_stale_llm_instances()
            return
    except subprocess.CalledProcessError:
        pass

    click.echo("  Starting LLM proxy on login node...")
    try:
        _run_llm_remote(
            "systemctl --user start imas-codex-llm",
            timeout=15,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        log_tail = _run_llm_remote(
            f"tail -20 {_SERVICES_DIR}/llm.log 2>/dev/null || true",
            timeout=10,
        )
        raise click.ClickException(f"Failed to start LLM proxy.\n{log_tail}") from exc

    # Secure database file permissions
    _run_llm_remote(
        f"chmod 600 {_SERVICES_DIR}/litellm.db 2>/dev/null || true",
        timeout=5,
    )

    _stop_stale_llm_instances()


def _install_llm_service_remote() -> None:
    """Install the LLM systemd user service on the remote login node via SSH.

    Includes ``ConditionHost=`` so the service only starts on the
    designated node — prevents NFS-shared service files from spawning
    redundant proxies on every login node.
    """
    import base64

    port = _llm_port()
    # Capture the static hostname (what systemd uses for ConditionHost=)
    # so the service only starts on this specific node.
    target_hostname = _run_llm_remote("hostname -f", timeout=10).strip()
    # systemd doesn't expand $HOME in any directive — use %h (home dir specifier)
    _project_h = _PROJECT.replace("$HOME", "%h")
    _services_h = _SERVICES_DIR.replace("$HOME", "%h")
    service_content = f"""[Unit]
Description=IMAS Codex LiteLLM Proxy
After=network.target
ConditionHost={target_hostname}

[Service]
Type=simple
WorkingDirectory={_project_h}
EnvironmentFile=-{_project_h}/.env
Environment="PATH=%h/.local/bin:/usr/local/bin:/usr/bin"
Environment="LITELLM_CALLBACKS=langfuse"
ExecStartPre=/bin/mkdir -p {_services_h}
ExecStart=%h/.local/bin/uv tool run --with 'litellm[proxy]>=1.81.0' --with 'langfuse>=2.0.0' -- litellm --config {_project_h}/imas_codex/config/litellm_config.yaml --host 0.0.0.0 --port {port} --drop_params
ExecStop=/bin/kill -15 $MAINPID
StandardOutput=append:{_services_h}/llm.log
StandardError=append:{_services_h}/llm.log
TimeoutStopSec=30
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"""
    content_b64 = base64.b64encode(service_content.encode()).decode()
    _run_llm_remote(
        'bash -c \'mkdir -p "$HOME/.config/systemd/user" && '
        f'echo "{content_b64}" | base64 -d > '
        '"$HOME/.config/systemd/user/imas-codex-llm.service" && '
        "systemctl --user daemon-reload && "
        "systemctl --user enable imas-codex-llm'",
        timeout=15,
        check=True,
    )
    click.echo(f"  Service installed (pinned to {target_hostname})")


def _stop_stale_llm_instances() -> None:
    """Stop LLM proxy instances running on sibling login nodes.

    When the service file is shared via NFS, nodes that had an active
    user session may have started their own copy before the
    ``ConditionHost=`` guard was added (or before the unit was
    re-written with the current target hostname).

    Discovers sibling login nodes from ``/etc/hosts`` on the target
    node and issues ``systemctl --user stop`` + port-based kill on
    each one that isn't the current host.  Failures are non-fatal —
    stale instances will also be prevented from restarting by
    ``ConditionHost=``.
    """
    port = _llm_port()
    try:
        # Use base64-encoded script to avoid nested quoting hell.
        # The script discovers sibling login nodes via /etc/hosts and
        # stops both systemd services and direct processes on each.
        import base64

        inner = (
            "this=$(hostname -s)\n"
            'prefix=$(echo "$this" | sed "s/-[0-9]*$//")\n'
            'for node in $(grep "$prefix" /etc/hosts '
            "| awk '{print $2}' | grep -v gpu); do\n"
            '  short=$(echo "$node" | cut -d. -f1)\n'
            '  [ "$short" = "$this" ] && continue\n'
            "  result=$(ssh -o BatchMode=yes -o ConnectTimeout=3 "
            '-o StrictHostKeyChecking=no "$node" '
            f"'systemctl --user stop imas-codex-llm 2>/dev/null; "
            f"pid=$(lsof -ti:{port} 2>/dev/null || true); "
            f'[ -n "$pid" ] && kill $pid 2>/dev/null; '
            "echo stopped' 2>/dev/null)\n"
            '  [ "$result" = "stopped" ] && echo "  Stopped stale proxy on $short"\n'
            "done\n"
        )
        inner_b64 = base64.b64encode(inner.encode()).decode()
        script = f"echo {inner_b64} | base64 -d | bash"
        output = _run_llm_remote(script, timeout=60)
        if output.strip():
            click.echo(output.strip())
    except Exception:
        pass


@llm.command("stop")
def llm_stop() -> None:
    """Stop the LLM proxy server on the login node.

    \b
    Examples:
        imas-codex llm stop
    """
    port = _llm_port()

    # Try systemd first
    try:
        result = _run_llm_remote(
            "systemctl --user is-active imas-codex-llm 2>/dev/null || true",
            timeout=10,
        )
        if "active" in result and "inactive" not in result:
            _run_llm_remote(
                "systemctl --user stop imas-codex-llm",
                timeout=15,
                check=True,
            )
            click.echo("LLM proxy stopped")
            return
    except subprocess.CalledProcessError:
        pass

    # Fall back to port-based kill (for direct-launch mode)
    try:
        result = _run_llm_remote(
            f"bash -c 'pid=$(lsof -ti:{port} 2>/dev/null || true); "
            f'if [ -n "$pid" ]; then kill $pid 2>/dev/null && echo killed; '
            f"else echo none; fi'",
            timeout=10,
        )
        if "killed" in result:
            click.echo("LLM proxy stopped (direct process)")
        else:
            click.echo("LLM proxy not running")
    except subprocess.CalledProcessError:
        click.echo("Failed to stop LLM proxy")


@llm.command("restart")
def llm_restart() -> None:
    """Restart the LLM proxy server on the login node.

    \b
    Examples:
        imas-codex llm restart
    """
    try:
        _run_llm_remote(
            "systemctl --user restart imas-codex-llm",
            timeout=30,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            "Failed to restart. Is the service installed?\n"
            "  Install: imas-codex llm service install"
        ) from exc

    port = _llm_port()
    _wait_for_health(
        "LLM proxy",
        f"curl -sf http://localhost:{port}/",
        timeout_s=60,
        ssh_host=_llm_ssh(),
    )
    click.echo(f"  URL: http://localhost:{port}")


@llm.command("logs")
@click.option("--follow", "-f", is_flag=True, help="Follow log output (tail -f)")
@click.option(
    "--lines", "-n", default=50, type=int, help="Number of lines (default: 50)"
)
def llm_logs(follow: bool, lines: int) -> None:
    """View LLM proxy service logs.

    \b
    Examples:
        imas-codex llm logs            # Last 50 lines
        imas-codex llm logs -f         # Follow live
        imas-codex llm logs -n 100     # Last 100 lines
    """
    log_file = f"{_SERVICES_DIR}/llm.log"
    _tail_log(log_file, follow, lines, ssh_host=_llm_ssh())


# ── Admin API helpers ────────────────────────────────────────────────────


def _llm_headers() -> dict[str, str]:
    """Get authorization headers for LiteLLM admin API."""
    master_key = os.environ.get("LITELLM_MASTER_KEY", "")
    if not master_key:
        raise click.ClickException(
            "LITELLM_MASTER_KEY not set. Export it or add to .env."
        )
    return {"Authorization": f"Bearer {master_key}", "Content-Type": "application/json"}


def _llm_api_url(path: str) -> str:
    """Build LiteLLM API URL."""
    from imas_codex.settings import get_llm_proxy_url

    return f"{get_llm_proxy_url()}{path}"


def _api_get(path: str, *, params: dict | None = None) -> dict:
    """GET request to LiteLLM admin API with error handling."""
    import httpx

    try:
        resp = httpx.get(
            _llm_api_url(path),
            headers=_llm_headers(),
            params=params,
            timeout=15.0,
        )
    except httpx.ConnectError as exc:
        raise click.ClickException(
            "Cannot connect to LiteLLM proxy. Is it running?\n"
            "  Start with: imas-codex llm start"
        ) from exc

    if resp.status_code == 401:
        raise click.ClickException("Authentication failed. Check LITELLM_MASTER_KEY.")
    if resp.status_code != 200:
        raise click.ClickException(
            f"API error (HTTP {resp.status_code}): {resp.text[:500]}"
        )
    return resp.json()


def _api_post(path: str, *, json: dict | None = None) -> dict:
    """POST request to LiteLLM admin API with error handling."""
    import httpx

    try:
        resp = httpx.post(
            _llm_api_url(path),
            headers=_llm_headers(),
            json=json or {},
            timeout=15.0,
        )
    except httpx.ConnectError as exc:
        raise click.ClickException(
            "Cannot connect to LiteLLM proxy. Is it running?\n"
            "  Start with: imas-codex llm start"
        ) from exc

    if resp.status_code == 401:
        raise click.ClickException("Authentication failed. Check LITELLM_MASTER_KEY.")
    if resp.status_code not in (200, 201):
        raise click.ClickException(
            f"API error (HTTP {resp.status_code}): {resp.text[:500]}"
        )
    return resp.json()


def _truncate(value: str, length: int = 12) -> str:
    """Truncate a string and append ellipsis."""
    if not value:
        return ""
    if len(value) <= length:
        return value
    return value[:length] + "..."


def _resolve_team_id(alias_or_id: str) -> str:
    """Resolve a team alias to its UUID, passing UUIDs through unchanged."""
    # If it looks like a UUID, return as-is
    if len(alias_or_id) > 30 and "-" in alias_or_id:
        return alias_or_id
    # Look up by alias
    data = _api_get("/team/list")
    teams = data if isinstance(data, list) else data.get("teams", [])
    for t in teams:
        if t.get("team_alias") == alias_or_id:
            return t.get("team_id", alias_or_id)
    return alias_or_id


# ── Keys subgroup ────────────────────────────────────────────────────────


@llm.group("keys")
def llm_keys() -> None:
    """Manage virtual API keys for the LiteLLM proxy.

    Virtual keys enable per-team spend tracking and access control.
    Requires LITELLM_MASTER_KEY for admin operations.

    \b
      imas-codex llm keys list       List all virtual keys
      imas-codex llm keys create     Generate a new key
      imas-codex llm keys revoke     Delete a key
      imas-codex llm keys rotate     Rotate (replace) a key
    """
    pass


@llm_keys.command("list")
def keys_list() -> None:
    """List all virtual API keys.

    Displays key alias, truncated token, team, budget, spend, and expiry.

    \b
    Examples:
        imas-codex llm keys list
    """
    data = _api_get(
        "/key/list", params={"return_full_object": "true", "page_size": "100"}
    )

    keys = data.get("keys", data if isinstance(data, list) else [])
    if not keys:
        click.echo("No virtual keys found.")
        return

    # Table header
    header = f"{'Alias':<25} {'Key':<16} {'Team':<20} {'Budget':>8} {'Spend':>8} {'Expires':<12}"
    click.echo(header)
    click.echo("─" * len(header))

    for k in keys:
        alias = k.get("key_alias") or k.get("key_name") or "—"
        token = _truncate(k.get("token", k.get("key", "")), 12)
        team = k.get("team_alias") or k.get("team_id") or "—"
        if isinstance(team, str) and len(team) > 20:
            team = _truncate(team, 17)
        budget = k.get("max_budget")
        budget_str = f"${budget:.0f}" if budget is not None else "—"
        spend = k.get("spend", 0) or 0
        spend_str = f"${spend:.2f}"
        expires = k.get("expires") or "—"
        if isinstance(expires, str) and len(expires) > 12:
            expires = expires[:10]

        click.echo(
            f"{alias:<25} {token:<16} {team:<20} {budget_str:>8} {spend_str:>8} {expires:<12}"
        )

    click.echo(f"\n{len(keys)} key(s) total")


@llm_keys.command("create")
@click.option(
    "--team", "team_id", required=True, help="Team ID or alias to assign the key to"
)
@click.option("--alias", required=True, help="Human-readable name for the key")
@click.option("--budget", type=float, default=None, help="Max budget in USD (optional)")
@click.option(
    "--duration", default=None, help="Key validity duration (e.g. '30d', '1y')"
)
def keys_create(
    team_id: str, alias: str, budget: float | None, duration: str | None
) -> None:
    """Generate a new virtual API key.

    The key is displayed only once — save it immediately.

    \b
    Examples:
        imas-codex llm keys create --team imas-codex-agents --alias worker-1
        imas-codex llm keys create --team imas-codex-dev --alias dev-key --budget 50 --duration 30d
    """
    payload: dict = {
        "team_id": team_id,
        "key_alias": alias,
    }
    if budget is not None:
        payload["max_budget"] = budget
    if duration is not None:
        payload["duration"] = duration

    data = _api_post("/key/generate", json=payload)

    key = data.get("key", data.get("token", ""))
    click.echo("")
    click.echo(
        click.style(
            "⚠  Save this key now — it is shown only once!", fg="yellow", bold=True
        )
    )
    click.echo("")
    click.echo(f"  Key:   {click.style(key, fg='green', bold=True)}")
    click.echo(f"  Alias: {alias}")
    click.echo(f"  Team:  {team_id}")
    if budget is not None:
        click.echo(f"  Budget: ${budget:.2f}")
    if duration is not None:
        click.echo(f"  Duration: {duration}")
    expires = data.get("expires")
    if expires:
        click.echo(f"  Expires: {expires}")
    click.echo("")


@llm_keys.command("revoke")
@click.argument("key")
def keys_revoke(key: str) -> None:
    """Revoke (delete) a virtual API key.

    Permanently deletes the key. This cannot be undone.

    \b
    Examples:
        imas-codex llm keys revoke sk-...
    """
    click.echo(f"Key to revoke: {_truncate(key, 20)}")
    if not click.confirm("Are you sure you want to revoke this key?"):
        click.echo("Cancelled.")
        return

    _api_post("/key/delete", json={"keys": [key]})
    click.echo(click.style("✓ Key revoked", fg="green"))


@llm_keys.command("rotate")
@click.option("--key", "old_key", required=True, help="The key to rotate")
def keys_rotate(old_key: str) -> None:
    """Rotate a virtual API key (delete old, create new).

    Deletes the existing key and generates a replacement with the same
    team and alias. Update all clients using the old key.

    \b
    Examples:
        imas-codex llm keys rotate --key sk-...
    """
    # Fetch key info to preserve team/alias
    info = _api_get("/key/info", params={"key": old_key})

    # The response may be a dict with "info" or a list
    key_info = info.get("info", info) if isinstance(info, dict) else info
    if isinstance(key_info, list) and key_info:
        key_info = key_info[0]

    team_id = key_info.get("team_id", "")
    alias = key_info.get("key_alias") or key_info.get("key_name") or "rotated-key"
    budget = key_info.get("max_budget")

    click.echo(f"Rotating key: {_truncate(old_key, 20)}")
    click.echo(f"  Team:  {team_id or '—'}")
    click.echo(f"  Alias: {alias}")

    if not click.confirm("Proceed with rotation?"):
        click.echo("Cancelled.")
        return

    # Delete old key
    _api_post("/key/delete", json={"keys": [old_key]})

    # Create new key with same metadata
    payload: dict = {"key_alias": alias}
    if team_id:
        payload["team_id"] = team_id
    if budget is not None:
        payload["max_budget"] = budget

    data = _api_post("/key/generate", json=payload)

    new_key = data.get("key", data.get("token", ""))
    click.echo("")
    click.echo(click.style("✓ Key rotated", fg="green"))
    click.echo("")
    click.echo(
        click.style(
            "⚠  Save the new key now — it is shown only once!", fg="yellow", bold=True
        )
    )
    click.echo(f"  New key: {click.style(new_key, fg='green', bold=True)}")
    click.echo("")
    click.echo(click.style("⚠  Update all clients that used the old key!", fg="yellow"))
    click.echo("")


# ── Teams subgroup ───────────────────────────────────────────────────────


@llm.group("teams")
def llm_teams() -> None:
    """Manage teams for budget and access control.

    Teams group virtual keys for collective spend tracking.
    Use the 'imas-codex' prefix for team aliases to avoid collisions.

    \b
      imas-codex llm teams list      List all teams
      imas-codex llm teams create    Create a new team
      imas-codex llm teams info      Show team details
    """
    pass


@llm_teams.command("list")
def teams_list() -> None:
    """List all teams.

    Displays team alias, truncated ID, budget, duration, and spend.

    \b
    Examples:
        imas-codex llm teams list
    """
    data = _api_get("/team/list")

    teams = data if isinstance(data, list) else data.get("teams", [])
    if not teams:
        click.echo("No teams found.")
        return

    header = (
        f"{'Alias':<30} {'Team ID':<16} {'Budget':>8} {'Duration':<10} {'Spend':>8}"
    )
    click.echo(header)
    click.echo("─" * len(header))

    for t in teams:
        alias = t.get("team_alias") or "—"
        team_id = _truncate(t.get("team_id", ""), 12)
        budget = t.get("max_budget")
        budget_str = f"${budget:.0f}" if budget is not None else "—"
        duration = t.get("budget_duration") or "—"
        spend = t.get("spend", 0) or 0
        spend_str = f"${spend:.2f}"

        click.echo(
            f"{alias:<30} {team_id:<16} {budget_str:>8} {duration:<10} {spend_str:>8}"
        )

    click.echo(f"\n{len(teams)} team(s) total")


@llm_teams.command("create")
@click.option("--alias", required=True, help="Team alias (use 'imas-codex-' prefix)")
@click.option(
    "--budget", type=float, default=100.0, show_default=True, help="Max budget in USD"
)
@click.option(
    "--duration",
    default="30d",
    show_default=True,
    help="Budget reset period (e.g. '30d')",
)
def teams_create(alias: str, budget: float, duration: str) -> None:
    """Create a new team.

    Teams group virtual keys and track collective spend.
    Use the 'imas-codex-' prefix for aliases to avoid collisions.

    \b
    Examples:
        imas-codex llm teams create --alias imas-codex-agents
        imas-codex llm teams create --alias imas-codex-dev --budget 50 --duration 7d
    """
    payload = {
        "team_alias": alias,
        "max_budget": budget,
        "budget_duration": duration,
    }

    data = _api_post("/team/new", json=payload)

    team_id = data.get("team_id", "")
    click.echo("")
    click.echo(click.style("✓ Team created", fg="green"))
    click.echo(f"  Alias:    {alias}")
    click.echo(f"  Team ID:  {team_id}")
    click.echo(f"  Budget:   ${budget:.2f}")
    click.echo(f"  Duration: {duration}")
    click.echo("")
    click.echo("Create keys for this team:")
    click.echo(f"  imas-codex llm keys create --team {team_id} --alias <key-name>")
    click.echo("")


@llm_teams.command("info")
@click.argument("team")
def teams_info(team: str) -> None:
    """Show detailed info for a team.

    TEAM can be a team_id or team_alias. Shows team metadata,
    member keys, and spend breakdown.

    \b
    Examples:
        imas-codex llm teams info imas-codex-agents
        imas-codex llm teams info <team-id>
    """
    # Resolve alias to UUID, then fetch via GET
    team_id = _resolve_team_id(team)
    data = _api_get("/team/info", params={"team_id": team_id})

    info = data.get("team_info", data)

    click.echo("")
    click.echo(f"Team: {info.get('team_alias', '—')}")
    click.echo(f"  ID:       {info.get('team_id', '—')}")
    budget = info.get("max_budget")
    click.echo(f"  Budget:   {'$' + f'{budget:.2f}' if budget is not None else '—'}")
    click.echo(f"  Duration: {info.get('budget_duration', '—')}")
    spend = info.get("spend", 0) or 0
    click.echo(f"  Spend:    ${spend:.2f}")
    if budget is not None and budget > 0:
        remaining = budget - spend
        click.echo(f"  Remaining: ${remaining:.2f}")
        pct = (spend / budget) * 100
        click.echo(f"  Used:     {pct:.1f}%")

    # Show member keys if available
    members = data.get("keys", [])
    if members:
        click.echo(f"\n  Keys ({len(members)}):")
        for k in members:
            k_alias = k.get("key_alias") or k.get("key_name") or "—"
            k_token = _truncate(k.get("token", k.get("key", "")), 12)
            k_spend = k.get("spend", 0) or 0
            click.echo(f"    {k_alias:<20} {k_token:<16} ${k_spend:.2f}")
    click.echo("")


# ── Spend command ────────────────────────────────────────────────────────


@llm.command("spend")
@click.option(
    "--team",
    "team_filter",
    default=None,
    help="Filter to a specific team (alias or ID)",
)
def llm_spend(team_filter: str | None) -> None:
    """View per-team spend summary.

    Shows budget usage across all teams, or detailed key-level spend
    for a specific team when --team is provided.

    \b
    Examples:
        imas-codex llm spend
        imas-codex llm spend --team imas-codex-agents
    """
    if team_filter:
        # Show detailed spend for a specific team
        # Resolve alias to UUID, then fetch via GET
        team_id = _resolve_team_id(team_filter)
        data = _api_get("/team/info", params={"team_id": team_id})
        info = data.get("team_info", data)

        click.echo("")
        click.echo(f"Spend report: {info.get('team_alias', team_filter)}")
        click.echo("")

        budget = info.get("max_budget")
        spend = info.get("spend", 0) or 0
        duration = info.get("budget_duration", "—")

        click.echo(
            f"  Budget:    {'$' + f'{budget:.2f}' if budget is not None else '—'}"
        )
        click.echo(f"  Spent:     ${spend:.2f}")
        if budget is not None and budget > 0:
            remaining = budget - spend
            pct = (spend / budget) * 100
            click.echo(f"  Remaining: ${remaining:.2f}")
            click.echo(f"  Used:      {pct:.1f}%")
        click.echo(f"  Period:    {duration}")

        # Key-level breakdown
        members = data.get("keys", [])
        if members:
            click.echo(f"\n  Key-level spend ({len(members)} keys):")
            header = f"    {'Alias':<25} {'Key':<16} {'Spend':>8}"
            click.echo(header)
            click.echo("    " + "─" * (len(header) - 4))
            for k in members:
                k_alias = k.get("key_alias") or k.get("key_name") or "—"
                k_token = _truncate(k.get("token", k.get("key", "")), 12)
                k_spend = k.get("spend", 0) or 0
                click.echo(f"    {k_alias:<25} {k_token:<16} ${k_spend:>7.2f}")
        click.echo("")
        return

    # Overview: all teams
    data = _api_get("/team/list")
    teams = data if isinstance(data, list) else data.get("teams", [])

    if not teams:
        click.echo("No teams found.")
        return

    click.echo("")
    click.echo("Per-team spend summary")
    click.echo("")
    header = f"{'Team':<30} {'Budget':>8} {'Spent':>8} {'Remaining':>10} {'Period':<10}"
    click.echo(header)
    click.echo("─" * len(header))

    total_spend = 0.0
    for t in teams:
        alias = t.get("team_alias") or "—"
        budget = t.get("max_budget")
        budget_str = f"${budget:.0f}" if budget is not None else "—"
        spend = t.get("spend", 0) or 0
        total_spend += spend
        spend_str = f"${spend:.2f}"
        if budget is not None and budget > 0:
            remaining = budget - spend
            remaining_str = f"${remaining:.2f}"
        else:
            remaining_str = "—"
        duration = t.get("budget_duration") or "—"

        click.echo(
            f"{alias:<30} {budget_str:>8} {spend_str:>8} {remaining_str:>10} {duration:<10}"
        )

    click.echo("─" * len(header))
    click.echo(f"{'Total':<30} {'':>8} ${total_spend:>7.2f}")
    click.echo("")


# ── Setup command ────────────────────────────────────────────────────────

# Default team definitions for initial provisioning.
_DEFAULT_TEAMS = [
    {
        "alias": "imas-codex",
        "budget": 500.0,
        "duration": "30d",
        "description": "Internal discovery workers, MCP agents, CLI tools",
        "keys": [
            {
                "alias": "imas-codex-workers",
                "description": "Discovery pipeline workers",
            },
            {"alias": "imas-codex-mcp", "description": "MCP server agents"},
        ],
    },
    {
        "alias": "claude-code",
        "budget": 200.0,
        "duration": "30d",
        "description": "Claude Code IDE clients",
        "keys": [
            {"alias": "claude-code-primary", "description": "Primary Claude Code key"},
        ],
    },
]


@llm.command("setup")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be created without making changes",
)
@click.option(
    "--force",
    is_flag=True,
    help="Create teams/keys even if they already exist",
)
def llm_setup(dry_run: bool, force: bool) -> None:
    """Initial setup: create default teams and virtual keys.

    Creates the standard team structure for multi-tenant operation:

    \b
      imas-codex      Internal use (discovery, agents, MCP)
        └─ imas-codex-workers    Discovery pipeline key
        └─ imas-codex-mcp        MCP server key

      claude-code     External Claude Code clients
        └─ claude-code-primary   Primary client key

    Keys are displayed only once — save them immediately.

    \b
    Examples:
        imas-codex llm setup             # Create teams and keys
        imas-codex llm setup --dry-run   # Preview without creating
    """
    click.echo("")
    click.echo("LiteLLM Gateway Setup")
    click.echo("═" * 40)

    # Check prerequisites
    if not dry_run:
        master_key = os.environ.get("LITELLM_MASTER_KEY", "")
        if not master_key:
            raise click.ClickException(
                "LITELLM_MASTER_KEY not set. Export it or add to .env."
            )

    # Check existing teams
    existing_teams: set[str] = set()
    if not dry_run:
        try:
            data = _api_get("/team/list")
            teams = data if isinstance(data, list) else data.get("teams", [])
            existing_teams = {t.get("team_alias", "") for t in teams}
        except click.ClickException:
            click.echo(
                click.style("⚠ Cannot connect to proxy. Is it running?", fg="yellow")
            )
            click.echo("  Start with: imas-codex llm start")
            raise

    created_keys: list[dict] = []

    for team_def in _DEFAULT_TEAMS:
        alias = team_def["alias"]
        click.echo(f"\n{'─' * 40}")
        click.echo(f"Team: {click.style(alias, bold=True)}")
        click.echo(f"  {team_def['description']}")
        click.echo(f"  Budget: ${team_def['budget']:.0f}/{team_def['duration']}")

        if alias in existing_teams and not force:
            click.echo(
                click.style("  → Already exists (use --force to recreate)", fg="yellow")
            )
            continue

        if dry_run:
            click.echo(click.style("  → Would create team", fg="cyan"))
            for key_def in team_def["keys"]:
                click.echo(f"    → Would create key: {key_def['alias']}")
            continue

        # Create team
        try:
            team_data = _api_post(
                "/team/new",
                json={
                    "team_alias": alias,
                    "max_budget": team_def["budget"],
                    "budget_duration": team_def["duration"],
                },
            )
            team_id = team_data.get("team_id", "")
            click.echo(
                click.style(
                    f"  ✓ Team created (ID: {_truncate(team_id, 12)})", fg="green"
                )
            )
        except click.ClickException as e:
            click.echo(click.style(f"  ✗ Failed: {e.message}", fg="red"))
            continue

        # Create keys for this team
        for key_def in team_def["keys"]:
            try:
                key_data = _api_post(
                    "/key/generate",
                    json={
                        "team_id": team_id,
                        "key_alias": key_def["alias"],
                    },
                )
                key_value = key_data.get("key", key_data.get("token", ""))
                created_keys.append(
                    {
                        "team": alias,
                        "alias": key_def["alias"],
                        "key": key_value,
                        "description": key_def["description"],
                    }
                )
                click.echo(
                    click.style(f"  ✓ Key created: {key_def['alias']}", fg="green")
                )
            except click.ClickException as e:
                click.echo(click.style(f"  ✗ Key failed: {e.message}", fg="red"))

    # Summary with keys
    if created_keys:
        click.echo(f"\n{'═' * 40}")
        click.echo(
            click.style(
                "⚠  SAVE THESE KEYS NOW — they are shown only once!",
                fg="yellow",
                bold=True,
            )
        )
        click.echo("")
        for k in created_keys:
            click.echo(f"  [{k['team']}] {k['alias']}:")
            click.echo(f"    {click.style(k['key'], fg='green', bold=True)}")
            click.echo(f"    {k['description']}")
            click.echo("")

        click.echo("Next steps:")
        click.echo("  1. Save keys to your .env or password manager")
        click.echo("  2. Set LITELLM_API_KEY for imas-codex workers")
        click.echo("  3. Configure Claude Code: see docs/client-setup.md")
    elif dry_run:
        click.echo(f"\n{'═' * 40}")
        click.echo("Dry run complete. Run without --dry-run to create resources.")
    else:
        click.echo(f"\n{'═' * 40}")
        click.echo("Setup complete — all teams already exist.")
    click.echo("")


# ── Security command ─────────────────────────────────────────────────────


@llm.group("security")
def llm_security() -> None:
    """Security audit and hardening for the LLM gateway.

    Check and enforce security settings for the multi-tenant proxy.
    Configurable via environment variables for cross-facility deployment.

    \b
      imas-codex llm security audit    Check security posture
      imas-codex llm security harden   Apply security fixes
    """
    pass


@llm_security.command("audit")
def security_audit() -> None:
    """Audit security posture of the LLM gateway.

    Checks file permissions, authentication, network binding,
    key hygiene, and database security.

    \b
    Examples:
        imas-codex llm security audit
    """
    import stat
    from pathlib import Path

    click.echo("")
    click.echo("LiteLLM Gateway Security Audit")
    click.echo("═" * 40)
    issues: list[str] = []
    warnings: list[str] = []

    # 1. Environment variables
    click.echo("\n1. Environment Variables")
    master_key = os.environ.get("LITELLM_MASTER_KEY", "")
    if master_key:
        click.echo(f"   {click.style('✓', fg='green')} LITELLM_MASTER_KEY is set")
        if len(master_key) < 20:
            warnings.append("LITELLM_MASTER_KEY is short (< 20 chars)")
            click.echo(
                f"   {click.style('⚠', fg='yellow')} Key is short — use a longer key"
            )
        if master_key.startswith("sk-") and "test" in master_key.lower():
            warnings.append("LITELLM_MASTER_KEY appears to be a test key")
            click.echo(f"   {click.style('⚠', fg='yellow')} Appears to be a test key")
    else:
        issues.append("LITELLM_MASTER_KEY not set")
        click.echo(f"   {click.style('✗', fg='red')} LITELLM_MASTER_KEY not set")

    imas_key = os.environ.get("OPENROUTER_API_KEY_IMAS_CODEX", "")
    if imas_key:
        click.echo(
            f"   {click.style('✓', fg='green')} OPENROUTER_API_KEY_IMAS_CODEX is set"
        )
    else:
        warnings.append("OPENROUTER_API_KEY_IMAS_CODEX not set")
        click.echo(
            f"   {click.style('⚠', fg='yellow')} OPENROUTER_API_KEY_IMAS_CODEX not set"
        )

    ext_key = os.environ.get("OPENROUTER_API_KEY_CLAUDE_CODE", "")
    if ext_key:
        click.echo(
            f"   {click.style('✓', fg='green')} OPENROUTER_API_KEY_CLAUDE_CODE is set"
        )
    else:
        click.echo(
            f"   {click.style('—', fg='cyan')} OPENROUTER_API_KEY_CLAUDE_CODE not set (optional)"
        )

    # 2. File permissions
    click.echo("\n2. File Permissions")
    sensitive_files = {
        ".env": Path.cwd() / ".env",
        "litellm.db": Path.home()
        / ".local"
        / "share"
        / "imas-codex"
        / "services"
        / "litellm.db",
    }
    for name, path in sensitive_files.items():
        if not path.exists():
            click.echo(f"   {click.style('—', fg='cyan')} {name} does not exist")
            continue
        mode = path.stat().st_mode
        world_readable = mode & (
            stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH
        )
        if world_readable:
            issues.append(f"{name} is world/group accessible")
            click.echo(
                f"   {click.style('✗', fg='red')} {name} is world/group accessible "
                f"(mode: {oct(mode & 0o777)})"
            )
            click.echo(f"     Fix: chmod 600 {path}")
        else:
            click.echo(
                f"   {click.style('✓', fg='green')} {name} permissions OK "
                f"(mode: {oct(mode & 0o777)})"
            )

    # 3. Proxy authentication
    click.echo("\n3. Proxy Authentication")
    try:
        import httpx

        from imas_codex.settings import get_llm_proxy_url

        url = get_llm_proxy_url()
        resp = httpx.get(f"{url}/v1/models", timeout=5.0)
        if resp.status_code == 401:
            click.echo(f"   {click.style('✓', fg='green')} API key required for access")
        elif resp.status_code == 200:
            issues.append("Proxy allows unauthenticated access")
            click.echo(
                f"   {click.style('✗', fg='red')} UNPROTECTED — no API key required!"
            )
            click.echo("     Set LITELLM_MASTER_KEY and restart the proxy")
        else:
            click.echo(
                f"   {click.style('?', fg='yellow')} Unexpected response: HTTP {resp.status_code}"
            )
    except Exception:
        click.echo(f"   {click.style('—', fg='cyan')} Proxy not reachable (skipped)")

    # 4. Network binding
    click.echo("\n4. Network Binding")
    bind_host = os.environ.get("LITELLM_HOST", "0.0.0.0")
    if bind_host == "127.0.0.1" or bind_host == "localhost":
        click.echo(f"   {click.style('✓', fg='green')} Bound to localhost only")
    elif bind_host == "0.0.0.0":
        warnings.append("Proxy bound to 0.0.0.0 (all interfaces)")
        click.echo(
            f"   {click.style('⚠', fg='yellow')} Bound to all interfaces (0.0.0.0)"
        )
        click.echo("     Set LITELLM_HOST=127.0.0.1 for localhost-only access")
        click.echo("     Or use SSH tunnels for remote access")
    else:
        click.echo(f"   {click.style('—', fg='cyan')} Bound to {bind_host}")

    # 5. Database
    click.echo("\n5. Database")
    db_url = os.environ.get("LITELLM_DATABASE_URL", "")
    if db_url:
        click.echo(f"   {click.style('✓', fg='green')} Database URL configured")
        if "sqlite" in db_url:
            click.echo(
                f"   {click.style('—', fg='cyan')} Using SQLite (OK for single-node)"
            )
    else:
        warnings.append("No database configured — virtual keys/teams will not persist")
        click.echo(f"   {click.style('⚠', fg='yellow')} No database configured")
        click.echo("     Set LITELLM_DATABASE_URL for persistent teams/keys")

    # Summary
    click.echo(f"\n{'═' * 40}")
    if issues:
        click.echo(click.style(f"✗ {len(issues)} issue(s) found:", fg="red", bold=True))
        for issue in issues:
            click.echo(f"  • {issue}")
    if warnings:
        click.echo(click.style(f"⚠ {len(warnings)} warning(s):", fg="yellow"))
        for warning in warnings:
            click.echo(f"  • {warning}")
    if not issues and not warnings:
        click.echo(click.style("✓ All checks passed", fg="green", bold=True))
    elif not issues:
        click.echo(click.style("✓ No critical issues", fg="green"))
    click.echo("")


@llm_security.command("harden")
def security_harden() -> None:
    """Apply security hardening to the LLM gateway.

    Fixes file permissions on sensitive files and validates
    configuration. Non-destructive — only tightens permissions.

    \b
    Examples:
        imas-codex llm security harden
    """
    import stat
    from pathlib import Path

    click.echo("")
    click.echo("LiteLLM Gateway Security Hardening")
    click.echo("═" * 40)
    fixed = 0

    sensitive_files = [
        Path.cwd() / ".env",
        Path.home() / ".local" / "share" / "imas-codex" / "services" / "litellm.db",
    ]

    for path in sensitive_files:
        if not path.exists():
            continue
        mode = path.stat().st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
            click.echo(
                f"  {click.style('✓', fg='green')} {path.name}: "
                f"{oct(mode & 0o777)} → 0o600"
            )
            fixed += 1
        else:
            click.echo(
                f"  {click.style('—', fg='cyan')} {path.name}: already secure "
                f"({oct(mode & 0o777)})"
            )

    # Check log directory permissions
    log_dir = Path.home() / ".local" / "share" / "imas-codex" / "logs"
    if log_dir.exists():
        for log_file in log_dir.glob("*.log*"):
            mode = log_file.stat().st_mode
            if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
                log_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
                click.echo(
                    f"  {click.style('✓', fg='green')} {log_file.name}: "
                    f"{oct(mode & 0o777)} → 0o600"
                )
                fixed += 1

    if fixed:
        click.echo(f"\n{click.style(f'✓ Fixed {fixed} file(s)', fg='green')}")
    else:
        click.echo(f"\n{click.style('✓ All files already secure', fg='green')}")
    click.echo("")


# ── Local LLM inference (Ollama on Titan) ────────────────────────────────


@llm.group("local")
def local() -> None:
    """Manage local LLM inference server on the Titan cluster.

    Runs Ollama on a Titan GPU node via SLURM for local model inference
    (no internet required — models must be pre-staged).

    \b
      imas-codex llm local start     Start local LLM on Titan
      imas-codex llm local stop      Stop local LLM
      imas-codex llm local status    Check local LLM health
      imas-codex llm local models    List local models
    """


@local.command("start")
@click.option("--gpu", default=6, type=int, help="GPU index to use (default: 6)")
@click.option(
    "--script",
    default=None,
    type=click.Path(exists=True),
    help="Custom SLURM script (default: slurm/ollama-llm.sh)",
)
def local_start(gpu: int, script: str | None) -> None:
    """Start local LLM inference server on the Titan cluster.

    Submits a SLURM job running Ollama on the Titan GPU node.
    Models must be pre-staged (compute nodes have no internet).

    \b
    Pre-requisites (run on login node):
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull qwen3:14b-q4_K_M

    \b
    Examples:
        imas-codex llm local start          # GPU 6 (default)
        imas-codex llm local start --gpu 7  # Use GPU 7
    """
    from pathlib import Path

    if script is None:
        script_path = Path(__file__).parent.parent.parent / "slurm" / "ollama-llm.sh"
        if not script_path.exists():
            raise click.ClickException(f"SLURM script not found: {script_path}")
        script = str(script_path)

    # Check if already running
    try:
        result = _run_llm_remote(
            "squeue -n codex-llm-local -h -o '%j %T' 2>/dev/null || true",
            timeout=10,
        )
        if "RUNNING" in result:
            click.echo("Local LLM server already running")
            return
        if "PENDING" in result:
            click.echo("Local LLM server job pending in SLURM queue")
            return
    except Exception:
        pass

    # Submit SLURM job with GPU override
    click.echo(f"Submitting Ollama SLURM job (GPU {gpu})...")
    try:
        result = _run_llm_remote(
            f"CUDA_VISIBLE_DEVICES={gpu} sbatch {_PROJECT}/slurm/ollama-llm.sh",
            timeout=15,
            check=True,
        )
        click.echo(f"  {result.strip()}")
        click.echo("  Monitor: imas-codex llm local status")
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"Failed to submit SLURM job: {exc}") from exc


@local.command("stop")
def local_stop() -> None:
    """Stop the local LLM inference server.

    Cancels the SLURM job running Ollama on the Titan cluster.

    \b
    Examples:
        imas-codex llm local stop
    """
    try:
        result = _run_llm_remote(
            "scancel -n codex-llm-local 2>/dev/null && echo cancelled || echo none",
            timeout=15,
        )
        if "cancelled" in result:
            click.echo("Local LLM server stopped")
        else:
            click.echo("Local LLM server not running")
    except Exception as e:
        click.echo(f"Failed to stop: {e}")


@local.command("status")
def local_status() -> None:
    """Check local LLM inference server status.

    Shows SLURM job state, Ollama health, and loaded models.

    \b
    Examples:
        imas-codex llm local status
    """
    import json

    # SLURM job status
    try:
        result = _run_llm_remote(
            "squeue -n codex-llm-local -h -o '%j %T %N %M %l' 2>/dev/null || true",
            timeout=10,
        )
        if result.strip():
            parts = result.strip().split()
            state = parts[1] if len(parts) > 1 else "unknown"
            node = parts[2] if len(parts) > 2 else "unknown"
            elapsed = parts[3] if len(parts) > 3 else "?"
            click.echo(
                f"SLURM Job: {click.style(state, fg='green' if state == 'RUNNING' else 'yellow')}"
            )
            click.echo(f"  Node: {node}")
            click.echo(f"  Elapsed: {elapsed}")
        else:
            click.echo(f"SLURM Job: {click.style('not running', fg='red')}")
            click.echo("  Start with: imas-codex llm local start")
            return
    except Exception:
        click.echo("SLURM Job: unavailable (cannot reach login node)")
        return

    # Ollama health check
    node = "98dci4-gpu-0002"
    try:
        result = _run_llm_remote(
            f"curl -sf http://{node}:11434/api/tags 2>/dev/null || true",
            timeout=10,
        )
        if result.strip():
            data = json.loads(result)
            models = data.get("models", [])
            click.echo(f"\nOllama: {click.style('healthy', fg='green')}")
            click.echo(f"  Models loaded: {len(models)}")
            for m in models:
                name = m.get("name", "unknown")
                size_gb = m.get("size", 0) / (1024**3)
                click.echo(f"    - {name} ({size_gb:.1f} GB)")
        else:
            click.echo(f"\nOllama: {click.style('not responding', fg='yellow')}")
            click.echo("  The SLURM job may still be starting up")
    except Exception:
        click.echo(f"\nOllama: {click.style('unreachable', fg='red')}")


@local.command("models")
def local_models() -> None:
    """List models available in local Ollama instance.

    \b
    Examples:
        imas-codex llm local models
    """
    import json

    node = "98dci4-gpu-0002"
    try:
        result = _run_llm_remote(
            f"curl -sf http://{node}:11434/api/tags 2>/dev/null || echo '{{}}'",
            timeout=10,
        )
        data = json.loads(result)
        models = data.get("models", [])
        if not models:
            click.echo("No models loaded")
            click.echo("Pull models on login node: ollama pull qwen3:14b-q4_K_M")
            return
        click.echo(f"{'Model':<35} {'Size':>8}  {'Modified'}")
        click.echo("-" * 65)
        for m in models:
            name = m.get("name", "unknown")
            size_gb = m.get("size", 0) / (1024**3)
            modified = m.get("modified_at", "")[:10]
            click.echo(f"{name:<35} {size_gb:>6.1f}G  {modified}")
    except Exception as e:
        click.echo(f"Cannot reach Ollama: {e}")
        click.echo("Is the local LLM server running? imas-codex llm local status")

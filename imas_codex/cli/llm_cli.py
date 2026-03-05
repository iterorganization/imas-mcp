"""LLM proxy server management commands."""

import logging
import os
import subprocess

import click

logger = logging.getLogger(__name__)


@click.group()
def llm() -> None:
    """Manage LiteLLM proxy server for centralized LLM routing.

    Routes all LLM calls through a single proxy with cost tracking via Langfuse.
    Provides OpenAI-compatible endpoint for agent teams and discovery workers.

    Runs on the login node via systemd (needs outbound HTTPS for API providers).

    \b
      imas-codex llm start      Start the proxy (systemd or foreground)
      imas-codex llm stop       Stop the proxy
      imas-codex llm restart    Restart (stop + start)
      imas-codex llm status     Check proxy health
      imas-codex llm logs       View service logs
      imas-codex llm service    Manage systemd service
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

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise click.ClickException(
            "OPENROUTER_API_KEY not set. Add to .env or export in shell."
        )

    master_key = os.environ.get("LITELLM_MASTER_KEY")
    if not master_key:
        raise click.ClickException(
            "LITELLM_MASTER_KEY not set. Add to .env or export in shell."
        )

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
    from imas_codex.cli.services import _llm_port, _wait_for_health

    _deploy_login_llm()
    port = _llm_port()
    _wait_for_health(
        "LLM proxy",
        f"curl -sf http://localhost:{port}/",
        timeout_s=60,
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
    """Show systemd service status with resource usage."""
    try:
        from imas_codex.cli.services import _colored_bar, _run_remote
    except ImportError:
        return

    try:
        # Get systemd service properties in one call
        props = _run_remote(
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
            uptime_out = _run_remote(
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
        cpu_out = _run_remote(
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

        service_content = f"""[Unit]
Description=IMAS Codex LiteLLM Proxy
After=network.target

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
        click.echo("✓ LLM proxy service installed and enabled")
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
    _run_remote,
    _tail_log,
    _wait_for_health,
)


def _deploy_login_llm() -> None:
    """Deploy LLM proxy to login node via systemd.

    Installs the systemd user service if not present, then starts it.
    The LLM proxy needs outbound HTTPS to reach API providers
    (OpenRouter, Anthropic, Google), so it must run on the login
    node which has internet access, not on a compute node.

    Idempotent: no-op if already running.
    """
    # Check if already running
    try:
        result = _run_remote(
            "systemctl --user is-active imas-codex-llm 2>/dev/null || true",
            timeout=10,
        )
        if "active" in result and "inactive" not in result:
            click.echo("  LLM proxy already running on login node")
            return
    except subprocess.CalledProcessError:
        pass

    # Install service if not present, then start
    try:
        _run_remote(
            "systemctl --user cat imas-codex-llm >/dev/null 2>&1",
            timeout=10,
            check=True,
        )
    except subprocess.CalledProcessError:
        # Service not installed — install it remotely
        click.echo("  Installing systemd service...")
        _install_llm_service_remote()

    click.echo("  Starting LLM proxy on login node...")
    try:
        _run_remote(
            "systemctl --user start imas-codex-llm",
            timeout=15,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        log_tail = _run_remote(
            f"tail -20 {_SERVICES_DIR}/llm.log 2>/dev/null || true",
            timeout=10,
        )
        raise click.ClickException(f"Failed to start LLM proxy.\n{log_tail}") from exc


def _install_llm_service_remote() -> None:
    """Install the LLM systemd user service on the remote login node via SSH."""
    import base64

    port = _llm_port()
    # systemd doesn't expand $HOME in any directive — use %h (home dir specifier)
    _project_h = _PROJECT.replace("$HOME", "%h")
    _services_h = _SERVICES_DIR.replace("$HOME", "%h")
    service_content = f"""[Unit]
Description=IMAS Codex LiteLLM Proxy
After=network.target

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
    _run_remote(
        'mkdir -p "$HOME/.config/systemd/user" && '
        f'echo "{content_b64}" | base64 -d > '
        '"$HOME/.config/systemd/user/imas-codex-llm.service" && '
        "systemctl --user daemon-reload && "
        "systemctl --user enable imas-codex-llm",
        timeout=15,
        check=True,
    )
    click.echo("  Service installed and enabled")


@llm.command("stop")
def llm_stop() -> None:
    """Stop the LLM proxy server on the login node.

    \b
    Examples:
        imas-codex llm stop
    """
    try:
        result = _run_remote(
            "systemctl --user is-active imas-codex-llm 2>/dev/null || true",
            timeout=10,
        )
        if "active" in result and "inactive" not in result:
            _run_remote(
                "systemctl --user stop imas-codex-llm",
                timeout=15,
                check=True,
            )
            click.echo("LLM proxy stopped")
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
        _run_remote(
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
    _tail_log(log_file, follow, lines)

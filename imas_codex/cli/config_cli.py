"""Configuration management CLI for private YAML and secrets.

This module provides the ``imas-codex config`` command group for:
- Private facility YAML management via GitHub Gist (``config private``)
- Secrets (.env) sync between project clones via SSH (``config secrets``)
- Local-host declarations for location-aware detection (``config local-hosts``)
"""

import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import click

from imas_codex import __version__

# ============================================================================
# Constants
# ============================================================================

PRIVATE_YAML_GLOB = "imas_codex/config/facilities/*_private.yaml"
PRIVATE_YAML_DIR = Path("imas_codex/config/facilities")
GIST_ID_FILE = Path.home() / ".config" / "imas-codex" / "private-gist-id"
SECRETS_DEFAULT_PROJECT_PATH = "~/Code/imas-codex"
RECOVERY_DIR = Path.home() / ".local" / "share" / "imas-codex" / "recovery"


# ============================================================================
# Helpers
# ============================================================================


def require_gh() -> None:
    if not shutil.which("gh"):
        raise click.ClickException(
            "gh CLI not found. Install from: https://cli.github.com/"
        )


def get_private_files() -> list[Path]:
    return list(Path(".").glob(PRIVATE_YAML_GLOB))


def get_saved_gist_id() -> str | None:
    if GIST_ID_FILE.exists():
        return GIST_ID_FILE.read_text().strip()
    return os.environ.get("IMAS_PRIVATE_GIST_ID")


def save_gist_id(gist_id: str) -> None:
    GIST_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    GIST_ID_FILE.write_text(gist_id)


def get_file_hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def show_file_diff(local_path: Path, remote_path: Path) -> bool:
    if not local_path.exists():
        click.echo(f"  + {local_path.name} (new file)")
        return True
    if not remote_path.exists():
        click.echo(f"  - {local_path.name} (would be removed)")
        return True

    local_hash = get_file_hash(local_path)
    remote_hash = get_file_hash(remote_path)

    if local_hash == remote_hash:
        click.echo(f"  = {local_path.name} (unchanged)")
        return False

    local_lines = len(local_path.read_text().splitlines())
    remote_lines = len(remote_path.read_text().splitlines())
    diff = remote_lines - local_lines
    diff_str = f"+{diff}" if diff > 0 else str(diff)
    click.echo(
        f"  ~ {local_path.name} ({diff_str} lines, {local_hash} → {remote_hash})"
    )
    return True


def _get_secrets_host() -> str | None:
    host_file = Path.home() / ".config" / "imas-codex" / "secrets-host"
    if host_file.exists():
        return host_file.read_text().strip()
    return os.environ.get("IMAS_SECRETS_HOST")


def _save_secrets_host(host: str) -> None:
    host_file = Path.home() / ".config" / "imas-codex" / "secrets-host"
    host_file.parent.mkdir(parents=True, exist_ok=True)
    host_file.write_text(host)


# ============================================================================
# Main Command Group
# ============================================================================


@click.group("config")
def config() -> None:
    """Manage private facility configs, secrets, and local-host settings.

    \b
      imas-codex config private push      Push private YAML to Gist
      imas-codex config private pull      Pull private YAML from Gist
      imas-codex config private status    Show gist configuration
      imas-codex config secrets push      Push .env to remote host
      imas-codex config secrets pull      Pull .env from remote host
      imas-codex config secrets status    Show .env status
      imas-codex config local-hosts       Show declared local hosts
      imas-codex config local-hosts add   Declare a host as local
      imas-codex config local-hosts remove Remove a local-host declaration
    """
    pass


# ============================================================================
# Private Data Subgroup (GitHub Gist)
# ============================================================================


@config.group("private")
def config_private() -> None:
    """Manage private facility YAML via GitHub Gist.

    Uses the ``gh`` CLI to create and manage a secret gist containing
    your private facility configuration files.

    \b
      imas-codex config private push    Create/update secret gist
      imas-codex config private pull    Download and restore files
      imas-codex config private status  Show gist URL and file status
    """
    pass


@config_private.command("push")
@click.option("--gist-id", envvar="IMAS_PRIVATE_GIST_ID", help="Existing gist ID")
@click.option("--dry-run", is_flag=True, help="Show what would be pushed")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output (for hooks)")
def private_push(gist_id: str | None, dry_run: bool, quiet: bool) -> None:
    """Push private YAML files to a secret GitHub Gist."""
    require_gh()

    echo = click.echo if not quiet else lambda *a, **kw: None

    private_files = get_private_files()
    if not private_files:
        echo("No private YAML files found")
        echo(f"  Expected pattern: {PRIVATE_YAML_GLOB}")
        return

    effective_gist_id = gist_id or get_saved_gist_id()

    echo(f"Private files to push: {len(private_files)}")
    for f in private_files:
        echo(f"  - {f.name}")

    if dry_run:
        if effective_gist_id:
            echo(f"\n[DRY RUN] Would update gist: {effective_gist_id}")
        else:
            echo("\n[DRY RUN] Would create new secret gist")
        return

    if effective_gist_id:
        echo(f"\nUpdating gist {effective_gist_id}...")

        with tempfile.TemporaryDirectory() as tmpdir:
            gist_dir = Path(tmpdir) / "gist"

            result = subprocess.run(
                ["gh", "gist", "clone", effective_gist_id, str(gist_dir)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                echo("Gist not found, creating new one...")
                effective_gist_id = None
            else:
                subprocess.run(
                    [
                        "git",
                        "remote",
                        "set-url",
                        "origin",
                        f"git@gist.github.com:{effective_gist_id}.git",
                    ],
                    cwd=gist_dir,
                    capture_output=True,
                )

                for f in private_files:
                    shutil.copy(f, gist_dir / f.name)

                subprocess.run(
                    ["git", "add", "-A"],
                    cwd=gist_dir,
                    capture_output=True,
                )
                result = subprocess.run(
                    ["git", "commit", "-m", f"Update from imas-codex {__version__}"],
                    cwd=gist_dir,
                    capture_output=True,
                    text=True,
                )
                if "nothing to commit" in result.stdout + result.stderr:
                    echo("  No changes to push")
                else:
                    result = subprocess.run(
                        ["git", "push"],
                        cwd=gist_dir,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        raise click.ClickException(f"Push failed: {result.stderr}")

    if not effective_gist_id:
        echo("\nCreating secret gist...")
        cmd = [
            "gh",
            "gist",
            "create",
            "--desc",
            "imas-codex private facility configs",
            *[str(f) for f in private_files],
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise click.ClickException(f"Failed to create gist: {result.stderr}")

        gist_url = result.stdout.strip()
        effective_gist_id = gist_url.split("/")[-1]

        save_gist_id(effective_gist_id)
        echo(f"Gist ID saved to: {GIST_ID_FILE}")

    echo(f"\n✓ Pushed to: https://gist.github.com/{effective_gist_id}")
    echo("  (This is a secret gist - only accessible with the URL)")


@config_private.command("pull")
@click.option("--gist-id", envvar="IMAS_PRIVATE_GIST_ID", help="Gist ID to pull from")
@click.option("--url", "gist_url", help="Gist URL (extracts ID and saves for future)")
@click.option("--force", is_flag=True, help="Overwrite without diff/prompt")
@click.option("--no-backup", is_flag=True, help="Skip backup of existing files")
def private_pull(
    gist_id: str | None, gist_url: str | None, force: bool, no_backup: bool
) -> None:
    """Pull private YAML files from GitHub Gist."""
    require_gh()

    if gist_url:
        gist_id = gist_url.rstrip("/").split("/")[-1]
        click.echo(f"Extracted gist ID: {gist_id}")
        save_gist_id(gist_id)
        click.echo(f"  Saved to: {GIST_ID_FILE}")

    effective_gist_id = gist_id or get_saved_gist_id()
    if not effective_gist_id:
        raise click.ClickException(
            "No gist ID configured. Either:\n"
            "  1. Run 'imas-codex config private push' first, or\n"
            "  2. Provide --url https://gist.github.com/<id>, or\n"
            "  3. Provide --gist-id, or\n"
            "  4. Set IMAS_PRIVATE_GIST_ID environment variable"
        )

    click.echo(f"Pulling from gist: {effective_gist_id}")

    existing_files = get_private_files()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        result = subprocess.run(
            ["gh", "gist", "clone", effective_gist_id, str(tmp / "gist")],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise click.ClickException(f"Failed to clone gist: {result.stderr}")

        gist_dir = tmp / "gist"
        yaml_files = list(gist_dir.glob("*_private.yaml"))

        if not yaml_files:
            click.echo("No *_private.yaml files found in gist")
            return

        if existing_files and not force:
            click.echo("\nChanges:")
            has_changes = False
            for gist_file in yaml_files:
                local_file = PRIVATE_YAML_DIR / gist_file.name
                if show_file_diff(local_file, gist_file):
                    has_changes = True

            gist_names = {f.name for f in yaml_files}
            for local_file in existing_files:
                if local_file.name not in gist_names:
                    click.echo(f"  ? {local_file.name} (local only, not in gist)")

            if has_changes:
                click.echo("")
                if not click.confirm("Apply these changes?"):
                    click.echo("Aborted.")
                    return
            else:
                click.echo("\nNo changes to apply.")
                return

        if existing_files and not no_backup:
            timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            backup_dir = RECOVERY_DIR / f"{timestamp}-private-pull"
            backup_dir.mkdir(parents=True, exist_ok=True)
            for f in existing_files:
                shutil.copy(f, backup_dir / f.name)
            click.echo(f"Backed up {len(existing_files)} files to: {backup_dir}")

        PRIVATE_YAML_DIR.mkdir(parents=True, exist_ok=True)
        for f in yaml_files:
            target = PRIVATE_YAML_DIR / f.name
            shutil.copy(f, target)
            click.echo(f"  ✓ {f.name}")

    if not get_saved_gist_id():
        save_gist_id(effective_gist_id)

    click.echo(f"\n✓ Pulled {len(yaml_files)} files")


@config_private.command("status")
def private_status() -> None:
    """Show private YAML file status and gist configuration."""
    private_files = get_private_files()
    gist_id = get_saved_gist_id()

    click.echo("Private YAML files:")
    if private_files:
        for f in private_files:
            size = f.stat().st_size
            click.echo(f"  - {f.name} ({size} bytes)")
    else:
        click.echo("  (none found)")
        click.echo(f"  Expected pattern: {PRIVATE_YAML_GLOB}")

    click.echo(f"\nGist ID: {gist_id or '(not configured)'}")
    if gist_id:
        click.echo(f"  URL: https://gist.github.com/{gist_id}")
        click.echo(f"  Config: {GIST_ID_FILE}")


# ============================================================================
# Secrets Subgroup (SSH-based transfer)
# ============================================================================


@config.group("secrets")
def config_secrets() -> None:
    """Sync .env between project clones via SSH.

    Transfer .env directly into the imas-codex project directory on remote hosts.
    Uses SSH/SCP with strict file permissions (0600).

    \b
      imas-codex config secrets push iter    Push .env to iter:~/Code/imas-codex
      imas-codex config secrets pull iter    Pull .env from iter:~/Code/imas-codex
      imas-codex config secrets status       Show local .env status

    Default remote path: ~/Code/imas-codex (override with --path)
    """
    pass


@config_secrets.command("push")
@click.argument("host", required=False, envvar="IMAS_SECRETS_HOST")
@click.option(
    "--path",
    default=SECRETS_DEFAULT_PROJECT_PATH,
    show_default=True,
    help="Remote project path",
)
@click.option("--dry-run", is_flag=True, help="Show what would be transferred")
def secrets_push(host: str | None, path: str, dry_run: bool) -> None:
    """Push .env to remote project directory."""
    effective_host = host or _get_secrets_host()
    if not effective_host:
        raise click.ClickException(
            "No host specified. Provide HOST argument or set IMAS_SECRETS_HOST"
        )

    env_file = Path(".env")
    if not env_file.exists():
        raise click.ClickException(".env file not found in project root")

    remote_env = f"{path}/.env"
    remote_path = f"{effective_host}:{remote_env}"

    click.echo(f"Push target: {effective_host}")
    click.echo(f"  Remote project: {path}")
    click.echo(f"  File: .env ({env_file.stat().st_size} bytes)")

    if dry_run:
        click.echo("\n[DRY RUN] Would:")
        click.echo(f"  1. Verify {path} exists on {effective_host}")
        click.echo(f"  2. Copy .env to {remote_path}")
        click.echo("  3. Set .env permissions to 0600")
        return

    click.echo("\nVerifying remote project...")
    result = subprocess.run(
        ["ssh", effective_host, f"test -d {path} && echo exists"],
        capture_output=True,
        text=True,
    )
    if "exists" not in result.stdout:
        raise click.ClickException(
            f"Project not found on {effective_host}\n"
            f"  Expected: {path}\n"
            f"  Use --path to specify a different location"
        )

    click.echo("Transferring .env...")
    result = subprocess.run(
        ["scp", "-q", str(env_file), remote_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise click.ClickException(f"Transfer failed: {result.stderr}")

    result = subprocess.run(
        ["ssh", effective_host, f"chmod 600 {remote_env}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"Warning: Could not set permissions: {result.stderr}", err=True)

    if host:
        _save_secrets_host(host)

    click.echo(f"\n✓ Pushed .env to {effective_host}:{path}")


@config_secrets.command("pull")
@click.argument("host", required=False, envvar="IMAS_SECRETS_HOST")
@click.option(
    "--path",
    default=SECRETS_DEFAULT_PROJECT_PATH,
    show_default=True,
    help="Remote project path",
)
@click.option("--force", is_flag=True, help="Overwrite existing .env without prompt")
def secrets_pull(host: str | None, path: str, force: bool) -> None:
    """Pull .env from remote project directory."""
    effective_host = host or _get_secrets_host()
    if not effective_host:
        raise click.ClickException(
            "No host specified. Provide HOST argument or set IMAS_SECRETS_HOST"
        )

    env_file = Path(".env")
    remote_env = f"{path}/.env"
    remote_path = f"{effective_host}:{remote_env}"

    if env_file.exists() and not force:
        click.echo("Local .env exists. Overwrite? (use --force to skip prompt)")
        if not click.confirm("Continue?"):
            return

    click.echo(f"Pulling from: {effective_host}")
    click.echo(f"  Remote project: {path}")

    result = subprocess.run(
        ["ssh", effective_host, f"test -f {remote_env} && echo exists"],
        capture_output=True,
        text=True,
    )
    if "exists" not in result.stdout:
        raise click.ClickException(
            f"No .env found on {effective_host}\n"
            f"  Expected: {remote_env}\n"
            f"  Use --path to specify a different location"
        )

    click.echo("Transferring .env...")
    result = subprocess.run(
        ["scp", "-q", remote_path, str(env_file)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise click.ClickException(f"Transfer failed: {result.stderr}")

    env_file.chmod(0o600)

    if host:
        _save_secrets_host(host)

    click.echo(f"\n✓ Pulled .env from {effective_host}:{path}")
    click.echo(f"  Size: {env_file.stat().st_size} bytes")


@config_secrets.command("status")
def secrets_status() -> None:
    """Show local .env status."""
    env_file = Path(".env")
    secrets_host = _get_secrets_host()

    click.echo("Local .env:")
    if env_file.exists():
        stat = env_file.stat()
        perms = oct(stat.st_mode)[-3:]
        click.echo(f"  Size: {stat.st_size} bytes")
        click.echo(f"  Permissions: {perms}")
        if perms != "600":
            click.echo("  ⚠ Permissions should be 600")
    else:
        click.echo("  (not found)")

    click.echo(f"\nDefault remote path: {SECRETS_DEFAULT_PROJECT_PATH}")
    click.echo(f"Last used host: {secrets_host or '(none)'}")


# ============================================================================
# Local Hosts Subgroup (per-machine locality declarations)
# ============================================================================


@config.group("local-hosts", invoke_without_command=True)
@click.pass_context
def config_local_hosts(ctx: click.Context) -> None:
    """Manage local-host declarations for location detection.

    Declares which SSH host aliases refer to this machine.  Persisted in
    ``~/.config/imas-codex/local-hosts`` — never travels with ``.env``
    or git, so ``config secrets push`` won't leak it to other machines.

    Most cases are detected automatically via FQDN domain matching
    (e.g. on ``*.iter.org``, ``is_local_host("iter")`` returns True).
    Use this command only when automatic detection fails.

    \b
      imas-codex config local-hosts           Show current declarations
      imas-codex config local-hosts add tcv   Declare 'tcv' as local
      imas-codex config local-hosts remove tcv Remove declaration
    """
    if ctx.invoked_subcommand is None:
        # Default action: show current local hosts
        from imas_codex.remote.executor import (
            _get_fqdn_domain_parts,
            get_local_hosts_file,
        )

        hosts_file = get_local_hosts_file()

        # Show FQDN-based auto-detection
        import socket

        fqdn = socket.getfqdn()
        domain_parts = _get_fqdn_domain_parts()
        click.echo(f"FQDN: {fqdn}")
        click.echo(f"  Domain components: {', '.join(domain_parts)}")

        # Show configured file
        click.echo(f"\nLocal-hosts file: {hosts_file}")
        if hosts_file.exists():
            lines = [
                line.strip()
                for line in hosts_file.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            if lines:
                for host in lines:
                    click.echo(f"  {host}")
            else:
                click.echo("  (empty)")
        else:
            click.echo("  (not created)")

        # Show env var override
        env_val = os.environ.get("IMAS_CODEX_LOCAL_HOSTS", "")
        if env_val:
            click.echo(f"\nIMAS_CODEX_LOCAL_HOSTS env: {env_val}")


@config_local_hosts.command("add")
@click.argument("host")
def local_hosts_add(host: str) -> None:
    """Declare a host alias as local to this machine.

    Example: imas-codex config local-hosts add iter
    """
    from imas_codex.remote.executor import get_local_hosts_file

    hosts_file = get_local_hosts_file()
    hosts_file.parent.mkdir(parents=True, exist_ok=True)

    # Read existing
    existing: list[str] = []
    if hosts_file.exists():
        existing = [
            line.strip()
            for line in hosts_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    if host.lower() in [h.lower() for h in existing]:
        click.echo(f"'{host}' is already declared as local")
        return

    # Append
    with hosts_file.open("a") as f:
        if not existing:
            f.write("# SSH aliases that refer to this machine\n")
            f.write("# Managed by: imas-codex config local-hosts\n")
        f.write(f"{host}\n")

    click.echo(f"Added '{host}' to {hosts_file}")


@config_local_hosts.command("remove")
@click.argument("host")
def local_hosts_remove(host: str) -> None:
    """Remove a local-host declaration.

    Example: imas-codex config local-hosts remove iter
    """
    from imas_codex.remote.executor import get_local_hosts_file

    hosts_file = get_local_hosts_file()

    if not hosts_file.exists():
        raise click.ClickException(f"No local-hosts file: {hosts_file}")

    lines = hosts_file.read_text().splitlines()
    new_lines = [
        line
        for line in lines
        if line.strip().lower() != host.lower() or line.strip().startswith("#")
    ]

    if len(new_lines) == len(lines):
        raise click.ClickException(f"'{host}' not found in {hosts_file}")

    hosts_file.write_text("\n".join(new_lines) + "\n" if new_lines else "")
    click.echo(f"Removed '{host}' from {hosts_file}")

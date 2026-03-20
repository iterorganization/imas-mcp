"""Release command: Semantic version bumps, graph publishing, and tagging.

Single-command release workflow:
1. Compute next version from bump type (major/minor/patch) + optional --rc
2. Validate graph and tag DDVersion
3. Push all graph variants (imas-only + full) to GHCR
4. Create and push git tag (triggers CI)
"""

from __future__ import annotations

import re
import subprocess

import click

# ============================================================================
# Version computation
# ============================================================================


def _get_latest_tag() -> str | None:
    """Get the most recent semver tag from git."""
    result = subprocess.run(
        ["git", "tag", "--sort=-v:refname"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.strip().splitlines():
        tag = line.strip()
        if re.match(r"^v\d+\.\d+\.\d+", tag):
            return tag
    return None


def _parse_version(tag: str) -> tuple[int, int, int, int | None]:
    """Parse a version tag into (major, minor, patch, rc_number|None).

    Handles: v5.0.0, v5.0.0-rc1, v5.0.0-rc12
    """
    tag = tag.lstrip("v")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:-rc(\d+))?$", tag)
    if not match:
        raise click.ClickException(f"Cannot parse version: {tag}")
    major, minor, patch = int(match[1]), int(match[2]), int(match[3])
    rc = int(match[4]) if match[4] else None
    return major, minor, patch, rc


def _format_git_tag(major: int, minor: int, patch: int, rc: int | None) -> str:
    """Format version components as a git tag (v5.0.0 or v5.0.0-rc1)."""
    base = f"v{major}.{minor}.{patch}"
    return f"{base}-rc{rc}" if rc else base


def _format_pep440(major: int, minor: int, patch: int, rc: int | None) -> str:
    """Format version as PEP 440 string (5.0.0 or 5.0.0rc1)."""
    base = f"{major}.{minor}.{patch}"
    return f"{base}rc{rc}" if rc else base


def _tag_exists(tag: str) -> bool:
    """Check if a git tag already exists."""
    result = subprocess.run(["git", "tag", "-l", tag], capture_output=True, text=True)
    return bool(result.stdout.strip())


def compute_next_version(
    bump: str | None,
    *,
    rc: bool = False,
    promote: bool = False,
) -> tuple[str, str]:
    """Compute the next version from the latest git tag.

    Returns (git_tag, pep440_version) tuple.
    """
    latest = _get_latest_tag()
    if latest is None:
        raise click.ClickException(
            "No existing version tags found. Create an initial tag first: "
            "git tag -a v0.1.0 -m 'Initial release'"
        )

    major, minor, patch, current_rc = _parse_version(latest)

    if promote:
        if current_rc is None:
            raise click.ClickException(
                f"Cannot promote {latest} — it's not a release candidate."
            )
        # Strip RC suffix: v5.0.0-rc1 → v5.0.0
        return _format_git_tag(major, minor, patch, None), _format_pep440(
            major, minor, patch, None
        )

    if bump is None:
        raise click.ClickException(
            "Specify a bump type (major, minor, patch) or use --promote."
        )

    # If latest is an RC and we're bumping the same base, auto-increment RC
    if rc and current_rc is not None:
        # Check if this is the same bump level — increment RC
        next_rc = current_rc + 1
        tag = _format_git_tag(major, minor, patch, next_rc)
        while _tag_exists(tag):
            next_rc += 1
            tag = _format_git_tag(major, minor, patch, next_rc)
        return tag, _format_pep440(major, minor, patch, next_rc)

    # Apply bump
    if bump == "major":
        major, minor, patch = major + 1, 0, 0
    elif bump == "minor":
        minor, patch = minor + 1, 0
    elif bump == "patch":
        # If current is an RC, patch bumps to the base version
        if current_rc is not None:
            pass  # Already at the right base
        else:
            patch += 1
    else:
        raise click.ClickException(f"Invalid bump type: {bump}")

    rc_num = 1 if rc else None
    tag = _format_git_tag(major, minor, patch, rc_num)

    # Check for collision and auto-increment RC
    if rc:
        while _tag_exists(tag):
            rc_num += 1
            tag = _format_git_tag(major, minor, patch, rc_num)

    return tag, _format_pep440(major, minor, patch, rc_num)


# ============================================================================
# Pre-flight checks
# ============================================================================


def _check_on_main() -> None:
    result = subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True
    )
    branch = result.stdout.strip()
    if branch != "main":
        raise click.ClickException(
            f"Not on main branch (current: {branch}). Switch first: git checkout main"
        )
    click.echo("  ✓ On main branch")


def _check_remote_exists(remote: str) -> None:
    result = subprocess.run(
        ["git", "remote", "get-url", remote], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise click.ClickException(
            f"Remote '{remote}' not found. Add it: git remote add {remote} <url>"
        )
    click.echo(f"  ✓ Remote '{remote}' exists")


def _check_clean_tree(dry_run: bool) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    if result.stdout.strip():
        msg = "Working tree has uncommitted changes. Commit or stash first."
        if dry_run:
            click.echo(f"  ⚠ {msg}", err=True)
        else:
            raise click.ClickException(msg)
    else:
        click.echo("  ✓ Working tree is clean")


def _check_synced(remote: str, dry_run: bool) -> None:
    subprocess.run(["git", "fetch", remote, "main"], capture_output=True)
    result = subprocess.run(
        ["git", "rev-list", "--left-right", "--count", f"main...{remote}/main"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"  ⚠ Could not check sync with {remote}/main", err=True)
        return

    parts = result.stdout.strip().split()
    if len(parts) != 2:
        return

    ahead, behind = int(parts[0]), int(parts[1])
    if behind > 0:
        msg = (
            f"Local is {behind} commits behind {remote}/main. "
            f"Pull first: git pull {remote} main"
        )
        if dry_run:
            click.echo(f"  ⚠ {msg}", err=True)
        else:
            raise click.ClickException(msg)
    if ahead > 0:
        msg = (
            f"Local is {ahead} commits ahead of {remote}/main. "
            "Ensure PR is merged first."
        )
        if dry_run:
            click.echo(f"  ⚠ {msg}", err=True)
        else:
            raise click.ClickException(msg)
    if ahead == 0 and behind == 0:
        click.echo(f"  ✓ Synced with {remote}/main")


# ============================================================================
# Graph operations
# ============================================================================


def _validate_graph_privacy() -> None:
    """Ensure no private fields are stored in graph Facility nodes."""
    try:
        from imas_codex.graph import GraphClient, get_schema

        schema = get_schema()
        private_slots = schema.get_private_slots("Facility")
        if not private_slots:
            click.echo("  ✓ No private slots defined in schema")
            return

        with GraphClient() as client:
            for slot in private_slots:
                result = client.query(
                    f"MATCH (f:Facility) WHERE f.{slot} IS NOT NULL "
                    f"RETURN f.id AS id, f.{slot} AS value LIMIT 5"
                )
                if result:
                    click.echo(f"  ✗ Private field '{slot}' found in graph!", err=True)
                    for r in result:
                        click.echo(f"    - Facility {r['id']}: {slot}={r['value']}")
                    raise click.ClickException(
                        "Private data must not be in graph before push. "
                        "Remove with: MATCH (f:Facility) REMOVE f.<field>"
                    )

        click.echo(f"  ✓ No private fields found (checked: {private_slots})")
    except click.ClickException:
        raise
    except Exception as e:
        click.echo(f"  ⚠ Could not validate graph: {e}", err=True)
        click.echo("    Is Neo4j running? Check with: imas-codex graph status")


def _tag_dd_version(version_number: str, message: str) -> None:
    """Tag the current DDVersion node with release metadata."""
    try:
        from imas_codex.graph import GraphClient

        with GraphClient() as client:
            client.query(
                """
                MATCH (d:DDVersion {is_current: true})
                SET d.release_version = $version,
                    d.release_message = $message,
                    d.release_at = datetime()
                """,
                version=version_number,
                message=message,
            )
            click.echo(f"  ✓ DDVersion tagged: release_version={version_number}")
    except Exception as e:
        click.echo(f"  ⚠ Could not tag DDVersion: {e}", err=True)
        click.echo("    Is Neo4j running? Check with: imas-codex graph status")


def _get_graph_facilities() -> list[str]:
    """Read facility list from GraphMeta to determine if full variant is needed."""
    try:
        from imas_codex.graph import GraphClient
        from imas_codex.graph.meta import get_graph_meta

        with GraphClient() as client:
            meta = get_graph_meta(client)
            if meta:
                return list(meta.get("facilities") or [])
    except Exception:
        pass
    return []


def _push_graph_variant(
    *,
    imas_only: bool = False,
    message: str | None = None,
    dry_run: bool = False,
) -> bool:
    """Push a single graph variant to GHCR via the graph push CLI.

    Returns True on success.
    """
    from imas_codex.graph.ghcr import get_package_name

    pkg_name = get_package_name(imas_only=imas_only)

    if dry_run:
        click.echo(f"  [would push {pkg_name} to GHCR]")
        return True

    cmd = ["uv", "run", "imas-codex", "graph", "push"]
    if imas_only:
        cmd.append("--imas-only")
    if message:
        cmd.extend(["-m", message])

    click.echo(f"  Pushing {pkg_name}...")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        click.echo(f"  ✗ Failed to push {pkg_name}", err=True)
        return False

    click.echo(f"  ✓ Pushed {pkg_name}")
    return True


def _push_all_graph_variants(message: str, dry_run: bool) -> None:
    """Push all applicable graph variants: imas-only (always) + full (if facilities)."""
    # Always push imas-only
    click.echo("\n  Variant 1: IMAS Data Dictionary only")
    if not _push_graph_variant(imas_only=True, message=message, dry_run=dry_run):
        raise click.ClickException(
            "IMAS-only graph push failed. Check: GHCR_TOKEN set, Neo4j running."
        )

    # Push full variant if graph has facilities
    facilities = _get_graph_facilities()
    if facilities:
        click.echo(f"\n  Variant 2: Full graph (facilities: {', '.join(facilities)})")
        if not _push_graph_variant(message=message, dry_run=dry_run):
            click.echo(
                "  ⚠ Full graph push failed — imas-only was already pushed.",
                err=True,
            )
    else:
        click.echo("\n  Variant 2: Full graph — skipped (no facilities in graph)")


# ============================================================================
# Git tag operations
# ============================================================================


def _create_and_push_tag(tag: str, message: str, remote: str, dry_run: bool) -> None:
    """Create an annotated git tag and push it to the remote."""
    if dry_run:
        click.echo(f"  [would create and push tag {tag} to {remote}]")
        return

    result = subprocess.run(
        ["git", "tag", "-a", tag, "-m", message],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "already exists" in result.stderr:
            click.echo(f"  ⚠ Tag {tag} already exists — pushing existing tag")
        else:
            raise click.ClickException(f"Failed to create tag: {result.stderr}")
    else:
        click.echo(f"  ✓ Created tag: {tag}")

    result = subprocess.run(
        ["git", "push", remote, tag],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise click.ClickException(f"Failed to push tag to {remote}: {result.stderr}")
    click.echo(f"  ✓ Pushed tag to {remote}")


# ============================================================================
# Main command
# ============================================================================


@click.command("release")
@click.argument(
    "bump",
    required=False,
    type=click.Choice(["major", "minor", "patch"]),
)
@click.option(
    "-m",
    "--message",
    required=True,
    help="Release message (used for git tag annotation and GHCR push).",
)
@click.option(
    "--rc",
    is_flag=True,
    help="Create a release candidate (e.g. v5.0.0-rc1).",
)
@click.option(
    "--promote",
    is_flag=True,
    help="Promote current RC to release (e.g. v5.0.0-rc1 → v5.0.0).",
)
@click.option(
    "--version",
    "explicit_version",
    default=None,
    help="Explicit version override (e.g. v5.0.0). Bypasses bump computation.",
)
@click.option(
    "--remote",
    type=click.Choice(["origin", "upstream"]),
    default="upstream",
    help="Target remote for git tag push (default: upstream).",
)
@click.option(
    "--skip-graph",
    is_flag=True,
    help="Skip graph validation and push (code-only release).",
)
@click.option(
    "--skip-git",
    is_flag=True,
    help="Skip git tag creation and push.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without making changes.",
)
def release(
    bump: str | None,
    message: str,
    rc: bool,
    promote: bool,
    explicit_version: str | None,
    remote: str,
    skip_graph: bool,
    skip_git: bool,
    dry_run: bool,
) -> None:
    """Create a release with semantic version bumps and graph publishing.

    Computes the next version from the latest git tag, pushes all graph
    variants to GHCR, creates a git tag, and pushes it to trigger CI.

    BUMP is the version bump type: major, minor, or patch.
    Omit when using --promote (RC → release).

    \b
    Examples:
        # Major RC (v4.0.0 → v5.0.0-rc1)
        imas-codex release major --rc -m 'IMAS DD 4.1.0 support'

        # Increment RC (v5.0.0-rc1 → v5.0.0-rc2)
        imas-codex release major --rc -m 'Fix CI issues'

        # Promote RC to release (v5.0.0-rc2 → v5.0.0)
        imas-codex release --promote -m 'Production release'

        # Patch release (v5.0.0 → v5.0.1)
        imas-codex release patch -m 'Bug fixes'

        # Test on fork
        imas-codex release minor --rc --remote origin -m 'Test'

        # Code-only release (no graph push)
        imas-codex release patch --skip-graph -m 'Docs update'

        # Dry run
        imas-codex release major --rc --dry-run -m 'Test'
    """
    # Resolve version
    if explicit_version:
        if not re.match(r"^v\d+\.\d+\.\d+(-rc\d+)?$", explicit_version):
            raise click.ClickException(
                f"Invalid version format: {explicit_version}. "
                "Expected: v1.0.0 or v1.0.0-rc1"
            )
        git_tag = explicit_version
        version_number = explicit_version.lstrip("v").replace("-rc", "rc")
    else:
        if not bump and not promote:
            raise click.ClickException(
                "Specify a bump type (major, minor, patch) or use --promote. "
                "Use --version for explicit override."
            )
        git_tag, version_number = compute_next_version(bump, rc=rc, promote=promote)

    is_rc = "-rc" in git_tag
    latest = _get_latest_tag()

    click.echo(f"{'[DRY RUN] ' if dry_run else ''}Release: {git_tag}")
    click.echo(f"  From: {latest or '(none)'}")
    click.echo(f"  PEP 440: {version_number}")
    click.echo(f"  Message: {message}")
    click.echo(f"  Remote: {remote}")
    click.echo(f"  RC: {'yes' if is_rc else 'no'}")
    click.echo()

    # Pre-flight checks
    click.echo("Pre-flight checks...")
    _check_on_main()
    _check_remote_exists(remote)
    _check_clean_tree(dry_run)
    _check_synced(remote, dry_run)
    click.echo()

    step = 0

    # Step: Validate graph privacy
    if not skip_graph:
        step += 1
        click.echo(f"Step {step}: Validating graph contains no private fields...")
        if not dry_run:
            _validate_graph_privacy()
        else:
            click.echo("  [would validate graph privacy]")

        # Step: Tag DDVersion
        step += 1
        click.echo(f"\nStep {step}: Tagging DDVersion with release info...")
        if not dry_run:
            _tag_dd_version(version_number, message)
        else:
            click.echo(f"  [would tag DDVersion: release_version={version_number}]")

        # Step: Push all graph variants
        step += 1
        click.echo(f"\nStep {step}: Pushing graph variants to GHCR...")
        _push_all_graph_variants(message, dry_run)
    else:
        click.echo("Graph operations: Skipped (--skip-graph)")

    # Step: Git tag
    if not skip_git:
        step += 1
        click.echo(f"\nStep {step}: Creating and pushing git tag...")
        _create_and_push_tag(git_tag, message, remote, dry_run)
    else:
        click.echo("\nGit tag: Skipped (--skip-git)")

    # Summary
    click.echo()
    if dry_run:
        click.echo("[DRY RUN] No changes made.")
    else:
        click.echo(f"✓ Release {git_tag} complete!")
        click.echo(f"  Tag pushed to {remote} — CI will build and publish.")
        if is_rc:
            click.echo(
                f"\n  To promote: imas-codex release --promote -m 'Release {git_tag.split('-')[0]}'"
            )

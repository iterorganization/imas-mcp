"""Release command: State-machine-driven semantic versioning and publishing.

Two-state release workflow (Stable ↔ RC mode):
  --bump major|minor|patch   Start a new RC series (or direct release with --final)
  --final                    Finalize current RC to stable release
  release status             Show current state and available commands

Pipeline steps:
1. Compute next version from state machine
2. Validate graph and tag DDVersion
3. Push all graph variants (dd-only, full, per-facility) to GHCR
4. Create and push git tag (triggers CI)
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

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


def _get_latest_stable_tag() -> str | None:
    """Get the most recent stable (non-RC) semver tag from git."""
    result = subprocess.run(
        ["git", "tag", "--sort=-v:refname"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.strip().splitlines():
        tag = line.strip()
        if re.match(r"^v\d+\.\d+\.\d+$", tag):  # No -rc suffix
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


# ============================================================================
# State detection
# ============================================================================


def _detect_state() -> dict:
    """Detect current release state from latest git tag.

    Returns dict with keys:
        state: 'stable' | 'rc' | None
        tag: str | None
        major, minor, patch: int
        rc: int | None
    """
    tag = _get_latest_tag()
    if tag is None:
        return {
            "state": None,
            "tag": None,
            "major": 0,
            "minor": 0,
            "patch": 0,
            "rc": None,
        }
    major, minor, patch, rc = _parse_version(tag)
    state = "rc" if rc is not None else "stable"
    return {
        "state": state,
        "tag": tag,
        "major": major,
        "minor": minor,
        "patch": patch,
        "rc": rc,
    }


def _commits_since_tag(tag: str) -> int:
    """Count commits since a tag."""
    result = subprocess.run(
        ["git", "rev-list", f"{tag}..HEAD", "--count"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return 0
    return int(result.stdout.strip())


def _apply_bump(major: int, minor: int, patch: int, bump: str) -> tuple[int, int, int]:
    """Apply a version bump to base version components."""
    if bump == "major":
        return major + 1, 0, 0
    elif bump == "minor":
        return major, minor + 1, 0
    elif bump == "patch":
        return major, minor, patch + 1
    else:
        raise click.ClickException(f"Invalid bump type: {bump}")


def compute_next_version(
    bump: str | None,
    *,
    final: bool = False,
) -> tuple[str, str]:
    """Compute the next version from the latest git tag using a state machine.

    State machine transitions:

      STABLE + --bump            → new RC series  (v5.0.0 + patch → v5.0.1-rc1)
      STABLE + --bump + --final  → direct release (v5.0.0 + patch → v5.0.1)
      STABLE + no bump           → error
      STABLE + --final only      → error (not in RC mode)
      RC + no bump               → increment RC   (v5.0.0-rc1 → v5.0.0-rc2)
      RC + --final               → finalize        (v5.0.0-rc1 → v5.0.0)
      RC + --bump                → abandon RC, bump from last stable
                                   (v6.0.0-rc3 + minor from v5.0.0 → v5.1.0-rc1)
      RC + --bump + --final      → abandon RC, direct release from last stable

    Returns (git_tag, pep440_version) tuple.
    """
    latest = _get_latest_tag()
    if latest is None:
        raise click.ClickException(
            "No existing version tags found. Create an initial tag first: "
            "git tag -a v0.1.0 -m 'Initial release'"
        )

    major, minor, patch, current_rc = _parse_version(latest)
    in_rc = current_rc is not None

    # --- Stable state ---
    if not in_rc:
        if bump is None and final:
            raise click.ClickException(
                f"Not in RC mode (latest: {latest}). "
                "Use --bump with --final for a direct release, "
                "or --bump alone to start an RC series."
            )
        if bump is None:
            raise click.ClickException(
                f"On stable release {latest}. Specify --bump (major|minor|patch) "
                "to start a new release candidate series."
            )
        new_major, new_minor, new_patch = _apply_bump(major, minor, patch, bump)
        rc_num = None if final else 1
        tag = _format_git_tag(new_major, new_minor, new_patch, rc_num)
        if not final:
            while _tag_exists(tag):
                rc_num += 1
                tag = _format_git_tag(new_major, new_minor, new_patch, rc_num)
        return tag, _format_pep440(new_major, new_minor, new_patch, rc_num)

    # --- RC mode ---
    if bump is None and final:
        # Finalize: strip RC suffix (v5.0.0-rc1 → v5.0.0)
        return _format_git_tag(major, minor, patch, None), _format_pep440(
            major, minor, patch, None
        )

    if bump is None:
        # Increment RC (v5.0.0-rc1 → v5.0.0-rc2)
        next_rc = current_rc + 1
        tag = _format_git_tag(major, minor, patch, next_rc)
        while _tag_exists(tag):
            next_rc += 1
            tag = _format_git_tag(major, minor, patch, next_rc)
        return tag, _format_pep440(major, minor, patch, next_rc)

    # --bump in RC mode: abandon current RC, bump from last STABLE tag
    stable_tag = _get_latest_stable_tag()
    if stable_tag is None:
        raise click.ClickException(
            "No stable (non-RC) tags found to bump from. "
            "Finalize the current RC first with --final."
        )
    s_major, s_minor, s_patch, _ = _parse_version(stable_tag)
    new_major, new_minor, new_patch = _apply_bump(s_major, s_minor, s_patch, bump)
    rc_num = None if final else 1
    tag = _format_git_tag(new_major, new_minor, new_patch, rc_num)
    if not final:
        while _tag_exists(tag):
            rc_num += 1
            tag = _format_git_tag(new_major, new_minor, new_patch, rc_num)
    return tag, _format_pep440(new_major, new_minor, new_patch, rc_num)


# ============================================================================
# Release status
# ============================================================================


def _show_release_status() -> None:
    """Show current release state and available commands."""
    info = _detect_state()

    if info["state"] is None:
        click.echo("Release state: No tags found")
        click.echo("  Create initial tag: git tag -a v0.1.0 -m 'Initial release'")
        return

    tag = info["tag"]
    major, minor, patch = info["major"], info["minor"], info["patch"]
    commits = _commits_since_tag(tag)
    commits_str = f" ({commits} commits since tag)" if commits else ""

    if info["state"] == "rc":
        target = _format_git_tag(major, minor, patch, None)
        next_rc = info["rc"] + 1
        click.echo("Release state: RC mode")
        click.echo(f"  Current:  {tag}{commits_str}")
        click.echo(f"  Target:   {target}")

        # Show bump targets from last stable tag
        stable_tag = _get_latest_stable_tag()
        if stable_tag:
            s_maj, s_min, s_pat, _ = _parse_version(stable_tag)
            click.echo(f"  Stable:   {stable_tag}")
        else:
            s_maj, s_min, s_pat = major, minor, patch

        click.echo()
        click.echo("Permitted commands:")
        click.echo(
            f"  imas-codex release -m '...'                  "
            f"→ {_format_git_tag(major, minor, patch, next_rc)}"
        )
        click.echo(f"  imas-codex release --final -m '...'          → {target}")
        click.echo(
            f"  imas-codex release --bump patch -m '...'     "
            f"→ {_format_git_tag(s_maj, s_min, s_pat + 1, 1)}  (abandon RC, bump from {stable_tag or 'stable'})"
        )
        click.echo(
            f"  imas-codex release --bump minor -m '...'     "
            f"→ {_format_git_tag(s_maj, s_min + 1, 0, 1)}  (abandon RC, bump from {stable_tag or 'stable'})"
        )
        click.echo(
            f"  imas-codex release --bump major -m '...'     "
            f"→ {_format_git_tag(s_maj + 1, 0, 0, 1)}  (abandon RC, bump from {stable_tag or 'stable'})"
        )
    else:
        click.echo("Release state: Stable")
        click.echo(f"  Current:  {tag}{commits_str}")
        click.echo()
        click.echo("Permitted commands:")
        click.echo(
            f"  imas-codex release --bump patch -m '...'     "
            f"→ {_format_git_tag(major, minor, patch + 1, 1)}"
        )
        click.echo(
            f"  imas-codex release --bump minor -m '...'     "
            f"→ {_format_git_tag(major, minor + 1, 0, 1)}"
        )
        click.echo(
            f"  imas-codex release --bump major -m '...'     "
            f"→ {_format_git_tag(major + 1, 0, 0, 1)}"
        )
        click.echo(
            f"  imas-codex release --bump patch --final -m '...'  "
            f"→ {_format_git_tag(major, minor, patch + 1, None)}"
        )


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
        raise click.ClickException(
            f"Graph privacy validation failed: {e}\n"
            "  Is Neo4j running? Check: imas-codex graph status"
        ) from e


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
        raise click.ClickException(
            f"Failed to tag DDVersion: {e}\n"
            "  Is Neo4j running? Check: imas-codex graph status"
        ) from e


def _get_graph_facilities() -> list[str]:
    """Read facility list from GraphMeta.

    Raises click.ClickException if Neo4j is unreachable or GraphMeta is missing.
    """
    from imas_codex.graph import GraphClient
    from imas_codex.graph.meta import get_graph_meta

    try:
        with GraphClient() as client:
            meta = get_graph_meta(client)
    except Exception as e:
        raise click.ClickException(
            f"Cannot read graph facilities: {e}\n"
            "  Is Neo4j running? Check: imas-codex graph status"
        ) from e
    if not meta:
        raise click.ClickException(
            "GraphMeta node not found — graph has no metadata.\n"
            "  Run 'imas-codex graph status' first."
        )
    return list(meta.get("facilities") or [])


def _push_graph_variant(
    *,
    dd_only: bool = False,
    facility: str | None = None,
    message: str | None = None,
    registry: str | None = None,
    version_tag: str | None = None,
    source_dump: str | None = None,
    dry_run: bool = False,
) -> bool:
    """Push a single graph variant to GHCR via the graph push CLI.

    Returns True on success.
    """
    from imas_codex.graph.ghcr import get_package_name

    facilities = [facility] if facility else None
    pkg_name = get_package_name(facilities=facilities, dd_only=dd_only)

    if dry_run:
        target = f" → {registry}" if registry else ""
        click.echo(f"  [would push {pkg_name}{target} to GHCR]")
        return True

    cmd = ["uv", "run", "imas-codex", "graph", "push"]
    if dd_only:
        cmd.append("--dd-only")
    if facility:
        cmd.extend(["--facility", facility])
    if message:
        cmd.extend(["-m", message])
    if registry:
        cmd.extend(["--registry", registry])
    if version_tag:
        cmd.extend(["--version", version_tag])
    if source_dump:
        cmd.extend(["--source-dump", source_dump])

    click.echo(f"  Pushing {pkg_name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(f"  ✗ Failed to push {pkg_name}: {result.stderr.strip()}", err=True)
        return False

    click.echo(f"  ✓ Pushed {pkg_name}")
    return True


def _resolve_target_registry(remote: str) -> str | None:
    """Determine the GHCR registry for graph push based on the target remote.

    When releasing to upstream, graph data must go to the upstream registry
    (ghcr.io/iterorganization) regardless of what the origin remote is.
    Returns None to use the default (origin-based) registry.
    """
    result = subprocess.run(
        ["git", "remote", "get-url", remote],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    url = result.stdout.strip()
    if "github.com" in url:
        if url.startswith("git@"):
            parts = url.split(":")[-1].replace(".git", "").split("/")
        else:
            parts = url.replace(".git", "").split("/")
        if len(parts) >= 2:
            owner = parts[-2].lower()
            return f"ghcr.io/{owner}"
    return None


def _create_shared_dump() -> str | None:
    """Create a single Neo4j dump for reuse across multiple graph push variants.

    Returns the path to the dump file, or None on failure.
    This stops Neo4j once, dumps, and restarts — subsequent pushes
    use --source-dump to avoid additional stop/start cycles.
    """
    from imas_codex.graph.neo4j_ops import Neo4jOperation

    try:
        from imas_codex.graph.profiles import resolve_neo4j

        profile = resolve_neo4j()
        dumps_dir = profile.data_dir / "dumps"
        dumps_dir.mkdir(parents=True, exist_ok=True)

        with Neo4jOperation("release dump", require_stopped=True):
            from imas_codex.graph.neo4j_ops import run_neo4j_dump

            run_neo4j_dump(profile, dumps_dir)

        dump_file = dumps_dir / "neo4j.dump"
        if dump_file.exists():
            # Move to a release-specific location to avoid collisions
            release_dump = dumps_dir / "release-cache.dump"
            shutil.move(str(dump_file), str(release_dump))
            size_mb = release_dump.stat().st_size / 1024 / 1024
            click.echo(f"  ✓ Shared dump created ({size_mb:.1f} MB)")
            return str(release_dump)
        click.echo("  ✗ Dump file not created", err=True)
        return None
    except Exception as e:
        click.echo(f"  ✗ Shared dump failed: {e}", err=True)
        return None


def _push_all_graph_variants(
    message: str, remote: str, dry_run: bool, git_tag: str | None = None
) -> None:
    """Push all graph variants: dd-only, full, and per-facility.

    Dumps the graph once and reuses the dump for filtered variants to avoid
    5× Neo4j stop/start cycles. The full graph push creates the dump, then
    dd-only and per-facility pushes filter from it via --source-dump.

    Raises click.ClickException if any variant push fails.
    """
    # Resolve target registry from the release remote (e.g. upstream → iterorganization)
    registry = _resolve_target_registry(remote)
    if registry:
        click.echo(f"  Target registry: {registry}")

    facilities = _get_graph_facilities()
    failed: list[str] = []

    if not facilities:
        # No facilities — just push dd-only (the only meaningful variant)
        click.echo("\n  Variant 1: IMAS Data Dictionary only")
        if not _push_graph_variant(
            dd_only=True,
            message=message,
            registry=registry,
            version_tag=git_tag,
            dry_run=dry_run,
        ):
            raise click.ClickException(
                "DD-only graph push failed. Check: GHCR_TOKEN set, Neo4j running."
            )
        return

    # Dump once, reuse for all variants.
    click.echo("\n  Creating shared graph dump (stops Neo4j once)...")

    if dry_run:
        cached_dump = None
    else:
        cached_dump = _create_shared_dump()
        if not cached_dump:
            raise click.ClickException(
                "Failed to create shared graph dump.\n"
                "  Is Neo4j running? Check: imas-codex graph status"
            )

    variant = 0

    # Push full graph (all facilities)
    variant += 1
    click.echo(
        f"\n  Variant {variant}: Full graph (facilities: {', '.join(facilities)})"
    )
    if not _push_graph_variant(
        message=message,
        registry=registry,
        version_tag=git_tag,
        source_dump=cached_dump,
        dry_run=dry_run,
    ):
        failed.append("full")

    # Push dd-only (filtered from cached dump)
    variant += 1
    click.echo(f"\n  Variant {variant}: IMAS Data Dictionary only")
    if not _push_graph_variant(
        dd_only=True,
        message=message,
        registry=registry,
        version_tag=git_tag,
        source_dump=cached_dump,
        dry_run=dry_run,
    ):
        failed.append("dd-only")

    # Per-facility graphs (filtered from cached dump)
    for fac in facilities:
        variant += 1
        click.echo(f"\n  Variant {variant}: {fac} + IMAS DD")
        if not _push_graph_variant(
            facility=fac,
            message=message,
            registry=registry,
            version_tag=git_tag,
            source_dump=cached_dump,
            dry_run=dry_run,
        ):
            failed.append(fac)

    # Clean up cached dump
    if cached_dump:
        try:
            Path(cached_dump).unlink(missing_ok=True)
            click.echo("\n  Cleaned up shared dump cache.")
        except OSError:
            pass

    if failed:
        raise click.ClickException(
            f"Graph push failed for {len(failed)} variant(s): "
            f"{', '.join(failed)}.\n"
            "  Check: GHCR_TOKEN set, Neo4j running, network access."
        )


# ============================================================================
# Git tag operations
# ============================================================================


def _create_local_tag(tag: str, message: str, dry_run: bool) -> None:
    """Create an annotated git tag locally (does not push)."""
    if dry_run:
        click.echo(f"  [would create tag {tag}]")
        return

    result = subprocess.run(
        ["git", "tag", "-a", tag, "-m", message],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "already exists" in result.stderr:
            click.echo(f"  ⚠ Tag {tag} already exists")
        else:
            raise click.ClickException(f"Failed to create tag: {result.stderr}")
    else:
        click.echo(f"  ✓ Created tag: {tag}")


def _push_tag(tag: str, remote: str, dry_run: bool) -> None:
    """Push an existing git tag to the remote."""
    if dry_run:
        click.echo(f"  [would push tag {tag} to {remote}]")
        return

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
    "action",
    required=False,
    default=None,
    type=click.Choice(["status"]),
)
@click.option(
    "--bump",
    type=click.Choice(["major", "minor", "patch"]),
    default=None,
    help="Version bump type. Starts a new RC series (default) or direct release (with --final).",
)
@click.option(
    "-m",
    "--message",
    default=None,
    help="Release message (used for git tag annotation and GHCR push).",
)
@click.option(
    "--final",
    "final",
    is_flag=True,
    help="Finalize: promote current RC to stable, or skip RC with --bump.",
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
    action: str | None,
    bump: str | None,
    message: str | None,
    final: bool,
    explicit_version: str | None,
    remote: str,
    skip_git: bool,
    dry_run: bool,
) -> None:
    """State-machine release: semantic version bumps, graph publishing, tagging.

    Detects current state (Stable or RC mode) from the latest git tag and
    computes the next version automatically. RC mode is the default for new
    bumps; use --final to skip RC or promote an existing RC to stable.

    \b
    States:
      Stable   Latest tag is vX.Y.Z       → must --bump to start RC
      RC mode  Latest tag is vX.Y.Z-rcN   → increment RC, --final, or --bump

    \b
    Examples:
        # Show current state and available commands
        imas-codex release status

        # Start RC series (v5.0.0 → v5.1.0-rc1)
        imas-codex release --bump minor -m 'New feature'

        # Increment RC (v5.1.0-rc1 → v5.1.0-rc2)
        imas-codex release -m 'Fix CI issues'

        # Finalize RC (v5.1.0-rc2 → v5.1.0)
        imas-codex release --final -m 'Production release'

        # Abandon RC, bump from last stable (v5.0.0 → v5.1.0-rc1)
        imas-codex release --bump minor -m 'Changed scope'

        # Direct release, skip RC (v5.0.0 → v5.0.1)
        imas-codex release --bump patch --final -m 'Hotfix'

        # Dry run
        imas-codex release --bump major --dry-run -m 'Test'
    """
    # --- Status subcommand ---
    if action == "status":
        _show_release_status()
        return

    # --- Release requires -m/--message ---
    if message is None:
        raise click.ClickException(
            "Missing required option '-m' / '--message'. "
            "Provide a release message: imas-codex release --bump patch -m 'description'"
        )

    # --- Resolve version ---
    if explicit_version:
        if not re.match(r"^v\d+\.\d+\.\d+(-rc\d+)?$", explicit_version):
            raise click.ClickException(
                f"Invalid version format: {explicit_version}. "
                "Expected: v1.0.0 or v1.0.0-rc1"
            )
        git_tag = explicit_version
        version_number = explicit_version.lstrip("v").replace("-rc", "rc")
    else:
        # Detect state once for error messages and abandonment warning
        info = _detect_state()

        if not bump and not final:
            if info["state"] == "rc":
                git_tag, version_number = compute_next_version(None)
            else:
                raise click.ClickException(
                    f"On stable release {info['tag'] or '(none)'}. "
                    "Specify --bump (major|minor|patch) to start a new release. "
                    "Use 'imas-codex release status' to see options."
                )
        else:
            git_tag, version_number = compute_next_version(bump, final=final)

        # Warn when abandoning an active RC series via --bump
        if bump and info["state"] == "rc":
            stable_tag = _get_latest_stable_tag()
            base_ref = stable_tag or "(none)"
            if final:
                click.echo(
                    f"  ⚠ Abandoning {info['tag']} "
                    f"— releasing {git_tag} directly (bumped from {base_ref})",
                    err=True,
                )
            else:
                click.echo(
                    f"  ⚠ Abandoning {info['tag']} "
                    f"— new RC series at {git_tag} (bumped from {base_ref})",
                    err=True,
                )

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

    # Step: Create local git tag (needed by graph push for version detection)
    if not skip_git:
        step += 1
        click.echo(f"Step {step}: Creating local git tag...")
        _create_local_tag(git_tag, message, dry_run)
    else:
        click.echo("Git tag: Skipped (--skip-git)")

    # Step: Validate graph privacy
    step += 1
    click.echo(f"\nStep {step}: Validating graph contains no private fields...")
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
    _push_all_graph_variants(message, remote, dry_run, git_tag=git_tag)

    # Step: Push git tag to remote (triggers CI)
    if not skip_git:
        step += 1
        click.echo(f"\nStep {step}: Pushing git tag to {remote}...")
        _push_tag(git_tag, remote, dry_run)
    click.echo()
    if dry_run:
        click.echo("[DRY RUN] No changes made.")
    else:
        click.echo(f"✓ Release {git_tag} complete!")
        click.echo(f"  Tag pushed to {remote} — CI will build and publish.")
        if is_rc:
            click.echo(
                f"\n  To finalize: imas-codex release --final "
                f"-m 'Release {git_tag.split('-')[0]}'"
            )

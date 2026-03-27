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


def _get_remote_owner(remote: str) -> str | None:
    """Extract the GitHub owner from a git remote URL.

    Handles both SSH (git@github.com:owner/repo.git) and HTTPS
    (https://github.com/owner/repo.git) URL formats.
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
            return parts[-2].lower()
    return None


def _check_final_targets_upstream(remote: str) -> None:
    """Verify that final releases target the upstream (iterorganization) repo.

    RC releases are allowed on any remote (forks), but final/stable releases
    must be pushed to the canonical upstream repository.
    """
    owner = _get_remote_owner(remote)
    if owner is None:
        click.echo(
            "  ⚠ Could not determine remote owner — skipping upstream check",
            err=True,
        )
        return

    if owner != "iterorganization":
        raise click.ClickException(
            f"Final releases must target upstream (iterorganization). "
            f"Remote '{remote}' points to '{owner}'. "
            f"Use --remote upstream"
        )
    click.echo("  ✓ Final release targets upstream (iterorganization)")


def _check_ci_passed(remote: str, dry_run: bool) -> None:
    """Verify that CI checks have passed for the HEAD commit.

    Uses the GitHub CLI (gh) to query commit status. Gracefully degrades
    if gh is not available. For --dry-run, downgrades failures to warnings.
    """
    import json as _json

    if not shutil.which("gh"):
        click.echo("  ⚠ gh CLI not found — skipping CI status check", err=True)
        return

    owner = _get_remote_owner(remote)
    if owner is None:
        click.echo(
            "  ⚠ Could not determine remote owner — skipping CI check",
            err=True,
        )
        return

    result = subprocess.run(
        ["git", "remote", "get-url", remote],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo("  ⚠ Could not read remote URL — skipping CI check", err=True)
        return

    url = result.stdout.strip()
    if url.startswith("git@"):
        repo = url.split(":")[-1].replace(".git", "").split("/")[-1]
    else:
        repo = url.replace(".git", "").split("/")[-1]

    sha_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if sha_result.returncode != 0:
        click.echo("  ⚠ Could not determine HEAD SHA — skipping CI check", err=True)
        return
    sha = sha_result.stdout.strip()

    api_result = subprocess.run(
        ["gh", "api", f"repos/{owner}/{repo}/commits/{sha}/check-runs"],
        capture_output=True,
        text=True,
    )
    if api_result.returncode != 0:
        click.echo("  ⚠ Could not query CI status — skipping CI check", err=True)
        return

    try:
        data = _json.loads(api_result.stdout)
    except _json.JSONDecodeError:
        click.echo("  ⚠ Could not parse CI response — skipping CI check", err=True)
        return

    check_runs = data.get("check_runs", [])
    if not check_runs:
        click.echo(f"  ⚠ No CI check runs found for {sha[:8]}", err=True)
        return

    pending = [cr["name"] for cr in check_runs if cr.get("status") != "completed"]
    failed = [
        cr["name"]
        for cr in check_runs
        if cr.get("status") == "completed"
        and cr.get("conclusion") not in ("success", "skipped", "neutral")
    ]

    if not pending and not failed:
        click.echo(f"  ✓ CI checks passed for {sha[:8]}")
        return

    details = []
    if failed:
        names = ", ".join(failed[:3])
        if len(failed) > 3:
            names += f" (+{len(failed) - 3} more)"
        details.append(f"failed: {names}")
    if pending:
        names = ", ".join(pending[:3])
        if len(pending) > 3:
            names += f" (+{len(pending) - 3} more)"
        details.append(f"pending: {names}")

    msg = (
        f"CI checks not passed for {sha[:8]}. "
        f"{'; '.join(details)}. "
        "Push and wait for CI before finalizing."
    )
    if dry_run:
        click.echo(f"  ⚠ {msg}", err=True)
    else:
        raise click.ClickException(msg)


# ============================================================================
# Changelog generation
# ============================================================================

_COMMIT_TYPE_RE = re.compile(
    r"^(feat|fix|refactor|docs|test|chore|perf|ci)(?:\(.+\))?!?:\s*(.+)$"
)

_TYPE_HEADINGS: dict[str, str] = {
    "feat": "Features",
    "fix": "Bug Fixes",
    "refactor": "Refactoring",
    "docs": "Documentation",
    "perf": "Performance",
    "ci": "CI / Build",
    "test": "Tests",
    "chore": "Maintenance",
}


def _generate_changelog(
    from_tag: str,
    to_ref: str = "HEAD",
    *,
    version: str = "",
    message: str = "",
) -> str:
    """Generate a changelog from commits and PRs since a previous tag.

    Groups commits by conventional commit type and optionally enriches
    with merged PR metadata from the GitHub CLI.
    """
    # Get commits since last tag
    result = subprocess.run(
        ["git", "log", f"{from_tag}..{to_ref}", "--oneline", "--no-decorate"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return f"## {version}\n\nNo changes since {from_tag}.\n"

    lines = result.stdout.strip().splitlines()

    # Parse conventional commits
    grouped: dict[str, list[str]] = {}
    other: list[str] = []
    for line in lines:
        # Strip SHA prefix
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        _, msg = parts[0], parts[1]
        m = _COMMIT_TYPE_RE.match(msg)
        if m:
            ctype, desc = m.group(1), m.group(2)
            grouped.setdefault(ctype, []).append(desc.strip())
        else:
            other.append(msg)

    # Try to get merged PRs for contributor info
    contributors: set[str] = set()
    pr_map: dict[str, str] = {}  # title → #number
    try:
        import json as _json

        pr_result = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "merged",
                "--base",
                "main",
                "--limit",
                "50",
                "--json",
                "number,title,author",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if pr_result.returncode == 0:
            prs = _json.loads(pr_result.stdout)
            for pr in prs:
                login = pr.get("author", {}).get("login", "")
                if login:
                    contributors.add(f"@{login}")
                title = pr.get("title", "")
                number = pr.get("number", "")
                if title and number:
                    pr_map[title.lower()] = f"#{number}"
    except (subprocess.TimeoutExpired, Exception):
        pass

    # Build markdown
    parts: list[str] = []
    heading = f"## {version}" if version else "## Changelog"
    if message:
        heading += f" — {message}"
    parts.append(heading)
    parts.append("")

    for ctype in ("feat", "fix", "refactor", "perf", "docs", "ci", "test", "chore"):
        items = grouped.get(ctype, [])
        if not items:
            continue
        parts.append(f"### {_TYPE_HEADINGS[ctype]}")
        for item in items:
            pr_ref = pr_map.get(f"{ctype}: {item}".lower(), "")
            suffix = f" ({pr_ref})" if pr_ref else ""
            parts.append(f"- {item}{suffix}")
        parts.append("")

    if other:
        parts.append("### Other Changes")
        for item in other:
            parts.append(f"- {item}")
        parts.append("")

    if contributors:
        parts.append("### Contributors")
        parts.append(", ".join(sorted(contributors)))
        parts.append("")

    parts.append(f"**Full diff:** `{from_tag}..{to_ref}`")
    return "\n".join(parts)


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

    # Verify push completed — check for PUSH_COMPLETE marker in output
    if "PUSH_COMPLETE" not in result.stdout and "pushed" not in result.stdout.lower():
        click.echo(
            "  ⚠ Push may have failed silently (no completion marker).", err=True
        )
        if result.stdout.strip():
            click.echo(f"  stdout: {result.stdout.strip()[-500:]}", err=True)
        if result.stderr.strip():
            click.echo(f"  stderr: {result.stderr.strip()[-500:]}", err=True)

    click.echo(f"  ✓ Pushed {pkg_name}")
    return True


def _resolve_target_registry(remote: str) -> str | None:
    """Determine the GHCR registry for graph push based on the target remote.

    When releasing to upstream, graph data must go to the upstream registry
    (ghcr.io/iterorganization) regardless of what the origin remote is.
    Returns None to use the default (origin-based) registry.
    """
    owner = _get_remote_owner(remote)
    if owner:
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


def _resolve_scheduler_for_release(profile) -> str:
    """Resolve scheduler type from Neo4j profile for release operations."""
    from imas_codex.remote.locations import resolve_location

    try:
        loc = resolve_location(profile.host)
        return loc.scheduler or "none"
    except Exception:
        return "none"


def _push_all_graph_variants(
    message: str,
    remote: str,
    dry_run: bool,
    git_tag: str | None = None,
    is_rc: bool = False,
) -> None:
    """Push graph variants to GHCR.

    RC releases push only the full graph and DD-only variant — these are
    the only two needed for CI validation and container builds.
    Per-facility variants are deferred to final releases where they serve
    as production deployment artifacts.

    When the graph is local, dumps once and reuses the dump for filtered
    variants to avoid repeated Neo4j stop/start cycles.

    When the graph is remote (e.g. running on ITER, accessed via tunnel),
    each variant push delegates independently to the remote host via SSH.

    Raises click.ClickException if any variant push fails.
    """
    # Resolve target registry from the release remote (e.g. upstream → iterorganization)
    registry = _resolve_target_registry(remote)
    if registry:
        click.echo(f"  Target registry: {registry}")

    # Detect whether the graph is remote (e.g. ITER via SSH tunnel)
    is_remote = False
    try:
        from imas_codex.graph.profiles import resolve_neo4j
        from imas_codex.graph.remote import is_remote_location

        profile = resolve_neo4j()
        is_remote = is_remote_location(profile.host)
        if is_remote:
            click.echo(f"  Graph location: {profile.host} (remote)")
    except Exception:
        pass

    facilities = _get_graph_facilities()

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

    # ── Unified remote push — single SSH session, single dump ───────────
    if is_remote:
        from imas_codex.cli.graph_progress import (
            GraphProgress,
            remote_operation_streaming,
        )
        from imas_codex.graph.ghcr import get_package_name
        from imas_codex.graph.remote import (
            build_remote_release_push_script,
            remote_check_imas_codex,
            remote_check_oras,
        )

        if not remote_check_oras(profile.host):
            raise click.ClickException(
                f"oras not found on {profile.host}. "
                "Install with: imas-codex tools install"
            )

        codex_cli_path = remote_check_imas_codex(profile.host)
        if not codex_cli_path:
            raise click.ClickException(
                f"imas-codex CLI not found on {profile.host}. "
                "Install with: cd ~/Code/imas-codex && uv sync"
            )

        # Build artifact refs for each variant
        from imas_codex.graph.ghcr import get_git_info

        git_info = get_git_info()
        full_pkg = get_package_name()
        full_ref = f"{registry}/{full_pkg}:{git_tag}"

        dd_pkg = get_package_name(dd_only=True)
        dd_ref = f"{registry}/{dd_pkg}:{git_tag}"

        facility_refs = None
        if not is_rc and facilities:
            facility_refs = {}
            for fac in facilities:
                fac_pkg = get_package_name(facilities=[fac])
                facility_refs[fac] = f"{registry}/{fac_pkg}:{git_tag}"

        if dry_run:
            click.echo(f"\n  [DRY RUN] Would push from {profile.host}:")
            click.echo(f"    Full: {full_ref}")
            click.echo(f"    DD-only: {dd_ref}")
            if facility_refs:
                for fac, fac_ref in facility_refs.items():
                    click.echo(f"    {fac}: {fac_ref}")
            return

        script = build_remote_release_push_script(
            profile.name,
            full_ref,
            dd_artifact_ref=dd_ref,
            facility_artifact_refs=facility_refs,
            version_tag=git_tag,
            git_commit=git_info["commit"],
            message=message,
            token=None,  # Use cached GHCR creds on remote
            is_dev=False,
            codex_cli_path=codex_cli_path,
            scheduler=_resolve_scheduler_for_release(profile),
        )

        _remote_markers = {
            "STOPPING": f"Stopping Neo4j on {profile.host}",
            "DUMPING": "Dumping graph database",
            "RECOVERY": "Recovery cycle (clean start/stop)",
            "ARCHIVING_FULL": "Creating full graph archive",
            "PUSHING_FULL": "Pushing full graph to GHCR",
            "FILTERING_DD": "Filtering to IMAS DD nodes only",
            "PUSHING_DD": "Pushing DD-only graph to GHCR",
            "STARTING": f"Starting Neo4j on {profile.host}",
            "TAGGING": "Tagging as latest",
            "COMPLETE": "All variants pushed",
        }
        if facility_refs:
            for fac in facility_refs:
                _remote_markers[f"FILTERING_FAC_{fac.upper()}"] = f"Filtering for {fac}"
                _remote_markers[f"PUSHING_FAC_{fac.upper()}"] = f"Pushing {fac} graph"

        with GraphProgress("push") as gp:
            gp.set_total_phases(1)
            gp.start_phase(f"Pushing all variants from {profile.host}")

            try:
                output = remote_operation_streaming(
                    script,
                    profile.host,
                    progress=gp,
                    progress_markers=_remote_markers,
                    timeout=1800,  # 30min for all variants
                )
            except Exception as e:
                gp.fail_phase(str(e))
                raise click.ClickException(
                    f"Remote release push on {profile.host} failed: {e}"
                ) from e

            sizes = {}
            for line in output.strip().splitlines():
                if line.startswith("SIZE_FULL="):
                    sizes["full"] = line.split("=", 1)[1].strip()
                elif line.startswith("SIZE_DD="):
                    sizes["dd"] = line.split("=", 1)[1].strip()

            size_summary = ", ".join(f"{k}: {v}" for k, v in sizes.items())
            gp.complete_phase(size_summary or None)

        click.echo("  ✓ All graph variants pushed")
        for k, v in sizes.items():
            click.echo(f"    {k}: {v}")

        # Dispatch graph-quality CI check
        from imas_codex.graph.ghcr import dispatch_graph_quality

        dispatch_graph_quality(git_info, git_tag, registry)
        return

    # ── Local push path — dump once, reuse for filtered variants ────────
    cached_dump = None
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

    failed: list[str] = []
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

    # Per-facility graphs — only for final releases (skip for RC)
    if is_rc:
        click.echo(
            f"\n  Skipping {len(facilities)} per-facility variant(s) (RC release)."
        )
    else:
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
    default="origin",
    help="Target remote for git tag push (default: origin).",
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
@click.option(
    "--changelog/--no-changelog",
    default=None,
    help="Generate changelog from commits since last tag. Default: on for --final.",
)
@click.option(
    "--changelog-file",
    type=click.Path(),
    default=None,
    help="Write changelog markdown to a file.",
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
    changelog: bool | None,
    changelog_file: str | None,
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
    if final:
        _check_final_targets_upstream(remote)
        _check_ci_passed(remote, dry_run)
    click.echo()

    # Changelog generation (before tag creation so it can be reviewed)
    generate_changelog = changelog if changelog is not None else final
    if generate_changelog:
        click.echo("Generating changelog...")
        cl_text = _generate_changelog(
            from_tag=latest or "HEAD~10",
            version=git_tag,
            message=message,
        )
        click.echo(cl_text)
        if changelog_file:
            Path(changelog_file).write_text(cl_text)
            click.echo(f"  Changelog written to {changelog_file}")
        if final:
            click.echo("  ℹ Include this changelog in your GitHub Release notes.")
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
    _push_all_graph_variants(message, remote, dry_run, git_tag=git_tag, is_rc=is_rc)

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

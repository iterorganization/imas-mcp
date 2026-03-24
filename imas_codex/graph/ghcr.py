"""GitHub Container Registry (GHCR) and ORAS helper functions.

Provides:
- Git state introspection (commit, tag, remote, dirty)
- GHCR registry resolution and authentication
- Version tag computation with dev revision tracking
- ORAS push/pull prerequisites
- GitHub REST API client for package management
- Local graph manifest persistence
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import click

from imas_codex import __version__

# ============================================================================
# Constants
# ============================================================================

LOCAL_GRAPH_MANIFEST = Path.home() / ".config" / "imas-codex" / "graph-manifest.json"

GITHUB_API = "https://api.github.com"

SCOPE_FIX_HINT = (
    "\n  Your token lacks the required GitHub API scopes for package management."
    "\n  Fix: set GHCR_TOKEN to a PAT with read:packages + delete:packages scopes."
    "\n  Create one at: https://github.com/settings/tokens/new"
    "\n    Required scopes: read:packages, write:packages, delete:packages"
)


# ============================================================================
# Git helpers
# ============================================================================


def get_git_info() -> dict:
    """Get current git state: commit, tag, remote, dirty status."""
    info = {
        "commit": None,
        "commit_short": None,
        "tag": None,
        "is_dirty": False,
        "remote_owner": None,
        "remote_url": None,
        "is_fork": False,
    }

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if result.returncode == 0:
        info["commit"] = result.stdout.strip()
        info["commit_short"] = info["commit"][:7]

    result = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        info["tag"] = result.stdout.strip()

    result = subprocess.run(
        ["git", "status", "--porcelain", "-uno"],
        capture_output=True,
        text=True,
    )
    info["is_dirty"] = bool(result.stdout.strip())

    result = subprocess.run(
        ["git", "remote", "get-url", "origin"], capture_output=True, text=True
    )
    if result.returncode == 0:
        url = result.stdout.strip()
        info["remote_url"] = url
        if "github.com" in url:
            if url.startswith("git@"):
                parts = url.split(":")[-1].replace(".git", "").split("/")
            else:
                parts = url.replace(".git", "").split("/")
            if len(parts) >= 2:
                info["remote_owner"] = parts[-2]

        info["is_fork"] = (
            info["remote_owner"] is not None
            and info["remote_owner"].lower() != "iterorganization"
        )

    return info


# ============================================================================
# Registry helpers
# ============================================================================


def get_registry(git_info: dict, force_registry: str | None = None) -> str:
    """Determine GHCR registry based on git remote or explicit override."""
    if force_registry:
        return force_registry

    if git_info["is_fork"] and git_info["remote_owner"]:
        return f"ghcr.io/{git_info['remote_owner'].lower()}"
    return "ghcr.io/iterorganization"


def get_version_tag(
    git_info: dict,
    dev: bool = False,
    version_override: str | None = None,
) -> str:
    """Determine version tag for push.

    For dev pushes, auto-increments a revision counter per base version
    so that successive pushes from the same commit produce unique tags:
      0.5.0.dev123-abc1234-r1, 0.5.0.dev123-abc1234-r2, ...

    The revision counter is tracked in the local graph manifest and
    resets when the base version changes (new commit or version bump).
    """
    if dev:
        version = version_override or __version__
        base = version.replace("+", "-")
        revision = next_dev_revision(base)
        return f"{base}-r{revision}"
    if git_info["tag"]:
        return git_info["tag"]
    raise click.ClickException(
        "Not on a git tag. Use --dev for development push, or create a tag first."
    )


def next_dev_revision(base_version: str) -> int:
    """Get next revision number for a dev push.

    Reads the graph manifest to find the last pushed revision for this
    base version. Returns last_revision + 1, or 1 if no previous push.
    """
    manifest = get_local_graph_manifest()
    if manifest:
        last_base = manifest.get("dev_base_version")
        last_rev = manifest.get("dev_revision", 0)
        if last_base == base_version:
            return last_rev + 1
    return 1


def save_dev_revision(base_version: str, revision: int) -> None:
    """Save the current dev revision to the graph manifest."""
    manifest = get_local_graph_manifest() or {}
    manifest["dev_base_version"] = base_version
    manifest["dev_revision"] = revision
    save_local_graph_manifest(manifest)


def get_package_name(
    facilities: list[str] | None = None,
    *,
    no_imas: bool = False,
    dd_only: bool = False,
) -> str:
    """Get the GHCR package name, optionally scoped to facilities.

    Args:
        facilities: If given, appends sorted facility IDs to the name.
        no_imas: If True, appends ``-no-imas`` suffix.
        dd_only: If True, uses ``imas-codex-graph-dd`` (DD-only graph).

    Returns:
        Package name, e.g. ``"imas-codex-graph-iter-tcv-no-imas"``.
    """
    if dd_only:
        return "imas-codex-graph-dd"
    parts = ["imas-codex-graph"]
    if facilities:
        parts.extend(sorted(facilities))
    if no_imas:
        parts.append("no-imas")
    return "-".join(parts)


# ============================================================================
# Prerequisites
# ============================================================================


def require_clean_git(git_info: dict) -> None:
    if git_info["is_dirty"]:
        raise click.ClickException(
            "Working tree has uncommitted changes. Commit or stash first."
        )


def require_oras() -> None:
    if not shutil.which("oras"):
        raise click.ClickException(
            "oras not found in PATH. Install from: "
            "https://github.com/oras-project/oras/releases"
        )


def require_apptainer() -> None:
    if not shutil.which("apptainer"):
        raise click.ClickException("apptainer not found in PATH")


def require_gh() -> None:
    if not shutil.which("gh"):
        raise click.ClickException(
            "gh CLI not found. Install from: https://cli.github.com/"
        )


# ============================================================================
# GHCR authentication
# ============================================================================


def login_to_ghcr(token: str | None) -> None:
    if not token:
        return

    result = subprocess.run(
        ["oras", "login", "ghcr.io", "-u", "token", "--password-stdin"],
        input=token,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise click.ClickException(f"GHCR login failed: {result.stderr}")


# ============================================================================
# Graph manifest persistence
# ============================================================================


def get_local_graph_manifest() -> dict | None:
    if LOCAL_GRAPH_MANIFEST.exists():
        return json.loads(LOCAL_GRAPH_MANIFEST.read_text())
    return None


def save_local_graph_manifest(manifest: dict) -> None:
    LOCAL_GRAPH_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest["loaded_at"] = datetime.now(UTC).isoformat()
    LOCAL_GRAPH_MANIFEST.write_text(json.dumps(manifest, indent=2))


# ============================================================================
# Version sync
# ============================================================================


def ensure_fresh_version() -> str:
    """Run ``uv sync`` to refresh the installed package version.

    hatch-vcs bakes the version at install time from the git state.
    Without a sync, __version__ can be stale (wrong commit hash and
    dev count).  Returns the refreshed version string.
    """
    click.echo("  Syncing package version (uv sync)...")
    result = subprocess.run(
        ["uv", "sync"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise click.ClickException(
            f"uv sync failed (version would be stale):\n{result.stderr}"
        )
    # Re-read the freshly computed version
    import importlib

    import imas_codex

    importlib.reload(imas_codex)
    return imas_codex.__version__


# ============================================================================
# Registry tag operations
# ============================================================================


def resolve_latest_tag(
    registry: str,
    token: str | None = None,
    pkg_name: str = "imas-codex-graph",
) -> str:
    """Resolve the most recent tag when 'latest' doesn't exist.

    Checks for 'latest' first. If not found, picks the most recent tag
    by sorting: release tags (semver) first, then dev tags by revision.
    """
    tags = list_registry_tags(registry, token, pkg_name)
    if not tags:
        raise click.ClickException(
            f"No graph versions found in {registry}/{pkg_name}.\n"
            "Push a graph first: imas-codex graph push --dev"
        )

    if "latest" in tags:
        return "latest"

    # Sort: prefer release tags (no 'dev'), then by revision number descending
    def _tag_sort_key(tag: str) -> tuple[int, int]:
        is_dev = 1 if ("dev" in tag or "-r" in tag) else 0
        rev = 0
        if "-r" in tag:
            try:
                rev = int(tag.rsplit("-r", 1)[-1])
            except ValueError:
                pass
        return (is_dev, -rev)

    tags.sort(key=_tag_sort_key)
    best = tags[0]
    click.echo(f"No 'latest' tag found. Using most recent: {best}")
    return best


def list_registry_tags(
    registry: str,
    token: str | None = None,
    pkg_name: str = "imas-codex-graph",
) -> list[str]:
    """List all tags in the GHCR registry."""
    login_to_ghcr(token)
    repo_ref = f"{registry}/{pkg_name}"
    result = subprocess.run(
        ["oras", "repo", "tags", repo_ref],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "not found" in result.stderr.lower():
            return []
        raise click.ClickException(f"Failed to list tags: {result.stderr}")
    return [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]


def fetch_tag_messages(
    registry: str,
    tags: list[str],
    *,
    pkg_name: str = "imas-codex-graph",
) -> dict[str, str | None]:
    """Fetch the push message for each tag from OCI manifest annotations.

    Returns a mapping of tag -> message (None if no message was set).
    """
    messages: dict[str, str | None] = {}
    for tag in tags:
        ref = f"{registry}/{pkg_name}:{tag}"
        result = subprocess.run(
            ["oras", "manifest", "fetch", ref],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            messages[tag] = None
            continue
        try:
            manifest = json.loads(result.stdout)
            annotations = manifest.get("annotations", {})
            messages[tag] = annotations.get("org.opencontainers.image.description")
        except (json.JSONDecodeError, AttributeError):
            messages[tag] = None
    return messages


# ============================================================================
# CI dispatch
# ============================================================================


def dispatch_graph_quality(git_info: dict, version_tag: str, registry: str) -> None:
    """Fire a repository_dispatch event to trigger graph quality CI.

    Uses the GitHub CLI (gh) to dispatch a graph-pushed event.
    Silently skips if gh is not available or the dispatch fails.
    """
    if not shutil.which("gh"):
        return

    owner = git_info.get("remote_owner", "iterorganization")
    repo = "imas-codex"

    body = json.dumps(
        {
            "event_type": "graph-pushed",
            "client_payload": {
                "tag": version_tag,
                "registry": registry,
                "commit": git_info.get("commit", ""),
            },
        }
    )

    try:
        result = subprocess.run(
            [
                "gh",
                "api",
                f"repos/{owner}/{repo}/dispatches",
                "--method",
                "POST",
                "--input",
                "-",
            ],
            input=body,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            click.echo("✓ Dispatched graph-quality CI")
        else:
            click.echo(
                f"Warning: graph-quality dispatch failed: {result.stderr.strip()}",
                err=True,
            )
    except (subprocess.TimeoutExpired, Exception) as e:
        click.echo(f"Warning: graph-quality dispatch skipped: {e}", err=True)


# ============================================================================
# GitHub REST API client
# ============================================================================


def github_api_request(
    path: str,
    token: str,
    method: str = "GET",
) -> tuple[int, dict | list | None]:
    """Make a GitHub REST API request. Returns (status_code, json_body)."""
    import urllib.error
    import urllib.request

    url = f"{GITHUB_API}{path}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")

    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode()
            return resp.status, json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            return e.code, json.loads(body) if body else None
        except json.JSONDecodeError:
            return e.code, {"message": body}


def github_api_paginated(
    path: str,
    token: str,
) -> tuple[int, list]:
    """Fetch all pages from a paginated GitHub API endpoint."""
    import urllib.error
    import urllib.request

    all_items: list = []
    url = f"{GITHUB_API}{path}?per_page=100"

    while url:
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")

        try:
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read().decode())
                if isinstance(body, list):
                    all_items.extend(body)
                else:
                    return resp.status, all_items

                # Parse Link header for next page
                link_header = resp.headers.get("Link", "")
                url = None
                for part in link_header.split(","):
                    if 'rel="next"' in part:
                        url = part.split("<")[1].split(">")[0]
        except urllib.error.HTTPError as e:
            body_str = e.read().decode() if e.fp else ""
            try:
                err = json.loads(body_str)
            except json.JSONDecodeError:
                err = {"message": body_str}
            return e.code, err if isinstance(err, list) else [err]

    return 200, all_items


def resolve_token(token: str | None) -> str:
    """Resolve a GitHub token from argument, env var, or gh CLI."""
    if token:
        return token

    # Try GHCR_TOKEN env var
    env_token = os.environ.get("GHCR_TOKEN")
    if env_token:
        return env_token

    # Fall back to gh auth token
    result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()

    raise click.ClickException(
        "No GitHub token found. Provide --token, set GHCR_TOKEN,"
        " or run 'gh auth login'."
    )


def get_ghcr_owner_and_type(registry: str, token: str) -> tuple[str, str]:
    """Extract the owner and API type from a GHCR registry string.

    Returns (owner, api_type) where api_type is 'orgs' or 'users'.
    """
    # registry is like "ghcr.io/owner-name"
    parts = registry.split("/")
    owner = parts[-1] if len(parts) >= 2 else parts[0]

    status, _ = github_api_request(f"/orgs/{owner}", token)
    api_type = "orgs" if status == 200 else "users"
    return owner, api_type


def get_package_version_id(
    owner: str,
    api_type: str,
    tag: str,
    token: str,
    pkg_name: str = "imas-codex-graph",
) -> int | None:
    """Find the GHCR package version ID for a given tag."""
    path = f"/{api_type}/{owner}/packages/container/{pkg_name}/versions"
    status, data = github_api_paginated(path, token)

    if status == 403:
        msg = ""
        if isinstance(data, list) and data:
            msg = data[0].get("message", "") if isinstance(data[0], dict) else ""
        click.echo(f"  Permission denied listing package versions: {msg}", err=True)
        click.echo(SCOPE_FIX_HINT, err=True)
        return None

    if status != 200:
        msg = ""
        if isinstance(data, list) and data:
            msg = (
                data[0].get("message", "")
                if isinstance(data[0], dict)
                else str(data[0])
            )
        click.echo(
            f"  Failed to query package versions (HTTP {status}): {msg}", err=True
        )
        return None

    if not isinstance(data, list):
        click.echo("  Unexpected API response format", err=True)
        return None

    for version in data:
        tags = version.get("metadata", {}).get("container", {}).get("tags", [])
        if tag in tags:
            return version["id"]

    return None


def delete_tag(
    registry: str,
    tag: str,
    token: str | None = None,
    pkg_name: str = "imas-codex-graph",
) -> bool:
    """Delete a specific tag from GHCR using the GitHub Packages API.

    GHCR does not support the OCI manifest delete endpoint (`oras manifest
    delete` returns "unsupported: The operation is unsupported"). We use
    the GitHub REST API directly to find and delete the package version.
    """
    resolved = resolve_token(token)
    owner, api_type = get_ghcr_owner_and_type(registry, resolved)
    version_id = get_package_version_id(owner, api_type, tag, resolved, pkg_name)

    if version_id is None:
        click.echo(f"  Could not find version for tag: {tag}", err=True)
        return False

    path = f"/{api_type}/{owner}/packages/container/{pkg_name}/versions/{version_id}"
    status, resp = github_api_request(path, resolved, method="DELETE")

    if status == 403:
        msg = resp.get("message", "") if isinstance(resp, dict) else ""
        click.echo(f"  Permission denied deleting {tag}: {msg}", err=True)
        click.echo(SCOPE_FIX_HINT, err=True)
        return False

    if status not in (200, 204):
        msg = resp.get("message", "") if isinstance(resp, dict) else str(resp)
        click.echo(f"  Failed to delete {tag} (HTTP {status}): {msg}", err=True)
        return False

    return True

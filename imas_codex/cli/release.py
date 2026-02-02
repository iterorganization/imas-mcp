"""Release command: Version tagging and graph publishing."""

from __future__ import annotations

import re
import subprocess

import click


@click.command("release")
@click.argument("version")
@click.option(
    "-m",
    "--message",
    required=True,
    help="Release message (used for git tag annotation)",
)
@click.option(
    "--remote",
    type=click.Choice(["origin", "upstream"]),
    default="upstream",
    help="Target remote: 'origin' prepares PR, 'upstream' finalizes release",
)
@click.option(
    "--skip-graph",
    is_flag=True,
    help="Skip graph dump and push (upstream mode only)",
)
@click.option(
    "--skip-git",
    is_flag=True,
    help="Skip git tag creation and push",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without making changes",
)
def release(
    version: str,
    message: str,
    remote: str,
    skip_graph: bool,
    skip_git: bool,
    dry_run: bool,
) -> None:
    """Create a new release with two modes based on remote.

    VERSION should be a semantic version with 'v' prefix (e.g., v1.0.0).
    The project version is derived from git tags via hatch-vcs.

    MODE: --remote origin (prepare PR)
    - Creates and pushes tag to origin
    - No graph operations (graph is local-only data)

    MODE: --remote upstream (finalize release - default)
    - Pre-flight: clean tree, synced with upstream
    - Updates _GraphMeta node with version
    - Dumps and pushes graph to GHCR
    - Creates and pushes tag to upstream (triggers CI)

    Workflow:
    1. imas-codex release vX.Y.Z -m 'message' --remote origin
    2. Create PR on GitHub, merge to upstream
    3. git pull upstream main
    4. imas-codex release vX.Y.Z -m 'message' --remote upstream

    Examples:
        # Prepare PR (pushes tag to fork)
        imas-codex release v1.0.0 -m 'Add EPFL' --remote origin

        # Finalize release (graph to GHCR, tag to upstream)
        imas-codex release v1.0.0 -m 'Add EPFL' --remote upstream

        # Dry run
        imas-codex release v1.0.0 -m 'Test' --dry-run
    """
    # Validate version format
    if not re.match(r"^v\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$", version):
        click.echo(f"Error: Invalid version format: {version}", err=True)
        click.echo("Expected format: v1.0.0 or v1.0.0-rc1")
        raise SystemExit(1)

    version_number = version.lstrip("v")

    # Determine mode
    is_origin_mode = remote == "origin"
    mode_desc = "PR preparation" if is_origin_mode else "release finalization"

    click.echo(f"{'[DRY RUN] ' if dry_run else ''}Release {version} ({mode_desc})")
    click.echo(f"Message: {message}")
    click.echo(f"Remote: {remote}")
    click.echo()

    # Pre-flight checks
    click.echo("Pre-flight checks...")

    # Check 1: On main branch
    branch_result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True,
        text=True,
    )
    current_branch = branch_result.stdout.strip()
    if current_branch != "main":
        click.echo(f"  ✗ Not on main branch (current: {current_branch})", err=True)
        click.echo("    Switch to main: git checkout main")
        raise SystemExit(1)
    click.echo("  ✓ On main branch")

    # Check 2: Remote exists
    remote_result = subprocess.run(
        ["git", "remote", "get-url", remote],
        capture_output=True,
        text=True,
    )
    if remote_result.returncode != 0:
        click.echo(f"  ✗ Remote '{remote}' not found", err=True)
        click.echo(f"    Add it: git remote add {remote} <url>")
        raise SystemExit(1)
    click.echo(f"  ✓ Remote '{remote}' exists")

    # For upstream mode: stricter checks
    if not is_origin_mode:
        # Check 3: Clean working tree (required for upstream)
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        if status_result.stdout.strip():
            click.echo("  ✗ Working tree has uncommitted changes", err=True)
            click.echo("    Commit or stash changes first")
            if not dry_run:
                raise SystemExit(1)
        else:
            click.echo("  ✓ Working tree is clean")

        # Check 4: Synced with upstream
        subprocess.run(["git", "fetch", remote, "main"], capture_output=True)
        ahead_behind = subprocess.run(
            ["git", "rev-list", "--left-right", "--count", f"main...{remote}/main"],
            capture_output=True,
            text=True,
        )
        if ahead_behind.returncode == 0:
            parts = ahead_behind.stdout.strip().split()
            if len(parts) == 2:
                ahead, behind = int(parts[0]), int(parts[1])
                if behind > 0:
                    click.echo(
                        f"  ✗ Local is {behind} commits behind {remote}/main", err=True
                    )
                    click.echo(f"    Pull first: git pull {remote} main")
                    if not dry_run:
                        raise SystemExit(1)
                if ahead > 0:
                    click.echo(
                        f"  ✗ Local is {ahead} commits ahead of {remote}/main",
                        err=True,
                    )
                    click.echo("    Ensure PR is merged first")
                    if not dry_run:
                        raise SystemExit(1)
                if ahead == 0 and behind == 0:
                    click.echo(f"  ✓ Synced with {remote}/main")

    click.echo()

    if is_origin_mode:
        _release_origin_mode(version, message, skip_git, dry_run)
    else:
        _release_upstream_mode(
            version, version_number, message, skip_graph, skip_git, dry_run
        )


def _release_origin_mode(
    version: str, message: str, skip_git: bool, dry_run: bool
) -> None:
    """Origin mode: Push branch + tag for PR preparation."""
    # Step 1: Push branch
    click.echo("Step 1: Pushing branch to origin...")
    if not dry_run:
        result = subprocess.run(
            ["git", "push", "origin", "main"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            click.echo(f"Error pushing branch: {result.stderr}", err=True)
            raise SystemExit(1)
        click.echo("  Pushed to origin/main")
    else:
        click.echo("  [would push to origin/main]")

    # Step 2: Create and push tag
    if not skip_git:
        click.echo("\nStep 2: Creating and pushing tag...")
        if not dry_run:
            # Create tag
            result = subprocess.run(
                ["git", "tag", "-a", version, "-m", message],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                if "already exists" in result.stderr:
                    click.echo(f"  Warning: Tag {version} already exists")
                else:
                    click.echo(f"Error creating tag: {result.stderr}", err=True)
                    raise SystemExit(1)
            else:
                click.echo(f"  Created tag: {version}")

            # Push tag to origin
            result = subprocess.run(
                ["git", "push", "origin", version],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                click.echo(f"Error pushing tag: {result.stderr}", err=True)
                raise SystemExit(1)
            click.echo("  Pushed tag to origin")
        else:
            click.echo(f"  [would create and push tag {version} to origin]")
    else:
        click.echo("\nStep 2: Skipped (--skip-git)")

    click.echo()
    if dry_run:
        click.echo("[DRY RUN] No changes made.")
    else:
        click.echo(f"PR preparation complete for {version}!")
        click.echo("\nNext steps:")
        click.echo("  1. Create PR on GitHub from origin/main to upstream/main")
        click.echo("  2. After merge: git pull upstream main")
        click.echo(f"  3. Run: imas-codex release {version} -m '{message}'")


def _release_upstream_mode(
    version: str,
    version_number: str,
    message: str,
    skip_graph: bool,
    skip_git: bool,
    dry_run: bool,
) -> None:
    """Upstream mode: Graph operations, push tag."""
    # Step 1: Validate no private fields in graph
    click.echo("Step 1: Validating graph contains no private fields...")
    if not dry_run:
        try:
            from imas_codex.graph import GraphClient, get_schema

            schema = get_schema()
            private_slots = schema.get_private_slots("Facility")

            if private_slots:
                with GraphClient() as client:
                    # Check Facility nodes for private fields
                    for slot in private_slots:
                        result = client.query(
                            f"MATCH (f:Facility) WHERE f.{slot} IS NOT NULL "
                            f"RETURN f.id AS id, f.{slot} AS value LIMIT 5"
                        )
                        if result:
                            click.echo(
                                f"  ✗ Private field '{slot}' found in graph!",
                                err=True,
                            )
                            for r in result:
                                click.echo(
                                    f"    - Facility {r['id']}: {slot}={r['value']}"
                                )
                            click.echo(
                                "\nPrivate data must not be in graph before OCI push."
                            )
                            click.echo(
                                "Remove with: MATCH (f:Facility) REMOVE f.<field>"
                            )
                            raise SystemExit(1)

                click.echo(f"  ✓ No private fields found (checked: {private_slots})")
            else:
                click.echo("  ✓ No private slots defined in schema")
        except SystemExit:
            raise
        except Exception as e:
            click.echo(f"Warning: Could not validate graph: {e}", err=True)
            click.echo("  Is Neo4j running? Check with: imas-codex data db status")
    else:
        click.echo("  [would validate no private fields in graph]")

    # Step 2: Update _GraphMeta node
    if not skip_graph:
        click.echo("\nStep 2: Updating graph metadata...")
        if not dry_run:
            try:
                from imas_codex.graph import GraphClient

                with GraphClient() as client:
                    facilities_result = client.query(
                        "MATCH (f:Facility) RETURN collect(f.id) as facilities"
                    )
                    facilities = (
                        facilities_result[0]["facilities"] if facilities_result else []
                    )
                    client.query(
                        """
                        MERGE (m:_GraphMeta {id: 'meta'})
                        SET m.version = $version,
                            m.message = $message,
                            m.updated_at = datetime(),
                            m.facilities = $facilities
                        """,
                        version=version_number,
                        message=message,
                        facilities=facilities,
                    )
                    click.echo(f"  _GraphMeta updated: version={version_number}")
                    click.echo(f"  Facilities: {', '.join(facilities)}")
            except Exception as e:
                click.echo(f"Warning: Could not update graph metadata: {e}", err=True)
                click.echo("  Is Neo4j running? Check with: imas-codex data db status")
        else:
            click.echo("  [would update _GraphMeta node in graph]")

        # Step 3: Dump and push graph (unified via data CLI)
        click.echo("\nStep 3: Dumping and pushing graph...")
        if not dry_run:
            # data push handles dump + push with auto stop/start
            result = subprocess.run(
                ["uv", "run", "imas-codex", "data", "push"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                click.echo(f"Error pushing graph: {result.stderr}", err=True)
                click.echo(result.stdout)
                click.echo("\n  Check: GHCR_TOKEN set, Neo4j running")
                raise SystemExit(1)
            click.echo(f"  Pushed to GHCR (version: {version})")
            click.echo("  (Neo4j was auto-stopped/restarted)")
        else:
            click.echo("  [would dump and push graph to GHCR]")
            click.echo("  [Neo4j would be auto-stopped/restarted]")
    else:
        click.echo("\nStep 2-3: Skipped (--skip-graph)")

    # Step 4: Git tag
    if not skip_git:
        click.echo("\nStep 4: Create and push git tag...")
        if not dry_run:
            result = subprocess.run(
                ["git", "tag", "-a", version, "-m", message],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                if "already exists" in result.stderr:
                    click.echo(f"  Warning: Tag {version} already exists")
                else:
                    click.echo(f"Error creating tag: {result.stderr}", err=True)
                    raise SystemExit(1)
            else:
                click.echo(f"  Created tag: {version}")

            result = subprocess.run(
                ["git", "push", "upstream", version],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                click.echo(f"Error pushing tag: {result.stderr}", err=True)
                raise SystemExit(1)
            click.echo("  Pushed tag to upstream")
        else:
            click.echo(f"  [would create tag: {version}]")
            click.echo("  [would push tag to: upstream]")
    else:
        click.echo("\nStep 4: Skipped (--skip-git)")

    click.echo()
    if dry_run:
        click.echo("[DRY RUN] No changes made.")
    else:
        click.echo(f"Release {version} complete!")
        click.echo("Tag pushed to upstream. CI will build and publish the package.")

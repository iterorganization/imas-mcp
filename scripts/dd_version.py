#!/usr/bin/env python
"""Utility script to manage IMAS Data Dictionary versions and build schemas.

This script helps users:
1. List all DD versions available in imas-data-dictionaries package
2. Show which versions have schemas built with detailed metadata
3. Switch to a specific DD version
4. Automatically build schemas for that version if needed
5. Verify the setup is complete

Usage:
    # List all available versions from package and built versions
    dd-version --list

    # List with detailed metadata
    dd-version --list --verbose

    # Switch to a specific version (builds schemas if missing)
    dd-version 3.42.2

    # Switch to git package version (imas-data-dictionary)
    dd-version dev

    # Force rebuild schemas even if they exist
    dd-version 4.0.0 --force-rebuild

    # Check current version and schema status
    dd-version --check
"""

import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import click

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from imas_codex import dd_version  # noqa: E402
from imas_codex.resource_path_accessor import ResourcePathAccessor  # noqa: E402


def get_all_dd_versions():
    """Get all DD versions available in imas-data-dictionaries package.

    Returns:
        List of version strings that can be built, or empty list if package not available.
    """
    try:
        import imas_data_dictionaries

        return imas_data_dictionaries.dd_xml_versions()
    except ImportError:
        return []
    except Exception as e:
        click.echo(
            f"Warning: Could not get DD versions from imas-data-dictionaries: {e}",
            err=True,
        )
        return []


def get_installed_packages():
    """Get information about installed DD packages.

    Returns:
        List of tuples (package_name, version, description, is_default)
    """
    packages = []

    # Check for imas-data-dictionary (git package) - this is the default
    try:
        import imas_data_dictionary

        git_version = getattr(imas_data_dictionary, "__version__", "unknown")
        if git_version != "unknown":
            packages.append(
                (
                    "imas-data-dictionary",
                    git_version,
                    "git development package (default when IMAS_DD_VERSION not set)",
                    True,
                )
            )
    except ImportError:
        pass

    # Check for imas-data-dictionaries (PyPI package)
    try:
        import imas_data_dictionaries

        # Get version using importlib.metadata
        try:
            import importlib.metadata

            pypi_version = importlib.metadata.version("imas-data-dictionaries")
        except Exception:
            pypi_version = "unknown"

        num_versions = len(imas_data_dictionaries.dd_xml_versions())
        packages.append(
            (
                "imas-data-dictionaries",
                pypi_version,
                f"PyPI package with {num_versions} versions available",
                False,
            )
        )
    except ImportError:
        pass

    return packages


def check_schemas_exist(version: str) -> tuple[bool, Path, list[Path]]:
    """Check if schemas exist for a given DD version.

    Returns:
        Tuple of (schemas_exist, schemas_dir, detailed_files)
    """
    path_accessor = ResourcePathAccessor(dd_version=version)
    schemas_dir = path_accessor.version_dir / "schemas"

    # Early return if schemas directory doesn't exist
    if not path_accessor.check_path_exists("schemas"):
        return False, schemas_dir, []

    # Check required schema files
    catalog_path = schemas_dir / "ids_catalog.json"
    detailed_dir = schemas_dir / "detailed"

    if not catalog_path.exists() or not detailed_dir.exists():
        return False, schemas_dir, []

    # Get detailed schema files
    detailed_files = list(detailed_dir.glob("*.json"))
    if not detailed_files:
        return False, schemas_dir, []

    return True, schemas_dir, detailed_files


def get_built_version_metadata(version: str) -> dict:
    """Get detailed metadata about a built DD version.

    Returns:
        Dictionary with metadata: schemas_dir, num_schemas, schema_names, embeddings_exist, database_exists
    """
    path_accessor = ResourcePathAccessor(dd_version=version)
    schemas_exist, schemas_dir, detailed_files = check_schemas_exist(version)

    if not schemas_exist:
        return {"built": False}

    # Get schema names
    schema_names = sorted([f.stem for f in detailed_files])

    # Check for embeddings without creating directory
    # Look for actual cache files with pattern: .{model_name}.pkl or .{model_name}_{hash}.pkl
    embeddings_exist = False
    if path_accessor.check_path_exists("embeddings"):
        embeddings_dir = path_accessor.version_dir / "embeddings"
        # Check for any .pkl files (embeddings cache files start with .)
        cache_files = list(embeddings_dir.glob(".*.pkl"))
        embeddings_exist = len(cache_files) > 0

    # Check for database without creating directory
    # Look for actual database files with pattern: imas_fts.db or imas_fts_{hash}.db
    database_exist = False
    if path_accessor.check_path_exists("database"):
        database_dir = path_accessor.version_dir / "database"
        # Check for any imas_fts*.db files
        db_files = list(database_dir.glob("imas_fts*.db"))
        database_exist = len(db_files) > 0

    return {
        "built": True,
        "schemas_dir": schemas_dir,
        "num_schemas": len(detailed_files),
        "schema_names": schema_names,
        "embeddings_exist": embeddings_exist,
        "database_exist": database_exist,
    }


def build_schemas(version: str, force: bool = False):
    """Build schemas for the specified DD version.

    Args:
        version: The DD version to build schemas for
        force: If True, force rebuild even if schemas exist
    """
    click.echo(
        f"\n{'Rebuilding' if force else 'Building'} schemas for DD version '{version}'..."
    )
    click.echo("This may take a few minutes...\n")

    # Prepare environment
    import os

    env = os.environ.copy()
    env["IMAS_DD_VERSION"] = version

    # Prepare command
    cmd = ["uv", "run", "python", "scripts/build_schemas.py"]
    if force:
        cmd.append("--force")

    # Run the build script
    try:
        subprocess.run(
            cmd,
            env=env,
            cwd=str(project_root),
            check=True,
            capture_output=False,
        )
        click.secho(
            f"\n✓ Schemas built successfully for DD version '{version}'", fg="green"
        )
        return True
    except subprocess.CalledProcessError as e:
        click.secho(f"\n✗ Failed to build schemas: {e}", fg="red", err=True)
        return False
    except FileNotFoundError:
        click.secho(
            "\n✗ Error: 'uv' command not found. Please install uv first:", fg="red"
        )
        click.echo("  https://github.com/astral-sh/uv")
        return False


@click.command()
@click.argument("version", required=False)
@click.option(
    "--list",
    "-l",
    "show_list",
    is_flag=True,
    help="List all available DD versions and their build status",
)
@click.option(
    "--check",
    "-c",
    "show_check",
    is_flag=True,
    help="Check current DD version and schema status",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed metadata (use with --list or --check)",
)
@click.option(
    "--force-rebuild",
    "-f",
    is_flag=True,
    help="Force rebuild schemas even if they exist",
)
def cli(version, show_list, show_check, verbose, force_rebuild):
    """Manage IMAS Data Dictionary versions and schemas.

    If VERSION is provided, switch to that version (building schemas if needed).
    Use 'dev' as VERSION to switch to the git package version (imas-data-dictionary).

    \b
    Examples:
        dd-version                  # Check current version status (default)
        dd-version --list           # List all available versions
        dd-version --list -v        # List with detailed metadata
        dd-version --check          # Check current version status
        dd-version --check -v       # Check with detailed metadata
        dd-version 3.42.2           # Switch to version 3.42.2
        dd-version dev              # Switch to git package version
        dd-version 4.0.0 --force-rebuild  # Force rebuild schemas
    """
    if show_list:
        list_versions(verbose)
    elif show_check:
        check_current(verbose)
    elif version:
        sys.exit(0 if switch_version(version, force_rebuild) else 1)
    else:
        # Default action: run check_current with the verbose flag
        check_current(verbose)


def list_versions(verbose: bool = False):
    """List all available DD versions and their build status."""
    # Show installed packages
    packages = get_installed_packages()
    if packages:
        click.echo("\nInstalled DD Packages:")
        for name, version, description, is_default in packages:
            current_marker = " ← ACTIVE" if version == dd_version else ""
            default_marker = " [DEFAULT]" if is_default else ""
            click.secho(
                f"  • {name} v{version}{default_marker}{current_marker}", bold=True
            )
            click.echo(f"    {description}")
        click.echo()
    else:
        click.secho(
            "\n⚠ No DD packages found. Install imas-data-dictionaries:", fg="yellow"
        )
        click.echo("  pip install imas-data-dictionaries")
        click.echo()
        return

    # Get all available DD versions
    all_versions = get_all_dd_versions()

    # Check for git package version (but don't add to main list since it's accessed via 'dev')
    git_version = None
    git_package_installed = False
    try:
        import imas_data_dictionary

        git_version = getattr(imas_data_dictionary, "__version__", None)
        git_package_installed = True
    except ImportError:
        pass

    if not all_versions and not git_package_installed:
        click.echo("Could not retrieve DD versions from packages.")
        return

    # Show available PyPI versions
    if all_versions:
        click.echo(
            f"Available DD Versions ({len(all_versions)} versions can be built):"
        )

        if not verbose:
            # Compact view: group by major.minor
            grouped = defaultdict(list)
            for v in all_versions:
                parts = v.split(".")
                if len(parts) >= 2:
                    key = f"{parts[0]}.{parts[1]}"
                    grouped[key].append(v)
                else:
                    grouped["other"].append(v)

            # Show grouped versions
            for major_minor in sorted(
                [k for k in grouped.keys() if k != "other"],
                key=lambda x: [int(p) for p in x.split(".")],
            ):
                versions_in_group = sorted(grouped[major_minor])
                # Mark current version with * (only for non-dev versions)
                versions_display = [
                    f"*{v}"
                    if v == dd_version and "dev" not in dd_version.lower()
                    else v
                    for v in versions_in_group
                ]
                click.echo(f"  {major_minor}.x: {', '.join(versions_display)}")

            if "other" in grouped:
                versions_display = [
                    f"*{v}"
                    if v == dd_version and "dev" not in dd_version.lower()
                    else v
                    for v in grouped["other"]
                ]
                click.echo(f"  Other: {', '.join(versions_display)}")
        else:
            # Verbose view: show each version
            for v in sorted(all_versions):
                active_marker = (
                    " ← ACTIVE"
                    if v == dd_version and "dev" not in dd_version.lower()
                    else ""
                )
                click.echo(f"  • {v}{active_marker}")

        click.echo()

    # Show git package version separately (accessed via 'dev' alias)
    if git_package_installed and git_version:
        click.secho("Development Version:", bold=True)
        active_marker = " ← ACTIVE" if "dev" in dd_version.lower() else ""
        click.echo(f"  • {git_version} (use 'dd-version dev'){active_marker}")
        click.echo()

    click.echo("=" * 80)
    click.echo("\nBuilt Versions (schemas already generated):")

    # Check which versions have been built
    built_versions = []

    # Check dev version if it exists
    if git_package_installed and git_version:
        dev_metadata = get_built_version_metadata(git_version)
        if dev_metadata["built"]:
            built_versions.append((git_version, dev_metadata))

    # Check PyPI versions
    for version in all_versions:
        metadata = get_built_version_metadata(version)
        if metadata["built"]:
            built_versions.append((version, metadata))

    if not built_versions:
        click.echo("\n  No versions have been built yet.")
        click.echo("\n  To build schemas for a version, run:")
        click.secho("    dd-version <version>", fg="cyan")
        click.echo("\n  Example:")
        click.secho("    dd-version 3.42.2", fg="cyan")
    else:
        click.echo()
        for version, meta in sorted(built_versions, key=lambda x: x[0]):
            current_marker = " ← ACTIVE" if version == dd_version else ""
            click.secho(f"  {version}{current_marker}", bold=True)
            click.echo(f"    Schemas: {meta['num_schemas']} IDS")

            if verbose:
                # Show detailed metadata only in verbose mode
                schema_list = ", ".join(meta["schema_names"][:10])
                if len(meta["schema_names"]) > 10:
                    schema_list += f", ... ({len(meta['schema_names']) - 10} more)"
                click.echo(f"      IDS: {schema_list}")
                click.echo(
                    f"      Embeddings: {'✓ Available' if meta['embeddings_exist'] else '✗ Missing'}"
                )
                click.echo(
                    f"      Database: {'✓ Available' if meta['database_exist'] else '✗ Missing'}"
                )
                click.echo(f"      Location: {meta['schemas_dir']}")
                click.echo()  # Add blank line between versions in verbose mode
            else:
                # Compact view
                extras = []
                if meta["embeddings_exist"]:
                    extras.append("embeddings")
                if meta["database_exist"]:
                    extras.append("database")
                if extras:
                    click.echo(f"    Extras: {', '.join(extras)}")
                click.echo()  # Add blank line between versions in compact mode

    click.echo()


def check_current(verbose: bool = False):
    """Check current DD version and schema status.

    Args:
        verbose: If True, show detailed metadata about schemas, embeddings, and database
    """
    click.echo("\nCurrent IMAS Data Dictionary version: ", nl=False)
    click.secho(dd_version, fg="cyan", bold=True)

    schemas_exist, schemas_dir, detailed_files = check_schemas_exist(dd_version)

    if schemas_exist:
        click.secho(
            f"\n✓ Schemas available: {len(detailed_files)} IDS schemas found",
            fg="green",
        )

        # Always show metadata
        metadata = get_built_version_metadata(dd_version)
        schema_list = ", ".join(metadata["schema_names"][:10])
        if len(metadata["schema_names"]) > 10:
            schema_list += f", ... ({len(metadata['schema_names']) - 10} more)"
        click.echo(f"  IDS: {schema_list}")
        click.echo(
            f"  Embeddings: {'✓ Available' if metadata['embeddings_exist'] else '✗ Missing'}"
        )
        click.echo(
            f"  Database: {'✓ Available' if metadata['database_exist'] else '✗ Missing'}"
        )
        click.echo(f"  Location: {schemas_dir}")
        click.secho("\nServer is ready to start.", fg="green", bold=True)
    else:
        click.secho(f"\n✗ Schemas missing for version '{dd_version}'", fg="red")
        click.echo(f"  Expected location: {schemas_dir}")
        click.echo("\nTo build schemas, run:")

        # Check if this is a git package version
        try:
            import imas_data_dictionary

            git_version = getattr(imas_data_dictionary, "__version__", None)
            if git_version and dd_version == git_version:
                click.secho("  dd-version dev", fg="cyan")
            else:
                click.secho(f"  dd-version {dd_version}", fg="cyan")
        except ImportError:
            click.secho(f"  dd-version {dd_version}", fg="cyan")

    click.echo()


def switch_version(target_version: str, force_rebuild: bool = False):
    """Switch to a specific DD version and ensure schemas exist.

    Args:
        target_version: The DD version to switch to (or 'dev' for git package version)
        force_rebuild: If True, rebuild schemas even if they exist
    """
    # Check if user is trying to specify an exact dev version string
    if "dev" in target_version.lower() and target_version.lower() != "dev":
        click.secho(
            f"\n✗ Cannot specify exact dev version '{target_version}'.", fg="red"
        )
        click.echo("\nDevelopment versions can only be accessed using the 'dev' alias:")
        click.secho("  dd-version dev", fg="cyan")
        click.echo(
            "\nThe 'dev' alias will automatically use the installed git package version."
        )
        return False

    # Handle 'dev' alias for git package version
    if target_version.lower() == "dev":
        try:
            import imas_data_dictionary

            git_version = getattr(imas_data_dictionary, "__version__", None)
            if not git_version:
                click.secho(
                    "\n✗ Could not determine git package version from imas-data-dictionary.",
                    fg="red",
                )
                return False
            click.echo("\n'dev' alias resolved to git package version: ", nl=False)
            click.secho(git_version, fg="cyan", bold=True)
            target_version = git_version
        except ImportError:
            click.secho(
                "\n✗ imas-data-dictionary (git package) not installed.", fg="red"
            )
            click.echo("\nInstall it with:")
            click.secho(
                "  pip install git+https://github.com/iterorganization/imas-data-dictionary.git",
                fg="cyan",
            )
            return False
    else:
        click.echo("\nSwitching to IMAS Data Dictionary version: ", nl=False)
        click.secho(target_version, fg="cyan", bold=True)

    # Verify the version is available (only check PyPI versions, not dev versions)
    all_versions = get_all_dd_versions()

    if not all_versions:
        click.secho("\n✗ No DD packages found.", fg="red")
        click.echo("\nInstall imas-data-dictionaries with:")
        click.secho("  pip install imas-data-dictionaries", fg="cyan")
        return False

    # For non-dev versions, verify they exist in PyPI package
    if "dev" not in target_version.lower() and target_version not in all_versions:
        click.secho(
            f"\n✗ Version '{target_version}' not available in installed packages.",
            fg="red",
        )
        click.echo(
            f"\nAvailable versions: {', '.join(sorted(all_versions)[:10])}"
            + (
                f", ... ({len(all_versions) - 10} more)"
                if len(all_versions) > 10
                else ""
            )
        )
        click.echo(f"\nUse --list to see all {len(all_versions)} available versions.")
        click.echo("\nTip: Use 'dev' to switch to the git package version.")
        return False

    # Check if schemas exist
    schemas_exist, schemas_dir, detailed_files = check_schemas_exist(target_version)

    if schemas_exist and not force_rebuild:
        click.secho(
            f"\n✓ Schemas already exist: {len(detailed_files)} IDS schemas found",
            fg="green",
        )
        click.echo(f"  Location: {schemas_dir}")
    else:
        # Build schemas
        if not build_schemas(target_version, force=force_rebuild):
            return False

        # Verify schemas were built
        schemas_exist, schemas_dir, detailed_files = check_schemas_exist(target_version)
        if not schemas_exist:
            click.secho(
                f"\n✗ Schema build completed but files not found at: {schemas_dir}",
                fg="red",
            )
            return False

    # Provide instructions for setting environment variable
    click.secho(
        f"\n✓ DD version '{target_version}' is ready to use!", fg="green", bold=True
    )

    # Check if this is a dev version
    if "dev" in target_version.lower():
        click.echo("\nThis is the default development version.")
        click.echo("To use it, either:")
        click.echo(
            "  1. Unset the IMAS_DD_VERSION variable (it will use this version by default):"
        )
        click.secho("     unset IMAS_DD_VERSION", fg="cyan")
        click.echo("  2. Or explicitly set it:")
        click.secho(f"     export IMAS_DD_VERSION={target_version}", fg="cyan")
    else:
        click.echo("\nTo use this version, set the environment variable:")
        click.secho(f"  export IMAS_DD_VERSION={target_version}", fg="cyan")
        click.echo("\nOr add it to your .env file:")
        click.secho(f"  echo 'IMAS_DD_VERSION={target_version}' >> .env", fg="cyan")

    click.echo("\nThen start the server:")
    click.secho("  uv run python -m imas_codex", fg="cyan")
    click.echo()

    return True


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build Pydantic models from LinkML schemas.

Source: imas_codex/schemas/facility.yaml
Output: imas_codex/graph/models.py

To regenerate:
    uv run build-models --force
"""

import logging
import subprocess
import sys
from pathlib import Path

import click


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_schemas_dir() -> Path:
    """Get the schemas directory containing LinkML schemas."""
    return get_project_root() / "imas_codex" / "schemas"


def get_graph_dir() -> Path:
    """Get the graph module directory."""
    return get_project_root() / "imas_codex" / "graph"


def get_core_dir() -> Path:
    """Get the core module directory."""
    return get_project_root() / "imas_codex" / "core"


def get_definitions_dir() -> Path:
    """Get the definitions directory."""
    return get_project_root() / "imas_codex" / "definitions"


def _generate_physics_domain(
    logger: logging.Logger,
    force: bool,
    dry_run: bool,
) -> int:
    """Generate physics domain enum from LinkML schema.

    Returns:
        0 on success, non-zero on failure.
    """
    from scripts.gen_physics_domains import generate_enum_code

    definitions_dir = get_definitions_dir()
    core_dir = get_core_dir()

    schema_file = definitions_dir / "physics" / "domains.yaml"
    output_file = core_dir / "physics_domain.py"

    if not schema_file.exists():
        logger.error(f"Physics domain schema not found: {schema_file}")
        return 1

    # Check if output already exists and is up to date
    if output_file.exists() and not force:
        if schema_file.stat().st_mtime <= output_file.stat().st_mtime:
            logger.info(f"Physics domain up to date at {output_file}")
            return 0
        logger.info("Physics schema newer than enum, regenerating...")

    logger.info(f"Generating physics domain enum from {schema_file}")

    if dry_run:
        click.echo(f"Would generate: {output_file}")
        return 0

    try:
        code = generate_enum_code(schema_file)
        output_file.write_text(code)
        logger.info(f"Generated physics domain written to {output_file}")
        click.echo(f"Generated: {output_file}")
        return 0
    except Exception as e:
        logger.error(f"Failed to generate physics domain: {e}")
        return 1


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all logging except errors")
@click.option(
    "--force", "-f", is_flag=True, help="Force rebuild even if files already exist"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be generated without writing files",
)
def build_models(
    verbose: bool,
    quiet: bool,
    force: bool,
    dry_run: bool,
) -> int:
    """Generate Pydantic models from LinkML schemas.

    This command generates:
    1. Physics domain enum from definitions/physics/domains.yaml
    2. Graph Pydantic models from schemas/facility.yaml
    3. IMAS DD models from schemas/imas_dd.yaml

    Examples:
        build-models              # Generate all models
        build-models -v           # Verbose output
        build-models --dry-run    # Preview without writing
        build-models -f           # Force regeneration
    """
    # Set up logging level
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    try:
        # Generate physics domain enum first (required by other modules)
        physics_result = _generate_physics_domain(logger, force, dry_run)
        if physics_result != 0:
            return physics_result

        # Generate graph models
        schemas_dir = get_schemas_dir()
        graph_dir = get_graph_dir()

        # Ensure graph directory exists
        graph_dir.mkdir(parents=True, exist_ok=True)

        schema_file = schemas_dir / "facility.yaml"
        output_file = graph_dir / "models.py"

        # Validate schema exists
        if not schema_file.exists():
            logger.error(f"Schema file not found: {schema_file}")
            click.echo(f"Error: Schema file not found: {schema_file}", err=True)
            return 1

        # Check if output already exists
        if output_file.exists() and not force:
            # Check if schema is newer than output
            if schema_file.stat().st_mtime <= output_file.stat().st_mtime:
                logger.info(f"Models up to date at {output_file}")
                click.echo(f"Models up to date: {output_file}")
            else:
                logger.info("Schema newer than models, regenerating...")
                force = True  # Force regeneration for this file

        if not output_file.exists() or force:
            logger.info(f"Generating Pydantic models from {schema_file}")

            if not dry_run:
                # Run gen-pydantic
                cmd = ["gen-pydantic", str(schema_file)]

                logger.debug(f"Running command: {' '.join(cmd)}")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if result.returncode != 0:
                    logger.error(f"gen-pydantic failed: {result.stderr}")
                    click.echo(
                        f"Error: gen-pydantic failed:\n{result.stderr}", err=True
                    )
                    return 1

                # Add header comment to generated code
                header = '''"""
Facility Knowledge Graph Pydantic Models.

AUTO-GENERATED from imas_codex/schemas/facility.yaml
DO NOT EDIT THIS FILE DIRECTLY - edit the LinkML schema instead.

To regenerate:
    uv run build-models --force
"""

'''
                generated_code = header + result.stdout

                # Write output
                output_file.write_text(generated_code)
                logger.info(f"Generated models written to {output_file}")
                click.echo(f"Generated: {output_file}")
            else:
                click.echo(f"Would generate: {output_file}")

        # Generate IMAS DD models
        imas_schema_file = schemas_dir / "imas_dd.yaml"
        imas_output_file = graph_dir / "dd_models.py"

        if imas_schema_file.exists():
            needs_regen = not imas_output_file.exists() or force
            if imas_output_file.exists() and not force:
                if imas_schema_file.stat().st_mtime > imas_output_file.stat().st_mtime:
                    needs_regen = True
                    logger.info("IMAS DD schema newer than models, regenerating...")

            if needs_regen:
                logger.info(f"Generating IMAS DD models from {imas_schema_file}")

                if not dry_run:
                    cmd = ["gen-pydantic", str(imas_schema_file)]

                    logger.debug(f"Running command: {' '.join(cmd)}")

                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                    if result.returncode != 0:
                        logger.error(
                            f"gen-pydantic failed for imas_dd: {result.stderr}"
                        )
                        click.echo(
                            f"Error: gen-pydantic failed:\n{result.stderr}", err=True
                        )
                        return 1

                    header = '''"""
IMAS Data Dictionary Knowledge Graph Pydantic Models.

AUTO-GENERATED from imas_codex/schemas/imas_dd.yaml
DO NOT EDIT THIS FILE DIRECTLY - edit the LinkML schema instead.

To regenerate:
    uv run build-models --force
"""

'''
                    generated_code = header + result.stdout
                    imas_output_file.write_text(generated_code)
                    logger.info(
                        f"Generated IMAS DD models written to {imas_output_file}"
                    )
                    click.echo(f"Generated: {imas_output_file}")
                else:
                    click.echo(f"Would generate: {imas_output_file}")
            else:
                logger.info(f"IMAS DD models up to date at {imas_output_file}")
                click.echo(f"IMAS DD models up to date: {imas_output_file}")
        else:
            logger.debug(f"IMAS DD schema not found: {imas_schema_file}")

        return 0

    except Exception as e:
        logger.error(f"Error generating models: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(build_models())

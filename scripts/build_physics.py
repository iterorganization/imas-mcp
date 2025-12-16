#!/usr/bin/env python3
"""
Build physics domain mappings for IDS using LLM inference.

This script generates IDS-to-physics-domain mappings by inferring
the appropriate domain from IDS names and descriptions using an LLM.
The output is stored in the resources directory as a build-time artifact.

Unlike cluster labels which are regenerated on each build, domain mappings
are stable and only regenerated when explicitly requested or when the
mapping file doesn't exist.
"""

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import click
from dotenv import load_dotenv

from imas_codex import dd_version
from imas_codex.core.data_model import PhysicsDomain
from imas_codex.embeddings.openrouter_client import OpenRouterClient
from imas_codex.resource_path_accessor import ResourcePathAccessor
from imas_codex.settings import get_language_model

# Load environment variables from .env file, overriding any existing values
load_dotenv(override=True)

logger = logging.getLogger(__name__)


def get_physics_mapping_path() -> Path:
    """Get the path to the physics domain mapping file."""
    accessor = ResourcePathAccessor(dd_version)
    return accessor.schemas_dir / "physics_domains.json"


def load_ids_catalog() -> dict:
    """Load the IDS catalog from the schemas directory."""
    accessor = ResourcePathAccessor(dd_version)
    catalog_path = accessor.schemas_dir / "ids_catalog.json"

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"IDS catalog not found at {catalog_path}. Run build-schemas first."
        )

    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def get_available_domains() -> list[str]:
    """Get list of available physics domain values."""
    return [domain.value for domain in PhysicsDomain]


def build_prompt(ids_entries: list[dict], domains: list[str]) -> str:
    """Build the LLM prompt for physics domain inference."""
    ids_data = []
    for entry in ids_entries:
        ids_data.append(
            {
                "name": entry["name"],
                "description": entry.get("description", ""),
                "path_count": entry.get("path_count", 0),
            }
        )

    return f"""You are an expert in fusion plasma physics and the IMAS (Integrated Modelling & Analysis Suite) data dictionary.

Classify each IDS (Interface Data Structure) into its most appropriate physics domain.

AVAILABLE PHYSICS DOMAINS:
{json.dumps(domains, indent=2)}

IDS TO CLASSIFY:
{json.dumps(ids_data, indent=2)}

INSTRUCTIONS:
1. For each IDS, select the SINGLE most appropriate domain from the list above
2. Consider the IDS name and description to determine the physics area
3. Use "general" only for IDS that truly don't fit any specific domain
4. Be consistent - similar IDS should have similar domain assignments

RESPOND WITH VALID JSON ONLY - no markdown, no explanation:
{{
  "ids_name_1": "domain_value",
  "ids_name_2": "domain_value",
  ...
}}"""


def infer_domains_with_llm(
    ids_entries: list[dict],
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    """Use LLM to infer physics domains for IDS entries."""
    model = model or get_language_model()
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        raise ValueError("API key required. Set OPENAI_API_KEY environment variable.")

    client = OpenRouterClient(
        model_name=model,
        api_key=api_key,
        base_url=base_url,
    )

    domains = get_available_domains()
    prompt = build_prompt(ids_entries, domains)
    messages = [{"role": "user", "content": prompt}]

    logger.info(f"Inferring physics domains for {len(ids_entries)} IDS using {model}")

    response = client.make_chat_request(messages, model=model, max_tokens=10000)

    # Parse the JSON response
    try:
        # Handle potential markdown code blocks
        content = response.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {e}")
        logger.debug(f"Response was: {response}")
        raise


def generate_fallback_mappings(ids_entries: list[dict]) -> dict[str, str]:
    """Generate fallback mappings when LLM is unavailable."""
    logger.warning("Generating fallback mappings (LLM unavailable)")
    return {entry["name"]: PhysicsDomain.GENERAL.value for entry in ids_entries}


def save_physics_mappings(
    mappings: dict[str, str],
    output_path: Path,
    model: str | None = None,
) -> None:
    """Save physics domain mappings to JSON file."""
    output_data = {
        "metadata": {
            "version": dd_version,
            "created": datetime.now(UTC).isoformat(),
            "description": "LLM-inferred IDS to physics domain mappings",
            "model": model or get_language_model(),
            "total_ids": len(mappings),
        },
        "mappings": mappings,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved physics domain mappings to {output_path}")


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all logging except errors")
@click.option(
    "--force", "-f", is_flag=True, help="Force rebuild even if file already exists"
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="LLM model to use (default from settings: imas-language-model)",
)
@click.option(
    "--fallback",
    is_flag=True,
    help="Use fallback mappings (all GENERAL) without LLM",
)
def build_physics(
    verbose: bool,
    quiet: bool,
    force: bool,
    model: str | None,
    fallback: bool,
) -> int:
    """Build physics domain mappings for IDS using LLM inference.

    This command generates IDS-to-physics-domain mappings by inferring
    the appropriate domain from IDS names and descriptions using an LLM.

    The output is stored in resources/schemas/physics_domains.json.

    Examples:
        build-physics                    # Build only if file doesn't exist
        build-physics -f                 # Force rebuild
        build-physics --fallback         # Generate fallback mappings without LLM
        build-physics --model "openai/gpt-4o"  # Use specific model
    """
    # Set up logging level
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        output_path = get_physics_mapping_path()

        # Check if we need to build
        if output_path.exists() and not force:
            logger.info(f"Physics domain mappings already exist at {output_path}")
            click.echo(f"Physics domain mappings already exist at {output_path}")
            click.echo("Use --force to rebuild")
            return 0

        # Load IDS catalog
        logger.info("Loading IDS catalog...")
        catalog = load_ids_catalog()
        ids_entries = list(catalog.get("ids_catalog", {}).values())

        if not ids_entries:
            logger.error("No IDS entries found in catalog")
            return 1

        logger.info(f"Found {len(ids_entries)} IDS entries")

        # Generate mappings
        if fallback:
            mappings = generate_fallback_mappings(ids_entries)
        else:
            try:
                mappings = infer_domains_with_llm(ids_entries, model=model)
            except Exception as e:
                logger.error(f"LLM inference failed: {e}")
                if verbose:
                    logger.exception("Full traceback:")
                logger.info("Falling back to default mappings")
                mappings = generate_fallback_mappings(ids_entries)

        # Save mappings
        save_physics_mappings(mappings, output_path, model=model)

        click.echo(f"Built physics domain mappings for {len(mappings)} IDS")
        return 0

    except FileNotFoundError as e:
        logger.error(str(e))
        click.echo(f"Error: {e}", err=True)
        click.echo("Run 'build-schemas' first to generate the IDS catalog.", err=True)
        return 1
    except Exception as e:
        logger.error(f"Error building physics mappings: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(build_physics())

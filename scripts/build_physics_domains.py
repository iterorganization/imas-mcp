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
from linkml_runtime.utils.schemaview import SchemaView

from imas_codex import dd_version
from imas_codex.core.data_model import PhysicsDomain
from imas_codex.definitions.physics import DOMAINS_SCHEMA
from imas_codex.embeddings.openrouter_client import OpenRouterClient
from imas_codex.resource_path_accessor import ResourcePathAccessor
from imas_codex.settings import get_language_model

# Load environment variables from .env file, overriding any existing values
load_dotenv(override=True)

logger = logging.getLogger(__name__)


class PhysicsDomainBuildError(Exception):
    """Raised when physics domain generation fails with all-general fallback."""

    pass


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


def compute_domain_counts(mappings: dict[str, str]) -> dict[str, int]:
    """Compute the count of IDS in each physics domain.

    Args:
        mappings: Dictionary mapping IDS names to domain values.

    Returns:
        Dictionary mapping domain values to their IDS counts.
    """
    counts: dict[str, int] = {}
    for domain in mappings.values():
        counts[domain] = counts.get(domain, 0) + 1
    return dict(sorted(counts.items()))


def get_domain_descriptions() -> dict[str, str]:
    """Get domain descriptions from the LinkML schema.

    Returns:
        Dictionary mapping domain values to their descriptions.
    """
    sv = SchemaView(str(DOMAINS_SCHEMA))
    enum_def = sv.get_enum("PhysicsDomain")
    if not enum_def:
        # Fallback to simple value list
        return {domain.value: domain.value for domain in PhysicsDomain}

    descriptions = {}
    for pv_name, pv in enum_def.permissible_values.items():
        descriptions[pv_name] = pv.description or pv_name
    return descriptions


def load_existing_mappings() -> dict[str, str]:
    """Load existing mappings from resources file if it exists.

    Returns:
        Dictionary mapping IDS names to physics domain strings.
        Returns empty dict if file doesn't exist.
    """
    mapping_path = get_physics_mapping_path()
    if not mapping_path.exists():
        return {}

    try:
        with open(mapping_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("ids_domain_mappings", {})
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load existing mappings: {e}")
        return {}


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

    # Get domain descriptions from LinkML schema for better categorization
    domain_descriptions = get_domain_descriptions()
    domains_with_desc = {
        domain: domain_descriptions.get(domain, domain) for domain in domains
    }

    return f"""You are an expert in fusion plasma physics and the IMAS (Integrated Modelling & Analysis Suite) data dictionary.

Classify each IDS (Interface Data Structure) into its most appropriate physics domain.

AVAILABLE PHYSICS DOMAINS (value: description):
{json.dumps(domains_with_desc, indent=2)}

IDS TO CLASSIFY:
{json.dumps(ids_data, indent=2)}

INSTRUCTIONS:
1. For each IDS, select the SINGLE most appropriate domain from the list above
2. Consider the IDS name and description to determine the physics area
3. Match IDS to domains based on the domain descriptions provided
4. Use "general" only for IDS that truly don't fit any specific domain
5. Be consistent - similar IDS should have similar domain assignments

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
            # Extract content between code fences
            lines = content.split("\n")
            # Find start and end of code block
            start_idx = 0
            end_idx = len(lines)
            for i, line in enumerate(lines):
                if line.startswith("```") and i == 0:
                    start_idx = 1
                elif line.startswith("```") and i > 0:
                    end_idx = i
                    break
            content = "\n".join(lines[start_idx:end_idx])
            # Remove json language identifier if present
            if content.startswith("json"):
                content = content[4:].strip()

        result = json.loads(content)

        # Validate that result is a dict with string values from valid domains
        valid_domains = set(get_available_domains())
        validated_result = {}
        for ids_name, domain in result.items():
            if domain in valid_domains:
                validated_result[ids_name] = domain
            else:
                logger.warning(
                    f"Invalid domain '{domain}' for IDS '{ids_name}', using 'general'"
                )
                validated_result[ids_name] = PhysicsDomain.GENERAL.value

        return validated_result
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {e}")
        logger.error(f"Response was: {response[:500]}...")
        raise


def generate_fallback_mappings(ids_entries: list[dict]) -> dict[str, str]:
    """Generate fallback mappings when LLM is unavailable."""
    logger.warning("Generating fallback mappings (LLM unavailable)")
    return {entry["name"]: PhysicsDomain.GENERAL.value for entry in ids_entries}


def validate_mappings(mappings: dict[str, str], total_ids: int) -> bool:
    """Validate that mappings are meaningful (not all general).

    Args:
        mappings: Dictionary mapping IDS names to domain values.
        total_ids: Total number of IDS expected.

    Returns:
        True if mappings are valid, False if all are 'general'.
    """
    if not mappings:
        return False

    general_count = sum(
        1 for d in mappings.values() if d == PhysicsDomain.GENERAL.value
    )
    # Allow up to 10% general or at least 3 general IDS
    max_general = max(3, int(total_ids * 0.1))
    return general_count <= max_general


def save_physics_mappings(
    mappings: dict[str, str],
    output_path: Path,
    model: str | None = None,
) -> None:
    """Save physics domain mappings to JSON file.

    Simplified structure without DD version key since resources
    are already organized by version.
    """
    # Compute domain counts for summary
    domain_counts = compute_domain_counts(mappings)

    output_data = {
        "metadata": {
            "created": datetime.now(UTC).isoformat(),
            "description": "LLM-inferred IDS to physics domain mappings",
            "model": model or get_language_model(),
            "total_ids": len(mappings),
        },
        "physics_domains": domain_counts,
        "ids_domain_mappings": mappings,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, sort_keys=True)

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
    help="Use fallback mappings (all GENERAL) without LLM - for testing only",
)
def build_physics_domains(
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

    Raises PhysicsDomainBuildError if LLM inference fails and all IDS
    would be assigned to 'general' (unless --fallback is explicitly set).

    Examples:
        build-physics-domains                    # Build only if file doesn't exist
        build-physics-domains -f                 # Force rebuild
        build-physics-domains --fallback         # Generate fallback mappings without LLM
        build-physics-domains --model "openai/gpt-4o"  # Use specific model
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

        # Load existing mappings from resources if available
        existing_mappings = load_existing_mappings()
        if existing_mappings:
            logger.info(f"Loaded {len(existing_mappings)} existing mappings")

        # Check if we need to build
        if output_path.exists() and not force:
            # Validate existing mappings are meaningful
            if validate_mappings(existing_mappings, len(existing_mappings)):
                logger.info(f"Physics domain mappings already exist at {output_path}")
                click.echo(f"Physics domain mappings already exist at {output_path}")
                click.echo("Use --force to rebuild")
                return 0
            else:
                logger.warning("Existing mappings are all 'general', rebuilding...")

        # Load IDS catalog
        logger.info("Loading IDS catalog...")
        catalog = load_ids_catalog()
        ids_entries = list(catalog.get("ids_catalog", {}).values())

        if not ids_entries:
            logger.error("No IDS entries found in catalog")
            return 1

        logger.info(f"Found {len(ids_entries)} IDS entries")
        total_ids = len(ids_entries)

        # Generate mappings
        if fallback:
            # Explicit fallback requested - allow all-general
            mappings = generate_fallback_mappings(ids_entries)
            logger.warning("Using fallback mappings (all GENERAL) as requested")
        else:
            try:
                logger.info("Inferring physics domains via LLM...")
                mappings = infer_domains_with_llm(ids_entries, model=model)

                # Validate mappings are meaningful
                if not validate_mappings(mappings, total_ids):
                    raise PhysicsDomainBuildError(
                        "LLM inference produced all 'general' mappings. "
                        "This indicates the LLM failed to classify IDS properly. "
                        "Check API key and model availability, or use --fallback "
                        "to explicitly accept all-general mappings."
                    )

            except PhysicsDomainBuildError:
                raise
            except Exception as e:
                logger.error(f"LLM inference failed: {e}")
                if verbose:
                    logger.exception("Full traceback:")

                # Check if we have valid existing mappings to fall back to
                if existing_mappings and validate_mappings(
                    existing_mappings, total_ids
                ):
                    logger.info("Using existing valid mappings as fallback")
                    mappings = existing_mappings
                else:
                    raise PhysicsDomainBuildError(
                        f"LLM inference failed: {e}. "
                        "No valid existing mappings available. "
                        "Set OPENAI_API_KEY and ensure model is accessible, "
                        "or use --fallback to generate placeholder mappings."
                    ) from e

        # Save to resources
        save_physics_mappings(mappings, output_path, model=model)

        # Show summary
        domain_counts = compute_domain_counts(mappings)
        click.echo(f"Built physics domain mappings for {len(mappings)} IDS")
        click.echo(f"Saved to {output_path}")
        click.echo(f"Domain distribution: {len(domain_counts)} domains used")
        return 0

    except FileNotFoundError as e:
        logger.error(str(e))
        click.echo(f"Error: {e}", err=True)
        click.echo("Run 'build-schemas' first to generate the IDS catalog.", err=True)
        return 1
    except PhysicsDomainBuildError as e:
        logger.error(str(e))
        click.echo(f"Error: {e}", err=True)
        return 1
    except Exception as e:
        logger.error(f"Error building physics mappings: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(build_physics_domains())

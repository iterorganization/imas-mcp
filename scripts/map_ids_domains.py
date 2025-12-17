#!/usr/bin/env python3
"""
Map IDS to physics domains using LLM inference.

This script generates IDS-to-physics-domain mappings by:
1. Collecting ALL IDS names across ALL versions from imas-data-dictionaries
2. Using an LLM to infer the appropriate physics domain for each IDS
3. Saving to definitions/physics/ids_domains.json (version-controlled)

The mappings are version-independent - they cover all IDS that have ever
existed in any DD version, so new users don't need an API key.
"""

import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

import click
from dotenv import load_dotenv
from imas_data_dictionaries import dd_xml_versions, get_dd_xml
from linkml_runtime.utils.schemaview import SchemaView

from imas_codex.core.data_model import PhysicsDomain
from imas_codex.definitions.physics import DOMAINS_SCHEMA, IDS_DOMAINS_FILE
from imas_codex.embeddings.openrouter_client import OpenRouterClient
from imas_codex.settings import get_language_model

# Load environment variables from .env file, overriding any existing values
load_dotenv(override=True)

logger = logging.getLogger(__name__)


class IdsDomainMappingError(Exception):
    """Raised when IDS domain mapping fails."""

    pass


def collect_all_ids() -> dict[str, dict]:
    """Collect all IDS across all DD versions from imas-data-dictionaries.

    Returns:
        Dictionary mapping IDS names to their metadata including:
        - description: IDS description (from latest version that has it)
        - versions: List of DD versions where this IDS exists
    """
    all_ids: dict[str, dict] = {}
    versions = dd_xml_versions()

    logger.info(f"Scanning {len(versions)} DD versions for IDS...")

    for version in versions:
        try:
            xml_content = get_dd_xml(version)
            root = ET.fromstring(xml_content)

            for ids_elem in root.findall(".//IDS"):
                ids_name = ids_elem.get("name")
                if not ids_name:
                    continue

                # Get description from documentation attribute or child element
                description = ids_elem.get("documentation", "")
                if not description:
                    doc_elem = ids_elem.find("documentation")
                    if doc_elem is not None and doc_elem.text:
                        description = doc_elem.text

                if ids_name not in all_ids:
                    all_ids[ids_name] = {
                        "name": ids_name,
                        "description": description,
                        "versions": [],
                    }
                all_ids[ids_name]["versions"].append(version)
                # Update description if we have a better one
                if description and not all_ids[ids_name]["description"]:
                    all_ids[ids_name]["description"] = description
        except Exception as e:
            logger.warning(f"Failed to load DD version {version}: {e}")

    logger.info(f"Found {len(all_ids)} unique IDS across all versions")
    return all_ids


def get_available_domains() -> list[str]:
    """Get list of available physics domain values."""
    return [domain.value for domain in PhysicsDomain]


def compute_domain_counts(mappings: dict[str, str]) -> dict[str, int]:
    """Compute the count of IDS in each physics domain."""
    counts: dict[str, int] = {}
    for domain in mappings.values():
        counts[domain] = counts.get(domain, 0) + 1
    return dict(sorted(counts.items()))


def get_domain_descriptions() -> dict[str, str]:
    """Get domain descriptions from the LinkML schema."""
    sv = SchemaView(str(DOMAINS_SCHEMA))
    enum_def = sv.get_enum("PhysicsDomain")
    if not enum_def:
        return {domain.value: domain.value for domain in PhysicsDomain}

    descriptions = {}
    for pv_name, pv in enum_def.permissible_values.items():
        descriptions[pv_name] = pv.description or pv_name
    return descriptions


def load_existing_mappings() -> dict[str, str]:
    """Load existing mappings from ids_domains.json if it exists."""
    if not IDS_DOMAINS_FILE.exists():
        return {}

    try:
        with open(IDS_DOMAINS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("ids_domain_mappings", {})
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load existing mappings: {e}")
        return {}


def build_prompt(ids_entries: list[dict], domains: list[str]) -> str:
    """Build the LLM prompt for physics domain inference."""
    ids_data = [
        {"name": entry["name"], "description": entry.get("description", "")}
        for entry in ids_entries
    ]

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
        content = response.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            start_idx = 0
            end_idx = len(lines)
            for i, line in enumerate(lines):
                if line.startswith("```") and i == 0:
                    start_idx = 1
                elif line.startswith("```") and i > 0:
                    end_idx = i
                    break
            content = "\n".join(lines[start_idx:end_idx])
            if content.startswith("json"):
                content = content[4:].strip()

        result = json.loads(content)

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


def validate_mappings(mappings: dict[str, str], total_ids: int) -> bool:
    """Validate that mappings are meaningful (not all general)."""
    if not mappings:
        return False

    general_count = sum(
        1 for d in mappings.values() if d == PhysicsDomain.GENERAL.value
    )
    # Allow up to 10% general or at least 3 general IDS
    max_general = max(3, int(total_ids * 0.1))
    return general_count <= max_general


def save_ids_domains(
    mappings: dict[str, str],
    ids_metadata: dict[str, dict],
    model: str | None = None,
) -> None:
    """Save IDS domain mappings to definitions/physics/ids_domains.json."""
    domain_counts = compute_domain_counts(mappings)

    # Build version coverage info
    version_coverage = sorted(
        {v for meta in ids_metadata.values() for v in meta.get("versions", [])}
    )

    output_data = {
        "metadata": {
            "created": datetime.now(UTC).isoformat(),
            "description": "IDS to physics domain mappings across all DD versions",
            "model": model or get_language_model(),
            "total_ids": len(mappings),
            "dd_versions_covered": version_coverage,
        },
        "physics_domains": domain_counts,
        "ids_domain_mappings": dict(sorted(mappings.items())),
    }

    IDS_DOMAINS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(IDS_DOMAINS_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved IDS domain mappings to {IDS_DOMAINS_FILE}")


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
    "--update-only",
    is_flag=True,
    help="Only add new IDS, preserve existing mappings",
)
def map_ids_domains(
    verbose: bool,
    quiet: bool,
    force: bool,
    model: str | None,
    update_only: bool,
) -> int:
    """Map IDS to physics domains using LLM inference.

    Collects ALL IDS names across ALL DD versions from imas-data-dictionaries
    and uses an LLM to infer physics domains. Output is saved to
    definitions/physics/ids_domains.json (version-controlled).

    Examples:
        map-ids-domains                  # Build only if file doesn't exist
        map-ids-domains -f               # Force complete rebuild
        map-ids-domains --update-only    # Only map new IDS, keep existing
        map-ids-domains --model "openai/gpt-4o"  # Use specific model
    """
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
        # Load existing mappings
        existing_mappings = load_existing_mappings()
        if existing_mappings:
            logger.info(f"Loaded {len(existing_mappings)} existing mappings")

        # Check if we need to build
        if IDS_DOMAINS_FILE.exists() and not force and not update_only:
            if validate_mappings(existing_mappings, len(existing_mappings)):
                logger.info(f"IDS domain mappings already exist at {IDS_DOMAINS_FILE}")
                click.echo(f"IDS domain mappings already exist at {IDS_DOMAINS_FILE}")
                click.echo("Use --force to rebuild or --update-only to add new IDS")
                return 0

        # Collect all IDS from all DD versions
        all_ids = collect_all_ids()

        if not all_ids:
            logger.error("No IDS found across DD versions")
            return 1

        # Determine which IDS need mapping
        if update_only and existing_mappings:
            ids_to_map = {
                name: meta
                for name, meta in all_ids.items()
                if name not in existing_mappings
            }
            if not ids_to_map:
                click.echo("All IDS already have mappings, nothing to update")
                return 0
            logger.info(f"{len(ids_to_map)} new IDS need mapping")
        else:
            ids_to_map = all_ids

        # Infer domains with LLM
        logger.info("Inferring physics domains via LLM...")
        ids_entries = list(ids_to_map.values())
        new_mappings = infer_domains_with_llm(ids_entries, model=model)

        # Validate new mappings
        if not validate_mappings(new_mappings, len(ids_entries)):
            raise IdsDomainMappingError(
                "LLM inference produced all 'general' mappings. "
                "Check API key and model availability."
            )

        # Merge with existing if update_only
        if update_only and existing_mappings:
            mappings = {**existing_mappings, **new_mappings}
        else:
            mappings = new_mappings

        # Save to definitions
        save_ids_domains(mappings, all_ids, model=model)

        domain_counts = compute_domain_counts(mappings)
        click.echo(f"Mapped {len(mappings)} IDS to physics domains")
        click.echo(f"Saved to {IDS_DOMAINS_FILE}")
        click.echo(f"Domain distribution: {len(domain_counts)} domains used")
        return 0

    except IdsDomainMappingError as e:
        logger.error(str(e))
        click.echo(f"Error: {e}", err=True)
        return 1
    except Exception as e:
        logger.error(f"Error mapping IDS domains: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(map_ids_domains())

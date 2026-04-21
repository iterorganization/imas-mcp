"""Rich grammar context for SN compose prompts.

Imports grammar context from imas_standard_names public API
(``get_grammar_context()``) and augments with codex-specific data
(curated examples, tokamak parameter ranges, enum lists).

Caches assembled context in-process (module-level dict).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cached context builder
# ---------------------------------------------------------------------------

_CONTEXT_CACHE: dict[str, Any] | None = None


def _get_isn_context() -> dict[str, Any]:
    """Return the ISN grammar context (cached by ISN internally)."""
    from imas_standard_names.grammar.context import get_grammar_context

    return get_grammar_context()


def build_compose_context() -> dict[str, Any]:
    """Build rich context dict for sn/compose_system.md template.

    Pulls all grammar, vocabulary, field-guidance, tag, and applicability
    data from ISN's ``get_grammar_context()`` public API, then augments
    with codex-specific data (curated examples, tokamak ranges, enum lists).

    Returns keys needed by both system and user prompts:
    - grammar_rules: canonical pattern, order constraint, template rules
    - vocabulary: per-segment token lists with descriptions
    - segment_descriptions: detailed segment usage guidance
    - field_guidance: per-field content rules and validation
    - examples: curated standard name examples (YAML)
    - tokamak_ranges: machine parameter data for grounding
    - exclusive_pairs: mutually exclusive segment pairs
    - naming_guidance, kind_definitions, anti_patterns, quick_start,
      common_patterns, critical_distinctions, vocabulary_usage_stats,
      base_requirements, type_specific_requirements, documentation_guidance
    - enum lists: subjects, positions, etc. (for user prompt backward compat)
    """
    global _CONTEXT_CACHE
    if _CONTEXT_CACHE is not None:
        return _CONTEXT_CACHE

    # Single call to ISN's public API provides all grammar context
    isn = _get_isn_context()

    ctx: dict[str, Any] = {}

    # Grammar rules (from ISN)
    ctx["canonical_pattern"] = isn["canonical_pattern"]
    ctx["segment_order"] = isn["segment_order"]
    ctx["template_rules"] = isn["template_rules"]
    ctx["exclusive_pairs"] = isn["exclusive_pairs"]

    # Vocabulary with descriptions (from ISN)
    ctx["vocabulary_sections"] = isn["vocabulary_sections"]

    # Segment descriptions and usage guidance (from ISN)
    ctx["segment_descriptions"] = isn["segment_descriptions"]

    # Field guidance for documentation generation (from ISN)
    ctx["field_guidance"] = isn["field_guidance"]

    # Tag descriptions — primary + secondary (from ISN)
    ctx["tag_descriptions"] = isn["tag_descriptions"]

    # Applicability rules (from ISN)
    ctx["applicability"] = isn["applicability"]

    # New ISN-provided keys
    ctx["naming_guidance"] = isn["naming_guidance"]
    ctx["kind_definitions"] = isn["kind_definitions"]
    ctx["anti_patterns"] = isn["anti_patterns"]
    ctx["quick_start"] = isn["quick_start"]
    ctx["common_patterns"] = isn["common_patterns"]
    ctx["critical_distinctions"] = isn["critical_distinctions"]
    ctx["vocabulary_usage_stats"] = isn["vocabulary_usage_stats"]
    ctx["base_requirements"] = isn["base_requirements"]
    ctx["type_specific_requirements"] = isn["type_specific_requirements"]
    ctx["documentation_guidance"] = isn["documentation_guidance"]

    # Codex-specific data (not from ISN)
    ctx["examples"] = _load_curated_examples()
    ctx["tokamak_ranges"] = _load_tokamak_ranges()

    # Physics domain enum (for prompt context — LLM doesn't set it but
    # needs domain awareness for better naming decisions)
    from imas_codex.core.physics_domain import PhysicsDomain

    ctx["physics_domains"] = [e.value for e in PhysicsDomain]

    # Bare enum lists (backward compat for user prompt)
    ctx.update(_build_enum_lists())

    _CONTEXT_CACHE = ctx
    return ctx


def clear_context_cache() -> None:
    """Clear cached context (for testing)."""
    global _CONTEXT_CACHE
    _CONTEXT_CACHE = None


# ---------------------------------------------------------------------------
# Curated examples
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_curated_examples() -> list[dict[str, Any]]:
    """Load all curated standard name examples from imas_standard_names resources."""
    import imas_standard_names

    pkg_path = Path(imas_standard_names.__path__[0])
    examples_dir = pkg_path / "resources" / "standard_name_examples"

    if not examples_dir.exists():
        logger.warning("No curated examples directory at %s", examples_dir)
        return []

    examples = []
    for yml_path in sorted(examples_dir.rglob("*.yml")):
        try:
            with open(yml_path) as f:
                data = yaml.safe_load(f)
            if data and isinstance(data, dict) and "name" in data:
                # Add the category from directory name
                data["category"] = yml_path.parent.name
                examples.append(data)
        except Exception:
            logger.debug("Failed to load example: %s", yml_path)

    logger.info("Loaded %d curated standard name examples", len(examples))
    return examples


# ---------------------------------------------------------------------------
# Tokamak parameters
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_tokamak_ranges() -> dict[str, dict[str, Any]]:
    """Load tokamak machine parameters for documentation grounding."""
    import imas_standard_names

    pkg_path = Path(imas_standard_names.__path__[0])
    params_dir = pkg_path / "resources" / "tokamak_parameters"

    if not params_dir.exists():
        logger.warning("No tokamak parameters directory at %s", params_dir)
        return {}

    machines: dict[str, dict[str, Any]] = {}
    for yml_path in sorted(params_dir.glob("*.yml")):
        if yml_path.name in ("schema.yml", "README.md"):
            continue
        try:
            with open(yml_path) as f:
                data = yaml.safe_load(f)
            if data and isinstance(data, dict) and "machine" in data:
                machines[data["machine"]] = data
        except Exception:
            logger.debug("Failed to load tokamak params: %s", yml_path)

    logger.info("Loaded %d tokamak parameter sets", len(machines))
    return machines


# ---------------------------------------------------------------------------
# Backward-compatible enum lists
# ---------------------------------------------------------------------------


def _build_enum_lists() -> dict[str, list[str]]:
    """Build bare enum lists for user prompt template variables."""
    from imas_standard_names.grammar import (
        BinaryOperator,
        Component,
        GeometricBase,
        Object,
        Position,
        Process,
        Subject,
        Transformation,
    )

    return {
        "subjects": [e.value for e in Subject],
        "positions": [e.value for e in Position],
        "components": [e.value for e in Component],
        "coordinates": [e.value for e in Component],  # same enum
        "processes": [e.value for e in Process],
        "transformations": [e.value for e in Transformation],
        "geometric_bases": [e.value for e in GeometricBase],
        "objects": [e.value for e in Object],
        "binary_operators": [e.value for e in BinaryOperator],
    }


def build_domain_vocabulary_preseed(domain: str | None) -> str:
    """Build a vocabulary pre-seed section for compose prompts.

    Queries the graph for all StandardName nodes in *domain* that are
    ``pipeline_status IN ['drafted', 'published', 'accepted']`` AND
    ``validation_status = 'valid'``, returning up to 40 canonical
    ``(name, description-first-sentence)`` pairs.

    Ordered by pipeline_status priority (accepted > published > drafted),
    then alphabetically by name.

    Returns empty string when *domain* is None or no names are found.
    """
    if not domain:
        return ""

    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            rows = gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.physics_domain = $domain
                  AND sn.pipeline_status IN ['drafted', 'published', 'accepted']
                  AND sn.validation_status = 'valid'
                RETURN sn.id AS name,
                       sn.description AS description,
                       sn.pipeline_status AS pipeline_status
                ORDER BY
                    CASE sn.pipeline_status
                        WHEN 'accepted' THEN 0
                        WHEN 'published' THEN 1
                        WHEN 'drafted' THEN 2
                    END,
                    sn.id
                LIMIT 40
                """,
                domain=domain,
            )
            if not rows:
                return ""

            lines = []
            for row in rows:
                name = row.get("name", "")
                desc = row.get("description", "")
                # Take first sentence only
                first_sentence = desc.split(". ")[0].rstrip(".") + "." if desc else ""
                lines.append(f"- `{name}`: {first_sentence}")

            return "\n".join(lines)
    except Exception:
        logger.debug("Domain vocabulary preseed unavailable", exc_info=True)
        return ""


def render_cocos_guidance(label: str, cocos_params: dict) -> str:
    """Render sign guidance for a transformation label using COCOS node properties.

    Args:
        label: COCOS transformation label (e.g., 'psi_like')
        cocos_params: Properties dict from the COCOS graph node
            (sigma_bp, psi_increasing_outward, phi_increasing_ccw, etc.)

    Returns:
        Rendered guidance string for the LLM prompt.
    """
    from imas_codex.llm.prompt_loader import load_prompt_config

    config = load_prompt_config("cocos_sign_guidance")
    label_config = config.get("labels", {}).get(label)
    if not label_config:
        return config.get("generic_fallback", "")

    guidance = label_config["guidance"]

    # Substitute raw Sauter parameters directly
    for param in ("sigma_bp", "sigma_r_phi_z", "sigma_rho_theta_phi", "e_bp"):
        guidance = guidance.replace(f"{{{param}}}", str(cocos_params.get(param, "?")))

    # Resolve template variables from COCOS node boolean/sign properties
    for var_name, var_spec in label_config.get("variables", {}).items():
        source_prop = var_spec["from"]
        source_val = cocos_params.get(source_prop)
        # Normalize lookup key: booleans → "true"/"false", numbers → str
        if isinstance(source_val, bool):
            lookup_key = str(source_val).lower()
        else:
            lookup_key = str(source_val)
        replacement = var_spec.get(lookup_key, f"[unknown {source_prop}]")
        guidance = guidance.replace(f"{{{var_name}}}", replacement)

    return guidance

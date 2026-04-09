"""Rich grammar context for SN compose prompts.

Imports segment rules, vocabulary, field guidance, and curated examples
from imas_standard_names backing functions.  Assembles them into template
variables for Jinja2 rendering.

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


def build_compose_context() -> dict[str, Any]:
    """Build rich context dict for sn/compose_system.md template.

    Returns keys needed by both system and user prompts:
    - grammar_rules: canonical pattern, order constraint, template rules
    - vocabulary: per-segment token lists with descriptions
    - segment_descriptions: detailed segment usage guidance
    - field_guidance: per-field content rules and validation
    - examples: curated standard name examples (YAML)
    - tokamak_ranges: machine parameter data for grounding
    - exclusive_pairs: mutually exclusive segment pairs
    - enum lists: subjects, positions, etc. (for user prompt backward compat)
    """
    global _CONTEXT_CACHE
    if _CONTEXT_CACHE is not None:
        return _CONTEXT_CACHE

    ctx: dict[str, Any] = {}

    # Grammar rules
    ctx["canonical_pattern"] = _get_canonical_pattern()
    ctx["segment_order"] = _get_segment_order()
    ctx["template_rules"] = _get_template_rules()
    ctx["exclusive_pairs"] = _get_exclusive_pairs()

    # Vocabulary with descriptions
    ctx["vocabulary_sections"] = _build_vocabulary_sections()

    # Segment descriptions and usage guidance
    ctx["segment_descriptions"] = _get_all_segment_descriptions()

    # Field guidance for documentation generation
    ctx["field_guidance"] = _get_field_guidance()

    # Curated examples
    ctx["examples"] = _load_curated_examples()

    # Tokamak parameter ranges for documentation grounding
    ctx["tokamak_ranges"] = _load_tokamak_ranges()

    # Bare enum lists (backward compat for user prompt)
    ctx.update(_build_enum_lists())

    _CONTEXT_CACHE = ctx
    return ctx


def clear_context_cache() -> None:
    """Clear cached context (for testing)."""
    global _CONTEXT_CACHE
    _CONTEXT_CACHE = None


# ---------------------------------------------------------------------------
# Grammar rules
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_canonical_pattern() -> str:
    from imas_standard_names.tools.grammar import _build_canonical_pattern

    return _build_canonical_pattern()


@lru_cache(maxsize=1)
def _get_segment_order() -> str:
    from imas_standard_names.tools.grammar import _build_segment_order_constraint

    return _build_segment_order_constraint()


@lru_cache(maxsize=1)
def _get_template_rules() -> str:
    from imas_standard_names.tools.grammar import _build_template_application_rule

    return _build_template_application_rule()


@lru_cache(maxsize=1)
def _get_exclusive_pairs() -> list[tuple[str, str]]:
    from imas_standard_names.grammar.constants import EXCLUSIVE_SEGMENT_PAIRS

    return list(EXCLUSIVE_SEGMENT_PAIRS)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def _build_vocabulary_sections() -> list[dict[str, Any]]:
    """Build per-segment vocabulary sections with tokens and descriptions."""
    from imas_standard_names.grammar.constants import SEGMENT_RULES
    from imas_standard_names.tools.grammar import _get_vocabulary_description

    sections = []
    for rule in SEGMENT_RULES:
        seg_id = rule.identifier
        desc = _get_vocabulary_description(seg_id)
        tokens = list(rule.tokens) if rule.tokens else []
        template = rule.template

        sections.append(
            {
                "segment": seg_id,
                "description": desc,
                "tokens": tokens,
                "template": template,
                "is_open": seg_id == "physical_base",
                "exclusive_with": list(rule.exclusive_with)
                if rule.exclusive_with
                else [],
            }
        )
    return sections


# ---------------------------------------------------------------------------
# Segment descriptions
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_all_segment_descriptions() -> dict[str, str]:
    from imas_standard_names.tools.grammar import _get_segment_descriptions

    return _get_segment_descriptions()


# ---------------------------------------------------------------------------
# Field guidance
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_field_guidance() -> dict[str, Any]:
    from imas_standard_names.grammar.field_schemas import (
        FIELD_GUIDANCE,
        TYPE_SPECIFIC_REQUIREMENTS,
    )

    return {
        "fields": dict(FIELD_GUIDANCE),
        "type_requirements": dict(TYPE_SPECIFIC_REQUIREMENTS),
    }


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

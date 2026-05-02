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
    """Build rich context dict for sn/generate_name_system.md template.

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

    # W2: full closed-vocabulary token map (per-segment) — injected verbatim
    # into the prompt so the LLM never has to guess whether a token is a
    # closed-vocab member.  This is the primary defence against decomposition
    # failures (closed tokens absorbed into physical_base).
    ctx["closed_vocab_full"] = _load_closed_vocab_full()

    # W2: curated examples + anti-patterns from the W0 snapshot YAMLs.  These
    # are static, cacheable, and survive `sn clear`; the graph-driven
    # `compose_scored_examples` injection still complements them at runtime
    # once the graph repopulates.
    ctx["w0_curated_examples"] = _load_w0_curated_examples()
    ctx["decomposition_anti_patterns"] = _load_decomposition_anti_patterns()

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
# W2: Full closed-vocabulary injection
# ---------------------------------------------------------------------------

# Aliased segments in the ISN SEGMENT_TOKEN_MAP that share an identical token
# list — emit only the canonical name to avoid duplicating ~400 tokens in the
# rendered prompt (and to keep the cached system prompt deterministic).
_SEGMENT_ALIASES: dict[str, str] = {
    # alias -> canonical
    "coordinate": "component",
    "object": "device",
    "position": "geometry",
}


@lru_cache(maxsize=1)
def _load_closed_vocab_full() -> list[dict[str, Any]]:
    """Return every closed-vocabulary segment with its FULL token list.

    The returned structure is a list of dicts ordered for stable, cache-friendly
    rendering::

        [
          {"segment": "component", "aliases": ["coordinate"],
           "tokens": ["binormal", "normal", ..., "z"]},
          ...
        ]

    Tokens within a segment are sorted alphabetically.  Open segments
    (``physical_base`` and any other segment with an empty token list) are
    omitted because their content is by-design free-form — listing them
    would mislead the LLM into treating them as closed.

    The data source is :data:`imas_standard_names.grammar.constants.SEGMENT_TOKEN_MAP`
    which is the single source of truth used by the parser, the
    ``is_known_token`` primitive, and the decomposition audit.  When the
    package is unavailable an empty list is returned so prompt rendering
    degrades gracefully rather than raising.
    """
    try:
        from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP
    except ImportError:
        logger.warning("imas_standard_names not available — closed_vocab_full empty")
        return []

    # Group aliased segments under their canonical name.
    canonical_to_aliases: dict[str, list[str]] = {}
    for segment in SEGMENT_TOKEN_MAP:
        canonical = _SEGMENT_ALIASES.get(segment, segment)
        if canonical == segment:
            canonical_to_aliases.setdefault(canonical, [])
        else:
            canonical_to_aliases.setdefault(canonical, []).append(segment)

    out: list[dict[str, Any]] = []
    for segment in sorted(canonical_to_aliases):
        tokens = SEGMENT_TOKEN_MAP.get(segment) or ()
        if not tokens:
            continue  # skip open segments
        out.append(
            {
                "segment": segment,
                "aliases": sorted(canonical_to_aliases[segment]),
                "tokens": sorted(tokens),
            }
        )
    return out


# ---------------------------------------------------------------------------
# W2: W0 snapshot — curated examples + decomposition anti-patterns
# ---------------------------------------------------------------------------


def _w0_examples_path() -> Path:
    return Path(__file__).parent / "examples_curated.yaml"


def _anti_patterns_path() -> Path:
    return Path(__file__).parent / "anti_patterns.yaml"


@lru_cache(maxsize=1)
def _load_w0_curated_examples() -> dict[str, list[dict[str, Any]]]:
    """Load the W0 snapshot ``examples_curated.yaml`` for prompt injection.

    Returns a dict keyed by tier — ``outstanding``, ``good``, ``adequate``,
    ``inadequate``, ``poor`` — each mapping to a list of example entries with
    ``id``, ``description``, ``documentation``, ``reviewer_comments_name``,
    ``grammar_decomposition``, etc.  When the YAML file is absent or
    malformed an empty dict is returned.

    The compose system prompt template selects the strongest entries
    (top of ``outstanding`` and ``good``) for the cacheable
    "EXEMPLAR DECOMPOSITIONS" section.
    """
    path = _w0_examples_path()
    if not path.exists():
        logger.warning("W0 curated examples not found at %s", path)
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return {}
        return {
            tier: [e for e in (entries or []) if isinstance(e, dict)]
            for tier, entries in data.items()
            if isinstance(entries, list)
        }
    except Exception:
        logger.exception("Failed to load W0 curated examples from %s", path)
        return {}


@lru_cache(maxsize=1)
def _load_decomposition_anti_patterns() -> list[dict[str, Any]]:
    """Load ``anti_patterns.yaml`` — curated decomposition-failure exemplars.

    Each entry contains ``bad_name``, ``issue_category``, ``reviewer_comment``,
    ``absorbed_tokens``, ``correct_decomposition``, and ``rewritten_name``.
    See ``imas_codex/standard_names/anti_patterns.yaml`` for the schema and
    ``tests/standard_names/test_anti_patterns_yaml.py`` for the validator.
    """
    path = _anti_patterns_path()
    if not path.exists():
        logger.warning("Decomposition anti-patterns YAML missing at %s", path)
        return []
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            return []
        return [e for e in data if isinstance(e, dict) and e.get("bad_name")]
    except Exception:
        logger.exception("Failed to load decomposition anti-patterns from %s", path)
        return []


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


# ---------------------------------------------------------------------------
# Reviewer neighbourhood (third-party-critic context)
# ---------------------------------------------------------------------------


def _path_ids_prefix(path: str) -> str | None:
    """Extract the leading IDS segment from a DD path.

    ``equilibrium/time_slice/0/global_quantities/ip`` -> ``equilibrium``.
    Returns ``None`` for empty input.
    """
    if not path:
        return None
    head = path.split("/", 1)[0]
    return head or None


def fetch_review_neighbours(
    sn: dict[str, Any],
    *,
    gc: Any = None,
    n_vector: int = 5,
    n_same_base: int = 3,
    n_same_path: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    """Fetch nearest-neighbour SNs to inject into reviewer prompts.

    Returns three lists used as third-party comparators by the reviewer:

    * ``vector_neighbours`` — up to ``n_vector`` accepted SNs nearest to the
      candidate description by embedding similarity (vector index lookup).
    * ``same_base_neighbours`` — up to ``n_same_base`` accepted SNs sharing
      the candidate's ``physical_base`` token (sibling-by-base comparator).
    * ``same_path_neighbours`` — up to ``n_same_path`` accepted SNs whose
      ``source_paths`` share the candidate's leading IDS prefix.

    All result lists exclude the candidate itself by ``id``. On any failure
    (no graph client, missing index, etc.) the corresponding list is empty
    and the function logs at DEBUG level — never raises.

    Each entry dict contains:
        ``id, name, description, kind, unit, score`` (score only for vector).

    The candidate ``sn`` dict must contain at least ``id``; ``description``
    drives the vector lookup, ``physical_base`` the same-base lookup, and
    ``source_paths`` the same-path lookup.
    """
    sn_id = sn.get("id") or sn.get("name") or ""
    desc = sn.get("description") or sn.get("name") or sn.get("id") or ""
    physical_base = sn.get("physical_base") or (
        (sn.get("grammar_fields") or {}).get("physical_base")
    )
    source_paths = sn.get("source_paths") or []
    ids_prefix = next(
        (p for p in (_path_ids_prefix(sp) for sp in source_paths) if p), None
    )

    out: dict[str, list[dict[str, Any]]] = {
        "vector_neighbours": [],
        "same_base_neighbours": [],
        "same_path_neighbours": [],
    }

    own_gc = False
    _gc_ctx: Any = None
    if gc is None:
        try:
            from imas_codex.graph.client import GraphClient

            _gc_ctx = GraphClient()
            gc = _gc_ctx.__enter__() if hasattr(_gc_ctx, "__enter__") else _gc_ctx
            own_gc = True
        except Exception:
            logger.debug("fetch_review_neighbours: GraphClient unavailable")
            return out

    try:
        # --- Vector nearest --------------------------------------------------
        if desc:
            try:
                from imas_codex.standard_names.search import (
                    search_standard_names_vector,
                )

                rows = search_standard_names_vector(
                    desc, k=n_vector + 1, gc=gc, include_superseded=False
                )
                out["vector_neighbours"] = [r for r in rows if r.get("id") != sn_id][
                    :n_vector
                ]
            except Exception:
                logger.debug("fetch_review_neighbours: vector lookup failed")

        # --- Same physical_base ---------------------------------------------
        if physical_base:
            try:
                rows = (
                    gc.query(
                        """
                        MATCH (sn:StandardName)
                        WHERE sn.physical_base = $base
                          AND sn.id <> $sn_id
                          AND coalesce(sn.validation_status, '') <> 'quarantined'
                          AND coalesce(sn.name_stage, '') = 'accepted'
                        OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
                        RETURN sn.id AS id,
                               sn.name AS name,
                               sn.description AS description,
                               sn.kind AS kind,
                               coalesce(u.id, sn.unit) AS unit
                        ORDER BY sn.id
                        LIMIT $k
                        """,
                        base=physical_base,
                        sn_id=sn_id,
                        k=n_same_base,
                    )
                    or []
                )
                out["same_base_neighbours"] = [dict(r) for r in rows]
            except Exception:
                logger.debug("fetch_review_neighbours: same-base lookup failed")

        # --- Same DD IDS prefix ---------------------------------------------
        if ids_prefix:
            try:
                rows = (
                    gc.query(
                        """
                        MATCH (sn:StandardName)
                        WHERE sn.id <> $sn_id
                          AND coalesce(sn.validation_status, '') <> 'quarantined'
                          AND coalesce(sn.name_stage, '') = 'accepted'
                          AND ANY(p IN sn.source_paths WHERE p STARTS WITH $prefix)
                        OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
                        RETURN sn.id AS id,
                               sn.name AS name,
                               sn.description AS description,
                               sn.kind AS kind,
                               coalesce(u.id, sn.unit) AS unit
                        ORDER BY sn.id
                        LIMIT $k
                        """,
                        sn_id=sn_id,
                        prefix=ids_prefix + "/",
                        k=n_same_path,
                    )
                    or []
                )
                out["same_path_neighbours"] = [dict(r) for r in rows]
            except Exception:
                logger.debug("fetch_review_neighbours: same-path lookup failed")

        return out
    finally:
        if own_gc and _gc_ctx is not None:
            try:
                if hasattr(_gc_ctx, "__exit__"):
                    _gc_ctx.__exit__(None, None, None)
                else:
                    gc.close()
            except Exception:
                pass

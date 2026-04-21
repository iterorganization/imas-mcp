"""Preferred ``physical_base`` vocabulary for standard name composition.

The ISN grammar defines ``physical_base`` as an OPEN vocabulary: any
snake_case compound is grammatically admissible. This is intentional
(new physics quantities must be representable), but it creates a
downstream problem: two grammatically-valid forms for the same concept
can co-exist (e.g. ``plasma_boundary_gap_angle`` vs
``angle_of_plasma_boundary_gap``).

This module loads a curated list of *preferred* ``physical_base``
anchors — tokens already used by ≥ 2 high-quality StandardNames in
the graph. The composer and reviewer prompts use this list to resolve
ordering ambiguities without closing the open segment.

Usage
-----

>>> from imas_codex.standard_names.preferred_bases import (
...     load_preferred_bases, suggest_anchor,
... )
>>> bases = load_preferred_bases()
>>> suggest_anchor("plasma_boundary_gap_angle")
AnchorSuggestion(anchor='angle', prefix='plasma_boundary_gap',
                 suggested='angle_of_plasma_boundary_gap',
                 original='plasma_boundary_gap_angle')
"""

from __future__ import annotations

import importlib.resources
import logging
from dataclasses import dataclass
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Package resource reference. Kept as a module-level constant so tests
# can monkey-patch it without touching the filesystem.
_RESOURCE_PACKAGE = "imas_codex.llm.config"
_RESOURCE_NAME = "preferred_physical_bases.yaml"

_CACHE: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_preferred_bases() -> dict[str, Any]:
    """Load the preferred ``physical_base`` anchor YAML (cached).

    Returns a dict with keys: ``version``, ``last_updated``,
    ``selection``, ``anchors``. Each anchor is a dict with
    ``token``, ``domain``, ``usage_count``, ``examples`` (and optional
    ``note``).
    """
    global _CACHE  # noqa: PLW0603
    if _CACHE is not None:
        return _CACHE

    try:
        ref = importlib.resources.files(_RESOURCE_PACKAGE) / _RESOURCE_NAME
        data = yaml.safe_load(ref.read_text()) or {}
    except Exception:
        logger.debug("Failed to load preferred_physical_bases.yaml", exc_info=True)
        data = {}

    data.setdefault("anchors", [])
    data.setdefault("version", 0)
    _CACHE = data
    return data


def clear_cache() -> None:
    """Clear cached preferred-bases data (for testing)."""
    global _CACHE  # noqa: PLW0603
    _CACHE = None


def get_anchor_tokens() -> list[str]:
    """Return the flat list of preferred anchor tokens."""
    return [
        a["token"]
        for a in load_preferred_bases().get("anchors", [])
        if isinstance(a, dict) and a.get("token")
    ]


def get_anchor_set() -> frozenset[str]:
    """Return anchor tokens as a frozenset (cached per call)."""
    return frozenset(get_anchor_tokens())


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def get_preferred_anchors_for_prompt() -> list[dict[str, Any]]:
    """Return anchors formatted for prompt rendering.

    Each entry has: ``token``, ``domain``, ``examples`` (up to 2),
    and ``note`` when present. Designed to be iterated in Jinja.
    """
    rendered: list[dict[str, Any]] = []
    for a in load_preferred_bases().get("anchors", []):
        if not isinstance(a, dict) or not a.get("token"):
            continue
        rendered.append(
            {
                "token": a["token"],
                "domain": a.get("domain", ""),
                "examples": list(a.get("examples") or [])[:2],
                "note": a.get("note", ""),
            }
        )
    return rendered


# ---------------------------------------------------------------------------
# Soft suggestion helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorSuggestion:
    """A soft suggestion to rewrite a physical_base around a preferred anchor."""

    anchor: str
    prefix: str
    suggested: str
    original: str


def suggest_anchor(
    physical_base: str,
    *,
    anchors: frozenset[str] | None = None,
) -> AnchorSuggestion | None:
    """Return a soft suggestion when ``physical_base`` ends with a preferred anchor.

    Example: ``plasma_boundary_gap_angle`` → suggest using anchor ``angle``
    with form ``angle_of_plasma_boundary_gap``.

    Returns ``None`` when:
    - The base is itself already an anchor (``major_radius``).
    - No anchor matches as a trailing ``_<anchor>`` suffix.
    - The base already starts with an anchor (``angle_of_*`` is fine).

    The check is intentionally conservative — this is a soft suggestion,
    not an error. Callers may surface it as a lint hint.
    """
    base = (physical_base or "").strip()
    if not base:
        return None

    anchor_set = anchors if anchors is not None else get_anchor_set()
    if not anchor_set:
        return None

    # If the base already IS an anchor, nothing to suggest.
    if base in anchor_set:
        return None

    # If the base already starts with an anchor (anchor-first form like
    # ``angle_of_plasma_boundary_gap``), the composer already did the
    # right thing.
    for anchor in anchor_set:
        if base == anchor or base.startswith(anchor + "_"):
            return None

    # Look for trailing ``_<anchor>`` (suffix form). Prefer the LONGEST
    # matching anchor so ``number_density`` wins over ``density`` when
    # both are anchors.
    matches: list[str] = []
    for anchor in anchor_set:
        if base.endswith("_" + anchor):
            matches.append(anchor)

    if not matches:
        return None

    anchor = max(matches, key=len)
    prefix = base[: -(len(anchor) + 1)]
    if not prefix:
        return None

    suggested = f"{anchor}_of_{prefix}"
    return AnchorSuggestion(
        anchor=anchor,
        prefix=prefix,
        suggested=suggested,
        original=base,
    )


# ---------------------------------------------------------------------------
# Graph mining (for regeneration)
# ---------------------------------------------------------------------------


def mine_preferred_bases(
    *,
    min_usage_count: int = 2,
    min_review_mean_score: float = 0.75,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Mine preferred physical_base anchors from the Neo4j graph.

    Selects ``grammar_physical_base`` tokens that have been used by at
    least ``min_usage_count`` distinct StandardNames whose individual
    ``review_mean_score`` is ≥ ``min_review_mean_score``. For each
    qualifying token, picks the most common ``physics_domain`` as the
    primary domain and collects up to three example names.

    Args:
        min_usage_count: Minimum distinct StandardName usages required.
        min_review_mean_score: Minimum per-name review mean score required.
        limit: Optional cap on the number of returned anchors.

    Returns:
        List of anchor dicts sorted by ``usage_count`` desc, then
        ``token`` asc. Each dict has ``token``, ``domain``, ``usage_count``,
        ``examples``.

    Raises:
        Exception when the graph is unreachable. The CLI catches this.
    """
    from imas_codex.graph.client import GraphClient

    cypher = """
        MATCH (sn:StandardName)
        WHERE sn.grammar_physical_base IS NOT NULL
          AND sn.review_mean_score IS NOT NULL
          AND sn.review_mean_score >= $min_score
        WITH sn.grammar_physical_base AS token,
             sn.physics_domain        AS domain,
             sn.id                    AS name
        WITH token, domain, collect(DISTINCT name) AS names
        WITH token,
             collect({domain: domain, names: names}) AS per_domain,
             sum(size(names)) AS usage_count
        WHERE usage_count >= $min_usage
        RETURN token, usage_count, per_domain
        ORDER BY usage_count DESC, token ASC
    """

    with GraphClient() as gc:
        rows = gc.query(
            cypher,
            min_score=float(min_review_mean_score),
            min_usage=int(min_usage_count),
        )

    out: list[dict[str, Any]] = []
    for r in rows:
        per_domain = sorted(
            (d for d in (r.get("per_domain") or [])),
            key=lambda d: -len(d.get("names") or []),
        )
        primary = (per_domain[0]["domain"] if per_domain else "") or "general"
        examples: list[str] = []
        for d in per_domain:
            for n in d.get("names") or []:
                if n not in examples:
                    examples.append(n)
                if len(examples) >= 3:
                    break
            if len(examples) >= 3:
                break
        out.append(
            {
                "token": r["token"],
                "domain": primary,
                "usage_count": int(r["usage_count"]),
                "examples": examples,
            }
        )
        if limit is not None and len(out) >= limit:
            break

    return out


def render_yaml(anchors: list[dict[str, Any]], *, last_updated: str) -> str:
    """Render a list of mined anchors to the canonical YAML layout.

    This produces the same file format as the committed
    ``preferred_physical_bases.yaml`` so ``sn anchors mine`` can print
    diffable output.
    """
    header = (
        "# Curated preferred physical_base anchors for standard name generation.\n"
        "#\n"
        "# These tokens resolve ordering ambiguities in the open-vocabulary\n"
        "# ``physical_base`` segment. When two grammatically-valid forms of a\n"
        "# name compete (e.g. ``plasma_boundary_gap_angle`` vs\n"
        "# ``angle_of_plasma_boundary_gap``), the one whose physical_base is\n"
        "# on this anchor list wins.\n"
        "#\n"
        "# IMPORTANT: This is **NOT** a closed list. ``physical_base`` remains\n"
        "# an open vocabulary — new compound tokens may still be coined freely.\n"
        "#\n"
        "# Regenerate with: ``uv run imas-codex sn anchors mine``\n"
    )
    lines = [
        header,
        "version: 1",
        f'last_updated: "{last_updated}"',
        "selection:",
        "  min_usage_count: 2",
        "  min_review_mean_score: 0.75",
        "anchors:",
    ]
    for a in anchors:
        lines.append(f"  - token: {a['token']}")
        lines.append(f"    domain: {a.get('domain', 'general')}")
        lines.append(f"    usage_count: {a.get('usage_count', 0)}")
        examples = list(a.get("examples") or [])[:3]
        if examples:
            lines.append("    examples:")
            for ex in examples:
                lines.append(f"      - {ex}")
        if a.get("note"):
            lines.append(f"    note: {a['note']!r}")
    return "\n".join(lines) + "\n"

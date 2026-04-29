"""Detect and promote saturated grammar tokens to ISN.

This module turns codex into a *miner* of grammar-vocabulary candidates — it
does not store the vocabulary itself. The workflow is:

1.  :func:`mine_promotion_candidates` scans the Neo4j graph for segment
    tokens reused across multiple high-quality ``StandardName`` nodes.
2.  :func:`persist_candidates` writes a ``PromotionCandidate`` node per
    saturated token plus ``EVIDENCED_BY`` edges to each supporting name.
3.  :func:`format_isn_pr_snippet` renders a PR-ready snippet matching the
    layout of ISN's ``grammar/vocabularies/*.yml`` files so a human can
    open a PR against :mod:`imas_standard_names`.

Once the ISN PR lands, codex picks up the new tokens automatically on the
next ISN dependency bump via ``get_grammar_context()`` — codex itself owns
no vocabulary data.

Thresholds default to:
    * ``min_usage_count = 3`` distinct StandardNames (independent triangulation)
    * ``min_review_mean_score = 0.75`` on every supporting name (quality gate)

See ``plans/research/standard-names/47-vocab-promotion.md`` for the
rationale and lifecycle diagram.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Existing-vocabulary lookup (ISN is the source of truth)
# ---------------------------------------------------------------------------

# Map from codex-side segment name to the ISN vocabulary_sections segment
# name. For physical_base the ISN vocabulary is *open* (no authoritative
# token list), but ``generic_physical_bases`` and ``geometric_bases`` both
# inform which tokens are already formally known.
_SEGMENT_VOCAB_CACHE: dict[str, set[str]] | None = None


def _load_isn_segment_vocab() -> dict[str, set[str]]:
    """Return {segment: {known_tokens}} from the installed ISN package.

    For ``physical_base``, this merges ``generic_physical_bases`` (tokens
    the grammar already calls out as needing qualification) with
    ``geometric_bases``; any additional tokens seen in the graph beyond
    these are "emergent" and candidates for promotion.
    """
    global _SEGMENT_VOCAB_CACHE
    if _SEGMENT_VOCAB_CACHE is not None:
        return _SEGMENT_VOCAB_CACHE

    from imas_standard_names.grammar import get_grammar_context

    ctx = get_grammar_context()
    by_segment: dict[str, set[str]] = {}
    for section in ctx.get("vocabulary_sections", []) or []:
        seg = section.get("segment")
        tokens = section.get("tokens") or []
        if seg:
            by_segment.setdefault(seg, set()).update(tokens)

    # Augment ``physical_base`` with already-formalised base lists. The ISN
    # ``vocabulary_sections`` entry for physical_base, if present, is usually
    # empty because the segment is open; we use the supporting YAML lists
    # to detect tokens that are already on ISN's radar.
    try:
        import importlib.resources as ir

        pkg = ir.files("imas_standard_names") / "grammar" / "vocabularies"
        for fname in ("generic_physical_bases.yml", "geometric_bases.yml"):
            fpath = pkg / fname
            if fpath.is_file():
                import yaml

                data = yaml.safe_load(fpath.read_text()) or []
                if isinstance(data, list):
                    by_segment.setdefault("physical_base", set()).update(
                        str(t) for t in data
                    )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not augment physical_base vocab from ISN: %s", exc)

    _SEGMENT_VOCAB_CACHE = by_segment
    return by_segment


def _existing_tokens(segment: str) -> set[str]:
    return _load_isn_segment_vocab().get(segment, set())


# ---------------------------------------------------------------------------
# Graph mining
# ---------------------------------------------------------------------------

# Segment → name of the StandardName property that stores its token.
_SEGMENT_TO_SN_PROPERTY = {
    "physical_base": "grammar_physical_base",
    "geometric_base": "grammar_geometric_base",
    "subject": "grammar_subject",
    "component": "grammar_component",
    "position": "grammar_position",
    "transformation": "grammar_transformation",
    "process": "grammar_process",
}


def mine_promotion_candidates(
    segment: str = "physical_base",
    min_usage_count: int = 3,
    min_review_mean_score: float = DEFAULT_MIN_SCORE,
    exclude_existing: bool = True,
) -> list[dict[str, Any]]:
    """Query the graph for saturated grammar tokens.

    A candidate is emitted iff:
    * the token appears as a single-token value in the given grammar
      segment slot on at least ``min_usage_count`` distinct StandardNames,
    * every supporting name has ``review_mean_score >= min_review_mean_score``,
    * and (when ``exclude_existing``) the token is not already present in
      ISN's vocabulary for that segment.

    Returns a list of dicts, one per candidate, with keys:
        ``token``, ``segment``, ``uses``, ``min_review_score``,
        ``physics_domains``, ``supporting_names``.
    """
    if segment not in _SEGMENT_TO_SN_PROPERTY:
        raise ValueError(
            f"Unknown segment {segment!r}. Known: {sorted(_SEGMENT_TO_SN_PROPERTY)}"
        )
    prop = _SEGMENT_TO_SN_PROPERTY[segment]

    # NOTE: the quality gate (review_mean_score >= threshold) is applied
    # per-name inside the MATCH; a token with any low-scoring supporter is
    # not promoted. This enforces the "all supporters ≥ 0.75" contract.
    cypher = f"""
        MATCH (sn:StandardName)
        WHERE sn.`{prop}` IS NOT NULL
          AND sn.review_mean_score IS NOT NULL
          AND sn.review_mean_score >= $min_score
        WITH sn.`{prop}` AS token,
             sn.id AS name,
             sn.review_mean_score AS score,
             CASE WHEN sn.physics_domain IS NOT NULL AND size(sn.physics_domain) > 0
                  THEN sn.physics_domain
                  ELSE ['unknown'] END AS pd_list
        UNWIND pd_list AS domain
        WITH token, name, score, domain
        WITH token,
             collect(DISTINCT name) AS names,
             min(score) AS min_score,
             collect(DISTINCT domain) AS domains
        WHERE size(names) >= $min_usage
        RETURN token,
               size(names) AS uses,
               min_score,
               domains,
               names
        ORDER BY uses DESC, token ASC
    """
    with GraphClient() as gc:
        rows = gc.query(
            cypher,
            min_score=float(min_review_mean_score),
            min_usage=int(min_usage_count),
        )

    known = _existing_tokens(segment) if exclude_existing else set()

    candidates: list[dict[str, Any]] = []
    for r in rows:
        token = r["token"]
        if exclude_existing and token in known:
            continue
        candidates.append(
            {
                "token": token,
                "segment": segment,
                "uses": int(r["uses"]),
                "min_review_score": float(r["min_score"]),
                "physics_domains": sorted(r["domains"] or []),
                "supporting_names": sorted(r["names"] or []),
            }
        )
    return candidates


# ---------------------------------------------------------------------------
# Graph persistence
# ---------------------------------------------------------------------------


def persist_candidates(candidates: list[dict[str, Any]]) -> int:
    """Write ``PromotionCandidate`` nodes + ``EVIDENCED_BY`` edges.

    Idempotent — MERGEs by ``id = promotion_candidate:{segment}:{token}``
    and refreshes ``uses``, ``min_review_score``, ``physics_domains`` on
    every call. ``submitted_to_isn_at`` is never overwritten here; it
    is set separately when a PR is opened against ISN.

    Returns the number of candidate nodes written.
    """
    if not candidates:
        return 0

    now = datetime.now(UTC).isoformat()

    node_batch = [
        {
            "id": f"promotion_candidate:{c['segment']}:{c['token']}",
            "segment": c["segment"],
            "token": c["token"],
            "uses": int(c["uses"]),
            "min_review_score": float(c["min_review_score"]),
            "physics_domains": ",".join(c.get("physics_domains") or []),
            "detected_at": now,
        }
        for c in candidates
    ]
    edge_batch: list[dict[str, str]] = []
    for c in candidates:
        cid = f"promotion_candidate:{c['segment']}:{c['token']}"
        for sn_id in c.get("supporting_names") or []:
            edge_batch.append({"candidate_id": cid, "sn_id": sn_id})

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (pc:PromotionCandidate {id: b.id})
            SET pc.segment = b.segment,
                pc.token = b.token,
                pc.uses = b.uses,
                pc.min_review_score = b.min_review_score,
                pc.physics_domains = b.physics_domains,
                pc.detected_at = coalesce(pc.detected_at, datetime(b.detected_at)),
                pc.last_detected_at = datetime(b.detected_at)
            """,
            batch=node_batch,
        )
        if edge_batch:
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (pc:PromotionCandidate {id: b.candidate_id})
                MATCH (sn:StandardName {id: b.sn_id})
                MERGE (pc)-[:EVIDENCED_BY]->(sn)
                """,
                batch=edge_batch,
            )

    logger.info(
        "Persisted %d PromotionCandidate nodes (%d evidence edges)",
        len(node_batch),
        len(edge_batch),
    )
    return len(node_batch)


# ---------------------------------------------------------------------------
# ISN-PR snippet formatter
# ---------------------------------------------------------------------------


def format_isn_pr_snippet(
    candidates: list[dict[str, Any]],
    segment: str = "physical_base",
) -> str:
    """Render a PR-ready snippet for ISN's ``grammar/vocabularies/`` files.

    The ``physical_base`` vocabulary is open in ISN, but promoted tokens
    would be added to a curated list (e.g. ``preferred_physical_bases.yml``
    — to be created on the ISN side). Format mirrors the existing
    vocabulary files: a header comment block explaining the source of the
    list, followed by a flat ``- token`` list. We also emit a short
    evidence table in a separate comment block so ISN maintainers can
    audit each entry.
    """
    if not candidates:
        return (
            f"# No promotion candidates for segment `{segment}` at the "
            "current thresholds.\n"
        )

    lines: list[str] = [
        f"# Promotion candidates for ISN segment `{segment}`",
        "#",
        "# Mined from the imas-codex StandardName graph. Each token below",
        "# appears as the single-token `{segment}` slot on {N}+ independent",
        "# StandardNames, all of which carry review_mean_score >= {S:.2f}.",
        "#",
        "# Codex does not own vocabulary; this snippet is intended to be",
        "# copied into the appropriate ISN file (e.g.",
        f"# `imas_standard_names/grammar/vocabularies/{segment}s.yml` or a",
        "# new `preferred_physical_bases.yml`) via a PR on imas-standard-names.",
        "#",
        "# Evidence table (token | uses | min score | domains):",
    ]
    # fill in dynamic summary values
    if candidates:
        min_uses = min(c["uses"] for c in candidates)
        min_score = min(c["min_review_score"] for c in candidates)
        lines[3] = (
            f"# appears as the single-token `{segment}` slot on {min_uses}+ independent"
        )
        lines[4] = (
            f"# StandardNames, all of which carry review_mean_score >= {min_score:.2f}."
        )

    for c in candidates:
        domains = ", ".join(c.get("physics_domains") or []) or "-"
        lines.append(
            f"#   {c['token']:<40s} "
            f"{c['uses']:>3d} uses  "
            f"{c['min_review_score']:.2f}  "
            f"[{domains}]"
        )

    lines += [
        "#",
        "# Example supporting StandardNames per token (first 3):",
    ]
    for c in candidates:
        examples = (c.get("supporting_names") or [])[:3]
        if examples:
            lines.append(f"#   {c['token']}: {', '.join(examples)}")

    lines += ["", "# ---- YAML list (append to target file) ----"]
    for c in candidates:
        lines.append(f"- {c['token']}")

    return "\n".join(lines) + "\n"

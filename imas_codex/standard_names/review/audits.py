"""Layer 1: Deterministic audits for standard-name review.

Provides embedding preflight, lexical lint, link integrity checks,
and near-duplicate detection — all runnable without LLM access.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic report models
# ---------------------------------------------------------------------------


class EmbeddingReport(BaseModel):
    """Result of embedding preflight — freshness check and re-embed."""

    total: int = 0
    missing_count: int = 0
    stale_count: int = 0
    refreshed_count: int = 0
    missing_ids: list[str] = Field(default_factory=list)
    stale_ids: list[str] = Field(default_factory=list)


class LintFinding(BaseModel):
    """Single lexical-lint finding."""

    name_id: str
    finding_type: str  # "round_trip_failure", "vocab_gap", "convention_violation"
    detail: str
    severity: str = "warning"  # "warning" or "error"


class LinkFinding(BaseModel):
    """Single link-integrity finding."""

    name_id: str
    finding_type: str  # "unresolved", "dead_link", "missing_reverse"
    target: str
    detail: str


class DuplicateComponent(BaseModel):
    """A connected component of near-duplicate names."""

    names: list[str]
    max_similarity: float
    pairs: list[tuple[str, str, float]] = Field(default_factory=list)


class AuditReport(BaseModel):
    """Combined output of all Layer 1 audits."""

    embedding: EmbeddingReport = Field(default_factory=EmbeddingReport)
    lint_findings: list[LintFinding] = Field(default_factory=list)
    link_findings: list[LinkFinding] = Field(default_factory=list)
    duplicate_components: list[DuplicateComponent] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------

_HASH_SCALAR_FIELDS = (
    "id",
    "description",
    "documentation",
    "kind",
    "unit",
)

_HASH_LIST_FIELDS = ("links",)

_HASH_GRAMMAR_FIELDS = (
    "physical_base",
    "subject",
    "component",
    "coordinate",
    "position",
    "process",
)


def compute_review_input_hash(name: dict[str, Any]) -> str:
    """Deterministic SHA-256 fingerprint of review-relevant fields.

    Fields are hashed in a fixed order.  ``None`` / missing values become
    the empty string; lists are sorted and joined with ``|``.
    """
    parts: list[str] = []

    # Scalar fields
    for field in _HASH_SCALAR_FIELDS:
        parts.append(str(name.get(field) or ""))

    # Sorted list fields (links)
    for field in _HASH_LIST_FIELDS:
        val = name.get(field)
        parts.append("|".join(sorted(val)) if val else "")

    # Grammar fields
    for field in _HASH_GRAMMAR_FIELDS:
        parts.append(str(name.get(field) or ""))

    # COCOS transformation type
    parts.append(str(name.get("cocos_transformation_type") or ""))

    # source_paths (sorted)
    source_paths = name.get("source_paths")
    parts.append("|".join(sorted(source_paths)) if source_paths else "")

    payload = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# 1. Embedding preflight
# ---------------------------------------------------------------------------


def run_embedding_preflight(names: list[dict[str, Any]]) -> EmbeddingReport:
    """Check embedding freshness, re-embed missing/stale entries.

    Compares each name's current ``review_input_hash`` against a freshly
    computed one and checks for missing embeddings.  Re-embeds in batch
    and persists updated embeddings + hashes to the graph.
    """
    report = EmbeddingReport(total=len(names))
    if not names:
        return report

    needs_embed: list[dict[str, Any]] = []

    for name in names:
        name_id = name.get("id", "")
        current_hash = compute_review_input_hash(name)
        stored_hash = name.get("review_input_hash")
        embedding = name.get("embedding")

        is_missing = embedding is None or (
            isinstance(embedding, list) and len(embedding) == 0
        )
        is_stale = stored_hash != current_hash

        if is_missing:
            report.missing_count += 1
            report.missing_ids.append(name_id)
            needs_embed.append(
                {
                    "id": name_id,
                    "description": name.get("description"),
                    "_hash": current_hash,
                }
            )
        elif is_stale:
            report.stale_count += 1
            report.stale_ids.append(name_id)
            needs_embed.append(
                {
                    "id": name_id,
                    "description": name.get("description"),
                    "_hash": current_hash,
                }
            )

    if not needs_embed:
        return report

    # Attempt batch re-embedding and graph persistence
    try:
        from imas_codex.embeddings.description import embed_descriptions_batch
        from imas_codex.graph.client import GraphClient

        embed_descriptions_batch(
            needs_embed, text_field="description", embedding_field="embedding"
        )

        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                SET sn.embedding = b.embedding,
                    sn.review_input_hash = b.hash,
                    sn.embedded_at = datetime()
                """,
                batch=[
                    {
                        "id": item["id"],
                        "embedding": item.get("embedding"),
                        "hash": item["_hash"],
                    }
                    for item in needs_embed
                ],
            )

        report.refreshed_count = len(needs_embed)
        logger.info(
            "Embedding preflight: refreshed %d (%d missing, %d stale)",
            report.refreshed_count,
            report.missing_count,
            report.stale_count,
        )
    except Exception:
        logger.warning(
            "Embedding service unavailable — skipping re-embed (%d missing, %d stale)",
            report.missing_count,
            report.stale_count,
            exc_info=True,
        )

    return report


# ---------------------------------------------------------------------------
# 2. Lexical lint
# ---------------------------------------------------------------------------

# Processing verbs that indicate derived / transformed quantities
_PROCESSING_SUFFIXES = (
    "_reconstructed",
    "_measured",
    "_computed",
    "_simulated",
    "_estimated",
    "_interpolated",
    "_extrapolated",
    "_smoothed",
    "_averaged",
    "_filtered",
    "_normalised",
    "_normalized",
)


def run_lexical_lint(names: list[dict[str, Any]]) -> list[LintFinding]:
    """Run deterministic grammar and convention checks.

    Uses ``imas_standard_names`` for parse/compose round-trip when available.
    Falls back to lightweight heuristic checks otherwise.
    """
    findings: list[LintFinding] = []

    # Attempt ISN import for round-trip checks
    isn_available = False
    parse_fn = None
    compose_fn = None
    try:
        from imas_standard_names import compose_standard_name, parse_standard_name

        parse_fn = parse_standard_name
        compose_fn = compose_standard_name
        isn_available = True
    except ImportError:
        logger.debug("imas_standard_names not available — skipping round-trip checks")

    # --- Per-name checks ---
    domain_positions: dict[str, set[str]] = defaultdict(set)

    for name in names:
        name_id = name.get("id", "")

        # 1. Grammar round-trip (ISN only)
        if isn_available and parse_fn and compose_fn:
            try:
                parsed = parse_fn(name_id)
                recomposed = compose_fn(parsed)
                if recomposed != name_id:
                    findings.append(
                        LintFinding(
                            name_id=name_id,
                            finding_type="round_trip_failure",
                            detail=f"Round-trip mismatch: '{name_id}' → parse → compose → '{recomposed}'",
                            severity="error",
                        )
                    )
            except Exception as exc:
                findings.append(
                    LintFinding(
                        name_id=name_id,
                        finding_type="round_trip_failure",
                        detail=f"Parse failed: {exc}",
                        severity="error",
                    )
                )

        # 2. Collect position forms per physics_domain for cross-name analysis
        physics_domain = name.get("physics_domain") or ""
        position = name.get("position") or ""
        if physics_domain and position:
            domain_positions[physics_domain].add(position)

        # 3. Processing verb detection
        for suffix in _PROCESSING_SUFFIXES:
            if name_id.endswith(suffix):
                findings.append(
                    LintFinding(
                        name_id=name_id,
                        finding_type="convention_violation",
                        detail=(
                            f"Name ends with processing verb '{suffix.lstrip('_')}'. "
                            "Consider whether a method-agnostic name is more appropriate."
                        ),
                        severity="warning",
                    )
                )
                break  # One suffix match is enough per name

    # --- Cross-name pattern detection ---
    # Detect mixed position forms within the same domain
    _POSITION_GROUPS = {
        "axis": {"at_magnetic_axis", "on_axis", "magnetic_axis"},
        "boundary": {"at_boundary", "on_boundary", "at_separatrix"},
        "midplane": {
            "at_midplane",
            "midplane_low_field_side",
            "midplane_high_field_side",
        },
    }

    for domain, positions in domain_positions.items():
        for group_name, group_variants in _POSITION_GROUPS.items():
            found = positions & group_variants
            if len(found) > 1:
                findings.append(
                    LintFinding(
                        name_id=f"[domain:{domain}]",
                        finding_type="convention_violation",
                        detail=(
                            f"Mixed {group_name} position forms in domain '{domain}': "
                            f"{sorted(found)}. Consider standardising to one form."
                        ),
                        severity="warning",
                    )
                )

    return findings


# ---------------------------------------------------------------------------
# 3. Link integrity
# ---------------------------------------------------------------------------


def run_link_integrity(names: list[dict[str, Any]]) -> list[LinkFinding]:
    """Check link resolution status and cross-reference consistency.

    Operates purely in-memory from the provided name catalog.
    """
    findings: list[LinkFinding] = []

    # Build lookup of all name IDs
    all_ids = {name.get("id", "") for name in names}

    # Build reverse-link index: target → set of sources that link to it
    reverse_links: dict[str, set[str]] = defaultdict(set)
    for name in names:
        name_id = name.get("id", "")
        for link in name.get("links") or []:
            # Strip known prefixes (name:, dd:, etc.) to get bare target
            target = _strip_link_prefix(link)
            reverse_links[target].add(name_id)

    for name in names:
        name_id = name.get("id", "")
        link_status = name.get("link_status")
        links = name.get("links") or []

        # Check link_status field
        if link_status == "unresolved":
            findings.append(
                LinkFinding(
                    name_id=name_id,
                    finding_type="unresolved",
                    target="",
                    detail="Name has unresolved link status (dd: prefixed links pending resolution).",
                )
            )

        for raw_link in links:
            target = _strip_link_prefix(raw_link)

            # Dead link: target not in catalog
            if target and target not in all_ids:
                findings.append(
                    LinkFinding(
                        name_id=name_id,
                        finding_type="dead_link",
                        target=target,
                        detail=f"Link target '{target}' not found in catalog.",
                    )
                )

            # Missing reverse: A→B exists but B→A does not
            if target and target in all_ids:
                target_linkers = reverse_links.get(name_id, set())
                if target not in target_linkers:
                    findings.append(
                        LinkFinding(
                            name_id=name_id,
                            finding_type="missing_reverse",
                            target=target,
                            detail=(
                                f"'{name_id}' links to '{target}', "
                                f"but '{target}' does not link back."
                            ),
                        )
                    )

    return findings


def _strip_link_prefix(link: str) -> str:
    """Remove known link prefixes (``name:``, ``dd:``, ``http(s)://``) to get bare ID."""
    if link.startswith(("http://", "https://")):
        return ""  # External URLs are not checked for existence
    for prefix in ("name:", "dd:"):
        if link.startswith(prefix):
            return link[len(prefix) :]
    return link


# ---------------------------------------------------------------------------
# 4. Duplicate detection
# ---------------------------------------------------------------------------


class UnionFind:
    """Simple union-find (disjoint set) for connected-component grouping."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py


def _tokenize_snake(name_id: str) -> set[str]:
    """Split a snake_case name into a set of lowercase tokens."""
    return set(name_id.split("_")) - {""}


def _token_overlap(a: set[str], b: set[str]) -> float:
    """Jaccard-like overlap ratio: |intersection| / |union|."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def run_duplicate_detection(names: list[dict[str, Any]]) -> list[DuplicateComponent]:
    """Multi-pass near-duplicate detection using blocking + similarity.

    Uses embedding-based semantic search when available, with lexical
    token-overlap as fallback.
    """
    if len(names) < 2:
        return []

    # --- Build blocking keys ---
    # Each block maps a blocking key → list of name dicts
    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for name in names:
        unit = name.get("unit") or ""
        kind = name.get("kind") or ""

        # Pass 1: (unit, kind, physics_domain)
        physics_domain = name.get("physics_domain") or ""
        if unit or kind or physics_domain:
            key1 = f"ukd:{unit}|{kind}|{physics_domain}"
            blocks[key1].append(name)

        # Pass 2: (unit, kind, physical_base) — when physical_base present
        physical_base = name.get("physical_base")
        if physical_base:
            key2 = f"ukp:{unit}|{kind}|{physical_base}"
            blocks[key2].append(name)

        # Pass 3: (unit, kind, geometric_base) — when present
        geometric_base = name.get("geometric_base")
        if geometric_base:
            key3 = f"ukg:{unit}|{kind}|{geometric_base}"
            blocks[key3].append(name)

        # Fallback: metadata-poor names (all blocking keys None)
        if (
            not unit
            and not kind
            and not physics_domain
            and not physical_base
            and not geometric_base
        ):
            blocks["_fallback"].append(name)

    # --- Semantic search helper (may be unavailable) ---
    semantic_available = False
    search_fn = None
    try:
        from imas_codex.standard_names.search import search_similar_names

        # Quick probe — don't fail if graph is down
        search_fn = search_similar_names
        semantic_available = True
    except ImportError:
        logger.debug(
            "search_similar_names not importable — lexical-only duplicate detection"
        )

    # --- Candidate pair discovery ---
    # pairs: set of (min_id, max_id, similarity)
    candidate_pairs: dict[tuple[str, str], float] = {}

    # Pre-tokenize all names for lexical comparison
    name_tokens: dict[str, set[str]] = {}
    for name in names:
        nid = name.get("id", "")
        name_tokens[nid] = _tokenize_snake(nid)

    # Process each block
    for _block_key, block_names in blocks.items():
        if len(block_names) < 2:
            continue

        block_ids = {n.get("id", "") for n in block_names}

        for name in block_names:
            nid = name.get("id", "")
            desc = name.get("description") or ""

            # Semantic similarity search (within block)
            if semantic_available and search_fn and desc:
                try:
                    results = search_fn(desc, k=5)
                    for r in results:
                        rid = r.get("id", "")
                        score = r.get("score", 0.0)
                        if rid != nid and rid in block_ids and score > 0.92:
                            pair = (min(nid, rid), max(nid, rid))
                            candidate_pairs[pair] = max(
                                candidate_pairs.get(pair, 0.0), score
                            )
                except Exception:
                    logger.debug("Semantic search failed for '%s'", nid, exc_info=True)

            # Lexical token overlap (within block)
            tokens_a = name_tokens.get(nid, set())
            for other in block_names:
                oid = other.get("id", "")
                if oid <= nid:  # avoid self and duplicate pairs
                    continue
                tokens_b = name_tokens.get(oid, set())
                overlap = _token_overlap(tokens_a, tokens_b)
                if overlap > 0.8:
                    pair = (min(nid, oid), max(nid, oid))
                    candidate_pairs[pair] = max(candidate_pairs.get(pair, 0.0), overlap)

    if not candidate_pairs:
        return []

    # --- Union-Find to get connected components ---
    uf = UnionFind()
    for (a, b), _sim in candidate_pairs.items():
        uf.union(a, b)

    # Group by component root
    components: dict[str, list[str]] = defaultdict(list)
    all_involved = {n for pair in candidate_pairs for n in pair}
    for name_id in sorted(all_involved):
        root = uf.find(name_id)
        components[root].append(name_id)

    # Build output
    results: list[DuplicateComponent] = []
    for _root, members in sorted(components.items()):
        if len(members) < 2:
            continue

        # Collect pairs within this component
        member_set = set(members)
        comp_pairs: list[tuple[str, str, float]] = []
        max_sim = 0.0
        for (a, b), sim in candidate_pairs.items():
            if a in member_set and b in member_set:
                comp_pairs.append((a, b, round(sim, 4)))
                max_sim = max(max_sim, sim)

        results.append(
            DuplicateComponent(
                names=sorted(members),
                max_similarity=round(max_sim, 4),
                pairs=comp_pairs,
            )
        )

    return results


# ---------------------------------------------------------------------------
# 5. Orchestrator
# ---------------------------------------------------------------------------


def run_all_audits(names: list[dict[str, Any]]) -> AuditReport:
    """Run all Layer 1 deterministic audits and return a combined report.

    Execution order:
      1. Embedding preflight (ensures fresh embeddings for duplicate detection)
      2. Lexical lint
      3. Link integrity
      4. Duplicate detection
    """
    logger.info("Starting Layer 1 audits on %d standard names", len(names))

    embedding = run_embedding_preflight(names)
    lint_findings = run_lexical_lint(names)
    link_findings = run_link_integrity(names)
    duplicate_components = run_duplicate_detection(names)

    report = AuditReport(
        embedding=embedding,
        lint_findings=lint_findings,
        link_findings=link_findings,
        duplicate_components=duplicate_components,
    )

    logger.info(
        "Layer 1 audits complete: %d lint findings, %d link findings, %d duplicate groups",
        len(lint_findings),
        len(link_findings),
        len(duplicate_components),
    )
    return report

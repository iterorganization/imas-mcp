"""Async workers for the standard-name enrich pipeline.

Five-phase enrich pipeline:

    EXTRACT → CONTEXTUALISE → DOCUMENT → VALIDATE → PERSIST

- **extract**: queries graph for ``pipeline_status='named'`` StandardNames,
  batches them for downstream processing.  Uses claim-token pattern to
  prevent parallel workers from double-processing.
- **contextualise**: gathers DD path descriptions, vector-similar
  neighbours, and domain siblings to build per-item context bundles.
- **document**: LLM call with enrich system/user prompts to generate
  descriptions and documentation.  Preserves DD-authoritative fields
  (unit, physics_domain).  Tracks cost, tokens, and respects budget limits.
- **validate**: (stub) spelling, link integrity checks.  C.4 will implement.
- **persist**: (stub) writes enriched data + REFERENCES rels to graph.
  C.4 will implement.

Workers follow the same async signature and error-handling convention
as ``workers.py`` workers: ``async def worker(state, **_kwargs)``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.claims import retry_on_deadlock

if TYPE_CHECKING:
    from imas_codex.standard_names.enrich_state import StandardNameEnrichState

logger = logging.getLogger(__name__)

# Default batch size for grouping SNs into enrichment batches.
_ENRICH_BATCH_SIZE = 10

# Defense-in-depth sanitizers for LLM enrichment output.
_SN_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_LINK_PREFIXES = ("name:", "http://", "https://")


def _valid_secondary_tags() -> set[str]:
    """Valid SECONDARY tags sourced from ISN grammar context (cached).

    The ISN ``tags`` field accepts **secondary tags only**. Primary tags
    (physics domains like ``edge-physics``, ``transport``) belong in the
    ``physics_domain`` field, not ``tags``.
    """
    global _VALID_TAGS_CACHE
    try:
        return _VALID_TAGS_CACHE  # type: ignore[name-defined]
    except NameError:
        pass
    try:
        from imas_standard_names.grammar import get_grammar_context

        ctx = get_grammar_context()
        td = ctx.get("tag_descriptions", {}) or {}
        _VALID_TAGS_CACHE = frozenset((td.get("secondary") or {}).keys())  # type: ignore[name-defined]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load ISN tag vocab: %s — tag filter disabled", exc)
        _VALID_TAGS_CACHE = frozenset()  # type: ignore[name-defined]
    return _VALID_TAGS_CACHE  # type: ignore[name-defined]


# Back-compat alias (callers may still import _valid_tags)
_valid_tags = _valid_secondary_tags


def _sanitize_tags(raw: list[str] | None) -> list[str]:
    """Filter tags against the ISN SECONDARY vocabulary.

    Primary tags are silently dropped — they are not valid in the ``tags``
    field (ISN uses ``physics_domain`` for that role).
    """
    if not raw:
        return []
    valid = _valid_secondary_tags()
    if not valid:
        return [t for t in raw if isinstance(t, str) and t]
    out: list[str] = []
    for t in raw:
        if not isinstance(t, str):
            continue
        norm = t.strip().lower()
        if norm in valid and norm not in out:
            out.append(norm)
    return out


def _sanitize_links(
    raw: list[str] | None, *, valid_names: set[str] | None = None
) -> list[str]:
    """Coerce LLM link strings to ``name:xxx`` / URL format and drop unknowns.

    If ``valid_names`` is provided, ``name:foo_bar`` links whose target is not
    in that set are silently dropped (prevents ``link_not_found`` quarantine).
    URLs and other prefixes pass through unchanged.
    """
    if not raw:
        return []
    out: list[str] = []
    for link in raw:
        if not isinstance(link, str):
            continue
        s = link.strip()
        if not s:
            continue
        if s.startswith(_LINK_PREFIXES):
            if s.startswith("name:") and valid_names is not None:
                target = s[len("name:") :].strip()
                if target not in valid_names:
                    continue
            out.append(s)
            continue
        for junk in ("dd:", "standard_name:", "sn:", "#"):
            if s.lower().startswith(junk):
                s = s[len(junk) :].strip()
                break
        if _SN_ID_RE.match(s):
            if valid_names is not None and s not in valid_names:
                continue
            candidate = f"name:{s}"
            if candidate not in out:
                out.append(candidate)
    return out


_SIGN_CONV_MENTION_RE = re.compile(r"\bsign\s+convention\b", re.IGNORECASE)
_SIGN_CONV_VALID_RE = re.compile(r"Sign convention:\s+Positive\s+")
_BRACKET_PLACEHOLDER_RE = re.compile(r"\[[a-z][a-z_ ]*\]")


def _sanitize_documentation(doc: str | None) -> str | None:
    """Strip malformed sign-convention sentences from documentation.

    ISN's validator rejects any documentation that mentions "sign convention"
    unless it contains the exact phrase ``Sign convention: Positive when ...``
    or ``Sign convention: Positive <quantity> ...`` (no bracketed placeholders).
    LLMs frequently emit placeholder variants like ``Sign convention: Positive
    when [condition].`` or lowercase/bold variants. Rather than quarantine the
    entire entry, we strip any sentence mentioning "sign convention" that does
    not match the valid form — ISN then accepts the docs without a sign-
    convention statement (which is acceptable for non-COCOS paths and graceful
    for COCOS paths where the LLM failed to produce one).
    """
    if not isinstance(doc, str) or not doc:
        return doc
    if not _SIGN_CONV_MENTION_RE.search(doc):
        return doc
    sentences = re.split(r"(?<=[.!?])\s+", doc)
    cleaned: list[str] = []
    for s in sentences:
        if _SIGN_CONV_MENTION_RE.search(s):
            if not _SIGN_CONV_VALID_RE.search(s):
                continue
            if _BRACKET_PLACEHOLDER_RE.search(s):
                continue
        cleaned.append(s)
    return " ".join(cleaned).strip() or doc


def _is_echoed_name(returned: str | None, expected: str) -> bool:
    """Check the LLM echoed the input name exactly and didn't hallucinate."""
    if not isinstance(returned, str):
        return False
    return returned.strip() == expected


# Claim timeout for enrichment claims (ISO 8601 duration string).
_CLAIM_TIMEOUT = "PT300S"  # 5 minutes


def _fetch_existing_sn_names() -> set[str]:
    """Return the set of all StandardName ids currently in the graph.

    Used by the enrich document worker to validate ``name:xxx`` cross-links
    before they reach the ISN validator (which would quarantine on unknown
    targets).
    """
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            rows = gc.query("MATCH (s:StandardName) RETURN s.id AS id")
            return {r["id"] for r in rows if r.get("id")}
    except Exception as exc:  # graph unavailable — skip validation
        logger.debug("Could not fetch existing SN names: %s", exc)
        return set()


# =============================================================================
# Graph helpers (extract phase)
# =============================================================================


@retry_on_deadlock()
def claim_names_for_enrichment(
    *,
    limit: int = 50,
    domain: str | list[str] | None = None,
    ids: str | None = None,
    force: bool = False,
    status_filter: list[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Atomically claim StandardNames at a given pipeline_status for enrichment.

    Uses the two-step claim-token pattern (SET + verify) to prevent
    parallel workers from double-processing.  Skips nodes already at
    ``enriched`` or later status unless *force* is True.

    Args:
        limit: Maximum number of names to claim.
        domain: One or more physics domains to filter by.
        ids: IDS name to filter by (via HAS_STANDARD_NAME relationships).
        force: If True, also claim already-enriched nodes.
        status_filter: Review statuses to claim from.  Defaults to
            ``['named']``.

    Returns ``(token, items)`` where *token* must be passed to
    downstream stages for release or mark-done.
    """
    from imas_codex.graph.client import GraphClient

    token = str(uuid.uuid4())

    # Resolve status filter
    statuses = status_filter or ["named"]

    # Build WHERE clauses
    where_parts = [
        "sn.pipeline_status IN $statuses",
        "(sn.enrich_claimed_at IS NULL"
        "  OR sn.enrich_claimed_at < datetime() - duration($timeout))",
    ]
    params: dict[str, Any] = {
        "limit": limit,
        "token": token,
        "timeout": _CLAIM_TIMEOUT,
        "statuses": statuses,
    }

    if not force:
        # Skip already-enriched nodes
        where_parts.append("sn.enriched_at IS NULL")

    if domain:
        domains = [domain] if isinstance(domain, str) else domain
        where_parts.append("sn.physics_domain IN $domains")
        params["domains"] = domains

    if ids:
        # Filter by IDS via source paths or HAS_STANDARD_NAME relationships
        where_parts.append(
            "EXISTS { MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn)"
            " MATCH (src)-[:IN_IDS]->(i:IDS {id: $ids}) }"
        )
        params["ids"] = ids

    where_clause = " AND ".join(where_parts)

    with GraphClient() as gc:
        # Step 1: Claim with random ordering and unique token
        gc.query(
            f"""
            MATCH (sn:StandardName)
            WHERE {where_clause}
            WITH sn ORDER BY rand() LIMIT $limit
            SET sn.enrich_claimed_at = datetime(),
                sn.enrich_claim_token = $token
            """,
            **params,
        )

        # Step 2: Verify — only nodes with our token
        results = gc.query(
            """
            MATCH (sn:StandardName {enrich_claim_token: $token})
            OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
            RETURN sn.id AS id,
                   sn.description AS description,
                   sn.documentation AS documentation,
                   sn.kind AS kind,
                   coalesce(u.id, sn.unit) AS unit,
                   sn.tags AS tags,
                   sn.links AS links,
                   sn.source_paths AS source_paths,
                   sn.grammar_physical_base AS physical_base,
                   sn.grammar_subject AS subject,
                   sn.grammar_component AS component,
                   sn.grammar_coordinate AS coordinate,
                   sn.grammar_position AS position,
                   sn.grammar_process AS process,
                   sn.physics_domain AS physics_domain,
                   sn.confidence AS confidence,
                   sn.model AS model
            """,
            token=token,
        )
        return token, [dict(r) for r in results]


def release_enrichment_claims(token: str) -> int:
    """Release enrichment claims on error. Token-verified."""
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sn:StandardName {enrich_claim_token: $token})
            SET sn.enrich_claimed_at = null,
                sn.enrich_claim_token = null
            RETURN count(sn) AS released
            """,
            token=token,
        )
        return result[0]["released"] if result else 0


# =============================================================================
# EXTRACT phase
# =============================================================================


async def enrich_extract_worker(state: StandardNameEnrichState, **_kwargs) -> None:
    """Extract ``pipeline_status='named'`` StandardNames into enrichment batches.

    Queries the graph for named SNs (filtered by domain/ids/limit),
    claims them with a token, and groups into batches for downstream
    processing.  Skips already-enriched nodes unless ``force=True``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="enrich_extract_worker")
    wlog.info(
        "Starting enrichment extraction (domain=%s, ids=%s, limit=%s)",
        state.domain,
        state.ids,
        state.limit,
    )

    claim_limit = state.limit or 500

    def _claim() -> tuple[str, list[dict[str, Any]]]:
        return claim_names_for_enrichment(
            limit=claim_limit,
            domain=state.domain,
            ids=state.ids,
            force=state.force,
            status_filter=state.status_filter,
        )

    from imas_codex.standard_names.batching import get_enrich_batch_config

    enrich_cfg = get_enrich_batch_config()

    if state.dry_run:
        # In dry-run mode, query without claiming
        token, items = await asyncio.to_thread(_claim)
        # Release immediately — no mutations in dry_run
        if items:
            await asyncio.to_thread(release_enrichment_claims, token)

        # Build batches from claimed items
        batches = _build_batches(
            items,
            batch_size=state.batch_size,
            max_tokens=enrich_cfg["max_tokens"],
        )
        state.batches = batches
        total_items = sum(len(b["items"]) for b in batches)
        state.extract_stats.total = total_items
        state.extract_stats.processed = total_items
        state.stats["extract_count"] = total_items
        state.stats["extract_batches"] = len(batches)
        wlog.info(
            "Dry run — extracted %d items in %d batches (claims released)",
            total_items,
            len(batches),
        )
        state.extract_stats.freeze_rate()
        state.extract_phase.mark_done()
        return

    # Live mode: claim and batch
    token, items = await asyncio.to_thread(_claim)

    if not items:
        wlog.info("No named StandardNames found for enrichment")
        state.extract_stats.freeze_rate()
        state.extract_phase.mark_done()
        return

    batches = _build_batches(
        items,
        batch_size=state.batch_size,
        token=token,
        max_tokens=enrich_cfg["max_tokens"],
    )
    state.batches = batches
    total_items = sum(len(b["items"]) for b in batches)
    state.extract_stats.total = total_items
    state.extract_stats.processed = total_items
    state.extract_stats.record_batch(total_items)

    wlog.info(
        "Extraction complete: %d items in %d batches",
        total_items,
        len(batches),
    )
    state.stats["extract_count"] = total_items
    state.stats["extract_batches"] = len(batches)

    state.extract_stats.freeze_rate()
    state.extract_phase.mark_done()


def _build_batches(
    items: list[dict[str, Any]],
    batch_size: int = _ENRICH_BATCH_SIZE,
    token: str | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Split flat item list into enrichment batches.

    Each batch is a dict with ``items`` (list of SN dicts) and
    ``claim_token`` for downstream release/mark.

    When *max_tokens* is set, a pre-flight token check binary-splits
    any batch whose estimated token count exceeds the budget.
    """
    if not items:
        return []

    batches = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        batches.append(
            {
                "items": chunk,
                "claim_token": token,
                "batch_index": len(batches),
            }
        )

    if max_tokens is not None:
        from imas_codex.standard_names.batching import pre_flight_enrich_token_check

        batches = pre_flight_enrich_token_check(batches, max_tokens=max_tokens)
        # Re-index after potential splits
        for idx, b in enumerate(batches):
            b["batch_index"] = idx

    return batches


# =============================================================================
# CONTEXTUALISE phase (C.2)
# =============================================================================

# Maximum characters for truncated description/documentation strings.
_DESC_TRUNCATE = 200

# How many vector-similar neighbours to fetch per SN.
_NEARBY_K = 6

# How many domain siblings to fetch per SN.
_SIBLINGS_LIMIT = 8

# Over-fetch factor for vector search (to compensate for status filtering).
_VECTOR_OVERFETCH = 3

# Grammar segment keys expected on each SN item from the generate phase.
_GRAMMAR_KEYS = (
    "physical_base",
    "subject",
    "component",
    "coordinate",
    "position",
    "process",
)


def _truncate(text: str | None, max_len: int = _DESC_TRUNCATE) -> str | None:
    """Truncate *text* to *max_len* chars, adding ellipsis if trimmed."""
    if not text:
        return text
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _build_grammar(item: dict[str, Any]) -> dict[str, str]:
    """Extract grammar decomposition dict from SN item qualifier fields."""
    return {k: item[k] for k in _GRAMMAR_KEYS if item.get(k)}


def _fetch_dd_paths_batch(
    gc: Any,
    sn_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Fetch DD path details and COCOS type for a batch of SN ids.

    Returns a mapping ``{sn_id: {"dd_paths": [...], "cocos": str|None}}``.
    Each ``dd_paths`` entry is ``{path, ids, description, documentation, unit}``.
    """
    if not sn_ids:
        return {}

    rows = gc.query(
        """
        MATCH (sn:StandardName)
        WHERE sn.id IN $ids
        OPTIONAL MATCH (n:IMASNode)-[:HAS_STANDARD_NAME]->(sn)
        RETURN sn.id AS sn_id,
               sn.cocos_transformation_type AS cocos,
               n.id AS path,
               n.ids AS ids,
               n.description AS description,
               n.documentation AS documentation,
               n.unit AS unit
        """,
        ids=sn_ids,
    )

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = row["sn_id"]
        if sid not in result:
            result[sid] = {"dd_paths": [], "cocos": row.get("cocos")}
        # Only add path entry if an IMASNode was matched
        if row.get("path"):
            result[sid]["dd_paths"].append(
                {
                    "path": row["path"],
                    "ids": row.get("ids"),
                    "description": _truncate(row.get("description")),
                    "documentation": _truncate(row.get("documentation")),
                    "unit": row.get("unit"),
                }
            )
    return result


def _fetch_nearby_sns(
    gc: Any,
    items: list[dict[str, Any]],
    k: int = _NEARBY_K,
) -> dict[str, list[dict[str, str | None]]]:
    """Fetch k vector-similar StandardNames for each item via the vector index.

    Iterates items but reuses the same *gc* session. Only returns neighbours
    with ``pipeline_status IN ['enriched', 'published', 'accepted']``.

    Returns ``{sn_id: [{name, description}, ...]}``.
    """
    result: dict[str, list[dict[str, str | None]]] = {}
    fetch_k = k * _VECTOR_OVERFETCH

    for item in items:
        sid = item["id"]
        try:
            rows = gc.query(
                """
                MATCH (target:StandardName {id: $target_id})
                WHERE target.embedding IS NOT NULL
                CALL db.index.vector.queryNodes(
                    'standard_name_desc_embedding', $fetch_k, target.embedding
                ) YIELD node, score
                WHERE node.id <> $target_id
                  AND node.pipeline_status IN ['enriched', 'published', 'accepted']
                RETURN node.id AS name,
                       node.description AS description
                LIMIT $k
                """,
                target_id=sid,
                fetch_k=fetch_k,
                k=k,
            )
            result[sid] = [
                {"name": r["name"], "description": _truncate(r.get("description"))}
                for r in rows
            ]
        except Exception:
            # No embedding or index unavailable — empty list is fine
            logger.debug("Vector search failed for %s — skipping nearby", sid)
            result[sid] = []

    return result


def _fetch_domain_siblings(
    gc: Any,
    items: list[dict[str, Any]],
    limit: int = _SIBLINGS_LIMIT,
) -> dict[str, list[dict[str, str | None]]]:
    """Fetch domain siblings for a batch of items.

    Groups items by ``physics_domain`` to minimise query count.
    Falls back to IDS-based siblings when ``physics_domain`` is unset.

    Returns ``{sn_id: [{name, description}, ...]}``.
    """
    result: dict[str, list[dict[str, str | None]]] = {}

    # Group items by domain
    domain_groups: dict[str, list[str]] = {}
    ids_fallback: list[dict[str, Any]] = []

    for item in items:
        sid = item["id"]
        domain = item.get("physics_domain")
        if domain:
            domain_groups.setdefault(domain, []).append(sid)
        else:
            ids_fallback.append(item)

    # One query per unique domain
    for domain, sn_ids in domain_groups.items():
        try:
            rows = gc.query(
                """
                MATCH (sibling:StandardName)
                WHERE sibling.physics_domain = $domain
                  AND NOT (sibling.id IN $exclude_ids)
                  AND sibling.pipeline_status IN [
                      'named', 'enriched', 'reviewable',
                      'published', 'accepted'
                  ]
                RETURN sibling.id AS name,
                       sibling.description AS description
                LIMIT $limit
                """,
                domain=domain,
                exclude_ids=sn_ids,
                limit=limit,
            )
            siblings = [
                {"name": r["name"], "description": _truncate(r.get("description"))}
                for r in rows
            ]
        except Exception:
            logger.warning("Sibling query failed for domain=%s", domain)
            siblings = []

        for sid in sn_ids:
            result[sid] = siblings

    # Fallback: IDS-based siblings for items without physics_domain
    for item in ids_fallback:
        sid = item["id"]
        source_paths = item.get("source_paths") or []
        # Extract IDS name from first source path (e.g. "equilibrium/time_slice/...")
        ids_name = source_paths[0].split("/")[0] if source_paths else None
        if not ids_name:
            result[sid] = []
            continue

        try:
            rows = gc.query(
                """
                MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sibling:StandardName)
                WHERE src.ids = $ids_name
                  AND sibling.id <> $exclude_id
                  AND sibling.pipeline_status IN [
                      'named', 'enriched', 'reviewable',
                      'published', 'accepted'
                  ]
                RETURN DISTINCT sibling.id AS name,
                       sibling.description AS description
                LIMIT $limit
                """,
                ids_name=ids_name,
                exclude_id=sid,
                limit=limit,
            )
            result[sid] = [
                {"name": r["name"], "description": _truncate(r.get("description"))}
                for r in rows
            ]
        except Exception:
            logger.warning("IDS sibling query failed for %s (ids=%s)", sid, ids_name)
            result[sid] = []

    return result


def _fetch_link_candidates(
    gc: Any,
    dd_data: dict[str, dict[str, Any]],
    items: list[dict[str, Any]],
    max_per_sn: int = 20,
) -> dict[str, list[dict]]:
    """Compute link candidates for each SN via hybrid DD search.

    For each SN, runs hybrid DD search across all its linked DD paths
    (from *dd_data*) and merges results.  Pre-resolves
    ``HAS_STANDARD_NAME`` to tag candidates as ``name:`` or ``dd:``.
    Deduplicates and caps per SN at *max_per_sn*.

    Returns ``{sn_id: [{tag, path, ids, kind_hint, doc_short, ...}, ...]}``.
    """
    from imas_codex.standard_names.workers import _hybrid_search_neighbours

    result: dict[str, list[dict]] = {}
    for item in items:
        sid = item["id"]
        dd_info = dd_data.get(sid, {})
        dd_paths = dd_info.get("dd_paths", [])

        if not dd_paths:
            result[sid] = []
            continue

        # Merge hybrid search results across all DD paths for this SN
        seen: dict[str, dict] = {}  # path → best candidate dict
        for dp in dd_paths:
            path = dp.get("path", "")
            if not path:
                continue
            try:
                neighbours = _hybrid_search_neighbours(
                    gc,
                    path,
                    description=dp.get("description"),
                    physics_domain=item.get("physics_domain"),
                    max_results=10,
                )
            except Exception:
                logger.debug(
                    "Hybrid search failed for %s (DD %s)",
                    sid,
                    path,
                    exc_info=True,
                )
                continue

            for n in neighbours:
                npath = n["path"]
                if npath not in seen:
                    # Add kind_hint: "name" if already minted, "dd" otherwise
                    n["kind_hint"] = "name" if n["tag"].startswith("name:") else "dd"
                    seen[npath] = n

        # Sort by pre-existing tag priority (name: first) then truncate
        candidates = sorted(
            seen.values(),
            key=lambda c: 0 if c["tag"].startswith("name:") else 1,
        )[:max_per_sn]

        result[sid] = candidates

    return result


def _fetch_related_neighbours(
    gc: Any,
    dd_data: dict[str, dict[str, Any]],
    items: list[dict[str, Any]],
) -> dict[str, list[dict]]:
    """Fetch graph-relationship neighbours for each SN's DD paths.

    For each SN, calls :func:`related_dd_search` across all its linked
    DD paths and merges the results.  Returns compact dicts with
    ``{path, ids, relationship_type, via}`` suitable for template injection.

    Returns ``{sn_id: [{path, ids, relationship_type, via}, ...]}``.
    """
    from imas_codex.standard_names.workers import _related_path_neighbours

    result: dict[str, list[dict]] = {}
    for item in items:
        sid = item["id"]
        dd_info = dd_data.get(sid, {})
        dd_paths = dd_info.get("dd_paths", [])

        if not dd_paths:
            result[sid] = []
            continue

        seen: dict[str, dict] = {}  # path → hit dict (dedup)
        for dp in dd_paths:
            path = dp.get("path", "")
            if not path:
                continue
            try:
                neighbours = _related_path_neighbours(gc, path)
            except Exception:
                logger.debug(
                    "Related search failed for %s (DD %s)",
                    sid,
                    path,
                    exc_info=True,
                )
                continue

            for n in neighbours:
                npath = n["path"]
                if npath not in seen:
                    seen[npath] = n

        result[sid] = list(seen.values())

    return result


async def enrich_contextualise_worker(
    state: StandardNameEnrichState, **_kwargs
) -> None:
    """Gather DD docs, nearby SNs, and domain siblings for enrichment context.

    For each batch of StandardName items, enriches them with:

    - **dd_paths**: linked IMASNode descriptions, documentation, and units.
    - **nearby**: k nearest-neighbour SNs by vector similarity.
    - **siblings**: SNs in the same physics domain or IDS.
    - **grammar**: decomposition dict from qualifier fields.
    - **cocos**: COCOS transformation type from the SN node.

    Context is written directly onto each item dict so that downstream
    workers (DOCUMENT) can template against it.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="enrich_contextualise_worker")

    if not state.batches:
        wlog.info("No batches to contextualise — skipping")
        state.contextualise_stats.freeze_rate()
        state.contextualise_phase.mark_done()
        return

    total_items = sum(len(b["items"]) for b in state.batches)
    state.contextualise_stats.total = total_items

    wlog.info(
        "Contextualising %d items across %d batches",
        total_items,
        len(state.batches),
    )

    processed = 0
    errors = 0

    for batch in state.batches:
        if state.stop_requested:
            wlog.info("Stop requested — aborting contextualise")
            break

        items = batch["items"]
        if not items:
            continue

        sn_ids = [it["id"] for it in items]

        # --- Fetch all context in threadpool (graph I/O) ---
        def _fetch_context(
            _sn_ids: list[str] = sn_ids,
            _items: list[dict[str, Any]] = items,
        ):
            from imas_codex.graph.client import GraphClient

            with GraphClient() as gc:
                dd_data = _fetch_dd_paths_batch(gc, _sn_ids)
                nearby_data = _fetch_nearby_sns(gc, _items)
                sibling_data = _fetch_domain_siblings(gc, _items)
                link_data = _fetch_link_candidates(gc, dd_data, _items)
                related_data = _fetch_related_neighbours(gc, dd_data, _items)
            return dd_data, nearby_data, sibling_data, link_data, related_data

        try:
            (
                dd_data,
                nearby_data,
                sibling_data,
                link_data,
                related_data,
            ) = await asyncio.to_thread(_fetch_context)
        except Exception:
            wlog.warning(
                "Graph error fetching context for batch %d — skipping",
                batch.get("batch_index", 0),
                exc_info=True,
            )
            errors += len(items)
            continue

        # --- Merge context onto each item ---
        for item in items:
            if state.stop_requested:
                wlog.info("Stop requested — aborting contextualise mid-batch")
                break

            sid = item["id"]
            try:
                dd_info = dd_data.get(sid, {})
                item["dd_paths"] = dd_info.get("dd_paths", [])
                item["cocos"] = dd_info.get("cocos")
                item["nearby"] = nearby_data.get(sid, [])
                item["siblings"] = sibling_data.get(sid, [])
                item["link_candidates"] = link_data.get(sid, [])
                item["related_neighbours"] = related_data.get(sid, [])
                item["grammar"] = _build_grammar(item)

                # Preserve existing description/documentation as "current"
                # so the LLM can improve upon prior enrichment attempts
                item["current"] = {
                    "description": item.get("description"),
                    "documentation": item.get("documentation"),
                    "tags": item.get("tags"),
                    "links": item.get("links"),
                }

                processed += 1
            except Exception:
                wlog.warning(
                    "Error merging context for %s — skipping", sid, exc_info=True
                )
                errors += 1

    state.contextualise_stats.processed = processed
    state.contextualise_stats.errors = errors
    state.stats["contextualise_processed"] = processed
    state.stats["contextualise_errors"] = errors

    wlog.info(
        "Contextualise complete: %d processed, %d errors",
        processed,
        errors,
    )

    state.contextualise_stats.freeze_rate()
    state.contextualise_phase.mark_done()


# =============================================================================
# DOCUMENT phase (C.3)
# =============================================================================

# Read-only fields: these come from the Data Dictionary and must never be
# overwritten by LLM output.
_READONLY_FIELDS = {"unit", "physics_domain"}


async def enrich_document_worker(state: StandardNameEnrichState, **_kwargs) -> None:
    """LLM call to generate descriptions and documentation for enrichment.

    For each batch of contextualised StandardName items:

    1. Render system/user prompts from ``sn/enrich_system`` and ``sn/enrich_user``.
    2. Call ``acall_llm_structured`` to produce ``StandardNameEnrichBatch``.
    3. Merge enriched fields back onto items, preserving read-only DD fields
       (``unit``, ``physics_domain``).
    4. Track cost/tokens on ``state`` and respect budget + stop signals.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="enrich_document_worker")

    if not state.batches:
        wlog.info("No batches to document — skipping")
        state.document_stats.freeze_rate()
        state.document_phase.mark_done()
        return

    if state.dry_run:
        total = sum(len(b["items"]) for b in state.batches)
        wlog.info("Dry run — skipping document for %d items", total)
        state.document_stats.total = total
        state.document_stats.processed = total
        state.document_stats.freeze_rate()
        state.document_phase.mark_done()
        return

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model
    from imas_codex.standard_names.models import StandardNameEnrichBatch

    model = state.model or get_model("sn-enrich")

    total_items = sum(len(b["items"]) for b in state.batches)
    state.document_stats.total = total_items

    wlog.info(
        "Documenting %d items in %d batches (model=%s)",
        total_items,
        len(state.batches),
        model,
    )

    # Render system prompt once (identical across batches → prompt-cacheable)
    # --- K3: Load scored examples for enrich prompt ---
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.example_loader import load_compose_examples

    enrich_domains: list[str] = list(state.domain or [])
    if not enrich_domains:
        _domains = {
            item.get("physics_domain")
            for batch in state.batches
            for item in batch.get("items", [])
            if item.get("physics_domain")
        }
        enrich_domains = sorted(_domains)

    def _load_enrich_scored() -> list[dict]:
        with GraphClient() as gc:
            return load_compose_examples(
                gc, physics_domains=enrich_domains, axis="docs"
            )

    compose_scored_examples = await asyncio.to_thread(_load_enrich_scored)
    if compose_scored_examples:
        wlog.info(
            "K3: Loaded %d scored examples for enrich (domains=%s)",
            len(compose_scored_examples),
            enrich_domains or "all",
        )
    system_prompt = render_prompt(
        "sn/enrich_system", {"compose_scored_examples": compose_scored_examples}
    )

    # Fetch set of existing SN ids once for link-validation in the sanitizer.
    valid_names = _fetch_existing_sn_names()
    if valid_names:
        wlog.info(
            "Link validation active: %d existing SN names known", len(valid_names)
        )

    processed = 0
    errors = 0

    for batch in state.batches:
        # --- Stop signals ---
        if state.stop_requested:
            wlog.info("Stop requested — aborting document phase")
            break

        if state.budget_exhausted:
            wlog.info("Document phase stopped: budget exhausted")
            break

        items = batch["items"]
        if not items:
            continue

        # Inject ``name`` alias for template compatibility (template uses
        # ``item.name``; items use ``id`` as their primary key).
        for item in items:
            item.setdefault("name", item["id"])

        # Render per-batch user prompt
        user_prompt = render_prompt(
            "sn/enrich_user",
            {"batch": batch, "items": items},
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            result, cost, tokens = await acall_llm_structured(
                model=model,
                messages=messages,
                response_model=StandardNameEnrichBatch,
                service="standard-names",
            )

            # Accumulate cost/tokens on state
            state.cost += cost
            state.tokens_in += tokens  # total tokens (in + out combined)
            state.document_stats.cost += cost

            # Distribute per-item cost so persist_enriched_batch can write
            # llm_cost_enrich on each StandardName node.
            per_item_cost = cost / len(items) if items else 0.0

            # Build lookup from LLM response by standard_name
            response_map: dict[str, Any] = {}
            for enriched_item in result.items:
                response_map[enriched_item.standard_name] = enriched_item

            # Merge enriched fields onto batch items (with sanitization)
            for item in items:
                name = item["id"]
                enriched = response_map.get(name)
                if not enriched:
                    wlog.debug(
                        "LLM response missing item %s — skipping enrichment", name
                    )
                    continue

                # Guard: reject items where the LLM didn't echo the input name
                # verbatim (catches runaway hallucination into standard_name field).
                if not _is_echoed_name(enriched.standard_name, name):
                    wlog.warning(
                        "Dropping enrichment for %s — LLM returned mismatched name "
                        "(%d chars)",
                        name,
                        len(enriched.standard_name or ""),
                    )
                    continue

                item["enriched_description"] = enriched.description
                item["enriched_documentation"] = _sanitize_documentation(
                    enriched.documentation
                )
                item["enriched_links"] = _sanitize_links(
                    enriched.links, valid_names=valid_names
                )
                item["enriched_tags"] = _sanitize_tags(enriched.tags)

                # Per-item cost for graph_ops.persist_enriched_batch
                item["enrich_cost_usd"] = per_item_cost
                # Optional enrichment fields (validity_domain, constraints)
                if enriched.validity_domain is not None:
                    item["enriched_validity_domain"] = enriched.validity_domain
                if enriched.constraints:
                    item["enriched_constraints"] = enriched.constraints

            processed += len(items)
            state.document_stats.record_batch(len(items))

        except (ValueError, Exception) as exc:
            wlog.warning(
                "LLM error for batch %d: %s — marking failed",
                batch.get("batch_index", 0),
                str(exc)[:200],
            )
            batch["failed"] = True
            errors += len(items)
            state.document_stats.errors += len(items)

    state.document_stats.processed = processed
    state.stats["document_processed"] = processed
    state.stats["document_errors"] = errors

    wlog.info(
        "Document complete: %d processed, %d errors, cost=$%.4f",
        processed,
        errors,
        state.cost,
    )

    state.document_stats.freeze_rate()
    state.document_phase.mark_done()


# =============================================================================
# VALIDATE phase (C.4)
# =============================================================================

# British → American spelling substitutions (warning only).
_BRITISH_SPELLING: dict[str, str] = {
    "colour": "color",
    "colours": "colors",
    "behaviour": "behavior",
    "behaviours": "behaviors",
    "characterise": "characterize",
    "characterised": "characterized",
    "characterises": "characterizes",
    "characterising": "characterizing",
    "optimise": "optimize",
    "optimised": "optimized",
    "optimises": "optimizes",
    "optimising": "optimizing",
    "analyse": "analyze",
    "analysed": "analyzed",
    "analyses": "analyzes",
    "analysing": "analyzing",
    "centre": "center",
    "centres": "centers",
    "modelled": "modeled",
    "modelling": "modeling",
    "ionisation": "ionization",
    "magnetised": "magnetized",
    "normalise": "normalize",
    "normalised": "normalized",
    "normalises": "normalizes",
    "normalising": "normalizing",
    "minimise": "minimize",
    "minimised": "minimized",
    "minimises": "minimizes",
    "minimising": "minimizing",
    "maximise": "maximize",
    "maximised": "maximized",
    "maximises": "maximizes",
    "maximising": "maximizing",
    "utilise": "utilize",
    "utilised": "utilized",
    "utilises": "utilizes",
    "utilising": "utilizing",
    "polarise": "polarize",
    "polarised": "polarized",
    "parameterise": "parameterize",
    "parameterised": "parameterized",
    "symmetrise": "symmetrize",
    "symmetrised": "symmetrized",
    "discretise": "discretize",
    "discretised": "discretized",
    "linearise": "linearize",
    "linearised": "linearized",
    "stabilise": "stabilize",
    "stabilised": "stabilized",
    "equalise": "equalize",
    "equalised": "equalized",
    "generalise": "generalize",
    "generalised": "generalized",
    "specialise": "specialize",
    "specialised": "specialized",
    "idealise": "idealize",
    "idealised": "idealized",
    "realise": "realize",
    "realised": "realized",
    "recognise": "recognize",
    "recognised": "recognized",
    "organisation": "organization",
    "organisations": "organizations",
    "localise": "localize",
    "localised": "localized",
    "harmonise": "harmonize",
    "harmonised": "harmonized",
    "vapour": "vapor",
    "vapours": "vapors",
    "fibre": "fiber",
    "fibres": "fibers",
    "metre": "meter",
    "metres": "meters",
    "litre": "liter",
    "litres": "liters",
    "calibre": "caliber",
}

# Compile a single regex from all British keys (word-boundary match).
_BRITISH_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _BRITISH_SPELLING) + r")\b",
    re.IGNORECASE,
)

# Statuses that indicate an existing valid link target.
_VALID_LINK_STATUSES = frozenset({"named", "enriched", "published", "accepted"})


def _check_british_spelling(text: str | None) -> list[str]:
    """Return warning strings for British spellings found in *text*."""
    if not text:
        return []
    issues: list[str] = []
    for match in _BRITISH_RE.finditer(text):
        word = match.group(0).lower()
        american = _BRITISH_SPELLING.get(word)
        if american:
            issues.append(f"british_spelling:{word}→{american}")
    return issues


def _check_latex_syntax(text: str | None) -> list[str]:
    """Return warning strings for unbalanced LaTeX in *text*."""
    if not text:
        return []
    issues: list[str] = []
    # Unbalanced dollar signs (inline math)
    dollar_count = text.count("$")
    # Subtract escaped dollars
    escaped_count = text.count("\\$")
    effective = dollar_count - escaped_count
    if effective % 2 != 0:
        issues.append("latex_syntax_warning:unbalanced $ delimiters")
    # \frac missing braces
    frac_pattern = re.compile(r"\\frac(?!\s*\{)")
    if frac_pattern.search(text):
        issues.append("latex_syntax_warning:\\frac missing braces")
    return issues


def _check_links_batch(
    items: list[dict[str, Any]],
    batch_ids: set[str],
) -> dict[str, list[str]]:
    """Check link integrity for a batch of items.

    Links that reference a name present in *batch_ids* are valid
    (they'll resolve once persisted).  Links referencing existing
    StandardName nodes with valid pipeline_status are also valid.
    Everything else gets a ``link_not_found`` warning.

    Returns ``{item_id: [issue_strings]}``.
    """
    # Collect all unique link targets across the batch
    all_links: set[str] = set()
    per_item: dict[str, list[str]] = {}
    for item in items:
        links = item.get("enriched_links") or []
        per_item[item["id"]] = list(links)
        for link in links:
            if link not in batch_ids:
                all_links.add(link)

    # Query graph for existing targets (single batch query)
    existing_links: set[str] = set()
    if all_links:
        try:
            from imas_codex.graph.client import GraphClient

            with GraphClient() as gc:
                rows = gc.query(
                    """
                    UNWIND $link_ids AS lid
                    MATCH (sn:StandardName {id: lid})
                    WHERE sn.pipeline_status IN $statuses
                    RETURN sn.id AS id
                    """,
                    link_ids=list(all_links),
                    statuses=list(_VALID_LINK_STATUSES),
                )
                existing_links = {r["id"] for r in rows}
        except Exception:
            logger.warning(
                "Graph query for link targets failed — all unknown links will warn"
            )

    # Build per-item issues
    result: dict[str, list[str]] = {}
    for item in items:
        item_id = item["id"]
        issues: list[str] = []
        for link in per_item.get(item_id, []):
            if link in batch_ids or link in existing_links:
                continue
            issues.append(f"link_not_found:{link}")
        result[item_id] = issues

    return result


def _validate_item_pydantic(item: dict[str, Any]) -> list[str]:
    """Run ISN Pydantic construction for a single item.

    Returns list of tagged issue strings. Empty list means success.
    """
    try:
        from imas_standard_names.models import create_standard_name_entry

        from imas_codex.standard_names.kind_derivation import to_isn_kind

        data = {
            "name": item["id"],
            "kind": to_isn_kind(item.get("kind") or "scalar"),
            "unit": item.get("unit") or "",
            "description": item.get("enriched_description") or "",
            "documentation": item.get("enriched_documentation") or "",
        }
        # Include optional fields if present
        if item.get("tags") or item.get("enriched_tags"):
            tags = list(item.get("enriched_tags") or item.get("tags") or [])
            data["tags"] = tags
        if item.get("enriched_links"):
            data["links"] = list(item["enriched_links"])

        create_standard_name_entry(data)
        return []
    except Exception as exc:
        return [f"[pydantic] {exc}"]


def _validate_item_description(item: dict[str, Any]) -> list[str]:
    """Run ISN description quality checks for a single item.

    Returns list of tagged issue strings.
    """
    try:
        from imas_standard_names.validation import validate_description

        entry = {
            "description": item.get("enriched_description") or "",
            "tags": list(item.get("enriched_tags") or item.get("tags") or []),
            "kind": item.get("kind") or "scalar",
        }
        issues = validate_description(entry)
        return [f"[description] {iss.get('message', str(iss))}" for iss in issues]
    except Exception as exc:
        logger.debug("validate_description error for %s: %s", item.get("id"), exc)
        return []


def _check_empty_documentation(item: dict[str, Any]) -> list[str]:
    """Check that description and documentation are not empty/whitespace.

    Returns quarantine-level issues if either is blank (D5/P0.1).
    """
    issues: list[str] = []
    desc = (item.get("enriched_description") or "").strip()
    doc = (item.get("enriched_documentation") or "").strip()
    if not desc:
        issues.append("[empty_documentation] description is empty or whitespace")
    if not doc:
        issues.append("[empty_documentation] documentation is empty or whitespace")
    return issues


def _check_unit_sanity(item: dict[str, Any]) -> list[str]:
    """Run dimensional-sanity checks on item name vs unit (D5/P0.2).

    Delegates to :func:`unit_audit.check_unit_sanity` and wraps results.
    """
    from imas_codex.standard_names.unit_audit import check_unit_sanity

    name = item.get("id") or ""
    unit = item.get("unit") or ""
    raw_issues = check_unit_sanity(name, unit)
    return [f"[{tag}]" for tag in raw_issues]


def _apply_kind_derivation(item: dict[str, Any]) -> None:
    """Override the LLM's ``kind`` field using name-based derivation (D5/P0.3).

    Mutates *item* in place — sets ``kind`` to the derived value.
    """
    from imas_codex.standard_names.kind_derivation import derive_kind

    name = item.get("id") or ""
    derived = derive_kind(name)
    if derived != (item.get("kind") or "scalar"):
        logger.debug(
            "Kind override for %s: %s → %s",
            name,
            item.get("kind"),
            derived,
        )
    item["kind"] = derived


async def enrich_validate_worker(state: StandardNameEnrichState, **_kwargs) -> None:
    """Validate enriched names: spelling, link integrity, description quality.

    For each item with ``enriched_description`` present, runs:
    1. British spelling check (warning only).
    2. Empty-doc check — quarantines items with blank description/documentation.
    3. ISN Pydantic construction — failure quarantines.
    4. Dimensional-sanity unit audit — quarantines on mismatch.
    5. Auto-derive kind from name (overrides LLM default).
    6. Description semantic checks — warnings.
    7. Link integrity — unknown links produce warnings.
    8. LaTeX/math syntax — warnings.

    Sets ``validation_status`` ('valid' or 'quarantined') and
    ``validation_issues`` (list of tagged strings) on each item.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="enrich_validate_worker")

    if not state.batches:
        wlog.info("No batches to validate — skipping")
        state.validate_stats.freeze_rate()
        state.validate_phase.mark_done()
        return

    total_items = sum(len(b["items"]) for b in state.batches)
    state.validate_stats.total = total_items

    wlog.info(
        "Validating %d items across %d batches",
        total_items,
        len(state.batches),
    )

    processed = 0
    quarantined = 0
    skipped = 0

    for batch in state.batches:
        if state.stop_requested:
            wlog.info("Stop requested — aborting validate")
            break

        items = batch["items"]
        if not items:
            continue

        # Build set of all item IDs in this batch for in-batch link resolution.
        batch_ids = {it["id"] for it in items}

        # --- Link integrity (batch query) ---
        link_issues = await asyncio.to_thread(_check_links_batch, items, batch_ids)

        for item in items:
            if state.stop_requested:
                break

            sid = item["id"]
            enriched_desc = item.get("enriched_description")

            if enriched_desc is None:
                # No new enriched description from this run. Check whether
                # the item already has a description persisted from a prior
                # enrich pass — if not, it is structurally incomplete and
                # must be quarantined (P0.1 — prevents empty-doc leaks).
                existing_desc = item.get("description")
                existing_doc = item.get("documentation")
                if not (existing_desc and str(existing_desc).strip()) and not (
                    existing_doc and str(existing_doc).strip()
                ):
                    item["validation_status"] = "quarantined"
                    item["validation_issues"] = ["empty_documentation"]
                    quarantined += 1
                else:
                    item["validation_status"] = "pending"
                    item["validation_issues"] = []
                    skipped += 1
                continue

            issues: list[str] = []
            is_quarantined = False

            # 1. British spelling check (warning only)
            issues.extend(_check_british_spelling(enriched_desc))
            issues.extend(_check_british_spelling(item.get("enriched_documentation")))

            # 2. Empty-doc check (P0.1 — hard quarantine)
            empty_doc_issues = _check_empty_documentation(item)
            if empty_doc_issues:
                issues.extend(empty_doc_issues)
                is_quarantined = True

            # 3. ISN Pydantic construction
            pydantic_issues = _validate_item_pydantic(item)
            if pydantic_issues:
                issues.extend(pydantic_issues)
                is_quarantined = True

            # 4. Dimensional-sanity unit audit (P0.2)
            unit_issues = _check_unit_sanity(item)
            if unit_issues:
                issues.extend(unit_issues)
                is_quarantined = True

            # 5. Auto-derive kind from name (P0.3 — override LLM default)
            _apply_kind_derivation(item)

            # 6. Description semantic checks (warning only)
            desc_issues = _validate_item_description(item)
            issues.extend(desc_issues)

            # 7. Link integrity (from batch query)
            issues.extend(link_issues.get(sid, []))

            # 8. LaTeX syntax (warning only)
            issues.extend(_check_latex_syntax(enriched_desc))
            issues.extend(_check_latex_syntax(item.get("enriched_documentation")))

            # Set status
            item["validation_status"] = "quarantined" if is_quarantined else "valid"
            item["validation_issues"] = issues

            if is_quarantined:
                quarantined += 1
                wlog.debug("Quarantined %s: %s", sid, issues)

            processed += 1

    state.validate_stats.processed = processed
    state.validate_stats.errors = quarantined
    state.stats["validate_processed"] = processed
    state.stats["validate_quarantined"] = quarantined
    state.stats["validate_skipped"] = skipped

    wlog.info(
        "Validate complete: %d processed, %d quarantined, %d skipped",
        processed,
        quarantined,
        skipped,
    )

    state.validate_stats.freeze_rate()
    state.validate_phase.mark_done()


# =============================================================================
# PERSIST phase (C.4)
# =============================================================================


async def enrich_persist_worker(state: StandardNameEnrichState, **_kwargs) -> None:
    """Write enriched data and REFERENCES relationships to graph.

    For each item marked ``valid`` (not quarantined) with ``enriched_description``
    present:
    1. Embeds enriched_description via ``embed_descriptions_batch``.
    2. Persists to graph via ``persist_enriched_batch`` (graph_ops).
    3. Releases enrichment claims.

    Quarantined items are skipped.  Items without ``enriched_description``
    are skipped.  Embedding failures quarantine the affected item.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="enrich_persist_worker")

    if state.dry_run:
        wlog.info("Dry run — skipping persist")
        state.persist_stats.freeze_rate()
        state.persist_phase.mark_done()
        return

    if not state.batches:
        wlog.info("No batches to persist — skipping")
        state.persist_stats.freeze_rate()
        state.persist_phase.mark_done()
        return

    total_items = sum(len(b["items"]) for b in state.batches)
    state.persist_stats.total = total_items

    wlog.info(
        "Persisting %d items across %d batches",
        total_items,
        len(state.batches),
    )

    persisted = 0
    skipped = 0
    errors = 0

    for batch in state.batches:
        if state.stop_requested:
            wlog.info("Stop requested — aborting persist")
            break

        items = batch["items"]
        token = batch.get("claim_token")

        if not items:
            continue

        # Filter to valid items with enriched descriptions.
        candidates = [
            it
            for it in items
            if it.get("validation_status") == "valid" and it.get("enriched_description")
        ]
        skip_count = len(items) - len(candidates)
        skipped += skip_count

        if not candidates:
            # Release claims if nothing to persist.
            if token:
                try:
                    await asyncio.to_thread(release_enrichment_claims, token)
                except Exception as e:
                    wlog.warning("Failed to release claims: %s", e)
            continue

        # 1. Embed enriched descriptions.
        try:
            from imas_codex.embeddings.description import embed_descriptions_batch

            await asyncio.to_thread(
                embed_descriptions_batch,
                candidates,
                "enriched_description",
                "embedding",
            )
        except Exception:
            wlog.warning(
                "Embedding server unavailable — all %d candidates quarantined",
                len(candidates),
                exc_info=True,
            )
            for item in candidates:
                item["embedding"] = None

        # Check per-item embedding success; quarantine failures.
        embeddable: list[dict[str, Any]] = []
        for item in candidates:
            if item.get("embedding"):
                embeddable.append(item)
            else:
                item["validation_status"] = "quarantined"
                existing_issues = list(item.get("validation_issues") or [])
                if "embedding_failed" not in existing_issues:
                    existing_issues.append("embedding_failed")
                item["validation_issues"] = existing_issues
                errors += 1

        # 2. Persist to graph.
        if embeddable:
            try:
                from imas_codex.standard_names.graph_ops import persist_enriched_batch

                written = await asyncio.to_thread(persist_enriched_batch, embeddable)
                persisted += written
            except Exception:
                wlog.warning(
                    "Graph persist failed for batch %d",
                    batch.get("batch_index", 0),
                    exc_info=True,
                )
                errors += len(embeddable)

        # 3. Release enrichment claims.
        if token:
            try:
                await asyncio.to_thread(release_enrichment_claims, token)
            except Exception as e:
                wlog.warning("Failed to release enrichment claims: %s", e)

    state.persist_stats.processed = persisted
    state.persist_stats.errors = errors
    state.stats["persist_written"] = persisted
    state.stats["persist_skipped"] = skipped
    state.stats["persist_errors"] = errors

    wlog.info(
        "Persist complete: %d written, %d skipped, %d errors",
        persisted,
        skipped,
        errors,
    )

    state.persist_stats.freeze_rate()
    state.persist_phase.mark_done()

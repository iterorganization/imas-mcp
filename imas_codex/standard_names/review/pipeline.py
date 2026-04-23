"""SN review pipeline orchestrator.

Wires the EXTRACT → ENRICH → REVIEW → PERSIST workers into the generic
discovery engine for the ``sn review`` command.

Unlike the *generate* pipeline (which creates new names), this pipeline
**reviews existing** StandardName nodes: loading them from the graph,
enriching with cluster/neighborhood context, running LLM quality scoring,
and persisting the review scores back.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from imas_codex.discovery.base.engine import WorkerSpec, run_discovery_engine

# Defense-in-depth: strict SN id pattern used to reject reviewer-hallucinated
# revised_name values (e.g. multi-hundred-char stream-of-consciousness strings).
_SN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_SN_ID_MAX_LEN = 120


def _axis_overwrite_blocked(
    name: dict,
    incoming: str,
) -> bool:
    """Return True if writing ``incoming`` axis would clobber existing data.

    Axis-split storage rules:
      * full-mode aggregates (``reviewer_scores``) block any axis-only write
      * same-axis presence blocks same-axis re-write (``reviewer_scores_name``
        blocks a new name-axis review; ``reviewer_scores_docs`` blocks a new
        docs-axis review)

    Callers bypass this guard with ``--force``.
    """
    if incoming != "full" and name.get("reviewer_scores") is not None:
        return True
    if incoming == "name" and name.get("reviewer_scores_name") is not None:
        return True
    if incoming == "docs" and name.get("reviewer_scores_docs") is not None:
        return True
    return False


def _valid_sn_id(candidate: str | None) -> bool:
    """Return True iff candidate is a plausible standard-name id."""
    if not candidate or not isinstance(candidate, str):
        return False
    if len(candidate) > _SN_ID_MAX_LEN:
        return False
    return bool(_SN_ID_PATTERN.match(candidate))


def _model_slug(model: str) -> str:
    """Return a filesystem/id-safe slug of a model string."""
    import re as _re

    return _re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-") or "unknown"


def _derive_model_family(model: str) -> str:
    """Map an OpenRouter-style model id to a family label."""
    if not model:
        return "other"
    m = model.lower()
    for needle, family in (
        ("anthropic", "anthropic"),
        ("claude", "anthropic"),
        ("openai", "openai"),
        ("gpt", "openai"),
        ("google", "google"),
        ("gemini", "google"),
        ("mistral", "mistral"),
        ("meta", "meta"),
        ("llama", "meta"),
        ("xai", "xai"),
        ("grok", "xai"),
        ("cohere", "cohere"),
    ):
        if needle in m:
            return family
    return "other"


def _build_review_record(
    item: dict,
    *,
    model: str,
    is_canonical: bool,
    reviewed_at: str,
    score: float | None = None,
    scores: Any = None,
    tier: str | None = None,
    comments: str | None = None,
    cost_usd: float | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    comments_per_dim: dict | None = None,
) -> dict:
    """Build a Review graph record for a scored item."""
    sn_id = item.get("id") or ""
    if not sn_id or not model or not reviewed_at:
        return {}
    rid = f"{sn_id}:{_model_slug(model)}:{reviewed_at}"
    raw_scores = scores if scores is not None else item.get("reviewer_scores")
    if isinstance(raw_scores, dict | list):
        scores_json = json.dumps(raw_scores)
    elif isinstance(raw_scores, str) and raw_scores:
        scores_json = raw_scores
    else:
        scores_json = "{}"
    return {
        "id": rid,
        "standard_name_id": sn_id,
        "model": model,
        "model_family": _derive_model_family(model),
        "is_canonical": bool(is_canonical),
        "score": float(
            score if score is not None else (item.get("reviewer_score") or 0.0)
        ),
        "scores_json": scores_json,
        "tier": (tier if tier is not None else item.get("review_tier")) or "unknown",
        "comments": (
            comments if comments is not None else item.get("reviewer_comments")
        )
        or "",
        "comments_per_dim_json": (
            json.dumps(comments_per_dim)
            if comments_per_dim
            else item.get("reviewer_comments_per_dim")
        ),
        "reviewed_at": reviewed_at,
        "llm_model": model,
        "llm_cost": cost_usd,
        "llm_tokens_in": tokens_in,
        "llm_tokens_out": tokens_out,
        "llm_at": reviewed_at,
        "llm_service": "standard-names",
    }


if TYPE_CHECKING:
    from imas_codex.standard_names.review.state import StandardNameReviewState

logger = logging.getLogger(__name__)


# =============================================================================
# EXTRACT phase — load names from graph, filter, batch
# =============================================================================


async def extract_review_worker(state: StandardNameReviewState, **_kwargs: Any) -> None:
    """Load StandardNames from graph, apply filters, and form review batches.

    1. Query full StandardName catalog into ``state.all_names``.
    2. Apply CLI filters (--ids, --domain, --status, --unreviewed, --re-review)
       to produce ``state.target_names``.
    3. Reconstruct clusters for targets via batch graph query.
    4. Group into review batches by (cluster × unit) with token budgets.
    5. Store in ``state.review_batches``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.review.enrichment import (
        group_into_review_batches,
        reconstruct_clusters_batch,
    )

    wlog = WorkerLogAdapter(logger, worker_name="sn_review_extract")
    wlog.info("Starting review extraction")
    state.extract_stats.status_text = "loading catalog…"

    # --- Step 1: Load full catalog -------------------------------------------

    def _load_all_names() -> list[dict]:
        with GraphClient() as gc:
            rows = gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.validation_status = 'valid'
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
                       sn.cocos_transformation_type AS cocos_transformation_type,
                       sn.physics_domain AS physics_domain,
                       sn.pipeline_status AS pipeline_status,
                       sn.reviewer_score AS reviewer_score,
                       sn.reviewer_scores AS reviewer_scores,
                       sn.reviewer_scores_name AS reviewer_scores_name,
                       sn.reviewer_scores_docs AS reviewer_scores_docs,
                       sn.review_input_hash AS review_input_hash,
                       sn.embedding AS embedding,
                       sn.review_tier AS review_tier,
                       sn.reviewer_comments AS reviewer_comments,
                       sn.source_types AS source_types,
                       sn.source_id AS source_id,
                       sn.generated_at AS generated_at,
                       sn.reviewed_at AS reviewed_at,
                       sn.reviewed_name_at AS reviewed_name_at,
                       sn.reviewed_docs_at AS reviewed_docs_at,
                       sn.link_status AS link_status
                """
            )
            return [dict(r) for r in rows] if rows else []

    all_names = await asyncio.to_thread(_load_all_names)
    state.all_names = all_names
    wlog.info("Loaded %d StandardNames from graph", len(all_names))

    # --- Step 2: Compute review_input_hash for staleness detection ----------

    _compute_hash = _get_hash_fn()
    if _compute_hash is not None:
        for name in all_names:
            name["_computed_hash"] = _compute_hash(name)

    # --- Step 3: Apply filters -----------------------------------------------

    state.extract_stats.status_text = "filtering targets…"

    # IDS filter: restrict to names linked to IMASNode in that IDS
    ids_eligible: set[str] | None = None
    if state.ids_filter:
        ids_eligible = await asyncio.to_thread(_query_ids_names, state.ids_filter)
        wlog.info(
            "IDS filter %r: %d eligible names", state.ids_filter, len(ids_eligible)
        )

    targets: list[dict] = []
    for name in all_names:
        nid = name.get("id", "")

        # --ids filter
        if ids_eligible is not None and nid not in ids_eligible:
            continue

        # --domain filter
        if state.domain_filter:
            domain = name.get("physics_domain") or ""
            if domain.lower() != state.domain_filter.lower():
                continue

        # --target docs gate: skip names without prior name review
        review_target = getattr(state, "target", "full")
        if review_target == "docs":
            if name.get("reviewed_name_at") is None:
                wlog.debug("Docs gate: skipping %r — reviewed_name_at IS NULL", nid)
                state.stats["docs_skipped_missing_name"] = (
                    state.stats.get("docs_skipped_missing_name", 0) + 1
                )
                continue

        # --status filter
        if state.status_filter:
            status = name.get("pipeline_status") or "drafted"
            if status != state.status_filter:
                continue

        # --unreviewed: no score OR stale hash
        # Target-aware freshness check: for target=docs, the canonical
        # reviewer_score is bootstrapped from name-only review, so checking
        # has_score would filter out every doc-target candidate. Use the
        # target-specific axis (reviewed_docs_at) instead so the docs phase
        # can see names that have passed name-review but not docs-review yet.
        if state.unreviewed_only:
            if review_target == "docs":
                has_axis_score = name.get("reviewed_docs_at") is not None
            elif review_target == "names":
                has_axis_score = name.get("reviewed_name_at") is not None
            else:
                has_axis_score = name.get("reviewer_score") is not None
            stored_hash = name.get("review_input_hash")
            computed_hash = name.get("_computed_hash")
            is_stale = (
                computed_hash is not None
                and stored_hash is not None
                and computed_hash != stored_hash
            )
            if has_axis_score and not is_stale:
                continue

        # --force: no filtering (include already-reviewed)
        # (default: skip already-reviewed unless --force) — also target-aware
        if not state.force_review and not state.unreviewed_only:
            if review_target == "docs":
                if name.get("reviewed_docs_at") is not None:
                    continue
            elif review_target == "names":
                if name.get("reviewed_name_at") is not None:
                    continue
            else:
                if name.get("reviewer_score") is not None:
                    continue

        # Downgrade guard: don't overwrite same-axis data without --force.
        # Axis-split storage means name and docs axes are independent; only
        # a full review has strictly higher fidelity than axis-only reviews.
        if not state.force_review:
            target = getattr(state, "target", None) or (
                "names" if state.name_only else "full"
            )
            incoming = "name" if target == "names" else target
            if _axis_overwrite_blocked(name, incoming):
                continue

        targets.append(name)

    state.target_names = targets
    wlog.info(
        "Filter result: %d targets from %d total (ids=%s, domain=%s, "
        "status=%s, unreviewed=%s, force_review=%s)",
        len(targets),
        len(all_names),
        state.ids_filter,
        state.domain_filter,
        state.status_filter,
        state.unreviewed_only,
        state.force_review,
    )

    if not targets:
        wlog.info("No names match review filters — nothing to review")
        state.extract_stats.total = 0
        state.extract_stats.processed = 0
        state.extract_stats.freeze_rate()
        state.extract_phase.mark_done()
        return

    # --- Step 4: Reconstruct clusters ----------------------------------------

    state.extract_stats.status_text = "reconstructing clusters…"

    def _batch_clusters() -> dict[str, dict | None]:
        with GraphClient() as gc:
            return reconstruct_clusters_batch(targets, gc)

    clusters = await asyncio.to_thread(_batch_clusters)
    wlog.info(
        "Cluster reconstruction: %d/%d names have clusters",
        sum(1 for v in clusters.values() if v is not None),
        len(targets),
    )

    # --- Step 5: Group into batches ------------------------------------------

    state.extract_stats.status_text = "forming batches…"
    batches = group_into_review_batches(
        targets,
        clusters,
        max_batch_size=state.batch_size,
        token_budget=8000,
    )
    state.review_batches = batches

    total_items = sum(len(b["names"]) for b in batches)
    state.extract_stats.total = total_items
    state.extract_stats.processed = total_items
    state.extract_stats.record_batch(total_items)

    wlog.info(
        "Extraction complete: %d batches, %d names",
        len(batches),
        total_items,
    )
    state.stats["extract_batches"] = len(batches)
    state.stats["extract_count"] = total_items

    state.extract_stats.freeze_rate()
    state.extract_phase.mark_done()
    state.extract_stats.stream_queue.add(
        [
            {
                "primary_text": "extract",
                "description": f"{total_items} names in {len(batches)} batches",
            }
        ]
    )


# =============================================================================
# ENRICH phase — neighborhood context + audit findings
# =============================================================================


async def enrich_review_worker(state: StandardNameReviewState, **_kwargs: Any) -> None:
    """Enrich each batch with neighborhood context and audit findings.

    For each batch:
    1. Build semantic neighborhood via vector search.
    2. Attach relevant audit findings (one-line summary per finding).
    """
    from imas_codex.cli.logging import WorkerLogAdapter
    from imas_codex.standard_names.review.enrichment import (
        build_neighborhood_context,
    )

    wlog = WorkerLogAdapter(logger, worker_name="sn_review_enrich")

    batches = state.review_batches
    if not batches:
        wlog.info("No batches to enrich — skipping")
        state.enrich_stats.freeze_rate()
        state.enrich_phase.mark_done()
        return

    state.enrich_stats.total = len(batches)
    wlog.info("Enriching %d batches with neighborhood context", len(batches))

    for i, batch in enumerate(batches):
        if state.should_stop():
            wlog.info("Stop requested during enrichment at batch %d", i)
            break

        # Neighborhood context via vector search
        neighbors = await asyncio.to_thread(
            build_neighborhood_context,
            batch,
            state.all_names,
            k=state.neighborhood_k,
        )
        batch["neighborhood"] = neighbors

        # DD source docs + hybrid neighbours + related-path neighbours
        batch_names = batch.get("names", [])
        if batch_names:
            await asyncio.to_thread(_fetch_review_dd_context, batch_names)

        # Audit findings (if audit was run)
        if state.audit_report is not None:
            batch_ids = {n.get("id", "") for n in batch.get("names", [])}
            findings = _extract_audit_findings(state.audit_report, batch_ids)
            batch["audit_findings"] = findings
        else:
            batch["audit_findings"] = []

        state.enrich_stats.processed = i + 1
        state.enrich_stats.record_batch(1)

    wlog.info(
        "Enrichment complete: %d batches enriched",
        state.enrich_stats.processed,
    )
    state.enrich_stats.freeze_rate()
    state.enrich_phase.mark_done()
    state.enrich_stats.stream_queue.add(
        [
            {
                "primary_text": "enrich",
                "description": (f"{state.enrich_stats.processed} batches enriched"),
            }
        ]
    )


def _fetch_review_dd_context(names: list[dict]) -> None:
    """Enrich review items with DD source docs, hybrid neighbours, and related paths.

    For each item with ``source_paths``, fetches:

    1. **DD source docs** — IMASNode documentation for each linked DD path.
    2. **Version notes** — notable DD version changes for the paths.
    3. **Hybrid neighbours** ``[hybrid]`` — concept-similar DD paths via
       :func:`~imas_codex.standard_names.workers._hybrid_search_neighbours`.
    4. **Related neighbours** ``[related]`` — cross-IDS structural siblings
       via :func:`~imas_codex.standard_names.workers._related_path_neighbours`
       (cluster, coordinate, unit, identifier, COCOS relationships).

    Modifies items in-place, adding ``dd_source_docs``, ``version_notes``,
    ``nearest_peers``, and ``related_neighbours``.
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.workers import (
        _hybrid_search_neighbours,
        _related_path_neighbours,
    )

    # Collect all source paths across items
    all_paths: set[str] = set()
    for item in names:
        sp = item.get("source_paths")
        if sp:
            if isinstance(sp, str):
                try:
                    sp = json.loads(sp)
                except (json.JSONDecodeError, TypeError):
                    sp = [sp]
            all_paths.update(sp)

    if not all_paths:
        return

    with GraphClient() as gc:
        # Batch-fetch DD path docs
        path_docs: dict[str, dict] = {}
        try:
            rows = gc.query(
                """
                UNWIND $paths AS pid
                MATCH (n:IMASNode {id: pid})
                RETURN n.id AS id,
                       n.unit AS unit,
                       n.description AS description,
                       n.documentation AS documentation
                """,
                paths=list(all_paths),
            )
            for r in rows or []:
                path_docs[r["id"]] = {
                    "id": r["id"],
                    "unit": r.get("unit", ""),
                    "description": r.get("description", ""),
                    "documentation": r.get("documentation", ""),
                }
        except Exception:
            logger.debug("Review DD source fetch failed", exc_info=True)

        # Batch-fetch version history for all source paths
        path_versions: dict[str, list[dict]] = {}
        try:
            vrows = gc.query(
                """
                UNWIND $paths AS pid
                MATCH (vc:IMASNodeChange)-[:FOR_IMAS_PATH]->(n:IMASNode {id: pid})
                WHERE vc.change_type IN [
                    'path_added', 'cocos_transformation_type',
                    'sign_convention', 'units', 'path_renamed',
                    'definition_clarification'
                ]
                RETURN n.id AS path, vc.id AS change_id,
                       vc.change_type AS change_type
                """,
                paths=list(all_paths),
            )
            for vr in vrows or []:
                vpath = vr["path"]
                change_id = vr.get("change_id") or ""
                change_type = vr.get("change_type") or ""
                parts = change_id.rsplit(":", 1)
                version = parts[-1] if len(parts) >= 2 else ""
                if version and change_type:
                    path_versions.setdefault(vpath, []).append(
                        {"version": version, "change_type": change_type}
                    )
        except Exception:
            logger.debug("Review version history fetch failed", exc_info=True)

        # Enrich each item
        for item in names:
            sp = item.get("source_paths")
            if sp:
                if isinstance(sp, str):
                    try:
                        sp = json.loads(sp)
                    except (json.JSONDecodeError, TypeError):
                        sp = [sp]

                docs = [path_docs[p] for p in sp if p in path_docs]
                if docs:
                    item["dd_source_docs"] = docs

                    # Version notes from DD path changes
                    vnotes: list[dict] = []
                    for p in sp:
                        vnotes.extend(path_versions.get(p, []))
                    if vnotes:
                        item["version_notes"] = vnotes

                    # Hybrid-search nearest peers [hybrid]
                    try:
                        peers = _hybrid_search_neighbours(
                            gc,
                            sp[0],
                            description=item.get("description"),
                            physics_domain=item.get("physics_domain"),
                            max_results=5,
                        )
                        if peers:
                            item["nearest_peers"] = peers
                    except Exception:
                        logger.debug(
                            "Review hybrid search failed for %s",
                            item.get("id"),
                            exc_info=True,
                        )

                    # Related-path neighbours [related]
                    try:
                        related = _related_path_neighbours(gc, sp[0], max_results=5)
                        if related:
                            item["related_neighbours"] = related
                    except Exception:
                        logger.debug(
                            "Review related-path search failed for %s",
                            item.get("id"),
                            exc_info=True,
                        )


# =============================================================================
# REVIEW phase — LLM quality scoring with budget management
# =============================================================================


async def review_review_worker(state: StandardNameReviewState, **_kwargs: Any) -> None:
    """Run LLM quality review on each enriched batch.

    Reuses the existing ``_review_batch()`` pattern from ``workers.py``
    with additions:
    - Budget management (reserve/reconcile)
    - Neighborhood context in prompts
    - Audit findings in prompts
    """
    from imas_codex.cli.logging import WorkerLogAdapter
    from imas_codex.settings import get_model

    wlog = WorkerLogAdapter(logger, worker_name="sn_review_review")

    batches = state.review_batches
    if not batches:
        wlog.info("No batches to review — skipping")
        state.review_stats.freeze_rate()
        state.review_phase.mark_done()
        return

    model = state.review_model or get_model("language")
    wlog.info("Reviewing %d batches (model=%s)", len(batches), model)

    # Shared context for prompt rendering
    grammar_enums = _get_grammar_enums()
    compose_ctx = _get_compose_context_for_review()

    # --- K3: Load scored examples for review calibration ---
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.example_loader import load_review_examples

    # Derive physics_domains from domain_filter or batch items
    review_domains: list[str] = []
    if state.domain_filter:
        review_domains = [state.domain_filter]
    else:
        _domains = {
            n.get("physics_domain")
            for batch in batches
            for n in batch.get("names", [])
            if n.get("physics_domain")
        }
        review_domains = sorted(_domains)

    # Derive axis from review target: name_only/names → "name",
    # docs → "docs", full → "full"
    _review_target = getattr(state, "target", None) or (
        "names" if state.name_only else "full"
    )
    if _review_target in ("names", "name"):
        _review_axis: Literal["name", "docs", "full"] = "name"
    elif _review_target == "docs":
        _review_axis = "docs"
    else:
        _review_axis = "full"

    def _load_review_scored() -> list[dict]:
        with GraphClient() as gc:
            return load_review_examples(
                gc, physics_domains=review_domains, axis=_review_axis
            )

    review_scored = await asyncio.to_thread(_load_review_scored)
    if review_scored:
        wlog.info(
            "K3: Loaded %d scored examples for review (domains=%s)",
            len(review_scored),
            review_domains or "all",
        )

    total_items = sum(len(b["names"]) for b in batches)
    state.review_stats.total = total_items

    sem = asyncio.Semaphore(state.concurrency)
    scored: list[dict] = []
    review_records: list[dict] = []
    total_cost = 0.0
    total_tokens = 0
    errors = 0
    revised = 0
    unscored = 0

    async def _process_batch(
        batch_idx: int, batch: dict
    ) -> tuple[list[dict], list[dict]]:
        nonlocal total_cost, total_tokens, errors, revised, unscored

        async with sem:
            if state.should_stop():
                return [], []

            names = batch.get("names", [])
            if not names:
                return [], []

            # Estimate cost and reserve budget
            estimated_cost = len(names) * 0.002  # ~$0.002 per name heuristic
            reserved = estimated_cost * 1.3
            actual_cost = 0.0

            if state.budget_manager:
                if not state.budget_manager.reserve(estimated_cost):
                    wlog.info(
                        "Budget exhausted at batch %d — stopping review",
                        batch_idx,
                    )
                    return [], []

            try:
                batch_scored = await _review_single_batch(
                    names=names,
                    model=model,
                    grammar_enums=grammar_enums,
                    compose_ctx=compose_ctx,
                    batch_context=batch.get("group_key", ""),
                    neighborhood=batch.get("neighborhood", []),
                    audit_findings=batch.get("audit_findings", []),
                    wlog=wlog,
                    name_only=state.name_only,
                    target=getattr(state, "target", None)
                    or ("names" if state.name_only else "full"),
                    review_scored_examples=review_scored,
                )
                batch_cost = batch_scored.pop("_cost", 0.0)
                batch_tokens = batch_scored.pop("_tokens", 0)
                batch_input_tokens = batch_scored.pop("_input_tokens", 0)
                batch_output_tokens = batch_scored.pop("_output_tokens", 0)
                batch_revised = batch_scored.pop("_revised", 0)
                batch_items = batch_scored.pop("_items", [])
                batch_unscored = batch_scored.pop("_unscored", 0)

                actual_cost = batch_cost
                total_cost += batch_cost
                total_tokens += batch_tokens
                revised += batch_revised
                unscored += batch_unscored

                # Only count actually scored items as processed
                actually_scored = len(batch_items)
                state.review_stats.cost += batch_cost
                state.review_stats.processed += actually_scored
                state.review_stats.record_batch(actually_scored)

                # Stream progress — distinguish scored from unscored
                desc = f"{actually_scored} scored  ${batch_cost:.3f}"
                if batch_unscored > 0:
                    desc += f"  ({batch_unscored} unscored)"
                state.review_stats.stream_queue.add(
                    [
                        {
                            "primary_text": f"batch {batch_idx + 1}",
                            "description": desc,
                        }
                    ]
                )
                # --- Build Review records for each configured model -----------
                batch_review_records: list[dict] = []
                models = state.review_models or (
                    [state.review_model] if state.review_model else []
                )
                if models and batch_items:
                    # Canonical model (models[0]) — items already scored above.
                    # Amortize batch-level cost/tokens across items so each
                    # Review node records its fair share.
                    canonical_ts = datetime.now(UTC).isoformat()
                    n_items = max(len(batch_items), 1)
                    per_cost = batch_cost / n_items
                    per_in = batch_input_tokens // n_items if batch_input_tokens else 0
                    per_out = (
                        batch_output_tokens // n_items if batch_output_tokens else 0
                    )
                    for item in batch_items:
                        rec = _build_review_record(
                            item,
                            model=models[0],
                            is_canonical=True,
                            reviewed_at=canonical_ts,
                            cost_usd=per_cost,
                            tokens_in=per_in,
                            tokens_out=per_out,
                        )
                        if rec:
                            batch_review_records.append(rec)

                    # Additional models — deep-copy canonical items, re-review
                    import copy as _copy

                    for sec_model in models[1:]:
                        try:
                            sec_input = _copy.deepcopy(batch_items)
                            sec_result = await _review_single_batch(
                                names=sec_input,
                                model=sec_model,
                                grammar_enums=grammar_enums,
                                compose_ctx=compose_ctx,
                                batch_context=batch.get("group_key", ""),
                                neighborhood=batch.get("neighborhood", []),
                                audit_findings=batch.get("audit_findings", []),
                                wlog=wlog,
                                name_only=state.name_only,
                                target=getattr(state, "target", None)
                                or ("names" if state.name_only else "full"),
                                review_scored_examples=review_scored,
                            )
                            sec_items = sec_result.get("_items", [])
                            sec_cost = sec_result.get("_cost", 0.0)
                            sec_input_tokens = sec_result.get("_input_tokens", 0)
                            sec_output_tokens = sec_result.get("_output_tokens", 0)
                            total_cost += sec_cost
                            state.review_stats.cost += sec_cost

                            sec_ts = datetime.now(UTC).isoformat()
                            sec_by_id = {s["id"]: s for s in sec_items if "id" in s}
                            n_sec = max(len(sec_items), 1)
                            sec_per_cost = sec_cost / n_sec
                            sec_per_in = (
                                sec_input_tokens // n_sec if sec_input_tokens else 0
                            )
                            sec_per_out = (
                                sec_output_tokens // n_sec if sec_output_tokens else 0
                            )
                            for item in batch_items:
                                sec = sec_by_id.get(item.get("id", ""))
                                if sec is None:
                                    continue
                                rec = _build_review_record(
                                    sec,
                                    model=sec_model,
                                    is_canonical=False,
                                    reviewed_at=sec_ts,
                                    cost_usd=sec_per_cost,
                                    tokens_in=sec_per_in,
                                    tokens_out=sec_per_out,
                                )
                                if rec:
                                    batch_review_records.append(rec)
                        except Exception:
                            wlog.warning(
                                "Additional reviewer %s failed for batch %d — skipped",
                                sec_model,
                                batch_idx,
                                exc_info=True,
                            )

                return batch_items, batch_review_records

            except Exception:
                wlog.debug(
                    "Review batch %d failed",
                    batch_idx,
                    exc_info=True,
                )
                errors += len(names)
                unscored += len(names)
                state.review_stats.errors += len(names)
                # Do NOT pass through unscored entries — they lack scores
                return [], []

            finally:
                if state.budget_manager:
                    state.budget_manager.reconcile(reserved, actual_cost)

    tasks = [_process_batch(i, batch) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, tuple):
            items, recs = r
            scored.extend(items)
            review_records.extend(recs)
        elif isinstance(r, Exception):
            wlog.warning("Batch task failed: %s", r)

    state.review_results = scored

    # Determine canonical model from review_models or fallback to review_model
    models_list = state.review_models or (
        [state.review_model] if state.review_model else []
    )
    state.canonical_review_model = models_list[0] if models_list else None
    state.review_records = review_records

    wlog.info(
        "Review complete: %d scored, %d unscored, %d revised, %d errors "
        "(cost=$%.4f, tokens=%d)",
        len(scored),
        unscored,
        revised,
        errors,
        total_cost,
        total_tokens,
    )
    state.stats["review_scored"] = len(scored)
    state.stats["review_unscored"] = unscored
    state.stats["review_revised"] = revised
    state.stats["review_errors"] = errors
    state.stats["review_cost"] = total_cost

    state.review_stats.freeze_rate()
    state.review_phase.mark_done()


# =============================================================================
# PERSIST phase — write review scores back to graph
# =============================================================================


async def persist_review_worker(state: StandardNameReviewState, **_kwargs: Any) -> None:
    """Persist review results to graph.

    For each reviewed name:
    1. Compute fresh ``review_input_hash``.
    2. Add ``reviewed_at`` timestamp and ``reviewer_model``.
    3. Write via ``write_standard_names()``.
    4. Write Review nodes via ``write_reviews()``.
    5. Update aggregates (review_count, review_mean_score, review_disagreement).
    """
    from imas_codex.cli.logging import WorkerLogAdapter
    from imas_codex.standard_names.graph_ops import (
        update_review_aggregates,
        write_review_results,
        write_reviews,
    )

    wlog = WorkerLogAdapter(logger, worker_name="sn_review_persist")

    results = state.review_results
    if not results:
        wlog.info("No review results to persist — skipping")
        state.persist_stats.freeze_rate()
        state.persist_phase.mark_done()
        return

    state.persist_stats.total = len(results)
    wlog.info("Persisting %d review results", len(results))

    # Stamp provenance
    model = state.review_model or "unknown"
    reviewed_at = datetime.now(UTC).isoformat()
    _compute_hash = _get_hash_fn()

    for entry in results:
        entry["reviewer_model"] = model
        entry["reviewed_at"] = reviewed_at

        # Compute fresh review_input_hash
        if _compute_hash is not None:
            entry["review_input_hash"] = _compute_hash(entry)

    # Determine review mode from state.target
    review_target = getattr(state, "target", "full")
    _target_to_mode = {"names": "name", "docs": "docs", "full": "full"}
    write_mode = _target_to_mode.get(review_target, "full")

    # Write canonical scores to StandardName nodes using mode-aware writer
    def _write() -> int:
        if write_mode in ("name", "docs"):
            return write_review_results(results, write_mode, stats=state.stats)
        # Full mode: write via write_review_results for 3-score slot support
        return write_review_results(results, "full", stats=state.stats)

    written = await asyncio.to_thread(_write)

    state.persist_stats.processed = written
    state.persist_stats.record_batch(written)

    # Write Review nodes
    review_records = state.review_records or []
    if review_records:

        def _write_reviews() -> int:
            return write_reviews(review_records)

        reviews_written = await asyncio.to_thread(_write_reviews)
        wlog.info("Wrote %d Review nodes", reviews_written)
        state.stats["review_nodes_written"] = reviews_written
    else:
        state.stats["review_nodes_written"] = 0

    # Update aggregates on reviewed names
    reviewed_ids = [r["id"] for r in results if r.get("id")]
    if reviewed_ids:

        def _update_agg() -> None:
            update_review_aggregates(
                reviewed_ids, threshold=state.disagreement_threshold
            )

        await asyncio.to_thread(_update_agg)

    wlog.info("Persisted %d/%d review results", written, len(results))
    state.stats["persist_count"] = written

    state.persist_stats.freeze_rate()
    state.persist_phase.mark_done()
    state.persist_stats.stream_queue.add(
        [
            {
                "primary_text": "persist",
                "description": f"{written} names updated",
            }
        ]
    )


# =============================================================================
# Pipeline entry point
# =============================================================================


async def run_sn_review_engine(
    state: StandardNameReviewState,
    *,
    stop_event: asyncio.Event | None = None,
    on_worker_status: Any | None = None,
) -> None:
    """Run the SN review pipeline.

    Pipeline::

        EXTRACT → ENRICH → REVIEW → PERSIST

    Extract loads and filters StandardNames, reconstructs clusters, batches.
    Enrich adds neighborhood context and audit findings.
    Review runs LLM quality scoring with budget management.
    Persist writes review scores back to the graph.

    Args:
        state: Populated ``StandardNameReviewState`` with filters and config.
        stop_event: Optional asyncio.Event for CLI shutdown signalling.
        on_worker_status: Optional callback for progress display updates.
    """

    # Persist must run to completion even when the review worker exhausts
    # budget. Only CLI shutdown (stop_requested) should stop it.
    def _downstream_should_stop() -> bool:
        return state.stop_requested

    workers = [
        WorkerSpec(
            "extract",
            "extract_phase",
            extract_review_worker,
        ),
        WorkerSpec(
            "enrich",
            "enrich_phase",
            enrich_review_worker,
            depends_on=["extract_phase"],
        ),
        WorkerSpec(
            "review",
            "review_phase",
            review_review_worker,
            depends_on=["enrich_phase"],
        ),
        WorkerSpec(
            "persist",
            "persist_phase",
            persist_review_worker,
            depends_on=["review_phase"],
            should_stop_fn=_downstream_should_stop,
        ),
    ]

    # The supervised loop must not exit on budget exhaustion — only on CLI
    # shutdown. Budget control happens inside the review worker; persist must
    # run to completion after review finishes.
    await run_discovery_engine(
        state,
        workers,
        stop_event=stop_event,
        on_worker_status=on_worker_status,
        stop_fn=lambda: state.stop_requested,
    )


# =============================================================================
# Internal helpers
# =============================================================================


def _get_hash_fn():
    """Return ``compute_review_input_hash`` or ``None`` if unavailable.

    The audits module (Layer 1) provides this function.  If it hasn't
    been created yet, we gracefully degrade — names are still reviewed
    but staleness detection won't work.
    """
    try:
        from imas_codex.standard_names.review.audits import compute_review_input_hash

        return compute_review_input_hash
    except (ImportError, ModuleNotFoundError):
        logger.debug("compute_review_input_hash not available (audits not yet created)")
        return None


def _query_ids_names(ids_prefix: str) -> set[str]:
    """Return StandardName ids linked to IMASNodes in a given IDS."""
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName)<-[:HAS_STANDARD_NAME]-(node:IMASNode)
            WHERE node.id STARTS WITH $ids_prefix
            RETURN DISTINCT sn.id AS id
            """,
            ids_prefix=ids_prefix + "/",
        )
        return {r["id"] for r in rows} if rows else set()


def _extract_audit_findings(audit_report: Any, batch_ids: set[str]) -> list[str]:
    """Extract one-line audit findings relevant to names in the batch.

    Processes the AuditReport from Layer 1, extracting per-name findings
    and capping verbosity.
    """
    findings: list[str] = []
    if audit_report is None:
        return findings

    # AuditReport has .findings: list of AuditFinding
    for finding in getattr(audit_report, "findings", []):
        affected = getattr(finding, "affected_names", []) or []
        # Check if any affected name is in this batch
        overlap = [nid for nid in affected if nid in batch_ids]
        if overlap:
            severity = getattr(finding, "severity", "info")
            message = getattr(finding, "message", str(finding))
            category = getattr(finding, "category", "general")
            # One-line summary, capped
            summary = f"[{severity}:{category}] {message[:200]}"
            findings.append(summary)

    # Cap total findings per batch
    return findings[:20]


def _match_reviews_to_entries(
    reviews: list,
    names: list[dict],
    wlog: logging.LoggerAdapter,
    name_only: bool = False,
    target: str | None = None,
) -> tuple[list[dict], list[dict], int]:
    """Match LLM review results to original entries.

    Returns ``(scored, unmatched, revised_count)`` where *scored* contains
    entries with review scores applied and *unmatched* contains entries that
    the LLM did not return a review for.

    ``target`` selects the ``review_mode`` stamped onto each scored entry:
    ``"names"`` → ``"names"``, ``"docs"`` → ``"docs"``, ``"full"`` →
    ``"full"``. When ``target`` is None, falls back to the legacy
    ``name_only`` boolean for back-compat.
    """
    from imas_codex.standard_names.models import StandardNameReviewVerdict

    if target is None:
        target = "names" if name_only else "full"
    review_mode = {
        "names": "names",
        "docs": "docs",
        "full": "full",
    }.get(target, "full")

    # Build lookup: source_id → entry, id → entry
    entry_map: dict[str, dict] = {}
    for entry in names:
        sid = entry.get("source_id") or entry.get("id") or ""
        entry_map[sid] = entry
        name_id = entry.get("id") or ""
        if name_id and name_id != sid:
            entry_map[name_id] = entry

    scored: list[dict] = []
    matched_entries: set[str] = set()  # track by entry id
    revised_count = 0

    for review in reviews:
        original = entry_map.get(review.source_id) or entry_map.get(
            review.standard_name
        )
        if original is None:
            wlog.debug(
                "Review returned unknown source_id=%s / standard_name=%s",
                review.source_id,
                review.standard_name,
            )
            continue

        entry_id = original.get("id") or ""
        matched_entries.add(entry_id)

        # Store review scores on the entry
        original["reviewer_score"] = review.scores.score
        original["reviewer_scores"] = json.dumps(review.scores.model_dump())
        original["reviewer_comments"] = review.reasoning
        original["review_tier"] = review.scores.tier
        if hasattr(review, "comments") and review.comments is not None:
            original["reviewer_comments_per_dim"] = json.dumps(
                review.comments.model_dump()
            )
        original["reviewer_verdict"] = review.verdict.value
        original["review_mode"] = review_mode

        # Review writes score/tier/comments but does NOT demote
        # validation_status. Regeneration targeting is driven by the
        # ``sn run --min-score F`` threshold instead, which selects names
        # purely by reviewer_score without touching lifecycle state.
        if review.verdict == StandardNameReviewVerdict.revise and getattr(
            review, "revised_name", None
        ):
            if not _valid_sn_id(review.revised_name):
                # Reviewer hallucinated garbage into revised_name (e.g. embedded
                # stream-of-consciousness reasoning). Keep original and log.
                wlog.warning(
                    "Rejecting malformed revised_name (len=%d) for %r — keeping original",
                    len(review.revised_name) if review.revised_name else 0,
                    review.standard_name,
                )
                scored.append(original)
                continue
            revised_entry = dict(original)
            revised_entry["id"] = review.revised_name
            if review.revised_fields:
                for key, value in review.revised_fields.items():
                    if key in revised_entry:
                        revised_entry[key] = value
            scored.append(revised_entry)
            revised_count += 1
            wlog.debug(
                "Revised %r → %r (score %.2f): %s",
                review.standard_name,
                review.revised_name,
                review.scores.score,
                review.reasoning[:120],
            )
        else:
            scored.append(original)
            if review.verdict == StandardNameReviewVerdict.reject:
                wlog.debug(
                    "Low score %r (%.2f, %s): %s",
                    review.standard_name,
                    review.scores.score,
                    review.scores.tier,
                    review.reasoning[:120],
                )

    # Identify entries that were NOT matched to any review
    unmatched: list[dict] = []
    for entry in names:
        entry_id = entry.get("id") or ""
        if entry_id not in matched_entries:
            unmatched.append(entry)

    return scored, unmatched, revised_count


async def _review_single_batch(
    *,
    names: list[dict],
    model: str,
    grammar_enums: dict[str, list[str]],
    compose_ctx: dict[str, Any],
    batch_context: str,
    neighborhood: list[dict],
    audit_findings: list[str],
    wlog: logging.LoggerAdapter,
    name_only: bool = False,
    target: str | None = None,
    _is_retry: bool = False,
    review_scored_examples: list[dict] | None = None,
) -> dict[str, Any]:
    """Review a single batch via LLM — mirrors ``_review_batch()`` from workers.py.

    Returns a dict with keys ``_items`` (scored entries), ``_cost``,
    ``_tokens``, ``_revised``, and ``_unscored`` (count of entries that
    could not be matched to an LLM review result).

    When ``_is_retry`` is False and some entries are unmatched, a single
    retry is attempted with only the unmatched entries.  Entries that
    remain unmatched after retry are **not** included in ``_items`` —
    they are counted in ``_unscored`` so callers never report unscored
    entries as reviewed.

    ``target`` selects the rubric:

    * ``"names"`` → ``sn/review_names`` prompt +
      ``StandardNameQualityReviewNameOnlyBatch`` (4 dims, /80).
    * ``"docs"`` → ``sn/review_docs`` prompt +
      ``StandardNameQualityReviewDocsBatch`` (4 docs dims, /80).
    * ``"full"`` → ``sn/review`` prompt + ``StandardNameQualityReviewBatch``
      (6 dims, /120).

    When ``target`` is None, the legacy ``name_only`` boolean selects
    between ``"names"`` and ``"full"`` for back-compat.
    """
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.models import (
        StandardNameQualityReviewBatch,
        StandardNameQualityReviewDocsBatch,
        StandardNameQualityReviewNameOnlyBatch,
    )

    if target is None:
        target = "names" if name_only else "full"

    if target == "names":
        prompt_name = "sn/review_names"
        response_model: type = StandardNameQualityReviewNameOnlyBatch
    elif target == "docs":
        prompt_name = "sn/review_docs"
        response_model = StandardNameQualityReviewDocsBatch
    else:
        prompt_name = "sn/review"
        response_model = StandardNameQualityReviewBatch

    # Enrich items with validation issues for reviewer context
    items_with_issues = []
    for item in names:
        item_data = dict(item)
        item_data["validation_issues"] = item.get("validation_issues", [])
        items_with_issues.append(item_data)

    # Merge compose context with review-specific keys
    base_ctx = dict(compose_ctx) if compose_ctx else {}
    _scored_examples = review_scored_examples if review_scored_examples else []

    # Build context for prompt rendering
    context = {
        **base_ctx,
        "items": items_with_issues,
        "existing_names": [],  # not needed for standalone review
        "review_scored_examples": _scored_examples,
        "batch_context": batch_context,
        "nearby_existing_names": neighborhood,
        "audit_findings": audit_findings,
        **grammar_enums,
    }

    # System prompt: rubric (cached across batches)
    system_context = {
        **base_ctx,
        "items": [],
        "existing_names": [],
        "review_scored_examples": _scored_examples,
        "batch_context": "",
        "nearby_existing_names": [],
        "audit_findings": [],
        **grammar_enums,
    }
    system_prompt = render_prompt(prompt_name, system_context)

    # User prompt: actual candidates + context
    user_prompt = render_prompt(prompt_name, context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    llm_out = await acall_llm_structured(
        model=model,
        messages=messages,
        response_model=response_model,
        service="standard-names",
    )
    result, cost, tokens = llm_out

    # --- Match reviews to original entries -----------------------------------
    scored, unmatched, revised_count = _match_reviews_to_entries(
        result.reviews, names, wlog, name_only=name_only, target=target
    )

    total_cost = cost
    total_tokens = tokens
    # llm_out is an LLMResult-like object carrying .input_tokens/.output_tokens;
    # tolerate legacy tuple returns (some tests mock acall_llm_structured as a
    # plain tuple) by falling back to 0.
    total_input_tokens = getattr(llm_out, "input_tokens", 0) or 0
    total_output_tokens = getattr(llm_out, "output_tokens", 0) or 0
    unscored_count = 0

    # --- Retry unmatched entries (once) --------------------------------------
    if unmatched and not _is_retry:
        unmatched_ids = [e.get("id", "?") for e in unmatched]
        logger.warning(
            "Batch incomplete: %d/%d entries unmatched after LLM review "
            "(unmatched: %s). Retrying unmatched only.",
            len(unmatched),
            len(names),
            ", ".join(unmatched_ids[:10]) + ("…" if len(unmatched_ids) > 10 else ""),
        )

        retry_result = await _review_single_batch(
            names=unmatched,
            model=model,
            grammar_enums=grammar_enums,
            compose_ctx=compose_ctx,
            batch_context=batch_context,
            neighborhood=neighborhood,
            audit_findings=audit_findings,
            wlog=wlog,
            name_only=name_only,
            target=target,
            _is_retry=True,
            review_scored_examples=review_scored_examples,
        )

        scored.extend(retry_result.get("_items", []))
        total_cost += retry_result.get("_cost", 0.0)
        total_tokens += retry_result.get("_tokens", 0)
        total_input_tokens += retry_result.get("_input_tokens", 0)
        total_output_tokens += retry_result.get("_output_tokens", 0)
        revised_count += retry_result.get("_revised", 0)
        unscored_count = retry_result.get("_unscored", 0)
    elif unmatched:
        # Retry already attempted — log remaining unmatched as unscored
        unmatched_ids = [e.get("id", "?") for e in unmatched]
        logger.warning(
            "Retry failed: %d entries still unscored after retry "
            "(ids: %s). NOT generating synthetic scores.",
            len(unmatched),
            ", ".join(unmatched_ids[:10]) + ("…" if len(unmatched_ids) > 10 else ""),
        )
        unscored_count = len(unmatched)

    return {
        "_items": scored,
        "_cost": total_cost,
        "_tokens": total_tokens,
        "_input_tokens": total_input_tokens,
        "_output_tokens": total_output_tokens,
        "_revised": revised_count,
        "_unscored": unscored_count,
    }


# ---------------------------------------------------------------------------
# Reused helpers from workers.py (avoid circular imports)
# ---------------------------------------------------------------------------


def _get_grammar_enums() -> dict[str, list[str]]:
    """Return grammar enum values for prompt context."""
    try:
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
            "components": [e.value for e in Component],
            "coordinates": [e.value for e in Component],
            "positions": [e.value for e in Position],
            "processes": [e.value for e in Process],
            "transformations": [e.value for e in Transformation],
            "geometric_bases": [e.value for e in GeometricBase],
            "objects": [e.value for e in Object],
            "binary_operators": [e.value for e in BinaryOperator],
        }
    except (ImportError, ModuleNotFoundError):
        logger.debug("imas_standard_names.grammar not available")
        return {}


def _get_compose_context_for_review() -> dict[str, Any]:
    """Return compose context keys needed by shared prompt includes."""
    try:
        from imas_codex.standard_names.context import build_compose_context

        return build_compose_context()
    except Exception:
        logger.debug("build_compose_context unavailable", exc_info=True)
        return {}

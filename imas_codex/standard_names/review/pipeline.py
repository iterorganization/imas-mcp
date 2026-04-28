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
from imas_codex.standard_names.budget import LLMCostEvent

# Defense-in-depth: strict SN id pattern used to reject reviewer-hallucinated
# revised_name values (e.g. multi-hundred-char stream-of-consciousness strings).
_SN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_SN_ID_MAX_LEN = 120


def _charge_review_cycle(
    lease,
    cost: float,
    result_dict: dict,
    model: str,
    items: list[dict],
    group_key: str,
    cycle: str,
    phase: str,
    loop_stats: Any | None = None,
) -> None:
    """Create charge event(s) for a review cycle (primary + optional retry)."""
    primary_cost = result_dict.get("_primary_cost", cost)
    retry_cost = cost - primary_cost

    _event = LLMCostEvent(
        model=model,
        tokens_in=result_dict.get(
            "_primary_input_tokens", result_dict.get("_input_tokens", 0)
        ),
        tokens_out=result_dict.get(
            "_primary_output_tokens", result_dict.get("_output_tokens", 0)
        ),
        sn_ids=tuple(item.get("id", "") for item in items),
        batch_id=group_key,
        cycle=cycle,
        phase=phase,
        service="standard-names",
    )
    lease.charge_event(primary_cost, _event)
    if loop_stats is not None:
        loop_stats.processed += 1
        loop_stats.cost += primary_cost
        _plabel = f"sn={_event.sn_ids[0]}" if _event.sn_ids else f"batch={group_key}"
        loop_stats.stream_queue.add(
            [
                {
                    "primary_text": _plabel,
                    "primary_text_style": "white",
                    "description": group_key,
                }
            ]
        )

    if retry_cost > 0:
        _retry = LLMCostEvent(
            model=model,
            tokens_in=max(
                result_dict.get("_input_tokens", 0)
                - result_dict.get("_primary_input_tokens", 0),
                0,
            ),
            tokens_out=max(
                result_dict.get("_output_tokens", 0)
                - result_dict.get("_primary_output_tokens", 0),
                0,
            ),
            sn_ids=tuple(item.get("id", "") for item in items),
            batch_id=f"{group_key}-retry",
            cycle=cycle,
            phase=phase,
            service="standard-names",
        )
        lease.charge_event(retry_cost, _retry)
        if loop_stats is not None:
            loop_stats.cost += retry_cost
            loop_stats.stream_queue.add(
                [
                    {
                        "primary_text": f"batch={group_key}-retry",
                        "primary_text_style": "white",
                        "description": group_key,
                    }
                ]
            )


def _axis_overwrite_blocked(
    name: dict,
    incoming: str,
) -> bool:
    """Return True if writing ``incoming`` axis would clobber existing data.

    Axis-split storage rules:
      * same-axis presence blocks same-axis re-write (``reviewer_scores_name``
        blocks a new name-axis review; ``reviewer_scores_docs`` blocks a new
        docs-axis review)

    Callers bypass this guard with ``--force``.
    """
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
    sn_id = item.get("_original_id") or item.get("id") or ""
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
        # reviewer_model is the consumer-facing alias for model — kept in sync
        # so that queries on rv.reviewer_model return the expected value.
        "reviewer_model": model,
        "model_family": _derive_model_family(model),
        "is_canonical": bool(is_canonical),
        "score": float(
            score if score is not None else (item.get("reviewer_score") or 0.0)
        ),
        "scores_json": scores_json,
        "tier": (tier if tier is not None else item.get("review_tier")) or "unknown",
        # verdict is the accept/reject/revise decision from the LLM reviewer.
        "verdict": item.get("reviewer_verdict") or "",
        "comments": (
            comments if comments is not None else item.get("reviewer_comments")
        )
        or "",
        "comments_per_dim_json": (
            json.dumps(comments_per_dim)
            if comments_per_dim
            else item.get("reviewer_comments_per_dim")
        ),
        "suggested_name": item.get("_suggested_name") or "",
        "suggestion_justification": item.get("_suggestion_justification") or "",
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
                       sn.reviewer_scores_name AS reviewer_scores_name,
                       sn.reviewer_scores_docs AS reviewer_scores_docs,
                       sn.review_input_hash AS review_input_hash,
                       sn.embedding AS embedding,
                       sn.review_tier AS review_tier,
                       sn.source_types AS source_types,
                       sn.source_id AS source_id,
                       sn.generated_at AS generated_at,
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
        review_target = getattr(state, "target", "names")
        if review_target == "docs":
            if name.get("reviewed_name_at") is None:
                wlog.debug("Docs gate: skipping %r — reviewed_name_at IS NULL", nid)
                state.stats["docs_skipped_missing_name"] = (
                    state.stats.get("docs_skipped_missing_name", 0) + 1
                )
                continue
            # Content gate: docs review of empty/stub content scores 0/80
            # and pollutes aggregate stats. Require minimum content before
            # the LLM is asked to grade documentation quality.
            doc_text = (name.get("documentation") or "").strip()
            desc_text = (name.get("description") or "").strip()
            if len(doc_text) < 50 and len(desc_text) < 20:
                wlog.debug(
                    "Docs gate: skipping %r — insufficient content "
                    "(doc=%d, desc=%d chars)",
                    nid,
                    len(doc_text),
                    len(desc_text),
                )
                state.stats["docs_skipped_empty_content"] = (
                    state.stats.get("docs_skipped_empty_content", 0) + 1
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
            else:
                has_axis_score = name.get("reviewed_name_at") is not None
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
            else:
                if name.get("reviewed_name_at") is not None:
                    continue

        # Downgrade guard: don't overwrite same-axis data without --force.
        # Axis-split storage means name and docs axes are independent; only
        # a full review has strictly higher fidelity than axis-only reviews.
        if not state.force_review:
            target = getattr(state, "target", None) or "names"
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
        _enrich_batch_items,
        _hybrid_search_neighbours_batch,
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

        # Enrich each item — first pass: non-hybrid enrichment
        # Collect data for batch hybrid search
        _hybrid_batch_tuples: list[tuple[str, str | None, str | None]] = []
        _hybrid_batch_item_indices: list[int] = []

        for item_idx, item in enumerate(names):
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

                    # Queue for batch hybrid search
                    _hybrid_batch_tuples.append(
                        (sp[0], item.get("description"), item.get("physics_domain"))
                    )
                    _hybrid_batch_item_indices.append(item_idx)

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

        # Batch hybrid-search nearest peers (single embed round-trip)
        if _hybrid_batch_tuples:
            try:
                _hybrid_results = _hybrid_search_neighbours_batch(
                    gc, _hybrid_batch_tuples, max_results=5
                )
                for bi, hr in zip(
                    _hybrid_batch_item_indices, _hybrid_results, strict=True
                ):
                    if hr:
                        names[bi]["nearest_peers"] = hr
            except Exception:
                logger.debug("Review batch hybrid search failed", exc_info=True)

    # Reuse compose's per-item enrichment so the reviewer sees the SAME 11
    # context channels the composer received: identifier_schema/values,
    # clusters, cross_ids_paths, sibling_fields, error_fields, hybrid_neighbours,
    # related_neighbours, parent_description, version_history, COCOS, etc.
    # Build stub items keyed on the primary source_path, then merge back.
    _CONTEXT_KEYS = (
        "coordinate_paths",
        "timebase",
        "cocos_label",
        "cocos_expression",
        "lifecycle_status",
        "identifier_schema",
        "identifier_schema_doc",
        "identifier_values",
        "sibling_fields",
        "clusters",
        "cross_ids_paths",
        "version_history",
        "hybrid_neighbours",
        "related_neighbours",
        "error_fields",
        "parent_path",
        "parent_description",
    )
    stubs: list[dict] = []
    stub_by_id: dict[str, dict] = {}
    for item in names:
        sp = item.get("source_paths")
        if not sp:
            continue
        if isinstance(sp, str):
            try:
                sp = json.loads(sp)
            except (json.JSONDecodeError, TypeError):
                sp = [sp]
        if not sp:
            continue
        stub = {
            "path": sp[0],
            "description": item.get("description"),
            "physics_domain": item.get("physics_domain"),
        }
        stubs.append(stub)
        stub_by_id[item.get("id") or ""] = stub

    if stubs:
        try:
            _enrich_batch_items(stubs)
        except Exception:
            logger.debug("Review compose-parity enrichment failed", exc_info=True)

        for item in names:
            stub = stub_by_id.get(item.get("id") or "")
            if not stub:
                continue
            for key in _CONTEXT_KEYS:
                # Don't clobber values already populated by the review-specific
                # enrichment above (e.g. nearest_peers/related_neighbours from
                # _hybrid_search_neighbours which uses different defaults).
                if key in stub and key not in item:
                    item[key] = stub[key]


# =============================================================================
# REVIEW phase — LLM quality scoring with budget management
# =============================================================================


async def review_review_worker(state: StandardNameReviewState, **_kwargs: Any) -> None:
    """Run LLM quality review on each enriched batch using RD-quorum protocol.

    RD-quorum tight loop per batch:
      cycle 0: primary (blind) — scores the full batch
      cycle 1: secondary (blind) — scores the same batch independently
      cycle 2: (optional) escalator — sees BOTH prior critiques and
               resolves disputed items only (per-item mini-batch)

    Budget management: one lease per batch, reserving worst-case cost
    upfront and charging each cycle's actual cost.
    """
    import copy as _copy
    import uuid as _uuid

    from imas_codex.cli.logging import WorkerLogAdapter
    from imas_codex.settings import (
        get_sn_review_disagreement_threshold,
        get_sn_review_docs_models,
        get_sn_review_max_cycles,
        get_sn_review_names_models,
    )

    wlog = WorkerLogAdapter(logger, worker_name="sn_review_review")

    batches = state.review_batches
    if not batches:
        wlog.info("No batches to review — skipping")
        state.review_stats.freeze_rate()
        state.review_phase.mark_done()
        return

    # --- Resolve axis-specific config -----------------------------------------
    review_target = getattr(state, "target", None) or (
        "names" if state.name_only else "names"
    )
    if review_target == "docs":
        models = state.review_models or get_sn_review_docs_models()
        review_axis = "docs"
    else:
        models = state.review_models or get_sn_review_names_models()
        review_axis = "names"

    max_cycles = get_sn_review_max_cycles()
    tolerance = get_sn_review_disagreement_threshold()

    wlog.info(
        "Reviewing %d batches (models=%s, max_cycles=%d, tolerance=%.2f, axis=%s)",
        len(batches),
        models,
        max_cycles,
        tolerance,
        review_axis,
    )

    # Shared context for prompt rendering
    grammar_enums = _get_grammar_enums()
    compose_ctx = _get_compose_context_for_review()

    # --- K3: Load scored examples for review calibration ---
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.example_loader import load_review_examples

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

    _review_axis_example: Literal["name", "docs", "full"] = (
        "docs" if review_target == "docs" else "name"
    )

    def _load_review_scored() -> list[dict]:
        with GraphClient() as gc:
            return load_review_examples(
                gc, physics_domains=review_domains, axis=_review_axis_example
            )

    review_scored = await asyncio.to_thread(_load_review_scored)
    if review_scored:
        wlog.info(
            "K3: Loaded %d scored examples for review (domains=%s)",
            len(review_scored),
            review_domains or "all",
        )

    # --- L4: Reviewer-theme injection (axis-aware) ---
    # Mine recurring themes from prior reviewer comments in the same axis
    # so the reviewer self-corrects domain-level patterns. Cache-stable:
    # injected into compose_ctx so it lands in the system prompt.
    from imas_codex.standard_names.review.themes import extract_reviewer_themes

    _theme_axis = "docs" if review_target == "docs" else "name"

    def _load_themes() -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for dom in review_domains:
            for theme in extract_reviewer_themes(dom, axis=_theme_axis):
                if theme not in seen:
                    seen.add(theme)
                    merged.append(theme)
                    if len(merged) >= 12:
                        return merged
        return merged

    reviewer_themes = await asyncio.to_thread(_load_themes)
    if reviewer_themes:
        wlog.info(
            "L4: Injected %d reviewer themes (axis=%s, domains=%s)",
            len(reviewer_themes),
            _theme_axis,
            review_domains or "all",
        )
    compose_ctx = dict(compose_ctx)
    compose_ctx["reviewer_themes"] = reviewer_themes

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
        """RD-quorum loop for a single review batch.

        Returns ``(scored_items, review_records)`` — scored items carry
        the final resolved scores, review records are one-per-cycle
        for graph persistence.
        """
        nonlocal total_cost, total_tokens, errors, revised, unscored

        async with sem:
            if state.should_stop():
                return [], []

            names = batch.get("names", [])
            if not names:
                return [], []

            # --- Budget reservation (worst-case: all cycles) -----------------
            # Per-name cost calibrated to observed Opus 4.6 + GPT-5.4 review
            # spend.  Each review cycle does one LLM call plus an optional
            # retry on unmatched items (~30% of batch).  For 3-model
            # RD-quorum, the escalator (cycle 2) only processes the disputed
            # subset, not the full batch.  A 1.5× per-model multiplier
            # conservatively covers "full batch + partial retry".
            #
            # Previous value was 3.0× per model (9.0× for 3 models), which
            # made the reservation ($6.75 for 15 names) exceed the entire
            # review phase budget ($3.00), blocking all batches.
            estimated_cost = len(names) * 0.05
            worst_case = estimated_cost * len(models) * 1.5
            lease = None

            if state.budget_manager:
                phase_tag = getattr(state, "budget_phase_tag", "") or "review"
                lease = state.budget_manager.reserve(worst_case, phase=phase_tag)
                if lease is None:
                    state.stats["budget_reservation_blocked"] = (
                        state.stats.get("budget_reservation_blocked", 0) + 1
                    )
                    wlog.info(
                        "Budget exhausted at batch %d — stopping review",
                        batch_idx,
                    )
                    return [], []

            review_group_id = str(_uuid.uuid4())
            batch_review_records: list[dict] = []
            review_phase = "review_docs" if review_target == "docs" else "review_names"
            _group_key = batch.get("group_key", "")

            try:
                # ============================================================
                # CYCLE 0 — Primary reviewer (blind)
                # ============================================================
                result_0 = await _review_single_batch(
                    names=_copy.deepcopy(names),
                    model=models[0],
                    grammar_enums=grammar_enums,
                    compose_ctx=compose_ctx,
                    batch_context=batch.get("group_key", ""),
                    neighborhood=batch.get("neighborhood", []),
                    audit_findings=batch.get("audit_findings", []),
                    wlog=wlog,
                    name_only=state.name_only,
                    target=review_target,
                    review_scored_examples=review_scored,
                    prior_reviews=None,  # blind — no prior context
                )
                c0_cost = result_0.get("_cost", 0.0)
                c0_items = result_0.get("_items", [])
                c0_tokens = result_0.get("_tokens", 0)
                c0_input = result_0.get("_input_tokens", 0)
                c0_output = result_0.get("_output_tokens", 0)
                revised += result_0.get("_revised", 0)

                total_cost += c0_cost
                total_tokens += c0_tokens
                state.review_stats.cost += c0_cost

                if lease:
                    _charge_review_cycle(
                        lease,
                        c0_cost,
                        result_0,
                        models[0],
                        c0_items,
                        _group_key,
                        "c0",
                        review_phase,
                        loop_stats=state.loop_stats,
                    )

                # Build cycle-0 Review records
                c0_ts = datetime.now(UTC).isoformat()
                n_c0 = max(len(c0_items), 1)
                for item in c0_items:
                    sn_id = item.get("id", "")
                    rec = _build_review_record(
                        item,
                        model=models[0],
                        is_canonical=True,
                        reviewed_at=c0_ts,
                        cost_usd=c0_cost / n_c0,
                        tokens_in=c0_input // n_c0 if c0_input else 0,
                        tokens_out=c0_output // n_c0 if c0_output else 0,
                    )
                    if rec:
                        rec["id"] = f"{sn_id}:{review_axis}:{review_group_id}:0"
                        rec["review_axis"] = review_axis
                        rec["cycle_index"] = 0
                        rec["review_group_id"] = review_group_id
                        rec["resolution_role"] = "primary"
                        rec["resolution_method"] = None
                        batch_review_records.append(rec)

                # Persist cycle-0 Review nodes immediately
                if batch_review_records:
                    await asyncio.to_thread(
                        _persist_review_records_sync, batch_review_records
                    )

                # --- Single-model shortcut ---
                if len(models) == 1:
                    # Stamp resolution_method on the just-persisted records
                    for rec in batch_review_records:
                        rec["resolution_method"] = "single_review"
                    await asyncio.to_thread(
                        _persist_review_records_sync, batch_review_records
                    )

                    _update_batch_stats(state, batch_idx, c0_items, c0_cost, result_0)
                    return c0_items, batch_review_records

                # ============================================================
                # CYCLE 1 — Secondary reviewer (blind, identical input)
                # ============================================================
                result_1 = await _review_single_batch(
                    names=_copy.deepcopy(names),
                    model=models[1],
                    grammar_enums=grammar_enums,
                    compose_ctx=compose_ctx,
                    batch_context=batch.get("group_key", ""),
                    neighborhood=batch.get("neighborhood", []),
                    audit_findings=batch.get("audit_findings", []),
                    wlog=wlog,
                    name_only=state.name_only,
                    target=review_target,
                    review_scored_examples=review_scored,
                    prior_reviews=None,  # blind — no primary context
                )
                c1_cost = result_1.get("_cost", 0.0)
                c1_items = result_1.get("_items", [])
                c1_tokens = result_1.get("_tokens", 0)
                c1_input = result_1.get("_input_tokens", 0)
                c1_output = result_1.get("_output_tokens", 0)
                revised += result_1.get("_revised", 0)

                total_cost += c1_cost
                total_tokens += c1_tokens
                state.review_stats.cost += c1_cost

                if lease:
                    _charge_review_cycle(
                        lease,
                        c1_cost,
                        result_1,
                        models[1],
                        c1_items,
                        _group_key,
                        "c1",
                        review_phase,
                        loop_stats=state.loop_stats,
                    )

                # Build cycle-1 Review records
                c1_ts = datetime.now(UTC).isoformat()
                n_c1 = max(len(c1_items), 1)
                c1_review_records: list[dict] = []
                for item in c1_items:
                    sn_id = item.get("id", "")
                    rec = _build_review_record(
                        item,
                        model=models[1],
                        is_canonical=False,
                        reviewed_at=c1_ts,
                        cost_usd=c1_cost / n_c1,
                        tokens_in=c1_input // n_c1 if c1_input else 0,
                        tokens_out=c1_output // n_c1 if c1_output else 0,
                    )
                    if rec:
                        rec["id"] = f"{sn_id}:{review_axis}:{review_group_id}:1"
                        rec["review_axis"] = review_axis
                        rec["cycle_index"] = 1
                        rec["review_group_id"] = review_group_id
                        rec["resolution_role"] = "secondary"
                        rec["resolution_method"] = None
                        c1_review_records.append(rec)

                # Persist cycle-1 Review nodes immediately
                if c1_review_records:
                    await asyncio.to_thread(
                        _persist_review_records_sync, c1_review_records
                    )
                batch_review_records.extend(c1_review_records)

                # --- Disagreement detection (per-dimension per-item) ---------
                c0_by_id = {it.get("id"): it for it in c0_items}
                c1_by_id = {it.get("id"): it for it in c1_items}

                all_ids = {n.get("id") for n in names}
                final_items: list[dict] = []
                disputed_ids: set[str] = set()
                resolution_methods: dict[str, str] = {}

                for nid in all_ids:
                    in_c0 = nid in c0_by_id
                    in_c1 = nid in c1_by_id

                    # Partial-failure handling
                    if not in_c0 and not in_c1:
                        # Both missing — item will be quarantined
                        resolution_methods[nid] = "retry_item"
                        unscored += 1
                        continue
                    if not in_c0:
                        # Cycle 0 missing, cycle 1 present → single_review
                        final_items.append(c1_by_id[nid])
                        resolution_methods[nid] = "single_review"
                        continue
                    if not in_c1:
                        # Cycle 1 missing, cycle 0 present → single_review
                        final_items.append(c0_by_id[nid])
                        resolution_methods[nid] = "single_review"
                        continue

                    # Both present — check per-dimension disagreement
                    item_0 = c0_by_id[nid]
                    item_1 = c1_by_id[nid]
                    is_disputed = _check_per_dim_disagreement(
                        item_0, item_1, tolerance, review_target
                    )

                    if is_disputed:
                        disputed_ids.add(nid)
                    else:
                        # Agreement → mean of both
                        merged = _merge_review_items(
                            item_0,
                            item_1,
                            review_target,
                        )
                        final_items.append(merged)
                        resolution_methods[nid] = "quorum_consensus"

                # --- Determine resolution and cycle-2 ----------------------
                if not disputed_ids:
                    # All items agreed → quorum_consensus
                    resolution = "quorum_consensus"
                    # Stamp resolution on last cycle records
                    for rec in c1_review_records:
                        rec["resolution_method"] = resolution
                    await asyncio.to_thread(
                        _persist_review_records_sync, c1_review_records
                    )

                    _update_batch_stats(
                        state, batch_idx, final_items, c0_cost + c1_cost, None
                    )
                    return final_items, batch_review_records

                # ============================================================
                # CYCLE 2 — Escalator (only disputed items, sees prior context)
                # ============================================================
                if len(models) < 3:
                    # No escalator available → max_cycles_reached
                    for nid in disputed_ids:
                        merged = _merge_review_items(
                            c0_by_id[nid],
                            c1_by_id[nid],
                            review_target,
                        )
                        final_items.append(merged)
                        resolution_methods[nid] = "max_cycles_reached"

                    # Stamp resolution on cycle-1 records
                    for rec in c1_review_records:
                        rec["resolution_method"] = "max_cycles_reached"
                    await asyncio.to_thread(
                        _persist_review_records_sync, c1_review_records
                    )

                    _update_batch_stats(
                        state,
                        batch_idx,
                        final_items,
                        c0_cost + c1_cost,
                        None,
                    )
                    return final_items, batch_review_records

                # Build escalator mini-batch with ONLY disputed items
                disputed_names = [n for n in names if n.get("id") in disputed_ids]

                # Build prior_reviews context for escalator prompt
                prior_reviews_ctx = _build_prior_reviews_context(
                    c0_items, c1_items, disputed_ids, models
                )

                result_2 = await _review_single_batch(
                    names=_copy.deepcopy(disputed_names),
                    model=models[2],
                    grammar_enums=grammar_enums,
                    compose_ctx=compose_ctx,
                    batch_context=batch.get("group_key", ""),
                    neighborhood=batch.get("neighborhood", []),
                    audit_findings=batch.get("audit_findings", []),
                    wlog=wlog,
                    name_only=state.name_only,
                    target=review_target,
                    review_scored_examples=review_scored,
                    prior_reviews=prior_reviews_ctx,
                )
                c2_cost = result_2.get("_cost", 0.0)
                c2_items = result_2.get("_items", [])
                c2_tokens = result_2.get("_tokens", 0)
                c2_input = result_2.get("_input_tokens", 0)
                c2_output = result_2.get("_output_tokens", 0)
                revised += result_2.get("_revised", 0)

                total_cost += c2_cost
                total_tokens += c2_tokens
                state.review_stats.cost += c2_cost

                if lease:
                    _charge_review_cycle(
                        lease,
                        c2_cost,
                        result_2,
                        models[2],
                        c2_items,
                        _group_key,
                        "c2",
                        review_phase,
                        loop_stats=state.loop_stats,
                    )

                # Build cycle-2 Review records
                c2_ts = datetime.now(UTC).isoformat()
                n_c2 = max(len(c2_items), 1)
                c2_review_records: list[dict] = []
                for item in c2_items:
                    sn_id = item.get("id", "")
                    rec = _build_review_record(
                        item,
                        model=models[2],
                        is_canonical=False,
                        reviewed_at=c2_ts,
                        cost_usd=c2_cost / n_c2,
                        tokens_in=c2_input // n_c2 if c2_input else 0,
                        tokens_out=c2_output // n_c2 if c2_output else 0,
                    )
                    if rec:
                        rec["id"] = f"{sn_id}:{review_axis}:{review_group_id}:2"
                        rec["review_axis"] = review_axis
                        rec["cycle_index"] = 2
                        rec["review_group_id"] = review_group_id
                        rec["resolution_role"] = "escalator"
                        rec["resolution_method"] = "authoritative_escalation"
                        c2_review_records.append(rec)

                # Persist cycle-2 Review nodes immediately
                if c2_review_records:
                    await asyncio.to_thread(
                        _persist_review_records_sync, c2_review_records
                    )
                batch_review_records.extend(c2_review_records)

                # Resolve disputed items: escalator result is authoritative
                c2_by_id = {it.get("id"): it for it in c2_items}
                for nid in disputed_ids:
                    if nid in c2_by_id:
                        final_items.append(c2_by_id[nid])
                        resolution_methods[nid] = "authoritative_escalation"
                    else:
                        # Escalator failed to score this item → mean fallback
                        merged = _merge_review_items(
                            c0_by_id[nid],
                            c1_by_id[nid],
                            review_target,
                        )
                        final_items.append(merged)
                        resolution_methods[nid] = "max_cycles_reached"

                _update_batch_stats(
                    state,
                    batch_idx,
                    final_items,
                    c0_cost + c1_cost + c2_cost,
                    None,
                )
                return final_items, batch_review_records

            except Exception:
                wlog.debug(
                    "Review batch %d failed",
                    batch_idx,
                    exc_info=True,
                )
                errors += len(names)
                unscored += len(names)
                state.review_stats.errors += len(names)
                return [], []

            finally:
                if lease:
                    lease.release_unused()

    # --- Polling-based work distribution (42-polling-workers) ---
    # N independent workers poll an asyncio.Queue for batches. Each worker
    # acquires the concurrency semaphore inside _process_batch (unchanged),
    # which handles budget reservation internally.
    review_queue: asyncio.Queue = asyncio.Queue()
    for _q_idx, _q_batch in enumerate(batches):
        review_queue.put_nowait((_q_idx, _q_batch))

    async def _review_polling_worker(worker_id: int) -> None:
        while not state.should_stop():
            try:
                idx, batch = review_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            try:
                result = await _process_batch(idx, batch)
                if isinstance(result, tuple):
                    items, recs = result
                    scored.extend(items)
                    review_records.extend(recs)
            except Exception as exc:
                wlog.warning("Worker %d: batch %d failed: %s", worker_id, idx, exc)

    n_review_workers = state.concurrency
    await asyncio.gather(*[_review_polling_worker(i) for i in range(n_review_workers)])

    state.review_results = scored

    # Canonical model is always models[0]
    state.canonical_review_model = models[0] if models else None
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
        write_docs_review_results,
        write_name_review_results,
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
    model = state.canonical_review_model or state.review_model or "unknown"
    reviewed_at = datetime.now(UTC).isoformat()
    _compute_hash = _get_hash_fn()

    for entry in results:
        entry["reviewer_model"] = model
        entry["reviewed_at"] = reviewed_at

        # Compute fresh review_input_hash
        if _compute_hash is not None:
            entry["review_input_hash"] = _compute_hash(entry)

    # Distribute total review cost equally across entries so each
    # StandardName gets a per-name cost share via the axis writer.
    total_cost = state.stats.get("review_cost", 0.0)
    if total_cost and results:
        cost_share = total_cost / len(results)
        for entry in results:
            entry.setdefault("llm_cost", cost_share)

    # Determine review mode from state.target
    review_target = getattr(state, "target", "names")

    # Write canonical scores to StandardName nodes using axis-specific writers
    def _write() -> int:
        if review_target == "docs":
            return write_docs_review_results(results, stats=state.stats)
        return write_name_review_results(results, stats=state.stats)

    written = await asyncio.to_thread(_write)

    state.persist_stats.processed = written
    state.persist_stats.record_batch(written)

    # Write Review nodes
    review_records = state.review_records or []
    if review_records:

        def _write_reviews() -> int:
            return write_reviews(review_records, skip_cost=True)

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
    """
    from imas_codex.standard_names.models import StandardNameReviewVerdict

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

        # Store review scores on the entry (in-memory dict keys).
        # NOTE: ``reviewer_score`` is a generic in-memory key; it is mapped
        # to axis-specific graph properties (``reviewer_score_name`` or
        # ``reviewer_score_docs``) by write_name_review_results /
        # write_docs_review_results.  There is no ``reviewer_score`` graph
        # property.
        original["reviewer_score"] = review.scores.score
        original["reviewer_scores"] = json.dumps(review.scores.model_dump())
        original["reviewer_comments"] = review.reasoning
        original["review_tier"] = review.scores.tier
        if hasattr(review, "comments") and review.comments is not None:
            original["reviewer_comments_per_dim"] = json.dumps(
                review.comments.model_dump()
            )
        original["reviewer_verdict"] = review.verdict.value

        # Capture reviewer's suggested-name + justification.
        # Per the W37 prompt rewrite, the reviewer offers a concrete
        # alternative for revise/reject verdicts (and null for accept).
        suggested_name = getattr(review, "suggested_name", None)
        suggestion_justification = getattr(review, "suggestion_justification", None)
        if suggested_name and not _valid_sn_id(suggested_name):
            wlog.warning(
                "Rejecting malformed suggested_name (len=%d) for %r",
                len(suggested_name),
                review.standard_name,
            )
            suggested_name = None
            suggestion_justification = None
        if suggested_name:
            original["_suggested_name"] = suggested_name
            if suggestion_justification:
                original["_suggestion_justification"] = suggestion_justification

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
            revised_entry["_original_id"] = original["id"]  # preserve real SN id
            revised_entry["_suggested_name"] = review.revised_name  # record suggestion
            # do NOT overwrite revised_entry["id"] — keeps MATCH on SN node correct
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
    prior_reviews: list[dict] | None = None,
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

    When ``target`` is None, the legacy ``name_only`` boolean selects
    between ``"names"`` and ``"docs"`` for back-compat.

    ``prior_reviews`` is used for escalator (cycle 2) calls — a list of
    dicts with ``role``, ``model``, and ``items`` keys, injected into
    the prompt template via the ``{% if prior_reviews %}`` block.
    """
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.models import (
        StandardNameQualityReviewDocsBatch,
        StandardNameQualityReviewNameOnlyBatch,
    )

    if target is None:
        target = "names" if name_only else "names"

    if target == "docs":
        prompt_name = "sn/review_docs"
        response_model: type = StandardNameQualityReviewDocsBatch
    else:
        prompt_name = "sn/review_names"
        response_model = StandardNameQualityReviewNameOnlyBatch

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
        "prior_reviews": prior_reviews or [],
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
        "prior_reviews": prior_reviews or [],
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
        "_primary_cost": cost,
        "_primary_input_tokens": getattr(llm_out, "input_tokens", 0) or 0,
        "_primary_output_tokens": getattr(llm_out, "output_tokens", 0) or 0,
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


# =============================================================================
# RD-quorum helper functions
# =============================================================================


def _check_per_dim_disagreement(
    item_0: dict,
    item_1: dict,
    tolerance: float,
    target: str,
) -> bool:
    """Return True if any dimension disagrees beyond tolerance.

    Dimensions are extracted from ``reviewer_scores`` JSON. Scores are
    normalized to 0-1 (divide by 20) before comparing against tolerance.
    """
    scores_0 = _parse_dim_scores(item_0, target)
    scores_1 = _parse_dim_scores(item_1, target)

    if not scores_0 or not scores_1:
        # Can't compare — treat as disagreement if we have partial data
        return bool(scores_0) != bool(scores_1)

    for dim in scores_0:
        if dim in scores_1:
            # Normalize 0-20 → 0-1
            s0 = scores_0[dim] / 20.0
            s1 = scores_1[dim] / 20.0
            if abs(s0 - s1) > tolerance:
                return True
    return False


def _parse_dim_scores(item: dict, target: str) -> dict[str, float]:
    """Extract per-dimension scores from an item's reviewer_scores field."""
    raw = item.get("reviewer_scores")
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    elif isinstance(raw, dict):
        data = raw
    else:
        return {}

    # The scores model stores raw dimension values (0-20)
    if target == "docs":
        dims = [
            "description_quality",
            "documentation_quality",
            "completeness",
            "physics_accuracy",
        ]
    else:
        dims = ["grammar", "semantic", "convention", "completeness"]

    return {d: float(data[d]) for d in dims if d in data}


def _merge_review_items(
    item_0: dict,
    item_1: dict,
    target: str,
) -> dict:
    """Produce a merged item with mean scores from two review items.

    Uses item_0 as the base (preserving its id, source_id, etc.) and
    averages the per-dimension scores. Comments are merged.

    The mean-score result feeds ``review_mean_score`` and the boolean
    ``review_disagreement`` aggregate (computed from Review node spread in
    ``update_review_aggregates``).  The former primary/secondary scalar
    mirrors (``reviewer_score_secondary`` etc.) are no longer written here;
    use :func:`~imas_codex.standard_names.review.projection.project_canonical_review`
    to derive canonical scoring from Review nodes directly.
    """
    merged = dict(item_0)

    scores_0 = _parse_dim_scores(item_0, target)
    scores_1 = _parse_dim_scores(item_1, target)

    if scores_0 and scores_1:
        mean_scores: dict[str, float] = {}
        all_dims = set(scores_0) | set(scores_1)
        for dim in all_dims:
            s0 = scores_0.get(dim, scores_1.get(dim, 0.0))
            s1 = scores_1.get(dim, scores_0.get(dim, 0.0))
            mean_scores[dim] = (s0 + s1) / 2.0

        merged["reviewer_scores"] = json.dumps(mean_scores)

        # Recompute aggregate score
        total = sum(mean_scores.values())
        max_score = len(mean_scores) * 20.0
        merged["reviewer_score"] = total / max_score if max_score > 0 else 0.0

        # Recompute tier
        norm = merged["reviewer_score"]
        if norm >= 0.85:
            merged["review_tier"] = "outstanding"
        elif norm >= 0.65:
            merged["review_tier"] = "good"
        elif norm >= 0.40:
            merged["review_tier"] = "inadequate"
        else:
            merged["review_tier"] = "poor"

    # Merge comments
    c0 = item_0.get("reviewer_comments") or ""
    c1 = item_1.get("reviewer_comments") or ""
    if c0 and c1 and c0 != c1:
        merged["reviewer_comments"] = f"[Primary] {c0}\n[Secondary] {c1}"
    elif c1:
        merged["reviewer_comments"] = c1

    return merged


def _build_prior_reviews_context(
    c0_items: list[dict],
    c1_items: list[dict],
    disputed_ids: set[str],
    models: list[str],
) -> list[dict]:
    """Build the ``prior_reviews`` list for escalator prompt injection.

    Only includes items that are in the disputed set.
    """
    prior = []
    for role, items, model_idx in [
        ("primary", c0_items, 0),
        ("secondary", c1_items, 1),
    ]:
        filtered = [
            {
                "standard_name": it.get("id", ""),
                "score": it.get("reviewer_score", 0.0),
                "tier": it.get("review_tier", "unknown"),
                "verdict": it.get("reviewer_verdict", "accept"),
                "scores_json": it.get("reviewer_scores", "{}"),
                "comments_per_dim_json": it.get("reviewer_comments_per_dim"),
                "reasoning": it.get("reviewer_comments", ""),
            }
            for it in items
            if it.get("id") in disputed_ids
        ]
        if filtered:
            prior.append(
                {
                    "role": role,
                    "model": models[model_idx]
                    if model_idx < len(models)
                    else "unknown",
                    "items": filtered,
                }
            )
    return prior


def _persist_review_records_sync(records: list[dict]) -> int:
    """Synchronously persist Review records to graph.

    Used via ``asyncio.to_thread()`` to persist each cycle's records
    immediately after the LLM call completes — ensures crash-safety.
    """
    from imas_codex.standard_names.graph_ops import write_reviews

    return write_reviews(records)


def _update_batch_stats(
    state: Any,
    batch_idx: int,
    items: list[dict],
    cost: float,
    _result: Any,
) -> None:
    """Update review_stats after a batch completes."""
    actually_scored = len(items)
    state.review_stats.processed += actually_scored
    state.review_stats.record_batch(actually_scored)
    state.review_stats.stream_queue.add(
        [
            {
                "primary_text": f"batch {batch_idx + 1}",
                "description": f"{actually_scored} scored  ${cost:.3f}",
            }
        ]
    )

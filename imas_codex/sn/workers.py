"""Async workers for the standard-name build pipeline.

Six-phase pipeline (review optional):

    EXTRACT → COMPOSE → [REVIEW] → VALIDATE → CONSOLIDATE → PERSIST

- **extract**: queries graph for DD paths or facility signals, builds batches
- **compose**: LLM-generates standard names from extraction batches
- **review**: cross-model review of composed candidates (optional)
- **validate**: validates names against grammar via round-trip + fields check
- **consolidate**: cross-batch dedup, conflict detection, coverage accounting
- **persist**: writes consolidated names to graph with provenance

Workers follow the ``dd_workers.py`` pattern: each is an async function
with signature ``async def worker(state, **_kwargs)`` that updates stats,
marks phases done, and respects ``state.should_stop()``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imas_codex.sn.sources.base import ExtractionBatch
    from imas_codex.sn.state import SNBuildState

logger = logging.getLogger(__name__)


# =============================================================================
# EXTRACT phase
# =============================================================================


async def extract_worker(state: SNBuildState, **_kwargs) -> None:
    """Extract candidate quantities from graph entities into batches.

    For DD source: queries IMASNode paths, groups by cluster/IDS/prefix.
    Skips sources already linked via HAS_STANDARD_NAME unless --force.
    Stores ExtractionBatch objects in ``state.extracted``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_extract_worker")
    wlog.info("Starting extraction (source=%s)", state.source)

    def _run() -> list:
        from imas_codex.sn.graph_ops import (
            get_existing_standard_names,
            get_named_source_ids,
        )
        from imas_codex.sn.sources.dd import extract_dd_candidates

        existing = get_existing_standard_names()

        # Source-level skip for resumability
        named_ids: set[str] = set()
        if not state.force:
            named_ids = get_named_source_ids()
            if named_ids:
                wlog.info("Skipping %d already-named sources", len(named_ids))

        if state.source == "dd":
            batches = extract_dd_candidates(
                ids_filter=state.ids_filter,
                domain_filter=state.domain_filter,
                limit=state.limit or 500,
                existing_names=existing,
            )
        else:
            wlog.error("Unknown source: %s", state.source)
            return []

        # Filter out already-named sources from batches
        if named_ids and not state.force:
            for batch in batches:
                batch.items = [
                    item
                    for item in batch.items
                    if item.get("path", item.get("signal_id")) not in named_ids
                ]
            # Remove empty batches
            batches = [b for b in batches if b.items]

        return batches

    batches = await asyncio.to_thread(_run)

    total_items = sum(len(b.items) for b in batches)
    state.extracted = batches
    state.extract_stats.total = total_items
    state.extract_stats.processed = total_items
    state.extract_stats.record_batch(total_items)

    wlog.info(
        "Extraction complete: %d batches, %d items",
        len(batches),
        total_items,
    )
    state.stats["extract_batches"] = len(batches)
    state.stats["extract_count"] = total_items

    state.extract_stats.freeze_rate()
    state.extract_phase.mark_done()


# =============================================================================
# COMPOSE phase
# =============================================================================


async def compose_worker(state: SNBuildState, **_kwargs) -> None:
    """LLM-generate standard names from extracted batches.

    Uses acall_llm_structured() with system/user prompt split for
    prompt caching.  Runs batches concurrently with semaphore.
    Results stored in ``state.composed``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_compose_worker")

    total_items = sum(len(b.items) for b in state.extracted)

    if state.dry_run:
        wlog.info("Dry run — skipping composition for %d items", total_items)
        state.compose_stats.total = total_items
        state.compose_stats.processed = total_items
        state.stats["compose_skipped"] = True
        state.compose_stats.freeze_rate()
        state.compose_phase.mark_done()
        return

    if not state.extracted:
        wlog.info("No batches to compose — skipping")
        state.compose_stats.freeze_rate()
        state.compose_phase.mark_done()
        return

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model
    from imas_codex.sn.context import build_compose_context
    from imas_codex.sn.models import SNComposeBatch

    model = get_model("language")
    context = build_compose_context()

    # Render system prompt once (cached via prompt caching)
    system_prompt = render_prompt("sn/compose_system", context)

    wlog.info(
        "Composing standard names for %d items in %d batches (model=%s)",
        total_items,
        len(state.extracted),
        model,
    )
    state.compose_stats.total = total_items

    sem = asyncio.Semaphore(5)

    async def _compose_batch(batch: ExtractionBatch) -> list[dict]:
        async with sem:
            if state.should_stop():
                return []

            user_context = {
                "items": batch.items,
                "ids_name": batch.group_key,
                "existing_names": sorted(batch.existing_names)[:200],
                "cluster_context": batch.context,
            }
            user_prompt = render_prompt("sn/compose_dd", {**context, **user_context})

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            result, cost, tokens = await acall_llm_structured(
                model=model,
                messages=messages,
                response_model=SNComposeBatch,
            )

            state.compose_stats.cost += cost
            state.compose_stats.processed += len(batch.items)
            state.compose_stats.record_batch(len(batch.items))

            candidates = []
            for c in result.candidates:
                # Find the matching source item to get authoritative unit
                source_item = next(
                    (item for item in batch.items if item.get("path") == c.source_id),
                    None,
                )
                candidates.append(
                    {
                        "id": c.standard_name,
                        "source_type": "dd" if state.source == "dd" else "signal",
                        "source_id": c.source_id,
                        "description": c.description,
                        "documentation": c.documentation,
                        "kind": c.kind,
                        "tags": c.tags,
                        "links": c.links,
                        "imas_paths": c.ids_paths,  # graph schema key
                        "fields": c.grammar_fields,
                        "confidence": c.confidence,
                        "reason": c.reason,
                        "validity_domain": c.validity_domain,
                        "constraints": c.constraints,
                        # Inject unit from DD (authoritative source, not LLM output)
                        "unit": source_item.get("unit") if source_item else None,
                    }
                )

            # Collect vocab gaps for later reporting
            if result.vocab_gaps:
                for vg in result.vocab_gaps:
                    state.stats.setdefault("vocab_gaps", []).append(
                        {
                            "source_id": vg.source_id,
                            "segment": vg.segment,
                            "needed_token": vg.needed_token,
                            "reason": vg.reason,
                        }
                    )

            wlog.info(
                "Batch %s: %d composed, %d skipped (cost=$%.4f)",
                batch.group_key,
                len(result.candidates),
                len(result.skipped),
                cost,
            )
            return candidates

    tasks = [_compose_batch(batch) for batch in state.extracted]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    composed: list[dict] = []
    errors = 0
    for r in results:
        if isinstance(r, list):
            composed.extend(r)
        elif isinstance(r, Exception):
            errors += 1
            wlog.warning("Batch failed: %s", r)

    state.composed = composed
    state.compose_stats.errors = errors

    wlog.info(
        "Composition complete: %d composed, %d errors (cost=$%.4f)",
        len(composed),
        errors,
        state.compose_stats.cost,
    )
    state.stats["compose_count"] = len(composed)
    state.stats["compose_errors"] = errors

    state.compose_stats.freeze_rate()
    state.compose_phase.mark_done()


# =============================================================================
# REVIEW phase
# =============================================================================

# Default batch size for review LLM calls
_REVIEW_BATCH_SIZE = 10


def _get_grammar_enums() -> dict[str, list[str]]:
    """Return grammar enum values for prompt context."""
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
        "coordinates": [e.value for e in Component],  # same enum
        "positions": [e.value for e in Position],
        "processes": [e.value for e in Process],
        "transformations": [e.value for e in Transformation],
        "geometric_bases": [e.value for e in GeometricBase],
        "objects": [e.value for e in Object],
        "binary_operators": [e.value for e in BinaryOperator],
    }


async def review_worker(state: SNBuildState, **_kwargs) -> None:
    """Cross-model review of composed standard name candidates.

    Uses a different LLM model family to review candidates produced by
    the compose phase.  Applies accept/reject/revise verdicts.

    Results stored in ``state.reviewed``; validate reads from reviewed
    (falling back to composed when review is skipped).
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_review_worker")

    if state.dry_run:
        wlog.info("Dry run — skipping review for %d candidates", len(state.composed))
        state.reviewed = list(state.composed)
        state.review_stats.total = len(state.composed)
        state.review_stats.processed = len(state.composed)
        state.stats["review_skipped"] = True
        state.review_stats.freeze_rate()
        state.review_phase.mark_done()
        return

    if not state.composed:
        wlog.info("No composed candidates to review — skipping")
        state.review_stats.freeze_rate()
        state.review_phase.mark_done()
        return

    wlog.info("Reviewing %d composed candidates", len(state.composed))
    state.review_stats.total = len(state.composed)

    from imas_codex.settings import get_model

    review_model = state.review_model or get_model("reasoning")
    wlog.info("Review model: %s", review_model)

    grammar_enums = _get_grammar_enums()
    existing_names = _get_existing_names_for_review()

    accepted: list[dict] = []
    rejected = 0
    revised = 0
    errors = 0
    total_cost = 0.0
    total_tokens = 0
    names_in_run: set[str] = set(existing_names)

    candidates = list(state.composed)
    for batch_start in range(0, len(candidates), _REVIEW_BATCH_SIZE):
        if state.should_stop():
            wlog.info("Stop requested at batch starting %d", batch_start)
            break

        batch = candidates[batch_start : batch_start + _REVIEW_BATCH_SIZE]

        try:
            batch_result = await _review_batch(
                batch,
                review_model,
                grammar_enums,
                names_in_run,
                wlog,
            )
            batch_accepted, batch_rejected, batch_revised, cost, tokens = batch_result
            total_cost += cost
            total_tokens += tokens

            for entry in batch_accepted:
                name = entry.get("id", "")
                if name and name not in names_in_run:
                    accepted.append(entry)
                    names_in_run.add(name)
                elif name in names_in_run:
                    wlog.debug("Duplicate name after review: %r — skipping", name)
                    rejected += 1
                else:
                    accepted.append(entry)

            rejected += batch_rejected
            revised += batch_revised

        except Exception:
            wlog.debug("Review batch failed at offset %d", batch_start, exc_info=True)
            errors += len(batch)
            accepted.extend(batch)

        state.review_stats.processed = min(batch_start + len(batch), len(candidates))
        state.review_stats.record_batch(len(batch))

    state.review_stats.errors = errors
    state.review_stats.cost = total_cost
    state.reviewed = accepted

    wlog.info(
        "Review complete: %d accepted, %d rejected, %d revised, %d errors "
        "(cost: $%.4f, tokens: %d)",
        len(accepted),
        rejected,
        revised,
        errors,
        total_cost,
        total_tokens,
    )
    state.stats["review_accepted"] = len(accepted)
    state.stats["review_rejected"] = rejected
    state.stats["review_revised"] = revised
    state.stats["review_errors"] = errors
    state.stats["review_cost"] = total_cost

    state.review_stats.freeze_rate()
    state.review_phase.mark_done()


def _get_existing_names_for_review() -> set[str]:
    """Fetch existing standard names from graph for duplicate checking."""
    try:
        from imas_codex.sn.graph_ops import get_existing_standard_names

        return get_existing_standard_names()
    except Exception:
        return set()


async def _review_batch(
    batch: list[dict],
    model: str,
    grammar_enums: dict[str, list[str]],
    existing_names: set[str],
    wlog: logging.LoggerAdapter,
) -> tuple[list[dict], int, int, float, int]:
    """Review a single batch of candidates via LLM."""
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.sn.models import SNReviewBatch, SNReviewVerdict

    context = {
        "items": batch,
        "existing_names": sorted(existing_names)[:200],
        **grammar_enums,
    }
    prompt_text = render_prompt("sn/review", context)

    messages = [{"role": "user", "content": prompt_text}]
    result, cost, tokens = await acall_llm_structured(
        model=model,
        messages=messages,
        response_model=SNReviewBatch,
    )

    entry_map: dict[str, dict] = {}
    for entry in batch:
        sid = entry.get("source_id") or entry.get("id") or ""
        entry_map[sid] = entry

    accepted: list[dict] = []
    rejected_count = 0
    revised_count = 0

    for review in result.reviews:
        original = entry_map.get(review.source_id)
        if original is None:
            wlog.debug("Review returned unknown source_id: %s", review.source_id)
            continue

        if review.verdict == SNReviewVerdict.accept:
            accepted.append(original)
        elif review.verdict == SNReviewVerdict.reject:
            rejected_count += 1
            wlog.debug("Rejected %r: %s", review.standard_name, review.reason)
        elif review.verdict == SNReviewVerdict.revise:
            if review.revised_name:
                revised_entry = dict(original)
                revised_entry["id"] = review.revised_name
                if review.revised_fields:
                    for key, value in review.revised_fields.items():
                        if key in revised_entry:
                            revised_entry[key] = value
                accepted.append(revised_entry)
                revised_count += 1
                wlog.debug(
                    "Revised %r → %r: %s",
                    review.standard_name,
                    review.revised_name,
                    review.reason,
                )
            else:
                accepted.append(original)

    reviewed_ids = {r.source_id for r in result.reviews}
    for entry in batch:
        sid = entry.get("source_id") or entry.get("id") or ""
        if sid not in reviewed_ids:
            accepted.append(entry)

    return accepted, rejected_count, revised_count, cost, tokens


# =============================================================================
# VALIDATE phase
# =============================================================================


async def validate_worker(state: SNBuildState, **_kwargs) -> None:
    """Validate composed names against grammar via round-trip + fields check.

    Reads from ``state.reviewed`` if review ran, else ``state.composed``.
    Reports distinct metrics: validate_valid, validate_invalid,
    validate_fields_consistent, validate_fields_inconsistent.
    Results stored in ``state.validated``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_validate_worker")

    if state.dry_run:
        wlog.info("Dry run — skipping validation")
        count = sum(len(b.items) for b in state.extracted) if state.extracted else 0
        state.validate_stats.total = count
        state.validate_stats.processed = count
        state.stats["validate_skipped"] = True
        state.validate_stats.freeze_rate()
        state.validate_phase.mark_done()
        return

    # Read from reviewed if available, else composed
    input_candidates = state.reviewed if state.reviewed else state.composed

    if not input_candidates:
        wlog.info("No composed names to validate — skipping")
        state.validate_stats.freeze_rate()
        state.validate_phase.mark_done()
        return

    from imas_standard_names.grammar import (
        StandardName,
        compose_standard_name,
        parse_standard_name,
    )

    # Load tag vocabulary for soft validation
    try:
        from typing import get_args

        from imas_standard_names.grammar.tag_types import PrimaryTag, SecondaryTag

        valid_primary_tags = set(get_args(PrimaryTag))
        valid_secondary_tags = set(get_args(SecondaryTag))
        valid_tags = valid_primary_tags | valid_secondary_tags
    except Exception:
        valid_tags = set()

    # Collect existing names for link validation
    existing_names: set[str] = set()
    for entry in input_candidates:
        existing_names.add(entry.get("id", ""))

    wlog.info("Validating %d composed names", len(input_candidates))
    state.validate_stats.total = len(input_candidates)

    valid: list[dict] = []
    invalid_count = 0
    fields_consistent = 0
    fields_inconsistent = 0

    # Soft validation counters
    desc_present = 0
    desc_too_long = 0
    doc_present = 0
    doc_too_short = 0
    unit_valid = 0
    kind_valid = 0
    tags_valid = 0
    links_valid = 0

    _VALID_KINDS = {"scalar", "vector", "metadata"}

    for i, entry in enumerate(input_candidates):
        name = entry.get("id", "")
        try:
            parsed = parse_standard_name(name)
            normalized = compose_standard_name(parsed)
            if normalized != name:
                wlog.debug("Normalization: %r → %r", name, normalized)
                entry["id"] = normalized

            # Fields consistency check
            fields = entry.get("fields", {})
            if fields:
                try:
                    sn_fields = _convert_fields_to_grammar(fields)
                    if sn_fields:
                        sn = StandardName(**sn_fields)
                        from_fields = compose_standard_name(sn)
                        if from_fields == normalized:
                            fields_consistent += 1
                            entry["fields_consistent"] = True
                        else:
                            fields_inconsistent += 1
                            entry["fields_consistent"] = False
                            wlog.debug(
                                "Fields inconsistent: %r vs %r",
                                from_fields,
                                normalized,
                            )
                except Exception:
                    fields_inconsistent += 1
                    entry["fields_consistent"] = False

            # --- Soft validation checks (metrics only, never reject) ---

            # 1. Description present + length check
            desc = entry.get("description", "")
            if desc:
                desc_present += 1
                if len(desc) > 120:
                    desc_too_long += 1
                    wlog.debug("Description >120 chars for %r: %d", name, len(desc))
            else:
                wlog.debug("Missing description for %r", name)

            # 2. Documentation present + minimum length
            doc = entry.get("documentation", "")
            if doc:
                doc_present += 1
                if len(doc) < 200:
                    doc_too_short += 1
                    wlog.debug("Documentation <200 chars for %r: %d", name, len(doc))
            else:
                wlog.debug("Missing documentation for %r", name)

            # 3. Unit validity — simple pattern check
            unit = entry.get("unit")
            if unit and isinstance(unit, str) and len(unit) < 50:
                unit_valid += 1

            # 4. Kind validity
            kind = entry.get("kind", "")
            if kind in _VALID_KINDS:
                kind_valid += 1
            elif kind:
                wlog.debug("Invalid kind %r for %r", kind, name)

            # 5. Tags from vocabulary
            entry_tags = entry.get("tags") or []
            if entry_tags and valid_tags:
                if all(t in valid_tags for t in entry_tags):
                    tags_valid += 1
                else:
                    bad_tags = [t for t in entry_tags if t not in valid_tags]
                    wlog.debug("Unknown tags for %r: %s", name, bad_tags)
            elif entry_tags:
                tags_valid += 1  # no vocabulary loaded, accept any

            # 6. Links reference existing names
            entry_links = entry.get("links") or []
            if entry_links:
                if all(lnk in existing_names for lnk in entry_links):
                    links_valid += 1
                else:
                    unknown = [lnk for lnk in entry_links if lnk not in existing_names]
                    wlog.debug("Unknown links for %r: %s", name, unknown)

            # 7. ids_paths look like IMAS paths (contain '/')
            imas_paths = entry.get("imas_paths") or []
            for p in imas_paths:
                if "/" not in p:
                    wlog.debug("Suspicious ids_path for %r: %r", name, p)

            valid.append(entry)
        except Exception:
            wlog.debug("Validation failed for name: %r", name)
            invalid_count += 1

        state.validate_stats.processed = i + 1

    if valid:
        state.validate_stats.record_batch(len(valid))
    state.validate_stats.errors = invalid_count
    state.validated = valid

    wlog.info(
        "Validation complete: %d valid, %d invalid, "
        "%d fields consistent, %d fields inconsistent",
        len(valid),
        invalid_count,
        fields_consistent,
        fields_inconsistent,
    )
    wlog.info(
        "Soft checks: desc=%d/%d (>120: %d), doc=%d/%d (<200: %d), "
        "unit=%d, kind=%d, tags=%d, links=%d",
        desc_present,
        len(valid),
        desc_too_long,
        doc_present,
        len(valid),
        doc_too_short,
        unit_valid,
        kind_valid,
        tags_valid,
        links_valid,
    )
    state.stats["validate_valid"] = len(valid)
    state.stats["validate_invalid"] = invalid_count
    state.stats["validate_fields_consistent"] = fields_consistent
    state.stats["validate_fields_inconsistent"] = fields_inconsistent
    # Soft validation metrics
    state.stats["validate_desc_present"] = desc_present
    state.stats["validate_desc_too_long"] = desc_too_long
    state.stats["validate_doc_present"] = doc_present
    state.stats["validate_doc_too_short"] = doc_too_short
    state.stats["validate_unit_valid"] = unit_valid
    state.stats["validate_kind_valid"] = kind_valid
    state.stats["validate_tags_valid"] = tags_valid
    state.stats["validate_links_valid"] = links_valid

    state.validate_stats.freeze_rate()
    state.validate_phase.mark_done()


def _convert_fields_to_grammar(fields: dict) -> dict:
    """Convert string field values to grammar enum instances."""
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

    enum_map = {
        "subject": Subject,
        "component": Component,
        "coordinate": Component,
        "position": Position,
        "process": Process,
        "transformation": Transformation,
        "geometric_base": GeometricBase,
        "object": Object,
        "binary_operator": BinaryOperator,
    }

    sn_fields: dict = {}
    for k, v in fields.items():
        if k == "physical_base":
            sn_fields[k] = v
        elif k in enum_map:
            sn_fields[k] = enum_map[k](v)
    return sn_fields


# =============================================================================
# CONSOLIDATE phase
# =============================================================================


async def consolidate_worker(state: SNBuildState, **_kwargs) -> None:
    """Cross-batch consolidation: dedup, conflict detection, coverage accounting.

    Runs after VALIDATE, before PERSIST.  Reads from ``state.validated``,
    writes to ``state.consolidated``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_consolidate_worker")

    if state.dry_run:
        wlog.info("Dry run — skipping consolidation")
        state.consolidated = list(state.validated)
        state.consolidate_stats.total = len(state.validated)
        state.consolidate_stats.processed = len(state.validated)
        state.stats["consolidation_skipped"] = True
        state.consolidate_stats.freeze_rate()
        state.consolidate_phase.mark_done()
        return

    if not state.validated:
        wlog.info("No validated names to consolidate — skipping")
        state.consolidate_stats.freeze_rate()
        state.consolidate_phase.mark_done()
        return

    from imas_codex.sn.consolidation import consolidate_candidates

    wlog.info("Consolidating %d validated candidates", len(state.validated))
    state.consolidate_stats.total = len(state.validated)

    # Collect all source paths for coverage accounting
    source_paths = None
    if state.extracted:
        source_paths = set()
        for batch in state.extracted:
            for item in batch.items:
                if item.get("path"):
                    source_paths.add(item["path"])

    result = await asyncio.to_thread(
        consolidate_candidates,
        state.validated,
        source_paths=source_paths,
    )

    state.consolidated = result.approved

    # Log results
    wlog.info(
        "Consolidation: %d approved, %d conflicts, %d coverage gaps, %d reused",
        len(result.approved),
        len(result.conflicts),
        len(result.coverage_gaps),
        len(result.reused),
    )

    # Record stats
    state.stats["consolidation"] = result.stats
    if result.conflicts:
        for conflict in result.conflicts:
            wlog.warning(
                "Conflict: %s (%s) — %s",
                conflict.standard_name,
                conflict.conflict_type,
                conflict.details,
            )
    if result.coverage_gaps:
        wlog.info("Coverage gaps: %d unmapped source paths", len(result.coverage_gaps))

    state.consolidate_stats.processed = len(state.validated)
    state.consolidate_stats.freeze_rate()
    state.consolidate_phase.mark_done()


# =============================================================================
# PERSIST phase
# =============================================================================


async def persist_worker(state: SNBuildState, **_kwargs) -> None:
    """Write consolidated standard names to graph with provenance.

    Creates StandardName nodes and HAS_STANDARD_NAME relationships.
    Enriches entries with model name, generation timestamp, review status.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_persist_worker")

    if state.dry_run:
        wlog.info(
            "Dry run — skipping persist for %d consolidated names",
            len(state.consolidated),
        )
        state.persist_stats.total = len(state.consolidated)
        state.persist_stats.processed = len(state.consolidated)
        state.stats["persist_skipped"] = True
        state.persist_stats.freeze_rate()
        state.persist_phase.mark_done()
        return

    if not state.consolidated:
        wlog.info("No consolidated names to persist — skipping")
        state.persist_stats.freeze_rate()
        state.persist_phase.mark_done()
        return

    from imas_codex.settings import get_model
    from imas_codex.sn.graph_ops import write_standard_names

    model = get_model("language")
    now = datetime.now(UTC).isoformat()

    # Enrich with provenance
    for entry in state.consolidated:
        entry.setdefault("model", model)
        entry.setdefault("review_status", "drafted")
        entry.setdefault("generated_at", now)
        # confidence comes from LLM output — never default to 1.0

        # Extract grammar fields into top-level properties for graph
        fields = entry.get("fields", {})
        for field_name in (
            "physical_base",
            "subject",
            "component",
            "coordinate",
            "position",
            "process",
            "geometric_base",
            "object",
        ):
            if field_name in fields and field_name not in entry:
                entry[field_name] = fields[field_name]

    wlog.info("Persisting %d consolidated standard names", len(state.consolidated))
    state.persist_stats.total = len(state.consolidated)

    written = await asyncio.to_thread(write_standard_names, state.consolidated)

    # Embed descriptions for vector search
    if written > 0:
        try:
            from imas_codex.embeddings.description import embed_descriptions_batch

            embed_items = [
                {"id": e["id"], "description": e.get("description", "")}
                for e in state.consolidated
                if e.get("description")
            ]
            if embed_items:
                enriched = await asyncio.to_thread(
                    embed_descriptions_batch, embed_items
                )
                # Write embeddings back to graph
                from imas_codex.graph.client import GraphClient

                def _write_embeddings():
                    with GraphClient() as gc:
                        gc.query(
                            """
                            UNWIND $batch AS b
                            MATCH (sn:StandardName {id: b.id})
                            SET sn.embedding = b.embedding,
                                sn.embedded_at = datetime()
                            """,
                            batch=[
                                {"id": e["id"], "embedding": e["embedding"]}
                                for e in enriched
                                if e.get("embedding")
                            ],
                        )

                await asyncio.to_thread(_write_embeddings)
                wlog.info("Embedded %d StandardName descriptions", len(embed_items))
        except Exception:
            wlog.warning(
                "Embedding generation failed — names persisted without embeddings",
                exc_info=True,
            )

    state.persist_stats.processed = written
    state.persist_stats.record_batch(written)
    state.stats["persist_written"] = written

    wlog.info("Persist complete: %d written", written)
    state.persist_stats.freeze_rate()
    state.persist_phase.mark_done()

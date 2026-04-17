"""Benchmark runner for comparing LLM models on standard name generation.

Extracts a fixed dataset, runs it through multiple models, validates
output via grammar round-trip, and compares against a reference set.
Produces a :class:`BenchmarkReport` with per-model metrics suitable
for Rich table display and JSON export.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_standard_names.grammar import (
    BinaryOperator,
    Component,
    GeometricBase,
    Object,
    Position,
    Process,
    StandardName,
    Subject,
    Transformation,
    compose_standard_name,
    parse_standard_name,
)

from imas_codex.standard_names.models import StandardNameComposeBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    models: list[str]
    source: str = "dd"
    ids_filter: str | None = None
    domain_filter: str | None = None
    facility: str | None = None
    max_candidates: int = 50
    runs_per_model: int = 1
    temperature: float = 0.0  # pinned for reproducibility
    reviewer_model: str | None = None  # frontier model for quality scoring


@dataclass
class ModelResult:
    """Results from running one model."""

    model: str
    candidates: list[dict] = field(default_factory=list)
    grammar_valid_count: int = 0
    grammar_invalid_count: int = 0
    fields_consistent_count: int = 0
    total_cost: float = 0.0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    names_per_minute: float = 0.0
    cost_per_name: float = 0.0
    skipped_count: int = 0
    batch_errors: int = 0
    # Quality against reference set
    reference_overlap: int = 0
    reference_total: int = 0
    reference_precision: float = 0.0
    reference_recall: float = 0.0
    # Quality scoring (reviewer model)
    quality_scores: list[dict] = field(default_factory=list)
    quality_distribution: dict[str, int] = field(default_factory=dict)
    avg_quality_score: float = 0.0
    avg_doc_length: float = 0.0
    avg_fields_populated: float = 0.0
    # Prompt-cache statistics
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class BenchmarkReport:
    """Full benchmark report."""

    config: BenchmarkConfig
    results: list[ModelResult]
    reference_names: list[str]
    extraction_count: int = 0
    timestamp: str = ""

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, data: str) -> BenchmarkReport:
        """Deserialize from JSON string."""
        raw = json.loads(data)
        config = BenchmarkConfig(**raw["config"])
        results = [ModelResult(**r) for r in raw["results"]]
        return cls(
            config=config,
            results=results,
            reference_names=raw["reference_names"],
            extraction_count=raw.get("extraction_count", 0),
            timestamp=raw.get("timestamp", ""),
        )


# ---------------------------------------------------------------------------
# Grammar validation
# ---------------------------------------------------------------------------


def validate_candidate(candidate: dict) -> tuple[bool, bool]:
    """Validate a single candidate via grammar round-trip.

    Returns:
        (grammar_valid, fields_consistent) tuple.
        grammar_valid: True if the name parses and round-trips.
        fields_consistent: True if composing from reported fields
            produces the same name (after normalization).
    """
    name = candidate.get("standard_name", "")
    fields = candidate.get("grammar_fields", {})

    grammar_valid = False
    fields_consistent = False

    # Check grammar round-trip
    try:
        parsed = parse_standard_name(name)
        normalized = compose_standard_name(parsed)
        grammar_valid = True  # parse+compose succeeded
    except Exception:
        return False, False

    # Check fields consistency: compose from reported fields
    try:
        # Convert string field values to enum instances
        sn_fields: dict[str, Any] = {}
        for k, v in fields.items():
            if k == "physical_base":
                sn_fields[k] = v
            elif k == "geometric_base":
                sn_fields[k] = GeometricBase(v)
            elif k == "subject":
                sn_fields[k] = Subject(v)
            elif k == "component":
                sn_fields[k] = Component(v)
            elif k == "coordinate":
                sn_fields[k] = Component(v)
            elif k == "position":
                sn_fields[k] = Position(v)
            elif k == "process":
                sn_fields[k] = Process(v)
            elif k == "transformation":
                sn_fields[k] = Transformation(v)
            elif k == "object":
                sn_fields[k] = Object(v)
            elif k == "binary_operator":
                sn_fields[k] = BinaryOperator(v)

        if sn_fields:
            sn = StandardName(**sn_fields)
            from_fields = compose_standard_name(sn)
            fields_consistent = from_fields == normalized
    except Exception:
        pass

    return grammar_valid, fields_consistent


# ---------------------------------------------------------------------------
# Reference comparison
# ---------------------------------------------------------------------------


def compare_to_reference(
    candidates: list[dict],
    reference: dict[str, dict],
) -> tuple[int, int, float, float]:
    """Compare model output against the reference set.

    Args:
        candidates: List of candidate dicts with source_id and standard_name.
        reference: REFERENCE_NAMES dict mapping source_path → {name, fields}.

    Returns:
        (overlap, ref_total, precision, recall) tuple.
        overlap: Number of candidates whose standard_name matches reference.
        ref_total: Total entries in reference set.
        precision: overlap / len(candidates) if candidates else 0.
        recall: overlap / ref_total if ref_total else 0.
    """
    # Build lookup from source_id → generated name
    generated = {}
    for c in candidates:
        sid = c.get("source_id", "")
        generated[sid] = c.get("standard_name", "")

    overlap = 0
    ref_total = len(reference)
    for path, ref_entry in reference.items():
        if path in generated:
            # Normalize both for comparison
            gen_name = generated[path]
            ref_name = ref_entry["name"]
            try:
                gen_parsed = parse_standard_name(gen_name)
                gen_normalized = compose_standard_name(gen_parsed)
            except Exception:
                gen_normalized = gen_name

            try:
                ref_parsed = parse_standard_name(ref_name)
                ref_normalized = compose_standard_name(ref_parsed)
            except Exception:
                ref_normalized = ref_name

            if gen_normalized == ref_normalized:
                overlap += 1

    n_candidates = len(candidates)
    precision = overlap / n_candidates if n_candidates else 0.0
    recall = overlap / ref_total if ref_total else 0.0

    return overlap, ref_total, precision, recall


# ---------------------------------------------------------------------------
# Quality tier labels
# ---------------------------------------------------------------------------


def load_calibration_entries() -> list[dict]:
    """Load calibration entries from benchmark_calibration.yaml.

    Returns a list of dicts, each with: name, tier, expected_score,
    description, documentation, unit, kind, tags, fields, reason.
    Returns empty list if file not found.
    """
    import yaml

    cal_path = Path(__file__).parent / "benchmark_calibration.yaml"
    if cal_path.exists():
        with open(cal_path) as f:
            data = yaml.safe_load(f) or {}
        return data.get("entries", [])
    return []


async def score_with_reviewer(
    candidates: list[dict],
    reviewer_model: str,
    calibration_entries: list[dict],
) -> list[dict]:
    """Score candidates using unified 6-dimensional quality review.

    Each candidate is scored across six dimensions (0-20 each):
    grammar, semantic, documentation, convention, completeness, compliance.
    Score is normalized (0-1).

    Returns list of dicts with: name, quality_tier, score,
    grammar_score, semantic_score, documentation_score,
    convention_score, completeness_score, compliance_score, reasoning.
    """
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.context import build_compose_context
    from imas_codex.standard_names.models import StandardNameQualityReviewBatch

    # Get compose context (includes grammar enums + shared include variables)
    compose_ctx = build_compose_context()
    grammar_enums = {
        k: compose_ctx[k]
        for k in (
            "subjects",
            "components",
            "coordinates",
            "positions",
            "processes",
            "transformations",
            "geometric_bases",
            "objects",
            "binary_operators",
        )
        if k in compose_ctx
    }

    # System prompt: rubric + calibration (cached across batches)
    system_prompt = render_prompt(
        "sn/review",
        {
            **compose_ctx,
            "calibration_entries": calibration_entries,
            "items": [],
            "existing_names": [],
            "batch_context": "",
            **grammar_enums,
        },
    )

    # Process in batches of 10
    all_reviews: list[dict] = []
    for i in range(0, len(candidates), 10):
        batch = candidates[i : i + 10]

        # Build per-batch user prompt with candidate details
        batch_items = []
        for c in batch:
            batch_items.append(
                {
                    "standard_name": c.get("standard_name", ""),
                    "source_id": c.get("source_id", ""),
                    "description": c.get("description", ""),
                    "documentation": (c.get("documentation", "") or "")[:500],
                    "unit": c.get("unit", "N/A"),
                    "kind": c.get("kind", "N/A"),
                    "tags": c.get("tags", []),
                    "grammar_fields": c.get("grammar_fields", {}),
                    "source_paths": c.get("source_paths", []),
                }
            )

        user_prompt = render_prompt(
            "sn/review",
            {
                **compose_ctx,
                "calibration_entries": calibration_entries,
                "items": batch_items,
                "existing_names": [],
                "batch_context": "",
                **grammar_enums,
            },
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            result, _, _ = await acall_llm_structured(
                model=reviewer_model,
                messages=messages,
                response_model=StandardNameQualityReviewBatch,
                service="standard-names",
            )
            for r in result.reviews:
                review_dict = {
                    "name": r.standard_name,
                    "quality_tier": r.scores.tier,
                    "score": r.scores.score,
                    "grammar_score": r.scores.grammar,
                    "semantic_score": r.scores.semantic,
                    "documentation_score": r.scores.documentation,
                    "convention_score": r.scores.convention,
                    "completeness_score": r.scores.completeness,
                    "compliance_score": r.scores.compliance,
                    "reasoning": r.reasoning,
                }
                all_reviews.append(review_dict)
        except Exception as e:
            logger.warning("Reviewer scoring failed for batch: %s", e)

    return all_reviews


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------


async def run_benchmark(
    config: BenchmarkConfig,
    extraction_batches: list[dict] | None = None,
) -> BenchmarkReport:
    """Run the benchmark across all configured models.

    Args:
        config: Benchmark configuration.
        extraction_batches: Pre-extracted candidate batches (list of dicts
            with items grouped by IDS). If None, extracts from graph.

    Returns:
        BenchmarkReport with per-model results.
    """
    from imas_codex.standard_names.benchmark_reference import REFERENCE_NAMES

    # --- 1. Extract candidates (same for all models) ---
    if extraction_batches is None:
        extraction_batches = _extract_candidates(config)

    # Flatten items for counting
    all_items = []
    for batch in extraction_batches:
        all_items.extend(batch.get("items", []))

    # Limit to max_candidates
    if len(all_items) > config.max_candidates:
        all_items = all_items[: config.max_candidates]
        # Rebuild batches with limited items
        extraction_batches = _rebuild_batches(extraction_batches, config.max_candidates)

    logger.info(
        "Benchmark: %d extraction items across %d batches",
        len(all_items),
        len(extraction_batches),
    )

    # --- 2. Run each model ---
    results: list[ModelResult] = []
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.context import build_compose_context

    context = build_compose_context()
    system_prompt = render_prompt("sn/compose_system", context)

    for model in config.models:
        logger.info("Benchmarking model: %s", model)
        model_result = await _run_model(
            model=model,
            extraction_batches=extraction_batches,
            config=config,
            reference=REFERENCE_NAMES,
            system_prompt=system_prompt,
            context=context,
        )
        results.append(model_result)

    # --- 2b. Reviewer scoring (optional) ---
    if config.reviewer_model:
        calibration_entries = load_calibration_entries()
        for result in results:
            if result.candidates:
                reviews = await score_with_reviewer(
                    result.candidates,
                    config.reviewer_model,
                    calibration_entries,
                )
                result.quality_scores = reviews
                # Compute distribution
                for r in reviews:
                    tier = r.get("quality_tier", "unknown")
                    result.quality_distribution[tier] = (
                        result.quality_distribution.get(tier, 0) + 1
                    )
                if reviews:
                    result.avg_quality_score = sum(
                        r.get("score", 0) for r in reviews
                    ) / len(reviews)

                # Compute doc length and field coverage metrics
                docs = [c.get("documentation", "") or "" for c in result.candidates]
                result.avg_doc_length = (
                    sum(len(d) for d in docs) / len(docs) if docs else 0.0
                )

                all_fields = {
                    "physical_base",
                    "subject",
                    "component",
                    "coordinate",
                    "position",
                    "process",
                }
                field_counts = []
                for c in result.candidates:
                    fields = c.get("grammar_fields", {})
                    field_counts.append(
                        len(set(fields.keys()) & all_fields) / len(all_fields)
                    )
                result.avg_fields_populated = (
                    sum(field_counts) / len(field_counts) if field_counts else 0.0
                )

    # --- 3. Build report ---
    report = BenchmarkReport(
        config=config,
        results=results,
        reference_names=list(REFERENCE_NAMES.keys()),
        extraction_count=len(all_items),
        timestamp=datetime.now(tz=UTC).isoformat(),
    )
    return report


def _extract_candidates(config: BenchmarkConfig) -> list[dict]:
    """Extract candidates from the graph DB.

    Returns list of batch dicts with keys: group_key, items, existing_names, context.
    """
    from imas_codex.standard_names.sources.dd import extract_dd_candidates

    batches = extract_dd_candidates(
        ids_filter=config.ids_filter,
        domain_filter=config.domain_filter,
        limit=config.max_candidates,
    )

    # Convert ExtractionBatch to plain dicts
    result = []
    for batch in batches:
        result.append(
            {
                "group_key": batch.group_key,
                "items": batch.items,
                "existing_names": list(batch.existing_names),
                "context": batch.context,
            }
        )
    return result


def _rebuild_batches(batches: list[dict], max_items: int) -> list[dict]:
    """Rebuild batches capping total items at max_items."""
    result = []
    count = 0
    for batch in batches:
        items = batch.get("items", [])
        remaining = max_items - count
        if remaining <= 0:
            break
        if len(items) > remaining:
            items = items[:remaining]
        result.append({**batch, "items": items})
        count += len(items)
    return result


async def _run_model(
    model: str,
    extraction_batches: list[dict],
    config: BenchmarkConfig,
    reference: dict[str, dict],
    system_prompt: str,
    context: dict[str, Any],
) -> ModelResult:
    """Run a single model across all extraction batches."""
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt

    result = ModelResult(model=model)
    all_candidates: list[dict] = []

    t0 = time.monotonic()

    for _run_idx in range(config.runs_per_model):
        for batch in extraction_batches:
            items = batch.get("items", [])
            if not items:
                continue

            group_key = batch.get("group_key", "unknown")
            existing = set(batch.get("existing_names", []))

            # Build user prompt context — mirrors workers.py pattern
            user_context = {
                "items": items,
                "ids_name": group_key,
                "existing_names": sorted(existing)[:200],
                "cluster_context": batch.get("context", ""),
            }

            try:
                user_prompt = render_prompt(
                    "sn/compose_dd", {**context, **user_context}
                )
            except Exception:
                logger.warning("Failed to render prompt for batch %s", group_key)
                result.batch_errors += 1
                continue

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # GPT-5.x models don't support temperature=0.0
            temp = config.temperature
            if "gpt-5" in model and temp == 0.0:
                temp = None  # let the provider use its default

            try:
                llm_response = await acall_llm_structured(
                    model=model,
                    messages=messages,
                    response_model=StandardNameComposeBatch,
                    temperature=temp,
                    service="standard-names",
                )
                llm_result, cost, tokens = llm_response
                result.total_cost += cost
                result.total_tokens += tokens
                result.cache_read_tokens += getattr(
                    llm_response, "cache_read_tokens", 0
                )
                result.cache_creation_tokens += getattr(
                    llm_response, "cache_creation_tokens", 0
                )
                logger.debug(
                    "Batch %s: cost=%.4f tokens=%d",
                    group_key,
                    cost,
                    tokens,
                )

                # Collect candidates
                for c in llm_result.candidates:
                    all_candidates.append(c.model_dump())
                result.skipped_count += len(llm_result.skipped)

            except Exception as exc:
                logger.warning(
                    "LLM call failed for model %s batch %s: %s",
                    model,
                    group_key,
                    exc,
                )
                result.batch_errors += 1

    elapsed = time.monotonic() - t0
    result.elapsed_seconds = round(elapsed, 2)
    result.candidates = all_candidates

    # --- Validate grammar ---
    valid = 0
    invalid = 0
    fields_ok = 0
    for c in all_candidates:
        g_valid, f_consistent = validate_candidate(c)
        if g_valid:
            valid += 1
        else:
            invalid += 1
        if f_consistent:
            fields_ok += 1

    result.grammar_valid_count = valid
    result.grammar_invalid_count = invalid
    result.fields_consistent_count = fields_ok

    # --- Derived metrics ---
    n = len(all_candidates)
    if elapsed > 0 and n > 0:
        result.names_per_minute = round(n / elapsed * 60, 1)
    if n > 0:
        result.cost_per_name = round(result.total_cost / n, 6)

    # --- Reference comparison ---
    overlap, ref_total, precision, recall = compare_to_reference(
        all_candidates, reference
    )
    result.reference_overlap = overlap
    result.reference_total = ref_total
    result.reference_precision = round(precision, 4)
    result.reference_recall = round(recall, 4)

    logger.info(
        "Model %s: %d names, %d valid, %d invalid, $%.4f cost, %.1f names/min",
        model,
        n,
        valid,
        invalid,
        result.total_cost,
        result.names_per_minute,
    )

    return result


# ---------------------------------------------------------------------------
# Rich table rendering
# ---------------------------------------------------------------------------


def render_comparison_table(report: BenchmarkReport) -> None:
    """Render a Rich comparison table to stdout."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Check if any result has quality scores
    has_quality = any(r.quality_scores for r in report.results)

    table = Table(
        title="SN Benchmark Results",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Model", style="bold")
    table.add_column("Names", justify="right")
    table.add_column("Valid %", justify="right")
    table.add_column("Fields %", justify="right")
    table.add_column("Ref Match", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Names/min", justify="right")
    table.add_column("$/name", justify="right")
    table.add_column("Cache %", justify="right")
    table.add_column("Errors", justify="right")
    if has_quality:
        table.add_column("Avg Quality", justify="right")
        table.add_column("Avg Doc Len", justify="right")
        table.add_column("Fields Pop%", justify="right")

    for r in report.results:
        n = len(r.candidates)
        valid_pct = f"{r.grammar_valid_count / n * 100:.0f}%" if n else "—"
        fields_pct = f"{r.fields_consistent_count / n * 100:.0f}%" if n else "—"
        ref_match = (
            f"{r.reference_overlap}/{r.reference_total}" if r.reference_total else "—"
        )
        cost_str = f"${r.total_cost:.4f}" if r.total_cost > 0 else "—"
        speed_str = f"{r.names_per_minute:.0f}" if r.names_per_minute > 0 else "—"
        cpn_str = f"${r.cost_per_name:.4f}" if r.cost_per_name > 0 else "—"
        cache_total = r.cache_read_tokens + r.cache_creation_tokens
        cache_pct = (
            f"{r.cache_read_tokens / cache_total * 100:.0f}%"
            if cache_total > 0
            else "—"
        )
        err_str = str(r.batch_errors) if r.batch_errors > 0 else "0"

        row_data = [
            r.model,
            str(n),
            valid_pct,
            fields_pct,
            ref_match,
            cost_str,
            speed_str,
            cpn_str,
            cache_pct,
            err_str,
        ]

        if has_quality:
            qual_str = f"{r.avg_quality_score:.2f}" if r.quality_scores else "—"
            doc_str = f"{r.avg_doc_length:.0f}" if r.quality_scores else "—"
            fp_str = f"{r.avg_fields_populated * 100:.0f}%" if r.quality_scores else "—"
            row_data.extend([qual_str, doc_str, fp_str])

        table.add_row(*row_data)

    console.print()
    console.print(table)

    # Quality distribution table (when reviewer was used)
    if has_quality:
        qual_table = Table(
            title="Quality Distribution",
            show_header=True,
            header_style="bold magenta",
        )
        qual_table.add_column("Model", style="bold")
        qual_table.add_column("Outstanding", justify="right")
        qual_table.add_column("Good", justify="right")
        qual_table.add_column("Adequate", justify="right")
        qual_table.add_column("Poor", justify="right")

        for r in report.results:
            if r.quality_scores:
                dist = r.quality_distribution
                n_reviews = len(r.quality_scores)

                qual_table.add_row(
                    r.model,
                    f"{dist.get('outstanding', 0)} ({dist.get('outstanding', 0) / n_reviews * 100:.0f}%)"
                    if n_reviews
                    else "—",
                    f"{dist.get('good', 0)} ({dist.get('good', 0) / n_reviews * 100:.0f}%)"
                    if n_reviews
                    else "—",
                    f"{dist.get('adequate', 0)} ({dist.get('adequate', 0) / n_reviews * 100:.0f}%)"
                    if n_reviews
                    else "—",
                    f"{dist.get('poor', 0)} ({dist.get('poor', 0) / n_reviews * 100:.0f}%)"
                    if n_reviews
                    else "—",
                )

        console.print()
        console.print(qual_table)

    # Summary line
    reviewer_str = (
        f" | Reviewer: {report.config.reviewer_model}"
        if report.config.reviewer_model
        else ""
    )
    console.print(
        f"\n[dim]Extraction: {report.extraction_count} items | "
        f"Temperature: {report.config.temperature}{reviewer_str} | "
        f"Timestamp: {report.timestamp}[/dim]"
    )

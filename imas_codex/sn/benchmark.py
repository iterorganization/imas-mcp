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

from imas_codex.sn.models import SNComposeBatch

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
# Grammar context builder
# ---------------------------------------------------------------------------


def build_grammar_context() -> dict[str, list[str]]:
    """Build the grammar enum values needed by the compose prompt.

    Returns a dict with keys matching the template variables in
    ``sn/compose_dd.md``: subjects, positions, components, coordinates,
    processes, transformations, geometric_bases, objects, binary_operators.
    """
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
    fields = candidate.get("fields", {})

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
    from imas_codex.sn.benchmark_reference import REFERENCE_NAMES

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
    for model in config.models:
        logger.info("Benchmarking model: %s", model)
        model_result = await _run_model(
            model=model,
            extraction_batches=extraction_batches,
            config=config,
            reference=REFERENCE_NAMES,
        )
        results.append(model_result)

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

    Returns list of batch dicts with keys: group_key, items, existing_names.
    """
    from imas_codex.sn.sources.dd import extract_dd_candidates

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
) -> ModelResult:
    """Run a single model across all extraction batches."""
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt

    grammar_ctx = build_grammar_context()
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

            # Build prompt context
            prompt_context = {
                "items": items,
                "ids_name": group_key,
                "existing_names": list(existing),
                **grammar_ctx,
            }

            try:
                prompt_text = render_prompt("sn/compose_dd", prompt_context)
            except Exception:
                logger.warning("Failed to render prompt for batch %s", group_key)
                result.batch_errors += 1
                continue

            messages = [{"role": "user", "content": prompt_text}]

            try:
                llm_result, cost, tokens = await acall_llm_structured(
                    model=model,
                    messages=messages,
                    response_model=SNComposeBatch,
                    temperature=config.temperature,
                )
                result.total_cost += cost
                result.total_tokens += tokens

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
    table.add_column("Errors", justify="right")

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
        err_str = str(r.batch_errors) if r.batch_errors > 0 else "0"

        table.add_row(
            r.model,
            str(n),
            valid_pct,
            fields_pct,
            ref_match,
            cost_str,
            speed_str,
            cpn_str,
            err_str,
        )

    console.print()
    console.print(table)

    # Summary line
    console.print(
        f"\n[dim]Extraction: {report.extraction_count} items | "
        f"Temperature: {report.config.temperature} | "
        f"Timestamp: {report.timestamp}[/dim]"
    )

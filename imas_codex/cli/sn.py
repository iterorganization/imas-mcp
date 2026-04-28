"""Standard name generation commands."""

from __future__ import annotations

import json
import logging
from typing import Any

import click
from rich.console import Console

from imas_codex.core.physics_domain import PhysicsDomain

logger = logging.getLogger(__name__)
console = Console()

_PHYSICS_DOMAIN_CHOICE = click.Choice(
    [d.value for d in PhysicsDomain], case_sensitive=False
)


@click.group()
def sn() -> None:
    """Standard name generation and management.

    \b
    Run (reconcile / generate / enrich / link / review / regen):
      sn run --source dd [--physics-domain NAME]
      sn run --source signals --facility NAME
      sn run --only link

    \b
    Status:
      sn status
    """
    pass


def _run_sn_loop_cmd(
    *,
    cost_limit: float,
    per_domain_limit: int | None,
    dry_run: bool,
    quiet: bool,
    only_domain: str | None = None,
    verbose: bool = False,
    turn_number: int = 1,
    min_score: float | None = None,
    skip_generate: bool = False,
    skip_enrich: bool = False,
    skip_review: bool = False,
    skip_regen: bool = False,
    source: str = "dd",
    override_edits: list[str] | None = None,
    only: str | None = None,
) -> None:
    """Execute the DD completion loop with Rich progress display.

    Uses the ``run_discovery()`` harness for 3-press shutdown,
    periodic ticker, graph-refresh, and service monitoring.
    Falls back to plain-mode logging when Rich is unavailable.
    """
    import uuid as _uuid

    from rich.console import Console
    from rich.table import Table

    from imas_codex.cli.discover.common import (
        DiscoveryConfig,
        run_discovery,
        setup_logging,
        use_rich_output,
    )
    from imas_codex.standard_names.loop import (
        run_sn_loop,
        summary_table,
    )
    from imas_codex.standard_names.progress import (
        SNLoopState,
        build_sn_loop_stages,
    )

    run_id = str(_uuid.uuid4())
    use_rich = not quiet and not dry_run and use_rich_output()

    # Pre-suppress noisy SN loggers BEFORE any heavy imports/inits so
    # rich-mode startup is silent. Repeated by run_discovery() after
    # display attach (defense in depth).
    if use_rich:
        import logging as _logging

        for mod in (
            "imas_codex.standard_names",
            "imas_codex.graph",
            "imas_codex.embeddings",
            "imas_codex.discovery",
            "imas_codex.llm",
            "litellm",
            "httpx",
            "httpcore",
            "openai",
            "urllib3",
            "neo4j",
        ):
            _logging.getLogger(mod).setLevel(_logging.ERROR)

    # Build Rich display or fall back to plain logging
    display = None
    loop_state: SNLoopState | None = None
    cli_console: Console | None = None
    if use_rich:
        cli_console = Console()
        setup_logging("sn", "sn-run", use_rich=True, verbose=verbose)

        # Cost gauge: graph-backed when available, else returns 0.0
        try:
            from imas_codex.standard_names.graph_ops import (
                aggregate_spend_for_run,
            )

            def _cost_fn() -> float:
                return aggregate_spend_for_run(run_id)

        except ImportError:

            def _cost_fn() -> float:
                return 0.0

        from imas_codex.discovery.base.progress import (
            DataDrivenProgressDisplay,
        )

        target = "full"
        if skip_enrich:
            target = "names"
        elif skip_review:
            target = "compose+enrich"

        loop_state = SNLoopState()

        # Per-group pending counters from the graph (row-level badges).
        # Cypher mirrors enrich_workers.claim_names_for_enrichment() and
        # review/pipeline.py docs-content gate (≥50 chars).
        from imas_codex.graph.client import GraphClient as _GC

        def _count_pending() -> dict[str, int]:
            try:
                with _GC() as gc:
                    rows = list(
                        gc.query(
                            """
                            CALL {
                              MATCH (s:StandardNameSource {status: 'extracted'})
                              RETURN count(s) AS draft
                            }
                            CALL {
                              MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                              WHERE $min_score IS NOT NULL
                                AND sn.reviewer_score_name IS NOT NULL
                                AND sn.reviewer_score_name < $min_score
                                AND coalesce(sn.regen_count, 0) < 1
                              RETURN count(DISTINCT src) AS revise
                            }
                            CALL {
                              MATCH (sn:StandardName)
                              WHERE sn.pipeline_status IN ['named']
                                AND sn.enriched_at IS NULL
                                AND (sn.enrich_claimed_at IS NULL
                                     OR sn.enrich_claimed_at < datetime() - duration({minutes: 30}))
                              RETURN count(sn) AS enrich
                            }
                            CALL {
                              MATCH (sn:StandardName)
                              WHERE sn.validation_status = 'valid'
                                AND sn.reviewed_name_at IS NULL
                              RETURN count(sn) AS review_names
                            }
                            CALL {
                              MATCH (sn:StandardName)
                              WHERE sn.validation_status = 'valid'
                                AND sn.reviewed_name_at IS NOT NULL
                                AND sn.reviewed_docs_at IS NULL
                                AND sn.documentation IS NOT NULL
                                AND size(sn.documentation) >= 50
                              RETURN count(sn) AS review_docs
                            }
                            RETURN draft, revise, enrich, review_names, review_docs
                            """,
                            min_score=min_score,
                        )
                    )
                if not rows:
                    return {
                        "draft": 0,
                        "revise": 0,
                        "enrich": 0,
                        "review_names": 0,
                        "review_docs": 0,
                    }
                r = rows[0]
                return {
                    "draft": int(r.get("draft", 0)),
                    "revise": int(r.get("revise", 0)),
                    "enrich": int(r.get("enrich", 0)),
                    "review_names": int(r.get("review_names", 0)),
                    "review_docs": int(r.get("review_docs", 0)),
                }
            except Exception:
                return {
                    "draft": 0,
                    "revise": 0,
                    "enrich": 0,
                    "review_names": 0,
                    "review_docs": 0,
                }

        _pending_cache: dict[str, tuple[float, dict[str, int]]] = {"v": (0.0, {})}

        def _pending_fn() -> list[tuple[str, int]]:
            import time as _t

            now = _t.monotonic()
            ts, val = _pending_cache["v"]
            if not val or (now - ts) > 1.0:
                val = _count_pending()
                _pending_cache["v"] = (now, val)
            return [
                ("draft", val.get("draft", 0)),
                ("revise", val.get("revise", 0)),
                ("enrich", val.get("enrich", 0)),
                ("rev-n", val.get("review_names", 0)),
                ("rev-d", val.get("review_docs", 0)),
            ]

        display = DataDrivenProgressDisplay(
            facility="sn",
            cost_limit=cost_limit,
            stages=build_sn_loop_stages(
                skip_generate=skip_generate,
                skip_enrich=skip_enrich,
                skip_review=skip_review,
                min_score=min_score,
            ),
            console=cli_console,
            title_suffix="Standard Name Loop",
            mode_label=f"target={target}",
            accumulated_cost_fn=_cost_fn,
            pending_fn=_pending_fn,
        )
        display.set_engine_state(loop_state)
    else:
        setup_logging("sn", "sn-run", use_rich=False, verbose=verbose)
        cli_console = Console(quiet=quiet)
        if not quiet:
            cli_console.print(
                f"[bold]DD completion loop[/bold] "
                f"(budget=${cost_limit:.2f}, turn={turn_number}"
                f"{f', min_score={min_score}' if min_score is not None else ''}"
                f"{', dry-run' if dry_run else ''})"
            )

    # Build harness config — SN loop wants graph + model status at top
    disc_config = DiscoveryConfig(
        domain="standard-names",
        facility="sn",
        facility_config={},  # SN has no facility YAML
        display=display,
        check_graph=not dry_run,
        check_embed=False,
        check_ssh=False,
        check_auth=False,
        check_model=not dry_run,
        verbose=verbose,
        suppress_loggers=[
            "imas_codex.standard_names",
            "imas_codex.graph",
            "imas_codex.embeddings",
            "imas_codex.discovery",
            "imas_codex.llm",
            "litellm",
            "httpx",
            "httpcore",
            "openai",
            "urllib3",
            "neo4j",
        ]
        if use_rich
        else [],
    )

    async def async_main(stop_event, service_monitor):
        summary = await run_sn_loop(
            cost_limit=cost_limit,
            per_domain_limit=per_domain_limit,
            dry_run=dry_run,
            only_domain=only_domain,
            turn_number=turn_number,
            min_score=min_score,
            skip_generate=skip_generate,
            skip_enrich=skip_enrich,
            skip_review=skip_review,
            skip_regen=skip_regen,
            source=source,
            override_edits=override_edits,
            only=only,
            loop_state=loop_state,
            stop_event=stop_event,
        )
        return {"summary": summary}

    result = run_discovery(disc_config, async_main)
    summary = result.get("summary")
    if summary is None:
        return

    row = summary_table(summary)

    if quiet:
        return

    # Print summary table (in both rich and plain mode, after display exits)
    out_console = cli_console or Console()
    table = Table(title=f"Run {row['run_id'][:8]}… (turn {row.get('turn_number', 1)})")
    table.add_column("field", style="cyan")
    table.add_column("value", style="white")
    for key in (
        "turn_number",
        "stop_reason",
        "cost_spent",
        "cost_limit",
        "names_composed",
        "names_enriched",
        "names_reviewed",
        "names_regenerated",
        "elapsed_s",
    ):
        if key in row:
            table.add_row(key, str(row[key]))
    table.add_row("domains_touched", ", ".join(row["domains_touched"]) or "—")
    out_console.print(table)


def _check_pipeline_clear_gate() -> None:
    """Check whether the pipeline version has changed since the last SNRun.

    Queries the graph for the most recent ``SNRun`` node that has a
    ``pipeline_hash`` set.  If the stored composite hash differs from the
    current one **and** there are ``StandardName`` nodes generated after
    that run, print a warning banner and raise ``SystemExit(1)``.

    Best-effort: if the graph is unreachable or the import fails, the
    function returns silently so it never blocks a legitimate first run.
    """
    try:
        import json as _json

        from rich.console import Console as _Console

        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.pipeline_version import (
            compute_pipeline_hash,
            diff_pipeline_hashes,
        )
    except ImportError:
        return  # bootstrap — skip gate

    try:
        current = compute_pipeline_hash()
        current_composite = current["_composite"]

        with GraphClient(timeout=5) as gc:
            # Fetch most recent SNRun that recorded a pipeline_hash
            rows = list(
                gc.query(
                    """
                    MATCH (r:SNRun)
                    WHERE r.pipeline_hash IS NOT NULL
                    RETURN r.pipeline_hash          AS composite,
                           r.pipeline_hash_detail   AS detail,
                           r.started_at             AS started_at,
                           r.id                     AS run_id
                    ORDER BY r.started_at DESC
                    LIMIT 1
                    """
                )
            )
            if not rows:
                return  # no prior run with hash — fresh graph, skip gate

            row = rows[0]
            prev_composite = row["composite"]
            if prev_composite == current_composite:
                return  # no change

            # Hashes differ — check whether there are generated names
            name_count_rows = list(
                gc.query("MATCH (sn:StandardName) RETURN count(sn) AS n")
            )
            name_count = name_count_rows[0]["n"] if name_count_rows else 0
            if name_count == 0:
                return  # empty graph — nothing to protect

            # Compute which keys changed for a useful message
            prev_detail: dict[str, str] = {}
            if row["detail"]:
                try:
                    prev_detail = _json.loads(row["detail"])
                except Exception:  # noqa: BLE001
                    pass
            changed_keys = diff_pipeline_hashes(prev_detail, current)

            _Console(stderr=True).print(
                "\n[bold yellow]⚠  Pipeline version changed since last cycle.[/bold yellow]\n"
                f"   Previous composite hash : [dim]{prev_composite}[/dim]\n"
                f"   Current  composite hash : [dim]{current_composite}[/dim]\n"
                f"   Keys that changed       : [yellow]{', '.join(changed_keys) or '(detail unavailable)'}[/yellow]\n"
                f"   Existing generated names: [cyan]{name_count}[/cyan]\n\n"
                "   Recommendation: run the following command before continuing:\n"
                "     [bold]imas-codex sn clear --all --force --include-sources[/bold]\n\n"
                "   To bypass this check and continue anyway:\n"
                "     [bold]imas-codex sn run --skip-clear-gate ...[/bold]\n"
            )
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        return  # graph unreachable or error — skip gate silently


@sn.command("run")
@click.option(
    "--source",
    type=click.Choice(["dd", "signals"]),
    default="dd",
    show_default=True,
    help="Source to extract candidates from",
)
@click.option(
    "--physics-domain",
    "domain_filter",
    type=_PHYSICS_DOMAIN_CHOICE,
    default=None,
    help=(
        "Filter to a specific physics domain (e.g. magnetics, equilibrium, "
        "transport). Applies to both DD and signals sources. Primary scope "
        "narrower for generation."
    ),
)
@click.option(
    "--facility",
    type=str,
    default=None,
    help="Facility ID (required for signals source)",
)
@click.option(
    "-c",
    "--cost-limit",
    type=float,
    default=5.0,
    help="Maximum LLM cost in USD",
)
@click.option("--dry-run", is_flag=True, help="Preview extraction without LLM calls")
@click.option(
    "--force", is_flag=True, help="Re-generate names for already-named sources"
)
@click.option(
    "--revalidate",
    is_flag=True,
    help=(
        "Before extraction, sweep StandardName nodes with validation_status='pending' "
        "in the current scope (source/domain/ids filters) and clear validated_at so "
        "validate_worker re-runs ISN checks against the current grammar. Use after "
        "an ISN vocab/grammar update to clear legacy quarantines without a full regen."
    ),
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Maximum number of DD paths to process",
)
@click.option(
    "--compose-model",
    type=str,
    default=None,
    help="LLM model for name composition (default: reasoning model)",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("-q", "--quiet", is_flag=True, help="Suppress non-error output")
@click.option(
    "--paths",
    "paths_list",
    type=str,
    multiple=True,
    help=(
        "DD paths to process directly. Accepts multiple --paths flags or "
        "space-separated paths within each flag (e.g., "
        "'--paths eq/.../psi eq/.../q' or '--paths eq/.../psi --paths eq/.../q'). "
        "Bypasses graph query, classifier, and already-named check. "
        "Overrides --physics-domain, --limit, and implies --force."
    ),
)
@click.option(
    "--reset-to",
    type=click.Choice(["extracted", "drafted"]),
    default=None,
    help=(
        "Reset standard names before generating. "
        "'extracted' clears matching SN nodes (full re-run); "
        "'drafted' resets existing drafted names (re-compose only)."
    ),
)
@click.option(
    "--from-model",
    type=str,
    default=None,
    help=(
        "Regenerate names produced by a specific model (substring match). "
        "Example: --from-model gemini matches 'google/gemini-3.1-flash-lite-preview'. "
        "Implies --force."
    ),
)
@click.option(
    "--reset-only",
    is_flag=True,
    default=False,
    help=(
        "Perform --reset-to cleanup then exit without running generation. "
        "Requires --reset-to. Useful for housekeeping without recomposing."
    ),
)
@click.option(
    "--since",
    type=str,
    default=None,
    help=(
        "Only reset/regenerate names with generated_at >= this ISO timestamp "
        "(e.g. '2026-04-19T10:00'). Combines with --reset-to and filters."
    ),
)
@click.option(
    "--before",
    type=str,
    default=None,
    help=(
        "Only reset/regenerate names with generated_at < this ISO timestamp. "
        "Combines with --since for a window."
    ),
)
@click.option(
    "--below-score",
    type=float,
    default=None,
    help=(
        "Only reset/regenerate names with reviewer_score_name < this value "
        "(0.0-1.0 scale). Requires prior `sn review` run."
    ),
)
@click.option(
    "--tier",
    type=str,
    default=None,
    help=(
        "Only reset/regenerate names with review_tier in this comma-separated "
        "list (e.g. 'poor,inadequate'). Requires prior `sn review` run."
    ),
)
@click.option(
    "--retry-quarantined",
    is_flag=True,
    default=False,
    help=("Shortcut: select names with validation_status=quarantined for regen."),
)
@click.option(
    "--retry-skipped",
    is_flag=True,
    default=False,
    help=(
        "Include StandardNameSource records with status=skipped in re-extraction "
        "(their underlying DD paths get re-queued). Useful after unit override "
        "table updates. (status='skipped' will be added in Phase B — it is OK "
        "for this flag to be a no-op today; Phase B will wire it up.)"
    ),
)
@click.option(
    "--retry-vocab-gap",
    is_flag=True,
    default=False,
    help=(
        "Select names with validation_status=quarantined AND a vocab_gap cause "
        "(or StandardNameSource.status=vocab_gap) for regen after ISN vocab "
        "updates."
    ),
)
@click.option(
    "--turn-number",
    "turn_number",
    type=int,
    default=1,
    show_default=True,
    help=(
        "User-supplied turn number for this run. Stamped on every "
        "StandardName touched via ``last_turn_number`` and on the "
        "created SNRun audit node. Does NOT auto-increment — call "
        "``sn run --turn-number 2`` manually to drive the next pass."
    ),
)
@click.option(
    "--min-score",
    "min_score",
    type=float,
    default=None,
    help=(
        "Reviewer-score threshold for regen phase. Names with "
        "``reviewer_score_name < min_score`` (that haven't already been "
        "regenerated) are re-composed with prior reviewer critique "
        "injected into the compose prompt. When None (default), the "
        "regen phase is skipped. Typical value: 0.6."
    ),
)
@click.option(
    "--skip-enrich",
    is_flag=True,
    default=False,
    help="Skip the enrich phase (documentation generation).",
)
@click.option(
    "--skip-review",
    is_flag=True,
    default=False,
    help="Skip the review phase (6-dimensional scoring).",
)
@click.option(
    "--skip-regen",
    is_flag=True,
    default=False,
    help=(
        "Skip the regen phase even when ``--min-score`` is set. "
        "Useful for A/B testing extract → compose → enrich → review "
        "in isolation from regeneration."
    ),
)
@click.option(
    "--target",
    type=click.Choice(["names", "docs"], case_sensitive=False),
    default="names",
    show_default=True,
    help=(
        "Which generation pass to run. 'names' runs the name-only compose "
        "pass (grammar + unit only, wide batches: group paths by "
        "physics_domain × unit with a lean user prompt). 'docs' routes "
        "through the five-phase enrich pipeline to fill description/"
        "documentation on already-named entries (EXTRACT→CONTEXTUALISE→"
        "DOCUMENT→VALIDATE→PERSIST)."
    ),
)
@click.option(
    "--name-only-batch-size",
    type=int,
    default=None,
    help=(
        "Max items per name-only batch. Defaults to "
        "[tool.imas-codex.sn-generate].name-only-batch-size in pyproject.toml "
        "(50). Only meaningful for --target=names."
    ),
)
@click.option(
    "--docs-status",
    "docs_status_filter",
    default="named",
    show_default=True,
    help=(
        "Review status(es) to enrich from when --target=docs "
        "(comma-separated or repeated). Ignored for other targets."
    ),
)
@click.option(
    "--docs-batch-size",
    type=int,
    default=None,
    help=(
        "LLM batch size for --target=docs. Defaults to "
        "[tool.imas-codex.sn-generate].docs-batch-size in pyproject.toml "
        "(12), then [tool.imas-codex.sn-enrich].batch-size as a fallback. "
        "Ignored for other targets."
    ),
)
@click.option(
    "--single-pass",
    is_flag=True,
    default=False,
    help=(
        "Force the single-pass extract → compose pipeline instead of "
        "the default domain-iterating completion loop. Useful for CI "
        "and regression tests."
    ),
)
@click.option(
    "--only",
    "only_phase",
    type=click.Choice(
        [
            "reconcile",
            "extract",
            "compose",
            "validate",
            "consolidate",
            "persist",
            "review",
            "link",
        ],
        case_sensitive=False,
    ),
    default=None,
    help=(
        "Run only this phase — all others are skipped. "
        "extract/compose/validate/consolidate/persist select the generate phase."
    ),
)
@click.option(
    "--override-edits",
    multiple=True,
    help=(
        "Standard name IDs to bypass pipeline protection for. "
        "Allows overwriting catalog-edited fields on these names only. "
        "Repeatable: --override-edits foo --override-edits bar."
    ),
)
@click.option(
    "--skip-clear-gate",
    is_flag=True,
    default=False,
    help=(
        "Bypass the pipeline-version change check. Normally ``sn run`` "
        "exits non-zero when prompt files or ISN vocab have changed since "
        "the last SNRun node and there are existing generated names. "
        "Pass this flag to suppress the gate and continue anyway."
    ),
)
@click.option(
    "--reviewer-profile",
    "reviewer_profile",
    type=click.Choice(
        ["default", "pilot", "opus-only", "haiku-only"], case_sensitive=False
    ),
    default="default",
    show_default=True,
    envvar="IMAS_CODEX_SN_REVIEW_PROFILE",
    help=(
        "Reviewer model chain profile for the review phase. "
        "'default' → Opus+GPT-5.4+Sonnet (3-model RD-quorum). "
        "'pilot' → Haiku×2+Opus arbiter (~85%% cost reduction). "
        "'opus-only' → single Opus reviewer. "
        "'haiku-only' → single Haiku reviewer (cheapest). "
        "Also read from IMAS_CODEX_SN_REVIEW_PROFILE env var."
    ),
)
def sn_run(
    source: str,
    domain_filter: str | None,
    facility: str | None,
    cost_limit: float,
    dry_run: bool,
    force: bool,
    limit: int | None,
    compose_model: str | None,
    verbose: bool,
    quiet: bool,
    paths_list: tuple[str, ...],
    reset_to: str | None,
    from_model: str | None,
    revalidate: bool,
    reset_only: bool,
    since: str | None,
    before: str | None,
    below_score: float | None,
    tier: str | None,
    retry_quarantined: bool,
    retry_skipped: bool,
    retry_vocab_gap: bool,
    turn_number: int,
    min_score: float | None,
    skip_enrich: bool,
    skip_review: bool,
    skip_regen: bool,
    target: str,
    name_only_batch_size: int | None,
    docs_status_filter: str,
    docs_batch_size: int | None,
    single_pass: bool,
    only_phase: str | None,
    override_edits: tuple[str, ...],
    skip_clear_gate: bool,
    reviewer_profile: str,
) -> None:
    """Generate standard names from a source.

    \b
    Scope routing:
      - With --paths: single-pass pipeline (explicit paths too narrow for looping)
      - Without --paths: domain-iterating completion loop (default)
      - With --single-pass: always single-pass pipeline

    \b
    Examples:
      imas-codex sn run -c 50                                 # loop over all domains
      imas-codex sn run --physics-domain equilibrium -c 5     # loop, one domain
      imas-codex sn run --physics-domain magnetics --dry-run
      imas-codex sn run --source signals --facility tcv --physics-domain magnetics
      imas-codex sn run --paths equilibrium/time_slice/profiles_1d/psi --paths equilibrium/time_slice/profiles_1d/q
      imas-codex sn run --single-pass --paths equilibrium/time_slice/profiles_1d/psi -c 1  # single compose pass on explicit paths
      imas-codex sn run --reset-to drafted --reset-only
      imas-codex sn run --reset-to drafted --below-score 0.6 --reset-only
      imas-codex sn run --turn-number 2 --min-score 0.6 -c 5  # regen reviewed names below 0.6
      imas-codex sn run --only link                   # resolve links only
      imas-codex sn run --override-edits foo --override-edits bar  # bypass protection on foo, bar
      imas-codex sn run --reviewer-profile pilot -c 5  # use cheap Haiku+Opus reviewer
    """
    import os as _os

    # --- Reviewer profile: propagate via env var so the review pipeline picks
    # it up automatically wherever it reads get_sn_review_names_models() /
    # get_sn_review_disagreement_threshold().
    reviewer_profile = reviewer_profile.lower()
    if reviewer_profile != "default":
        _os.environ["IMAS_CODEX_SN_REVIEW_PROFILE"] = reviewer_profile

    # --- Pipeline-version clear gate ---
    # Check if prompt/vocab/code has changed since the last SNRun.
    # Exits non-zero with a warning banner unless --skip-clear-gate is set
    # or there are no existing generated names (fresh graph).
    if not skip_clear_gate and not dry_run:
        _check_pipeline_clear_gate()

    # --- Apply --only overrides ---
    if only_phase:
        from imas_codex.standard_names.turn import skip_flags_from_only

        overrides = skip_flags_from_only(only_phase)
        if overrides.get("skip_generate", False):
            # When --only skips generate, also skip related pre-processing
            force = False
        if overrides.get("skip_enrich", False):
            skip_enrich = True
        if overrides.get("skip_review", False):
            skip_review = True
        if overrides.get("skip_regen", False):
            skip_regen = True
        # skip_generate handled via the overrides dict below
        skip_generate_from_only = overrides.get("skip_generate", False)
    else:
        skip_generate_from_only = False
    # --- Resolve --target ---
    target_normalized = target.lower()
    # Downstream compose routing uses a derived name_only boolean.
    name_only = target_normalized == "names"

    # --target=names skips enrich (docs-enrichment) so its budget is not
    # spent before compose/review_names get their share.  Enrich requires
    # a prior compose pass to exist; when the user wants only names review
    # they explicitly do not want docs enrichment to consume budget.
    if name_only:
        skip_enrich = True

    # --target=docs routes through the five-phase enrich pipeline.
    # The rotator and the compose single-pass do not produce docs on
    # already-named entries; delegate to the enrich engine instead.
    if target_normalized == "docs":
        # Parse --physics-domain from domain_filter (single value) into the
        # list[str] form the enrich pipeline expects.
        _docs_domains: list[str] | None = [domain_filter] if domain_filter else None
        _run_sn_docs_generation(
            domain_list=_docs_domains,
            status_filter=docs_status_filter,
            cost_limit=cost_limit,
            limit=limit,
            batch_size=docs_batch_size,
            dry_run=dry_run,
            force=force,
            model_override=compose_model,
            verbose=verbose,
            quiet=quiet,
        )
        return

    # Scope-routing: --paths → single-pass; else → loop (unless --single-pass).
    # The loop drives the full extract→enrich→review→regen cycle per domain
    # and persists an SNRun audit node. --physics-domain is forwarded to scope
    # the loop to a single domain (single-element iteration).
    use_loop = not single_pass and not paths_list and source == "dd"

    # Coerce override_edits tuple to list for downstream
    _override_edits = list(override_edits) if override_edits else None

    if use_loop:
        _run_sn_loop_cmd(
            cost_limit=cost_limit,
            per_domain_limit=limit,
            dry_run=dry_run,
            quiet=quiet,
            only_domain=domain_filter,
            verbose=verbose,
            turn_number=turn_number,
            min_score=min_score,
            skip_generate=skip_generate_from_only,
            skip_enrich=skip_enrich,
            skip_review=skip_review,
            skip_regen=skip_regen,
            source=source,
            override_edits=_override_edits,
            only=only_phase,
        )
        return

    # --ids has been removed from this command; scope narrowing is domain-based
    # so it works uniformly across DD and facility-signals sources.
    ids_filter: str | None = None

    # Validate: signals source requires facility
    if source == "signals" and not facility:
        raise click.UsageError("--facility is required when --source is signals")

    # --paths implies DD source, force, and overrides filters
    # Flatten multiple --paths args and space-separated paths within each arg
    flat_paths = " ".join(paths_list).split() if paths_list else []
    if flat_paths:
        source = "dd"
        domain_filter = None
        limit = None
        force = True  # Targeted paths always regenerate

        # Resolve wildcard patterns (e.g., "*/profiles_1d/q" or "equilibrium/*/data")
        raw_paths = flat_paths
        resolved_paths = []
        has_wildcards = any("*" in p for p in raw_paths)

        if has_wildcards:
            import re

            from imas_codex.graph.client import GraphClient

            _MAX_WILDCARD_MATCHES = 50

            with GraphClient() as gc:
                for pattern in raw_paths:
                    if "*" in pattern:
                        # Escape regex metacharacters except *, then convert * to [^/]+
                        escaped = re.escape(pattern).replace(r"\*", "[^/]+")
                        regex = f"^{escaped}$"
                        matches = list(
                            gc.query(
                                """
                                MATCH (n:IMASNode)
                                WHERE n.id =~ $regex
                                  AND NOT (n.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
                                RETURN n.id AS path
                                ORDER BY n.id
                                LIMIT $max_matches
                                """,
                                regex=regex,
                                max_matches=_MAX_WILDCARD_MATCHES,
                            )
                        )
                        found = [r["path"] for r in matches]
                        if found:
                            console.print(
                                f"  [dim]{pattern}[/dim] → {len(found)} paths"
                            )
                            resolved_paths.extend(found)
                        else:
                            console.print(
                                f"  [yellow]⚠ {pattern}[/yellow] — no matches"
                            )
                    else:
                        resolved_paths.append(pattern)

            # Deduplicate preserving order
            seen: set[str] = set()
            unique_paths = []
            for p in resolved_paths:
                if p not in seen:
                    seen.add(p)
                    unique_paths.append(p)
            resolved_paths = unique_paths

            console.print(
                f"  Resolved {len(resolved_paths)} unique paths from "
                f"{len(raw_paths)} patterns"
            )
            resolved_paths_final = resolved_paths
        else:
            # No wildcards — just use raw paths, still deduplicate
            seen_paths: set[str] = set()
            unique = []
            for p in raw_paths:
                if p not in seen_paths:
                    seen_paths.add(p)
                    unique.append(p)
            resolved_paths_final = unique
    else:
        resolved_paths_final = None

    # --from-model implies --force (selecting by model only makes sense for regeneration)
    if from_model:
        force = True

    # Validate --reset-only
    if reset_only and reset_to is None:
        raise click.UsageError("--reset-only requires --reset-to")

    # Build filter kwargs for reset/clear functions.
    _tiers = [t.strip() for t in tier.split(",")] if tier else None
    _validation_status: str | None = None
    if retry_quarantined:
        _validation_status = "quarantined"
    _reset_filter_kwargs: dict[str, Any] = {
        "since": since,
        "before": before,
        "below_score": below_score,
        "tiers": _tiers,
        "validation_status": _validation_status,
    }

    # Handle --reset-to before the main pipeline
    if reset_to is not None and not dry_run:
        source_arg = "dd" if source == "dd" else "signals"
        from imas_codex.standard_names.graph_ops import (
            clear_standard_names,
            reset_standard_names,
        )

        if reset_to == "extracted":
            n = clear_standard_names(
                source_filter=source_arg,
                ids_filter=ids_filter,
                **_reset_filter_kwargs,
            )
            console.print(
                f"[yellow]--reset-to extracted:[/yellow] cleared {n} SN nodes"
            )
        elif reset_to == "drafted":
            n = reset_standard_names(
                from_status="drafted",
                source_filter=source_arg,
                ids_filter=ids_filter,
                **_reset_filter_kwargs,
            )
            console.print(f"[yellow]--reset-to drafted:[/yellow] reset {n} SN nodes")

    if reset_only:
        console.print(
            "[green]--reset-only:[/green] reset complete, exiting without generation"
        )
        return

    # Log Phase B/C flags that are pending wire-up
    if retry_skipped:
        logger.info("--retry-skipped set (pending Phase B wire-up)")
    if retry_vocab_gap:
        logger.info("--retry-vocab-gap set (pending Phase B wire-up)")

    # Handle --revalidate: clear validated_at on pending SNs in current scope so
    # validate_worker re-runs ISN checks without a full regen. Safe with any source.
    if revalidate and not dry_run:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            where_clauses = [
                "sn.validation_status = 'pending'",
                "sn.validated_at IS NOT NULL",
            ]
            params: dict[str, Any] = {}
            if domain_filter:
                where_clauses.append("$domain IN sn.physics_domain")
                params["domain"] = domain_filter
            if source == "dd":
                where_clauses.append(
                    "EXISTS { MATCH (sn)<-[:HAS_STANDARD_NAME]-(:IMASNode) }"
                )
            elif source == "signals":
                where_clauses.append(
                    "EXISTS { MATCH (sn)<-[:HAS_STANDARD_NAME]-(:FacilitySignal) }"
                )
            q = f"""
                MATCH (sn:StandardName)
                WHERE {" AND ".join(where_clauses)}
                WITH sn, sn.id AS id
                SET sn.validated_at = NULL, sn.claimed_at = NULL, sn.claim_token = NULL
                RETURN count(sn) AS n
            """
            rows = list(gc.query(q, **params))
            n = rows[0]["n"] if rows else 0
            console.print(
                f"[yellow]--revalidate:[/yellow] cleared validated_at on {n} pending SN node(s)"
            )

    from imas_codex.discovery.base.llm import set_litellm_offline_env

    set_litellm_offline_env()

    from imas_codex.cli.discover.common import (
        DiscoveryConfig,
        make_log_print,
        run_discovery,
        setup_logging,
        use_rich_output,
    )

    use_rich = use_rich_output()
    console_obj = setup_logging("sn", "sn", use_rich, verbose=verbose)
    log_print = make_log_print("sn", console_obj)

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Determine effective facility for state
    effective_facility = facility if source == "signals" else "dd"

    log_print("\n[bold]Standard Name Build[/bold]")
    log_print(f"  Source: {source}")
    if resolved_paths_final:
        log_print(f"  Targeted paths: {len(resolved_paths_final)} paths")
    if domain_filter:
        log_print(f"  Domain filter: {domain_filter}")
    if facility:
        log_print(f"  Facility: {facility}")
    if dry_run:
        log_print("  Mode: dry run")
    if force:
        log_print("  Force: re-generating all names")
    if from_model:
        log_print(f"  From model: {from_model} (substring match)")
    if limit:
        log_print(f"  Limit: {limit} paths")
    if compose_model:
        log_print(f"  Compose model: {compose_model}")
    log_print(f"  Cost limit: ${cost_limit:.2f}")
    log_print("")

    from imas_codex.standard_names.budget import BudgetManager
    from imas_codex.standard_names.pipeline import run_sn_pipeline
    from imas_codex.standard_names.state import StandardNameBuildState

    # Build progress display
    display = None
    if use_rich and not quiet:
        try:
            from imas_codex.standard_names.progress import StandardNameProgressDisplay

            display = StandardNameProgressDisplay(
                source=source,
                console=console_obj,
                cost_limit=cost_limit,
                mode_label="DRY RUN" if dry_run else None,
            )
        except Exception:
            logger.debug("Could not create progress display", exc_info=True)

    # Resolve name-only batch size from pyproject default when unspecified.
    if name_only_batch_size is None:
        from imas_codex.settings import _get_section

        name_only_batch_size = int(
            _get_section("sn-run").get("name-only-batch-size", 50)
        )
    if name_only:
        logger.info("Mode: name-only (batch_size=%d)", name_only_batch_size)

    state = StandardNameBuildState(
        facility=effective_facility,
        source=source,
        ids_filter=ids_filter,
        domain_filter=domain_filter,
        facility_filter=facility,
        paths_list=resolved_paths_final,
        cost_limit=cost_limit,
        dry_run=dry_run,
        force=force,
        regen=min_score is not None,
        min_score=min_score,
        turn_number=turn_number,
        limit=limit,
        compose_model=compose_model,
        from_model=from_model,
        name_only=name_only,
        name_only_batch_size=name_only_batch_size,
        budget_manager=BudgetManager(cost_limit),
    )

    if display:
        display.set_engine_state(state)

    async def _run(stop_event, service_monitor):
        if service_monitor:
            state.service_monitor = service_monitor
        await run_sn_pipeline(
            state,
            stop_event=stop_event,
            on_worker_status=display.on_worker_status if display else None,
        )
        return state.stats

    config = DiscoveryConfig(
        facility=effective_facility,
        domain="sn",
        facility_config={},
        display=display,
        check_graph=True,
        check_embed=False,
        check_ssh=False,
        check_auth=source != "dd",  # signals source might need auth
        check_model=not dry_run,
        model_section="sn-run",
        suppress_loggers=[
            "imas_codex.standard_names",
        ],
        verbose=verbose,
    )

    result = run_discovery(config, _run)

    # Print summary
    if result:
        extracted = result.get("extract_count", 0)
        composed = result.get("compose_count", 0)
        attached = result.get("attachments", 0)
        validated = result.get("validate_valid", 0)
        compose_cost = result.get("compose_cost", 0.0)
        compose_model_name = result.get("compose_model", "")
        parts = [
            f"Extracted: {extracted}",
            f"Composed: {composed}",
        ]
        if attached:
            parts.append(f"Attached: {attached}")
        parts.append(f"Validated: {validated}")
        if compose_cost > 0:
            parts.append(f"Cost: ${compose_cost:.4f}")
        log_print(", ".join(parts))
        if compose_model_name:
            log_print(f"Model: {compose_model_name}")
        if dry_run:
            log_print("(dry run — no LLM calls or graph writes)")


@sn.command("benchmark")
@click.option(
    "--source",
    type=click.Choice(["dd"]),
    default="dd",
    help="Source to extract candidates from",
)
@click.option(
    "--ids",
    "ids_filter",
    type=str,
    default=None,
    help="Filter to specific IDS (for DD source)",
)
@click.option(
    "--physics-domain",
    "domain_filter",
    type=_PHYSICS_DOMAIN_CHOICE,
    default=None,
    help="Filter to physics domain.",
)
@click.option(
    "--facility",
    type=str,
    default=None,
    help="Facility ID (reserved for future signals source)",
)
@click.option(
    "--models",
    type=str,
    default=None,
    help="Comma-separated model list. Defaults to [sn.benchmark].compose-models.",
)
@click.option(
    "--max-candidates",
    type=int,
    default=50,
    help="Maximum extraction candidates",
)
@click.option(
    "--runs",
    type=int,
    default=1,
    help="Runs per model for consistency check",
)
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    help="LLM temperature (0.0 for reproducibility)",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="JSON report output path",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--reviewer-model",
    type=str,
    default=None,
    help="Judge model for quality scoring. Defaults to [sn.benchmark].reviewer-model.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help=(
        "Re-run against DD paths that already have a StandardNameSource. "
        "Required when the target IDS has been fully processed (most of the "
        "DD catalog in mature deployments)."
    ),
)
@click.option(
    "--review-target",
    type=click.Choice(["names"]),
    default="names",
    show_default=True,
    help=(
        "Reviewer rubric. 'names' uses the 4-dim name rubric "
        "(sn/review_names, 0-80) matching the compose-stage output."
    ),
)
def sn_benchmark(
    source: str,
    ids_filter: str | None,
    domain_filter: str | None,
    facility: str | None,
    models: str | None,
    max_candidates: int,
    runs: int,
    temperature: float,
    output: str | None,
    verbose: bool,
    reviewer_model: str | None,
    force: bool,
    review_target: str,
) -> None:
    """Benchmark LLM models on standard name generation.

    Runs a fixed dataset through multiple models and compares results
    on grammar validity, reference overlap, cost, and speed.

    When --models is omitted, loads the model list from
    [tool.imas-codex.sn.benchmark].compose-models in pyproject.toml.

    \b
    Examples:
      imas-codex sn benchmark --ids equilibrium
      imas-codex sn benchmark --models anthropic/claude-sonnet-4.6,openai/gpt-5.4
      imas-codex sn benchmark --max-candidates 20 -v
      imas-codex sn benchmark --reviewer-model anthropic/claude-opus-4.6
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    from imas_codex.settings import (
        get_sn_benchmark_compose_models,
        get_sn_benchmark_reviewer_model,
    )

    # Resolve model list: CLI flag → pyproject.toml → built-in defaults
    if models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]
    else:
        model_list = get_sn_benchmark_compose_models()

    if not model_list:
        raise click.UsageError(
            "No models configured. Pass --models or set "
            "[tool.imas-codex.sn.benchmark].compose-models in pyproject.toml."
        )

    # Resolve reviewer model: CLI flag → pyproject.toml → built-in default
    if reviewer_model is None:
        reviewer_model = get_sn_benchmark_reviewer_model()

    from imas_codex.standard_names.benchmark import (
        BenchmarkConfig,
        render_comparison_table,
        run_benchmark,
    )

    config = BenchmarkConfig(
        models=model_list,
        source=source,
        ids_filter=ids_filter,
        domain_filter=domain_filter,
        facility=facility,
        max_candidates=max_candidates,
        runs_per_model=runs,
        temperature=temperature,
        reviewer_model=reviewer_model,
        force=force,
        review_target=review_target,
    )

    console.print("[bold]SN Benchmark[/bold]")
    console.print(f"  Models: {', '.join(model_list)}")
    console.print(f"  Source: {source}")
    if ids_filter:
        console.print(f"  IDS filter: {ids_filter}")
    if domain_filter:
        console.print(f"  Domain filter: {domain_filter}")
    console.print(f"  Max candidates: {max_candidates}")
    console.print(f"  Runs per model: {runs}")
    console.print(f"  Temperature: {temperature}")
    console.print(f"  Reviewer (judge): {reviewer_model}")
    console.print()

    from imas_codex.cli.utils import run_async

    report = run_async(run_benchmark(config))

    # Display comparison table
    render_comparison_table(report)

    # Save JSON report
    if output is None:
        ts = report.timestamp.replace(":", "").replace("-", "")[:15]
        output = f"sn_benchmark_{ts}.json"

    from pathlib import Path

    out_path = Path(output)
    out_path.write_text(report.to_json())
    console.print(f"\n[green]Report saved:[/green] {out_path}")


@sn.command("coverage")
@click.option(
    "--physics-domain",
    "physics_domain",
    type=_PHYSICS_DOMAIN_CHOICE,
    default=None,
    help="Restrict eligibility counts to this physics domain.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Emit machine-readable JSON instead of rich tables.",
)
def sn_coverage(physics_domain: str | None, as_json: bool) -> None:
    """Pre-run coverage report: how many names do we expect to mint?

    Prints a three-section report:

    \b
      1. DD extract eligibility — leaves admitted by the B3' invariant.
      2. Already-minted coverage — existing StandardName catalog.
      3. Work remaining — uncovered paths + rough LLM cost estimate.

    \b
    Examples:
      imas-codex sn coverage
      imas-codex sn coverage --physics-domain equilibrium
      imas-codex sn coverage --json | jq .to_compose
    """
    from imas_codex.standard_names.coverage import compute_coverage

    try:
        report = compute_coverage(physics_domain=physics_domain)
    except Exception as exc:  # pragma: no cover
        console.print(f"[red]Error computing coverage:[/red] {exc}")
        raise SystemExit(1) from exc

    if as_json:
        click.echo(report.to_json())
        return

    # --- Rich output --------------------------------------------------------
    from rich.table import Table

    scope_label = (
        f"[bold cyan]{physics_domain}[/bold cyan]"
        if physics_domain
        else "[bold cyan]all domains[/bold cyan]"
    )
    console.print(f"\n[bold]SN Coverage Report[/bold] — scope: {scope_label}\n")

    # Section 1: DD eligibility
    elig_table = Table(
        title="1 · DD Extract Eligibility (B3' invariant)", show_header=True
    )
    elig_table.add_column("Metric", style="cyan")
    elig_table.add_column("Count", justify="right")
    elig_table.add_row(
        "[bold]Total eligible leaves[/bold]", f"[bold]{report.eligible_total:,}[/bold]"
    )
    elig_table.add_row(
        "  … with HAS_ERROR edges (B9 parents)", f"{report.eligible_with_errors:,}"
    )
    console.print(elig_table)

    cat_table = Table(title="By node_category", show_header=True)
    cat_table.add_column("node_category", style="dim")
    cat_table.add_column("Count", justify="right")
    for k, v in sorted(report.eligible_by_category.items(), key=lambda x: -x[1]):
        cat_table.add_row(k, f"{v:,}")
    console.print(cat_table)

    nt_table = Table(title="By node_type", show_header=True)
    nt_table.add_column("node_type", style="dim")
    nt_table.add_column("Count", justify="right")
    for k, v in sorted(report.eligible_by_node_type.items(), key=lambda x: -x[1]):
        nt_table.add_row(k, f"{v:,}")
    console.print(nt_table)

    # Only show domain table when not filtered (it's redundant when filtered)
    if not physics_domain:
        dom_table = Table(title="By physics_domain (top 15)", show_header=True)
        dom_table.add_column("physics_domain", style="dim")
        dom_table.add_column("Count", justify="right")
        for k, v in sorted(report.eligible_by_domain.items(), key=lambda x: -x[1])[:15]:
            dom_table.add_row(k, f"{v:,}")
        console.print(dom_table)

    # Section 2: Already-minted
    console.print()
    minted_table = Table(title="2 · Already-Minted Coverage", show_header=True)
    minted_table.add_column("Metric", style="cyan")
    minted_table.add_column("Count", justify="right")
    minted_table.add_row(
        "[bold]Total StandardName nodes[/bold]", f"[bold]{report.sn_total:,}[/bold]"
    )
    minted_table.add_row(
        "  Error-sibling names (deterministic:dd_error_modifier)",
        f"{report.error_siblings_minted:,}",
    )
    minted_table.add_row(
        "  IMASNodes covered (HAS_STANDARD_NAME)", f"{report.covered_parents:,}"
    )
    console.print(minted_table)

    ps_table = Table(title="By pipeline_status", show_header=True)
    ps_table.add_column("pipeline_status", style="dim")
    ps_table.add_column("Count", justify="right")
    for k, v in sorted(report.sn_by_pipeline_status.items(), key=lambda x: -x[1]):
        ps_table.add_row(k, f"{v:,}")
    console.print(ps_table)

    vs_table = Table(title="By validation_status", show_header=True)
    vs_table.add_column("validation_status", style="dim")
    vs_table.add_column("Count", justify="right")
    for k, v in sorted(report.sn_by_validation_status.items(), key=lambda x: -x[1]):
        vs_table.add_row(k, f"{v:,}")
    console.print(vs_table)

    # Section 3: Work remaining
    console.print()
    work_table = Table(title="3 · Work Remaining", show_header=True)
    work_table.add_column("Metric", style="cyan")
    work_table.add_column("Value", justify="right")
    work_table.add_row(
        "[bold]To compose (uncovered leaves)[/bold]",
        f"[bold]{report.to_compose:,}[/bold]",
    )
    work_table.add_row("  … with HAS_ERROR edges", f"{report.to_compose_with_errors:,}")
    work_table.add_row(
        "  Expected error siblings (3×)", f"{report.expected_error_siblings:,}"
    )

    if report.cost_per_name is not None:
        work_table.add_row(
            "Avg cost/name (from SNRun telemetry)",
            f"${report.cost_per_name:.5f}",
        )
        if report.estimated_compose_cost is not None:
            work_table.add_row(
                "Estimated total compose cost",
                f"${report.estimated_compose_cost:.2f}",
            )
    else:
        work_table.add_row(
            "Cost estimate",
            "[dim]unknown — no prior SNRun telemetry[/dim]",
        )
    console.print(work_table)
    console.print()


@sn.command("status")
def sn_status() -> None:
    """Show standard name statistics."""
    from imas_codex.graph.client import GraphClient

    try:
        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (sn:StandardName)
                RETURN count(sn) AS total,
                       count(CASE WHEN 'dd' IN sn.source_types THEN 1 END) AS from_dd,
                       count(CASE WHEN 'signals' IN sn.source_types THEN 1 END) AS from_signals,
                       count(CASE WHEN 'manual' IN sn.source_types THEN 1 END) AS from_manual
            """
            )
            row = next(iter(result), None)
            if row:
                console.print(f"[bold]Standard Names:[/bold] {row['total']}")
                console.print(f"  From DD: {row['from_dd']}")
                console.print(f"  From signals: {row['from_signals']}")
                console.print(f"  From manual: {row['from_manual']}")
            else:
                console.print("No standard names in graph")

            # Validation status breakdown
            vstatus_result = gc.query(
                """
                MATCH (sn:StandardName)
                RETURN coalesce(sn.validation_status, 'unset') AS status,
                       count(sn) AS cnt
                ORDER BY cnt DESC
            """
            )
            if vstatus_result:
                from rich.table import Table as RichTable

                console.print()
                console.print("[bold]Validation Status[/bold]")
                vtable = RichTable(show_header=True)
                vtable.add_column("Status")
                vtable.add_column("Count", justify="right")
                for vrow in vstatus_result:
                    vtable.add_row(vrow["status"], str(vrow["cnt"]))
                console.print(vtable)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")

    # StandardNameSource status
    from rich.table import Table

    from imas_codex.standard_names.graph_ops import get_standard_name_source_stats

    source_stats = get_standard_name_source_stats()
    if source_stats:
        console.print()
        console.print("[bold]StandardNameSource Pipeline Status[/bold]")
        source_table = Table(show_header=True)
        source_table.add_column("Status")
        source_table.add_column("Count", justify="right")
        total = 0
        for status_name in [
            "extracted",
            "composed",
            "attached",
            "vocab_gap",
            "failed",
            "stale",
            "skipped",
        ]:
            count = source_stats.get(status_name, 0)
            total += count
            if count > 0:
                source_table.add_row(status_name, str(count))
        source_table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")
        console.print(source_table)

    # Skipped sources breakdown by reason (DD unit overrides, etc.)
    try:
        from imas_codex.standard_names.graph_ops import get_skipped_source_counts

        skip_counts = get_skipped_source_counts()
    except Exception as exc:  # pragma: no cover — graph connection issues
        skip_counts = {}
        logger.debug("Could not fetch skipped source counts: %s", exc)

    if skip_counts:
        console.print()
        console.print("[bold]Skipped sources (by reason)[/bold]")
        skip_table = Table(show_header=True)
        skip_table.add_column("Skip Reason")
        skip_table.add_column("Count", justify="right")
        skip_total = 0
        for reason, count in skip_counts.items():
            skip_table.add_row(reason, str(count))
            skip_total += count
        skip_table.add_row("[bold]Total[/bold]", f"[bold]{skip_total}[/bold]")
        console.print(skip_table)

    # Latest SNRun (from sn run rotator)
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            rr_rows = list(
                gc.query(
                    """
                MATCH (rr:SNRun)
                RETURN rr.id AS id,
                       rr.started_at AS started_at,
                       rr.ended_at AS ended_at,
                       rr.stop_reason AS stop_reason,
                       rr.passes AS passes,
                       rr.cost_spent AS cost_spent,
                       rr.cost_limit AS cost_limit,
                       rr.names_composed AS names_composed,
                       rr.names_enriched AS names_enriched,
                       rr.names_reviewed AS names_reviewed,
                       rr.names_regenerated AS names_regenerated,
                       rr.domains_touched AS domains_touched
                ORDER BY rr.started_at DESC
                LIMIT 1
                """
                )
            )
    except Exception as exc:  # pragma: no cover
        rr_rows = []
        logger.debug("Could not fetch latest SNRun: %s", exc)

    if rr_rows:
        rr = rr_rows[0]
        console.print()
        console.print("[bold]Latest Rotation (sn run)[/bold]")
        rr_table = Table(show_header=True)
        rr_table.add_column("Field")
        rr_table.add_column("Value")
        rr_table.add_row("id", str(rr["id"]))
        rr_table.add_row("started_at", str(rr["started_at"]))
        rr_table.add_row("ended_at", str(rr["ended_at"] or "—"))
        rr_table.add_row("stop_reason", str(rr["stop_reason"] or "—"))
        rr_table.add_row("passes", str(rr["passes"] or 0))
        rr_table.add_row(
            "cost",
            f"${float(rr['cost_spent'] or 0):.4f} / ${float(rr['cost_limit'] or 0):.2f}",
        )
        rr_table.add_row("names_composed", str(rr["names_composed"] or 0))
        rr_table.add_row("names_enriched", str(rr["names_enriched"] or 0))
        rr_table.add_row("names_reviewed", str(rr["names_reviewed"] or 0))
        rr_table.add_row("names_regenerated", str(rr["names_regenerated"] or 0))
        rr_table.add_row(
            "domains_touched",
            ", ".join(rr["domains_touched"] or []) or "—",
        )
        console.print(rr_table)

    # --- Linking integrity & review cost --------------------------------
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            integrity = next(
                iter(
                    gc.query(
                        """
                        MATCH (sn:StandardName)
                        WITH count(sn) AS total_sn,
                             count(CASE WHEN NOT (sn)<-[:PRODUCED_NAME]-()
                                         AND sn.model <> 'deterministic:dd_error_modifier'
                                   THEN 1 END) AS orphan_sn,
                             count(CASE WHEN sn.model = 'deterministic:dd_error_modifier'
                                   THEN 1 END) AS error_siblings
                        MATCH (s:StandardNameSource)
                        WITH total_sn, orphan_sn, error_siblings,
                             count(CASE WHEN s.status IN ['composed','attached']
                                         AND NOT (s)-[:PRODUCED_NAME]->()
                                   THEN 1 END) AS orphan_src
                        RETURN total_sn, orphan_sn, orphan_src, error_siblings
                        """
                    )
                ),
                None,
            )
    except Exception as exc:  # pragma: no cover — graph connection issues
        integrity = None
        logger.debug("Could not fetch linking integrity: %s", exc)

    if integrity:
        console.print()
        console.print("[bold]Linking Integrity[/bold]")
        li_table = Table(show_header=True)
        li_table.add_column("Metric")
        li_table.add_column("Count", justify="right")
        li_table.add_row(
            "Orphan StandardName (no PRODUCED_NAME edge, excl. error siblings)",
            str(integrity.get("orphan_sn", 0)),
        )
        li_table.add_row(
            "Error-sibling StandardNames (deterministic, no source link expected)",
            str(integrity.get("error_siblings", 0)),
        )
        li_table.add_row(
            "Orphan composed/attached source (no PRODUCED_NAME edge)",
            str(integrity.get("orphan_src", 0)),
        )
        console.print(li_table)

    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            review_totals = next(
                iter(
                    gc.query(
                        """
                        MATCH (r:StandardNameReview)
                        RETURN count(r) AS total_reviews,
                               count(r.cost_usd) AS reviews_with_cost,
                               coalesce(sum(r.cost_usd), 0.0) AS total_cost,
                               coalesce(sum(r.tokens_in), 0) AS total_tokens_in,
                               coalesce(sum(r.tokens_out), 0) AS total_tokens_out
                        """
                    )
                ),
                None,
            )
            cost_by_model = list(
                gc.query(
                    """
                    MATCH (r:StandardNameReview)
                    WHERE r.cost_usd IS NOT NULL
                    RETURN r.model AS model,
                           count(r) AS n,
                           sum(r.cost_usd) AS cost_usd,
                           sum(r.tokens_in) AS tokens_in,
                           sum(r.tokens_out) AS tokens_out
                    ORDER BY cost_usd DESC
                    """
                )
            )
    except Exception as exc:  # pragma: no cover
        review_totals = None
        cost_by_model = []
        logger.debug("Could not fetch review cost: %s", exc)

    if review_totals:
        console.print()
        console.print("[bold]Review Cost[/bold]")
        rc_table = Table(show_header=True)
        rc_table.add_column("Metric")
        rc_table.add_column("Value", justify="right")
        total_reviews = review_totals.get("total_reviews", 0) or 0
        with_cost = review_totals.get("reviews_with_cost", 0) or 0
        rc_table.add_row("Review nodes", str(total_reviews))
        rc_table.add_row(
            "With cost recorded",
            f"{with_cost} ({100 * with_cost / max(total_reviews, 1):.1f}%)",
        )
        rc_table.add_row(
            "Total cost (USD)", f"${float(review_totals.get('total_cost') or 0):.4f}"
        )
        rc_table.add_row(
            "Total tokens in", str(review_totals.get("total_tokens_in") or 0)
        )
        rc_table.add_row(
            "Total tokens out", str(review_totals.get("total_tokens_out") or 0)
        )
        console.print(rc_table)

    if cost_by_model:
        console.print()
        console.print("[bold]Review Cost by Reviewer Model[/bold]")
        cm_table = Table(show_header=True)
        cm_table.add_column("Model")
        cm_table.add_column("Reviews", justify="right")
        cm_table.add_column("Cost (USD)", justify="right")
        cm_table.add_column("Tokens in", justify="right")
        cm_table.add_column("Tokens out", justify="right")
        for row in cost_by_model:
            cm_table.add_row(
                str(row.get("model") or "—"),
                str(row.get("n") or 0),
                f"${float(row.get('cost_usd') or 0):.4f}",
                str(row.get("tokens_in") or 0),
                str(row.get("tokens_out") or 0),
            )
        console.print(cm_table)


def _emit_yaml_output(
    sections: list[str], export_format: str, output: str | None
) -> None:
    """Write combined YAML sections to stdout or file (used by sn gaps)."""
    if export_format != "yaml":
        return
    if not sections:
        console.print("[dim]No vocabulary gaps to export.[/dim]")
        return
    combined = "\n".join(sections)
    if output:
        from pathlib import Path

        Path(output).write_text(combined)
        console.print(f"[green]Wrote ISN PR snippet to[/green] {output}")
    else:
        click.echo(combined)


@sn.command("gaps")
@click.option(
    "--direction",
    type=click.Choice(["missing", "saturated", "both"]),
    default="both",
    show_default=True,
    help=(
        "Which vocabulary gaps to report. 'missing' = tokens the LLM wanted "
        "but ISN lacks. 'saturated' = open-segment tokens reused enough to "
        "propose as new ISN anchors. 'both' shows the full ISN-PR picture."
    ),
)
@click.option(
    "--segment",
    default=None,
    help=(
        "Filter by grammar segment (e.g., transformation, process, "
        "physical_base). For --direction saturated, defaults to physical_base."
    ),
)
@click.option(
    "--include-open-segments/--closed-only",
    "include_open",
    default=False,
    help=(
        "Include missing-direction gaps on open-vocabulary segments "
        "(e.g. physical_base) and pseudo segments (grammar_ambiguity). "
        "Hidden by default because physical_base admits any compound "
        "token by design — use --direction saturated instead to propose "
        "common bases as ISN anchors."
    ),
)
@click.option(
    "--min-uses",
    type=int,
    default=3,
    show_default=True,
    help="Saturated-direction: minimum distinct supporting StandardNames.",
)
@click.option(
    "--min-score",
    type=float,
    default=0.75,
    show_default=True,
    help="Saturated-direction: minimum review_mean_score on every "
    "supporting name (quality gate).",
)
@click.option(
    "--include-existing/--exclude-existing",
    default=False,
    show_default=True,
    help="Saturated-direction: include tokens already in ISN's vocabulary.",
)
@click.option(
    "--persist/--no-persist",
    default=False,
    show_default=True,
    help="Saturated-direction: write PromotionCandidate nodes to the graph.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write YAML export to this path (default: stdout when --format yaml).",
)
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["table", "yaml"]),
    default="table",
    show_default=True,
    help="Output format. --format yaml yields ISN-PR-ready snippets.",
)
def sn_gaps(
    direction: str,
    segment: str | None,
    include_open: bool,
    min_uses: int,
    min_score: float,
    include_existing: bool,
    persist: bool,
    output: str | None,
    export_format: str,
) -> None:
    """Report vocabulary gaps for ISN issue filing.

    Reports two complementary ISN-boundary flows:

    * ``missing`` — VocabGap nodes recording tokens the LLM wanted but
      ISN lacks (closed-segment gaps by default).
    * ``saturated`` — open-segment tokens (``physical_base``) reused on
      enough high-quality StandardNames to propose as new ISN anchors.

    The default ``--direction both`` shows both: one table for missing
    tokens to add, one for saturated tokens ready for promotion. Both
    feed the same ISN grammar PR workflow.

    \b
    Examples:
      imas-codex sn gaps                        # both directions, table
      imas-codex sn gaps --direction missing    # tokens ISN should add
      imas-codex sn gaps --direction saturated  # promotion candidates
      imas-codex sn gaps --format yaml          # ISN PR snippet (both)
      imas-codex sn gaps --direction saturated --persist --format yaml \\
          --output promotions.yml               # persist + PR snippet
    """
    from rich.table import Table

    from imas_codex.graph.client import GraphClient

    yaml_sections: list[str] = []

    # ------------------------------------------------------------------
    # Saturated direction — promotion candidates for open segments
    # ------------------------------------------------------------------
    if direction in ("saturated", "both"):
        from imas_codex.standard_names.vocab_promotion import (
            format_isn_pr_snippet,
            mine_promotion_candidates,
            persist_candidates,
        )

        sat_segment = segment or "physical_base"
        candidates = mine_promotion_candidates(
            segment=sat_segment,
            min_usage_count=min_uses,
            min_review_mean_score=min_score,
            exclude_existing=not include_existing,
        )

        if persist and candidates:
            try:
                n = persist_candidates(candidates)
                console.print(f"[green]Persisted {n} PromotionCandidate nodes.[/green]")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Failed to persist candidates:[/red] {exc}")

        if export_format == "yaml":
            if candidates:
                yaml_sections.append(
                    format_isn_pr_snippet(candidates, segment=sat_segment)
                )
        else:
            console.print(
                f"[bold]Saturated `{sat_segment}` tokens[/bold] "
                f"(>= {min_uses} uses, all >= {min_score:.2f})"
            )
            if not candidates:
                console.print(
                    "[dim]No promotion candidates at current thresholds.[/dim]"
                )
            else:
                table = Table(show_header=True, header_style="bold")
                table.add_column("Token")
                table.add_column("Uses", justify="right")
                table.add_column("Min score", justify="right")
                table.add_column("Domains")
                table.add_column("Example name")
                for c in candidates:
                    example = (c.get("supporting_names") or [""])[0]
                    table.add_row(
                        c["token"],
                        str(c["uses"]),
                        f"{c['min_review_score']:.2f}",
                        ", ".join((c.get("physics_domains") or [])[:3]),
                        example,
                    )
                console.print(table)
                console.print(f"[dim]{len(candidates)} candidate(s).[/dim]")

    # ------------------------------------------------------------------
    # Missing direction — VocabGap nodes for closed segments
    # ------------------------------------------------------------------
    if direction not in ("missing", "both"):
        _emit_yaml_output(yaml_sections, export_format, output)
        return

    with GraphClient() as gc:
        from imas_codex.standard_names.gap_harvest import harvest_vocab_gaps

        results = harvest_vocab_gaps(
            gc, segment_filter=segment, include_open=include_open
        )

    if export_format == "yaml":
        from imas_codex.standard_names.gap_harvest import (
            _dd_version,
            _isn_version,
            format_pr_yaml,
        )

        if results:
            yaml_sections.append(
                format_pr_yaml(
                    results,
                    isn_version=_isn_version(),
                    dd_version=_dd_version(),
                )
            )
        _emit_yaml_output(yaml_sections, export_format, output)
        return

    # Table format (default)
    if not results:
        console.print("[dim]No missing vocabulary gaps found.[/dim]")
        if segment:
            console.print(f"[dim]  (filtered by segment={segment})[/dim]")
    else:
        table = Table(title="Missing Vocabulary Tokens")
        table.add_column("Segment", style="cyan")
        table.add_column("Needed Token", style="bold")
        table.add_column("Sources", justify="right")
        table.add_column("Example Count", justify="right")
        table.add_column("First Seen")
        table.add_column("Last Seen")

        for r in results:
            first_seen = str(r["first_seen"])[:10] if r.get("first_seen") else "—"
            last_seen = str(r["last_seen"])[:10] if r.get("last_seen") else "—"
            table.add_row(
                r["segment"],
                r["needed_token"],
                str(r["occurrences"]),
                str(r.get("example_count") or "—"),
                first_seen,
                last_seen,
            )

        console.print(table)
        console.print(f"[dim]{len(results)} missing token(s).[/dim]")


# =============================================================================
# Export / Preview / Publish / Import — Phase 3+4 CLI integration
# =============================================================================


@sn.command("export")
@click.option(
    "--staging",
    type=click.Path(),
    required=True,
    help="Output staging directory for YAML files",
)
@click.option(
    "--min-score",
    type=float,
    default=0.65,
    show_default=True,
    help="Minimum reviewer_score_name for inclusion",
)
@click.option(
    "--include-unreviewed",
    is_flag=True,
    help="Include names without a reviewer_score_name",
)
@click.option(
    "--min-description-score",
    type=float,
    default=None,
    help="Secondary threshold on description sub-score",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite non-empty staging directory without prompting",
)
@click.option(
    "--skip-gate",
    is_flag=True,
    help="Skip all quality gates (debugging only)",
)
@click.option(
    "--gate-only",
    is_flag=True,
    help="Run quality gates and report without emitting YAML",
)
@click.option(
    "--gate-scope",
    type=click.Choice(["all", "a", "b", "c", "d"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Which gates to run",
)
@click.option(
    "--domain",
    type=str,
    default=None,
    help="Filter export to a single physics domain",
)
@click.option(
    "--override-edits",
    type=str,
    multiple=True,
    help="Name(s) to reset from catalog_edit to pipeline origin (repeatable; or 'all')",
)
@click.option(
    "--include-sources/--no-include-sources",
    default=True,
    show_default=True,
    help="Populate sources field in each entry with graph provenance (debug aid)",
)
def sn_export(
    staging: str,
    min_score: float,
    include_unreviewed: bool,
    min_description_score: float | None,
    force: bool,
    skip_gate: bool,
    gate_only: bool,
    gate_scope: str,
    domain: str | None,
    override_edits: tuple[str, ...],
    include_sources: bool,
) -> None:
    """Export validated standard names from graph to a staging directory.

    \b
    Reads StandardName nodes, applies quality gates (A/B/C/D),
    and writes YAML files to <staging>/standard_names/<domain>/<name>.yml.

    \b
    Examples:
      imas-codex sn export --staging ./staging
      imas-codex sn export --staging ./staging --gate-only
      imas-codex sn export --staging ./staging --domain equilibrium
      imas-codex sn export --staging ./staging --min-score 0.8 --force
      imas-codex sn export --staging ./staging --no-include-sources
    """
    from pathlib import Path

    from rich.table import Table

    from imas_codex.standard_names.export import run_export

    edits_list = list(override_edits) if override_edits else None

    console.print("\n[bold]Standard Name Export[/bold]")
    console.print(f"  Staging: {staging}")
    console.print(f"  Min score: {min_score}")
    if domain:
        console.print(f"  Domain: {domain}")
    if gate_only:
        console.print("  Mode: [yellow]gate-only (no YAML output)[/yellow]")
    if skip_gate:
        console.print("  Gates: [yellow]skipped[/yellow]")
    if not include_sources:
        console.print("  Sources: [dim]excluded (--no-include-sources)[/dim]")
    if edits_list:
        console.print(f"  Override edits: {', '.join(edits_list)}")
    console.print("")

    try:
        report = run_export(
            staging_dir=Path(staging),
            min_score=min_score,
            include_unreviewed=include_unreviewed,
            min_description_score=min_description_score,
            domain=domain,
            force=force,
            skip_gate=skip_gate,
            gate_only=gate_only,
            gate_scope=gate_scope,
            override_edits=edits_list,
            include_sources=include_sources,
        )
    except FileExistsError as exc:
        console.print(f"[red]Precondition failure:[/red] {exc}")
        console.print("[dim]Use --force to overwrite.[/dim]")
        raise SystemExit(2) from exc
    except Exception as exc:
        console.print(f"[red]Export error:[/red] {exc}")
        raise SystemExit(3) from exc

    # ── Summary table ──────────────────────────────────────
    table = Table(title="Export Summary")
    table.add_column("metric", style="cyan")
    table.add_column("value", style="white")
    table.add_row("total candidates", str(report.total_candidates))
    table.add_row("exported", str(report.exported_count))
    table.add_row("excluded (below score)", str(report.excluded_below_score))
    table.add_row("excluded (unreviewed)", str(report.excluded_unreviewed))
    table.add_row("excluded (domain)", str(report.excluded_by_domain))
    console.print(table)

    # ── Gate results ───────────────────────────────────────
    if report.gate_results:
        console.print("\n[bold]Gate Results[/bold]")
        for gr in report.gate_results:
            status = "[green]PASS[/green]" if gr.passed else "[red]FAIL[/red]"
            if gr.skipped:
                status = "[dim]SKIP[/dim]"
            issues = f" ({len(gr.issues)} issue(s))" if gr.issues else ""
            console.print(f"  {gr.gate}: {status}{issues}")

    # ── Divergence entries ─────────────────────────────────
    if report.divergence_entries:
        console.print(
            f"\n[yellow]Divergence:[/yellow] {len(report.divergence_entries)} entries"
        )
        for de in report.divergence_entries[:10]:
            console.print(f"  ~ {de.name} ({de.field}): {de.detail}")
        if len(report.divergence_entries) > 10:
            console.print(f"  ... and {len(report.divergence_entries) - 10} more")

    # ── Exit code ──────────────────────────────────────────
    if not report.all_gates_passed:
        failed_gates = [g.gate for g in report.gate_results if not g.passed]
        console.print(
            f"\n[red]✗ Blocking gate failure(s):[/red] {', '.join(failed_gates)}"
        )
        raise SystemExit(1)

    console.print("\n[green]✓ Export complete[/green]")


@sn.command("preview")
@click.option(
    "--staging",
    type=click.Path(exists=True),
    required=True,
    help="Staging directory to preview",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port for the local preview server (default: 8000)",
)
@click.option(
    "--host",
    type=str,
    default=None,
    help=(
        "Host to bind to (default: 127.0.0.1). Pass 0.0.0.0 to expose "
        "the dev server to other machines on the network — useful when "
        "previewing over an SSH tunnel that other collaborators on the "
        "same cluster can reach."
    ),
)
def sn_preview(
    staging: str,
    port: int | None,
    host: str | None,
) -> None:
    """Preview a staging directory in the browser via ISN catalog-site.

    \b
    Launches a local MkDocs dev server. Press Ctrl-C to stop.

    \b
    Examples:
      imas-codex sn preview --staging ./staging
      imas-codex sn preview --staging ./staging --port 9090
      imas-codex sn preview --staging ./staging --host 0.0.0.0
    """
    from imas_codex.standard_names.preview import run_preview

    console.print("\n[bold]Standard Name Preview[/bold]")
    console.print(f"  Staging: {staging}")
    if host:
        console.print(f"  Host: {host}")
    if port:
        console.print(f"  Port: {port}")
    console.print("")

    try:
        handle = run_preview(staging, port=port, host=host)
    except FileNotFoundError as exc:
        console.print(f"[red]Precondition failure:[/red] {exc}")
        raise SystemExit(2) from exc
    except Exception as exc:
        console.print(f"[red]Preview error:[/red] {exc}")
        raise SystemExit(3) from exc

    if handle.process is None:
        console.print(
            "[red]Could not start preview server.[/red]\n"
            "Ensure imas-standard-names is installed: "
            "uv pip install imas-standard-names"
        )
        raise SystemExit(3)

    console.print(f"  Preview URL: [link={handle.url}]{handle.url}[/link]")
    console.print("  Press [bold]Ctrl-C[/bold] to stop.\n")

    try:
        handle.process.wait()
    except KeyboardInterrupt:
        console.print("\n[dim]Shutting down preview server...[/dim]")
    finally:
        handle.stop()


@sn.command("publish")
@click.option(
    "--staging",
    type=click.Path(exists=True),
    required=True,
    help="Staging directory produced by 'sn export'",
)
@click.option(
    "--isnc",
    type=click.Path(exists=True),
    required=True,
    help="Path to local imas-standard-names-catalog git checkout",
)
@click.option(
    "--push/--no-push",
    default=False,
    show_default=True,
    help="Push the commit to origin after creating it",
)
@click.option(
    "--dry-run", is_flag=True, help="Validate and report without modifying ISNC"
)
def sn_publish(
    staging: str,
    isnc: str,
    push: bool,
    dry_run: bool,
) -> None:
    """Transport a staging directory to an ISNC checkout.

    \b
    Mirrors the staging directory (from 'sn export') into an ISNC git
    checkout, creates a commit, and optionally pushes to origin.
    No quality gates are re-run — they already ran at export time.

    \b
    Examples:
      imas-codex sn publish --staging ./staging --isnc ../isnc
      imas-codex sn publish --staging ./staging --isnc ../isnc --push
      imas-codex sn publish --staging ./staging --isnc ../isnc --dry-run
    """
    from rich.table import Table

    from imas_codex.standard_names.publish import run_publish

    console.print("\n[bold]Standard Name Publish[/bold]")
    console.print(f"  Staging: {staging}")
    console.print(f"  ISNC: {isnc}")
    if push:
        console.print("  Push: [green]yes[/green]")
    if dry_run:
        console.print("  Mode: [yellow]dry run[/yellow]")
    console.print("")

    try:
        report = run_publish(
            staging_dir=staging,
            isnc_path=isnc,
            push=push,
            dry_run=dry_run,
        )
    except Exception as exc:
        console.print(f"[red]Publish error:[/red] {exc}")
        raise SystemExit(3) from exc

    if report.errors:
        console.print(f"[red]Errors: {len(report.errors)}[/red]")
        for err in report.errors[:10]:
            console.print(f"  - {err}")
        if len(report.errors) > 10:
            console.print(f"  ... and {len(report.errors) - 10} more")
        raise SystemExit(2)

    # ── Summary ────────────────────────────────────────────
    table = Table(title="Publish Summary")
    table.add_column("metric", style="cyan")
    table.add_column("value", style="white")
    table.add_row("files copied", str(report.files_copied))
    table.add_row("commit SHA", report.commit_sha or "(no changes)")
    table.add_row("pushed", "yes" if report.pushed else "no")
    table.add_row("dry run", "yes" if report.dry_run else "no")
    console.print(table)

    console.print("\n[green]✓ Publish complete[/green]")


@sn.command("import")
@click.option(
    "--isnc",
    type=click.Path(exists=True),
    required=True,
    help="Path to ISNC repository root (containing standard_names/ subtree)",
)
@click.option(
    "--accept-unit-override",
    is_flag=True,
    help="Accept unit mismatches against DD values",
)
@click.option(
    "--accept-cocos-override",
    is_flag=True,
    help="Accept COCOS transformation type mismatches against graph",
)
@click.option(
    "--dry-run", is_flag=True, help="Parse and validate without writing to graph"
)
def sn_import(
    isnc: str,
    accept_unit_override: bool,
    accept_cocos_override: bool,
    dry_run: bool,
) -> None:
    """Import reviewed catalog entries from ISNC into the graph.

    \b
    Reads YAML files from the ISNC standard_names/ subtree, validates
    them, derives grammar fields, applies diff-based origin tracking,
    and MERGEs into the graph with pipeline_status='accepted'.

    \b
    Examples:
      imas-codex sn import --isnc ../imas-standard-names-catalog
      imas-codex sn import --isnc ../isnc --dry-run
      imas-codex sn import --isnc ../isnc --accept-unit-override
    """
    from pathlib import Path

    from rich.table import Table

    from imas_codex.standard_names.catalog_import import run_import

    console.print("\n[bold]Standard Name Import[/bold]")
    console.print(f"  ISNC: {isnc}")
    if accept_unit_override:
        console.print("  Unit override: [yellow]accepted[/yellow]")
    if accept_cocos_override:
        console.print("  COCOS override: [yellow]accepted[/yellow]")
    if dry_run:
        console.print("  Mode: [yellow]dry run[/yellow]")
    console.print("")

    try:
        report = run_import(
            catalog_dir=Path(isnc),
            dry_run=dry_run,
            accept_unit_override=accept_unit_override,
            accept_cocos_override=accept_cocos_override,
        )
    except Exception as exc:
        console.print(f"[red]Import error:[/red] {exc}")
        raise SystemExit(3) from exc

    # ── Errors ─────────────────────────────────────────────
    if report.errors:
        console.print(f"[red]Errors: {len(report.errors)}[/red]")
        for err in report.errors[:10]:
            console.print(f"  - {err}")
        if len(report.errors) > 10:
            console.print(f"  ... and {len(report.errors) - 10} more")

    # ── Summary table ──────────────────────────────────────
    action = "Would import" if dry_run else "Imported"
    table = Table(title=f"Import Summary ({action})")
    table.add_column("metric", style="cyan")
    table.add_column("value", style="white")
    table.add_row("imported", str(report.imported))
    table.add_row("created", str(report.created))
    table.add_row("updated", str(report.updated))
    table.add_row("skipped", str(report.skipped))
    table.add_row("errors", str(len(report.errors)))
    if report.catalog_commit_sha:
        table.add_row("catalog SHA", report.catalog_commit_sha[:12])
    if report.pr_numbers:
        table.add_row("PR numbers", ", ".join(f"#{n}" for n in report.pr_numbers))
    table.add_row("watermark advanced", "yes" if report.watermark_advanced else "no")
    console.print(table)

    if dry_run and report.entries:
        console.print("\n[bold]Preview:[/bold]")
        for entry in report.entries[:20]:
            units = f" [{entry.get('unit', '')}]" if entry.get("unit") else ""
            console.print(f"  - {entry.get('id', '?')}{units}")
        if len(report.entries) > 20:
            console.print(f"  ... and {len(report.entries) - 20} more")

    if report.errors and not dry_run:
        raise SystemExit(2)

    console.print(f"\n[green]✓ {action}: {report.imported} entries[/green]")


@sn.command("clear")
@click.option("--dry-run", is_flag=True, help="Preview without modifying the graph")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--no-comment-export",
    "no_comment_export",
    is_flag=True,
    default=False,
    help="Skip the pre-clear Review comment export to JSONL.",
)
def sn_clear(dry_run: bool, force: bool, no_comment_export: bool) -> None:
    """Wipe every Standard Name the pipeline has produced.

    Deletes the six pipeline-output labels: StandardName, StandardNameReview,
    StandardNameSource, VocabGap, SNRun, LLMCost. ISN grammar nodes
    (GrammarToken, GrammarSegment, GrammarTemplate, ISNGrammarVersion)
    are ISN-authoritative reference data and stay in the graph — use
    ``sn sync-grammar`` to refresh them after an ISN release.

    For scoped deletes (by status, source, IDS, score tier, …) use
    ``sn prune`` instead.

    Before deleting, any existing StandardNameReview nodes are exported to a JSONL
    file in ``research/`` so reviewer feedback survives across clear
    cycles.  Pass ``--no-comment-export`` to skip the dump (e.g. in
    automated tests).

    \b
    Examples:
      imas-codex sn clear --dry-run    # Preview the wipe
      imas-codex sn clear --force      # Full wipe (non-interactive)
    """
    import datetime

    from imas_codex.standard_names.graph_ops import clear_sn_subsystem

    try:
        preview = clear_sn_subsystem(dry_run=True)
        total = sum(preview.values())
        if total == 0:
            console.print("No SN pipeline nodes to delete.")
            return

        console.print("[bold]SN pipeline wipe preview:[/bold]")
        for label, n in preview.items():
            if n:
                console.print(f"  {label}: {n}")
        console.print(f"[bold]Total: {total}[/bold]")

        if dry_run:
            return

        if not force:
            click.confirm(
                f"This will delete {total} SN pipeline nodes. Continue?",
                abort=True,
            )

        # Pre-clear comment export (Phase F)
        if not no_comment_export and preview.get("StandardNameReview", 0) > 0:
            import pathlib

            from imas_codex.standard_names.graph_ops import export_review_comments

            ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
            export_dir = pathlib.Path("research")
            export_path = export_dir / f"comments-{ts}.jsonl"
            try:
                n_exported = export_review_comments(export_path)
                if n_exported:
                    console.print(
                        f"[dim]Exported {n_exported} StandardNameReview records → {export_path}[/dim]"
                    )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Comment export skipped: {exc}[/yellow]")

        deleted = clear_sn_subsystem(dry_run=False)
        total_deleted = sum(deleted.values())
        console.print(f"[green]Deleted {total_deleted} nodes[/green]")
        for label, n in deleted.items():
            if n:
                console.print(f"  {label}: {n}")
    except click.Abort:
        raise
    except Exception as e:
        console.print(f"[red]Clear error:[/red] {e}")
        raise SystemExit(1) from e


@sn.command("reset-domain")
@click.option(
    "--domain",
    "physics_domain",
    required=True,
    type=_PHYSICS_DOMAIN_CHOICE,
    help="Physics domain whose StandardNameSource nodes should be reset.",
)
@click.option(
    "--hard",
    is_flag=True,
    default=False,
    help="DETACH DELETE the SNS nodes entirely (recreated on next sn run extract).",
)
@click.option("--dry-run", is_flag=True, help="Report counts only — no writes.")
def sn_reset_domain(physics_domain: str, hard: bool, dry_run: bool) -> None:
    """Reset stuck StandardNameSource nodes for a physics domain.

    Default (soft reset): clears ``claimed_at`` / ``claim_token`` and
    resets ``status`` to ``extracted`` so abandoned nodes can be reclaimed
    on the next ``sn run``.

    ``--hard``: DETACH DELETE the SNS nodes entirely.  They will be
    re-created automatically when ``sn run`` next extracts that domain.

    ``--dry-run``: print the affected count only; no graph writes.

    \b
    Examples:
      imas-codex sn reset-domain --domain magnetics --dry-run
      imas-codex sn reset-domain --domain magnetics
      imas-codex sn reset-domain --domain equilibrium --hard
    """
    from imas_codex.graph.client import GraphClient

    try:
        with GraphClient() as gc:
            # Count affected nodes
            count_rows = gc.query(
                """
                MATCH (sns:StandardNameSource)
                WHERE (sns)-[:FROM_DD_PATH]->(:IMASNode {physics_domain: $domain})
                   OR sns.physics_domain = $domain
                RETURN count(sns) AS n
                """,
                domain=physics_domain,
            )
            affected = count_rows[0]["n"] if count_rows else 0

            if affected == 0:
                console.print(
                    f"No StandardNameSource nodes found for domain [bold]{physics_domain}[/bold]."
                )
                return

            action = "DETACH DELETE" if hard else "soft reset (clear claims)"
            console.print(
                f"[bold]{affected}[/bold] SNS node(s) for domain "
                f"[bold]{physics_domain}[/bold] would be affected ({action})."
            )

            if dry_run:
                return

            if hard:
                gc.query(
                    """
                    MATCH (sns:StandardNameSource)
                    WHERE (sns)-[:FROM_DD_PATH]->(:IMASNode {physics_domain: $domain})
                       OR sns.physics_domain = $domain
                    DETACH DELETE sns
                    """,
                    domain=physics_domain,
                )
                console.print(
                    f"[green]Deleted {affected} SNS node(s) for domain {physics_domain}.[/green]"
                )
            else:
                gc.query(
                    """
                    MATCH (sns:StandardNameSource)
                    WHERE (sns)-[:FROM_DD_PATH]->(:IMASNode {physics_domain: $domain})
                       OR sns.physics_domain = $domain
                    SET sns.status = 'extracted',
                        sns.claimed_at = null,
                        sns.claim_token = null
                    """,
                    domain=physics_domain,
                )
                console.print(
                    f"[green]Reset {affected} SNS node(s) for domain {physics_domain} "
                    f"to extracted status.[/green]"
                )
    except Exception as e:
        console.print(f"[red]reset-domain error:[/red] {e}")
        raise SystemExit(1) from e


@sn.command("analyse-comments")
@click.option(
    "--input-glob",
    "input_glob",
    default="research/comments-*.jsonl",
    show_default=True,
    help="Glob pattern for JSONL files to analyse.",
)
@click.option(
    "--domain",
    default=None,
    help="Restrict analysis to a specific physics domain.",
)
@click.option(
    "--top",
    type=int,
    default=10,
    show_default=True,
    help="Number of top phrases to show per dimension.",
)
@click.option(
    "--output-file",
    "output_file",
    default=None,
    help="Write markdown report to this file instead of stdout.",
)
def sn_analyse_comments(
    input_glob: str,
    domain: str | None,
    top: int,
    output_file: str | None,
) -> None:
    """Mine recurring criticisms from exported Review comment JSONL files.

    Reads JSONL files produced by ``sn clear`` (pre-clear comment dumps)
    and analyses per-dimension reviewer comments to surface recurring
    criticism patterns.  Produces a markdown report showing:

    * Top recurring noun phrases per review dimension
    * Per-dimension score distribution (lowest-scoring dimensions)
    * Repeat-reviewed names and score trajectory

    \b
    Examples:
      imas-codex sn analyse-comments
      imas-codex sn analyse-comments --domain equilibrium --top 15
      imas-codex sn analyse-comments --input-glob "research/comments-eq-*.jsonl"
      imas-codex sn analyse-comments --output-file research/analysis.md
    """
    import glob as glob_module
    import re
    from collections import Counter, defaultdict

    # ── Collect files ─────────────────────────────────────────────────────────
    files = sorted(glob_module.glob(input_glob))
    if not files:
        console.print(
            f"[yellow]No files matched '{input_glob}'. "
            "Run 'sn clear' first to produce JSONL exports.[/yellow]"
        )
        return

    # ── Load records ──────────────────────────────────────────────────────────
    records: list[dict] = []
    for fpath in files:
        with open(fpath, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if domain and rec.get("domain") != domain:
                    continue
                records.append(rec)

    if not records:
        scope = f" for domain '{domain}'" if domain else ""
        console.print(f"[yellow]No review records found{scope}.[/yellow]")
        return

    # ── Simple NLP: extract noun phrases (stopword-filtered bigrams/unigrams) ─
    _STOPWORDS = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "and",
        "or",
        "in",
        "for",
        "with",
        "on",
        "at",
        "by",
        "from",
        "as",
        "this",
        "that",
        "it",
        "its",
        "not",
        "no",
        "but",
        "also",
        "should",
        "must",
        "can",
        "may",
        "would",
        "could",
        "will",
        "has",
        "have",
        "had",
        "does",
        "do",
        "did",
        "very",
        "more",
        "than",
        "too",
        "just",
        "only",
        "per",
        "e.g",
        "i.e",
        "etc",
        "use",
        "used",
        "using",
    }

    def _tokenise(text: str) -> list[str]:
        tokens = re.findall(r"[a-z][a-z_\-]*", text.lower())
        return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]

    def _phrases(tokens: list[str]) -> list[str]:
        """Unigrams + bigrams from token list."""
        result = list(tokens)
        for i in range(len(tokens) - 1):
            result.append(f"{tokens[i]} {tokens[i + 1]}")
        return result

    # ── Per-dimension phrase counts ────────────────────────────────────────────
    dim_phrase_counts: dict[str, Counter] = defaultdict(Counter)
    dim_scores: dict[str, list[float]] = defaultdict(list)

    # Track per-name score history across files
    name_scores: dict[str, list[float]] = defaultdict(list)

    for rec in records:
        name = rec.get("name") or "unknown"
        score = rec.get("score")
        if score is not None:
            name_scores[name].append(float(score))

        cpd = rec.get("comments_per_dim") or {}
        if isinstance(cpd, str):
            try:
                cpd = json.loads(cpd)
            except (json.JSONDecodeError, ValueError):
                cpd = {}

        if cpd:
            for dim, text in cpd.items():
                if isinstance(text, str) and text.strip():
                    tokens = _tokenise(text)
                    for phrase in _phrases(tokens):
                        dim_phrase_counts[dim][phrase] += 1
                    # use overall score as proxy for dim score
                    if score is not None:
                        dim_scores[dim].append(float(score))
        else:
            # Fall back to free-text comments field
            text = rec.get("comments") or ""
            if text.strip():
                dim = "general"
                tokens = _tokenise(text)
                for phrase in _phrases(tokens):
                    dim_phrase_counts[dim][phrase] += 1
                if score is not None:
                    dim_scores[dim].append(float(score))

    # ── Build markdown report ─────────────────────────────────────────────────
    lines: list[str] = []
    scope_label = f" — domain: `{domain}`" if domain else ""
    lines.append(f"# Review Comment Analysis{scope_label}")
    lines.append("")
    lines.append(
        f"**Files scanned**: {len(files)}  |  "
        f"**Records loaded**: {len(records)}  |  "
        f"**Top N**: {top}"
    )
    lines.append("")

    # Section 1: Top criticisms per dimension
    lines.append("## Top Criticisms by Dimension")
    lines.append("")
    if not dim_phrase_counts:
        lines.append("_No per-dimension comments found in records._")
        lines.append("")
    else:
        # Sort dimensions by mean score (lowest first = most problematic)
        sorted_dims = sorted(
            dim_phrase_counts.keys(),
            key=lambda d: (
                (sum(dim_scores[d]) / len(dim_scores[d])) if dim_scores[d] else 1.0
            ),
        )
        for dim in sorted_dims:
            counter = dim_phrase_counts[dim]
            mean_score = (
                sum(dim_scores[dim]) / len(dim_scores[dim]) if dim_scores[dim] else None
            )
            score_str = (
                f" (mean score: {mean_score:.3f})" if mean_score is not None else ""
            )
            lines.append(f"### `{dim}`{score_str}")
            lines.append("")
            lines.append("| Phrase | Occurrences |")
            lines.append("|--------|-------------|")
            for phrase, cnt in counter.most_common(top):
                lines.append(f"| {phrase} | {cnt} |")
            lines.append("")

    # Section 2: Per-dimension score distribution
    lines.append("## Per-Dimension Score Distribution")
    lines.append("")
    if not dim_scores:
        lines.append("_No score data available._")
        lines.append("")
    else:
        lines.append("| Dimension | Reviews | Mean Score | Min Score |")
        lines.append("|-----------|---------|-----------|-----------|")
        for dim in sorted(
            dim_scores, key=lambda d: sum(dim_scores[d]) / len(dim_scores[d])
        ):
            scores_list = dim_scores[dim]
            mean_s = sum(scores_list) / len(scores_list)
            min_s = min(scores_list)
            lines.append(f"| {dim} | {len(scores_list)} | {mean_s:.3f} | {min_s:.3f} |")
        lines.append("")

    # Section 3: Repeat-reviewed names and score trajectory
    lines.append("## Repeat-Reviewed Names")
    lines.append("")
    repeat_names = {n: s for n, s in name_scores.items() if len(s) > 1}
    if not repeat_names:
        lines.append("_No names reviewed more than once in these files._")
        lines.append("")
    else:
        lines.append("| Name | Reviews | First Score | Last Score | Δ |")
        lines.append("|------|---------|-------------|------------|---|")
        for name, scores_list in sorted(repeat_names.items()):
            first = scores_list[0]
            last = scores_list[-1]
            delta = last - first
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            lines.append(
                f"| {name} | {len(scores_list)} | {first:.3f} | {last:.3f} | {delta_str} |"
            )
        lines.append("")

    report = "\n".join(lines)

    if output_file:
        from pathlib import Path as _Path

        out = _Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        console.print(f"[green]Report written to {output_file}[/green]")
    else:
        # Print to stdout via console (markdown renders better without markup)
        import sys

        sys.stdout.write(report + "\n")


@sn.command("prune")
@click.option(
    "--status",
    default=None,
    help="Delete names with this pipeline_status (e.g. drafted)",
)
@click.option(
    "--all",
    "prune_all",
    is_flag=True,
    help="Delete all standard names (still respects --include-accepted)",
)
@click.option(
    "--source",
    type=click.Choice(["dd", "signals"]),
    default=None,
    help="Filter by source ('dd' or 'signals')",
)
@click.option("--ids", "ids_filter", default=None, help="Filter to specific IDS")
@click.option(
    "--include-accepted",
    is_flag=True,
    help="Also delete accepted names (dangerous — use with care)",
)
@click.option("--dry-run", is_flag=True, help="Preview without modifying the graph")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--include-sources",
    is_flag=True,
    default=False,
    help="Also delete StandardNameSource nodes matching the same scope",
)
def sn_prune(
    status: str | None,
    prune_all: bool,
    source: str | None,
    ids_filter: str | None,
    include_accepted: bool,
    dry_run: bool,
    force: bool,
    include_sources: bool,
) -> None:
    """Delete a subset of StandardName nodes (scoped by filters).

    Relationship-first safety model: HAS_STANDARD_NAME edges are removed
    before deleting nodes; scoped deletes only remove orphaned nodes.
    Review nodes attached to pruned StandardNames are deleted alongside
    them; a final sweep removes any orphan StandardNameReview nodes left by prior
    runs.

    Use this for targeted cleanup while iterating on generation. For a
    full subsystem wipe (all nodes + grammar re-seed), use ``sn clear``.

    \b
    Examples:
      imas-codex sn prune --status drafted
      imas-codex sn prune --all --source dd --ids equilibrium
      imas-codex sn prune --all --include-accepted --dry-run
    """
    if not status and not prune_all:
        raise click.UsageError("Provide --status <value> or --all to select names.")

    status_filter = None if prune_all else ([status] if status else None)

    from imas_codex.standard_names.graph_ops import clear_standard_names

    try:
        # Always preview first
        count = clear_standard_names(
            status_filter=status_filter,
            source_filter=source,
            ids_filter=ids_filter,
            include_accepted=include_accepted,
            dry_run=True,
        )

        if count == 0:
            console.print("No matching StandardName nodes to delete.")
            return

        # Build scope description for the confirmation message
        scope_parts: list[str] = []
        if status:
            scope_parts.append(f"status={status}")
        if source:
            scope_parts.append(f"source={source}")
        if ids_filter:
            scope_parts.append(f"ids={ids_filter}")
        if include_accepted:
            scope_parts.append("including accepted")
        scope = f" ({', '.join(scope_parts)})" if scope_parts else ""

        if dry_run:
            console.print(f"Would delete {count} StandardName node(s){scope}")
            return

        if not force:
            click.confirm(
                f"This will delete {count} StandardName node(s){scope}. Continue?",
                abort=True,
            )

        deleted = clear_standard_names(
            status_filter=status_filter,
            source_filter=source,
            ids_filter=ids_filter,
            include_accepted=include_accepted,
            dry_run=False,
        )
        console.print(f"Deleted {deleted} StandardName node(s)")

        if include_sources:
            from imas_codex.graph.client import GraphClient

            with GraphClient() as gc:
                sns_where_clauses = []
                sns_params: dict = {}
                if source:
                    sns_where_clauses.append("sns.source_type = $source_type")
                    sns_params["source_type"] = source
                if ids_filter:
                    sns_where_clauses.append("sns.ids_name = $ids_filter")
                    sns_params["ids_filter"] = ids_filter
                where_clause = (
                    "WHERE " + " AND ".join(sns_where_clauses)
                    if sns_where_clauses
                    else ""
                )
                count_result = gc.query(
                    f"MATCH (sns:StandardNameSource) {where_clause} RETURN count(sns) AS count",
                    **sns_params,
                )
                sns_count = count_result[0]["count"] if count_result else 0
                if sns_count > 0:
                    gc.query(
                        f"MATCH (sns:StandardNameSource) {where_clause} DETACH DELETE sns",
                        **sns_params,
                    )
                    console.print(f"  Deleted {sns_count} StandardNameSource nodes")

    except click.Abort:
        raise
    except Exception as e:
        console.print(f"[red]Prune error:[/red] {e}")
        raise SystemExit(1) from e


def _run_sync_grammar(*, dry_run: bool, verbose: bool) -> None:
    """Shared implementation for ``sn sync-grammar`` and ``sn clear``'s re-seed."""
    from imas_codex.standard_names.grammar_sync import sync_isn_grammar_to_graph

    try:
        report = sync_isn_grammar_to_graph(dry_run=dry_run)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    console.print(
        f"ISN version: {report.isn_version}  spec: {report.spec_version}  "
        f"segments: {report.segments}  templates: {report.templates}"
    )
    console.print(f"[green]✓ Grammar sync complete (dry_run={dry_run})[/green]")

    if verbose:
        planned = report.raw_report.pop("planned_statements", None)
        for key, val in report.raw_report.items():
            console.print(f"  {key}: {val}")
        if planned:
            console.print(f"  planned_statements: {len(planned)}")
            for cypher, params in planned:
                console.print(f"    {cypher}  params={params}")

    console.print(
        f"[green]✓ Active grammar version → {report.finalise_report.get('target_version')}"
        f" (applied={report.finalise_report.get('applied')})[/green]"
    )


@sn.command("themes")
@click.option(
    "--physics-domain",
    "domain",
    type=_PHYSICS_DOMAIN_CHOICE,
    default=None,
    help="Filter by physics domain.",
)
@click.option(
    "--source",
    type=click.Choice(["graph", "reviews"], case_sensitive=False),
    default="graph",
    show_default=True,
    help=(
        "Theme source. 'graph' queries StandardName reviewer comments, "
        "'reviews' queries StandardNameReview nodes directly."
    ),
)
@click.option("--limit", type=int, default=50, help="Max comments to sample.")
@click.option(
    "--since",
    default=None,
    help="Only reviews after this ISO date (e.g. 2025-01-01).",
)
def sn_themes(
    domain: str | None,
    source: str,
    limit: int,
    since: str | None,
) -> None:
    """Extract and display recurring reviewer themes.

    Runs n-gram frequency analysis over reviewer comments to surface
    the most common criticisms.  Useful for understanding what the
    reviewer fleet consistently flags so compose prompts can be tuned.

    \b
    Examples:
      imas-codex sn themes --physics-domain equilibrium
      imas-codex sn themes --source reviews --limit 100
      imas-codex sn themes --since 2025-06-01
    """
    from rich.table import Table

    if source == "reviews" or since:
        # Query Review nodes directly for richer filtering
        try:
            from imas_codex.graph.client import GraphClient

            where_clauses: list[str] = ["r.comments IS NOT NULL"]
            params: dict[str, Any] = {"limit": limit}
            if domain:
                where_clauses.append("$domain IN sn.physics_domain")
                params["domain"] = domain
            if since:
                where_clauses.append("r.reviewed_at >= $since")
                params["since"] = since

            where = " AND ".join(where_clauses)
            cypher = f"""
                MATCH (sn:StandardName)-[:HAS_REVIEW]->(r:StandardNameReview)
                WHERE {where}
                RETURN r.comments AS comments
                ORDER BY r.reviewed_at DESC
                LIMIT $limit
            """
            with GraphClient() as gc:
                rows = gc.query(cypher, **params)

            if not rows:
                console.print("[yellow]No review comments found.[/yellow]")
                return

            comments = [r["comments"] for r in rows if r.get("comments")]
            if not comments:
                console.print("[yellow]No non-empty comments found.[/yellow]")
                return

            from imas_codex.standard_names.review.themes import (
                _extract_themes_from_texts,
            )

            themes = _extract_themes_from_texts(comments)
        except Exception as exc:
            console.print(f"[red]Error querying graph: {exc}[/red]")
            return
    else:
        # Use the existing domain-scoped helper
        from imas_codex.standard_names.review.themes import extract_reviewer_themes

        if not domain:
            console.print(
                "[yellow]--physics-domain is required for 'graph' source. "
                "Use --source reviews to query all domains.[/yellow]"
            )
            return
        themes = extract_reviewer_themes(domain=domain, limit=limit)

    if not themes:
        console.print("[yellow]No recurring themes found.[/yellow]")
        return

    table = Table(title="Recurring Reviewer Themes", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Theme", style="bold")
    table.add_column("Source", style="dim")

    source_label = f"domain={domain}" if domain else "all domains"
    for i, theme in enumerate(themes, 1):
        table.add_row(str(i), theme, source_label)

    console.print(table)
    console.print(f"\n[dim]{len(themes)} themes extracted from ≤{limit} comments[/dim]")


@sn.command("sync-grammar")
@click.option("--dry-run", is_flag=True, help="Log Cypher without executing.")
@click.option("-v", "--verbose", is_flag=True, help="Show planned Cypher statements.")
def sn_sync_grammar(dry_run: bool, verbose: bool) -> None:
    """Sync the ISN grammar spec into Neo4j.

    Writes ``ISNGrammarVersion``, ``GrammarSegment``, ``GrammarToken``,
    and ``GrammarTemplate`` nodes plus ``NEXT`` / ``DEFINES`` /
    ``HAS_TOKEN`` edges from the installed ``imas_standard_names``
    package. Idempotent — safe to re-run.

    Run this after ``sn clear --no-reseed``, after upgrading the ISN
    package, or manually during release preparation.
    """
    _run_sync_grammar(dry_run=dry_run, verbose=verbose)


@sn.command("review")
@click.option("--ids", default=None, help="Scope to names linked to specific IDS")
@click.option(
    "--physics-domain",
    "domain",
    type=_PHYSICS_DOMAIN_CHOICE,
    default=None,
    help="Scope to physics domain.",
)
@click.option(
    "--status",
    "status_filter",
    default="drafted",
    help="Filter by pipeline_status (default: drafted)",
)
@click.option(
    "--unreviewed",
    is_flag=True,
    help="Only names with no reviewer_score_name or stale review",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force re-review of already-scored names",
)
@click.option(
    "--models",
    "models_override",
    default=None,
    help=(
        "Ad-hoc override for the reviewer list as comma-separated model "
        "ids. Overrides [sn.review.names].models or [sn.review.docs].models "
        "depending on --target."
    ),
)
@click.option(
    "--batch-size",
    type=int,
    default=15,
    help="Max names per batch (hard cap: 25)",
)
@click.option(
    "--neighborhood",
    type=int,
    default=10,
    help="Similar names for context",
)
@click.option(
    "-c",
    "--cost-limit",
    type=float,
    default=5.0,
    help="Max LLM spend in USD",
)
@click.option(
    "--dry-run", is_flag=True, help="Run Layer 1 audits, show batch plan, no LLM calls"
)
@click.option("--skip-audit", is_flag=True, help="Skip Layer 1 audits (debug only)")
@click.option("--concurrency", type=int, default=2, help="Parallel review batches")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--target",
    type=click.Choice(["names", "docs"], case_sensitive=False),
    default="names",
    show_default=True,
    help=(
        "Which review rubric to apply. 'names' → 4-dim name rubric "
        "(grammar/semantic/convention/completeness, /80). 'docs' → 4-dim "
        "docs rubric (description_quality/documentation_quality/"
        "completeness/physics_accuracy, /80). A lower-fidelity target will "
        "not overwrite a higher-fidelity prior review unless --force."
    ),
)
@click.option(
    "--reviewer-profile",
    "reviewer_profile",
    type=click.Choice(
        ["default", "pilot", "opus-only", "haiku-only"], case_sensitive=False
    ),
    default="default",
    show_default=True,
    envvar="IMAS_CODEX_SN_REVIEW_PROFILE",
    help=(
        "Reviewer model chain profile. "
        "'default' → Opus+GPT-5.4+Sonnet (3-model RD-quorum, $0.027/name). "
        "'pilot' → Haiku×2+Opus arbiter ($0.004/name, ~85%% cost reduction). "
        "'opus-only' → single Opus reviewer (no quorum). "
        "'haiku-only' → single Haiku reviewer (cheapest). "
        "Overridden by --models if both are specified. "
        "Also read from IMAS_CODEX_SN_REVIEW_PROFILE env var."
    ),
)
def sn_review(
    ids: str | None,
    domain: str | None,
    status_filter: str,
    unreviewed: bool,
    force: bool,
    models_override: str | None,
    batch_size: int,
    neighborhood: int,
    cost_limit: float,
    dry_run: bool,
    skip_audit: bool,
    concurrency: int,
    verbose: bool,
    target: str,
    reviewer_profile: str,
) -> None:
    """Review standard names with 3-layer pipeline.

    \b
    Layer 1: Deterministic audits (embedding, lint, links, duplicates)
    Layer 2: Batched LLM quality scoring with neighborhood context
    Layer 3: Cross-batch consolidation and summary report

    \b
    Examples:
      imas-codex sn review --unreviewed --cost-limit 5.0
      imas-codex sn review --ids equilibrium --dry-run
      imas-codex sn review --force --physics-domain magnetics
      imas-codex sn review --target names --unreviewed
      imas-codex sn review --target docs --physics-domain equilibrium
      imas-codex sn review --reviewer-profile pilot --unreviewed -c 2.0
    """
    import asyncio

    from imas_codex.cli.discover.common import setup_logging
    from imas_codex.standard_names.budget import BudgetManager
    from imas_codex.standard_names.review.state import StandardNameReviewState

    setup_logging("sn", "sn-review", use_rich=False, verbose=verbose)

    # Resolve --target.
    target_normalized = target.lower()
    # Downstream state uses a derived name_only boolean.
    name_only = target_normalized == "names"

    # Enforce batch-size cap
    batch_size = min(batch_size, 25)

    # Load reviewer list (N>=1). Priority: --models > --reviewer-profile > config.
    from imas_codex.settings import (
        get_sn_review_disagreement_threshold,
        get_sn_review_docs_models,
        get_sn_review_names_models,
        get_sn_review_profile_models,
        get_sn_review_profile_threshold,
    )

    reviewer_profile = reviewer_profile.lower()
    if models_override:
        # Ad-hoc --models takes precedence over profile.
        review_models = [m.strip() for m in models_override.split(",") if m.strip()]
        disagreement_threshold = get_sn_review_disagreement_threshold()
    elif reviewer_profile != "default":
        # Explicit non-default profile selected.
        review_models = get_sn_review_profile_models(reviewer_profile)
        disagreement_threshold = get_sn_review_profile_threshold(reviewer_profile)
    elif target_normalized == "names":
        review_models = get_sn_review_names_models()
        disagreement_threshold = get_sn_review_disagreement_threshold()
    else:
        review_models = get_sn_review_docs_models()
        disagreement_threshold = get_sn_review_disagreement_threshold()

    # Build state
    state = StandardNameReviewState(
        facility="dd",
        cost_limit=cost_limit,
        ids_filter=ids,
        domain_filter=domain,
        status_filter=status_filter,
        unreviewed_only=unreviewed,
        force_review=force,
        skip_audit=skip_audit,
        review_model=(review_models[0] if review_models else None),
        batch_size=batch_size,
        neighborhood_k=neighborhood,
        concurrency=concurrency,
        dry_run=dry_run,
        name_only=name_only,
        target=target_normalized,
        budget_manager=BudgetManager(cost_limit),
        review_models=review_models,
        disagreement_threshold=disagreement_threshold,
    )

    async def _run() -> None:
        # Layer 1: Audits (on full catalog, unless --skip-audit)
        if not skip_audit:
            console.print(
                "[bold]Layer 1:[/bold] Running deterministic audits on full catalog…"
            )
            from imas_codex.graph.client import GraphClient

            def _load_catalog() -> list[dict]:
                with GraphClient() as gc:
                    rows = gc.query(
                        """
                        MATCH (sn:StandardName)
                        OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
                        RETURN sn.id AS id, sn.description AS description,
                               sn.documentation AS documentation,
                               sn.kind AS kind,
                               coalesce(u.id, sn.unit) AS unit,
                               sn.tags AS tags, sn.links AS links,
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
                               sn.reviewer_score_name AS reviewer_score,
                               sn.review_input_hash AS review_input_hash,
                               sn.embedding AS embedding,
                               sn.review_tier AS review_tier,
                               sn.link_status AS link_status,
                               sn.source_types AS source_types,
                               sn.geometric_base AS geometric_base
                        """
                    )
                    return [dict(r) for r in rows] if rows else []

            all_names = await asyncio.to_thread(_load_catalog)

            if not all_names:
                console.print("[yellow]No standard names found in graph[/yellow]")
                return

            console.print(f"  Loaded {len(all_names)} standard names")

            from imas_codex.standard_names.review.audits import run_all_audits

            state.audit_report = await asyncio.to_thread(run_all_audits, all_names)
            state.all_names = all_names

            # Print audit summary
            ar = state.audit_report
            console.print(
                f"  Embeddings: {ar.embedding.missing_count} missing, "
                f"{ar.embedding.stale_count} stale, "
                f"{ar.embedding.refreshed_count} refreshed"
            )
            console.print(f"  Lint findings: {len(ar.lint_findings)}")
            console.print(f"  Link issues: {len(ar.link_findings)}")
            console.print(f"  Duplicate components: {len(ar.duplicate_components)}")

        if dry_run:
            # In dry-run mode, show batch plan but don't run LLM
            console.print("\n[bold]Dry run:[/bold] Showing batch plan (no LLM calls)")

            from imas_codex.graph.client import GraphClient
            from imas_codex.standard_names.review.enrichment import (
                group_into_review_batches,
                reconstruct_clusters_batch,
            )

            # Apply filters to get target names
            targets = list(state.all_names) if state.all_names else []
            if status_filter:
                targets = [
                    n for n in targets if n.get("pipeline_status") == status_filter
                ]
            if ids:
                targets = [
                    n
                    for n in targets
                    if any(
                        p.startswith(ids + "/") for p in (n.get("source_paths") or [])
                    )
                ]
            if domain:
                targets = [n for n in targets if n.get("physics_domain") == domain]
            if unreviewed:
                from imas_codex.standard_names.review.audits import (
                    compute_review_input_hash,
                )

                targets = [
                    n
                    for n in targets
                    if n.get("reviewer_score_name") is None
                    or n.get("review_input_hash") != compute_review_input_hash(n)
                ]

            console.print(f"  Targets for review: {len(targets)} names")

            if targets:
                try:

                    def _get_clusters() -> dict:
                        with GraphClient() as gc:
                            return reconstruct_clusters_batch(targets, gc)

                    clusters = await asyncio.to_thread(_get_clusters)
                    batches = group_into_review_batches(
                        targets,
                        clusters,
                        max_batch_size=batch_size,
                    )
                    console.print(f"  Would create {len(batches)} review batches:")
                    for i, b in enumerate(batches[:10]):
                        n_names = len(b.get("names", []))
                        tokens = b.get("estimated_tokens", 0)
                        console.print(
                            f"    Batch {i + 1}: {n_names} names, ~{tokens} tokens"
                            f" — {b.get('group_key', 'unknown')}"
                        )
                    if len(batches) > 10:
                        console.print(f"    … and {len(batches) - 10} more batches")
                except Exception as exc:
                    console.print(
                        f"  [yellow]Could not compute batch plan: {exc}[/yellow]"
                    )
            return

        # Layer 2: Batched LLM Review
        console.print("\n[bold]Layer 2:[/bold] Running batched LLM review…")

        from imas_codex.standard_names.review.pipeline import run_sn_review_engine

        stop_event = asyncio.Event()
        await run_sn_review_engine(state, stop_event=stop_event)

        # Layer 3: Consolidation
        console.print("\n[bold]Layer 3:[/bold] Running cross-batch consolidation…")
        from imas_codex.standard_names.review.consolidation import run_consolidation

        summary = run_consolidation(state)

        # Print summary report
        console.print("\n[bold]═══ Review Summary ═══[/bold]")
        scored_info = f"  Scored: {summary.total_scored} / {summary.total_catalog_size}"
        scored_info += f" names ({summary.coverage_pct:.1f}%)"
        if summary.total_unscored > 0:
            scored_info += f"  [yellow]({summary.total_unscored} unscored)[/yellow]"
        console.print(scored_info)
        console.print(f"  LLM cost: ${summary.total_cost:.4f}")

        if summary.tier_distribution:
            tier_str = ", ".join(
                f"{t}: {c}" for t, c in sorted(summary.tier_distribution.items())
            )
            console.print(f"  Tier distribution: {tier_str}")

        if summary.duplicate_candidates:
            console.print(
                f"  Duplicate candidates: {len(summary.duplicate_candidates)}"
            )
            for dc in summary.duplicate_candidates[:3]:
                console.print(f"    {dc.names} (sim={dc.max_similarity:.3f})")

        if summary.drift_warnings:
            console.print(f"  Convention drift warnings: {len(summary.drift_warnings)}")
            for dw in summary.drift_warnings[:3]:
                console.print(
                    f"    [{dw.physics_domain}] {dw.drift_type}: {dw.detail[:80]}"
                )

        if summary.outliers:
            console.print(f"  Score outliers: {len(summary.outliers)}")
            for ol in summary.outliers[:5]:
                console.print(
                    f"    {ol.name_id}: {ol.score:.2f}"
                    f" (z={ol.z_score:.1f}, {ol.recommendation})"
                )

        if summary.lowest_scorers:
            console.print("  Lowest scorers:")
            for ls in summary.lowest_scorers[:5]:
                console.print(
                    f"    {ls.get('id', '?')}: {ls.get('reviewer_score_name', 0):.2f}"
                    f" ({ls.get('review_tier', '?')})"
                )

        # Budget summary
        if state.budget_manager:
            bs = state.budget_manager.summary
            console.print(
                f"  Budget: ${bs['total_spent']:.4f} used of"
                f" ${bs['total_budget']:.2f} ({bs['batch_count']} batches)"
            )

    asyncio.run(_run())


def _run_sn_docs_generation(
    *,
    domain_list: list[str] | None,
    status_filter: str,
    cost_limit: float,
    limit: int | None,
    batch_size: int | None,
    dry_run: bool,
    force: bool,
    model_override: str | None,
    verbose: bool,
    quiet: bool,
) -> None:
    """Run the five-phase enrichment pipeline to fill docs on named SNs.

    Shared by ``sn run --target docs`` so the
    docs-generation pathway has a single implementation. Does NOT change
    name, grammar fields, kind, or unit — only description/documentation/
    tags/links.
    """
    from imas_codex.discovery.base.llm import set_litellm_offline_env

    set_litellm_offline_env()

    from imas_codex.cli.discover.common import (
        DiscoveryConfig,
        make_log_print,
        run_discovery,
        setup_logging,
        use_rich_output,
    )

    # Resolve docs batch size from pyproject defaults when unspecified.
    if batch_size is None:
        from imas_codex.settings import _get_section

        sn_generate_cfg = _get_section("sn-run")
        sn_enrich_cfg = _get_section("sn-enrich")
        batch_size = int(
            sn_generate_cfg.get("docs-batch-size", sn_enrich_cfg.get("batch-size", 8))
        )

    # status_filter can be comma-separated or repeated
    statuses = [
        s.strip() for part in status_filter.split(",") for s in [part] if s.strip()
    ]

    # --- Validation warning ---
    if not domain_list and limit is None:
        console.print(
            "[yellow]⚠ Enriching all named SNs — "
            "use --physics-domain or --limit to scope[/yellow]"
        )

    use_rich = use_rich_output()
    console_obj = setup_logging("sn", "sn-enrich", use_rich, verbose=verbose)
    log_print = make_log_print("sn-enrich", console_obj)

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    log_print("\n[bold]Standard Name Enrichment[/bold]")
    if domain_list:
        log_print(f"  Domain filter: {', '.join(domain_list)}")
    log_print(f"  Status filter: {', '.join(statuses)}")
    log_print(f"  Batch size: {batch_size}")
    log_print(f"  Cost limit: ${cost_limit:.2f}")
    if limit:
        log_print(f"  Limit: {limit} names")
    if force:
        log_print("  Force: re-enriching already-enriched names")
    if model_override:
        log_print(f"  Model override: {model_override}")
    if dry_run:
        log_print("  Mode: dry run")
    log_print("")

    from imas_codex.standard_names.enrich_pipeline import run_sn_enrich_engine
    from imas_codex.standard_names.enrich_state import StandardNameEnrichState

    state = StandardNameEnrichState(
        facility="dd",
        domain=domain_list,
        status_filter=statuses,
        cost_limit=cost_limit,
        limit=limit,
        batch_size=batch_size,
        dry_run=dry_run,
        force=force,
        model=model_override,
    )

    async def _run(stop_event, service_monitor):
        if service_monitor:
            state.service_monitor = service_monitor
        await run_sn_enrich_engine(
            state,
            stop_event=stop_event,
        )
        return state.stats

    config = DiscoveryConfig(
        facility="dd",
        domain="sn-enrich",
        facility_config={},
        display=None,
        check_graph=True,
        check_embed=False,
        check_ssh=False,
        check_auth=False,
        check_model=not dry_run,
        model_section="sn-enrich",
        suppress_loggers=[
            "imas_codex.standard_names",
        ],
        verbose=verbose,
    )

    result = run_discovery(config, _run)

    # --- Print summary ---
    if result:
        extracted = result.get("extract_count", 0)
        enriched_count = result.get("persist_written", 0)
        failed_count = result.get("document_errors", 0)
        quarantined = result.get("validate_quarantined", 0)
        doc_cost = result.get("document_cost", 0.0)

        log_print("\n[bold]Enrichment Summary[/bold]")
        log_print(f"  Extracted:    {extracted}")
        log_print(f"  Enriched:     {enriched_count}")
        if failed_count:
            log_print(f"  Failed:       {failed_count}")
        if quarantined:
            log_print(f"  Quarantined:  {quarantined}")
        log_print(f"  Cost:         ${doc_cost:.4f} / ${cost_limit:.2f}")

        # Per-phase timings
        phase_stats = [
            ("extract", state.extract_stats),
            ("contextualise", state.contextualise_stats),
            ("document", state.document_stats),
            ("validate", state.validate_stats),
            ("persist", state.persist_stats),
        ]
        timing_parts = []
        for name, stats in phase_stats:
            elapsed = stats.elapsed
            if elapsed and elapsed > 0 and stats.processed > 0:
                timing_parts.append(f"{name}={elapsed:.1f}s")
        if timing_parts:
            log_print(f"  Timings:      {', '.join(timing_parts)}")

        # Per-domain breakdown (if available)
        domain_stats = result.get("domain_breakdown")
        if domain_stats:
            log_print("  By domain:")
            for dom, count in sorted(domain_stats.items()):
                log_print(f"    {dom}: {count}")

        if dry_run:
            log_print("  (dry run — no LLM calls or graph writes)")
    elif not quiet:
        log_print("[yellow]No enrichment results returned[/yellow]")

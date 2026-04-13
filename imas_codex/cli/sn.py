"""Standard name generation commands."""

from __future__ import annotations

import logging

import click
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


@click.group()
def sn() -> None:
    """Standard name generation and management.

    \b
    Generate:
      imas-codex sn generate --source dd [--ids NAME] [--domain NAME]
      imas-codex sn generate --source signals --facility NAME

    \b
    Status:
      imas-codex sn status
    """
    pass


@sn.command("generate")
@click.option(
    "--source",
    type=click.Choice(["dd", "signals"]),
    required=True,
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
    "--domain",
    "domain_filter",
    type=str,
    default=None,
    help="Filter to physics domain",
)
@click.option(
    "--facility",
    type=str,
    default=None,
    help="Facility ID (required for signals source)",
)
@click.option(
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
    "--limit",
    type=int,
    default=None,
    help="Maximum number of DD paths to process",
)
@click.option(
    "--review-model",
    type=str,
    default=None,
    help="LLM model for cross-model review (default: reasoning model)",
)
@click.option("--skip-review", is_flag=True, help="Skip the cross-model review phase")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("-q", "--quiet", is_flag=True, help="Suppress non-error output")
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
def sn_generate(
    source: str,
    ids_filter: str | None,
    domain_filter: str | None,
    facility: str | None,
    cost_limit: float,
    dry_run: bool,
    force: bool,
    limit: int | None,
    review_model: str | None,
    skip_review: bool,
    verbose: bool,
    quiet: bool,
    reset_to: str | None,
) -> None:
    """Generate standard names from a source.

    \b
    Examples:
      imas-codex sn generate --source dd --ids equilibrium --dry-run
      imas-codex sn generate --source dd --domain magnetics --cost-limit 2
      imas-codex sn generate --source signals --facility tcv
    """
    # Validate: signals source requires facility
    if source == "signals" and not facility:
        raise click.UsageError("--facility is required when --source is signals")

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
            )
            console.print(
                f"[yellow]--reset-to extracted:[/yellow] cleared {n} SN nodes"
            )
        elif reset_to == "drafted":
            n = reset_standard_names(
                from_status="drafted",
                source_filter=source_arg,
                ids_filter=ids_filter,
            )
            console.print(f"[yellow]--reset-to drafted:[/yellow] reset {n} SN nodes")

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
    if ids_filter:
        log_print(f"  IDS filter: {ids_filter}")
    if domain_filter:
        log_print(f"  Domain filter: {domain_filter}")
    if facility:
        log_print(f"  Facility: {facility}")
    if dry_run:
        log_print("  Mode: dry run")
    if force:
        log_print("  Force: re-generating all names")
    if limit:
        log_print(f"  Limit: {limit} paths")
    if skip_review:
        log_print("  Review: skipped")
    elif review_model:
        log_print(f"  Review model: {review_model}")
    log_print(f"  Cost limit: ${cost_limit:.2f}")
    log_print("")

    from imas_codex.standard_names.pipeline import run_sn_generate_engine
    from imas_codex.standard_names.state import SNBuildState

    # Build progress display
    display = None
    if use_rich and not quiet:
        try:
            from imas_codex.standard_names.progress import SNProgressDisplay

            display = SNProgressDisplay(
                source=source,
                console=console_obj,
                cost_limit=cost_limit,
                mode_label="DRY RUN" if dry_run else None,
            )
        except Exception:
            logger.debug("Could not create progress display", exc_info=True)

    state = SNBuildState(
        facility=effective_facility,
        source=source,
        ids_filter=ids_filter,
        domain_filter=domain_filter,
        facility_filter=facility,
        cost_limit=cost_limit,
        dry_run=dry_run,
        force=force,
        limit=limit,
        skip_review=skip_review,
        review_model=review_model,
    )

    if display:
        display.set_engine_state(state)

    async def _run(stop_event, service_monitor):
        if service_monitor:
            state.service_monitor = service_monitor
        await run_sn_generate_engine(
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
        model_section="language",
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
        reviewed = result.get("review_accepted", composed)
        validated = result.get("validate_valid", 0)
        parts = [f"Extracted: {extracted}", f"Composed: {composed}"]
        if not skip_review:
            rejected = result.get("review_rejected", 0)
            revised = result.get("review_revised", 0)
            parts.append(
                f"Reviewed: {reviewed} (rejected: {rejected}, revised: {revised})"
            )
        parts.append(f"Validated: {validated}")
        log_print(", ".join(parts))
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
    "--domain",
    "domain_filter",
    type=str,
    default=None,
    help="Filter to physics domain",
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
                       count(CASE WHEN sn.source = 'dd' THEN 1 END) AS from_dd,
                       count(CASE WHEN sn.source = 'signals' THEN 1 END) AS from_signals,
                       count(CASE WHEN sn.source = 'manual' THEN 1 END) AS from_manual
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
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")


@sn.command("publish")
@click.option(
    "--ids",
    "ids_filter",
    type=str,
    default=None,
    help="Filter to specific IDS name",
)
@click.option(
    "--domain",
    "domain_filter",
    type=str,
    default=None,
    help="Filter to physics domain (applied to tags)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="sn_catalog_output",
    help="Directory for YAML files",
)
@click.option(
    "--group-by",
    type=click.Choice(["ids", "domain", "confidence"]),
    default="ids",
    help="Batching strategy for PR grouping",
)
@click.option(
    "--confidence-min",
    type=float,
    default=0.0,
    help="Minimum confidence threshold (0.0-1.0)",
)
@click.option(
    "--catalog-dir",
    type=click.Path(exists=False),
    default=None,
    help="Existing catalog directory for duplicate checking",
)
@click.option(
    "--create-pr",
    is_flag=True,
    help="Create GitHub PR (requires gh CLI)",
)
@click.option(
    "--catalog-repo",
    type=str,
    default="iterorganization/imas-standard-names-catalog",
    help="Target GitHub repo for PR creation",
)
@click.option("--dry-run", is_flag=True, help="Preview without writing files")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def sn_publish(
    ids_filter: str | None,
    domain_filter: str | None,
    output_dir: str,
    group_by: str,
    confidence_min: float,
    catalog_dir: str | None,
    create_pr: bool,
    catalog_repo: str,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Publish validated standard names to YAML catalog files.

    \b
    Reads StandardName nodes from the graph, converts them to YAML
    files matching the imas-standard-names-catalog format, and
    optionally creates batched GitHub pull requests.

    \b
    Examples:
      imas-codex sn publish --dry-run
      imas-codex sn publish --ids equilibrium --output-dir catalog/
      imas-codex sn publish --group-by confidence --confidence-min 0.8
      imas-codex sn publish --create-pr --catalog-repo org/repo
    """
    from pathlib import Path

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    console.print("\n[bold]Standard Name Publish[/bold]")
    if ids_filter:
        console.print(f"  IDS filter: {ids_filter}")
    if domain_filter:
        console.print(f"  Domain filter: {domain_filter}")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Group by: {group_by}")
    console.print(f"  Confidence min: {confidence_min}")
    if dry_run:
        console.print("  Mode: [yellow]dry run[/yellow]")
    console.print("")

    # Step 1: Load validated names from graph
    try:
        from imas_codex.standard_names.graph_ops import get_validated_standard_names

        records = get_validated_standard_names(
            ids_filter=ids_filter,
            confidence_min=confidence_min,
        )
    except Exception as e:
        console.print(f"[red]Error reading from graph:[/red] {e}")
        raise SystemExit(1) from e

    if not records:
        console.print("[yellow]No validated standard names found in graph.[/yellow]")
        return

    console.print(f"  Loaded [bold]{len(records)}[/bold] validated names from graph")

    # Step 2: Convert to publish entries
    from imas_codex.standard_names.publish import (
        check_catalog_duplicates,
        create_catalog_pr,
        generate_catalog_files,
        graph_records_to_entries,
        make_publish_batches,
    )

    entries = graph_records_to_entries(records)

    # Apply domain filter on tags if specified
    if domain_filter:
        entries = [e for e in entries if domain_filter in e.tags]
        console.print(f"  After domain filter: [bold]{len(entries)}[/bold] entries")

    if not entries:
        console.print("[yellow]No entries after filtering.[/yellow]")
        return

    # Step 3: Check for duplicates
    catalog_path = Path(catalog_dir) if catalog_dir else None
    new_entries, duplicates = check_catalog_duplicates(entries, catalog_path)

    if duplicates:
        console.print(f"  Skipping [yellow]{len(duplicates)}[/yellow] duplicate(s)")
    entries = new_entries

    if not entries:
        console.print(
            "[yellow]All entries are duplicates — nothing to publish.[/yellow]"
        )
        return

    # Step 4: Create batches
    batches = make_publish_batches(entries, group_by)

    # Step 5: Print summary table
    console.print(f"\n[bold]Publish Summary[/bold] ({len(entries)} entries)")
    console.print("")
    for batch in batches:
        console.print(
            f"  [bold]{batch.group_key}[/bold]: "
            f"{len(batch.entries)} entries "
            f"(confidence: {batch.confidence_tier})"
        )
        if verbose:
            for entry in batch.entries:
                conf = f"{entry.provenance.confidence:.2f}"
                console.print(f"    - {entry.name} [{conf}]")

    # Step 6: Generate YAML files
    if dry_run:
        console.print(
            f"\n[yellow]Dry run — would write {len(entries)} files to {output_dir}[/yellow]"
        )
    else:
        out = Path(output_dir)
        written = generate_catalog_files(entries, out)
        console.print(f"\n[green]Wrote {len(written)} YAML files to {out}[/green]")

        # Step 6b: Update review_status in graph
        from imas_codex.standard_names.graph_ops import update_review_status

        published_names = [e.name for e in entries]
        updated = update_review_status(published_names, status="published")
        console.print(
            f"  Updated [bold]{updated}[/bold] names to review_status='published'"
        )

    # Step 7: Optionally create PRs
    if create_pr:
        for batch in batches:
            branch = f"sn/{batch.group_key}/{batch.confidence_tier}"
            branch = branch.replace(" ", "-").lower()
            yaml_files = (
                [Path(output_dir) / f"{e.name}.yaml" for e in batch.entries]
                if not dry_run
                else []
            )
            pr_url = create_catalog_pr(
                batch=batch,
                catalog_repo=catalog_repo,
                branch_name=branch,
                yaml_files=yaml_files,
                dry_run=dry_run,
            )
            if pr_url:
                console.print(f"  PR: {pr_url}")
            elif dry_run:
                console.print(
                    f"  [yellow]Would create PR for {batch.group_key}[/yellow]"
                )


@sn.command("import")
@click.option(
    "--catalog-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to catalog directory containing YAML entries",
)
@click.option("--tags", type=str, default=None, help="Comma-separated tag filter")
@click.option("--dry-run", is_flag=True, help="Preview without writing to graph")
@click.option(
    "--check",
    "check_mode",
    is_flag=True,
    help="Compare catalog vs graph without importing; report sync status",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def sn_import(
    catalog_dir: str,
    tags: str | None,
    dry_run: bool,
    check_mode: bool,
    verbose: bool,
) -> None:
    """Import reviewed catalog entries into the graph.

    \b
    Reads YAML files from the catalog directory, validates them against
    the imas-standard-names catalog model, derives grammar fields, and
    MERGEs into the graph with review_status='accepted'.

    \b
    Use --check to compare catalog vs graph without importing.

    \b
    Examples:
      imas-codex sn import --catalog-dir ../imas-standard-names-catalog/standard_names
      imas-codex sn import --catalog-dir <path> --dry-run
      imas-codex sn import --catalog-dir <path> --tags equilibrium,core-physics
      imas-codex sn import --catalog-dir <path> --check
    """
    from pathlib import Path

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    tag_filter = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    # -- Check mode --
    if check_mode:
        console.print("\n[bold]Standard Name Catalog Check[/bold]")
        console.print(f"  Catalog: {catalog_dir}")
        if tag_filter:
            console.print(f"  Tag filter: {', '.join(tag_filter)}")
        console.print("")

        try:
            from imas_codex.standard_names.catalog_import import check_catalog

            cr = check_catalog(
                catalog_dir=Path(catalog_dir),
                tag_filter=tag_filter,
            )
        except ImportError as e:
            console.print(
                f"[red]Missing dependency:[/red] {e}\n"
                "Install with: uv pip install imas-standard-names"
            )
            raise SystemExit(1) from e
        except Exception as e:
            console.print(f"[red]Check error:[/red] {e}")
            raise SystemExit(1) from e

        # Print check results
        if cr.catalog_commit_sha:
            console.print(f"  Catalog SHA: {cr.catalog_commit_sha[:12]}")
        if cr.graph_commit_sha:
            console.print(f"  Graph SHA:   {cr.graph_commit_sha[:12]}")
        if cr.catalog_commit_sha and cr.graph_commit_sha:
            if cr.catalog_commit_sha == cr.graph_commit_sha:
                console.print("  [green]SHAs match[/green]")
            else:
                console.print("  [yellow]SHAs differ[/yellow]")
        console.print("")

        console.print(f"  In sync: [green]{cr.in_sync}[/green]")
        if cr.only_in_catalog:
            console.print(
                f"  Only in catalog: [yellow]{len(cr.only_in_catalog)}[/yellow]"
            )
            for name in cr.only_in_catalog[:10]:
                console.print(f"    + {name}")
            if len(cr.only_in_catalog) > 10:
                console.print(f"    ... and {len(cr.only_in_catalog) - 10} more")
        if cr.only_in_graph:
            console.print(f"  Only in graph: [yellow]{len(cr.only_in_graph)}[/yellow]")
            for name in cr.only_in_graph[:10]:
                console.print(f"    - {name}")
            if len(cr.only_in_graph) > 10:
                console.print(f"    ... and {len(cr.only_in_graph) - 10} more")
        if cr.diverged:
            console.print(f"  Diverged: [red]{len(cr.diverged)}[/red]")
            for item in cr.diverged[:10]:
                fields = ", ".join(item["fields"].keys())
                console.print(f"    ~ {item['name']} ({fields})")
            if len(cr.diverged) > 10:
                console.print(f"    ... and {len(cr.diverged) - 10} more")
        if not cr.only_in_catalog and not cr.only_in_graph and not cr.diverged:
            console.print("\n  [green]✓ Catalog and graph are in sync[/green]")
        return

    console.print("\n[bold]Standard Name Catalog Import[/bold]")
    console.print(f"  Catalog: {catalog_dir}")
    if tag_filter:
        console.print(f"  Tag filter: {', '.join(tag_filter)}")
    if dry_run:
        console.print("  Mode: [yellow]dry run[/yellow]")
    console.print("")

    try:
        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(
            catalog_dir=Path(catalog_dir),
            dry_run=dry_run,
            tag_filter=tag_filter,
        )
    except ImportError as e:
        console.print(
            f"[red]Missing dependency:[/red] {e}\n"
            "Install with: uv pip install imas-standard-names"
        )
        raise SystemExit(1) from e
    except Exception as e:
        console.print(f"[red]Import error:[/red] {e}")
        raise SystemExit(1) from e

    # Print results
    if result.catalog_commit_sha:
        console.print(f"  Catalog SHA: {result.catalog_commit_sha[:12]}")

    if result.errors:
        console.print(f"  [red]Errors: {len(result.errors)}[/red]")
        for err in result.errors[:10]:
            console.print(f"    - {err}")
        if len(result.errors) > 10:
            console.print(f"    ... and {len(result.errors) - 10} more")

    if result.skipped:
        console.print(f"  [yellow]Skipped: {result.skipped}[/yellow] (tag filter)")

    action = "Would import" if dry_run else "Imported"
    console.print(f"\n  [green]{action}: {result.imported}[/green] entries")

    if dry_run and result.entries:
        console.print("\n  [bold]Preview:[/bold]")
        for entry in result.entries[:20]:
            units = f" [{entry.get('units', '')}]" if entry.get("units") else ""
            console.print(f"    - {entry['id']}{units}")
        if len(result.entries) > 20:
            console.print(f"    ... and {len(result.entries) - 20} more")


@sn.command("reset")
@click.option("--status", required=True, help="Reset names with this review_status")
@click.option(
    "--to",
    "to_status",
    default=None,
    help="Target review_status after reset (default: clear fields only)",
)
@click.option(
    "--source",
    type=click.Choice(["dd", "signals"]),
    default=None,
    help="Filter by source ('dd' or 'signals')",
)
@click.option("--ids", "ids_filter", default=None, help="Filter to specific IDS")
@click.option("--dry-run", is_flag=True, help="Preview without modifying the graph")
def sn_reset(
    status: str,
    to_status: str | None,
    source: str | None,
    ids_filter: str | None,
    dry_run: bool,
) -> None:
    """Reset standard names for re-processing.

    Clears transient fields (embedding, model, confidence, generated_at) and
    removes HAS_STANDARD_NAME / CANONICAL_UNITS relationships for matching
    nodes, optionally changing their review_status.

    \b
    Examples:
      imas-codex sn reset --status drafted --dry-run
      imas-codex sn reset --status drafted --to extracted --ids equilibrium
      imas-codex sn reset --status drafted --source dd
    """
    from imas_codex.standard_names.graph_ops import reset_standard_names

    try:
        count = reset_standard_names(
            from_status=status,
            to_status=to_status,
            source_filter=source,
            ids_filter=ids_filter,
            dry_run=dry_run,
        )
    except Exception as e:
        console.print(f"[red]Reset error:[/red] {e}")
        raise SystemExit(1) from e

    qualifier = "Would reset" if dry_run else "Reset"
    to_note = f" → {to_status}" if to_status else " (fields cleared)"
    console.print(f"{qualifier} {count} StandardName node(s){to_note}")


@sn.command("clear")
@click.option(
    "--status",
    default=None,
    help="Delete names with this review_status (e.g. drafted)",
)
@click.option(
    "--all",
    "clear_all",
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
def sn_clear(
    status: str | None,
    clear_all: bool,
    source: str | None,
    ids_filter: str | None,
    include_accepted: bool,
    dry_run: bool,
) -> None:
    """Delete standard names from the graph.

    Relationship-first safety model: HAS_STANDARD_NAME edges are removed
    before deleting nodes; scoped deletes only remove orphaned nodes.

    \b
    Examples:
      imas-codex sn clear --status drafted --dry-run
      imas-codex sn clear --all --source dd --ids equilibrium --dry-run
      imas-codex sn clear --all --include-accepted --dry-run
    """
    if not status and not clear_all:
        raise click.UsageError("Provide --status <value> or --all to select names.")

    status_filter = None if clear_all else ([status] if status else None)

    from imas_codex.standard_names.graph_ops import clear_standard_names

    try:
        count = clear_standard_names(
            status_filter=status_filter,
            source_filter=source,
            ids_filter=ids_filter,
            include_accepted=include_accepted,
            dry_run=dry_run,
        )
    except Exception as e:
        console.print(f"[red]Clear error:[/red] {e}")
        raise SystemExit(1) from e

    qualifier = "Would delete" if dry_run else "Deleted"
    console.print(f"{qualifier} {count} StandardName node(s)")


@sn.command("seed")
@click.option(
    "--source",
    type=click.Choice(["isn", "west", "all"]),
    default="all",
    help="Which source to import: ISN reference examples, WEST catalog, or both",
)
@click.option(
    "--west-dir",
    type=click.Path(exists=True),
    default=None,
    help="Path to west-standard-names/standard_names (default: ~/Code/west-standard-names/standard_names)",
)
@click.option(
    "--dry-run", is_flag=True, help="Validate and count but don't write to graph"
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def sn_seed(
    source: str,
    west_dir: str | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Seed the graph with reference standard names from external sources.

    \b
    Imports curated standard names from ISN reference examples (42 entries,
    accepted) and/or the WEST catalog (~305 entries, drafted).

    \b
    ISN examples are shipped with imas-standard-names and serve as
    calibration anchors. WEST entries receive physics_domain and tag
    cleanup before ISN validation.

    \b
    Examples:
      imas-codex sn seed                          # import both ISN + WEST
      imas-codex sn seed --source isn --dry-run    # preview ISN only
      imas-codex sn seed --source west --west-dir /path/to/standard_names
    """
    from pathlib import Path

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    console.print("\n[bold]Standard Name Seed[/bold]")
    if dry_run:
        console.print("  Mode: [yellow]dry run[/yellow]")
    console.print("")

    try:
        from imas_codex.standard_names.seed import seed_isn_examples, seed_west_catalog
    except ImportError as e:
        console.print(
            f"[red]Missing dependency:[/red] {e}\n"
            "Install with: uv pip install imas-standard-names"
        )
        raise SystemExit(1) from e

    # -- ISN reference examples --
    if source in ("isn", "all"):
        console.print("[bold]ISN reference examples[/bold]")
        try:
            isn_result = seed_isn_examples(dry_run=dry_run)
        except Exception as e:
            console.print(f"  [red]Error:[/red] {e}")
            raise SystemExit(1) from e

        console.print(f"  Loaded:    {isn_result.loaded}")
        console.print(f"  Validated: [green]{isn_result.validated}[/green]")

        if isn_result.validation_errors:
            console.print(
                f"  Errors:    [red]{len(isn_result.validation_errors)}[/red]"
            )
            for err in isn_result.validation_errors[:5]:
                console.print(f"    - {err}")
            if len(isn_result.validation_errors) > 5:
                console.print(
                    f"    ... and {len(isn_result.validation_errors) - 5} more"
                )

        if isn_result.grammar_mismatches:
            console.print(
                f"  Grammar mismatches: [yellow]{len(isn_result.grammar_mismatches)}[/yellow]"
            )
            for m in isn_result.grammar_mismatches[:5]:
                console.print(f"    ⚠ {m}")

        action = "Would write" if dry_run else "Wrote"
        written = isn_result.validated if dry_run else isn_result.written
        console.print(f"  {action}:   [green]{written}[/green] entries")
        console.print("")

    # -- WEST catalog --
    if source in ("west", "all"):
        west_path = Path(west_dir) if west_dir else None
        console.print("[bold]WEST catalog[/bold]")
        if west_path:
            console.print(f"  Directory: {west_path}")

        try:
            west_result = seed_west_catalog(west_dir=west_path, dry_run=dry_run)
        except Exception as e:
            console.print(f"  [red]Error:[/red] {e}")
            raise SystemExit(1) from e

        console.print(f"  Loaded:    {west_result.loaded}")
        console.print(f"  Validated: [green]{west_result.validated}[/green]")

        if west_result.validation_errors:
            console.print(
                f"  Errors:    [red]{len(west_result.validation_errors)}[/red]"
            )
            for err in west_result.validation_errors[:5]:
                console.print(f"    - {err}")
            if len(west_result.validation_errors) > 5:
                console.print(
                    f"    ... and {len(west_result.validation_errors) - 5} more"
                )

        if west_result.grammar_mismatches:
            console.print(
                f"  Grammar mismatches: [yellow]{len(west_result.grammar_mismatches)}[/yellow]"
            )
            for m in west_result.grammar_mismatches[:5]:
                console.print(f"    ⚠ {m}")

        action = "Would write" if dry_run else "Wrote"
        written = west_result.validated if dry_run else west_result.written
        console.print(f"  {action}:   [green]{written}[/green] entries")
        console.print("")

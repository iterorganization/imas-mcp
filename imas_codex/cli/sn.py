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
      imas-codex sn generate --source dd [--physics-domain NAME]
      imas-codex sn generate --source signals --facility NAME

    \b
    Links:
      imas-codex sn resolve-links

    \b
    Status:
      imas-codex sn status
    """
    pass


@sn.command("generate")
@click.option(
    "--source",
    type=click.Choice(["dd", "signals"]),
    default="dd",
    show_default=True,
    help="Source to extract candidates from",
)
@click.option(
    "--physics-domain",
    "--domain",
    "domain_filter",
    type=str,
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
def sn_generate(
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
) -> None:
    """Generate standard names from a source.

    \b
    Examples:
      imas-codex sn generate --physics-domain equilibrium --dry-run
      imas-codex sn generate --physics-domain magnetics -c 2
      imas-codex sn generate --source signals --facility tcv --physics-domain magnetics
      imas-codex sn generate --paths equilibrium/time_slice/profiles_1d/psi --paths equilibrium/time_slice/profiles_1d/q
      imas-codex sn generate --paths "equilibrium/time_slice/profiles_1d/psi equilibrium/time_slice/profiles_1d/q"
    """
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

    from imas_codex.standard_names.pipeline import run_sn_generate_engine
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
        limit=limit,
        compose_model=compose_model,
        from_model=from_model,
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
        model_section="sn-generate",
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
        ]:
            count = source_stats.get(status_name, 0)
            total += count
            if count > 0:
                source_table.add_row(status_name, str(count))
        source_table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")
        console.print(source_table)


@sn.command("gaps")
@click.option(
    "--segment",
    default=None,
    help="Filter by grammar segment (e.g., transformation, process)",
)
@click.option(
    "--export",
    "export_format",
    type=click.Choice(["table", "yaml"]),
    default="table",
    show_default=True,
    help="Output format (yaml for ISN issue filing)",
)
def sn_gaps(segment: str | None, export_format: str) -> None:
    """List grammar vocabulary gaps identified during composition.

    VocabGap nodes record missing grammar tokens found when the LLM
    could not compose a valid standard name. Use --export yaml to
    generate ISN issue filing data.

    \b
    Examples:
      imas-codex sn gaps
      imas-codex sn gaps --segment transformation
      imas-codex sn gaps --export yaml
    """
    from rich.table import Table

    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        # Query VocabGap nodes with source counts
        params: dict = {}
        where_clause = ""
        if segment:
            where_clause = "WHERE vg.segment = $segment"
            params["segment"] = segment

        results = list(
            gc.query(
                f"""
                MATCH (vg:VocabGap)
                {where_clause}
                OPTIONAL MATCH (src)-[:HAS_STANDARD_NAME_VOCAB_GAP]->(vg)
                WITH vg, count(src) AS source_count,
                     collect(DISTINCT labels(src)[0]) AS source_types
                RETURN vg.id AS id,
                       vg.segment AS segment,
                       vg.needed_token AS needed_token,
                       vg.example_count AS example_count,
                       source_count,
                       source_types,
                       vg.first_seen_at AS first_seen,
                       vg.last_seen_at AS last_seen
                ORDER BY vg.segment, source_count DESC
                """,
                **params,
            )
        )

    if not results:
        console.print("[dim]No vocabulary gaps found.[/dim]")
        if segment:
            console.print(f"[dim]  (filtered by segment={segment})[/dim]")
        return

    if export_format == "yaml":
        import yaml

        export_data = []
        for r in results:
            entry = {
                "segment": r["segment"],
                "needed_token": r["needed_token"],
                "example_count": r["example_count"] or r["source_count"],
                "sources": r["source_types"],
            }
            export_data.append(entry)

        console.print(yaml.dump(export_data, default_flow_style=False, sort_keys=False))
        return

    # Table format (default)
    table = Table(title="Vocabulary Gaps")
    table.add_column("Segment", style="cyan")
    table.add_column("Needed Token", style="bold")
    table.add_column("Sources", justify="right")
    table.add_column("Example Count", justify="right")
    table.add_column("First Seen")
    table.add_column("Last Seen")

    for r in results:
        first_seen = str(r["first_seen"])[:10] if r["first_seen"] else "—"
        last_seen = str(r["last_seen"])[:10] if r["last_seen"] else "—"
        table.add_row(
            r["segment"],
            r["needed_token"],
            str(r["source_count"]),
            str(r["example_count"] or "—"),
            first_seen,
            last_seen,
        )

    console.print(table)
    console.print(f"\n[dim]{len(results)} gaps found[/dim]")


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
    removes HAS_STANDARD_NAME / HAS_UNIT relationships for matching
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
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--include-sources",
    is_flag=True,
    default=False,
    help="Also delete StandardNameSource nodes",
)
def sn_clear(
    status: str | None,
    clear_all: bool,
    source: str | None,
    ids_filter: str | None,
    include_accepted: bool,
    dry_run: bool,
    force: bool,
    include_sources: bool,
) -> None:
    """Delete standard names from the graph.

    Relationship-first safety model: HAS_STANDARD_NAME edges are removed
    before deleting nodes; scoped deletes only remove orphaned nodes.

    Shows a preview count and requires confirmation before deleting.
    Use --force to skip the confirmation prompt.

    \b
    Examples:
      imas-codex sn clear --status drafted              # Clear drafted names
      imas-codex sn clear --all --source dd --ids equilibrium
      imas-codex sn clear --all --include-accepted --dry-run
      imas-codex sn clear --all --force                 # Skip confirmation
    """
    if not status and not clear_all:
        raise click.UsageError("Provide --status <value> or --all to select names.")

    status_filter = None if clear_all else ([status] if status else None)

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
                # Build filter for StandardNameSource nodes matching the same scope
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
        console.print(f"[red]Clear error:[/red] {e}")
        raise SystemExit(1) from e


@sn.command("reconcile")
@click.option(
    "--source-type",
    type=click.Choice(["dd", "signals"]),
    default="dd",
    help="Source type to reconcile",
)
def reconcile(source_type: str) -> None:
    """Reconcile StandardNameSource nodes after DD/signal rebuild.

    Re-links sources to upstream entities, marks missing as stale,
    and revives previously-stale sources that reappear.
    """
    from imas_codex.standard_names.graph_ops import reconcile_standard_name_sources

    console.print(f"Reconciling {source_type} sources...")

    result = reconcile_standard_name_sources(source_type)

    console.print(f"  Stale marked: {result['stale_marked']}")
    console.print(f"  Revived: {result['revived']}")
    console.print(f"  Re-linked: {result['relinked']}")
    console.print("[green]Reconciliation complete[/green]")


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


@sn.command("resolve-links")
@click.option("--limit", type=int, default=50, help="Max names to process per round")
@click.option(
    "--rounds",
    type=int,
    default=3,
    help="Number of resolution rounds (names may become resolvable between rounds)",
)
@click.option("--dry-run", is_flag=True, help="Check resolvability without writing")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def sn_resolve_links(
    limit: int,
    rounds: int,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Resolve dd: links to name: links in standard names.

    \b
    Links with dd: prefix point to DD paths that may now have standard names.
    This command checks each unresolved link and replaces it with name: if
    the target path has been named.

    \b
    Examples:
      imas-codex sn resolve-links
      imas-codex sn resolve-links --rounds 5 --limit 100
    """
    from imas_codex.cli.utils import setup_logging

    setup_logging("DEBUG" if verbose else "WARNING")

    from imas_codex.standard_names.graph_ops import (
        claim_unresolved_links,
        resolve_links_batch,
    )

    total_resolved = 0
    total_unresolved = 0
    total_failed = 0

    for round_num in range(1, rounds + 1):
        if dry_run:
            # Just count unresolved
            from imas_codex.graph.client import GraphClient

            with GraphClient() as gc:
                rows = list(
                    gc.query(
                        """
                        MATCH (sn:StandardName)
                        WHERE sn.link_status = 'unresolved'
                        RETURN count(sn) AS count
                        """
                    )
                )
                count = rows[0]["count"] if rows else 0
            console.print(
                f"[dim]Round {round_num}:[/dim] {count} names with unresolved links"
            )
            break

        items = claim_unresolved_links(limit=limit)
        if not items:
            console.print(
                f"[dim]Round {round_num}:[/dim] No unresolved links remaining"
            )
            break

        result = resolve_links_batch(items)
        total_resolved += result["resolved"]
        total_unresolved += result["unresolved"]
        total_failed += result["failed"]

        console.print(
            f"[dim]Round {round_num}:[/dim] "
            f"[green]{result['resolved']}[/green] resolved, "
            f"[yellow]{result['unresolved']}[/yellow] still unresolved, "
            f"[red]{result['failed']}[/red] failed"
        )

        if result["unresolved"] == 0:
            break

    console.print(
        f"\n[bold]Total:[/bold] "
        f"[green]{total_resolved}[/green] resolved, "
        f"[yellow]{total_unresolved}[/yellow] unresolved, "
        f"[red]{total_failed}[/red] failed"
    )


@sn.command("review")
@click.option("--ids", default=None, help="Scope to names linked to specific IDS")
@click.option("--domain", default=None, help="Scope to physics domain")
@click.option(
    "--status",
    "status_filter",
    default="drafted",
    help="Filter by review_status (default: drafted)",
)
@click.option(
    "--unreviewed",
    is_flag=True,
    help="Only names with no reviewer_score or stale review",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force re-review of already-scored names",
)
@click.option("--model", default=None, help="Override review model")
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
def sn_review(
    ids: str | None,
    domain: str | None,
    status_filter: str,
    unreviewed: bool,
    force: bool,
    model: str | None,
    batch_size: int,
    neighborhood: int,
    cost_limit: float,
    dry_run: bool,
    skip_audit: bool,
    concurrency: int,
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
      imas-codex sn review --force --domain magnetics
    """
    import asyncio

    from imas_codex.standard_names.review.budget import ReviewBudgetManager
    from imas_codex.standard_names.review.state import StandardNameReviewState

    # Enforce batch-size cap
    batch_size = min(batch_size, 25)

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
        review_model=model,
        batch_size=batch_size,
        neighborhood_k=neighborhood,
        concurrency=concurrency,
        dry_run=dry_run,
        budget_manager=ReviewBudgetManager(cost_limit),
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
                               sn.physical_base AS physical_base,
                               sn.subject AS subject,
                               sn.component AS component,
                               sn.coordinate AS coordinate,
                               sn.position AS position,
                               sn.process AS process,
                               sn.cocos_transformation_type AS cocos_transformation_type,
                               sn.physics_domain AS physics_domain,
                               sn.review_status AS review_status,
                               sn.reviewer_score AS reviewer_score,
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
                    n for n in targets if n.get("review_status") == status_filter
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
                    if n.get("reviewer_score") is None
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
                    f"    {ls.get('id', '?')}: {ls.get('reviewer_score', 0):.2f}"
                    f" ({ls.get('review_tier', '?')})"
                )

        # Budget summary
        if state.budget_manager:
            bs = state.budget_manager.summary
            console.print(
                f"  Budget: ${bs['total_actual']:.4f} used of"
                f" ${bs['total_budget']:.2f} ({bs['batch_count']} batches)"
            )

    asyncio.run(_run())


@sn.command("enrich")
@click.option(
    "--domain",
    multiple=True,
    default=None,
    help="Filter by physics domain (repeatable, e.g. --domain equilibrium --domain transport)",
)
@click.option(
    "--status",
    "status_filter",
    default="named",
    show_default=True,
    help="Review status(es) to enrich from (comma-separated or repeated)",
)
@click.option(
    "-c",
    "--cost-limit",
    type=float,
    default=2.0,
    show_default=True,
    help="Maximum LLM spend in USD",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Cap total standard names to enrich",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    show_default=True,
    help="LLM batch size (names per request)",
)
@click.option("--dry-run", is_flag=True, help="Preview candidates without graph writes")
@click.option(
    "--force",
    is_flag=True,
    help="Re-enrich names already at review_status='enriched'",
)
@click.option(
    "--model",
    "model_override",
    type=str,
    default=None,
    help="Override LLM model (default: sn-enrich from settings)",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("-q", "--quiet", is_flag=True, help="Suppress progress output")
def sn_enrich(
    domain: tuple[str, ...],
    status_filter: str,
    cost_limit: float,
    limit: int | None,
    batch_size: int,
    dry_run: bool,
    force: bool,
    model_override: str | None,
    verbose: bool,
    quiet: bool,
) -> None:
    """Enrich existing standard names with documentation.

    \b
    Takes named standard names as input and generates documentation
    fields (description, documentation, tags, links) using linked
    DD paths as context.  Runs the five-phase pipeline:
    EXTRACT → CONTEXTUALISE → DOCUMENT → VALIDATE → PERSIST.

    \b
    Does NOT change: name, grammar fields, kind, or unit.

    \b
    Examples:
      imas-codex sn enrich --domain equilibrium -c 2.0
      imas-codex sn enrich --domain transport --domain magnetics --limit 50 --dry-run
      imas-codex sn enrich --force --model openrouter/anthropic/claude-opus-4.7
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

    # --- Parse domain / status lists ---
    domain_list: list[str] | None = list(domain) if domain else None
    # status_filter can be comma-separated or repeated
    statuses = [
        s.strip() for part in status_filter.split(",") for s in [part] if s.strip()
    ]

    # --- Validation warning ---
    if not domain_list and limit is None:
        console.print(
            "[yellow]⚠ Enriching all named SNs — "
            "use --domain or --limit to scope[/yellow]"
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

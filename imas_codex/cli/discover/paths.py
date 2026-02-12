"""Paths discovery command: Directory structure scanning and scoring."""

from __future__ import annotations

import asyncio
import logging
import re
import sys

import click
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)


@click.command()
@click.argument("facility")
@click.option(
    "--root",
    "-r",
    multiple=True,
    type=str,
    help="Restrict discovery to these root paths (can specify multiple)",
)
@click.option(
    "--cost-limit",
    "-c",
    type=float,
    default=10.0,
    help="Maximum LLM spend in USD (default: $10)",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=None,
    help="Maximum paths to process (for debugging)",
)
@click.option(
    "--focus",
    "-f",
    type=str,
    help="Natural language focus (e.g., 'equilibrium codes')",
)
@click.option(
    "--threshold",
    "-t",
    default=0.7,
    type=float,
    help="Minimum score to expand paths",
)
@click.option(
    "--scan-workers",
    default=1,
    type=int,
    help="Number of scan workers (default: 1, single SSH connection)",
)
@click.option(
    "--score-workers",
    default=2,
    type=int,
    help="Number of score workers (default: 2, parallel LLM calls)",
)
@click.option(
    "--scan-only",
    is_flag=True,
    default=False,
    help="SSH scan only, no LLM scoring (fast, requires SSH access)",
)
@click.option(
    "--score-only",
    is_flag=True,
    default=False,
    help="LLM scoring only, no SSH scanning (offline, graph-only)",
)
@click.option(
    "--no-rich",
    is_flag=True,
    default=False,
    help="Use logging output instead of rich progress display",
)
@click.option(
    "--add-roots",
    is_flag=True,
    default=False,
    help="Add missing discovery_roots from facility config",
)
@click.option(
    "--enrich-threshold",
    type=float,
    default=None,
    help="Auto-enrich paths scoring >= threshold (e.g., 0.75)",
)
@click.option(
    "--time",
    "time_limit",
    type=int,
    default=None,
    help="Maximum runtime in minutes (e.g., 10). Discovery halts when time expires.",
)
def paths(
    facility: str,
    root: tuple[str, ...],
    cost_limit: float,
    limit: int | None,
    focus: str | None,
    threshold: float,
    scan_workers: int,
    score_workers: int,
    scan_only: bool,
    score_only: bool,
    no_rich: bool,
    add_roots: bool,
    enrich_threshold: float | None,
    time_limit: int | None,
) -> None:
    """Discover and score directory structure at a facility.

    Parallel scan workers enumerate directories via SSH while score workers
    classify paths using LLM. Both run concurrently with the graph as
    coordination. Discovery is idempotent - rerun to continue from current state.

    \b
    Examples:
      imas-codex discover paths <facility>                 # Default $10 limit
      imas-codex discover paths <facility> -c 20.0         # $20 limit
      imas-codex discover paths iter --focus "equilibrium"  # Focus scoring
      imas-codex discover paths iter --scan-only            # SSH only, no LLM
      imas-codex discover paths iter --score-only           # LLM only, no SSH
      imas-codex discover paths tcv -r /home/codes/astra    # Deep dive
    """
    console = Console()

    # Validate mutually exclusive flags
    if scan_only and score_only:
        console.print(
            "[red]Error: --scan-only and --score-only are mutually exclusive[/red]"
        )
        raise SystemExit(1)

    # Convert root tuple to list or None
    root_filter = list(root) if root else None

    _run_iterative_discovery(
        facility=facility,
        budget=cost_limit,
        limit=limit,
        focus=focus,
        threshold=threshold,
        num_scan_workers=scan_workers,
        num_score_workers=score_workers,
        scan_only=scan_only,
        score_only=score_only,
        no_rich=no_rich,
        root_filter=root_filter,
        add_roots=add_roots,
        enrich_threshold=enrich_threshold,
        timeout_minutes=time_limit,
    )


# =============================================================================
# Internal Implementation
# =============================================================================


def _run_iterative_discovery(
    facility: str,
    budget: float,
    limit: int | None,
    focus: str | None,
    threshold: float,
    num_scan_workers: int = 1,
    num_score_workers: int = 3,
    scan_only: bool = False,
    score_only: bool = False,
    no_rich: bool = False,
    root_filter: list[str] | None = None,
    add_roots: bool = False,
    enrich_threshold: float | None = None,
    timeout_minutes: int | None = None,
) -> None:
    """Run parallel scan/score discovery."""
    import time as time_module

    from imas_codex.discovery import (
        get_discovery_stats,
        seed_facility_roots,
        seed_missing_roots,
    )
    from imas_codex.settings import get_model

    # Compute deadline from timeout
    start_time = time_module.time()
    deadline: float | None = None
    if timeout_minutes is not None:
        deadline = start_time + (timeout_minutes * 60)

    # Auto-detect if rich can run (TTY check) or use no_rich flag
    use_rich = not no_rich and sys.stdout.isatty()

    # Always configure file logging (DEBUG level to disk)
    from imas_codex.cli.logging import configure_cli_logging

    configure_cli_logging("paths")

    if use_rich:
        console = Console()
    else:
        console = None
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )

    disc_logger = logging.getLogger("imas_codex.discovery")
    if not use_rich:
        disc_logger.setLevel(logging.INFO)

    def log_print(msg: str, style: str = "") -> None:
        """Print to console or log, stripping rich markup."""
        clean_msg = re.sub(r"\[[^\]]+\]", "", msg)
        if console:
            console.print(msg)
        else:
            disc_logger.info(clean_msg)

    # Get initial stats to determine next steps
    stats = get_discovery_stats(facility)

    # Handle --add-roots flag
    if add_roots:
        log_print("[cyan]Checking for missing discovery_roots...[/cyan]")
        seeded = seed_missing_roots(facility)
        if seeded > 0:
            log_print(f"[green]Added {seeded} new root path(s) from config[/green]")
        else:
            log_print("[dim]All discovery_roots already in graph[/dim]")
        stats = get_discovery_stats(facility)

    # Handle targeted deep dive with --root
    if root_filter:
        log_print(f"[cyan]Targeted discovery: {len(root_filter)} root(s)[/cyan]")
        for r in root_filter:
            log_print(f"  • {r}")
        seeded = seed_facility_roots(facility, root_paths=root_filter)
        if seeded > 0:
            log_print(f"[green]Added {seeded} new root path(s)[/green]")
        stats = get_discovery_stats(facility)
    elif stats["total"] == 0:
        if score_only:
            log_print(
                "[red]Error: --score-only requires existing paths in the graph.[/red]"
            )
            log_print(
                f"[yellow]Run 'imas-codex discover paths {facility}' or "
                "'--scan-only' first to populate the graph.[/yellow]"
            )
            raise SystemExit(1)
        log_print(f"[cyan]Seeding root paths for {facility}...[/cyan]")
        seed_facility_roots(facility)
        stats = get_discovery_stats(facility)

    if score_only and stats.get("scanned", 0) == 0:
        log_print("[yellow]Warning: No 'scanned' paths available for scoring.[/yellow]")
        log_print("Checking for already-scored paths to expand...")

    # Adjust worker counts based on mode flags
    effective_scan_workers = 0 if score_only else num_scan_workers
    effective_score_workers = 0 if scan_only else num_score_workers

    # Get model name for display
    model_name = get_model("language")
    if model_name.startswith("anthropic/"):
        model_name = model_name[len("anthropic/") :]

    # Display mode
    mode_str = ""
    if scan_only:
        mode_str = " [bold cyan](SCAN ONLY)[/bold cyan]"
    elif score_only:
        mode_str = " [bold green](SCORE ONLY)[/bold green]"

    log_print(
        f"[bold]Starting parallel discovery for {facility.upper()}[/bold]{mode_str}"
    )
    if not scan_only:
        log_print(f"Cost limit: ${budget:.2f}")
    if limit:
        log_print(f"Path limit: {limit}")
    if timeout_minutes is not None:
        log_print(f"Time limit: {timeout_minutes} min")
    if not scan_only:
        log_print(f"Model: {model_name}")

    # Build worker count display
    worker_parts = []
    if not score_only and effective_scan_workers > 0:
        worker_parts.append(f"{effective_scan_workers} scan")
    if not score_only:
        worker_parts.append("1 expand")
    if not scan_only and effective_score_workers > 0:
        worker_parts.append(f"{effective_score_workers} score")
    if not scan_only:
        worker_parts.append("1 enrich")
        worker_parts.append("1 rescore")
    log_print(f"Workers: {', '.join(worker_parts)}")
    if focus and not scan_only:
        log_print(f"Focus: {focus}")

    # Run the async discovery loop
    try:
        result, scored_this_run = asyncio.run(
            _async_discovery_loop(
                facility=facility,
                budget=budget,
                limit=limit,
                focus=focus,
                threshold=threshold,
                console=console,
                num_scan_workers=effective_scan_workers,
                num_score_workers=effective_score_workers,
                scan_only=scan_only,
                score_only=score_only,
                use_rich=use_rich,
                root_filter=root_filter,
                auto_enrich_threshold=enrich_threshold,
                deadline=deadline,
            )
        )

        _print_discovery_summary(
            console, facility, result, scored_this_run, scan_only=scan_only
        )

    except KeyboardInterrupt:
        log_print("\n[yellow]Discovery interrupted by user[/yellow]")
        raise SystemExit(130) from None
    except Exception as e:
        log_print(f"\n[red]Error: {e}[/red]")
        raise SystemExit(1) from e


def _print_discovery_summary(
    console,
    facility: str,
    result: dict,
    scored_this_run: set[str] | None = None,
    scan_only: bool = False,
) -> None:
    """Print detailed discovery summary with statistics."""
    from imas_codex.discovery import get_discovery_stats
    from imas_codex.discovery.paths.frontier import (
        get_accumulated_cost,
        get_high_value_paths,
    )

    _ansi_pattern = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

    def clean_text(text: str) -> str:
        return escape(_ansi_pattern.sub("", text))

    disc_logger = logging.getLogger("imas_codex.discovery")

    # Get final graph stats
    stats = get_discovery_stats(facility)
    coverage = stats["scored"] / stats["total"] * 100 if stats["total"] > 0 else 0
    elapsed = result.get("elapsed_seconds", 0)

    if elapsed >= 3600:
        hours, rem = divmod(int(elapsed), 3600)
        mins = rem // 60
        elapsed_str = f"{hours}h {mins:02d}m" if mins else f"{hours}h"
    elif elapsed >= 60:
        mins, secs = divmod(int(elapsed), 60)
        elapsed_str = f"{mins}m {secs:02d}s" if secs else f"{mins}m"
    else:
        elapsed_str = f"{int(elapsed)}s"

    scan_rate = result.get("scan_rate")
    score_rate = result.get("score_rate")

    # Non-rich mode: log simple summary
    if console is None:
        disc_logger.info(
            f"Discovery complete: scanned={result['scanned']}, "
            f"scored={result['scored']}, cost=${result['cost']:.3f}, "
            f"elapsed={elapsed_str}"
        )
        disc_logger.info(
            f"Graph state: total={stats['total']}, scored={stats['scored']} "
            f"({coverage:.1f}%), pending={stats.get('pending', 0)}"
        )
        return

    # Rich mode: use panels
    console.print()

    facility_upper = facility.upper()
    summary = Text()
    panel_width = console.width or 100

    # Row 1: SCAN stats
    scanned = result["scanned"]
    expanded = result.get("expanded", 0)
    enriched = result.get("enriched", 0)
    summary.append("  SCAN  ", style="bold blue")
    summary.append(f"scanned={scanned:,}", style="white")
    summary.append(f"  expanded={expanded:,}", style="white")
    summary.append(f"  enriched={enriched:,}", style="white")
    if scan_rate:
        summary.append(f"  {scan_rate:.1f}/s", style="dim")
    summary.append("\n")

    # Row 2: SCORE stats
    if not scan_only:
        scored = result["scored"]
        cost = result.get("cost", 0.0)
        rescored = result.get("rescored", 0)
        summary.append("  SCORE ", style="bold green")
        summary.append(f"scored={scored:,}", style="white")
        summary.append(f"  rescored={rescored:,}", style="white")
        summary.append(f"  cost=${cost:.3f}", style="yellow")
        if score_rate:
            summary.append(f"  {score_rate:.1f}/s", style="dim")
        summary.append("\n")

    # Row 3: ENRICH stats
    enrichment_aggs = result.get("enrichment_aggregates", {})
    if enrichment_aggs and enrichment_aggs.get("total_bytes", 0) > 0:
        total_bytes = enrichment_aggs.get("total_bytes", 0)
        total_lines = enrichment_aggs.get("total_lines", 0)
        multiformat = enrichment_aggs.get("multiformat_count", 0)
        pattern_cats = enrichment_aggs.get("pattern_categories", {})

        def fmt_bytes(b: int) -> str:
            if b >= 1_000_000_000:
                return f"{b / 1_000_000_000:.1f}GB"
            if b >= 1_000_000:
                return f"{b / 1_000_000:.1f}MB"
            if b >= 1_000:
                return f"{b / 1_000:.1f}KB"
            return f"{b}B"

        def fmt_lines(ln: int) -> str:
            if ln >= 1_000_000:
                return f"{ln / 1_000_000:.1f}M"
            if ln >= 1_000:
                return f"{ln / 1_000:.1f}K"
            return str(ln)

        summary.append("  ENRICH", style="bold magenta")
        summary.append(f"  size={fmt_bytes(total_bytes)}", style="white")
        summary.append(f"  LOC={fmt_lines(total_lines)}", style="white")
        if multiformat:
            summary.append(f"  multiformat={multiformat}", style="cyan")

        if pattern_cats:
            top_cats = sorted(pattern_cats.items(), key=lambda x: x[1], reverse=True)[
                :4
            ]
            cat_strs = [f"{cat}={count}" for cat, count in top_cats]
            summary.append(f"  patterns: {', '.join(cat_strs)}", style="dim")
        summary.append("\n")

    # Row 4: USAGE
    summary.append("  USAGE ", style="bold cyan")
    summary.append(f"time={elapsed_str}", style="white")
    if not scan_only:
        cost_data = get_accumulated_cost(facility)
        total_cost = cost_data.get("total_cost", 0.0)
        summary.append(f"  total_cost=${total_cost:.2f}", style="yellow")

    # Title based on mode
    if scan_only:
        title = f"[bold blue]{facility_upper} Scan Complete[/bold blue]"
        border = "blue"
    else:
        title = f"[bold green]{facility_upper} Discovery Complete[/bold green]"
        border = "green"

    console.print(
        Panel(
            summary,
            title=title,
            border_style=border,
            width=panel_width,
        )
    )

    # Show high-value paths found IN THIS RUN only
    if scan_only:
        console.print()
        console.print(
            f"[dim]Next step: Run 'imas-codex discover paths {facility} --score-only' "
            "to score listed paths.[/dim]"
        )
        return

    all_high_value = get_high_value_paths(facility, min_score=0.7, limit=200)

    if scored_this_run:
        high_value = [p for p in all_high_value if p["path"] in scored_this_run]
    else:
        high_value = all_high_value

    if not high_value:
        return

    score_categories = {
        "score_modeling_code": "Modeling Code",
        "score_analysis_code": "Analysis Code",
        "score_operations_code": "Operations Code",
        "score_modeling_data": "Modeling Data",
        "score_experimental_data": "Experimental Data",
        "score_data_access": "Data Access",
        "score_workflow": "Workflow",
        "score_visualization": "Visualization",
        "score_documentation": "Documentation",
        "score_imas": "IMAS",
    }
    category_order = list(score_categories.keys())

    by_category: dict[str, list] = {cat: [] for cat in category_order}

    for p in high_value:
        max_cat = None
        max_score = -1.0
        for cat in category_order:
            cat_score = p.get(cat) or 0.0
            if cat_score > max_score:
                max_score = cat_score
                max_cat = cat
        if max_cat:
            by_category[max_cat].append(p)

    console.print()
    console.print(f"[bold]High-value paths this run ({len(high_value)}):[/bold]")

    def clip_path_inner(path: str, max_len: int) -> str:
        if len(path) <= max_len:
            return path
        keep_start = max_len // 3
        keep_end = max_len - keep_start - 5
        return f"{path[:keep_start]}/.../{path[-keep_end:]}"

    for cat_key in category_order:
        cat_paths = by_category.get(cat_key, [])
        if not cat_paths:
            continue

        sorted_paths = sorted(
            cat_paths, key=lambda p: p.get(cat_key) or 0.0, reverse=True
        )

        cat_name = score_categories[cat_key]
        console.print(f"  [bold cyan]{cat_name}[/bold cyan] ({len(sorted_paths)})")

        for p in sorted_paths[:3]:
            cat_score = p.get(cat_key) or 0.0
            path = p.get("path", "")
            description = p.get("description", "")
            should_expand = p.get("should_expand", True)
            terminal_reason = p.get("terminal_reason", "")

            max_path_len = 70 if terminal_reason else 88
            clipped_path = clip_path_inner(path, max_path_len)

            if cat_score >= 1.0:
                score_style = "bold green"
            elif cat_score >= 0.8:
                score_style = "green"
            else:
                score_style = "yellow"

            path_line = (
                f"    [{score_style}]{cat_score:.2f}[/{score_style}] "
                f"{clean_text(clipped_path)}"
            )
            if terminal_reason:
                reason_display = terminal_reason.replace("_", " ")
                path_line += f" [magenta]{reason_display}[/magenta]"
            elif not should_expand:
                path_line += " [magenta]terminal[/magenta]"

            reason = description[:80] + "..." if len(description) > 80 else description

            console.print(path_line, highlight=False)
            if reason:
                console.print(
                    f"         [dim]{clean_text(reason)}[/dim]", highlight=False
                )

        remaining = len(sorted_paths) - 3
        if remaining > 0:
            console.print(f"    [dim]... +{remaining} more[/dim]")


async def _async_discovery_loop(
    facility: str,
    budget: float,
    limit: int | None,
    focus: str | None,
    threshold: float,
    console,
    num_scan_workers: int = 2,
    num_score_workers: int = 2,
    scan_only: bool = False,
    score_only: bool = False,
    use_rich: bool = True,
    root_filter: list[str] | None = None,
    auto_enrich_threshold: float = 0.75,
    deadline: float | None = None,
) -> tuple[dict, set[str]]:
    """Async discovery loop with parallel scan/score workers."""
    from imas_codex.discovery.paths.parallel import run_parallel_discovery

    disc_logger = logging.getLogger("imas_codex.discovery")
    scored_this_run: set[str] = set()

    if use_rich:
        from imas_codex.discovery.paths.progress import ParallelProgressDisplay
        from imas_codex.settings import get_model

        model_name = get_model("language")

        with ParallelProgressDisplay(
            facility=facility,
            cost_limit=budget,
            path_limit=limit,
            model=model_name,
            console=console,
            focus=focus or "",
            scan_only=scan_only,
            score_only=score_only,
        ) as display:

            async def refresh_graph_state():
                while True:
                    display.refresh_from_graph(facility)
                    await asyncio.sleep(0.5)

            async def queue_ticker():
                while True:
                    display.tick()
                    await asyncio.sleep(0.15)

            refresh_task = asyncio.create_task(refresh_graph_state())
            ticker_task = asyncio.create_task(queue_ticker())

            def on_scan(msg, stats, paths=None, scan_results=None):
                display.update_scan(msg, stats, paths=paths, scan_results=scan_results)

            def on_expand(msg, stats, paths=None, scan_results=None):
                display.update_expand(
                    msg, stats, paths=paths, scan_results=scan_results
                )

            def on_score(msg, stats, results=None):
                display.update_score(msg, stats, results=results)

            def on_enrich(msg, stats, results=None):
                display.update_enrich(msg, stats, results=results)

            def on_rescore(msg, stats, results=None):
                display.update_rescore(msg, stats, results=results)

            try:
                result = await run_parallel_discovery(
                    facility=facility,
                    cost_limit=budget,
                    path_limit=limit,
                    focus=focus,
                    threshold=threshold,
                    root_filter=root_filter,
                    auto_enrich_threshold=auto_enrich_threshold,
                    num_scan_workers=num_scan_workers,
                    num_score_workers=num_score_workers,
                    on_scan_progress=on_scan,
                    on_expand_progress=on_expand,
                    on_score_progress=on_score,
                    on_enrich_progress=on_enrich,
                    on_rescore_progress=on_rescore,
                    deadline=deadline,
                )
            finally:
                refresh_task.cancel()
                ticker_task.cancel()
                try:
                    await refresh_task
                except asyncio.CancelledError:
                    pass
                try:
                    await ticker_task
                except asyncio.CancelledError:
                    pass

            display.refresh_from_graph(facility)
            scored_this_run = display.get_paths_scored_this_run()
            enrichment_aggregates = display.get_enrichment_aggregates()

    else:
        # Logging-based progress
        def on_scan_log(msg: str, stats, paths=None, scan_results=None):
            if scan_results and len(scan_results) > 0:
                disc_logger.info(
                    f"SCAN batch: {len(scan_results)} paths, "
                    f"total: {stats.processed}, rate: {stats.rate:.1f}/s"
                )

        def on_score_log(msg: str, stats, results=None):
            if results and len(results) > 0:
                for r in results:
                    if r.get("path"):
                        scored_this_run.add(r["path"])
                disc_logger.info(
                    f"SCORE batch: {len(results)} paths, "
                    f"total: {stats.processed}, cost: ${stats.cost:.3f}"
                )

        result = await run_parallel_discovery(
            facility=facility,
            cost_limit=budget,
            path_limit=limit,
            focus=focus,
            threshold=threshold,
            root_filter=root_filter,
            auto_enrich_threshold=auto_enrich_threshold,
            num_scan_workers=num_scan_workers,
            num_score_workers=num_score_workers,
            on_scan_progress=on_scan_log,
            on_score_progress=on_score_log,
            deadline=deadline,
        )
        enrichment_aggregates = {}

    return (
        {
            "cycles": 1,
            "scanned": result["scanned"],
            "scored": result["scored"],
            "expanded": result.get("expanded", 0),
            "enriched": result.get("enriched", 0),
            "rescored": result.get("rescored", 0),
            "cost": result["cost"],
            "elapsed_seconds": result["elapsed_seconds"],
            "scan_rate": result.get("scan_rate"),
            "score_rate": result.get("score_rate"),
            "enrichment_aggregates": enrichment_aggregates,
        },
        scored_this_run,
    )

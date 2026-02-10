"""Discovery commands: Graph-led facility exploration."""

from __future__ import annotations

import asyncio
import logging
import re
import sys

import click
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)


class DefaultGroup(click.Group):
    """Click group that routes to a default command when no subcommand matches.

    If the first argument is not a known subcommand (e.g. a facility name),
    it is treated as an argument to the default command.
    """

    def __init__(self, *args, default_cmd_name: str = "run", **kwargs):
        super().__init__(*args, **kwargs)
        self.default_cmd_name = default_cmd_name

    def resolve_command(self, ctx, args):
        cmd_name = args[0] if args else None
        if cmd_name and cmd_name not in self.commands:
            args.insert(0, self.default_cmd_name)
        return super().resolve_command(ctx, args)


@click.group()
def discover():
    """Discover facility resources with graph-led exploration.

    \b
    Top-level Commands:
      status             Show discovery statistics for all domains
      clear              Clear ALL discovery data (nuclear reset)
      seed               Seed root paths from config
      inspect            Debug view of scanned/scored paths

    \b
    Domain Subgroups (each has status, clear):
      paths              Directory structure discovery
      wiki               Wiki page discovery and ingestion
      signals            Facility signal discovery

    \b
    Examples:
      imas-codex discover status jet          # All domains
      imas-codex discover paths jet            # Run paths discovery
      imas-codex discover paths status jet     # Paths status only
      imas-codex discover wiki jet             # Run wiki discovery
      imas-codex discover wiki clear jet       # Clear wiki only
      imas-codex discover signals jet          # Run signals discovery
      imas-codex discover clear jet            # Clear ALL domains

    The graph is the single source of truth. All discovery operations
    are idempotent and resume from the current graph state.
    """
    pass


# =============================================================================
# Paths Subgroup
# =============================================================================


@click.group(cls=DefaultGroup)
def paths():
    """Directory structure discovery.

    Discovers and scores directory structures at remote facilities via
    parallel SSH scanning and LLM classification.

    \b
    Examples:
      imas-codex discover paths jet              # Default $10 limit
      imas-codex discover paths jet -c 20.0      # $20 limit
      imas-codex discover paths status jet       # Show statistics
      imas-codex discover paths clear jet        # Clear paths data
    """
    pass


discover.add_command(paths)


# =============================================================================
# Wiki Subgroup
# =============================================================================


@click.group(cls=DefaultGroup)
def wiki():
    """Wiki page discovery and ingestion.

    Discovers, scores, and ingests wiki pages into the documentation graph.

    \b
    Examples:
      imas-codex discover wiki jet               # Full discovery
      imas-codex discover wiki jet --scan-only   # Scan pages only
      imas-codex discover wiki status jet        # Show statistics
      imas-codex discover wiki clear jet         # Clear wiki data
    """
    pass


discover.add_command(wiki)


# =============================================================================
# Signals Subgroup
# =============================================================================


@click.group(cls=DefaultGroup)
def signals():
    """Facility signal discovery.

    Scans TDI function files to discover and classify data signals.

    \b
    Examples:
      imas-codex discover signals tcv              # Full pipeline
      imas-codex discover signals tcv --scan-only  # Scan only
      imas-codex discover signals status tcv       # Show statistics
      imas-codex discover signals clear tcv        # Clear signals data
    """
    pass


discover.add_command(signals)


# =============================================================================
# Paths Commands
# =============================================================================


@paths.command("run", hidden=True)
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
def paths_run(
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
) -> None:
    """Discover and score directory structure at a facility.

    \b
    Examples:
      imas-codex discover paths <facility>              # Default $10 limit
      imas-codex discover paths <facility> -c 20.0      # $20 limit
      imas-codex discover paths iter --focus "equilibrium codes"
      imas-codex discover paths iter --scan-only        # SSH only, no LLM
      imas-codex discover paths iter --score-only       # LLM only, no SSH
      imas-codex discover paths tcv -r /home/codes/astra  # Deep dive

    \b
    Targeted deep dives:
      --root, -r    Restrict discovery to specific roots.

    Parallel scan workers enumerate directories via SSH while score workers
    classify paths using LLM. Both run concurrently with the graph as
    coordination. Discovery is idempotent - rerun to continue from current state.
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
    )


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
) -> None:
    """Run parallel scan/score discovery."""
    from imas_codex.agentic.agents import get_model_for_task
    from imas_codex.discovery import (
        get_discovery_stats,
        seed_facility_roots,
        seed_missing_roots,
    )

    # Auto-detect if rich can run (TTY check) or use no_rich flag
    use_rich = not no_rich and sys.stdout.isatty()

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
    model_name = get_model_for_task("score")
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
        paths = by_category.get(cat_key, [])
        if not paths:
            continue

        sorted_paths = sorted(paths, key=lambda p: p.get(cat_key) or 0.0, reverse=True)

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
) -> tuple[dict, set[str]]:
    """Async discovery loop with parallel scan/score workers."""
    from imas_codex.discovery.paths.parallel import run_parallel_discovery

    disc_logger = logging.getLogger("imas_codex.discovery")
    scored_this_run: set[str] = set()

    if use_rich:
        from imas_codex.agentic.agents import get_model_for_task
        from imas_codex.discovery.paths.progress import ParallelProgressDisplay

        model_name = get_model_for_task("score")

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


# Status and management commands


@discover.command("status")
@click.argument("facility")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option(
    "--no-rich",
    "no_rich",
    is_flag=True,
    help="Disable rich output (for LLM tools in non-TTY)",
)
@click.option(
    "--domain",
    "-d",
    type=click.Choice(["paths", "wiki", "signals"]),
    help="Show status for specific domain only (default: all)",
)
def discover_status(
    facility: str, as_json: bool, no_rich: bool, domain: str | None
) -> None:
    """Show discovery statistics for a facility.

    By default shows status for all discovery domains: paths, wiki, and signals.
    Use --domain/-d to filter to a specific domain.

    \b
    Examples:
      imas-codex discover status jet           # All domains
      imas-codex discover status jet -d paths  # Paths only
      imas-codex discover status jet -d wiki   # Wiki only
      imas-codex discover status jet --json    # JSON output
    """
    import json as json_module
    import sys

    from imas_codex.discovery import get_discovery_stats, get_high_value_paths
    from imas_codex.discovery.data.parallel import get_data_discovery_stats
    from imas_codex.discovery.wiki.parallel import get_wiki_discovery_stats

    # Auto-detect TTY if --no-rich not explicitly set
    use_rich = not no_rich and sys.stdout.isatty()

    try:
        if as_json:
            output: dict = {"facility": facility}

            if domain is None or domain == "paths":
                stats = get_discovery_stats(facility)
                high_value = get_high_value_paths(facility, min_score=0.7, limit=20)
                output["paths"] = {"stats": stats, "high_value_paths": high_value}

            if domain is None or domain == "wiki":
                wiki_stats = get_wiki_discovery_stats(facility)
                output["wiki"] = wiki_stats

            if domain is None or domain == "signals":
                signal_stats = get_data_discovery_stats(facility)
                output["signals"] = signal_stats

            click.echo(json_module.dumps(output, indent=2))
        else:
            from imas_codex.discovery.paths.progress import print_discovery_status

            print_discovery_status(facility, use_rich=use_rich, domain=domain)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@discover.command("inspect")
@click.argument("facility")
@click.option(
    "--scanned", "-s", default=5, type=int, help="Number of scanned paths to show"
)
@click.option(
    "--scored", "-r", default=5, type=int, help="Number of scored paths to show"
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def discover_inspect(facility: str, scanned: int, scored: int, as_json: bool) -> None:
    """Inspect scanned and scored paths from the graph."""
    import json

    from imas_codex.graph import GraphClient

    console = Console()

    try:
        with GraphClient() as gc:
            scanned_paths = gc.query(
                """
                MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
                WHERE p.status = 'scanned'
                RETURN p.path AS path, p.total_files AS total_files,
                       p.total_dirs AS total_dirs, p.has_readme AS has_readme,
                       p.has_makefile AS has_makefile, p.has_git AS has_git,
                       p.depth AS depth, p.scanned_at AS scanned_at
                ORDER BY p.scanned_at DESC
                LIMIT $limit
                """,
                facility=facility,
                limit=scanned,
            )

            scored_paths = gc.query(
                """
                MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
                WHERE p.status = 'scored' AND p.score IS NOT NULL
                RETURN p.path AS path, p.score AS score,
                       p.score_modeling_code AS score_modeling_code,
                       p.score_analysis_code AS score_analysis_code,
                       p.score_imas AS score_imas, p.path_purpose AS path_purpose,
                       p.description AS description, p.total_files AS total_files,
                       p.scored_at AS scored_at
                ORDER BY p.score DESC
                LIMIT $limit
                """,
                facility=facility,
                limit=scored,
            )

        if as_json:
            output = {
                "facility": facility,
                "scanned_paths": list(scanned_paths),
                "scored_paths": list(scored_paths),
            }
            console.print_json(json.dumps(output, default=str))
            return

        console.print(f"\n[bold cyan]Scanned Paths ({len(scanned_paths)})[/bold cyan]")
        if scanned_paths:
            scan_table = Table(show_header=True, header_style="bold")
            scan_table.add_column("Path", style="cyan", no_wrap=True, max_width=40)
            scan_table.add_column("Files", justify="right")
            scan_table.add_column("Dirs", justify="right")
            scan_table.add_column("README", justify="center")
            scan_table.add_column("Makefile", justify="center")
            scan_table.add_column("Git", justify="center")
            scan_table.add_column("Depth", justify="right")

            for p in scanned_paths:
                path_display = p["path"]
                if len(path_display) > 40:
                    path_display = "..." + path_display[-37:]
                scan_table.add_row(
                    path_display,
                    str(p.get("total_files", 0) or 0),
                    str(p.get("total_dirs", 0) or 0),
                    "✓" if p.get("has_readme") else "",
                    "✓" if p.get("has_makefile") else "",
                    "✓" if p.get("has_git") else "",
                    str(p.get("depth", 0) or 0),
                )
            console.print(scan_table)
        else:
            console.print("  (no scanned paths found)")

        console.print(f"\n[bold green]Scored Paths ({len(scored_paths)})[/bold green]")
        if scored_paths:
            score_table = Table(show_header=True, header_style="bold")
            score_table.add_column("Path", style="cyan", no_wrap=True, max_width=35)
            score_table.add_column("Score", justify="right", style="bold")
            score_table.add_column("Model", justify="right")
            score_table.add_column("Anlys", justify="right")
            score_table.add_column("IMAS", justify="right")
            score_table.add_column("Purpose", max_width=15)
            score_table.add_column("Description", max_width=30)

            for p in scored_paths:
                path_display = p["path"]
                if len(path_display) > 35:
                    path_display = "..." + path_display[-32:]

                score_val = p.get("score", 0) or 0
                if score_val >= 0.7:
                    score_str = f"[green]{score_val:.2f}[/green]"
                elif score_val >= 0.4:
                    score_str = f"[yellow]{score_val:.2f}[/yellow]"
                else:
                    score_str = f"[red]{score_val:.2f}[/red]"

                desc = p.get("description", "") or ""
                if len(desc) > 30:
                    desc = desc[:27] + "..."

                score_table.add_row(
                    path_display,
                    score_str,
                    f"{p.get('score_modeling_code', 0) or 0:.2f}",
                    f"{p.get('score_analysis_code', 0) or 0:.2f}",
                    f"{p.get('score_imas', 0) or 0:.2f}",
                    p.get("path_purpose", "") or "",
                    desc,
                )
            console.print(score_table)
        else:
            console.print("  (no scored paths found)")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@discover.command("clear")
@click.argument("facility")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--domain",
    "-d",
    type=click.Choice(["paths", "wiki", "signals"]),
    help="Clear specific domain only (default: all)",
)
def discover_clear(facility: str, force: bool, domain: str | None) -> None:
    """Clear discovered data for a facility.

    By default clears ALL domains: paths, wiki, and signals.
    Use --domain/-d to clear a specific domain only.

    \b
    Examples:
      imas-codex discover clear jet           # All domains
      imas-codex discover clear jet -d paths  # Paths only
      imas-codex discover clear jet -d wiki   # Wiki only
      imas-codex discover clear jet --force   # Skip confirmation
    """
    _clear_facility_domain(facility, domain, force)


@paths.command("clear")
@click.argument("facility")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
def paths_clear(facility: str, force: bool) -> None:
    """Clear path discovery data for a facility.

    Deletes FacilityPath, SourceFile, and FacilityUser nodes.

    \b
    Examples:
      imas-codex discover paths clear jet
      imas-codex discover paths clear tcv --force
    """
    _clear_facility_domain(facility, "paths", force)


@paths.command("status")
@click.argument("facility")
def paths_status(facility: str) -> None:
    """Show paths discovery statistics for a facility.

    \b
    Examples:
      imas-codex discover paths status jet
      imas-codex discover paths status tcv
    """
    from imas_codex.discovery import get_discovery_stats
    from imas_codex.discovery.paths.progress import print_discovery_status

    stats = get_discovery_stats(facility)
    if stats.get("total", 0) == 0:
        click.echo(f"No paths discovered for {facility}")
        return

    print_discovery_status(facility, domain="paths")


# =============================================================================
# Wiki Clear & Status
# =============================================================================


@wiki.command("clear")
@click.argument("facility")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
def wiki_clear(facility: str, force: bool) -> None:
    """Clear wiki discovery data for a facility.

    Deletes WikiPage, WikiChunk, WikiArtifact nodes.

    \b
    Examples:
      imas-codex discover wiki clear jet
      imas-codex discover wiki clear tcv --force
    """
    _clear_facility_domain(facility, "wiki", force)


@wiki.command("status")
@click.argument("facility")
def wiki_status(facility: str) -> None:
    """Show wiki discovery statistics for a facility.

    \b
    Examples:
      imas-codex discover wiki status jet
      imas-codex discover wiki status tcv
    """
    from imas_codex.discovery.paths.progress import print_discovery_status

    print_discovery_status(facility, domain="wiki")


# =============================================================================
# Signals Clear & Status
# =============================================================================


@signals.command("clear")
@click.argument("facility")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
def signals_clear(facility: str, force: bool) -> None:
    """Clear signal discovery data for a facility.

    Deletes FacilitySignal, DataAccess, TreeModelVersion nodes,
    and epoch checkpoint files.

    \b
    Examples:
      imas-codex discover signals clear jet
      imas-codex discover signals clear tcv --force
    """
    _clear_facility_domain(facility, "signals", force)


@signals.command("status")
@click.argument("facility")
def signals_status(facility: str) -> None:
    """Show signal discovery statistics for a facility.

    \b
    Examples:
      imas-codex discover signals status jet
      imas-codex discover signals status tcv
    """
    from imas_codex.discovery.paths.progress import print_discovery_status

    print_discovery_status(facility, domain="signals")


# =============================================================================
# Shared Clear Function
# =============================================================================


def _clear_facility_domain(
    facility: str, domain: str | None, force: bool = False
) -> None:
    """Clear discovery data for facility, optionally filtered by domain.

    Args:
        facility: Facility ID
        domain: "paths", "wiki", "signals", or None for all
        force: Skip confirmation prompt
    """
    from imas_codex.discovery import clear_facility_paths, get_discovery_stats
    from imas_codex.discovery.data import (
        clear_facility_signals,
        get_data_discovery_stats,
    )
    from imas_codex.discovery.wiki import clear_facility_wiki, get_wiki_stats

    try:
        items_to_clear: list[tuple[str, int, callable]] = []

        # Paths domain
        if domain is None or domain == "paths":
            stats = get_discovery_stats(facility)
            total = stats.get("total", 0)
            if total > 0:
                items_to_clear.append(("paths + related", total, clear_facility_paths))

        # Wiki domain
        if domain is None or domain == "wiki":
            wiki_stats = get_wiki_stats(facility)
            pages = wiki_stats.get("pages", 0)
            chunks = wiki_stats.get("chunks", 0)
            from imas_codex.graph import GraphClient

            with GraphClient() as gc:
                artifact_result = gc.query(
                    "MATCH (wa:WikiArtifact {facility_id: $f}) RETURN count(wa) AS cnt",
                    f=facility,
                )
                artifacts = artifact_result[0]["cnt"] if artifact_result else 0
            if pages > 0 or artifacts > 0:
                label = f"wiki pages + {chunks} chunks + {artifacts} artifacts"
                items_to_clear.append((label, pages or artifacts, clear_facility_wiki))

        # Signals domain
        if domain is None or domain == "signals":
            signal_stats = get_data_discovery_stats(facility)
            signal_total = signal_stats.get("total", 0)
            if signal_total > 0:
                items_to_clear.append(
                    ("signals + epochs", signal_total, clear_facility_signals)
                )

        if not items_to_clear:
            domain_msg = f" {domain}" if domain else ""
            click.echo(f"No{domain_msg} discovery data to clear for {facility}")
            return

        summary_parts = [f"{count} {name}" for name, count, _ in items_to_clear]
        summary = " and ".join(summary_parts)

        if not force:
            click.confirm(
                f"This will delete {summary} for {facility}. Continue?",
                abort=True,
            )

        for name, _, clear_func in items_to_clear:
            result = clear_func(facility)
            _print_clear_result(name, result, facility)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


def _print_clear_result(name: str, result: dict | int, facility: str) -> None:
    """Format and print clear operation result."""
    if isinstance(result, dict):
        parts = []
        for key in (
            "pages_deleted",
            "chunks_deleted",
            "artifacts_deleted",
            "signals_deleted",
            "data_access_deleted",
            "epochs_deleted",
            "checkpoints_deleted",
            "paths_deleted",
            "source_files_deleted",
            "users_deleted",
        ):
            if result.get(key):
                label = key.replace("_deleted", "").replace("_", " ")
                parts.append(f"{result[key]} {label}")
        click.echo(f"✓ Deleted {', '.join(parts)} for {facility}")
    else:
        click.echo(f"✓ Deleted {result} {name} for {facility}")


@discover.command("seed")
@click.argument("facility")
@click.option("--path", "-p", multiple=True, help="Additional root paths to seed")
def discover_seed(facility: str, path: tuple[str, ...]) -> None:
    """Seed facility root paths without scanning."""
    from imas_codex.discovery import seed_facility_roots

    try:
        additional_paths = list(path) if path else None
        created = seed_facility_roots(facility, additional_paths)
        click.echo(f"✓ Created {created} root path(s) for {facility}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


# Placeholder Discovery Commands


@discover.command("code")
@click.argument("facility")
@click.option("--dry-run", is_flag=True, help="Show what would be discovered")
def discover_code(facility: str, dry_run: bool) -> None:
    """Discover source files in scored paths.

    NOT YET IMPLEMENTED. Use 'imas-codex ingest queue' to manually
    queue source files for ingestion.
    """
    from imas_codex.discovery import get_discovery_stats

    console = Console()
    stats = get_discovery_stats(facility)

    if stats.get("scored", 0) == 0:
        console.print(
            f"[yellow]⚠ No scored paths found for {facility.upper()}[/yellow]\n"
        )
        console.print("Discovery pipeline:")
        console.print(
            "  1. [bold]discover paths[/bold] → 2. discover code → 3. ingest code"
        )
        console.print(f"\nNext step: [cyan]imas-codex discover paths {facility}[/cyan]")
        raise SystemExit(1)

    console.print("[yellow]discover code: Not yet implemented[/yellow]")
    console.print(f"\nCurrent paths status for {facility.upper()}:")
    console.print(f"  Scored paths: {stats.get('scored', 0)}")
    console.print(f"  High-value (≥0.7): {stats.get('high_value', 'unknown')}")
    console.print("\nThis feature will scan scored paths for source files.")
    raise SystemExit(1)


@wiki.command("run", hidden=True)
@click.argument("facility")
@click.option("--source", "-s", help="Specific wiki site URL or index")
@click.option(
    "--cost-limit", "-c", type=float, default=10.0, help="Maximum LLM spend in USD"
)
@click.option(
    "--max-pages", "-n", type=int, default=None, help="Maximum pages to process"
)
@click.option(
    "--max-depth", type=int, default=None, help="Maximum link depth from portal"
)
@click.option(
    "--focus", "-f", help="Focus discovery (e.g., 'equilibrium', 'diagnostics')"
)
@click.option(
    "--scan-only", is_flag=True, help="Only scan pages, skip scoring and ingestion"
)
@click.option(
    "--score-only",
    is_flag=True,
    help="Only score already-discovered pages, skip ingestion",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
@click.option(
    "--no-rich",
    is_flag=True,
    default=False,
    help="Use logging output instead of rich progress display",
)
@click.option(
    "--rescan",
    is_flag=True,
    default=False,
    help="Re-scan pages even if already in graph",
)
@click.option(
    "--score-workers",
    type=int,
    default=3,
    help="Number of parallel score workers (default: 3)",
)
@click.option(
    "--ingest-workers",
    type=int,
    default=4,
    help="Number of parallel ingest workers (default: 4)",
)
@click.option(
    "--rescan-artifacts",
    is_flag=True,
    default=False,
    help="Re-scan artifacts even if already in graph",
)
def wiki_run(
    facility: str,
    source: str | None,
    cost_limit: float,
    max_pages: int | None,
    max_depth: int | None,
    focus: str | None,
    scan_only: bool,
    score_only: bool,
    verbose: bool,
    no_rich: bool,
    rescan: bool,
    score_workers: int,
    ingest_workers: int,
    rescan_artifacts: bool,
) -> None:
    """Discover wiki pages and build documentation graph.

    Runs parallel wiki discovery workers:

    \b
    - SCAN: Enumerate all pages per site (runs once, cached in graph)
    - SCORE: LLM relevance evaluation with content fetch
    - INGEST: Chunk and embed high-score pages
    - ARTIFACTS: Score and embed wiki attachments (PDFs, images, etc.)

    Page scanning runs automatically on first invocation. Use --rescan
    to re-enumerate pages (adds new pages, keeps existing).
    """
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.discovery.wiki import get_wiki_stats
    from imas_codex.discovery.wiki.parallel import (
        reset_transient_pages,
        run_parallel_wiki_discovery,
    )

    # Auto-detect if rich can run (TTY check) or use no_rich flag
    use_rich = not no_rich and sys.stdout.isatty()

    if use_rich:
        console = Console()
    else:
        console = None
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )

    wiki_logger = logging.getLogger("imas_codex.discovery.wiki")
    if not use_rich:
        wiki_logger.setLevel(logging.INFO)

    def log_print(msg: str, style: str = "") -> None:
        """Print to console or log, stripping rich markup."""
        clean_msg = re.sub(r"\[[^\]]+\]", "", msg)
        if console:
            console.print(msg)
        else:
            wiki_logger.info(clean_msg)

    try:
        config = get_facility(facility)
        wiki_sites = config.get("wiki_sites", [])
    except Exception as e:
        log_print(f"[red]Error loading facility config: {e}[/red]")
        raise SystemExit(1) from e

    if not wiki_sites:
        log_print(
            f"[yellow]No wiki sites configured for {facility}.[/yellow]\n"
            "Configure wiki sites in the facility YAML under 'wiki_sites:'"
        )
        raise SystemExit(1)

    # Check embedding server availability upfront for ingestion mode.
    # Uses centralized readiness check which handles the full lifecycle:
    # SSH tunnel (if off-ITER), systemd service start, health check.
    if not scan_only and not score_only:
        from imas_codex.embeddings.config import EmbeddingBackend
        from imas_codex.settings import get_embedding_backend

        backend_str = get_embedding_backend()
        try:
            backend = EmbeddingBackend(backend_str)
        except ValueError:
            backend = EmbeddingBackend.LOCAL

        if backend == EmbeddingBackend.REMOTE:
            from rich.markup import escape as rich_escape

            from imas_codex.embeddings.readiness import ensure_embedding_ready

            _style_map = {
                "info": ("", ""),
                "dim": ("[dim]", "[/dim]"),
                "warning": ("[yellow]", "[/yellow]"),
                "success": ("[green]", "[/green]"),
                "error": ("[red]", "[/red]"),
            }

            def _readiness_log(msg: str, style: str = "info") -> None:
                prefix, suffix = _style_map.get(style, ("", ""))
                log_print(f"{prefix}{rich_escape(msg)}{suffix}")

            ok, msg = ensure_embedding_ready(log_fn=_readiness_log, timeout=60.0)
            if not ok:
                log_print(f"[red]{rich_escape(msg)}[/red]")
                log_print(
                    "[dim]Or use --scan-only / --score-only to skip embedding[/dim]"
                )
                raise SystemExit(1)

    # Display wiki sites
    multi_site_table = use_rich and len(wiki_sites) > 3
    if multi_site_table:
        from urllib.parse import urlparse as _parse_url

        site_table = Table(
            show_header=True,
            header_style="bold",
            border_style="dim",
            padding=(0, 1),
        )
        site_table.add_column("Site", style="cyan", no_wrap=True)
        site_table.add_column("Type", style="dim")
        site_table.add_column("Description", style="white")

        for site in wiki_sites:
            parsed = _parse_url(site.get("url", ""))
            name = parsed.path.rstrip("/").rsplit("/", 1)[-1] or site.get("url", "")
            site_type_str = site.get("site_type", "mediawiki")
            desc = site.get("description", "")
            site_table.add_row(name, site_type_str, desc)

        console.print(site_table)

        # Show auth info once (aggregated across all sites)
        auth_types = {s.get("auth_type") for s in wiki_sites if s.get("auth_type")}
        cred_services = {
            s.get("credential_service")
            for s in wiki_sites
            if s.get("credential_service")
        }
        if auth_types and cred_services:
            if "basic" in auth_types:
                auth_label = f"HTTP Basic ({', '.join(sorted(cred_services))})"
            elif "tequila" in auth_types:
                auth_label = "Tequila"
            else:
                auth_label = ", ".join(sorted(auth_types))
            console.print(f"[dim]Auth: {auth_label}[/dim]")
        console.print()
    else:
        log_print(f"[bold]Documentation sources for {facility}:[/bold]")
        if len(wiki_sites) > 3 and not verbose:
            from urllib.parse import urlparse as _parse_url

            names = []
            for site in wiki_sites:
                parsed = _parse_url(site.get("url", ""))
                name = parsed.path.rstrip("/").rsplit("/", 1)[-1] or site.get("url", "")
                names.append(name)
            log_print(f"  {len(wiki_sites)} sites: {', '.join(names)}")
        else:
            for i, site in enumerate(wiki_sites):
                site_type_str = site.get("site_type", "mediawiki")
                url = site.get("url", "")
                desc = site.get("description", "")
                log_print(f"  [{i}] {site_type_str}: {url}")
                if desc and verbose:
                    log_print(f"      {desc}")
        if use_rich:
            console.print()

    site_indices = list(range(len(wiki_sites)))
    if source:
        try:
            idx = int(source)
            if 0 <= idx < len(wiki_sites):
                site_indices = [idx]
            else:
                log_print(f"[red]Invalid site index: {idx}[/red]")
                raise SystemExit(1)
        except ValueError:
            matched = [
                i
                for i, s in enumerate(wiki_sites)
                if source.lower() in s.get("url", "").lower()
            ]
            if matched:
                site_indices = matched
            else:
                log_print(f"[red]No site matching '{source}'[/red]")
                raise SystemExit(1) from None

    # ================================================================
    # Phase 1: Preflight - validate credentials, bulk discovery
    # ================================================================

    # Check wiki stats once (facility-level, not per-site)
    wiki_stats = get_wiki_stats(facility)
    existing_pages = wiki_stats.get("pages", 0) or wiki_stats.get("total", 0)
    should_bulk_discover = rescan or existing_pages == 0

    if existing_pages > 0 and not should_bulk_discover:
        log_print(
            f"[dim]Found {existing_pages} existing wiki pages, skipping scan[/dim]"
        )
        log_print("[dim]Use --rescan to re-enumerate pages[/dim]")
    elif rescan and existing_pages > 0:
        log_print(
            f"[yellow]Rescan: adding new pages (keeping {existing_pages} existing)[/yellow]"
        )

    # Reset orphaned pages once (facility-level)
    reset_counts = reset_transient_pages(facility, silent=True)
    if any(reset_counts.values()):
        total_reset = sum(reset_counts.values())
        log_print(f"[dim]Reset {total_reset} orphaned pages from previous run[/dim]")

    # Validate all sites and run bulk discovery for each
    site_configs: list[dict] = []
    validated_cred_services: set[str] = set()

    for site_idx in site_indices:
        site = wiki_sites[site_idx]
        site_type = site.get("site_type", "mediawiki")
        base_url = site.get("url", "")
        portal_page = site.get("portal_page", "Main_Page")
        auth_type = site.get("auth_type")
        credential_service = site.get("credential_service")

        # Determine SSH host
        ssh_host = site.get("ssh_host")
        if not ssh_host and site.get("ssh_available", False):
            ssh_host = config.get("ssh_host")

        # Short name for multi-site display (e.g. "pog" from URL path)
        from urllib.parse import urlparse as _urlparse

        parsed_url = _urlparse(base_url)
        short_name = parsed_url.path.rstrip("/").rsplit("/", 1)[-1] or base_url

        # Site header - skip if sites table was already displayed
        if not multi_site_table:
            if len(site_indices) > 1:
                site_n = site_indices.index(site_idx) + 1
                log_print(
                    f"[bold cyan]({site_n}/{len(site_indices)}) {short_name}[/bold cyan]"
                )
            else:
                log_print(f"\n[bold cyan]Processing: {base_url}[/bold cyan]")

            if site_type == "twiki":
                log_print("[cyan]TWiki: using HTTP scanner via SSH[/cyan]")
            elif auth_type in ("tequila", "session"):
                log_print("[cyan]Using Tequila authentication[/cyan]")
            elif auth_type == "basic" and credential_service:
                log_print(
                    f"[cyan]Using HTTP Basic authentication ({credential_service})[/cyan]"
                )
            elif ssh_host:
                # Check if SOCKS tunnel is available (faster path)
                from imas_codex.discovery.wiki.adapters import _ensure_socks_tunnel

                if _ensure_socks_tunnel():
                    log_print("[cyan]Using SOCKS proxy via laptop[/cyan]")
                else:
                    log_print(f"[cyan]Using SSH proxy via {ssh_host}[/cyan]")

        # Validate credentials once per credential_service
        if auth_type in ("tequila", "session", "basic") and credential_service:
            if credential_service not in validated_cred_services:
                from imas_codex.discovery.wiki.auth import CredentialManager

                cred_mgr = CredentialManager()
                creds = cred_mgr.get_credentials(
                    credential_service, prompt_if_missing=True
                )
                if creds is None:
                    log_print(
                        f"[red]Credentials required for {credential_service}.[/red]\n"
                        f"Set them with: imas-codex credentials set {credential_service}\n"
                        f"Or set environment variables:\n"
                        f"  export {cred_mgr._env_var_name(credential_service, 'username')}=your_username\n"
                        f"  export {cred_mgr._env_var_name(credential_service, 'password')}=your_password"
                    )
                    raise SystemExit(1)
                log_print(f"[dim]Authenticated as: {creds[0]}[/dim]")
                validated_cred_services.add(credential_service)

        # Bulk page discovery
        bulk_discovered = 0
        if should_bulk_discover and not score_only:
            if site_type == "mediawiki":
                from imas_codex.discovery.wiki.parallel import (
                    bulk_discover_all_pages_basic_auth,
                    bulk_discover_all_pages_http,
                    bulk_discover_all_pages_keycloak,
                    bulk_discover_all_pages_mediawiki,
                )

                def bulk_progress_log(msg, _):
                    wiki_logger.info(f"BULK: {msg}")

                if use_rich:
                    from rich.status import Status

                    with Status(
                        f"[cyan]Bulk discovery: {short_name}...[/cyan]",
                        console=console,
                        spinner="dots",
                    ) as status:

                        def bulk_progress_rich(msg, _, _sn=short_name):
                            if "pages" in msg:
                                status.update(f"[cyan]{_sn}: {msg}[/cyan]")
                            elif "creating" in msg:
                                status.update(f"[cyan]{_sn}: {msg}[/cyan]")
                            elif "created" in msg:
                                status.update(f"[green]{_sn}: {msg}[/green]")

                        if auth_type == "tequila" and credential_service:
                            bulk_discovered = bulk_discover_all_pages_http(
                                facility,
                                base_url,
                                credential_service,
                                bulk_progress_rich,
                            )
                        elif auth_type == "basic" and credential_service:
                            bulk_discovered = bulk_discover_all_pages_basic_auth(
                                facility,
                                base_url,
                                credential_service,
                                bulk_progress_rich,
                            )
                        elif auth_type == "keycloak" and credential_service:
                            bulk_discovered = bulk_discover_all_pages_keycloak(
                                facility,
                                base_url,
                                credential_service,
                                bulk_progress_rich,
                            )
                        elif ssh_host:
                            bulk_discovered = bulk_discover_all_pages_mediawiki(
                                facility, base_url, ssh_host, bulk_progress_rich
                            )

                    if bulk_discovered > 0:
                        log_print(
                            f"[green]Discovered {bulk_discovered:,} pages[/green]"
                        )
                else:
                    if auth_type == "tequila" and credential_service:
                        bulk_discovered = bulk_discover_all_pages_http(
                            facility,
                            base_url,
                            credential_service,
                            bulk_progress_log,
                        )
                    elif auth_type == "basic" and credential_service:
                        bulk_discovered = bulk_discover_all_pages_basic_auth(
                            facility,
                            base_url,
                            credential_service,
                            bulk_progress_log,
                        )
                    elif auth_type == "keycloak" and credential_service:
                        bulk_discovered = bulk_discover_all_pages_keycloak(
                            facility,
                            base_url,
                            credential_service,
                            bulk_progress_log,
                        )
                    elif ssh_host:
                        bulk_discovered = bulk_discover_all_pages_mediawiki(
                            facility, base_url, ssh_host, bulk_progress_log
                        )
                    if bulk_discovered > 0:
                        wiki_logger.info(
                            f"Discovered {bulk_discovered} pages from {short_name}"
                        )

            elif site_type in ("twiki_static", "static_html"):
                from imas_codex.discovery.wiki.parallel import (
                    bulk_discover_all_pages_static_html,
                    bulk_discover_all_pages_twiki_static,
                )

                def twiki_progress_log(msg, _):
                    wiki_logger.info(f"BULK: {msg}")

                if site_type == "twiki_static":
                    discover_func = bulk_discover_all_pages_twiki_static
                    discover_args = (facility, base_url, ssh_host)
                    label = "TWiki"
                else:
                    from imas_codex.discovery.wiki.parallel import (
                        _get_exclude_prefixes,
                    )

                    exclude_prefixes = _get_exclude_prefixes(facility, base_url)
                    discover_func = bulk_discover_all_pages_static_html
                    discover_args = (facility, base_url, exclude_prefixes, ssh_host)
                    label = "Static HTML"

                if use_rich:
                    from rich.status import Status

                    with Status(
                        f"[cyan]{label} discovery: {short_name}...[/cyan]",
                        console=console,
                        spinner="dots",
                    ) as status:

                        def twiki_progress_rich(msg, _, _sn=short_name):
                            status.update(f"[cyan]{_sn}: {msg}[/cyan]")

                        bulk_discovered = discover_func(
                            *discover_args, twiki_progress_rich
                        )

                    if bulk_discovered > 0:
                        log_print(
                            f"[green]Discovered {bulk_discovered:,} pages[/green]"
                        )
                else:
                    bulk_discovered = discover_func(*discover_args, twiki_progress_log)
                    if bulk_discovered > 0:
                        wiki_logger.info(
                            f"Discovered {bulk_discovered} pages from {short_name}"
                        )

        # Artifact scanning
        should_discover_artifacts_site = (
            rescan_artifacts or (bulk_discovered > 0)
        ) and not score_only
        if should_discover_artifacts_site and site_type == "mediawiki":
            from imas_codex.discovery.wiki.parallel import bulk_discover_artifacts

            def artifact_progress_log(msg, _):
                wiki_logger.info(f"ARTIFACTS: {msg}")

            wiki_client = None
            if auth_type == "tequila" and credential_service:
                from imas_codex.discovery.wiki.mediawiki import MediaWikiClient

                wiki_client = MediaWikiClient(
                    base_url=base_url,
                    credential_service=credential_service,
                    verify_ssl=False,
                )
                wiki_client.authenticate()

            if use_rich:
                from rich.status import Status

                with Status(
                    f"[cyan]Artifact discovery: {short_name}...[/cyan]",
                    console=console,
                    spinner="dots",
                ) as status:

                    def artifact_progress_rich(msg, _, _sn=short_name):
                        if "batch" in msg:
                            status.update(f"[cyan]{_sn} artifacts: {msg}[/cyan]")
                        elif "created" in msg:
                            status.update(f"[green]{_sn} artifacts: {msg}[/green]")

                    artifacts_discovered = bulk_discover_artifacts(
                        facility=facility,
                        base_url=base_url,
                        site_type=site_type,
                        ssh_host=ssh_host,
                        wiki_client=wiki_client,
                        credential_service=credential_service,
                        on_progress=artifact_progress_rich,
                    )
            else:
                artifacts_discovered = bulk_discover_artifacts(
                    facility=facility,
                    base_url=base_url,
                    site_type=site_type,
                    ssh_host=ssh_host,
                    wiki_client=wiki_client,
                    credential_service=credential_service,
                    on_progress=artifact_progress_log,
                )

            if wiki_client:
                wiki_client.close()

            if artifacts_discovered > 0:
                log_print(
                    f"[green]Discovered {artifacts_discovered:,} artifacts[/green]"
                )

        site_configs.append(
            {
                "site_type": site_type,
                "base_url": base_url,
                "portal_page": portal_page,
                "ssh_host": ssh_host,
                "auth_type": auth_type,
                "credential_service": credential_service,
                "short_name": short_name,
            }
        )

    if not site_configs:
        log_print("[yellow]No sites to process[/yellow]")
        raise SystemExit(1)

    # ================================================================
    # Phase 2: Score/ingest all sites with unified progress display
    # ================================================================

    worker_parts = []
    if not scan_only:
        worker_parts.append(f"{score_workers} score")
        worker_parts.append(f"{ingest_workers} ingest")
        worker_parts.append("2 artifact")
    log_print(f"\nWorkers: {', '.join(worker_parts)}")
    if not scan_only:
        log_print(f"Cost limit: ${cost_limit:.2f}")
    if max_pages:
        log_print(f"Page limit: {max_pages}")
    if focus and not scan_only:
        log_print(f"Focus: {focus}")
    if len(site_configs) > 1:
        log_print(f"Sites to process: {len(site_configs)}")

    try:

        async def run_all_sites_unified(
            _facility: str,
            _site_configs: list[dict],
            _cost_limit: float,
            _max_pages: int | None,
            _max_depth: int | None,
            _focus: str | None,
            _scan_only: bool,
            _score_only: bool,
            _use_rich: bool,
            _num_score_workers: int,
            _num_ingest_workers: int,
        ):
            combined: dict = {
                "scanned": 0,
                "scored": 0,
                "ingested": 0,
                "artifacts": 0,
                "cost": 0.0,
                "elapsed_seconds": 0.0,
            }
            remaining_budget = _cost_limit
            remaining_pages = _max_pages

            # Logging-only callbacks for non-rich mode
            def log_on_scan(msg, stats, results=None):
                if msg != "idle":
                    wiki_logger.info(f"SCAN: {msg}")

            def log_on_score(msg, stats, results=None):
                if msg != "idle":
                    wiki_logger.info(f"SCORE: {msg}")

            def log_on_ingest(msg, stats, results=None):
                if msg != "idle":
                    wiki_logger.info(f"INGEST: {msg}")

            if not _use_rich:
                _multi = len(_site_configs) > 1
                for i, sc in enumerate(_site_configs):
                    if remaining_budget <= 0:
                        wiki_logger.info("Budget exhausted, skipping remaining sites")
                        break
                    if remaining_pages is not None and remaining_pages <= 0:
                        wiki_logger.info("Page limit reached, skipping remaining sites")
                        break

                    wiki_logger.info(
                        "Processing site %d/%d: %s",
                        i + 1,
                        len(_site_configs),
                        sc["base_url"],
                    )

                    try:
                        result = await run_parallel_wiki_discovery(
                            facility=_facility,
                            site_type=sc["site_type"],
                            base_url=sc["base_url"],
                            portal_page=sc["portal_page"],
                            ssh_host=sc["ssh_host"],
                            auth_type=sc["auth_type"],
                            credential_service=sc["credential_service"],
                            cost_limit=remaining_budget,
                            page_limit=remaining_pages,
                            max_depth=_max_depth,
                            focus=_focus,
                            num_scan_workers=1,
                            num_score_workers=_num_score_workers,
                            num_ingest_workers=_num_ingest_workers,
                            scan_only=_scan_only,
                            score_only=_score_only,
                            bulk_discover=False,
                            skip_reset=_multi,
                            on_scan_progress=log_on_scan,
                            on_score_progress=log_on_score,
                            on_ingest_progress=log_on_ingest,
                        )
                    except Exception as e:
                        wiki_logger.warning("Site %s failed: %s", sc["base_url"], e)
                        continue

                    for key in ("scanned", "scored", "ingested", "artifacts"):
                        combined[key] += result.get(key, 0)
                    combined["cost"] += result.get("cost", 0)
                    combined["elapsed_seconds"] += result.get("elapsed_seconds", 0)

                    remaining_budget -= result.get("cost", 0)
                    if remaining_pages is not None:
                        remaining_pages -= result.get("scored", 0)

                return combined

            # Rich mode: single unified display across all sites
            from imas_codex.discovery.wiki.progress import WikiProgressDisplay

            multi_site = len(_site_configs) > 1

            # Suppress noisy INFO logs from embedding modules during rich display
            # (source tracking, model init etc. are shown in the progress panel)
            for mod in ("imas_codex.embeddings",):
                logging.getLogger(mod).setLevel(logging.WARNING)

            with WikiProgressDisplay(
                facility=_facility,
                cost_limit=_cost_limit,
                page_limit=_max_pages,
                focus=_focus or "",
                console=console,
                scan_only=_scan_only,
                score_only=_score_only,
            ) as display:
                # Set multi-site info on first iteration
                if multi_site:
                    display.set_site_info(
                        site_name=_site_configs[0]["short_name"],
                        site_index=0,
                        total_sites=len(_site_configs),
                    )

                # Periodic graph refresh (facility-level, works across sites)
                async def refresh_graph_state():
                    from imas_codex.discovery.wiki.parallel import (
                        get_wiki_discovery_stats,
                    )

                    while True:
                        try:
                            stats = get_wiki_discovery_stats(_facility)
                            display.update_from_graph(
                                total_pages=stats.get("total", 0),
                                pages_scanned=stats.get("scanned", 0),
                                pages_scored=stats.get("scored", 0),
                                pages_ingested=stats.get("ingested", 0),
                                pages_skipped=stats.get("skipped", 0),
                                pending_score=stats.get(
                                    "pending_score",
                                    stats.get("scanned", 0),
                                ),
                                pending_ingest=stats.get("pending_ingest", 0),
                                pending_artifact_score=stats.get(
                                    "pending_artifact_score", 0
                                ),
                                pending_artifact_ingest=stats.get(
                                    "pending_artifact_ingest", 0
                                ),
                                accumulated_cost=stats.get("accumulated_cost", 0.0),
                                artifacts_ingested=stats.get("artifacts_ingested", 0),
                                artifacts_scored=stats.get("artifacts_scored", 0),
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            wiki_logger.debug("Graph refresh failed: %s", e)
                        await asyncio.sleep(0.5)

                async def queue_ticker():
                    while True:
                        try:
                            display.tick()
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            wiki_logger.debug("Display tick failed: %s", e)
                        await asyncio.sleep(0.15)

                refresh_task = asyncio.create_task(refresh_graph_state())
                ticker_task = asyncio.create_task(queue_ticker())

                def on_scan(msg, stats, results=None):
                    result_dicts = None
                    if results:
                        result_dicts = [
                            {
                                "title": r.get("id", "?").split(":")[-1][:50],
                                "out_links": r.get("out_degree", 0),
                                "depth": r.get("depth", 0),
                            }
                            for r in results[:5]
                        ]
                    display.update_scan(msg, stats, result_dicts)

                def on_score(msg, stats, results=None):
                    result_dicts = None
                    if results:
                        result_dicts = [
                            {
                                "title": r.get("id", "?").split(":")[-1][:60],
                                "score": r.get("score"),
                                "physics_domain": r.get("physics_domain"),
                                "description": r.get("description", ""),
                                "is_physics": r.get("is_physics", False),
                                "skipped": r.get("skipped", False),
                                "skip_reason": r.get("skip_reason", ""),
                            }
                            for r in results[:5]
                        ]
                    display.update_score(msg, stats, result_dicts)

                def on_ingest(msg, stats, results=None):
                    result_dicts = None
                    if results:
                        result_dicts = [
                            {
                                "title": r.get("id", "?").split(":")[-1][:60],
                                "score": r.get("score"),
                                "description": r.get("description", ""),
                                "physics_domain": r.get("physics_domain"),
                                "chunk_count": r.get("chunk_count", 0),
                            }
                            for r in results[:5]
                        ]
                    display.update_ingest(msg, stats, result_dicts)

                def on_artifact(msg, stats, results=None):
                    result_dicts = None
                    if results:
                        result_dicts = [
                            {
                                "filename": r.get("filename", "unknown"),
                                "artifact_type": r.get("artifact_type", ""),
                                "score": r.get("score"),
                                "physics_domain": r.get("physics_domain"),
                                "description": r.get("description", ""),
                                "chunk_count": r.get("chunk_count", 0),
                            }
                            for r in results[:5]
                        ]
                    display.update_artifact(msg, stats, result_dicts)

                def on_artifact_score(msg, stats, results=None):
                    result_dicts = None
                    if results:
                        result_dicts = [
                            {
                                "filename": r.get("filename", "unknown"),
                                "artifact_type": r.get("artifact_type", ""),
                                "score": r.get("score"),
                                "physics_domain": r.get("physics_domain"),
                                "description": r.get("description", ""),
                            }
                            for r in results[:5]
                        ]
                    display.update_artifact_score(msg, stats, result_dicts)

                def on_worker_status(worker_group):
                    display.update_worker_status(worker_group)

                try:
                    for i, sc in enumerate(_site_configs):
                        # Advance display to next site (skip first)
                        if i > 0 and multi_site:
                            display.advance_site(sc["short_name"], i)

                        # Stop if budget or page limit exhausted
                        if remaining_budget <= 0:
                            break
                        if remaining_pages is not None and remaining_pages <= 0:
                            break

                        try:
                            result = await run_parallel_wiki_discovery(
                                facility=_facility,
                                site_type=sc["site_type"],
                                base_url=sc["base_url"],
                                portal_page=sc["portal_page"],
                                ssh_host=sc["ssh_host"],
                                auth_type=sc["auth_type"],
                                credential_service=sc["credential_service"],
                                cost_limit=remaining_budget,
                                page_limit=remaining_pages,
                                max_depth=_max_depth,
                                focus=_focus,
                                num_scan_workers=1,
                                num_score_workers=_num_score_workers,
                                num_ingest_workers=_num_ingest_workers,
                                scan_only=_scan_only,
                                score_only=_score_only,
                                bulk_discover=False,
                                skip_reset=multi_site,
                                on_scan_progress=on_scan,
                                on_score_progress=on_score,
                                on_ingest_progress=on_ingest,
                                on_artifact_progress=on_artifact,
                                on_artifact_score_progress=on_artifact_score,
                                on_worker_status=on_worker_status,
                            )
                        except Exception as e:
                            wiki_logger.warning("Site %s failed: %s", sc["base_url"], e)
                            continue

                        for key in (
                            "scanned",
                            "scored",
                            "ingested",
                            "artifacts",
                        ):
                            combined[key] += result.get(key, 0)
                        combined["cost"] += result.get("cost", 0)
                        combined["elapsed_seconds"] += result.get("elapsed_seconds", 0)

                        remaining_budget -= result.get("cost", 0)
                        if remaining_pages is not None:
                            remaining_pages -= result.get("scored", 0)

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

            # Print summary AFTER Live display exits (outside `with` block)
            # so it appears cleanly below the frozen display
            display.print_summary()

            return combined

        result = asyncio.run(
            run_all_sites_unified(
                _facility=facility,
                _site_configs=site_configs,
                _cost_limit=cost_limit,
                _max_pages=max_pages,
                _max_depth=max_depth,
                _focus=focus,
                _scan_only=scan_only,
                _score_only=score_only,
                _use_rich=use_rich,
                _num_score_workers=score_workers,
                _num_ingest_workers=ingest_workers,
            )
        )

        # Display final results
        log_print(
            f"  [green]{result.get('scanned', 0)} pages scanned, "
            f"{result.get('scored', 0)} scored, "
            f"{result.get('ingested', 0)} ingested, "
            f"{result.get('artifacts', 0)} artifacts[/green]"
        )
        log_print(
            f"  [dim]Cost: ${result.get('cost', 0):.2f}, "
            f"Time: {result.get('elapsed_seconds', 0):.1f}s[/dim]"
        )
    except Exception as e:
        log_print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback

            traceback.print_exc()

    log_print("\n[green]Documentation discovery complete.[/green]")


@signals.command("run", hidden=True)
@click.argument("facility")
@click.option(
    "--cost-limit",
    "-c",
    type=float,
    default=10.0,
    help="Maximum LLM spend in USD (default: $10)",
)
@click.option(
    "--signal-limit",
    "-l",
    type=int,
    default=None,
    help="Maximum signals to enrich (for debugging)",
)
@click.option(
    "--tdi-path",
    type=str,
    default=None,
    help="Override TDI function directory (default: from facility config)",
)
@click.option(
    "--focus",
    "-f",
    type=str,
    default=None,
    help="Focus on specific physics domain (e.g., 'equilibrium')",
)
@click.option(
    "--scan-only",
    is_flag=True,
    default=False,
    help="Only scan signals, skip LLM enrichment",
)
@click.option(
    "--enrich-only",
    is_flag=True,
    default=False,
    help="Only enrich already-discovered signals",
)
@click.option(
    "--enrich-workers",
    type=int,
    default=2,
    help="Number of parallel enrich workers (default: 2)",
)
@click.option(
    "--check-workers",
    type=int,
    default=1,
    help="Number of parallel check workers (default: 1)",
)
@click.option(
    "--no-rich",
    is_flag=True,
    default=False,
    help="Use logging output instead of rich progress display",
)
def signals_run(
    facility: str,
    cost_limit: float,
    signal_limit: int | None,
    tdi_path: str | None,
    focus: str | None,
    scan_only: bool,
    enrich_only: bool,
    enrich_workers: int,
    check_workers: int,
    no_rich: bool,
) -> None:
    """Discover and document facility data signals.

    Scans TDI function files (.fun) to discover physics-level data accessors.
    TDI functions are the primary data access layer, abstracting over raw
    MDSplus paths with built-in versioning and source selection.

    \b
    Pipeline stages:
      SCAN: Parse TDI .fun files, extract quantities, create signals
      ENRICH: LLM classification with .fun source code context
      CHECK: Test TDI accessor execution against reference shot

    \b
    Examples:
      # Full discovery pipeline (scan + enrich + check)
      imas-codex discover signals tcv

      # Scan only, no LLM enrichment
      imas-codex discover signals tcv --scan-only

      # Enrich already-discovered signals
      imas-codex discover signals tcv --enrich-only -c 5.0

      # Override TDI path
      imas-codex discover signals tcv --tdi-path /usr/local/CRPP/tdi/tcv
    """
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.discovery.data import (
        get_data_discovery_stats,
        reset_transient_signals,
        run_parallel_data_discovery,
    )

    # Auto-detect if rich can run (TTY check) or use no_rich flag
    use_rich = not no_rich and sys.stdout.isatty()

    if use_rich:
        console = Console()
    else:
        console = None
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )

    data_logger = logging.getLogger("imas_codex.discovery.data")
    if not use_rich:
        data_logger.setLevel(logging.INFO)

    def log_print(msg: str, style: str = "") -> None:
        """Print to console or log, stripping rich markup."""
        clean_msg = re.sub(r"\[[^\]]+\]", "", msg)
        if console:
            console.print(msg)
        else:
            data_logger.info(clean_msg)

    # Get facility config
    try:
        config = get_facility(facility)
        ssh_host = config.get("ssh_host", facility)
    except Exception as e:
        log_print(f"[red]Error loading facility config: {e}[/red]")
        raise SystemExit(1) from e

    # Get facility data source config
    data_sources = config.get("data_sources", {})
    tdi_config = data_sources.get("tdi", {})

    # Auto-configure TDI path from facility config (unless CLI override)
    if not enrich_only and not tdi_path:
        tdi_primary = tdi_config.get("primary_path")
        if tdi_primary:
            tdi_path = tdi_primary
            log_print(f"[dim]Using TDI path from config: {tdi_path}[/dim]")

    # Get reference shot from TDI config
    reference_shot = tdi_config.get("reference_shot")

    # Require TDI path for scan
    if not enrich_only and not tdi_path:
        log_print(
            "[yellow]No TDI path specified. Use --tdi-path or configure "
            "data_sources.tdi.primary_path in facility config.[/yellow]"
        )
        raise SystemExit(1)

    # Reset orphaned claims
    log_print(f"[bold]Starting data discovery for {facility.upper()}[/bold]")
    reset_transient_signals(facility, silent=True)

    # Show current state
    stats = get_data_discovery_stats(facility)
    if stats.get("total", 0) > 0:
        log_print(
            f"[dim]Existing signals: {stats.get('total', 0)} "
            f"(discovered={stats.get('discovered', 0)}, "
            f"enriched={stats.get('enriched', 0)}, "
            f"checked={stats.get('checked', 0)})[/dim]"
        )

    # Display configuration
    if not scan_only:
        log_print(f"Cost limit: ${cost_limit:.2f}")
    if signal_limit:
        log_print(f"Signal limit: {signal_limit}")
    if reference_shot:
        log_print(f"Reference shot: {reference_shot}")
    if tdi_path:
        log_print(f"TDI path: {tdi_path}")
    if focus:
        log_print(f"Focus: {focus}")

    worker_parts = []
    if not enrich_only:
        worker_parts.append("1 scan")
    if not scan_only:
        worker_parts.append(f"{enrich_workers} enrich")
        worker_parts.append(f"{check_workers} check")
    log_print(f"Workers: {', '.join(worker_parts)}")

    try:

        async def run_data_discovery():
            if use_rich:
                from imas_codex.discovery.data.progress import DataProgressDisplay

                with DataProgressDisplay(
                    facility=facility,
                    cost_limit=cost_limit,
                    signal_limit=signal_limit,
                    focus=focus or "",
                    console=console,
                    discover_only=scan_only,
                    enrich_only=enrich_only,
                ) as display:
                    # Periodic graph state refresh
                    async def refresh_graph_state():
                        while True:
                            try:
                                graph_stats = get_data_discovery_stats(facility)
                                display.update_from_graph(
                                    total_signals=graph_stats.get("total", 0),
                                    signals_discovered=graph_stats.get("discovered", 0),
                                    signals_enriched=graph_stats.get("enriched", 0),
                                    signals_checked=graph_stats.get("checked", 0),
                                    signals_skipped=graph_stats.get("skipped", 0),
                                    signals_failed=graph_stats.get("failed", 0),
                                    pending_enrich=graph_stats.get("pending_enrich", 0),
                                    pending_check=graph_stats.get("pending_check", 0),
                                    accumulated_cost=graph_stats.get(
                                        "accumulated_cost", 0.0
                                    ),
                                )
                            except asyncio.CancelledError:
                                raise
                            except Exception as e:
                                data_logger.debug("Graph refresh failed: %s", e)
                            await asyncio.sleep(0.5)

                    async def queue_ticker():
                        while True:
                            try:
                                display.tick()
                            except asyncio.CancelledError:
                                raise
                            except Exception as e:
                                data_logger.debug("Display tick failed: %s", e)
                            await asyncio.sleep(0.15)

                    refresh_task = asyncio.create_task(refresh_graph_state())
                    ticker_task = asyncio.create_task(queue_ticker())

                    def on_discover(msg, stats, results=None):
                        display.update_discover(msg, stats, results)

                    def on_enrich(msg, stats, results=None):
                        display.update_enrich(msg, stats, results)

                    def on_check(msg, stats, results=None):
                        display.update_check(msg, stats, results)

                    def on_worker_status(worker_group):
                        display.update_worker_status(worker_group)

                    try:
                        result = await run_parallel_data_discovery(
                            facility=facility,
                            ssh_host=ssh_host,
                            tdi_path=tdi_path,
                            reference_shot=reference_shot,
                            cost_limit=cost_limit,
                            signal_limit=signal_limit,
                            focus=focus,
                            num_enrich_workers=enrich_workers,
                            num_check_workers=check_workers,
                            discover_only=scan_only,
                            enrich_only=enrich_only,
                            on_discover_progress=on_discover,
                            on_enrich_progress=on_enrich,
                            on_check_progress=on_check,
                            on_worker_status=on_worker_status,
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

                    display.print_summary()

                return result
            else:
                # Non-rich mode: logging only
                def log_on_discover(msg, stats, results=None):
                    if msg != "idle":
                        data_logger.info(f"SCAN: {msg} (processed={stats.processed})")

                def log_on_enrich(msg, stats, results=None):
                    if msg != "idle":
                        data_logger.info(
                            f"ENRICH: {msg} (processed={stats.processed}, "
                            f"cost=${stats.cost:.3f})"
                        )

                def log_on_check(msg, stats, results=None):
                    if msg != "idle":
                        data_logger.info(f"CHECK: {msg} (processed={stats.processed})")

                result = await run_parallel_data_discovery(
                    facility=facility,
                    ssh_host=ssh_host,
                    tdi_path=tdi_path,
                    reference_shot=reference_shot,
                    cost_limit=cost_limit,
                    signal_limit=signal_limit,
                    focus=focus,
                    num_enrich_workers=enrich_workers,
                    num_check_workers=check_workers,
                    discover_only=scan_only,
                    enrich_only=enrich_only,
                    on_discover_progress=log_on_discover,
                    on_enrich_progress=log_on_enrich,
                    on_check_progress=log_on_check,
                )
                return result

        result = asyncio.run(run_data_discovery())

        # Final summary for non-rich mode
        if not use_rich:
            data_logger.info(
                f"Discovery complete: discovered={result.get('discovered', 0)}, "
                f"enriched={result.get('enriched', 0)}, "
                f"checked={result.get('checked', 0)}, "
                f"cost=${result.get('cost', 0):.3f}, "
                f"elapsed={result.get('elapsed_seconds', 0):.1f}s"
            )

    except KeyboardInterrupt:
        log_print("\n[yellow]Discovery interrupted by user[/yellow]")
        raise SystemExit(130) from None
    except Exception as e:
        log_print(f"\n[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from e

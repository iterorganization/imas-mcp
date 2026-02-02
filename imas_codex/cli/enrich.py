"""Enrich commands: AI-assisted metadata generation for graph nodes."""

from __future__ import annotations

import asyncio
import logging

import click
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console()


@click.group()
def enrich() -> None:
    """Enrich graph nodes with AI-generated metadata.

    Uses CodeAgent to analyze and describe data from multiple sources:
    - TreeNodes from MDSplus/HDF5 trees
    - Wiki pages from facility documentation
    - Code files from ingested source

    \b
      imas-codex enrich nodes     Enrich TreeNode metadata
      imas-codex enrich run       Run a custom enrichment task
    """
    pass


@enrich.command("run")
@click.argument("task")
@click.option(
    "--type",
    "agent_type",
    default="enrichment",
    type=click.Choice(["enrichment", "mapping", "exploration"]),
    help="Agent type to use",
)
@click.option(
    "--cost-limit",
    "-c",
    default=None,
    type=float,
    help="Maximum cost budget in USD",
)
@click.option("--verbose", "-v", is_flag=True, help="Show agent reasoning")
def agent_run(
    task: str, agent_type: str, cost_limit: float | None, verbose: bool
) -> None:
    """Run an agent with a task using smolagents CodeAgent.

    The agent generates Python code to autonomously:
    - Query the Neo4j knowledge graph
    - Search code examples and IMAS paths
    - Adapt and self-debug to solve problems

    Examples:
        imas-codex enrich run "Describe what \\RESULTS::ASTRA is used for"

        imas-codex enrich run "Find IMAS paths for electron temperature" --type mapping

        imas-codex enrich run "Explore EPFL for equilibrium codes" --type exploration -c 1.0
    """
    from imas_codex.agentic import quick_task_sync

    click.echo(f"Running {agent_type} agent (CodeAgent)...")
    if verbose:
        click.echo(f"Task: {task}")
    if cost_limit:
        click.echo(f"Cost limit: ${cost_limit:.2f}")
    click.echo()

    try:
        result = quick_task_sync(task, agent_type, verbose, cost_limit)
        click.echo("\n=== Agent Response ===")
        click.echo(result)
    except Exception as e:
        click.echo(f"Agent error: {e}", err=True)
        raise SystemExit(1) from None


@enrich.command("nodes")
@click.argument("paths", nargs=-1)
@click.option(
    "--prompt",
    "-p",
    default=None,
    help="Guidance for enrichment (e.g., 'Focus on equilibrium signals')",
)
@click.option(
    "--limit", "-n", default=None, type=int, help="Max nodes to enrich (default: all)"
)
@click.option("--tree", default=None, help="Filter to specific tree name")
@click.option(
    "--status", default="pending", help="Target status (pending, enriched, stale)"
)
@click.option("--force", is_flag=True, help="Include all nodes regardless of status")
@click.option(
    "--linked",
    is_flag=True,
    help="Only nodes with code context (more reliable enrichment)",
)
@click.option(
    "--batch-size",
    "-b",
    default=None,
    type=int,
    help="Paths per batch (auto-selected if not set: 100 for Flash, 200 for Pro)",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="LLM model to use (default: from config for 'enrichment' task)",
)
@click.option(
    "--cost-limit",
    "-c",
    default=None,
    type=float,
    help="Maximum cost budget in USD (default: no limit)",
)
@click.option("--dry-run", is_flag=True, help="Preview without persisting to graph")
@click.option("--verbose", "-v", is_flag=True, help="Show verbose output")
def enrich_nodes(
    paths: tuple[str, ...],
    prompt: str | None,
    limit: int | None,
    tree: str | None,
    status: str,
    force: bool,
    linked: bool,
    batch_size: int | None,
    model: str | None,
    cost_limit: float | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Enrich TreeNode metadata using CodeAgent.

    The agent generates Python code to gather context from the
    knowledge graph and code examples, then produces physics-accurate
    descriptions. Uses adaptive problem-solving and self-debugging.

    \b
    EXAMPLES:
        # Enrich all pending nodes in the results tree
        imas-codex enrich nodes --tree results

        # Focus on specific physics with guidance
        imas-codex enrich nodes --tree results -p "Focus on equilibrium signals"

        # Enrich all pending nodes across all trees
        imas-codex enrich nodes

        # Limit to first 100 nodes
        imas-codex enrich nodes --tree tcv_shot --limit 100

        # Process stale nodes (marked for re-enrichment)
        imas-codex enrich nodes --status stale

        # Only enrich nodes with code context (more reliable)
        imas-codex enrich nodes --linked

        # Use Pro model for higher quality
        imas-codex enrich nodes --model google/gemini-3-pro-preview -b 200

        # Preview without saving
        imas-codex enrich nodes --dry-run
    """
    from imas_codex.agentic import (
        BatchProgress,
        batch_enrich_paths,
        compose_batches,
        discover_nodes_to_enrich,
        estimate_enrichment_cost,
        get_model_for_task,
        get_parent_path,
    )

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Resolve model from config if not specified
    effective_model = model or get_model_for_task("enrichment")

    # Determine target status
    target_status = "all" if force else status

    # Get paths - either from args or discover from graph
    if paths:
        path_list = list(paths)
        console.print(f"[cyan]Enriching {len(path_list)} specified paths...[/cyan]")
        tree_name = tree or "unknown"
    else:
        if force:
            filter_desc = "all statuses (--force)"
        else:
            filter_desc = f"status='{target_status}'"
        if linked:
            filter_desc += ", with code context"
        console.print(f"[cyan]Discovering nodes with {filter_desc}...[/cyan]")
        nodes = discover_nodes_to_enrich(
            tree_name=tree,
            status=target_status,
            with_context_only=linked,
            limit=limit,
        )
        if not nodes:
            console.print(f"[yellow]No nodes found with {filter_desc}.[/yellow]")
            return
        path_list = [n["path"] for n in nodes]
        with_ctx = sum(1 for n in nodes if n["has_context"])
        console.print(
            f"[green]Found {len(path_list)} nodes[/green] "
            f"([dim]{with_ctx} with code context[/dim])"
        )
        tree_name = tree or nodes[0].get("tree", "unknown") if nodes else "unknown"

    # Auto-select batch size based on model
    effective_batch_size = batch_size
    if effective_batch_size is None:
        if "pro" in effective_model.lower():
            effective_batch_size = 200
        else:
            effective_batch_size = 100

    # Compose smart batches grouped by parent (for preview)
    batches = compose_batches(
        path_list, batch_size=effective_batch_size, group_by_parent=True
    )

    # Show cost estimate
    cost_est = estimate_enrichment_cost(len(path_list), effective_batch_size)
    cost_info = (
        f"[dim]Batches: {len(batches)} | "
        f"Est. time: {cost_est['estimated_hours'] * 60:.0f}min | "
        f"Est. cost: ${cost_est['estimated_cost']:.2f}"
    )
    if cost_limit is not None:
        cost_info += f" | Limit: ${cost_limit:.2f}"
    cost_info += "[/dim]"
    console.print()
    console.print(cost_info)
    console.print(f"[dim]Model: {effective_model} (smolagents CodeAgent)[/dim]")

    if dry_run:
        console.print("\n[yellow][DRY RUN] Will not persist to graph[/yellow]")
        console.print("\n[cyan]Batch preview:[/cyan]")
        for i, batch in enumerate(batches[:5], 1):
            parent = get_parent_path(batch[0]) if batch else "?"
            console.print(f"  Batch {i}: {len(batch)} paths from [bold]{parent}[/bold]")
            for p in batch[:3]:
                console.print(f"    {p}")
            if len(batch) > 3:
                console.print(f"    [dim]... and {len(batch) - 3} more[/dim]")
        if len(batches) > 5:
            console.print(f"\n  [dim]... and {len(batches) - 5} more batches[/dim]")
        return

    # State for progress display (updated by callback)
    class ProgressState:
        def __init__(self) -> None:
            self.batch_num = 0
            self.total_batches = len(batches)
            self.parent_path = ""
            self.paths_processed = 0
            self.paths_total = len(path_list)
            self.enriched = 0
            self.errors = 0
            self.high_conf = 0
            self.elapsed = 0.0

        def rate(self) -> float:
            return self.paths_processed / self.elapsed if self.elapsed > 0 else 0

    state = ProgressState()

    def create_progress_display() -> Group:
        """Create the rich progress display."""
        # Overall progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Enriching"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("→"),
            TimeRemainingColumn(),
        )
        task = progress.add_task("", total=state.paths_total)
        progress.update(task, completed=state.paths_processed)

        # Current batch info
        batch_info = Table.grid(padding=(0, 2))
        batch_info.add_column(style="dim")
        batch_info.add_column(style="bold")
        batch_info.add_row(
            "Batch:",
            f"{state.batch_num}/{state.total_batches}",
        )
        batch_info.add_row("Tree:", tree_name)
        batch_info.add_row("Group:", state.parent_path or "—")

        # Statistics
        stats = Table.grid(padding=(0, 2))
        stats.add_column(style="dim")
        stats.add_column(justify="right")
        stats.add_row("Enriched:", f"[green]{state.enriched}[/green]")
        stats.add_row("Errors:", f"[red]{state.errors}[/red]")
        stats.add_row("High conf:", f"[cyan]{state.high_conf}[/cyan]")
        stats.add_row("Rate:", f"{state.rate():.1f} paths/sec")

        # Combine into panels
        info_panel = Panel(
            batch_info,
            title="[bold]Current Batch[/bold]",
            border_style="blue",
            padding=(0, 1),
        )
        stats_panel = Panel(
            stats,
            title="[bold]Statistics[/bold]",
            border_style="green",
            padding=(0, 1),
        )

        # Layout with side-by-side panels
        layout = Table.grid(expand=True)
        layout.add_column(ratio=1)
        layout.add_column(ratio=1)
        layout.add_row(info_panel, stats_panel)

        return Group(progress, layout)

    # Progress callback that updates state
    live_display: Live | None = None

    def on_progress(p: BatchProgress) -> None:
        state.batch_num = p.batch_num
        state.total_batches = p.total_batches
        state.parent_path = p.parent_path
        state.paths_processed = p.paths_processed
        state.enriched = p.enriched
        state.errors = p.errors
        state.high_conf = p.high_confidence
        state.elapsed = p.elapsed_seconds
        if live_display:
            live_display.update(create_progress_display())

    async def run_with_progress() -> list:
        nonlocal live_display
        with Live(
            create_progress_display(), console=console, refresh_per_second=4
        ) as live:
            live_display = live
            return await batch_enrich_paths(
                paths=path_list,
                tree_name=tree_name,
                batch_size=effective_batch_size,
                verbose=verbose,
                dry_run=False,
                model=effective_model,
                progress_callback=on_progress,
            )

    console.print()
    results = asyncio.run(run_with_progress())
    console.print()

    # Final summary
    enriched_count = sum(1 for r in results if r.description)
    error_count = sum(1 for r in results if r.error)
    high_conf_count = sum(1 for r in results if r.confidence == "high")

    summary = Table(title="Enrichment Summary", show_header=False, box=None)
    summary.add_column(style="dim")
    summary.add_column(justify="right")
    summary.add_row("Total paths:", str(len(results)))
    summary.add_row("Enriched:", f"[green]{enriched_count}[/green]")
    summary.add_row("Errors:", f"[red]{error_count}[/red]")
    summary.add_row("High confidence:", f"[cyan]{high_conf_count}[/cyan]")
    summary.add_row("Time:", f"{state.elapsed:.1f}s")
    if state.elapsed > 0:
        summary.add_row("Rate:", f"{len(results) / state.elapsed:.1f} paths/sec")
    summary.add_row("", "[green]✓ Persisted to graph[/green]")

    console.print(Panel(summary, border_style="green"))


@enrich.command("mark-stale")
@click.argument("pattern")
@click.option("--tree", default=None, help="Filter to specific tree name")
@click.option("--dry-run", is_flag=True, help="Preview without updating")
def enrich_mark_stale(
    pattern: str,
    tree: str | None,
    dry_run: bool,
) -> None:
    """Mark TreeNodes as stale for re-enrichment.

    Matches nodes by path pattern and sets enrichment_status='stale'.
    Use this when new context is available and you want to re-process nodes.

    \b
    EXAMPLES:
        # Mark all LIUQE nodes as stale
        imas-codex enrich mark-stale "LIUQE"

        # Mark nodes in results tree matching pattern
        imas-codex enrich mark-stale "THOMSON" --tree results

        # Preview what would be marked
        imas-codex enrich mark-stale "BOLO" --dry-run
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        where_clauses = [
            "t.enrichment_status = 'enriched'",
            f"t.path CONTAINS '{pattern}'",
        ]
        if tree:
            where_clauses.append(f't.tree_name = "{tree}"')

        # Count matching nodes
        count_query = f"""
            MATCH (t:TreeNode)
            WHERE {" AND ".join(where_clauses)}
            RETURN count(t) AS count
        """
        result = gc.query(count_query)
        count = result[0]["count"] if result else 0

        if count == 0:
            click.echo(f"No enriched nodes matching '{pattern}' found.")
            return

        if dry_run:
            click.echo(f"[DRY RUN] Would mark {count} nodes as stale")
            # Show sample
            sample_query = f"""
                MATCH (t:TreeNode)
                WHERE {" AND ".join(where_clauses)}
                RETURN t.path AS path LIMIT 10
            """
            samples = gc.query(sample_query)
            for s in samples:
                click.echo(f"  {s['path']}")
            if count > 10:
                click.echo(f"  ... and {count - 10} more")
            return

        # Mark as stale
        update_query = f"""
            MATCH (t:TreeNode)
            WHERE {" AND ".join(where_clauses)}
            SET t.enrichment_status = 'stale'
            RETURN count(t) AS updated
        """
        result = gc.query(update_query)
        updated = result[0]["updated"] if result else 0
        click.echo(f"✓ Marked {updated} nodes as stale")

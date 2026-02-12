"""Ingest command group: Code file ingestion from remote facilities."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()


@click.group()
def ingest() -> None:
    """Ingest code examples from remote facilities.

    \b
      imas-codex ingest run <facility>   Process discovered SourceFile nodes
      imas-codex ingest status <facility> Show queue statistics
      imas-codex ingest list <facility>   List discovered files
    """
    pass


@ingest.command("run")
@click.argument("facility")
@click.option(
    "--limit",
    "-n",
    default=None,
    type=int,
    help="Maximum files to process (default: all discovered files)",
)
@click.option(
    "--min-score",
    default=0.0,
    type=float,
    help="Minimum interest score threshold (default: 0.0)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-ingest files even if already present",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be processed without ingesting",
)
def ingest_run(
    facility: str,
    limit: int | None,
    min_score: float,
    force: bool,
    dry_run: bool,
) -> None:
    """Process discovered SourceFile nodes for a facility.

    Scouts discover files for ingestion using the queue_source_files MCP tool.
    This command fetches those files, generates embeddings, and creates
    CodeExample nodes with searchable chunks.

    Examples:
        # Process all discovered files
        imas-codex ingest run tcv

        # Process only high-priority files
        imas-codex ingest run tcv --min-score 0.7

        # Limit to 100 files
        imas-codex ingest run tcv -n 100

        # Preview what would be processed
        imas-codex ingest run tcv --dry-run
    """
    import asyncio

    from imas_codex.ingestion import get_pending_files, ingest_files

    # Get pending files (discovered, not yet ingested)
    console.print(f"[cyan]Fetching discovered files for {facility}...[/cyan]")
    query_limit = limit if limit is not None else 10000  # Large number for "all"
    pending = get_pending_files(
        facility, limit=query_limit, min_interest_score=min_score
    )

    if not pending:
        console.print("[yellow]No discovered files awaiting ingestion.[/yellow]")
        console.print(
            "Scouts can discover files using the queue_source_files MCP tool."
        )
        return

    console.print(f"[green]Found {len(pending)} discovered files[/green]")

    if dry_run:
        console.print("\n[cyan]Files that would be processed:[/cyan]")
        for i, f in enumerate(pending[:20], 1):
            score = f.get("interest_score", 0.5)
            console.print(f"  {i}. [{score:.2f}] {f['path']}")
        if len(pending) > 20:
            console.print(f"  ... and {len(pending) - 20} more")
        return

    # Run ingestion with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting...", total=len(pending))

        def progress_callback(current: int, total: int, message: str) -> None:
            # Update total if pipeline reports a different count (e.g., after skip)
            progress.update(
                task, completed=current, total=total, description=message[:50]
            )

        try:
            stats = asyncio.run(
                ingest_files(
                    facility=facility,
                    remote_paths=None,  # Use graph queue
                    progress_callback=progress_callback,
                    force=force,
                    limit=limit,
                )
            )

            # Final update to ensure 100%
            progress.update(task, completed=stats["files"] + stats["skipped"])

        except Exception as e:
            console.print(f"[red]Error during ingestion: {e}[/red]")
            raise SystemExit(1) from e

    # Print summary
    console.print("\n[green]Ingestion complete![/green]")
    console.print(f"  Files processed: {stats['files']}")
    console.print(f"  Chunks created:  {stats['chunks']}")
    console.print(f"  IDS references:  {stats['ids_found']}")
    console.print(f"  MDSplus paths:   {stats['mdsplus_paths']}")
    console.print(f"  TreeNodes linked: {stats['tree_nodes_linked']}")
    console.print(f"  Skipped:         {stats['skipped']}")


@ingest.command("status")
@click.argument("facility")
def ingest_status(facility: str) -> None:
    """Show queue statistics for a facility.

    Examples:
        imas-codex ingest status tcv
    """
    from imas_codex.ingestion import get_queue_stats

    stats = get_queue_stats(facility)

    if not stats:
        console.print(f"[yellow]No SourceFile nodes for {facility}[/yellow]")
        return

    table = Table(title=f"SourceFile Queue: {facility}")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right", style="green")

    total = 0
    for status, count in sorted(stats.items()):
        table.add_row(status, str(count))
        total += count

    table.add_row("─" * 10, "─" * 5)
    table.add_row("Total", str(total), style="bold")

    console.print(table)


@ingest.command("queue")
@click.argument("facility")
@click.argument("paths", nargs=-1)
@click.option(
    "--from-file",
    "-f",
    "from_file",
    type=click.Path(exists=True),
    help="Read file paths from a text file (one per line)",
)
@click.option(
    "--stdin",
    is_flag=True,
    help="Read file paths from stdin",
)
@click.option(
    "--interest-score",
    "-s",
    default=0.5,
    type=float,
    help="Interest score for all files (0.0-1.0, default: 0.5)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be queued without making changes",
)
def ingest_queue(
    facility: str,
    paths: tuple[str, ...],
    from_file: str | None,
    stdin: bool,
    interest_score: float,
    dry_run: bool,
) -> None:
    """Discover source files for ingestion.

    Accepts paths as arguments, from a file, or from stdin. Creates
    SourceFile nodes with status='discovered'. Already-discovered or ingested
    files are skipped automatically.

    Examples:
        # Discover paths directly (LLM-friendly)
        imas-codex ingest queue tcv /path/a.py /path/b.py /path/c.py

        # Discover from file (for large batches)
        imas-codex ingest queue tcv -f files.txt

        # Discover from stdin (pipe from rg)
        ssh tcv 'rg -l "IMAS" /home' | imas-codex ingest queue tcv --stdin

        # Set priority score
        imas-codex ingest queue tcv /path/a.py -s 0.9

        # Preview
        imas-codex ingest queue tcv /path/a.py --dry-run
    """
    from imas_codex.ingestion import queue_source_files

    # Read file paths from arguments, file, or stdin
    path_list: list[str] = []
    if paths:
        path_list = list(paths)
    elif from_file:
        path_list = Path(from_file).read_text().strip().splitlines()
    elif stdin:
        path_list = sys.stdin.read().strip().splitlines()
    else:
        console.print(
            "[red]Error: Provide paths as arguments, --from-file, or --stdin[/red]"
        )
        raise SystemExit(1)

    # Filter empty lines and comments
    path_list = [
        p.strip() for p in path_list if p.strip() and not p.strip().startswith("#")
    ]

    if not path_list:
        console.print("[yellow]No file paths provided[/yellow]")
        return

    console.print(f"[cyan]Discovering {len(path_list)} files for {facility}...[/cyan]")

    if dry_run:
        console.print("\n[cyan]Files that would be discovered:[/cyan]")
        for i, path in enumerate(path_list[:20], 1):
            console.print(f"  {i}. {path}")
        if len(path_list) > 20:
            console.print(f"  ... and {len(path_list) - 20} more")
        console.print(f"\n[dim]Interest score: {interest_score}[/dim]")
        return

    result = queue_source_files(
        facility=facility,
        file_paths=path_list,
        interest_score=interest_score,
        discovered_by="cli",
    )

    console.print(f"[green]✓ Discovered: {result['discovered']}[/green]")
    console.print(
        f"[yellow]↷ Skipped: {result['skipped']} (already discovered/ingested)[/yellow]"
    )
    if result["errors"]:
        for err in result["errors"]:
            console.print(f"[red]✗ Error: {err}[/red]")

    console.print(
        f"\n[dim]Run ingestion: imas-codex ingest run {facility} "
        f"-n {min(result['discovered'], 500)}[/dim]"
    )


@ingest.command("list")
@click.argument("facility")
@click.option(
    "--status",
    "-s",
    default="discovered",
    type=click.Choice(["discovered", "ingested", "failed", "stale", "all"]),
    help="Filter by status (default: discovered)",
)
@click.option(
    "--limit",
    "-n",
    default=50,
    type=int,
    help="Maximum files to show (default: 50)",
)
def ingest_list(facility: str, status: str, limit: int) -> None:
    """List SourceFile nodes for a facility.

    Examples:
        # List discovered files
        imas-codex ingest list tcv

        # List failed files
        imas-codex ingest list tcv -s failed

        # List all files
        imas-codex ingest list tcv -s all
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as client:
        if status == "all":
            result = client.query(
                """
                MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
                RETURN sf.path AS path, sf.status AS status,
                       sf.interest_score AS score, sf.error AS error
                ORDER BY sf.interest_score DESC
                LIMIT $limit
                """,
                facility=facility,
                limit=limit,
            )
        else:
            result = client.query(
                """
                MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
                WHERE sf.status = $status
                RETURN sf.path AS path, sf.status AS status,
                       sf.interest_score AS score, sf.error AS error
                ORDER BY sf.interest_score DESC
                LIMIT $limit
                """,
                facility=facility,
                status=status,
                limit=limit,
            )

    if not result:
        console.print(f"[yellow]No SourceFile nodes with status '{status}'[/yellow]")
        return

    table = Table(title=f"SourceFiles ({status}): {facility}")
    table.add_column("Path", style="cyan", max_width=60)
    table.add_column("Status", style="green")
    table.add_column("Score", justify="right")
    if status == "failed":
        table.add_column("Error", style="red", max_width=30)

    for row in result:
        score = f"{row['score']:.2f}" if row["score"] is not None else "-"
        if status == "failed":
            table.add_row(row["path"], row["status"], score, row["error"] or "")
        else:
            table.add_row(row["path"], row["status"], score)

    console.print(table)
    console.print(f"\n[dim]Showing {len(result)} of possibly more files[/dim]")

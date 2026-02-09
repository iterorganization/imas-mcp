"""Embedding management commands."""

import logging

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

logger = logging.getLogger(__name__)


@click.group()
def embed():
    """Manage embeddings for graph nodes.

    \b
    Commands:
      update       Add embeddings to nodes missing them
      indexes      Show or create vector indexes
      status       Show embedding coverage statistics
    """
    pass


@embed.command("update")
@click.option(
    "--label",
    "-l",
    required=True,
    type=click.Choice(
        ["FacilitySignal", "FacilityPath", "TreeNode", "WikiArtifact"],
        case_sensitive=True,
    ),
    help="Node label to update embeddings for",
)
@click.option(
    "--facility",
    "-f",
    type=str,
    default=None,
    help="Restrict to specific facility (optional)",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=100,
    help="Batch size for embedding (default: 100)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be embedded without making changes",
)
@click.option("--no-rich", is_flag=True, help="Disable rich output")
def update(
    label: str,
    facility: str | None,
    batch_size: int,
    dry_run: bool,
    no_rich: bool,
) -> None:
    """Update description embeddings for nodes.

    Finds nodes with description but no embedding and
    generates embeddings in batches.

    \b
    Examples:
        # Dry run to see what would be embedded
        imas-codex embed update --label FacilitySignal --dry-run

        # Update all FacilitySignal nodes
        imas-codex embed update --label FacilitySignal

        # Update only TCV signals
        imas-codex embed update --label FacilitySignal --facility tcv
    """
    from imas_codex.embeddings.description import embed_descriptions_batch
    from imas_codex.graph.client import GraphClient

    console = Console() if not no_rich else None

    # Build query based on label
    facility_filter = ""
    if facility:
        facility_filter = f"AND n.facility_id = '{facility}'"

    count_query = f"""
        MATCH (n:{label})
        WHERE n.description IS NOT NULL
          AND n.description <> ''
          AND n.embedding IS NULL
          {facility_filter}
        RETURN count(n) AS total
    """

    fetch_query = f"""
        MATCH (n:{label})
        WHERE n.description IS NOT NULL
          AND n.description <> ''
          AND n.embedding IS NULL
          {facility_filter}
        RETURN n.id AS id, n.description AS description
        LIMIT $batch_size
    """

    update_query = f"""
        UNWIND $items AS item
        MATCH (n:{label} {{id: item.id}})
        SET n.embedding = item.embedding
    """

    with GraphClient() as gc:
        # Get total count
        result = gc.query(count_query)
        total = result[0]["total"] if result else 0

        if total == 0:
            msg = f"No {label} nodes need embedding update"
            if facility:
                msg += f" (facility={facility})"
            if console:
                console.print(f"[green]{msg}[/green]")
            else:
                click.echo(msg)
            return

        facility_msg = f" for {facility}" if facility else ""
        if dry_run:
            msg = f"[DRY RUN] Would embed {total} {label} descriptions{facility_msg}"
            if console:
                console.print(f"[yellow]{msg}[/yellow]")
            else:
                click.echo(msg)
            return

        if console:
            console.print(f"Updating {total} {label} embeddings{facility_msg}...")

        processed = 0

        if console and not no_rich:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Embedding {label}...", total=total)

                while True:
                    # Fetch batch
                    rows = gc.query(fetch_query, batch_size=batch_size)
                    if not rows:
                        break

                    items = [
                        {"id": r["id"], "description": r["description"]} for r in rows
                    ]

                    # Batch embed
                    items = embed_descriptions_batch(items)

                    # Update graph
                    gc.query(update_query, items=items)

                    processed += len(items)
                    progress.update(task, advance=len(items))
                    progress.update(
                        task, description=f"Embedded {processed}/{total} {label}"
                    )
        else:
            while True:
                rows = gc.query(fetch_query, batch_size=batch_size)
                if not rows:
                    break

                items = [{"id": r["id"], "description": r["description"]} for r in rows]
                items = embed_descriptions_batch(items)
                gc.query(update_query, items=items)

                processed += len(items)
                click.echo(f"Embedded {processed}/{total} {label}")

        msg = f"Updated {processed} {label} description embeddings"
        if console:
            console.print(f"[green]{msg}[/green]")
        else:
            click.echo(msg)


@embed.command("indexes")
@click.option("--create", is_flag=True, help="Create missing indexes")
@click.option("--no-rich", is_flag=True, help="Disable rich output")
def indexes(create: bool, no_rich: bool) -> None:
    """Show or create vector indexes for description embeddings.

    \b
    Examples:
        # List current indexes
        imas-codex embed indexes

        # Create missing indexes
        imas-codex embed indexes --create
    """
    from imas_codex.graph.client import GraphClient

    console = Console() if not no_rich else None

    with GraphClient() as gc:
        # Get existing vector indexes
        result = gc.query("""
            SHOW INDEXES
            YIELD name, type, labelsOrTypes, properties
            WHERE type = 'VECTOR'
            RETURN name, labelsOrTypes, properties
            ORDER BY name
        """)

        if console:
            from rich.table import Table

            table = Table(title="Vector Indexes")
            table.add_column("Name")
            table.add_column("Label")
            table.add_column("Property")

            for row in result:
                table.add_row(
                    row["name"],
                    ", ".join(row["labelsOrTypes"]) if row["labelsOrTypes"] else "",
                    ", ".join(row["properties"]) if row["properties"] else "",
                )
            console.print(table)
        else:
            click.echo("Vector Indexes:")
            for row in result:
                labels = ", ".join(row["labelsOrTypes"]) if row["labelsOrTypes"] else ""
                props = ", ".join(row["properties"]) if row["properties"] else ""
                click.echo(f"  {row['name']}: {labels} -> {props}")

        if create:
            gc.ensure_vector_indexes()
            if console:
                console.print("[green]Ensured all vector indexes exist[/green]")
            else:
                click.echo("Ensured all vector indexes exist")


@embed.command("status")
@click.option("--no-rich", is_flag=True, help="Disable rich output")
def status(no_rich: bool) -> None:
    """Show embedding coverage across node types.

    Shows how many nodes have descriptions and how many have been embedded.
    """
    from imas_codex.graph.client import GraphClient

    console = Console() if not no_rich else None

    labels_with_desc_embedding = [
        "FacilitySignal",
        "FacilityPath",
        "TreeNode",
        "WikiArtifact",
    ]

    stats = []
    with GraphClient() as gc:
        for label in labels_with_desc_embedding:
            result = gc.query(f"""
                MATCH (n:{label})
                WITH count(n) AS total,
                     count(CASE WHEN n.description IS NOT NULL
                                 AND n.description <> '' THEN 1 END) AS with_desc,
                     count(CASE WHEN n.embedding IS NOT NULL THEN 1 END) AS with_emb
                RETURN total, with_desc, with_emb
            """)
            if result:
                r = result[0]
                stats.append(
                    {
                        "label": label,
                        "total": r["total"],
                        "with_desc": r["with_desc"],
                        "with_emb": r["with_emb"],
                        "needs_update": r["with_desc"] - r["with_emb"],
                    }
                )

    if console:
        from rich.table import Table

        table = Table(title="Description Embedding Status")
        table.add_column("Node Type")
        table.add_column("Total", justify="right")
        table.add_column("Has Description", justify="right")
        table.add_column("Has Embedding", justify="right")
        table.add_column("Needs Update", justify="right")

        for s in stats:
            needs = s["needs_update"]
            needs_str = f"[yellow]{needs}[/yellow]" if needs > 0 else str(needs)
            table.add_row(
                s["label"],
                str(s["total"]),
                str(s["with_desc"]),
                str(s["with_emb"]),
                needs_str,
            )
        console.print(table)
    else:
        click.echo("Description Embedding Status:")
        for s in stats:
            click.echo(
                f"  {s['label']}: {s['with_emb']}/{s['with_desc']} embedded "
                f"({s['needs_update']} need update)"
            )

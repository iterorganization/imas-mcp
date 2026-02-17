"""Discovery CLI: Graph-led facility exploration.

Modular CLI package with domain commands as direct subcommands:

    imas-codex discover paths tcv          # Run paths discovery
    imas-codex discover wiki jt-60sa        # Run wiki discovery
    imas-codex discover signals tcv        # Run signals discovery
    imas-codex discover files tcv          # Run files discovery
    imas-codex discover status tcv         # Status (all domains)
    imas-codex discover status tcv -d wiki # Status (wiki only)
    imas-codex discover clear tcv          # Clear (all domains)
    imas-codex discover clear tcv -d paths # Clear (paths only)
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from imas_codex.cli.discover.common import DISCOVERY_DOMAINS, domain_option


@click.group()
def discover():
    """Discover facility resources with graph-led exploration.

    \b
    Domain Commands (each runs discovery directly):
      paths              Directory structure discovery
      files              Source file discovery from scored paths
      wiki               Wiki page discovery and ingestion
      signals            Facility signal discovery

    \b
    Management Commands:
      status             Show discovery statistics
      clear              Clear discovery data
      seed               Seed root paths from config
      inspect            Debug view of scanned/scored paths

    \b
    Examples:
      imas-codex discover paths jet            # Run paths discovery
      imas-codex discover wiki jt-60sa          # Run wiki discovery
      imas-codex discover status jet           # All domains status
      imas-codex discover status jet -d wiki   # Wiki status only
      imas-codex discover clear jet -d paths   # Clear paths only
    """
    pass


# =============================================================================
# Status Command
# =============================================================================


@discover.command("status")
@click.argument("facility")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@domain_option()
def discover_status(facility: str, as_json: bool, domain: str | None) -> None:
    """Show discovery statistics for a facility.

    By default shows status for all discovery domains.
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

    from imas_codex.cli.discover.common import use_rich_output
    from imas_codex.discovery import get_discovery_stats, get_high_value_paths
    from imas_codex.discovery.signals.parallel import get_data_discovery_stats
    from imas_codex.discovery.wiki.parallel import get_wiki_discovery_stats

    use_rich = use_rich_output()

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

            if domain is None or domain == "files":
                from imas_codex.discovery.files.scanner import (
                    get_file_discovery_stats,
                )

                file_stats = get_file_discovery_stats(facility)
                output["files"] = file_stats

            click.echo(json_module.dumps(output, indent=2))
        else:
            from imas_codex.discovery.paths.progress import print_discovery_status

            print_discovery_status(facility, use_rich=use_rich, domain=domain)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


# =============================================================================
# Clear Command
# =============================================================================


@discover.command("clear")
@click.argument("facility")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
@domain_option()
def discover_clear(facility: str, force: bool, domain: str | None) -> None:
    """Clear discovered data for a facility.

    By default clears ALL domains. Use --domain/-d to target specific domains.

    \b
    Examples:
      imas-codex discover clear jet              # All domains
      imas-codex discover clear jet -d paths     # Paths only
      imas-codex discover clear jet -d wiki      # Wiki only
      imas-codex discover clear jet -d files     # Files only
      imas-codex discover clear jet --force      # Skip confirmation
    """
    from imas_codex.discovery import clear_facility_paths, get_discovery_stats
    from imas_codex.discovery.signals import (
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
                image_result = gc.query(
                    "MATCH (i:Image {facility_id: $f}) RETURN count(i) AS cnt",
                    f=facility,
                )
                images = image_result[0]["cnt"] if image_result else 0
            if pages > 0 or artifacts > 0 or images > 0:
                label = f"wiki pages + {chunks} chunks + {artifacts} artifacts + {images} images"
                items_to_clear.append(
                    (label, pages or artifacts or images, clear_facility_wiki)
                )

        # Signals domain
        if domain is None or domain == "signals":
            signal_stats = get_data_discovery_stats(facility)
            signal_total = signal_stats.get("total", 0)
            if signal_total > 0:
                items_to_clear.append(
                    ("signals + epochs", signal_total, clear_facility_signals)
                )

        # Files domain
        if domain is None or domain == "files":
            from imas_codex.discovery.files.scanner import (
                clear_facility_files,
                get_file_discovery_stats,
            )

            file_stats = get_file_discovery_stats(facility)
            file_total = file_stats.get("total", 0)
            if file_total > 0:
                items_to_clear.append(
                    ("source files", file_total, clear_facility_files)
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
            "images_deleted",
            "signals_deleted",
            "data_access_deleted",
            "epochs_deleted",
            "checkpoints_deleted",
            "paths_deleted",
            "source_files_deleted",
            "code_chunks_deleted",
            "data_references_deleted",
            "users_deleted",
        ):
            if result.get(key):
                label = key.replace("_deleted", "").replace("_", " ")
                parts.append(f"{result[key]} {label}")
        click.echo(f"✓ Deleted {', '.join(parts)} for {facility}")
    else:
        click.echo(f"✓ Deleted {result} {name} for {facility}")


# =============================================================================
# Seed Command
# =============================================================================


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


# =============================================================================
# Inspect Command
# =============================================================================


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
                MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
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
                MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
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


# =============================================================================
# Register Domain Commands
# =============================================================================

# Import and register domain commands as direct subcommands.
# Each domain module exposes a single @click.command that runs its pipeline.
# `discover paths tcv` runs paths discovery directly (no subgroup).
from imas_codex.cli.discover.files import files  # noqa: E402
from imas_codex.cli.discover.paths import paths  # noqa: E402
from imas_codex.cli.discover.signals import signals  # noqa: E402
from imas_codex.cli.discover.wiki import wiki  # noqa: E402

discover.add_command(paths)
discover.add_command(wiki)
discover.add_command(signals)
discover.add_command(files)

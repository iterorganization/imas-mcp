"""Files discovery command: Remote file scanning and LLM scoring."""

from __future__ import annotations

import logging
import sys

import click
from rich.console import Console

logger = logging.getLogger(__name__)


@click.command()
@click.argument("facility")
@click.option(
    "--min-score",
    type=float,
    default=0.5,
    help="Minimum FacilityPath score to include (default: 0.5)",
)
@click.option(
    "--max-paths",
    type=int,
    default=None,
    help="Maximum FacilityPaths to scan for files",
)
@click.option(
    "--focus",
    "-f",
    help="Focus on specific patterns",
)
@click.option(
    "--cost-limit",
    "-c",
    type=float,
    default=5.0,
    help="Maximum LLM spend in USD",
)
@click.option(
    "--score-batch-size",
    type=int,
    default=50,
    help="Batch size for LLM scoring (default: 50)",
)
@click.option(
    "--scan-only",
    is_flag=True,
    help="Only scan for files, skip scoring",
)
@click.option(
    "--score-only",
    is_flag=True,
    help="Score already discovered files (skip scanning)",
)
@click.option(
    "--no-rich",
    is_flag=True,
    default=False,
    help="Use logging output instead of rich progress display",
)
def files(
    facility: str,
    min_score: float,
    max_paths: int | None,
    focus: str | None,
    cost_limit: float,
    score_batch_size: int,
    scan_only: bool,
    score_only: bool,
    no_rich: bool,
) -> None:
    """Discover source files in scored facility paths.

    Two-stage pipeline:
    1. SCAN: SSH to facility, list files in high-scoring FacilityPaths
    2. SCORE: LLM batch-scores discovered files for IMAS relevance

    \b
    Examples:
      imas-codex discover files tcv
      imas-codex discover files tcv --min-score 0.7 --scan-only
      imas-codex discover files tcv -c 2.0 --score-batch-size 100
      imas-codex discover files tcv -f equilibrium
    """
    from imas_codex.discovery.base.facility import get_facility

    # Auto-detect rich output
    use_rich = not no_rich and sys.stdout.isatty()

    # Always configure file logging (DEBUG level to disk)
    from imas_codex.cli.logging import configure_cli_logging

    configure_cli_logging("files")

    if use_rich:
        console = Console()
    else:
        console = None
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )

    file_logger = logging.getLogger("imas_codex.discovery.files")

    def log_print(msg: str) -> None:
        import re

        clean_msg = re.sub(r"\[[^\]]+\]", "", msg)
        if console:
            console.print(msg)
        else:
            file_logger.info(clean_msg)

    try:
        config = get_facility(facility)
    except Exception as e:
        log_print(f"[red]Error loading facility config: {e}[/red]")
        raise SystemExit(1) from e

    ssh_host = config.get("ssh_host")
    if not ssh_host:
        log_print(f"[red]No SSH host configured for {facility}[/red]")
        raise SystemExit(1)

    log_print(f"\n[bold]File Discovery: {facility}[/bold]")
    log_print(f"  SSH host: {ssh_host}")
    log_print(f"  Min score: {min_score}")
    log_print(f"  Cost limit: ${cost_limit:.2f}")
    if max_paths:
        log_print(f"  Max paths: {max_paths}")
    if focus:
        log_print(f"  Focus: {focus}")
    log_print("")

    # Stage 1: Scan files via SSH
    total_scanned = 0
    total_scored = 0
    total_cost = 0.0

    if not score_only:
        log_print("[bold]Stage 1: Scanning files via SSH[/bold]")

        try:
            from imas_codex.discovery.files import scan_facility_files

            if use_rich:
                from rich.status import Status

                with Status(
                    "[cyan]Scanning files...[/cyan]",
                    console=console,
                    spinner="dots",
                ) as status:

                    def scan_progress(current, total, msg):
                        status.update(f"[cyan]{msg}[/cyan]")

                    scan_kwargs = {
                        "facility": facility,
                        "ssh_host": ssh_host,
                        "min_score": min_score,
                        "progress_callback": scan_progress,
                    }
                    if max_paths is not None:
                        scan_kwargs["max_paths"] = max_paths
                    scan_result = scan_facility_files(**scan_kwargs)
            else:

                def scan_log(current, total, msg):
                    file_logger.info(f"SCAN: [{current}/{total}] {msg}")

                scan_kwargs = {
                    "facility": facility,
                    "ssh_host": ssh_host,
                    "min_score": min_score,
                    "progress_callback": scan_log,
                }
                if max_paths is not None:
                    scan_kwargs["max_paths"] = max_paths
                scan_result = scan_facility_files(**scan_kwargs)

            total_scanned = scan_result.get("new_files", 0)
            paths_scanned = scan_result.get("total_paths", 0)
            log_print(
                f"  [green]{total_scanned} files discovered "
                f"in {paths_scanned} paths[/green]"
            )

        except Exception as e:
            log_print(f"[red]Scan failed: {e}[/red]")
            import traceback

            traceback.print_exc()
            if scan_only:
                raise SystemExit(1) from e

    if scan_only:
        log_print("\n[green]File scanning complete.[/green]")
        return

    # Stage 2: Score files via LLM
    log_print("\n[bold]Stage 2: Scoring files via LLM[/bold]")

    try:
        from imas_codex.discovery.files import score_facility_files

        if use_rich:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task("[cyan]Scoring files...", total=None)

                def score_progress(current, total, msg):
                    if total > 0:
                        progress.update(task_id, total=total, completed=current)
                    progress.update(
                        task_id,
                        description=f"[cyan]Scoring: {msg}",
                    )

                score_result = score_facility_files(
                    facility=facility,
                    cost_limit=cost_limit,
                    batch_size=score_batch_size,
                    focus=focus,
                    progress_callback=score_progress,
                )
        else:

            def score_log(current, total, msg):
                file_logger.info(f"SCORE: [{current}/{total}] {msg}")

            score_result = score_facility_files(
                facility=facility,
                cost_limit=cost_limit,
                batch_size=score_batch_size,
                focus=focus,
                progress_callback=score_log,
            )

        total_scored = score_result.get("total_scored", 0)
        total_cost = score_result.get("cost", 0.0)
        total_skipped = score_result.get("total_skipped", 0)

        log_print(
            f"  [green]{total_scored} files scored, {total_skipped} skipped[/green]"
        )
        log_print(f"  [dim]Cost: ${total_cost:.2f}[/dim]")

    except Exception as e:
        log_print(f"[red]Scoring failed: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from e

    # Summary
    log_print(f"\n  [green]{total_scanned} scanned, {total_scored} scored[/green]")
    log_print(f"  [dim]Total cost: ${total_cost:.2f}[/dim]")
    log_print("\n[green]File discovery complete.[/green]")

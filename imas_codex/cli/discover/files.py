"""Files discovery command: Parallel file scanning, scoring, and ingestion."""

from __future__ import annotations

import asyncio
import logging
import re
import time

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
    default=100,
    help="Maximum FacilityPaths to scan for files (default: 100)",
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
    "--scan-workers",
    type=int,
    default=2,
    help="Number of parallel scan workers (default: 2)",
)
@click.option(
    "--score-workers",
    type=int,
    default=2,
    help="Number of parallel score workers (default: 2)",
)
@click.option(
    "--code-workers",
    type=int,
    default=2,
    help="Number of parallel code ingest workers (default: 2)",
)
@click.option(
    "--scan-only",
    is_flag=True,
    help="Only scan for files, skip scoring and ingestion",
)
@click.option(
    "--score-only",
    is_flag=True,
    help="Score already discovered files (skip scanning and ingestion)",
)
@click.option(
    "--time",
    "time_limit",
    type=int,
    default=None,
    help="Maximum runtime in minutes",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
def files(
    facility: str,
    min_score: float,
    max_paths: int,
    focus: str | None,
    cost_limit: float,
    scan_workers: int,
    score_workers: int,
    code_workers: int,
    scan_only: bool,
    score_only: bool,
    time_limit: int | None,
    verbose: bool,
) -> None:
    """Discover and ingest source files from scored facility paths.

    Runs parallel workers through a multi-stage pipeline:

    \b
    - SCAN: SSH to facility, enumerate files in scored FacilityPaths
    - SCORE: LLM batch-scores discovered files for relevance
    - CODE: Fetch, chunk, embed high-scoring code files
    - ARTIFACT: Ingest documents, notebooks, configs

    \b
    Examples:
      imas-codex discover files tcv
      imas-codex discover files tcv --min-score 0.7 --scan-only
      imas-codex discover files tcv -c 2.0 --code-workers 4
      imas-codex discover files tcv -f equilibrium --time 10
    """
    from imas_codex.cli.logging import configure_cli_logging
    from imas_codex.cli.rich_output import should_use_rich
    from imas_codex.discovery.base.facility import get_facility

    use_rich = should_use_rich()
    configure_cli_logging("files", facility=facility, verbose=verbose)

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
    if focus:
        log_print(f"  Focus: {focus}")

    # Worker summary
    worker_parts = [f"{scan_workers} scan"]
    if not scan_only:
        worker_parts.append(f"{score_workers} score")
    if not scan_only and not score_only:
        worker_parts.append(f"{code_workers} code")
        worker_parts.append("1 artifact")
    log_print(f"  Workers: {', '.join(worker_parts)}")
    if time_limit is not None:
        log_print(f"  Time limit: {time_limit} min")
    log_print("")

    deadline: float | None = None
    if time_limit is not None:
        deadline = time.time() + (time_limit * 60)

    # Run parallel discovery
    try:
        from imas_codex.discovery.files.parallel import run_parallel_file_discovery

        async def _run():
            # Logging callbacks for non-rich mode
            def log_scan(msg, stats, results=None):
                if msg != "idle":
                    file_logger.info("SCAN: %s", msg)

            def log_score(msg, stats, results=None):
                if msg != "idle":
                    file_logger.info("SCORE: %s", msg)

            def log_code(msg, stats, results=None):
                if msg != "idle":
                    file_logger.info("CODE: %s", msg)

            def log_artifact(msg, stats, results=None):
                if msg != "idle":
                    file_logger.info("ARTIFACT: %s", msg)

            return await run_parallel_file_discovery(
                facility=facility,
                ssh_host=ssh_host,
                cost_limit=cost_limit,
                min_score=min_score,
                max_paths=max_paths,
                focus=focus,
                num_scan_workers=scan_workers,
                num_score_workers=score_workers,
                num_code_workers=code_workers,
                num_artifact_workers=1,
                scan_only=scan_only,
                score_only=score_only,
                deadline=deadline,
                on_scan_progress=log_scan,
                on_score_progress=log_score,
                on_code_progress=log_code,
                on_artifact_progress=log_artifact,
            )

        result = asyncio.run(_run())

        # Summary
        log_print(
            f"\n  [green]{result.get('scanned', 0)} scanned, "
            f"{result.get('scored', 0)} scored, "
            f"{result.get('code_ingested', 0)} code ingested, "
            f"{result.get('artifacts_ingested', 0)} artifacts ingested[/green]"
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

    log_print("\n[green]File discovery complete.[/green]")

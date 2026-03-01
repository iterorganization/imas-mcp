"""Code discovery command: Parallel code scanning, scoring, and ingestion."""

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
    default=None,
    help="Minimum FacilityPath score to include (default: from settings)",
)
@click.option(
    "--max-paths",
    type=int,
    default=100,
    help="Maximum FacilityPaths to scan per batch (default: 100)",
)
@click.option(
    "--focus",
    "-f",
    help="Focus on specific patterns (e.g. 'equilibrium', 'transport')",
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
    default=1,
    help="Number of parallel score workers (default: 1)",
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
@click.option(
    "--rescan",
    is_flag=True,
    default=False,
    help="Rescan paths that were previously scanned",
)
def code(
    facility: str,
    min_score: float | None,
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
    rescan: bool,
) -> None:
    """Discover and ingest source code from scored facility paths.

    Scans for code files (Python, Fortran, MATLAB, Julia, C/C++, IDL, TDI)
    using tree-sitter for chunking and embedding.

    \b
    Pipeline stages:
      SCAN:    SSH to facility, enumerate code files + rg pattern enrichment
      TRIAGE:  Fast LLM pass to keep/skip files per directory
      SCORE:   Detailed multi-dimensional LLM scoring with pattern evidence
      INGEST:  Fetch, tree-sitter chunk, embed, extract IDS/MDSplus refs

    Paths are processed highest-value first using weighted dimension scores
    (data_access, IMAS, convention, analysis, modeling).

    \b
    Examples:
      imas-codex discover code tcv
      imas-codex discover code tcv --min-score 0.8 --scan-only
      imas-codex discover code tcv -c 2.0 --code-workers 4
      imas-codex discover code tcv -f equilibrium --time 10
      imas-codex discover code tcv --rescan
    """
    from imas_codex.cli.logging import configure_cli_logging
    from imas_codex.cli.rich_output import should_use_rich
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.settings import get_discovery_threshold

    if min_score is None:
        min_score = get_discovery_threshold()

    use_rich = should_use_rich()
    configure_cli_logging("code", facility=facility, verbose=verbose)

    if use_rich:
        console = Console()
    else:
        console = None
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )

    file_logger = logging.getLogger("imas_codex.discovery.code")

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

    if rescan:
        from imas_codex.discovery.code.graph_ops import set_files_scan_after

        set_files_scan_after(facility)
        log_print(
            "[yellow]Rescan enabled — previously scanned paths will be re-processed[/yellow]"
        )

    deadline: float | None = None
    if time_limit is not None:
        deadline = time.time() + (time_limit * 60)

    try:
        from imas_codex.discovery.code.parallel import run_parallel_code_discovery

        # Logging callbacks for non-rich mode
        def log_scan(msg, stats, results=None):
            if msg != "idle":
                file_logger.info("SCAN: %s", msg)

        def log_score(msg, stats, results=None):
            if msg != "idle":
                file_logger.info("SCORE: %s", msg)

        def log_code_cb(msg, stats, results=None):
            if msg != "idle":
                file_logger.info("CODE: %s", msg)

        def log_enrich(msg, stats, results=None):
            if msg != "idle":
                file_logger.info("ENRICH: %s", msg)

        if not use_rich:
            log_print(f"\n[bold]Code Discovery: {facility}[/bold]")
            log_print(f"  SSH host: {ssh_host}")
            log_print(f"  Min score: {min_score}")
            log_print(f"  Cost limit: ${cost_limit:.2f}")
            if focus:
                log_print(f"  Focus: {focus}")
            log_print("")

            result = asyncio.run(
                run_parallel_code_discovery(
                    facility=facility,
                    ssh_host=ssh_host,
                    cost_limit=cost_limit,
                    min_score=min_score,
                    max_paths=max_paths,
                    focus=focus,
                    num_scan_workers=scan_workers,
                    num_score_workers=score_workers,
                    num_code_workers=code_workers,
                    scan_only=scan_only,
                    score_only=score_only,
                    deadline=deadline,
                    on_scan_progress=log_scan,
                    on_score_progress=log_score,
                    on_enrich_progress=log_enrich,
                    on_code_progress=log_code_cb,
                )
            )
        else:
            # Rich progress display
            from imas_codex.cli.discover.common import create_discovery_monitor
            from imas_codex.discovery.code.progress import FileProgressDisplay

            service_monitor = create_discovery_monitor(
                config,
                check_graph=True,
                check_embed=not scan_only and not score_only,
                check_model=not scan_only,
                check_ssh=True,
                check_auth=False,
            )

            # Suppress noisy INFO during rich display
            for mod in (
                "imas_codex.embeddings",
                "imas_codex.discovery.code",
            ):
                logging.getLogger(mod).setLevel(logging.WARNING)

            with FileProgressDisplay(
                facility=facility,
                cost_limit=cost_limit,
                focus=focus or "",
                console=console,
                scan_only=scan_only,
                score_only=score_only,
            ) as display:
                display.service_monitor = service_monitor

                async def run_with_display():
                    await service_monitor.__aenter__()

                    async def refresh_graph_state():
                        while True:
                            try:
                                display.refresh_from_graph(facility)
                            except asyncio.CancelledError:
                                raise
                            except Exception:
                                pass
                            await asyncio.sleep(0.5)

                    async def queue_ticker():
                        while True:
                            try:
                                display.tick()
                            except asyncio.CancelledError:
                                raise
                            except Exception:
                                pass
                            await asyncio.sleep(0.15)

                    refresh_task = asyncio.create_task(refresh_graph_state())
                    ticker_task = asyncio.create_task(queue_ticker())

                    def on_scan(msg, stats, results=None):
                        display.update_scan(msg, stats, results)

                    def on_score(msg, stats, results=None):
                        display.update_score(msg, stats, results)

                    def on_code(msg, stats, results=None):
                        display.update_code(msg, stats, results)

                    def on_enrich(msg, stats, results=None):
                        display.update_enrich(msg, stats, results)

                    def on_worker_status(worker_group):
                        display.update_worker_status(worker_group)

                    try:
                        return await run_parallel_code_discovery(
                            facility=facility,
                            ssh_host=ssh_host,
                            cost_limit=cost_limit,
                            min_score=min_score,
                            max_paths=max_paths,
                            focus=focus,
                            num_scan_workers=scan_workers,
                            num_score_workers=score_workers,
                            num_code_workers=code_workers,
                            scan_only=scan_only,
                            score_only=score_only,
                            deadline=deadline,
                            on_scan_progress=on_scan,
                            on_score_progress=on_score,
                            on_enrich_progress=on_enrich,
                            on_code_progress=on_code,
                            on_worker_status=on_worker_status,
                            service_monitor=service_monitor,
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
                        await service_monitor.__aexit__(None, None, None)

                result = asyncio.run(run_with_display())

                # Final graph refresh for accurate summary
                try:
                    display.refresh_from_graph(facility)
                except Exception:
                    pass

                display.print_summary()

        # Final output
        scanned = result.get("scanned", 0)
        scored = result.get("scored", 0)
        code_ingested = result.get("code_ingested", 0)
        cost = result.get("cost", 0)
        elapsed = result.get("elapsed_seconds", 0)

        log_print(
            f"\n  [green]{scanned} scanned, {scored} scored, "
            f"{code_ingested} code ingested[/green]"
        )
        log_print(f"  [dim]Cost: ${cost:.2f}, Time: {elapsed:.1f}s[/dim]")

    except Exception as e:
        log_print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback

            traceback.print_exc()
        raise SystemExit(1) from e

    log_print("\n[green]Code discovery complete.[/green]")

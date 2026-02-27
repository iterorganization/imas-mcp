"""Image discovery command: Find and process images from scored facility paths."""

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
    default=0.7,
    help="Minimum FacilityPath score to include (default: 0.7)",
)
@click.option(
    "--max-paths",
    type=int,
    default=50,
    help="Maximum FacilityPaths to scan for images (default: 50)",
)
@click.option(
    "--cost-limit",
    "-c",
    type=float,
    default=2.0,
    help="Maximum VLM spend in USD (default: 2.0)",
)
@click.option(
    "--workers",
    type=int,
    default=2,
    help="Number of parallel image workers (default: 2)",
)
@click.option(
    "--vlm-workers",
    type=int,
    default=1,
    help="Number of parallel VLM captioning workers (default: 1)",
)
@click.option(
    "--store-bytes",
    is_flag=True,
    default=False,
    help="Keep image bytes in graph after VLM scoring (default: clear)",
)
@click.option(
    "--scan-only",
    is_flag=True,
    help="Only scan for image files, skip VLM captioning",
)
@click.option(
    "--time",
    "time_limit",
    type=int,
    default=None,
    help="Maximum runtime in minutes",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
def images(
    facility: str,
    min_score: float,
    max_paths: int,
    cost_limit: float,
    workers: int,
    vlm_workers: int,
    store_bytes: bool,
    scan_only: bool,
    time_limit: int | None,
    verbose: bool,
) -> None:
    """Discover and process images from scored facility paths.

    Scans for image files (PNG, JPG, SVG, etc.) in scored FacilityPaths,
    fetches via SCP, downsamples, and optionally captions with a VLM.

    Operates on SourceFile nodes with file_category='image' that were
    discovered by the code discovery pipeline scan phase.

    \b
    Pipeline stages:
      FETCH:    Download image files via SCP batch tar
      PROCESS:  Downsample, extract metadata, create Image nodes
      CAPTION:  VLM captioning and relevance scoring

    \b
    Examples:
      imas-codex discover images tcv
      imas-codex discover images tcv --scan-only
      imas-codex discover images tcv -c 1.0 --vlm-workers 2
      imas-codex discover images tcv --store-bytes
    """
    from imas_codex.cli.logging import configure_cli_logging
    from imas_codex.cli.rich_output import should_use_rich
    from imas_codex.discovery.base.facility import get_facility

    use_rich = should_use_rich()
    configure_cli_logging("images", facility=facility, verbose=verbose)

    if use_rich:
        console = Console()
    else:
        console = None
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )

    img_logger = logging.getLogger("imas_codex.discovery.files")

    def log_print(msg: str) -> None:
        clean_msg = re.sub(r"\[[^\]]+\]", "", msg)
        if console:
            console.print(msg)
        else:
            img_logger.info(clean_msg)

    try:
        config = get_facility(facility)
    except Exception as e:
        log_print(f"[red]Error loading facility config: {e}[/red]")
        raise SystemExit(1) from e

    ssh_host = config.get("ssh_host")
    if not ssh_host:
        log_print(f"[red]No SSH host configured for {facility}[/red]")
        raise SystemExit(1)

    deadline: float | None = None
    if time_limit is not None:
        deadline = time.time() + (time_limit * 60)

    try:
        from imas_codex.discovery.files.image_pipeline import (
            run_image_discovery,
        )

        log_print(f"\n[bold]Image Discovery: {facility}[/bold]")
        log_print(f"  SSH host: {ssh_host}")
        log_print(f"  Min score: {min_score}")
        log_print(f"  Cost limit: ${cost_limit:.2f}")
        log_print("")

        result = asyncio.run(
            run_image_discovery(
                facility=facility,
                ssh_host=ssh_host,
                cost_limit=cost_limit,
                min_score=min_score,
                max_paths=max_paths,
                num_image_workers=workers,
                num_vlm_workers=vlm_workers,
                store_images=store_bytes,
                scan_only=scan_only,
                deadline=deadline,
            )
        )

        fetched = result.get("images_fetched", 0)
        captioned = result.get("images_captioned", 0)
        cost = result.get("cost", 0)
        elapsed = result.get("elapsed_seconds", 0)

        log_print(f"\n  [green]{fetched} images fetched, {captioned} captioned[/green]")
        log_print(f"  [dim]Cost: ${cost:.2f}, Time: {elapsed:.1f}s[/dim]")

    except Exception as e:
        log_print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback

            traceback.print_exc()
        raise SystemExit(1) from e

    log_print("\n[green]Image discovery complete.[/green]")

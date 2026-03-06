"""Document discovery command: scan, fetch images, VLM captioning."""

from __future__ import annotations

import logging
import time

import click

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
    default=50,
    help="Maximum FacilityPaths to scan (default: 50)",
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
    help="Number of parallel image fetch workers (default: 2)",
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
    help="Only scan for document files, skip image processing",
)
@click.option(
    "--focus",
    "-f",
    help="Focus for VLM scoring (e.g. 'diagnostics', 'equilibrium')",
)
@click.option(
    "--time",
    "time_limit",
    type=int,
    default=None,
    help="Maximum runtime in minutes",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed progress")
def documents(
    facility: str,
    min_score: float,
    max_paths: int,
    cost_limit: float,
    workers: int,
    vlm_workers: int,
    store_bytes: bool,
    scan_only: bool,
    focus: str | None,
    time_limit: int | None,
    verbose: bool,
) -> None:
    """Discover documents and images from scored facility paths.

    Scans for document files (PDF, Markdown, notebooks) and images
    (PNG, JPG, SVG, etc.) in scored FacilityPaths. Images are fetched,
    downsampled, and optionally captioned with a VLM.

    \b
    Pipeline stages:
      SCAN:    SSH enumerate document + image files, create Document nodes
      FETCH:   Download images via SCP, create Image nodes
      CAPTION: VLM captioning and relevance scoring

    \b
    Examples:
      imas-codex discover documents tcv
      imas-codex discover documents tcv --scan-only
      imas-codex discover documents tcv -c 1.0 --vlm-workers 2
      imas-codex discover documents tcv -f diagnostics
    """
    from imas_codex.cli.discover.common import (
        DiscoveryConfig,
        make_log_print,
        run_discovery,
        setup_logging,
        use_rich_output,
    )
    from imas_codex.discovery.base.facility import get_facility

    use_rich = use_rich_output()
    console = setup_logging("documents", facility, use_rich=use_rich, verbose=verbose)
    log_print = make_log_print("documents", console)

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
        # Step 1: Scan for document files (synchronous)
        from imas_codex.discovery.documents.scanner import scan_facility_documents

        log_print(f"\n[bold]Document Discovery: {facility}[/bold]")
        log_print(f"  SSH host: {ssh_host}")
        log_print(f"  Min score: {min_score}")
        log_print(f"  Cost limit: ${cost_limit:.2f}")
        if focus:
            log_print(f"  Focus: {focus}")
        log_print("")

        scan_stats = scan_facility_documents(
            facility=facility,
            min_score=min_score,
            max_paths=max_paths,
            ssh_host=ssh_host,
        )

        log_print(
            f"  [green]Scanned: {scan_stats['new_files']} new documents "
            f"in {scan_stats['total_paths']} paths[/green]"
        )

        if scan_only:
            log_print("\n[green]Document scan complete (--scan-only).[/green]")
            return

        # Step 2: Process images (fetch + VLM captioning) via harness
        from imas_codex.discovery.documents.pipeline import (
            DocumentDiscoveryState,
            run_document_discovery,
        )

        # Create state externally so the display can observe it
        state = DocumentDiscoveryState(
            facility=facility,
            ssh_host=ssh_host,
            cost_limit=cost_limit,
            min_score=min_score,
            deadline=deadline,
            store_images=store_bytes,
            scan_only=False,
            focus=focus,
        )

        # Build display for rich mode
        display = None
        if use_rich:
            from imas_codex.discovery.base.progress import (
                DataDrivenProgressDisplay,
                StageDisplaySpec,
            )

            display = DataDrivenProgressDisplay(
                facility=facility,
                cost_limit=cost_limit,
                stages=[
                    StageDisplaySpec(
                        "FETCH", "bold blue", "image", "image_stats", "image_phase"
                    ),
                    StageDisplaySpec(
                        "VLM",
                        "bold magenta",
                        "vlm",
                        "image_score_stats",
                        "image_score_phase",
                    ),
                ],
                console=console,
                focus=focus or "",
                title_suffix="Document Discovery",
            )
            display.set_engine_state(state)

        disc_config = DiscoveryConfig(
            domain="documents",
            facility=facility,
            facility_config=config,
            display=display,
            check_graph=False,
            check_embed=False,
            check_ssh=False,
            verbose=verbose,
        )

        async def async_main(stop_event, service_monitor):
            return await run_document_discovery(
                state,
                num_image_workers=workers,
                num_vlm_workers=vlm_workers,
                stop_event=stop_event,
                on_worker_status=(display.update_worker_status if display else None),
            )

        result = run_discovery(disc_config, async_main)

        if not display:
            fetched = result.get("images_fetched", 0)
            captioned = result.get("images_captioned", 0)
            cost = result.get("cost", 0)
            elapsed = result.get("elapsed_seconds", 0)

            log_print(
                f"\n  [green]{fetched} images fetched, {captioned} captioned[/green]"
            )
            log_print(f"  [dim]Cost: ${cost:.2f}, Time: {elapsed:.1f}s[/dim]")

    except Exception as e:
        log_print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback

            traceback.print_exc()
        raise SystemExit(1) from e

    log_print("\n[green]Document discovery complete.[/green]")

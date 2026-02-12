"""Signals discovery command: TDI expression scanning and enrichment."""

from __future__ import annotations

import asyncio
import logging
import sys

import click
from rich.console import Console

logger = logging.getLogger(__name__)


@click.command()
@click.argument("facility")
@click.option(
    "--cost-limit",
    "-c",
    type=float,
    default=5.0,
    help="Maximum LLM spend in USD",
)
@click.option(
    "--signal-limit",
    "-n",
    type=int,
    default=None,
    help="Maximum signals to process",
)
@click.option(
    "--tdi-path",
    help="TDI expression path to scan (e.g., /home/tdi/...)",
)
@click.option(
    "--focus",
    "-f",
    help="Focus on specific signal patterns (e.g., 'equilibrium')",
)
@click.option(
    "--scan-only",
    is_flag=True,
    help="Only scan for signals, skip enrichment",
)
@click.option(
    "--enrich-only",
    is_flag=True,
    help="Only enrich already-discovered signals, skip scan",
)
@click.option(
    "--enrich-workers",
    type=int,
    default=2,
    help="Number of parallel enrichment workers",
)
@click.option(
    "--check-workers",
    type=int,
    default=4,
    help="Number of parallel check workers",
)
@click.option(
    "--no-rich",
    is_flag=True,
    default=False,
    help="Use logging output instead of rich progress display",
)
def signals(
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
    """Discover signals from MDSplus TDI expressions.

    Scans TDI expression files to discover facility signals, then enriches
    them with descriptions, physics domains, and IMAS mappings.

    \b
    Examples:
      imas-codex discover signals tcv
      imas-codex discover signals tcv --scan-only
      imas-codex discover signals tcv -f equilibrium -c 2.0
      imas-codex discover signals tcv --tdi-path /home/tdi/magnetics
    """
    from imas_codex.discovery.base.facility import get_facility

    # Auto-detect rich output
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

    sig_logger = logging.getLogger("imas_codex.discovery.data")

    def log_print(msg: str, style: str | None = None) -> None:
        import re

        clean_msg = re.sub(r"\[[^\]]+\]", "", msg)
        if console:
            console.print(msg)
        else:
            sig_logger.info(clean_msg)

    try:
        config = get_facility(facility)
    except Exception as e:
        log_print(f"[red]Error loading facility config: {e}[/red]")
        raise SystemExit(1) from e

    ssh_host = config.get("ssh_host")
    if not ssh_host:
        log_print(f"[red]No SSH host configured for {facility}[/red]")
        raise SystemExit(1)

    # Auto-configure TDI path and reference shot from schema-backed config
    data_sources = config.get("data_sources", {})
    tdi_config = data_sources.get("tdi", {})
    mdsplus_config = data_sources.get("mdsplus", {})

    if tdi_path is None:
        tdi_path = tdi_config.get("primary_path")

    if tdi_path is None:
        log_print(
            f"[red]No TDI path configured for {facility}.[/red]\n"
            "Specify with --tdi-path or configure data_sources.tdi.primary_path "
            "in facility YAML"
        )
        raise SystemExit(1)

    # Get reference shot from TDI config (primary) or MDSplus config (fallback)
    reference_shot = tdi_config.get("reference_shot") or mdsplus_config.get(
        "reference_shot"
    )

    # Get exclude functions for TDI scanning
    exclude_functions = tdi_config.get("exclude_functions", [])

    log_print(f"\n[bold]Signal Discovery: {facility}[/bold]")
    log_print(f"  TDI path: {tdi_path}")
    log_print(f"  SSH host: {ssh_host}")
    if reference_shot:
        log_print(f"  Reference shot: {reference_shot}")
    log_print(f"  Cost limit: ${cost_limit:.2f}")
    if signal_limit:
        log_print(f"  Signal limit: {signal_limit}")
    if focus:
        log_print(f"  Focus: {focus}")
    log_print(f"  Workers: {enrich_workers} enrich, {check_workers} check")
    if exclude_functions:
        log_print(f"  Exclude functions: {len(exclude_functions)}")
    log_print("")

    try:
        from imas_codex.discovery.data.parallel import run_parallel_data_discovery

        def log_on_scan(msg, stats, results=None):
            if msg != "idle":
                sig_logger.info(f"SCAN: {msg}")

        def log_on_enrich(msg, stats, results=None):
            if msg != "idle":
                sig_logger.info(f"ENRICH: {msg}")

        def log_on_check(msg, stats, results=None):
            if msg != "idle":
                sig_logger.info(f"CHECK: {msg}")

        if not use_rich:
            result = asyncio.run(
                run_parallel_data_discovery(
                    facility=facility,
                    tdi_path=tdi_path,
                    ssh_host=ssh_host,
                    reference_shot=reference_shot,
                    cost_limit=cost_limit,
                    signal_limit=signal_limit,
                    focus=focus,
                    discover_only=scan_only,
                    enrich_only=enrich_only,
                    num_enrich_workers=enrich_workers,
                    num_check_workers=check_workers,
                    on_discover_progress=log_on_scan,
                    on_enrich_progress=log_on_enrich,
                    on_check_progress=log_on_check,
                )
            )
        else:
            # Rich progress display
            from imas_codex.discovery.base.services import create_service_monitor
            from imas_codex.discovery.data.progress import DataProgressDisplay

            service_monitor = create_service_monitor(
                facility=facility,
                ssh_host=ssh_host,
                check_graph=True,
                check_embed=not scan_only,
                check_ssh=True,
            )

            # Suppress noisy INFO during rich display
            for mod in (
                "imas_codex.embeddings",
                "imas_codex.discovery.data",
            ):
                logging.getLogger(mod).setLevel(logging.WARNING)

            with DataProgressDisplay(
                facility=facility,
                cost_limit=cost_limit,
                signal_limit=signal_limit,
                focus=focus or "",
                console=console,
                scan_only=scan_only,
            ) as display:
                display.state.service_monitor = service_monitor

                async def run_with_display():
                    await service_monitor.__aenter__()

                    async def refresh_graph_state():
                        while True:
                            try:
                                display.tick()
                            except asyncio.CancelledError:
                                raise
                            except Exception:
                                pass
                            await asyncio.sleep(0.15)

                    ticker_task = asyncio.create_task(refresh_graph_state())

                    def on_scan(msg, stats, results=None):
                        display.update_scan(msg, stats, results)

                    def on_enrich(msg, stats, results=None):
                        display.update_enrich(msg, stats, results)

                    def on_check(msg, stats, results=None):
                        display.update_check(msg, stats, results)

                    def on_worker_status(worker_group):
                        display.update_worker_status(worker_group)

                    try:
                        return await run_parallel_data_discovery(
                            facility=facility,
                            tdi_path=tdi_path,
                            ssh_host=ssh_host,
                            reference_shot=reference_shot,
                            cost_limit=cost_limit,
                            signal_limit=signal_limit,
                            focus=focus,
                            discover_only=scan_only,
                            enrich_only=enrich_only,
                            num_enrich_workers=enrich_workers,
                            num_check_workers=check_workers,
                            on_discover_progress=on_scan,
                            on_enrich_progress=on_enrich,
                            on_check_progress=on_check,
                            on_worker_status=on_worker_status,
                        )
                    finally:
                        ticker_task.cancel()
                        try:
                            await ticker_task
                        except asyncio.CancelledError:
                            pass
                        await service_monitor.__aexit__(None, None, None)

                result = asyncio.run(run_with_display())
                display.print_summary()

        # Final output
        scanned = result.get("scanned", 0)
        enriched = result.get("enriched", 0)
        checked = result.get("checked", 0)
        cost = result.get("cost", 0)
        elapsed = result.get("elapsed_seconds", 0)

        log_print(
            f"\n  [green]{scanned} scanned, {enriched} enriched, "
            f"{checked} checked[/green]"
        )
        log_print(f"  [dim]Cost: ${cost:.2f}, Time: {elapsed:.1f}s[/dim]")

    except Exception as e:
        log_print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from e

    log_print("\n[green]Signal discovery complete.[/green]")

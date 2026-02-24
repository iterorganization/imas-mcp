"""Signals discovery command: facility-agnostic signal scanning and enrichment.

Dispatches to registered scanner plugins based on facility config data_sources.
Scanner plugins handle facility-specific enumeration (TDI, PPF, EDAS, MDSplus,
IMAS, wiki), while shared infrastructure handles LLM enrichment and validation.
"""

from __future__ import annotations

import asyncio
import logging
import time

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
    "--scanners",
    "-s",
    type=str,
    default=None,
    help="Comma-separated scanner types to run (e.g., 'tdi,mdsplus'). "
    "Default: auto-detect from facility config data_sources.",
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
    "--time",
    "time_limit",
    type=int,
    default=None,
    help="Maximum runtime in minutes (e.g., 5). Discovery halts when time expires.",
)
@click.option(
    "--reference-shot",
    type=int,
    default=None,
    help="Override reference shot/pulse for validation",
)
def signals(
    facility: str,
    cost_limit: float,
    signal_limit: int | None,
    scanners: str | None,
    focus: str | None,
    scan_only: bool,
    enrich_only: bool,
    enrich_workers: int,
    check_workers: int,
    time_limit: int | None,
    reference_shot: int | None,
) -> None:
    """Discover signals from facility data sources.

    Scans configured data sources to discover facility signals, then enriches
    them with descriptions, physics domains, and IMAS mappings. Scanners are
    auto-detected from facility config data_sources section.

    \b
    Examples:
      imas-codex discover signals tcv
      imas-codex discover signals tcv --scan-only
      imas-codex discover signals jet -s ppf -c 2.0
      imas-codex discover signals jt-60sa -s edas --scan-only
      imas-codex discover signals tcv -s tdi,mdsplus -f equilibrium
    """
    # Auto-detect rich output
    from imas_codex.cli.rich_output import should_use_rich
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.discovery.signals.scanners.base import (
        get_scanners_for_facility,
        list_scanners,
    )

    use_rich = should_use_rich()

    # Always configure file logging (DEBUG level to disk)
    from imas_codex.cli.logging import configure_cli_logging

    configure_cli_logging("signals", facility=facility)

    if use_rich:
        console = Console()
    else:
        console = None
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )

    sig_logger = logging.getLogger("imas_codex.discovery.signals")

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

    # Resolve scanner types
    if scanners:
        scanner_types = [s.strip() for s in scanners.split(",")]
        # Validate requested scanner types exist
        available = list_scanners()
        invalid = [s for s in scanner_types if s not in available]
        if invalid:
            log_print(
                f"[red]Unknown scanner types: {invalid}. Available: {available}[/red]"
            )
            raise SystemExit(1)
    else:
        # Auto-detect from facility config
        scanner_instances = get_scanners_for_facility(facility)
        scanner_types = [s.scanner_type for s in scanner_instances]

    if not scanner_types:
        log_print(
            f"[red]No data sources configured for {facility}.[/red]\n"
            "Configure data_sources in facility YAML or specify --scanners."
        )
        raise SystemExit(1)

    # Resolve reference shot from config if not specified
    data_sources = config.get("data_sources", {})
    if reference_shot is None:
        for source_config in data_sources.values():
            if isinstance(source_config, dict):
                ref = source_config.get("reference_shot") or source_config.get(
                    "reference_pulse"
                )
                if ref:
                    reference_shot = int(ref)
                    break

    log_print(f"\n[bold]Signal Discovery: {facility}[/bold]")
    log_print(f"  Scanners: {', '.join(scanner_types)}")
    log_print(f"  SSH host: {ssh_host}")
    if reference_shot:
        log_print(f"  Reference shot: {reference_shot}")
    log_print(f"  Cost limit: ${cost_limit:.2f}")
    if signal_limit:
        log_print(f"  Signal limit: {signal_limit}")
    if time_limit is not None:
        log_print(f"  Time limit: {time_limit} min")
    if focus:
        log_print(f"  Focus: {focus}")
    log_print(f"  Workers: {enrich_workers} enrich, {check_workers} check")
    log_print("")

    try:
        from imas_codex.discovery.signals.parallel import run_parallel_data_discovery

        # Compute deadline from time limit
        deadline: float | None = None
        if time_limit is not None:
            deadline = time.time() + (time_limit * 60)

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
                    ssh_host=ssh_host,
                    scanner_types=scanner_types,
                    reference_shot=reference_shot,
                    cost_limit=cost_limit,
                    signal_limit=signal_limit,
                    focus=focus,
                    discover_only=scan_only,
                    enrich_only=enrich_only,
                    deadline=deadline,
                    num_enrich_workers=enrich_workers,
                    num_check_workers=check_workers,
                    on_discover_progress=log_on_scan,
                    on_enrich_progress=log_on_enrich,
                    on_check_progress=log_on_check,
                )
            )
        else:
            # Rich progress display
            from imas_codex.cli.discover.common import create_discovery_monitor
            from imas_codex.discovery.signals.progress import DataProgressDisplay

            service_monitor = create_discovery_monitor(
                config,
                check_graph=True,
                check_embed=not scan_only,
                check_model=not scan_only,
                check_auth=False,
            )

            # Suppress noisy INFO during rich display
            for mod in (
                "imas_codex.embeddings",
                "imas_codex.discovery.signals",
            ):
                logging.getLogger(mod).setLevel(logging.WARNING)

            with DataProgressDisplay(
                facility=facility,
                cost_limit=cost_limit,
                signal_limit=signal_limit,
                focus=focus or "",
                console=console,
                discover_only=scan_only,
                enrich_only=enrich_only,
            ) as display:
                display.service_monitor = service_monitor

                async def run_with_display():
                    await service_monitor.__aenter__()

                    async def refresh_graph_state():
                        from imas_codex.discovery.signals.parallel import (
                            get_data_discovery_stats,
                        )

                        while True:
                            try:
                                stats = get_data_discovery_stats(facility)
                                if stats:
                                    display.update_from_graph(
                                        total_signals=stats.get("total", 0),
                                        signals_discovered=stats.get("discovered", 0),
                                        signals_enriched=stats.get("enriched", 0),
                                        signals_checked=stats.get("checked", 0),
                                        signals_skipped=stats.get("skipped", 0),
                                        signals_failed=stats.get("failed", 0),
                                        pending_enrich=stats.get("pending_enrich", 0),
                                        pending_check=stats.get("pending_check", 0),
                                        accumulated_cost=stats.get(
                                            "accumulated_cost", 0.0
                                        ),
                                    )
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

                    def on_enrich(msg, stats, results=None):
                        display.update_enrich(msg, stats, results)

                    def on_check(msg, stats, results=None):
                        display.update_check(msg, stats, results)

                    def on_worker_status(worker_group):
                        display.update_worker_status(worker_group)

                    try:
                        return await run_parallel_data_discovery(
                            facility=facility,
                            ssh_host=ssh_host,
                            scanner_types=scanner_types,
                            reference_shot=reference_shot,
                            cost_limit=cost_limit,
                            signal_limit=signal_limit,
                            focus=focus,
                            discover_only=scan_only,
                            enrich_only=enrich_only,
                            deadline=deadline,
                            num_enrich_workers=enrich_workers,
                            num_check_workers=check_workers,
                            on_discover_progress=on_scan,
                            on_enrich_progress=on_enrich,
                            on_check_progress=on_check,
                            on_worker_status=on_worker_status,
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
                    from imas_codex.discovery.signals.parallel import (
                        get_data_discovery_stats,
                    )

                    final_stats = get_data_discovery_stats(facility)
                    if final_stats:
                        display.update_from_graph(
                            total_signals=final_stats.get("total", 0),
                            signals_discovered=final_stats.get("discovered", 0),
                            signals_enriched=final_stats.get("enriched", 0),
                            signals_checked=final_stats.get("checked", 0),
                            signals_skipped=final_stats.get("skipped", 0),
                            signals_failed=final_stats.get("failed", 0),
                            pending_enrich=final_stats.get("pending_enrich", 0),
                            pending_check=final_stats.get("pending_check", 0),
                            accumulated_cost=final_stats.get("accumulated_cost", 0.0),
                        )
                except Exception:
                    pass

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

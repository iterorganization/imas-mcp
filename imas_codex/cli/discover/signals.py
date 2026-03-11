"""Signals discovery command: facility-agnostic signal scanning and enrichment.

Dispatches to registered scanner plugins based on facility config data_systems.
Scanner plugins handle facility-specific enumeration (TDI, PPF, EDAS, MDSplus,
IMAS, wiki), while shared infrastructure handles LLM enrichment and validation.
"""

from __future__ import annotations

import asyncio
import logging
import time

import click

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
    "Default: auto-detect from facility config data_systems.",
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
    auto-detected from facility config data_systems section.

    \b
    Examples:
      imas-codex discover signals tcv
      imas-codex discover signals tcv --scan-only
      imas-codex discover signals jet -s ppf -c 2.0
      imas-codex discover signals jt-60sa -s edas --scan-only
      imas-codex discover signals tcv -s tdi,mdsplus -f equilibrium
    """
    # Auto-detect rich output
    from imas_codex.cli.discover.common import (
        DiscoveryConfig,
        make_log_print,
        run_discovery,
        setup_logging,
        use_rich_output,
    )
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.discovery.signals.scanners.base import (
        get_scanners_for_facility,
        list_scanners,
    )

    use_rich = use_rich_output()
    console = setup_logging("signals", facility, use_rich)
    log_print = make_log_print("signals", console)

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
            "Configure data_systems in facility YAML or specify --scanners."
        )
        raise SystemExit(1)

    # Resolve reference shot from config if not specified
    data_systems = config.get("data_systems", {})
    if reference_shot is None:
        for source_config in data_systems.values():
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

        sig_logger = logging.getLogger("imas_codex.discovery.signals")

        # Build display for rich mode
        display = None
        if use_rich:
            from imas_codex.discovery.signals.progress import DataProgressDisplay

            display = DataProgressDisplay(
                facility=facility,
                cost_limit=cost_limit,
                signal_limit=signal_limit,
                focus=focus or "",
                console=console,
                discover_only=scan_only,
                enrich_only=enrich_only,
            )

        # Custom async graph refresh for signals (uses update_from_graph with kwargs)
        async def signals_graph_refresh():
            from imas_codex.discovery.signals.parallel import (
                get_data_discovery_stats,
            )

            stats = await asyncio.to_thread(get_data_discovery_stats, facility)
            if stats and display:
                display.update_from_graph(
                    total_signals=stats.get("total", 0),
                    signals_discovered=stats.get("discovered", 0),
                    signals_enriched=stats.get("enriched", 0),
                    signals_checked=stats.get("checked", 0),
                    signals_skipped=stats.get("skipped", 0),
                    signals_failed=stats.get("failed", 0),
                    pending_enrich=stats.get("pending_enrich", 0),
                    pending_check=stats.get("pending_check", 0),
                    accumulated_cost=stats.get("accumulated_cost", 0.0),
                    signal_groups=stats.get("signal_groups", 0),
                    grouped_signals=stats.get("grouped_signals", 0),
                )

        disc_config = DiscoveryConfig(
            domain="signals",
            facility=facility,
            facility_config=config,
            display=display,
            check_graph=True,
            check_embed=not scan_only,
            check_model=not scan_only,
            check_ssh=True,
            check_auth=False,
            graph_refresh_interval=2.0,
            graph_refresh_fn=signals_graph_refresh if use_rich else None,
            suppress_loggers=[
                "imas_codex.embeddings",
                "imas_codex.discovery.signals",
            ],
        )

        # Callbacks — wire to display or logging
        if display:

            def on_scan(msg, stats, results=None):
                display.update_scan(msg, stats, results)

            def on_extract(msg, stats, results=None):
                display.update_extract(msg, stats, results)

            def on_promote(msg, stats, results=None):
                display.update_promote(msg, stats, results)

            def on_enrich(msg, stats, results=None):
                display.update_enrich(msg, stats, results)

            def on_check(msg, stats, results=None):
                display.update_check(msg, stats, results)

            def on_worker_status(worker_group):
                display.update_worker_status(worker_group)
        else:

            def on_scan(msg, stats, results=None):
                if msg != "idle":
                    sig_logger.info("SEED: %s", msg)

            def on_extract(msg, stats, results=None):
                if msg != "idle":
                    sig_logger.info("EXTRACT: %s", msg)

            def on_promote(msg, stats, results=None):
                if msg != "idle":
                    sig_logger.info("PROMOTE: %s", msg)

            def on_enrich(msg, stats, results=None):
                if msg != "idle":
                    sig_logger.info("ENRICH: %s", msg)

            def on_check(msg, stats, results=None):
                if msg != "idle":
                    sig_logger.info("CHECK: %s", msg)

            on_worker_status = None

        async def async_main(stop_event, service_monitor):
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
                on_extract_progress=on_extract,
                on_promote_progress=on_promote,
                on_enrich_progress=on_enrich,
                on_check_progress=on_check,
                on_worker_status=on_worker_status,
                stop_event=stop_event,
            )

        def on_complete(result):
            if display:
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

        result = run_discovery(disc_config, async_main, on_complete=on_complete)

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

    except KeyboardInterrupt:
        log_print("\n[yellow]Discovery interrupted by user[/yellow]")
        from imas_codex.remote.executor import cleanup_ssh_on_exit

        cleanup_ssh_on_exit()
        raise SystemExit(130) from None
    except Exception as e:
        log_print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        raise SystemExit(1) from e

    log_print("\n[green]Signal discovery complete.[/green]")

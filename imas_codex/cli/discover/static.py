"""Static tree discovery: extract and ingest machine-description MDSplus trees.

Parallel worker pipeline:
  EXTRACT → Claim TreeModelVersion, SSH extract, ingest to graph
  UNITS   → Batched unit extraction for NUMERIC/SIGNAL nodes
  ENRICH  → LLM batch descriptions of tree node physics
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from imas_codex.discovery.base.progress import (
    BaseProgressDisplay,
    PipelineRowConfig,
    ResourceConfig,
    StreamQueue,
    build_pipeline_section,
    build_resource_section,
    format_time,
)

if TYPE_CHECKING:
    from imas_codex.discovery.base.progress import WorkerStats

logger = logging.getLogger(__name__)


# =============================================================================
# Progress state
# =============================================================================


@dataclass
class StaticProgressState:
    """Mutable state for the static tree progress display."""

    start_time: float = field(default_factory=time.time)

    # EXTRACT — bar tracks cumulative nodes, versions in STATS row
    extract_completed: int = 0  # versions completed
    extract_total: int = 1  # versions total
    extract_nodes: int = 0  # cumulative nodes extracted
    extract_nodes_total: int = 0  # estimated total nodes
    extract_rate: float | None = None  # nodes/s
    extract_version: str = ""
    extract_detail: str = ""

    # UNITS
    units_completed: int = 0
    units_total: int = 0
    units_found: int = 0
    units_rate: float | None = None
    units_path: str = ""
    units_detail: str = ""

    # ENRICH
    enrich_completed: int = 0
    enrich_total: int = 0
    enrich_rate: float | None = None
    enrich_path: str = ""
    enrich_description: str = ""
    enrich_cost: float = 0.0

    # Accumulated cost from graph
    accumulated_cost: float = 0.0

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def run_cost(self) -> float:
        return self.enrich_cost

    @property
    def eta_seconds(self) -> float | None:
        """Estimated time to completion based on active work."""
        etas: list[float] = []

        # Extract ETA — based on remaining estimated nodes ÷ nodes/s
        if self.extract_rate and self.extract_rate > 0 and self.extract_nodes_total > 0:
            remaining_nodes = self.extract_nodes_total - self.extract_nodes
            if remaining_nodes > 0:
                etas.append(remaining_nodes / self.extract_rate)

        # Enrich ETA
        if self.enrich_rate and self.enrich_rate > 0:
            remaining = self.enrich_total - self.enrich_completed
            if remaining > 0:
                etas.append(remaining / self.enrich_rate)

        return max(etas) if etas else None


# =============================================================================
# Progress display
# =============================================================================


class StaticProgressDisplay(BaseProgressDisplay):
    """Rich progress display for static tree discovery.

    Pipeline phases with worker annotations:
      EXTRACT  SSH extraction + ingestion per version
      UNITS    Batched unit extraction
      ENRICH   LLM batch descriptions
    """

    def __init__(
        self,
        facility: str,
        cost_limit: float,
        console: Console | None = None,
        enrich: bool = True,
    ) -> None:
        self.facility = facility
        self.cost_limit = cost_limit
        self.enrich = enrich
        self.state = StaticProgressState()
        self.extract_queue = StreamQueue(rate=1.0, max_rate=2.0, min_display_time=0.8)
        self.units_queue = StreamQueue(rate=1.0, max_rate=3.0, min_display_time=0.5)
        self.enrich_queue = StreamQueue(rate=0.5, max_rate=2.0, min_display_time=0.5)
        self._console = console or Console()

        super().__init__(
            facility=facility,
            cost_limit=cost_limit,
            console=self._console,
            title_suffix="Static Tree Discovery",
        )

    def _build_pipeline_section(self) -> Text:
        s = self.state

        # Worker counts
        extract_count, extract_ann = self._count_group_workers("extract")
        units_count, units_ann = self._count_group_workers("units")
        enrich_count, enrich_ann = self._count_group_workers("enrich")

        # Worker completion
        extract_complete = self._worker_complete("extract")
        units_complete = self._worker_complete("units")
        enrich_complete = self._worker_complete("enrich")

        # Estimate total nodes from average per completed version
        if s.extract_completed > 0 and s.extract_total > 0:
            avg_nodes = s.extract_nodes / s.extract_completed
            s.extract_nodes_total = int(avg_nodes * s.extract_total)

        rows = [
            PipelineRowConfig(
                name="EXTRACT",
                style="bold blue",
                completed=s.extract_nodes,
                total=max(s.extract_nodes_total, 1),
                rate=s.extract_rate,
                primary_text=s.extract_version,
                description=s.extract_detail,
                is_complete=extract_complete
                or (s.extract_completed >= s.extract_total and s.extract_total > 0),
                worker_count=extract_count,
                worker_annotation=extract_ann,
            ),
            PipelineRowConfig(
                name="UNITS",
                style="bold cyan",
                completed=s.units_completed,
                total=max(s.units_total, 1),
                rate=s.units_rate,
                primary_text=s.units_path,
                description=s.units_detail,
                is_complete=units_complete
                or (s.units_completed >= s.units_total and s.units_total > 0),
                worker_count=units_count,
                worker_annotation=units_ann,
            ),
            PipelineRowConfig(
                name="ENRICH",
                style="bold magenta",
                completed=s.enrich_completed,
                total=max(s.enrich_total, 1),
                cost=s.enrich_cost if s.enrich_cost > 0 else None,
                rate=s.enrich_rate,
                primary_text=s.enrich_path,
                description=s.enrich_description,
                is_complete=enrich_complete
                or (s.enrich_completed >= s.enrich_total and s.enrich_total > 0),
                disabled=not self.enrich,
                disabled_msg="--no-enrich",
                worker_count=enrich_count,
                worker_annotation=enrich_ann,
            ),
        ]
        return build_pipeline_section(rows, self.bar_width)

    def _build_resources_section(self) -> Text:
        s = self.state

        # Compute ETC
        etc = None
        if s.enrich_cost > 0 and s.enrich_completed > 0 and s.enrich_total > 0:
            cost_per_node = s.enrich_cost / s.enrich_completed
            remaining = s.enrich_total - s.enrich_completed
            if remaining > 0:
                etc = s.enrich_cost + cost_per_node * remaining

        stats: list[tuple[str, str, str]] = [
            ("versions", f"{s.extract_completed}/{s.extract_total}", "blue"),
            ("nodes", f"{s.extract_nodes:,}", "blue"),
            ("units", str(s.units_found), "cyan"),
        ]
        if self.enrich:
            stats.append(("enriched", str(s.enrich_completed), "magenta"))

        pending: list[tuple[str, int]] = []
        if s.enrich_total > s.enrich_completed:
            pending.append(("enrich", s.enrich_total - s.enrich_completed))

        config = ResourceConfig(
            elapsed=s.elapsed,
            eta=s.eta_seconds,
            run_cost=s.enrich_cost if self.enrich else None,
            cost_limit=self.cost_limit if self.enrich else None,
            accumulated_cost=s.accumulated_cost if self.enrich else 0.0,
            etc=etc,
            stats=stats,
            pending=pending,
        )
        return build_resource_section(config, self.gauge_width)

    def tick(self) -> None:
        """Drain streaming queues into display state."""
        s = self.state

        item = self.extract_queue.pop()
        if item:
            s.extract_version = item.get("version", "")
            s.extract_detail = item.get("detail", "")
        elif self.extract_queue.is_stale() and (
            s.extract_completed >= s.extract_total and s.extract_total > 0
        ):
            s.extract_version = ""
            s.extract_detail = ""

        item = self.units_queue.pop()
        if item:
            s.units_path = item.get("path", "")
            s.units_detail = item.get("detail", "")
        elif self.units_queue.is_stale() and (
            s.units_completed >= s.units_total and s.units_total > 0
        ):
            s.units_path = ""
            s.units_detail = ""

        item = self.enrich_queue.pop()
        if item:
            s.enrich_path = item.get("path", "")
            s.enrich_description = item.get("description", "")
        elif self.enrich_queue.is_stale() and (
            s.enrich_completed >= s.enrich_total and s.enrich_total > 0
        ):
            s.enrich_path = ""
            s.enrich_description = ""

        self._refresh()

    def update_extract(self, msg: str, stats: WorkerStats, results: Any) -> None:
        """Callback from extract worker."""
        s = self.state
        if results:
            for r in results:
                if "node_count" in r:
                    s.extract_completed = stats.processed
                    node_count = r.get("node_count", 0)
                    s.extract_nodes += node_count
                    # Compute nodes/s from cumulative nodes over elapsed time
                    elapsed = s.elapsed
                    if elapsed > 0:
                        s.extract_rate = s.extract_nodes / elapsed
                    self.extract_queue.add(
                        [
                            {
                                "version": f"v{r['version']} done — {node_count:,} nodes",
                                "detail": f"{r.get('tags', 0)} tags, {r.get('nodes_created', 0):,} ingested",
                            }
                        ]
                    )
                elif r.get("phase") == "extract":
                    self.extract_queue.add(
                        [
                            {
                                "version": f"v{r['version']} extracting...",
                                "detail": "SSH extraction in progress",
                            }
                        ]
                    )

    def update_units(self, msg: str, stats: WorkerStats, results: Any) -> None:
        """Callback from units worker."""
        s = self.state
        s.units_rate = stats.ema_rate
        if results:
            for r in results:
                s.units_completed = r.get("checked", stats.processed)
                s.units_total = r.get("total", s.units_total)
                s.units_found = r.get("found", s.units_found)
                batch = r.get("batch", "")
                self.units_queue.add(
                    [
                        {
                            "path": f"{s.units_completed:,}/{s.units_total:,} checked",
                            "detail": f"batch {batch}, {s.units_found} with units"
                            if batch
                            else f"{s.units_found} with units",
                        }
                    ]
                )

    def update_enrich(self, msg: str, stats: WorkerStats, results: Any) -> None:
        """Callback from enrich worker."""
        s = self.state
        s.enrich_rate = stats.ema_rate
        s.enrich_completed = stats.processed
        s.enrich_cost = stats.cost
        if results:
            for r in results:
                self.enrich_queue.add(
                    [
                        {
                            "path": r.get("path", ""),
                            "description": r.get("description", ""),
                        }
                    ]
                )

    def refresh_from_graph(self, facility: str, tree_name: str) -> None:
        """Refresh display state from graph statistics."""
        from imas_codex.discovery.static.graph_ops import get_static_discovery_stats

        stats = get_static_discovery_stats(facility, tree_name)
        s = self.state

        s.extract_total = stats.get("versions_total", s.extract_total)
        s.extract_completed = stats.get("versions_ingested", s.extract_completed)
        s.extract_nodes = stats.get("nodes_total", s.extract_nodes)
        # Pin node total when all versions are ingested
        if s.extract_completed >= s.extract_total and s.extract_total > 0:
            s.extract_nodes_total = s.extract_nodes
        s.enrich_total = stats.get("nodes_enrichable", s.enrich_total)
        s.enrich_completed = stats.get("nodes_enriched", s.enrich_completed)

    def print_summary(self) -> None:
        """Print final summary."""
        s = self.state
        summary = Text()
        summary.append(
            f"  {s.extract_completed} versions, {s.extract_nodes:,} nodes extracted",
            style="blue",
        )
        if s.units_found > 0:
            summary.append(f", {s.units_found} nodes with units", style="cyan")
        if self.enrich:
            summary.append(f", {s.enrich_completed} nodes enriched", style="magenta")
        summary.append(f"\n  Time: {format_time(s.elapsed)}", style="dim")
        if self.enrich and s.enrich_cost > 0:
            summary.append(f", Cost: ${s.enrich_cost:.2f}", style="dim")
        self._console.print(
            Panel(summary, title="Static Tree Discovery Complete", border_style="green")
        )


# =============================================================================
# CLI command
# =============================================================================


@click.command()
@click.argument("facility")
@click.option(
    "--tree", "-t", "tree_name", help="Static tree name (default: from config)"
)
@click.option(
    "--versions",
    help="Comma-separated version numbers (default: all from config)",
)
@click.option("--dry-run", is_flag=True, help="Preview without ingesting")
@click.option(
    "--timeout", type=int, default=600, help="SSH timeout per version in seconds"
)
@click.option(
    "--cost-limit",
    "-c",
    type=float,
    default=2.0,
    help="Maximum LLM spend in USD for enrichment",
)
@click.option(
    "--enrich/--no-enrich",
    default=True,
    help="Enable/disable LLM enrichment of tree nodes",
)
@click.option(
    "--batch-size",
    type=int,
    default=40,
    help="Nodes per LLM enrichment batch",
)
@click.option(
    "--extract-workers",
    type=int,
    default=1,
    help="Number of parallel extract workers (default: 1)",
)
@click.option(
    "--enrich-workers",
    type=int,
    default=1,
    help="Number of parallel enrich workers (default: 1)",
)
@click.option(
    "--force", is_flag=True, help="Re-extract all versions even if already in graph"
)
@click.option(
    "--time",
    "time_limit",
    type=int,
    default=None,
    help="Maximum runtime in minutes",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output except errors")
def static(
    facility: str,
    tree_name: str | None,
    versions: str | None,
    dry_run: bool,
    timeout: int,
    cost_limit: float,
    enrich: bool,
    batch_size: int,
    extract_workers: int,
    enrich_workers: int,
    force: bool,
    time_limit: int | None,
    verbose: bool,
    quiet: bool,
) -> None:
    """Discover and ingest static/machine-description MDSplus trees.

    FACILITY is the SSH host alias (e.g., "tcv").

    Static trees contain time-invariant constructional data versioned by
    machine configuration changes. This command extracts tree structure,
    tags, and metadata, then ingests them into the knowledge graph as
    TreeModelVersion and TreeNode nodes.

    \b
    Pipeline (parallel workers):
      EXTRACT  Claim version, SSH extract, ingest to graph
      UNITS    Batched unit extraction for NUMERIC/SIGNAL nodes
      ENRICH   LLM batch descriptions of tree node physics

    \b
    Examples:
      imas-codex discover static tcv
      imas-codex discover static tcv --tree static --versions 1,3,8
      imas-codex discover static tcv --dry-run -v
      imas-codex discover static tcv --no-enrich
      imas-codex discover static tcv --extract-workers 2 --time 10
    """
    from imas_codex.cli.logging import configure_cli_logging
    from imas_codex.cli.rich_output import should_use_rich
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.mdsplus.static import get_static_tree_config

    use_rich = should_use_rich()
    configure_cli_logging("static", facility=facility, verbose=verbose)

    if use_rich and not quiet:
        console = Console()
    else:
        console = None

    static_logger = logging.getLogger("imas_codex.mdsplus.static")

    def log_print(msg: str) -> None:
        clean_msg = re.sub(r"\[[^\]]+\]", "", msg)
        if console:
            console.print(msg)
        else:
            static_logger.info(clean_msg)

    # Load static tree configs
    configs = get_static_tree_config(facility)
    if not configs:
        log_print(
            f"[red]No static_trees configured for {facility}.[/red] "
            f"Add static_trees to data_sources.mdsplus in the facility YAML.",
        )
        raise SystemExit(1)

    # Filter to specific tree if requested
    if tree_name:
        configs = [c for c in configs if c.get("tree_name") == tree_name]
        if not configs:
            log_print(
                f"[red]No static tree named '{tree_name}' in {facility} config[/red]"
            )
            raise SystemExit(1)

    # Load facility config for ssh_host
    try:
        facility_config = get_facility(facility)
    except Exception as e:
        log_print(f"[red]Error loading facility config: {e}[/red]")
        raise SystemExit(1) from e

    ssh_host = facility_config.get("ssh_host")
    if not ssh_host:
        log_print(f"[red]No SSH host configured for {facility}[/red]")
        raise SystemExit(1)

    # Check Neo4j connectivity upfront (unless dry-run)
    if not dry_run:
        from imas_codex.graph import GraphClient

        try:
            with GraphClient() as client:
                client.query("RETURN 1")
            logger.info("Neo4j connection verified")
        except Exception as e:
            log_print(
                f"[red]Neo4j is not available:[/red] {e}\n"
                "Use --dry-run to skip ingestion.",
            )
            raise SystemExit(1) from e

    deadline: float | None = None
    if time_limit is not None:
        deadline = time.time() + (time_limit * 60)

    for cfg in configs:
        tname = cfg["tree_name"]
        ver_list = _parse_versions(versions, cfg)
        if not ver_list:
            ver_list = [1]

        if use_rich and not quiet:
            _run_with_rich(
                facility=facility,
                ssh_host=ssh_host,
                cfg=cfg,
                tname=tname,
                ver_list=ver_list,
                dry_run=dry_run,
                timeout=timeout,
                cost_limit=cost_limit,
                enrich=enrich,
                batch_size=batch_size,
                extract_workers=extract_workers,
                enrich_workers=enrich_workers,
                force=force,
                deadline=deadline,
                console=console,
            )
        else:
            _run_plain(
                facility=facility,
                ssh_host=ssh_host,
                cfg=cfg,
                tname=tname,
                ver_list=ver_list,
                dry_run=dry_run,
                timeout=timeout,
                cost_limit=cost_limit,
                enrich=enrich,
                batch_size=batch_size,
                extract_workers=extract_workers,
                enrich_workers=enrich_workers,
                force=force,
                deadline=deadline,
            )


# =============================================================================
# Plain mode (no rich progress)
# =============================================================================


def _run_plain(
    *,
    facility: str,
    ssh_host: str,
    cfg: dict,
    tname: str,
    ver_list: list[int],
    dry_run: bool,
    timeout: int,
    cost_limit: float,
    enrich: bool,
    batch_size: int,
    extract_workers: int,
    enrich_workers: int,
    force: bool,
    deadline: float | None,
) -> None:
    """Run static discovery with plain logging output."""
    from imas_codex.discovery.static.parallel import run_parallel_static_discovery

    logger.info("Processing static tree: %s:%s", facility, tname)
    logger.info("  Versions: %s", ver_list)
    if force:
        logger.info("  --force: re-extracting all versions")

    def log_extract(msg, stats, results=None):
        if msg != "idle":
            logger.info("EXTRACT: %s", msg)

    def log_units(msg, stats, results=None):
        if msg != "idle":
            logger.info("UNITS: %s", msg)

    def log_enrich(msg, stats, results=None):
        if msg != "idle":
            logger.info("ENRICH: %s", msg)

    result = asyncio.run(
        run_parallel_static_discovery(
            facility=facility,
            ssh_host=ssh_host,
            tree_name=tname,
            tree_config=cfg,
            ver_list=ver_list,
            cost_limit=cost_limit,
            timeout=timeout,
            batch_size=batch_size,
            enrich=enrich,
            force=force,
            num_extract_workers=extract_workers,
            num_enrich_workers=enrich_workers,
            deadline=deadline,
            on_extract_progress=log_extract,
            on_units_progress=log_units,
            on_enrich_progress=log_enrich,
            dry_run=dry_run,
        )
    )

    logger.info(
        "Complete: %d versions extracted, %d nodes enriched, $%.2f cost, %.0fs",
        result["versions_extracted"],
        result["nodes_enriched"],
        result["cost"],
        result["elapsed_seconds"],
    )


# =============================================================================
# Rich progress mode
# =============================================================================


def _run_with_rich(
    *,
    facility: str,
    ssh_host: str,
    cfg: dict,
    tname: str,
    ver_list: list[int],
    dry_run: bool,
    timeout: int,
    cost_limit: float,
    enrich: bool,
    batch_size: int,
    extract_workers: int,
    enrich_workers: int,
    force: bool,
    deadline: float | None,
    console: Console | None,
) -> None:
    """Run static discovery with rich progress display."""
    from imas_codex.cli.discover.common import create_discovery_monitor
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.discovery.static.parallel import run_parallel_static_discovery

    facility_config = get_facility(facility)

    service_monitor = create_discovery_monitor(
        facility_config,
        check_graph=True,
        check_embed=False,
        check_model=enrich,
        check_ssh=True,
        check_auth=False,
    )

    # Suppress noisy INFO during rich display
    for mod in (
        "imas_codex.mdsplus",
        "imas_codex.discovery.static",
    ):
        logging.getLogger(mod).setLevel(logging.WARNING)

    display = StaticProgressDisplay(
        facility=facility,
        cost_limit=cost_limit,
        console=console,
        enrich=enrich,
    )

    # Set initial version count
    display.state.extract_total = len(ver_list)

    with display:
        display.service_monitor = service_monitor

        async def run_with_display():
            await service_monitor.__aenter__()

            async def refresh_graph_state():
                while True:
                    try:
                        display.refresh_from_graph(facility, tname)
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

            def on_extract(msg, stats, results=None):
                display.update_extract(msg, stats, results)

            def on_units(msg, stats, results=None):
                display.update_units(msg, stats, results)

            def on_enrich(msg, stats, results=None):
                display.update_enrich(msg, stats, results)

            def on_worker_status(worker_group):
                display.update_worker_status(worker_group)

            try:
                return await run_parallel_static_discovery(
                    facility=facility,
                    ssh_host=ssh_host,
                    tree_name=tname,
                    tree_config=cfg,
                    ver_list=ver_list,
                    cost_limit=cost_limit,
                    timeout=timeout,
                    batch_size=batch_size,
                    enrich=enrich,
                    force=force,
                    num_extract_workers=extract_workers,
                    num_enrich_workers=enrich_workers,
                    deadline=deadline,
                    on_extract_progress=on_extract,
                    on_units_progress=on_units,
                    on_enrich_progress=on_enrich,
                    on_worker_status=on_worker_status,
                    service_monitor=service_monitor,
                    dry_run=dry_run,
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

        asyncio.run(run_with_display())

        # Final graph refresh for accurate summary
        try:
            display.refresh_from_graph(facility, tname)
        except Exception:
            pass

        display.print_summary()


# =============================================================================
# Shared helpers
# =============================================================================


def _parse_versions(versions: str | None, cfg: dict) -> list[int] | None:
    """Parse version list from CLI arg or config."""
    if versions:
        return [int(v.strip()) for v in versions.split(",")]
    ver_configs = cfg.get("versions", [])
    if ver_configs:
        return [v["version"] for v in ver_configs]
    return None

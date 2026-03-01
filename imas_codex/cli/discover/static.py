"""Static tree discovery: extract and ingest machine-description MDSplus trees.

Four-phase pipeline:
  EXTRACT → SSH to facility, walk MDSplus static tree versions
  UNITS   → Batched unit extraction for NUMERIC/SIGNAL nodes
  ENRICH  → LLM batch descriptions of tree node physics
  INGEST  → Write TreeModelVersion + TreeNode to Neo4j
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

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
    clip_text,
    format_time,
)

if TYPE_CHECKING:
    from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)


# =============================================================================
# Progress display
# =============================================================================


@dataclass
class StaticProgressState:
    """Mutable state for the static tree progress display."""

    start_time: float = field(default_factory=time.time)

    # EXTRACT phase — progress tracks nodes across all versions
    extract_versions_done: int = 0
    extract_versions_total: int = 1
    extract_nodes: int = 0  # cumulative nodes extracted
    extract_nodes_total: int = 0  # estimated total (grows as versions complete)
    extract_rate: float | None = None  # nodes/s
    extract_version: str = ""  # e.g. "v3 tcv:static"
    extract_detail: str = ""  # e.g. "47,976 nodes, 312 tags"

    # UNITS phase
    units_completed: int = 0
    units_total: int = 0
    units_found: int = 0
    units_rate: float | None = None
    units_path: str = ""  # current node path being checked
    units_detail: str = ""  # e.g. "NUMERIC nodes, batch 3/7"

    # ENRICH phase
    enrich_completed: int = 0
    enrich_total: int = 0
    enrich_rate: float | None = None
    enrich_path: str = ""  # current node path
    enrich_description: str = ""  # LLM-generated description
    enrich_cost: float = 0.0

    # INGEST phase
    ingest_completed: int = 0
    ingest_total: int = 0
    ingest_rate: float | None = None
    ingest_label: str = ""  # e.g. "v3 — TreeModelVersion + 47,976 TreeNodes"
    ingest_detail: str = ""  # e.g. "MERGE batch 2/5"

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time


class StaticProgressDisplay(BaseProgressDisplay):
    """Rich progress display for static tree discovery.

    Shows four pipeline phases with streaming per-item activity:
      EXTRACT  version name, node count, tags
      UNITS    batch progress, nodes with units found
      ENRICH   node path, LLM description, cost
      INGEST   batch writes, versions + nodes created
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
        self.ingest_queue = StreamQueue(rate=2.0, max_rate=4.0, min_display_time=0.3)
        self._console = console or Console()
        self._extract_start: float | None = None
        self._units_start: float | None = None
        self._enrich_start: float | None = None
        self._ingest_start: float | None = None

        super().__init__(
            facility=facility,
            cost_limit=cost_limit,
            console=self._console,
            title_suffix="Static Tree Discovery",
        )

    def _build_pipeline_section(self) -> Text:
        s = self.state

        rows = [
            PipelineRowConfig(
                name="EXTRACT",
                style="bold blue",
                completed=s.extract_nodes,
                total=max(s.extract_nodes_total, 1),
                rate=s.extract_rate,
                primary_text=s.extract_version,
                description=s.extract_detail,
                is_complete=s.extract_versions_done >= s.extract_versions_total
                and s.extract_versions_total > 0,
            ),
            PipelineRowConfig(
                name="UNITS",
                style="bold cyan",
                completed=s.units_completed,
                total=max(s.units_total, 1),
                rate=s.units_rate,
                primary_text=s.units_path,
                description=s.units_detail,
                is_complete=s.units_completed >= s.units_total and s.units_total > 0,
            ),
        ]
        rows.append(
            PipelineRowConfig(
                name="ENRICH",
                style="bold magenta",
                completed=s.enrich_completed,
                total=max(s.enrich_total, 1),
                cost=s.enrich_cost if s.enrich_cost > 0 else None,
                rate=s.enrich_rate,
                primary_text=s.enrich_path,
                description=s.enrich_description,
                is_complete=s.enrich_completed >= s.enrich_total and s.enrich_total > 0,
                disabled=not self.enrich,
                disabled_msg="--no-enrich",
            )
        )
        rows.append(
            PipelineRowConfig(
                name="INGEST",
                style="bold green",
                completed=s.ingest_completed,
                total=max(s.ingest_total, 1),
                rate=s.ingest_rate,
                primary_text=s.ingest_label,
                description=s.ingest_detail,
                is_complete=s.ingest_completed >= s.ingest_total and s.ingest_total > 0,
            ),
        )
        return build_pipeline_section(rows, self.bar_width)

    def _build_resources_section(self) -> Text:
        s = self.state
        stats: list[tuple[str, str, str]] = [
            (
                "versions",
                f"{s.extract_versions_done}/{s.extract_versions_total}",
                "blue",
            ),
            ("nodes", f"{s.extract_nodes:,}", "blue"),
            ("units", str(s.units_found), "cyan"),
        ]
        if self.enrich:
            stats.append(("enriched", str(s.enrich_completed), "magenta"))
        stats.append(("ingested", f"{s.ingest_completed:,}", "green"))

        config = ResourceConfig(
            elapsed=s.elapsed,
            run_cost=s.enrich_cost if self.enrich else None,
            cost_limit=self.cost_limit if self.enrich else None,
            accumulated_cost=s.enrich_cost,
            stats=stats,
        )
        return build_resource_section(config, self.gauge_width)

    def _phase_complete(self, completed: int, total: int) -> bool:
        """Check if a phase has finished all its work."""
        return completed >= total and total > 0

    def tick(self) -> None:
        """Drain streaming queues into display state.

        Only clears text on stale when the phase is complete — during
        active SSH calls the last message stays visible instead of
        flashing to idle after 3 seconds.
        """
        s = self.state

        item = self.extract_queue.pop()
        if item:
            s.extract_version = item.get("version", "")
            s.extract_detail = item.get("detail", "")
        elif self.extract_queue.is_stale() and self._phase_complete(
            s.extract_versions_done, s.extract_versions_total
        ):
            s.extract_version = ""
            s.extract_detail = ""

        item = self.units_queue.pop()
        if item:
            s.units_path = item.get("path", "")
            s.units_detail = item.get("detail", "")
        elif self.units_queue.is_stale() and self._phase_complete(
            s.units_completed, s.units_total
        ):
            s.units_path = ""
            s.units_detail = ""

        item = self.enrich_queue.pop()
        if item:
            s.enrich_path = item.get("path", "")
            s.enrich_description = item.get("description", "")
        elif self.enrich_queue.is_stale() and self._phase_complete(
            s.enrich_completed, s.enrich_total
        ):
            s.enrich_path = ""
            s.enrich_description = ""

        item = self.ingest_queue.pop()
        if item:
            s.ingest_label = item.get("label", "")
            s.ingest_detail = item.get("detail", "")
        elif self.ingest_queue.is_stale() and self._phase_complete(
            s.ingest_completed, s.ingest_total
        ):
            s.ingest_label = ""
            s.ingest_detail = ""

        self._refresh()

    def _update_rate(self, phase: str) -> None:
        """Update rate for a phase based on elapsed time."""
        s = self.state
        now = time.time()
        if phase == "extract" and self._extract_start:
            dt = now - self._extract_start
            if dt > 0 and s.extract_nodes > 0:
                s.extract_rate = s.extract_nodes / dt
        elif phase == "units" and self._units_start:
            dt = now - self._units_start
            if dt > 0 and s.units_completed > 0:
                s.units_rate = s.units_completed / dt
        elif phase == "enrich" and self._enrich_start:
            dt = now - self._enrich_start
            if dt > 0 and s.enrich_completed > 0:
                s.enrich_rate = s.enrich_completed / dt
        elif phase == "ingest" and self._ingest_start:
            dt = now - self._ingest_start
            if dt > 0 and s.ingest_completed > 0:
                s.ingest_rate = s.ingest_completed / dt

    def print_summary(self) -> None:
        """Print final summary after pipeline completes."""
        s = self.state
        summary = Text()
        summary.append(
            f"  {s.extract_versions_done} versions, {s.extract_nodes:,} nodes extracted",
            style="blue",
        )
        if s.units_found > 0:
            summary.append(f", {s.units_found} nodes with units", style="cyan")
        if self.enrich:
            summary.append(f", {s.enrich_completed} nodes enriched", style="magenta")
        summary.append(f", {s.ingest_completed:,} nodes ingested", style="green")
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
@click.option(
    "--values/--no-values",
    "extract_values",
    default=None,
    help="Extract numerical values (R/Z, matrices). Default: from config.",
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
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output except errors")
def static(
    facility: str,
    tree_name: str | None,
    versions: str | None,
    extract_values: bool | None,
    dry_run: bool,
    timeout: int,
    cost_limit: float,
    enrich: bool,
    batch_size: int,
    verbose: bool,
    quiet: bool,
) -> None:
    """Discover and ingest static/machine-description MDSplus trees.

    FACILITY is the SSH host alias (e.g., "tcv").

    Static trees contain time-invariant constructional data versioned by
    machine configuration changes. This command extracts tree structure,
    tags, metadata, and optionally numerical values, then ingests them
    into the knowledge graph as TreeModelVersion and TreeNode nodes.

    \b
    Pipeline phases:
      EXTRACT  SSH to facility, walk MDSplus static tree versions
      ENRICH   LLM batch descriptions of tree node physics
      INGEST   Write TreeModelVersion + TreeNode to Neo4j

    \b
    Examples:
      imas-codex discover static tcv
      imas-codex discover static tcv --values
      imas-codex discover static tcv --tree static --versions 1,3,8
      imas-codex discover static tcv --dry-run -v
      imas-codex discover static tcv --no-enrich
    """
    from imas_codex.cli.logging import configure_cli_logging
    from imas_codex.cli.rich_output import should_use_rich
    from imas_codex.mdsplus.static import (
        get_static_tree_config,
    )

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

    if use_rich and not quiet:
        _run_with_progress(
            facility=facility,
            configs=configs,
            versions=versions,
            extract_values=extract_values,
            dry_run=dry_run,
            timeout=timeout,
            cost_limit=cost_limit,
            enrich=enrich,
            batch_size=batch_size,
            console=console,
        )
    else:
        _run_plain(
            facility=facility,
            configs=configs,
            versions=versions,
            extract_values=extract_values,
            dry_run=dry_run,
            timeout=timeout,
            cost_limit=cost_limit,
            enrich=enrich,
            batch_size=batch_size,
        )


# =============================================================================
# Plain mode (no rich progress)
# =============================================================================


def _run_plain(
    *,
    facility: str,
    configs: list[dict],
    versions: str | None,
    extract_values: bool | None,
    dry_run: bool,
    timeout: int,
    cost_limit: float,
    enrich: bool,
    batch_size: int,
) -> None:
    """Run static discovery with plain logging output."""
    from concurrent.futures import Future, ThreadPoolExecutor

    from imas_codex.mdsplus.static import (
        discover_static_tree_version,
        extract_units_for_version,
        ingest_static_tree,
        merge_units_into_data,
        merge_version_results,
    )

    for cfg in configs:
        tname = cfg["tree_name"]
        logger.info("Processing static tree: %s:%s", facility, tname)

        ver_list = _parse_versions(versions, cfg)
        do_extract = (
            extract_values
            if extract_values is not None
            else cfg.get("extract_values", False)
        )
        if not ver_list:
            ver_list = [1]

        logger.info("  Versions: %s, Extract values: %s", ver_list, do_extract)

        # Phase 1: Extract — one version at a time
        version_results = []
        for ver in ver_list:
            logger.info("  Extracting v%d...", ver)
            try:
                data = discover_static_tree_version(
                    facility=facility,
                    tree_name=tname,
                    version=ver,
                    extract_values=do_extract,
                    timeout=timeout,
                )
                version_results.append(data)
            except Exception:
                logger.exception(
                    "Extraction failed for %s:%s v%d", facility, tname, ver
                )

        data = merge_version_results(version_results)
        if not data.get("versions"):
            logger.warning("No versions extracted for %s:%s", facility, tname)
            continue

        _log_version_summary(data, facility, tname)

        # Phase 2: Units + Enrichment (concurrent)
        # Units extraction runs in background thread with batched SSH calls.
        # Each batch processes ~5000 nodes with its own timeout (~90s).
        latest_version = max(
            int(v) for v in data["versions"] if "error" not in data["versions"][v]
        )

        def _plain_progress(checked: int, total: int, found: int) -> None:
            logger.info("  Units: %d/%d checked, %d found", checked, total, found)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="units")
        try:
            units_future: Future[dict[str, str]] = executor.submit(
                extract_units_for_version,
                facility,
                tname,
                latest_version,
                timeout=180,
                batch_size=5000,
                on_progress=_plain_progress,
            )
            logger.info(
                "  Units extraction started (v%d, batched, background)", latest_version
            )

            # LLM enrichment runs in main thread
            enrichment_results = []
            enrich_cost = 0.0
            if enrich and not dry_run:
                enrichment_results, enrich_cost = _run_enrichment(
                    data, facility, tname, cfg, cost_limit, batch_size
                )

            # Collect units result — each batch has its own 120s timeout,
            # so total wait = batches × 120s. 34k nodes / 5000 = 7 batches.
            units = units_future.result(timeout=7 * 180 + 60)
            if units:
                updated = merge_units_into_data(data, units)
                logger.info("  Merged units into %d nodes", updated)
        except Exception:
            logger.exception("  Units collection failed")
            units = {}
        finally:
            executor.shutdown(wait=False)

        # Phase 3: Ingest
        if not dry_run:
            from imas_codex.graph import GraphClient

            with GraphClient() as client:
                stats = ingest_static_tree(client, facility, data, dry_run=False)
                if enrichment_results:
                    _apply_enrichment(client, facility, tname, enrichment_results)
                logger.info(
                    "Ingested: %d versions, %d nodes, %d values",
                    stats["versions_created"],
                    stats["nodes_created"],
                    stats["values_stored"],
                )
        else:
            stats = ingest_static_tree(None, facility, data, dry_run=True)
            logger.info(
                "[DRY RUN] Would create: %d versions, %d nodes",
                stats["versions_created"],
                stats["nodes_created"],
            )


# =============================================================================
# Rich progress mode
# =============================================================================


def _run_pipeline(
    *,
    display: StaticProgressDisplay,
    facility: str,
    cfg: dict,
    tname: str,
    ver_list: list[int],
    do_extract: bool,
    dry_run: bool,
    timeout: int,
    cost_limit: float,
    enrich: bool,
    batch_size: int,
) -> None:
    """Execute the async static discovery pipeline inside a live display."""
    import asyncio

    asyncio.run(
        _async_pipeline(
            display=display,
            facility=facility,
            cfg=cfg,
            tname=tname,
            ver_list=ver_list,
            do_extract=do_extract,
            dry_run=dry_run,
            timeout=timeout,
            cost_limit=cost_limit,
            enrich=enrich,
            batch_size=batch_size,
        )
    )


async def _async_pipeline(
    *,
    display: StaticProgressDisplay,
    facility: str,
    cfg: dict,
    tname: str,
    ver_list: list[int],
    do_extract: bool,
    dry_run: bool,
    timeout: int,
    cost_limit: float,
    enrich: bool,
    batch_size: int,
) -> None:
    """Async pipeline with sequential extract and concurrent workers.

    Architecture:
    - EXTRACT runs versions sequentially for clear incremental progress
    - UNITS worker starts as soon as the latest version finishes
    - ENRICH worker starts as soon as any version yields enrichable nodes
    - INGEST runs after all extract + units + enrich complete
    """
    import asyncio

    from imas_codex.mdsplus.static import (
        async_discover_static_tree_version,
        async_extract_units_for_version,
        ingest_static_tree,
        merge_units_into_data,
        merge_version_results,
    )

    display._extract_start = time.time()
    display.state.extract_versions_total = len(ver_list)

    # Shared state between async workers
    version_results: list[dict] = []
    enrichable_nodes: list[dict] = []
    enrichable_lock = asyncio.Lock()
    enrichment_results: list = []
    enrichment_cost = 0.0
    units: dict[str, str] = {}
    latest_version = max(ver_list)
    latest_version_event = asyncio.Event()  # Set when max version is extracted

    # ── EXTRACT worker (one task per version) ─────────────────────────
    async def extract_version(ver: int) -> dict | None:
        ver_idx = ver_list.index(ver) + 1
        display.extract_queue.add(
            [
                {
                    "version": f"v{ver} {facility}:{tname} ({ver_idx}/{len(ver_list)})",
                    "detail": "connecting via SSH, walking MDSplus tree...",
                }
            ]
        )

        try:
            data = await async_discover_static_tree_version(
                facility=facility,
                tree_name=tname,
                version=ver,
                extract_values=do_extract,
                timeout=timeout,
            )
            version_results.append(data)

            ver_data = data.get("versions", {}).get(str(ver), {})
            if "error" in ver_data:
                display.extract_queue.add(
                    [
                        {
                            "version": f"v{ver} — error",
                            "detail": ver_data["error"][:60],
                        }
                    ]
                )
            else:
                nc = ver_data.get("node_count", 0)
                tags = len(ver_data.get("tags", {}))
                detail_parts = [f"{nc:,} nodes"]
                if tags:
                    detail_parts.append(f"{tags} tags")
                display.extract_queue.add(
                    [
                        {
                            "version": f"v{ver} done — {nc:,} nodes, {tags} tags",
                            "detail": f"version {ver_idx}/{len(ver_list)} complete",
                        }
                    ]
                )

                # Update node counts for progress bar
                display.state.extract_nodes += nc
                # Estimate total based on avg nodes per version
                avg = display.state.extract_nodes / (
                    display.state.extract_versions_done + 1
                )
                remaining = len(ver_list) - display.state.extract_versions_done - 1
                display.state.extract_nodes_total = int(
                    display.state.extract_nodes + avg * remaining
                )

                # Feed enrichable nodes as they arrive
                async with enrichable_lock:
                    new_nodes = _collect_enrichable_nodes(data)
                    seen = {n["path"] for n in enrichable_nodes}
                    for n in new_nodes:
                        if n["path"] not in seen:
                            enrichable_nodes.append(n)
                            seen.add(n["path"])

            display.state.extract_versions_done += 1
            display._update_rate("extract")

            # Signal units worker when the latest version completes
            if ver == latest_version:
                latest_version_event.set()

            return data

        except Exception:
            logger.exception("Extraction failed for %s:%s v%d", facility, tname, ver)
            display.extract_queue.add(
                [
                    {
                        "version": f"v{ver} — FAILED",
                        "detail": "extraction error, see log",
                    }
                ]
            )
            display.state.extract_versions_done += 1
            display._update_rate("extract")

            # If this was the latest version, still signal so units doesn't hang
            if ver == latest_version:
                latest_version_event.set()

            return None

    # ── UNITS worker ──────────────────────────────────────────────────
    async def units_worker() -> None:
        nonlocal units

        # Wait for the latest version to be extracted
        await latest_version_event.wait()

        display._units_start = time.time()
        display.units_queue.add(
            [
                {
                    "path": f"v{latest_version} — batched extraction",
                    "detail": "NUMERIC + SIGNAL nodes",
                }
            ]
        )

        def _on_progress(
            checked: int,
            total: int,
            found: int,
            _display: StaticProgressDisplay = display,
        ) -> None:
            _display.state.units_total = total
            _display.state.units_completed = checked
            _display.state.units_found = found
            _display._update_rate("units")
            batch_num = (checked // 5000) + 1
            total_batches = (total + 4999) // 5000
            _display.units_queue.add(
                [
                    {
                        "path": f"{checked:,}/{total:,} nodes checked",
                        "detail": f"batch {batch_num}/{total_batches}, {found} with units",
                    }
                ]
            )

        try:
            units = await async_extract_units_for_version(
                facility,
                tname,
                latest_version,
                timeout=180,
                batch_size=5000,
                on_progress=_on_progress,
            )
            if units:
                display.state.units_completed = display.state.units_total
                display.state.units_found = len(units)
                display.units_queue.add(
                    [
                        {
                            "path": f"{len(units)} paths with units",
                            "detail": "complete",
                        }
                    ]
                )
            else:
                display.state.units_completed = max(display.state.units_total, 1)
                display.units_queue.add([{"path": "no units found", "detail": ""}])
        except Exception:
            logger.exception("Units extraction failed")
            display.state.units_completed = max(display.state.units_total, 1)
            display.units_queue.add(
                [{"path": "failed", "detail": "partial results kept"}]
            )

    # ── ENRICH worker ─────────────────────────────────────────────────
    async def enrich_worker() -> None:
        nonlocal enrichment_results, enrichment_cost

        if not enrich or dry_run:
            return

        # Wait for at least one version to produce enrichable nodes
        while not enrichable_nodes:
            if display.state.extract_versions_done >= len(ver_list):
                return  # All done, nothing to enrich
            await asyncio.sleep(0.5)

        # Snapshot current enrichable nodes
        async with enrichable_lock:
            nodes_to_enrich = list(enrichable_nodes)

        display.state.enrich_total = len(nodes_to_enrich)

        if nodes_to_enrich:
            # Run in thread to avoid blocking the event loop (LLM calls are sync)
            enrichment_results, enrichment_cost = await asyncio.to_thread(
                _run_enrichment_with_progress,
                nodes_to_enrich,
                facility,
                tname,
                cfg,
                cost_limit,
                batch_size,
                display,
            )
            display.state.enrich_cost = enrichment_cost

    # ── Async ticker (replaces the threading ticker) ─────────────────
    async def _ticker() -> None:
        while True:
            display.tick()
            await asyncio.sleep(0.25)

    # ── EXTRACT sequencer ────────────────────────────────────────────
    async def sequential_extract() -> None:
        """Extract versions one at a time for clear incremental progress.

        Running all versions concurrently via SSH causes them to finish
        at roughly the same time (SSH multiplexing + MDSplus locking
        serialise them anyway), so the progress bar sits at 0% the
        whole time.  Sequential extraction lets the bar advance after
        each version and lets ENRICH start processing nodes from early
        versions while later ones are still extracting.
        """
        for ver in ver_list:
            await extract_version(ver)
        # Pin total to actual count so bar reaches 100%
        display.state.extract_nodes_total = display.state.extract_nodes

    # ── Run all workers concurrently ──────────────────────────────────
    ticker_task = asyncio.create_task(_ticker())

    all_tasks = [
        sequential_extract(),
        units_worker(),
        enrich_worker(),
    ]

    try:
        await asyncio.gather(*all_tasks, return_exceptions=True)

        # Merge all version results
        data = merge_version_results(version_results)
        if not data.get("versions"):
            display.print_summary()
            return

        total_nodes = sum(
            v.get("node_count", 0)
            for v in data["versions"].values()
            if "error" not in v
        )

        # Merge units into data
        if units:
            updated = merge_units_into_data(data, units)
            display.units_queue.add(
                [{"path": f"merged into {updated} nodes", "detail": "complete"}]
            )

        # ── Phase 4: INGEST (runs while ticker is alive) ─────────────
        if not dry_run:
            from imas_codex.graph import GraphClient

            n_versions = len(data["versions"])
            display.state.ingest_total = total_nodes + n_versions
            display._ingest_start = time.time()
            display.ingest_queue.add(
                [
                    {
                        "label": f"{n_versions} TreeModelVersions + {total_nodes:,} TreeNodes",
                        "detail": "writing to Neo4j...",
                    }
                ]
            )

            def _on_ingest_progress(written: int, total: int, detail: str) -> None:
                display.state.ingest_completed = n_versions + written
                display._update_rate("ingest")
                display.ingest_queue.add(
                    [
                        {
                            "label": f"{written:,}/{total:,} TreeNodes",
                            "detail": detail,
                        }
                    ]
                )

            # Run ingest in thread so ticker keeps refreshing the display
            def _do_ingest() -> dict[str, int]:
                with GraphClient() as client:
                    stats = ingest_static_tree(
                        client,
                        facility,
                        data,
                        dry_run=False,
                        on_progress=_on_ingest_progress,
                    )
                    if enrichment_results:
                        _apply_enrichment(client, facility, tname, enrichment_results)
                    return stats

            stats = await asyncio.to_thread(_do_ingest)
            display.state.ingest_completed = (
                stats["versions_created"] + stats["nodes_created"]
            )
            display.ingest_queue.add(
                [
                    {
                        "label": f"{stats['versions_created']} versions, {stats['nodes_created']:,} nodes",
                        "detail": "complete",
                    }
                ]
            )
        else:
            stats = ingest_static_tree(None, facility, data, dry_run=True)
            display.state.ingest_total = 1
            display.state.ingest_completed = 1
            display.ingest_queue.add(
                [
                    {
                        "label": f"{stats['versions_created']} versions, {stats['nodes_created']:,} nodes",
                        "detail": "DRY RUN — no writes",
                    }
                ]
            )

        display.print_summary()

    finally:
        ticker_task.cancel()
        try:
            await ticker_task
        except asyncio.CancelledError:
            pass


def _run_with_progress(
    *,
    facility: str,
    configs: list[dict],
    versions: str | None,
    extract_values: bool | None,
    dry_run: bool,
    timeout: int,
    cost_limit: float,
    enrich: bool,
    batch_size: int,
    console: Console | None,
) -> None:
    """Run static discovery with rich progress display."""
    for cfg in configs:
        tname = cfg["tree_name"]
        ver_list = _parse_versions(versions, cfg)
        do_extract = (
            extract_values
            if extract_values is not None
            else cfg.get("extract_values", False)
        )
        if not ver_list:
            ver_list = [1]

        display = StaticProgressDisplay(
            facility=facility,
            cost_limit=cost_limit,
            console=console,
            enrich=enrich,
        )

        with display:
            _run_pipeline(
                display=display,
                facility=facility,
                cfg=cfg,
                tname=tname,
                ver_list=ver_list,
                do_extract=do_extract,
                dry_run=dry_run,
                timeout=timeout,
                cost_limit=cost_limit,
                enrich=enrich,
                batch_size=batch_size,
            )


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


def _log_version_summary(data: dict, facility: str, tname: str) -> None:
    """Log version extraction summary."""
    for ver_str, ver_data in sorted(
        data.get("versions", {}).items(), key=lambda x: int(x[0])
    ):
        if "error" in ver_data:
            logger.warning("  v%s: %s", ver_str, ver_data["error"][:80])
        else:
            logger.info(
                "  v%s: %d nodes, %d tags",
                ver_str,
                ver_data.get("node_count", 0),
                len(ver_data.get("tags", {})),
            )


def _collect_enrichable_nodes(data: dict) -> list[dict]:
    """Collect unique enrichable nodes across all versions.

    Deduplicates by path, preferring nodes with more metadata.
    Only includes data-bearing nodes (NUMERIC, SIGNAL, AXIS, TEXT).
    """
    seen: dict[str, dict] = {}
    enrichable_types = {"NUMERIC", "SIGNAL", "AXIS", "TEXT"}

    for ver_data in data.get("versions", {}).values():
        if "error" in ver_data:
            continue
        for node in ver_data.get("nodes", []):
            path = node.get("path", "")
            node_type = node.get("node_type", "STRUCTURE")
            if node_type not in enrichable_types:
                continue
            # Keep the version with more metadata
            if path not in seen or _node_richness(node) > _node_richness(seen[path]):
                seen[path] = node

    return list(seen.values())


def _node_richness(node: dict) -> int:
    """Score how much metadata a node has (more = richer)."""
    score = 0
    if node.get("tags"):
        score += 2
    if node.get("units"):
        score += 1
    if node.get("description"):
        score += 1
    if node.get("dtype"):
        score += 1
    if node.get("shape"):
        score += 1
    if node.get("value") is not None or node.get("scalar_value") is not None:
        score += 1
    return score


def _run_enrichment(
    data: dict,
    facility: str,
    tname: str,
    cfg: dict,
    cost_limit: float,
    batch_size: int,
) -> tuple[list, float]:
    """Run LLM enrichment on extracted nodes (plain mode)."""
    from imas_codex.mdsplus.enrichment import enrich_static_nodes

    nodes = _collect_enrichable_nodes(data)
    if not nodes:
        logger.info("No enrichable nodes found")
        return [], 0.0

    logger.info(
        "Enriching %d nodes with LLM (batch_size=%d)...", len(nodes), batch_size
    )

    # Build version descriptions
    version_descs = _build_version_descriptions(cfg)

    results, cost = enrich_static_nodes(
        nodes=nodes,
        facility=facility,
        tree_name=tname,
        version_descriptions=version_descs,
        batch_size=batch_size,
    )

    logger.info("Enriched %d nodes ($%.4f)", len(results), cost)
    return results, cost


def _run_enrichment_with_progress(
    nodes: list[dict],
    facility: str,
    tname: str,
    cfg: dict,
    cost_limit: float,
    batch_size: int,
    display: StaticProgressDisplay,
) -> tuple[list, float]:
    """Run LLM enrichment with progress display updates."""
    from imas_codex.discovery.base.llm import call_llm_structured
    from imas_codex.mdsplus.enrichment import (
        StaticNodeBatch,
        _build_system_prompt,
        _build_user_prompt,
    )
    from imas_codex.settings import get_model

    model = get_model("language")
    system_prompt = _build_system_prompt(facility, tname)
    version_descs = _build_version_descriptions(cfg)
    all_results = []
    total_cost = 0.0
    display._enrich_start = time.time()

    for i in range(0, len(nodes), batch_size):
        if total_cost >= cost_limit:
            logger.info("Cost limit reached ($%.2f >= $%.2f)", total_cost, cost_limit)
            break

        batch = nodes[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(nodes) + batch_size - 1) // batch_size

        # Show what we're enriching
        first_path = batch[0].get("path", "?")
        display.enrich_queue.add(
            [
                {
                    "path": clip_text(first_path, 50),
                    "description": f"batch {batch_num}/{total_batches}, {len(batch)} nodes",
                }
            ]
        )

        user_prompt = _build_user_prompt(batch, version_descs)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            result, cost, _tokens = call_llm_structured(
                model=model,
                messages=messages,
                response_model=StaticNodeBatch,
            )
            parsed = cast(StaticNodeBatch, result)
            all_results.extend(parsed.results)
            total_cost += cost
            display.state.enrich_completed += len(parsed.results)
            display.state.enrich_cost = total_cost
            display._update_rate("enrich")
            # Stream individual results for per-node visibility
            for r in parsed.results:
                display.enrich_queue.add(
                    [
                        {
                            "path": clip_text(r.path, 50),
                            "description": clip_text(r.description, 60)
                            if r.description
                            else "",
                        }
                    ]
                )
        except Exception:
            logger.exception("Failed to enrich batch %d", batch_num)

    return all_results, total_cost


def _apply_enrichment(
    client: GraphClient,
    facility: str,
    tree_name: str,
    results: list,
) -> None:
    """Apply enrichment results to TreeNode nodes in Neo4j."""
    from imas_codex.mdsplus.ingestion import normalize_mdsplus_path

    updates = []
    for r in results:
        normalized = normalize_mdsplus_path(r.path)
        node_id = f"{facility}:{tree_name}:{normalized}"
        update = {"node_id": node_id, "description": r.description}
        if r.keywords:
            update["keywords"] = r.keywords
        if r.category:
            update["category"] = r.category
        updates.append(update)

    if not updates:
        return

    # Batch update
    batch_size = 200
    for i in range(0, len(updates), batch_size):
        batch = updates[i : i + batch_size]
        client.query(
            """
            UNWIND $updates AS u
            MATCH (n:TreeNode {id: u.node_id})
            SET n.description = u.description,
                n.keywords = u.keywords,
                n.category = u.category
            """,
            updates=batch,
        )

    logger.info("Applied enrichment to %d TreeNode nodes", len(updates))


def _build_version_descriptions(cfg: dict) -> dict[int, str]:
    """Build version→description map from config."""
    descs: dict[int, str] = {}
    for vc in cfg.get("versions", []):
        if "description" in vc:
            descs[vc["version"]] = vc["description"]
    return descs

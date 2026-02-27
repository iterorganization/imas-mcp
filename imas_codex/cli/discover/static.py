"""Static tree discovery: extract and ingest machine-description MDSplus trees.

Three-phase pipeline:
  EXTRACT → SSH to facility, walk MDSplus static tree versions
  ENRICH  → LLM batch descriptions of tree node physics
  INGEST  → Write TreeModelVersion + TreeNode to Neo4j
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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

    # EXTRACT phase
    extract_completed: int = 0
    extract_total: int = 1
    extract_activity: str = ""
    extract_description: str = ""

    # UNITS phase (concurrent with enrich)
    units_completed: int = 0
    units_total: int = 0
    units_found: int = 0
    units_activity: str = ""
    units_description: str = ""

    # ENRICH phase
    enrich_completed: int = 0
    enrich_total: int = 0
    enrich_activity: str = ""
    enrich_description: str = ""
    enrich_cost: float = 0.0

    # INGEST phase
    ingest_completed: int = 0
    ingest_total: int = 0
    ingest_activity: str = ""
    ingest_description: str = ""

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time


class StaticProgressDisplay(BaseProgressDisplay):
    """Rich progress display for static tree discovery."""

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
        self.extract_queue = StreamQueue(rate=1.0, min_display_time=0.8)
        self.units_queue = StreamQueue(rate=1.0, min_display_time=0.8)
        self.enrich_queue = StreamQueue(rate=1.5, min_display_time=0.5)
        self.ingest_queue = StreamQueue(rate=2.0, min_display_time=0.3)
        self._console = console or Console()

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
                completed=s.extract_completed,
                total=max(s.extract_total, 1),
                primary_text=s.extract_activity,
                description=s.extract_description,
                is_complete=s.extract_completed >= s.extract_total
                and s.extract_total > 0,
            ),
            PipelineRowConfig(
                name="UNITS",
                style="bold cyan",
                completed=s.units_completed,
                total=max(s.units_total, 1),
                primary_text=s.units_activity,
                description=s.units_description,
                is_complete=s.units_completed >= s.units_total and s.units_total > 0,
                disabled=s.units_total == 0,
                disabled_msg="waiting for extract",
            ),
        ]
        if self.enrich:
            rows.append(
                PipelineRowConfig(
                    name="ENRICH",
                    style="bold magenta",
                    completed=s.enrich_completed,
                    total=max(s.enrich_total, 1),
                    cost=s.enrich_cost if s.enrich_cost > 0 else None,
                    primary_text=s.enrich_activity,
                    description=s.enrich_description,
                    is_complete=s.enrich_completed >= s.enrich_total
                    and s.enrich_total > 0,
                    disabled=s.enrich_total == 0
                    and s.extract_completed < s.extract_total,
                    disabled_msg="waiting for extract",
                )
            )
        rows.append(
            PipelineRowConfig(
                name="INGEST",
                style="bold green",
                completed=s.ingest_completed,
                total=max(s.ingest_total, 1),
                primary_text=s.ingest_activity,
                description=s.ingest_description,
                is_complete=s.ingest_completed >= s.ingest_total and s.ingest_total > 0,
                disabled=s.ingest_total == 0,
                disabled_msg="waiting",
            ),
        )
        return build_pipeline_section(rows, self.bar_width)

    def _build_resources_section(self) -> Text:
        s = self.state
        stats: list[tuple[str, str, str]] = [
            ("versions", str(s.extract_completed), "blue"),
            ("units", str(s.units_found), "cyan"),
            ("enriched", str(s.enrich_completed), "magenta"),
            ("ingested", str(s.ingest_completed), "green"),
        ]
        config = ResourceConfig(
            elapsed=s.elapsed,
            run_cost=s.enrich_cost if self.enrich else None,
            cost_limit=self.cost_limit if self.enrich else None,
            accumulated_cost=s.enrich_cost,
            stats=stats,
        )
        return build_resource_section(config, self.gauge_width)

    def tick(self) -> None:
        """Drain streaming queues into display state."""
        item = self.extract_queue.pop()
        if item:
            self.state.extract_activity = item.get("activity", "")
            self.state.extract_description = item.get("description", "")
        elif self.extract_queue.is_stale():
            self.state.extract_activity = ""
            self.state.extract_description = ""

        item = self.units_queue.pop()
        if item:
            self.state.units_activity = item.get("activity", "")
            self.state.units_description = item.get("description", "")
        elif self.units_queue.is_stale():
            self.state.units_activity = ""
            self.state.units_description = ""

        item = self.enrich_queue.pop()
        if item:
            self.state.enrich_activity = item.get("activity", "")
            self.state.enrich_description = item.get("description", "")
        elif self.enrich_queue.is_stale():
            self.state.enrich_activity = ""
            self.state.enrich_description = ""

        item = self.ingest_queue.pop()
        if item:
            self.state.ingest_activity = item.get("activity", "")
            self.state.ingest_description = item.get("description", "")
        elif self.ingest_queue.is_stale():
            self.state.ingest_activity = ""
            self.state.ingest_description = ""

    def print_summary(self) -> None:
        """Print final summary after pipeline completes."""
        s = self.state
        summary = Text()
        summary.append(f"  {s.extract_completed} versions extracted", style="blue")
        if s.units_found > 0:
            summary.append(f", {s.units_found} nodes with units", style="cyan")
        if self.enrich:
            summary.append(f", {s.enrich_completed} nodes enriched", style="magenta")
        summary.append(f", {s.ingest_completed} nodes ingested", style="green")
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
        ver_list = _parse_versions(versions, cfg)
        do_extract = (
            extract_values
            if extract_values is not None
            else cfg.get("extract_values", False)
        )
        if not ver_list:
            ver_list = [1]

        with StaticProgressDisplay(
            facility=facility,
            cost_limit=cost_limit,
            console=console,
            enrich=enrich,
        ) as display:
            display.state.extract_total = len(ver_list)

            # Phase 1: Extract — one version at a time
            version_results = []
            for ver in ver_list:
                display.extract_queue.add(
                    [
                        {
                            "activity": f"v{ver} — extracting...",
                            "description": f"{facility}:{tname}",
                        }
                    ]
                )
                display.tick()

                try:
                    data = discover_static_tree_version(
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
                                    "activity": f"v{ver} — error",
                                    "description": ver_data["error"][:60],
                                }
                            ]
                        )
                    else:
                        nc = ver_data.get("node_count", 0)
                        display.extract_queue.add(
                            [
                                {
                                    "activity": f"v{ver} — {nc:,} nodes",
                                    "description": "done",
                                }
                            ]
                        )
                    display.state.extract_completed += 1
                except Exception:
                    logger.exception(
                        "Extraction failed for %s:%s v%d", facility, tname, ver
                    )
                    display.extract_queue.add(
                        [
                            {
                                "activity": f"v{ver} — FAILED",
                                "description": "extraction error",
                            }
                        ]
                    )
                    display.state.extract_completed += 1

                display.tick()

            # Merge per-version results
            data = merge_version_results(version_results)
            if not data.get("versions"):
                display.print_summary()
                continue

            total_nodes = sum(
                v.get("node_count", 0)
                for v in data["versions"].values()
                if "error" not in v
            )

            # Phase 2: Units + Enrichment (concurrent)
            # Units run in background thread with batched SSH calls (~5000 nodes each).
            # LLM enrichment runs in main thread. Both are I/O-bound.
            latest_version = max(
                int(v) for v in data["versions"] if "error" not in data["versions"][v]
            )

            units_future: Future[dict[str, str]] | None = None
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="units")

            try:
                display.units_queue.add(
                    [
                        {
                            "activity": f"v{latest_version} — extracting units...",
                            "description": "batched, NUMERIC + SIGNAL nodes",
                        }
                    ]
                )
                display.tick()

                def _rich_progress(checked: int, total: int, found: int) -> None:
                    display.state.units_total = total
                    display.state.units_completed = checked
                    display.state.units_found = found
                    display.units_queue.add(
                        [
                            {
                                "activity": f"{checked:,}/{total:,} nodes checked",
                                "description": f"{found} with units",
                            }
                        ]
                    )

                units_future = executor.submit(
                    extract_units_for_version,
                    facility,
                    tname,
                    latest_version,
                    timeout=180,
                    batch_size=5000,
                    on_progress=_rich_progress,
                )

                # LLM enrichment in main thread (concurrent with units)
                enrichment_results = []
                if enrich and not dry_run:
                    enrich_nodes = _collect_enrichable_nodes(data)
                    display.state.enrich_total = len(enrich_nodes)

                    if enrich_nodes:
                        enrichment_results, enrich_cost = _run_enrichment_with_progress(
                            enrich_nodes,
                            facility,
                            tname,
                            cfg,
                            cost_limit,
                            batch_size,
                            display,
                        )
                        display.state.enrich_cost = enrich_cost

                # Collect units result — batched, so total wait = batches × timeout
                units = units_future.result(timeout=7 * 180 + 60)
                if units:
                    updated = merge_units_into_data(data, units)
                    display.state.units_completed = display.state.units_total
                    display.state.units_found = len(units)
                    display.units_queue.add(
                        [
                            {
                                "activity": f"{len(units)} paths with units",
                                "description": f"merged into {updated} nodes",
                            }
                        ]
                    )
                else:
                    display.state.units_completed = max(display.state.units_total, 1)
                    display.units_queue.add(
                        [{"activity": "no units found", "description": ""}]
                    )
                display.tick()
            except Exception:
                logger.exception("Units collection failed")
                display.state.units_completed = max(display.state.units_total, 1)
                display.units_queue.add(
                    [{"activity": "failed", "description": "partial results kept"}]
                )
                display.tick()
            finally:
                executor.shutdown(wait=False)

            # Phase 3: Ingest
            if not dry_run:
                from imas_codex.graph import GraphClient

                display.state.ingest_total = total_nodes + len(data["versions"])
                display.ingest_queue.add(
                    [{"activity": "Writing to Neo4j...", "description": ""}]
                )
                display.tick()

                with GraphClient() as client:
                    stats = ingest_static_tree(client, facility, data, dry_run=False)
                    display.state.ingest_completed = (
                        stats["versions_created"] + stats["nodes_created"]
                    )
                    if enrichment_results:
                        _apply_enrichment(client, facility, tname, enrichment_results)
                    display.ingest_queue.add(
                        [
                            {
                                "activity": "Complete",
                                "description": (
                                    f"{stats['versions_created']} versions, "
                                    f"{stats['nodes_created']} nodes"
                                ),
                            }
                        ]
                    )
                    display.tick()
            else:
                stats = ingest_static_tree(None, facility, data, dry_run=True)
                display.state.ingest_total = 1
                display.state.ingest_completed = 1
                display.ingest_queue.add(
                    [
                        {
                            "activity": "[DRY RUN]",
                            "description": (
                                f"{stats['versions_created']} versions, "
                                f"{stats['nodes_created']} nodes"
                            ),
                        }
                    ]
                )
                display.tick()

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
                    "activity": clip_text(first_path, 50),
                    "description": f"batch {batch_num}/{total_batches} ({len(batch)} nodes)",
                }
            ]
        )
        display.tick()

        user_prompt = _build_user_prompt(batch, version_descs)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            parsed, cost, _tokens = call_llm_structured(
                model=model,
                messages=messages,
                response_model=StaticNodeBatch,
            )
            all_results.extend(parsed.results)
            total_cost += cost
            display.state.enrich_completed += len(parsed.results)
            display.state.enrich_cost = total_cost
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

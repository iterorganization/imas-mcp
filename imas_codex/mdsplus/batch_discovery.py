"""Optimized MDSplus tree discovery using batched SSH and binary search.

Key optimizations:
1. Batch multiple shots per SSH connection (50 shots/query)
2. Binary search to find exact epoch boundaries
3. Checkpoint/resume support via JSON file
4. Incremental updates from existing graph state
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)

# Remote script template for batch structure queries
BATCH_QUERY_SCRIPT = """
import json
import MDSplus

tree_name = "{tree_name}"
shots = {shots}

results = {{}}
for shot in shots:
    try:
        t = MDSplus.Tree(tree_name, shot)
        paths = sorted(str(n.path) for n in t.getNodeWild("***"))
        # Use hash of sorted paths as structure fingerprint
        fingerprint = hash(tuple(paths))
        results[shot] = {{"count": len(paths), "fingerprint": fingerprint}}
    except Exception as e:
        results[shot] = {{"error": str(e)}}

print(json.dumps(results))
"""

# Remote script for getting full paths at a specific shot
FULL_PATHS_SCRIPT = """
import json
import MDSplus

tree_name = "{tree_name}"
shot = {shot}

try:
    t = MDSplus.Tree(tree_name, shot)
    paths = sorted(str(n.path) for n in t.getNodeWild("***"))
    print(json.dumps({{"paths": paths}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""


@dataclass
class DiscoveryCheckpoint:
    """Checkpoint state for resumable discovery."""

    facility: str
    tree_name: str
    current_shot: int
    start_shot: int
    coarse_step: int
    # Structure fingerprints: shot -> (count, fingerprint)
    structures: dict[int, tuple[int, int]] = field(default_factory=dict)
    # Discovered boundaries: list of (low_shot, high_shot) pairs
    boundaries: list[tuple[int, int]] = field(default_factory=list)
    # Refined boundaries: shot where structure changed
    refined_boundaries: list[int] = field(default_factory=list)
    # Phase: "coarse", "refine", "complete"
    phase: str = "coarse"

    def save(self, path: Path) -> None:
        """Save checkpoint to JSON file."""
        data = {
            "facility": self.facility,
            "tree_name": self.tree_name,
            "current_shot": self.current_shot,
            "start_shot": self.start_shot,
            "coarse_step": self.coarse_step,
            "structures": {str(k): v for k, v in self.structures.items()},
            "boundaries": self.boundaries,
            "refined_boundaries": self.refined_boundaries,
            "phase": self.phase,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.debug(f"Saved checkpoint to {path}")

    @classmethod
    def load(cls, path: Path) -> "DiscoveryCheckpoint":
        """Load checkpoint from JSON file."""
        data = json.loads(path.read_text())
        return cls(
            facility=data["facility"],
            tree_name=data["tree_name"],
            current_shot=data["current_shot"],
            start_shot=data["start_shot"],
            coarse_step=data["coarse_step"],
            structures={int(k): tuple(v) for k, v in data["structures"].items()},
            boundaries=[tuple(b) for b in data["boundaries"]],
            refined_boundaries=data["refined_boundaries"],
            phase=data["phase"],
        )


class BatchDiscovery:
    """Optimized MDSplus tree discovery with batching and binary search."""

    def __init__(
        self,
        facility: str,
        tree_name: str,
        shot_tree: str = "tcv_shot",
        batch_size: int = 50,
        ssh_timeout: int = 300,
    ):
        self.facility = facility
        self.tree_name = tree_name
        self.shot_tree = shot_tree
        self.batch_size = batch_size
        self.ssh_timeout = ssh_timeout

    def get_current_shot(self) -> int:
        """Get current shot number from facility."""
        script = f'''
import MDSplus
t = MDSplus.Tree("{self.shot_tree}", -1)
print(t.getCurrent())
'''
        result = self._run_ssh(script, timeout=30)
        return int(result.strip())

    def batch_query_structures(
        self, shots: list[int]
    ) -> dict[int, tuple[int, int] | None]:
        """Query structure fingerprints for multiple shots in one SSH call.

        Returns dict of shot -> (count, fingerprint) or None for errors.
        """
        if not shots:
            return {}

        script = BATCH_QUERY_SCRIPT.format(
            tree_name=self.tree_name,
            shots=shots,
        )

        try:
            result = self._run_ssh(script)
            data = json.loads(result)

            structures = {}
            for shot_str, info in data.items():
                shot = int(shot_str)
                if "error" in info:
                    logger.debug(f"Shot {shot}: {info['error']}")
                    structures[shot] = None
                else:
                    structures[shot] = (info["count"], info["fingerprint"])
            return structures

        except Exception as e:
            logger.warning(f"Batch query failed: {e}")
            return dict.fromkeys(shots)

    def get_full_paths(self, shot: int) -> list[str] | None:
        """Get full list of paths at a specific shot."""
        script = FULL_PATHS_SCRIPT.format(
            tree_name=self.tree_name,
            shot=shot,
        )

        try:
            result = self._run_ssh(script)
            data = json.loads(result)
            if "error" in data:
                logger.warning(f"Shot {shot}: {data['error']}")
                return None
            return data["paths"]
        except Exception as e:
            logger.warning(f"Failed to get paths for shot {shot}: {e}")
            return None

    def binary_search_boundary(
        self,
        low_shot: int,
        high_shot: int,
        low_fingerprint: int,
        high_fingerprint: int,
    ) -> int | None:
        """Find exact shot where structure changed using binary search.

        Returns the first shot with high_fingerprint structure.
        Handles gaps (missing shots) by probing nearby shots in batches.
        """
        logger.debug(f"Binary search: {low_shot} -> {high_shot}")

        failed_shots: set[int] = set()
        max_probe_attempts = 20

        while high_shot - low_shot > 1:
            mid_shot = (low_shot + high_shot) // 2

            # Skip known failed shots - find nearest untried
            if mid_shot in failed_shots:
                found = False
                for offset in range(1, (high_shot - low_shot) // 2 + 1):
                    for candidate in [mid_shot + offset, mid_shot - offset]:
                        if (
                            low_shot < candidate < high_shot
                            and candidate not in failed_shots
                        ):
                            mid_shot = candidate
                            found = True
                            break
                    if found:
                        break
                if not found:
                    logger.warning(
                        f"Cannot resolve boundary {low_shot}-{high_shot}: "
                        f"too many failed shots ({len(failed_shots)})"
                    )
                    return None

            result = self.batch_query_structures([mid_shot])
            mid_info = result.get(mid_shot)

            if mid_info is None:
                failed_shots.add(mid_shot)
                # Batch probe nearby shots for efficiency
                candidates = []
                for offset in range(1, max_probe_attempts + 1):
                    for candidate in [mid_shot + offset, mid_shot - offset]:
                        if (
                            candidate not in failed_shots
                            and low_shot < candidate < high_shot
                        ):
                            candidates.append(candidate)
                        if len(candidates) >= 10:
                            break
                    if len(candidates) >= 10:
                        break

                if not candidates:
                    logger.warning(
                        f"Cannot resolve boundary {low_shot}-{high_shot}: "
                        f"no candidate shots available"
                    )
                    return None

                # Query all candidates in one batch
                probe_result = self.batch_query_structures(candidates)
                found_valid = False
                # Check candidates in order (closest to mid first)
                for candidate in candidates:
                    probe_info = probe_result.get(candidate)
                    if probe_info is not None:
                        mid_shot = candidate
                        mid_info = probe_info
                        found_valid = True
                        break
                    failed_shots.add(candidate)

                if not found_valid:
                    logger.warning(
                        f"Cannot resolve boundary {low_shot}-{high_shot}: "
                        f"no valid shots found near {(low_shot + high_shot) // 2}"
                    )
                    return None

            mid_fingerprint = mid_info[1]

            if mid_fingerprint == low_fingerprint:
                low_shot = mid_shot
            else:
                high_shot = mid_shot

        return high_shot

    def _run_ssh(self, script: str, timeout: int | None = None) -> str:
        """Execute Python script on remote facility via SSH."""
        timeout = timeout or self.ssh_timeout
        # Escape single quotes in script
        escaped_script = script.replace("'", "'\"'\"'")
        cmd = ["ssh", self.facility, f"python3 -c '{escaped_script}'"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        return result.stdout


def discover_epochs_optimized(
    facility: str,
    tree_name: str,
    start_shot: int | None = None,
    end_shot: int | None = None,
    coarse_step: int = 1000,
    checkpoint_path: Path | None = None,
    client: "GraphClient | None" = None,
) -> tuple[list[dict], dict[int, list[str]]]:
    """Discover structural epochs using optimized batch + binary search.

    Args:
        facility: SSH host alias
        tree_name: MDSplus tree name
        start_shot: Start of scan (default: 3000)
        end_shot: End of scan (default: current shot)
        coarse_step: Step for initial coarse scan
        checkpoint_path: Path to save/load checkpoint
        client: Optional GraphClient for incremental mode

    Returns:
        Tuple of (epochs, representative_structures) where:
        - epochs: List of epoch dicts ready for ingestion
        - representative_structures: Dict of representative_shot -> paths
    """
    discovery = BatchDiscovery(facility, tree_name)

    # Load checkpoint if exists
    checkpoint = None
    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint = DiscoveryCheckpoint.load(checkpoint_path)
            logger.info(f"Resuming from checkpoint: phase={checkpoint.phase}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    # Get shot range
    if end_shot is None:
        end_shot = discovery.get_current_shot()
        logger.info(f"Current shot: {end_shot}")

    if start_shot is None:
        start_shot = 3000

    # Check graph for existing epochs (incremental mode)
    existing_max_shot = None
    if client is not None:
        existing = client.query(
            """
            MATCH (v:TreeModelVersion {tree_name: $tree, facility_id: $facility})
            RETURN min(v.first_shot) as min_shot, max(v.first_shot) as max_shot,
                   count(v) as epoch_count
            """,
            tree=tree_name,
            facility=facility,
        )
        if existing and existing[0]["min_shot"]:
            existing_min = existing[0]["min_shot"]
            existing_max = existing[0]["max_shot"]
            epoch_count = existing[0]["epoch_count"]
            existing_max_shot = existing_max
            logger.info(
                f"Graph has {epoch_count} epochs covering shots "
                f"{existing_min}-{existing_max}"
            )
            # Adjust start_shot for incremental: scan from where graph ends
            if start_shot < existing_max:
                logger.info(
                    f"Incremental mode: adjusting start from {start_shot} "
                    f"to {existing_max} (existing coverage)"
                )
                start_shot = existing_max

    # Initialize or resume checkpoint
    if checkpoint is None:
        checkpoint = DiscoveryCheckpoint(
            facility=facility,
            tree_name=tree_name,
            current_shot=end_shot,
            start_shot=start_shot,
            coarse_step=coarse_step,
        )

    # Phase 1: Coarse scan with batching
    if checkpoint.phase == "coarse":
        logger.info(
            f"Phase 1: Coarse scan from {end_shot} to {start_shot} (step={coarse_step})"
        )
        checkpoint = _coarse_scan(discovery, checkpoint, checkpoint_path)

    # Phase 2: Binary search refinement
    if checkpoint.phase == "refine":
        logger.info(f"Phase 2: Refining {len(checkpoint.boundaries)} boundaries")
        checkpoint = _refine_boundaries(discovery, checkpoint, checkpoint_path)

    # Phase 3: Build epoch records
    logger.info("Phase 3: Building epoch records")
    epochs, structures = _build_epoch_records(discovery, checkpoint)

    # Tag epochs as new vs existing for reporting
    if existing_max_shot is not None:
        for epoch in epochs:
            epoch["is_new"] = epoch["first_shot"] > existing_max_shot
    else:
        for epoch in epochs:
            epoch["is_new"] = True

    if checkpoint_path:
        checkpoint.phase = "complete"
        checkpoint.save(checkpoint_path)

    logger.info(f"Discovered {len(epochs)} epochs")
    return epochs, structures


def _coarse_scan(
    discovery: BatchDiscovery,
    checkpoint: DiscoveryCheckpoint,
    checkpoint_path: Path | None,
) -> DiscoveryCheckpoint:
    """Phase 1: Coarse scan to find approximate boundaries."""
    shots_to_scan = list(
        range(
            checkpoint.current_shot,
            checkpoint.start_shot - 1,
            -checkpoint.coarse_step,
        )
    )

    # Process in batches
    batch_size = discovery.batch_size
    for i in range(0, len(shots_to_scan), batch_size):
        batch = shots_to_scan[i : i + batch_size]
        logger.info(f"Scanning shots {batch[0]} to {batch[-1]}...")

        results = discovery.batch_query_structures(batch)

        for shot in batch:
            info = results.get(shot)
            if info is not None:
                checkpoint.structures[shot] = info

        checkpoint.current_shot = batch[-1]

        # Save checkpoint periodically
        if checkpoint_path and i % (batch_size * 5) == 0:
            checkpoint.save(checkpoint_path)

    # Find boundaries from coarse scan
    sorted_shots = sorted(checkpoint.structures.keys())
    for i in range(1, len(sorted_shots)):
        prev_shot = sorted_shots[i - 1]
        curr_shot = sorted_shots[i]
        prev_fp = checkpoint.structures[prev_shot][1]
        curr_fp = checkpoint.structures[curr_shot][1]

        if prev_fp != curr_fp:
            checkpoint.boundaries.append((prev_shot, curr_shot))
            logger.info(f"Found boundary: {prev_shot} -> {curr_shot}")

    checkpoint.phase = "refine"
    if checkpoint_path:
        checkpoint.save(checkpoint_path)

    return checkpoint


def _refine_boundaries(
    discovery: BatchDiscovery,
    checkpoint: DiscoveryCheckpoint,
    checkpoint_path: Path | None,
) -> DiscoveryCheckpoint:
    """Phase 2: Binary search to find exact boundaries."""
    for low_shot, high_shot in checkpoint.boundaries:
        low_fp = checkpoint.structures[low_shot][1]
        high_fp = checkpoint.structures[high_shot][1]

        boundary = discovery.binary_search_boundary(
            low_shot, high_shot, low_fp, high_fp
        )

        if boundary is not None:
            checkpoint.refined_boundaries.append(boundary)
            logger.info(f"Refined boundary: {boundary}")

        if checkpoint_path:
            checkpoint.save(checkpoint_path)

    checkpoint.refined_boundaries.sort()
    checkpoint.phase = "build"
    if checkpoint_path:
        checkpoint.save(checkpoint_path)

    return checkpoint


def _build_epoch_records(
    discovery: BatchDiscovery,
    checkpoint: DiscoveryCheckpoint,
) -> tuple[list[dict], dict[int, list[str]]]:
    """Phase 3: Build epoch records with full path lists."""
    epochs = []
    structures = {}

    # Get sorted list of all boundary shots
    sorted_shots = sorted(checkpoint.structures.keys())
    if not sorted_shots:
        return [], {}

    # Add refined boundaries to create epoch ranges
    boundary_shots = [sorted_shots[0]] + checkpoint.refined_boundaries

    version = 0
    prev_paths: set[str] = set()

    for i, first_shot in enumerate(boundary_shots):
        version += 1

        # Determine last_shot for this epoch
        if i + 1 < len(boundary_shots):
            last_shot = boundary_shots[i + 1] - 1
        else:
            last_shot = None  # Current epoch

        # Get full paths for this epoch's representative shot
        paths_list = discovery.get_full_paths(first_shot)
        if paths_list is None:
            # Try a nearby shot
            for offset in [1, 10, 100]:
                paths_list = discovery.get_full_paths(first_shot + offset)
                if paths_list:
                    break

        if paths_list is None:
            logger.warning(f"Could not get paths for epoch starting at {first_shot}")
            continue

        paths = set(paths_list)
        structures[first_shot] = paths_list

        # Calculate diff from previous epoch
        added = paths - prev_paths
        removed = prev_paths - paths

        # Get subtree names
        def get_subtree(path: str) -> str:
            parts = path.replace("\\", "").split("::")
            if len(parts) > 1:
                return parts[1].split(":")[0].split(".")[0]
            return "TOP"

        added_subtrees = sorted({get_subtree(p) for p in added})[:10]
        removed_subtrees = sorted({get_subtree(p) for p in removed})[:10]

        epoch = {
            "id": f"{checkpoint.facility}:{checkpoint.tree_name}:v{version}",
            "tree_name": checkpoint.tree_name,
            "facility_id": checkpoint.facility,
            "version": version,
            "first_shot": first_shot,
            "last_shot": last_shot,
            "node_count": len(paths),
            "nodes_added": len(added),
            "nodes_removed": len(removed),
            "added_subtrees": added_subtrees,
            "removed_subtrees": removed_subtrees,
            "added_paths": list(added),
            "removed_paths": list(removed),
        }

        if version > 1:
            epoch["predecessor"] = (
                f"{checkpoint.facility}:{checkpoint.tree_name}:v{version - 1}"
            )

        epochs.append(epoch)
        prev_paths = paths

        logger.info(
            f"Epoch v{version}: shots {first_shot}-{last_shot or 'current'}, "
            f"{len(paths)} nodes (+{len(added)}/-{len(removed)})"
        )

    return epochs, structures


def refine_boundaries(
    client: "GraphClient",
    facility: str,
    tree_name: str,
    dry_run: bool = False,
) -> dict[str, int]:
    """Refine rough epoch boundaries using binary search.

    Takes existing epochs (from sequential discovery with rough boundaries)
    and uses binary search to find exact shot where structure changed.

    Args:
        client: Neo4j GraphClient
        facility: SSH host alias
        tree_name: MDSplus tree name
        dry_run: If True, log but don't update

    Returns:
        Dict with counts: epochs_checked, boundaries_refined
    """
    # Get existing epochs sorted by first_shot
    result = client.query(
        """
        MATCH (v:TreeModelVersion {facility_id: $facility, tree_name: $tree})
        RETURN v.id as id, v.first_shot as first_shot
        ORDER BY v.first_shot
        """,
        facility=facility,
        tree=tree_name,
    )

    epochs = list(result)
    if len(epochs) < 2:
        logger.info("Not enough epochs to refine (need at least 2)")
        return {"epochs_checked": 0, "boundaries_refined": 0}

    logger.info(f"Checking {len(epochs) - 1} boundaries for refinement")

    discovery = BatchDiscovery(facility, tree_name)
    refined_count = 0

    for i in range(1, len(epochs)):
        prev = epochs[i - 1]
        curr = epochs[i]

        prev_shot = prev["first_shot"]
        curr_shot = curr["first_shot"]
        gap = curr_shot - prev_shot

        # Skip if already at shot-level precision (gap <= 1)
        if gap <= 1:
            logger.debug(f"Epoch {curr['id']}: already exact (gap={gap})")
            continue

        # Get fingerprints at boundaries
        fp_result = discovery.batch_query_structures([prev_shot, curr_shot])
        prev_info = fp_result.get(prev_shot)
        curr_info = fp_result.get(curr_shot)

        if prev_info is None or curr_info is None:
            logger.warning(f"Cannot query shots {prev_shot}/{curr_shot}, skipping")
            continue

        prev_fp = prev_info[1]
        curr_fp = curr_info[1]

        if prev_fp == curr_fp:
            logger.debug(f"Epoch {curr['id']}: fingerprints match, no change")
            continue

        # Binary search for exact boundary
        exact_shot = discovery.binary_search_boundary(
            prev_shot, curr_shot, prev_fp, curr_fp
        )

        if exact_shot is None:
            logger.warning(f"Could not refine boundary for {curr['id']}")
            continue

        if exact_shot != curr_shot:
            logger.info(
                f"Epoch {curr['id']}: refined {curr_shot} -> {exact_shot} "
                f"(was off by {curr_shot - exact_shot} shots)"
            )

            if not dry_run:
                client.query(
                    """
                    MATCH (v:TreeModelVersion {id: $id})
                    SET v.first_shot = $shot,
                        v.boundary_refined = true
                    """,
                    id=curr["id"],
                    shot=exact_shot,
                )
            refined_count += 1
        else:
            logger.debug(f"Epoch {curr['id']}: boundary already correct")

    result_dict = {
        "epochs_checked": len(epochs) - 1,
        "boundaries_refined": refined_count,
    }

    if refined_count > 0:
        logger.info(f"Refined {refined_count} boundaries")
    else:
        logger.info("All boundaries already at exact precision")

    return result_dict

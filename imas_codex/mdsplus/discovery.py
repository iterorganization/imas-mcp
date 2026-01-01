"""MDSplus tree structure discovery.

Provides facility-agnostic tree introspection via SSH.
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TreeDiscovery:
    """Configuration for MDSplus tree discovery at a facility.

    Attributes:
        facility: SSH host alias (e.g., "epfl", "iter")
        shot_tree: Tree name for getting current shot number (facility-specific)
        python_cmd: Python command on remote (default: "python3")
        ssh_timeout: Timeout for SSH commands in seconds
    """

    facility: str
    shot_tree: str = "tcv_shot"  # Override per facility
    python_cmd: str = "python3"
    ssh_timeout: int = 120

    # Cache of discovered structures: shot -> (count, paths)
    _structures: dict[int, tuple[int, frozenset[str]]] = field(
        default_factory=dict, repr=False
    )

    def get_current_shot(self) -> int:
        """Get the current shot number from the facility's shot tree."""
        script = f'''
import MDSplus
t = MDSplus.Tree("{self.shot_tree}", -1)
print(t.getCurrent())
'''
        result = self._run_remote(script, timeout=30)
        return int(result.strip())

    def get_tree_structure(
        self, tree_name: str, shot: int
    ) -> tuple[int, frozenset[str]] | tuple[None, None]:
        """Get tree structure at a specific shot.

        Args:
            tree_name: MDSplus tree name
            shot: Shot number

        Returns:
            Tuple of (node_count, frozenset of paths) or (None, None) on error
        """
        # Check cache first
        cache_key = (tree_name, shot)
        if cache_key in self._structures:
            return self._structures[cache_key]

        script = f'''
import json
import MDSplus
try:
    t = MDSplus.Tree("{tree_name}", {shot})
    paths = [str(n.path) for n in t.getNodeWild("***")]
    print(json.dumps({{"count": len(paths), "paths": paths}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
'''
        try:
            result = self._run_remote(script)
            data = json.loads(result)
            if "error" in data:
                logger.warning(f"Shot {shot}: {data['error']}")
                return None, None

            count = data["count"]
            paths = frozenset(data["paths"])
            self._structures[cache_key] = (count, paths)
            return count, paths

        except Exception as e:
            logger.warning(f"Shot {shot}: {e}")
            return None, None

    def get_node_details(
        self, tree_name: str, shot: int, paths: list[str] | None = None
    ) -> list[dict]:
        """Get detailed node information.

        Args:
            tree_name: MDSplus tree name
            shot: Shot number
            paths: Optional list of paths to query (default: all nodes)

        Returns:
            List of node dicts with path, node_type, units, description
        """
        paths_filter = ""
        if paths:
            # Escape paths for Python string
            paths_json = json.dumps(paths)
            paths_filter = f"filter_paths = {paths_json}"

        script = f'''
import json
import MDSplus

{paths_filter}

t = MDSplus.Tree("{tree_name}", {shot})
nodes = list(t.getNodeWild("***"))

# Filter if specified
if 'filter_paths' in dir():
    filter_set = set(filter_paths)
    nodes = [n for n in nodes if str(n.path) in filter_set]

result = []
for node in nodes:
    try:
        usage = str(node.usage)
        usage_map = {{
            "STRUCTURE": "STRUCTURE",
            "SIGNAL": "SIGNAL",
            "NUMERIC": "NUMERIC",
            "TEXT": "TEXT",
            "AXIS": "AXIS",
            "SUBTREE": "SUBTREE",
            "ACTION": "ACTION",
            "DISPATCH": "DISPATCH",
            "TASK": "TASK",
        }}
        node_type = usage_map.get(usage, "STRUCTURE")

        try:
            units = str(node.units) if hasattr(node, "units") else ""
        except:
            units = ""

        try:
            desc = str(node.node_name) if hasattr(node, "node_name") else ""
        except:
            desc = ""

        result.append({{
            "path": str(node.path),
            "node_type": node_type,
            "units": units or "dimensionless",
            "description": desc,
        }})
    except Exception:
        pass

print(json.dumps(result))
'''
        result = self._run_remote(script, timeout=300)
        return json.loads(result)

    def _run_remote(self, script: str, timeout: int | None = None) -> str:
        """Execute a Python script on the remote facility via SSH."""
        timeout = timeout or self.ssh_timeout
        cmd = ["ssh", self.facility, f"{self.python_cmd} -c '{script}'"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        return result.stdout


def get_tree_structure(
    facility: str, tree_name: str, shot: int, **kwargs
) -> tuple[int, frozenset[str]] | tuple[None, None]:
    """Convenience function for one-off structure queries."""
    discovery = TreeDiscovery(facility=facility, **kwargs)
    return discovery.get_tree_structure(tree_name, shot)


def discover_epochs(
    facility: str,
    tree_name: str,
    start_shot: int | None = None,
    end_shot: int | None = None,
    step: int = 500,
    **kwargs,
) -> tuple[list[dict], dict[int, tuple[int, frozenset[str]]]]:
    """Discover structural epochs by scanning shot history.

    Args:
        facility: SSH host alias (e.g., "epfl")
        tree_name: MDSplus tree name (e.g., "results")
        start_shot: Start of scan range (default: 3000)
        end_shot: End of scan range (default: current shot)
        step: Shot step for coarse scan
        **kwargs: Additional args for TreeDiscovery

    Returns:
        Tuple of (epochs list, structures dict) where:
        - epochs: List of epoch dicts with version, shot ranges, diffs
        - structures: Dict mapping shot -> (count, paths) for super tree building
    """
    discovery = TreeDiscovery(facility=facility, **kwargs)

    # Get current shot if not specified
    if end_shot is None:
        end_shot = discovery.get_current_shot()
        logger.info(f"Current shot: {end_shot}")

    if start_shot is None:
        start_shot = 3000  # Reasonable default for most facilities

    # Ensure we scan from high to low (recent to old)
    high_shot = max(start_shot, end_shot)
    low_shot = min(start_shot, end_shot)

    logger.info(f"Scanning {tree_name} from {high_shot} to {low_shot} (step={step})")

    # Phase 1: Scan structure at each step
    structures: dict[int, tuple[int, frozenset[str]]] = {}
    for shot in range(high_shot, low_shot - 1, -step):
        count, paths = discovery.get_tree_structure(tree_name, shot)
        if count is not None:
            structures[shot] = (count, paths)
            logger.debug(f"Shot {shot}: {count} nodes")
        else:
            logger.warning(f"Shot {shot}: failed to get structure")

    # Phase 2: Find boundaries
    sorted_shots = sorted(structures.keys())
    boundaries: list[tuple[int, int]] = []

    for i in range(1, len(sorted_shots)):
        prev_shot = sorted_shots[i - 1]
        curr_shot = sorted_shots[i]
        prev_count, prev_paths = structures[prev_shot]
        curr_count, curr_paths = structures[curr_shot]

        if prev_paths != curr_paths:
            boundaries.append((prev_shot, curr_shot))
            logger.info(
                f"Boundary: {prev_shot}->{curr_shot} ({prev_count}->{curr_count} nodes)"
            )

    # Phase 3: Build epochs
    epochs = _build_epochs(facility, tree_name, sorted_shots, boundaries, structures)

    logger.info(f"Discovered {len(epochs)} epochs")
    return epochs, structures


def _build_epochs(
    facility: str,
    tree_name: str,
    sorted_shots: list[int],
    boundaries: list[tuple[int, int]],
    structures: dict[int, tuple[int, frozenset[str]]],
) -> list[dict]:
    """Build epoch records from boundaries."""
    epochs = []
    version = 1

    def get_subtree(path: str) -> str:
        parts = path.replace("\\", "").split("::")
        if len(parts) > 1:
            return parts[1].split(":")[0].split(".")[0]
        return "TOP"

    # First epoch: earliest known structure
    if sorted_shots:
        first_shot = sorted_shots[0]
        first_count, first_paths = structures[first_shot]
        epochs.append(
            {
                "id": f"{facility}:{tree_name}:v{version}",
                "tree_name": tree_name,
                "facility_id": facility,
                "version": version,
                "first_shot": first_shot,
                "last_shot": None,
                "node_count": first_count,
                "nodes_added": first_count,
                "nodes_removed": 0,
                "added_subtrees": [],
                "removed_subtrees": [],
                "added_paths": list(first_paths),  # Store all paths for super tree
            }
        )

    # Subsequent epochs from boundaries
    for prev_shot, curr_shot in boundaries:
        if epochs:
            epochs[-1]["last_shot"] = prev_shot

        version += 1
        prev_count, prev_paths = structures[prev_shot]
        curr_count, curr_paths = structures[curr_shot]

        added = curr_paths - prev_paths
        removed = prev_paths - curr_paths

        added_subtrees = sorted({get_subtree(p) for p in added})
        removed_subtrees = sorted({get_subtree(p) for p in removed})

        epochs.append(
            {
                "id": f"{facility}:{tree_name}:v{version}",
                "tree_name": tree_name,
                "facility_id": facility,
                "version": version,
                "first_shot": curr_shot,
                "last_shot": None,
                "node_count": curr_count,
                "nodes_added": len(added),
                "nodes_removed": len(removed),
                "added_subtrees": added_subtrees[:10],
                "removed_subtrees": removed_subtrees[:10],
                "added_paths": list(added),  # Store for super tree
                "removed_paths": list(removed),
            }
        )

    # Set predecessor links
    for i, epoch in enumerate(epochs):
        if i > 0:
            epoch["predecessor"] = epochs[i - 1]["id"]

    return epochs

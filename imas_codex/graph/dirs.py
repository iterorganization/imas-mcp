"""Graph directory store — physical storage for graph instances.

Each graph lives in a name-based directory under
``~/.local/share/imas-codex/.neo4j/``::

    .neo4j/
        codex/                 ← graph name is the directory name
            data/              ← Neo4j data
            logs/
            conf/
            import/
        tcv/
            ...

The **active** graph is selected via a symlink::

    neo4j → .neo4j/codex

Switching graphs means repointing the symlink and restarting Neo4j.
The symlink target is transparent to Neo4j — it sees a regular directory.

All metadata (name, facilities) lives in the ``(:GraphMeta)`` node
inside the running database, not on disk.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Paths ──────────────────────────────────────────────────────────────────

DATA_BASE_DIR = Path.home() / ".local" / "share" / "imas-codex"
GRAPH_STORE = DATA_BASE_DIR / ".neo4j"
ACTIVE_LINK = DATA_BASE_DIR / "neo4j"

DEFAULT_GRAPH_NAME = "codex"


# ─── Data class ─────────────────────────────────────────────────────────────


@dataclass
class GraphDirInfo:
    """Metadata for a graph directory in the store."""

    name: str
    path: Path
    active: bool = False
    warnings: list[str] = field(default_factory=list)


# ─── Store management ──────────────────────────────────────────────────────


def ensure_graph_store() -> Path:
    """Create the ``.neo4j/`` store with secure permissions (700).

    Returns:
        Path to the store directory.
    """
    GRAPH_STORE.mkdir(parents=True, exist_ok=True)
    os.chmod(GRAPH_STORE, 0o700)
    return GRAPH_STORE


def create_graph_dir(name: str, *, force: bool = False) -> GraphDirInfo:
    """Create a new graph directory in the store.

    Creates ``.neo4j/<name>/`` with standard Neo4j subdirectories
    (``data/``, ``logs/``, ``conf/``, ``import/``).

    Args:
        name: Graph name (used as directory name).
        force: If True, allow using an existing directory.

    Returns:
        :class:`GraphDirInfo` for the new directory.

    Raises:
        FileExistsError: If a directory with this name already exists
            and ``force`` is False.
    """
    store = ensure_graph_store()
    graph_dir = store / name

    if graph_dir.exists() and not force:
        raise FileExistsError(
            f"Graph directory already exists: {graph_dir}\nUse --force to overwrite."
        )

    for subdir in ("data", "logs", "conf", "import"):
        (graph_dir / subdir).mkdir(parents=True, exist_ok=True)
    os.chmod(graph_dir, 0o700)

    logger.info("Created graph dir: %s", name)
    return GraphDirInfo(name=name, path=graph_dir)


def list_local_graphs() -> list[GraphDirInfo]:
    """List all graph directories in the store.

    Returns:
        Sorted list of :class:`GraphDirInfo` (by name).
    """
    if not GRAPH_STORE.exists():
        return []

    active_target = None
    if ACTIVE_LINK.is_symlink():
        try:
            active_target = ACTIVE_LINK.resolve()
        except OSError:
            pass

    results: list[GraphDirInfo] = []
    for entry in sorted(GRAPH_STORE.iterdir()):
        if not entry.is_dir():
            continue

        is_active = active_target is not None and entry.resolve() == active_target

        info = GraphDirInfo(
            name=entry.name,
            path=entry,
            active=is_active,
        )

        # Warn if missing Neo4j data subdirectory
        if not (entry / "data").exists():
            info.warnings.append("Missing data/ subdirectory")

        results.append(info)

    return sorted(results, key=lambda g: g.name)


def get_active_graph() -> GraphDirInfo | None:
    """Get info for the currently active graph (symlink target).

    Returns ``None`` if:
    - The ``neo4j/`` path doesn't exist
    - It's a real directory (legacy, not yet migrated)
    - The symlink target doesn't exist
    """
    if not ACTIVE_LINK.exists():
        return None

    if not ACTIVE_LINK.is_symlink():
        # Legacy: real directory, not symlinked yet
        return None

    target = ACTIVE_LINK.resolve()
    if not target.exists():
        return None

    return GraphDirInfo(name=target.name, path=target, active=True)


def is_legacy_data_dir() -> bool:
    """Check if ``neo4j/`` is a real directory (pre-migration)."""
    return ACTIVE_LINK.exists() and not ACTIVE_LINK.is_symlink()


def switch_active_graph(name: str) -> GraphDirInfo:
    """Repoint the ``neo4j/`` symlink to a different graph directory.

    Does NOT stop/start Neo4j — the caller is responsible for that.

    Args:
        name: Name of the target graph directory.

    Returns:
        :class:`GraphDirInfo` for the newly active graph.

    Raises:
        FileNotFoundError: Target directory doesn't exist.
        FileExistsError: ``neo4j/`` exists as a real directory (not symlink).
    """
    target = GRAPH_STORE / name
    if not target.exists():
        raise FileNotFoundError(
            f"No graph directory: {target}\n"
            f"Run 'imas-codex graph list' to see available graphs."
        )

    # Repoint symlink
    if ACTIVE_LINK.is_symlink():
        ACTIVE_LINK.unlink()
    elif ACTIVE_LINK.exists():
        raise FileExistsError(
            f"{ACTIVE_LINK} is a real directory, not a symlink.\n"
            "Manual migration needed: move contents into .neo4j/ first.\n"
            "See: imas-codex graph init --help"
        )

    ACTIVE_LINK.symlink_to(target)
    logger.info("Switched active graph to %s", name)
    return GraphDirInfo(name=name, path=target, active=True)


def find_graph(name: str) -> GraphDirInfo:
    """Find a graph by name.

    Args:
        name: Graph name (e.g. ``"codex"``).

    Returns:
        :class:`GraphDirInfo` for the matched graph.

    Raises:
        LookupError: No match found.
    """
    target = GRAPH_STORE / name
    if target.exists() and target.is_dir():
        active_target = None
        if ACTIVE_LINK.is_symlink():
            try:
                active_target = ACTIVE_LINK.resolve()
            except OSError:
                pass
        is_active = active_target is not None and target.resolve() == active_target
        return GraphDirInfo(name=name, path=target, active=is_active)

    raise LookupError(
        f"No graph found matching '{name}'.\n"
        "Run 'imas-codex graph list' to see available graphs."
    )


def delete_graph_dir(name: str) -> None:
    """Delete a graph directory from the store.

    Will NOT delete the active graph. Unlink symlink first.

    Args:
        name: Name of the graph directory to delete.

    Raises:
        FileNotFoundError: Directory doesn't exist.
        ValueError: Trying to delete the active graph.
    """
    import shutil

    target = GRAPH_STORE / name
    if not target.exists():
        raise FileNotFoundError(f"No graph directory: {target}")

    # Safety: don't delete active graph
    if ACTIVE_LINK.is_symlink() and ACTIVE_LINK.resolve() == target.resolve():
        raise ValueError(
            f"Cannot delete active graph '{name}'. Switch to another graph first."
        )

    shutil.rmtree(target)
    logger.info("Deleted graph dir: %s", name)


__all__ = [
    "ACTIVE_LINK",
    "DATA_BASE_DIR",
    "DEFAULT_GRAPH_NAME",
    "GRAPH_STORE",
    "GraphDirInfo",
    "create_graph_dir",
    "delete_graph_dir",
    "ensure_graph_store",
    "find_graph",
    "get_active_graph",
    "is_legacy_data_dir",
    "list_local_graphs",
    "switch_active_graph",
]

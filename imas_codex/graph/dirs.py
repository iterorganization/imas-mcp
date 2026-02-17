"""Graph directory store — physical storage for graph instances.

Each graph lives in a hash-named directory under ``~/.local/share/imas-codex/.neo4j/``::

    .neo4j/
        a3f8c1d09e2b/          ← SHA-256(name:fac1,fac2,...)[:12]
            data/              ← Neo4j data
            logs/
            conf/
            import/
            .meta.json         ← {name, facilities, hash, created_at}
        7c2e9f1a4b5d/
            ...

The **active** graph is selected via a symlink::

    neo4j → .neo4j/a3f8c1d09e2b

Switching graphs means repointing the symlink and restarting Neo4j.
The symlink target is transparent to Neo4j — it sees a regular directory.

Hash computation::

    hash = SHA-256(name + ":" + ",".join(sorted(facilities)))[:12]

The ``"imas"`` pseudo-facility is included in the facilities list when
the IMAS Data Dictionary is present (treated identically to real facilities
for hashing, ``--keep``/``--only`` in ``graph clear``, etc.).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Paths ──────────────────────────────────────────────────────────────────

DATA_BASE_DIR = Path.home() / ".local" / "share" / "imas-codex"
GRAPH_STORE = DATA_BASE_DIR / ".neo4j"
ACTIVE_LINK = DATA_BASE_DIR / "neo4j"


# ─── Hash ───────────────────────────────────────────────────────────────────


def compute_graph_hash(name: str, facilities: list[str]) -> str:
    """Deterministic hash of graph identity (name + facilities).

    Sorted, joined with ``,``, prefixed by ``name:``, SHA-256, first 12 hex
    chars.  This identifies the on-disk directory uniquely.

    Args:
        name: Graph name (e.g. ``"codex"``).
        facilities: Facility IDs (order-insensitive).  Include ``"imas"``
            when the Data Dictionary is loaded.

    Returns:
        12-char hex digest (e.g. ``"a3f8c1d09e2b"``).
    """
    canonical = name + ":" + ",".join(sorted(facilities))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ─── Metadata ───────────────────────────────────────────────────────────────

META_FILENAME = ".meta.json"


@dataclass
class GraphDirInfo:
    """Metadata for a graph directory in the store."""

    name: str
    facilities: list[str]
    hash: str
    path: Path
    created_at: str = ""
    warnings: list[str] = field(default_factory=list)


def write_dir_meta(
    graph_dir: Path,
    name: str,
    facilities: list[str],
    graph_hash: str,
) -> None:
    """Write ``.meta.json`` into a graph directory."""
    meta = {
        "name": name,
        "facilities": sorted(facilities),
        "hash": graph_hash,
        "created_at": datetime.now(UTC).isoformat(),
    }
    meta_path = graph_dir / META_FILENAME
    meta_path.write_text(json.dumps(meta, indent=2))
    os.chmod(meta_path, 0o600)


def read_dir_meta(graph_dir: Path) -> GraphDirInfo | None:
    """Read ``.meta.json`` from a graph directory.

    Returns ``None`` if the file is missing or corrupt.
    """
    meta_path = graph_dir / META_FILENAME
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Corrupt %s in %s: %s", META_FILENAME, graph_dir, exc)
        return None
    return GraphDirInfo(
        name=data.get("name", ""),
        facilities=data.get("facilities", []),
        hash=data.get("hash", ""),
        path=graph_dir,
        created_at=data.get("created_at", ""),
    )


# ─── Store management ──────────────────────────────────────────────────────


def ensure_graph_store() -> Path:
    """Create the ``.neo4j/`` store with secure permissions (700).

    Returns:
        Path to the store directory.
    """
    GRAPH_STORE.mkdir(parents=True, exist_ok=True)
    os.chmod(GRAPH_STORE, 0o700)
    return GRAPH_STORE


def create_graph_dir(name: str, facilities: list[str]) -> GraphDirInfo:
    """Create a new graph directory in the store.

    Creates ``.neo4j/<hash>/`` with standard Neo4j subdirectories
    (``data/``, ``logs/``, ``conf/``, ``import/``) and a ``.meta.json``.

    Args:
        name: Graph name.
        facilities: Facility IDs.

    Returns:
        :class:`GraphDirInfo` for the new directory.

    Raises:
        FileExistsError: If a directory with this hash already exists.
    """
    graph_hash = compute_graph_hash(name, facilities)
    store = ensure_graph_store()
    graph_dir = store / graph_hash

    if graph_dir.exists():
        raise FileExistsError(
            f"Graph directory already exists: {graph_dir}\n"
            f"  name={name}, facilities={sorted(facilities)}, hash={graph_hash}"
        )

    for subdir in ("data", "logs", "conf", "import"):
        (graph_dir / subdir).mkdir(parents=True)
    os.chmod(graph_dir, 0o700)

    write_dir_meta(graph_dir, name, facilities, graph_hash)
    logger.info(
        "Created graph dir: %s (name=%s, facilities=%s)", graph_hash, name, facilities
    )

    return GraphDirInfo(
        name=name,
        facilities=sorted(facilities),
        hash=graph_hash,
        path=graph_dir,
        created_at=datetime.now(UTC).isoformat(),
    )


def list_local_graphs() -> list[GraphDirInfo]:
    """List all graph directories in the store.

    Scans ``.neo4j/``, reads ``.meta.json`` from each subdirectory,
    validates the hash, and returns metadata with warnings for any
    hash drift.

    Returns:
        Sorted list of :class:`GraphDirInfo` (by name, then hash).
    """
    if not GRAPH_STORE.exists():
        return []

    results: list[GraphDirInfo] = []
    for entry in sorted(GRAPH_STORE.iterdir()):
        if not entry.is_dir():
            continue

        info = read_dir_meta(entry)
        if info is None:
            # Directory without metadata — report as unknown
            results.append(
                GraphDirInfo(
                    name="<unknown>",
                    facilities=[],
                    hash=entry.name,
                    path=entry,
                    warnings=[f"No {META_FILENAME} found"],
                )
            )
            continue

        # Validate hash matches
        expected = compute_graph_hash(info.name, info.facilities)
        if expected != entry.name:
            info.warnings.append(
                f"Hash drift: dir={entry.name}, "
                f"expected={expected} from name={info.name}, "
                f"facilities={info.facilities}"
            )

        results.append(info)

    return sorted(results, key=lambda g: (g.name, g.hash))


def get_active_graph() -> GraphDirInfo | None:
    """Get metadata for the currently active graph (symlink target).

    Returns ``None`` if:
    - The ``neo4j/`` path doesn't exist
    - It's a real directory (legacy, not yet migrated)
    - The symlink target has no ``.meta.json``
    """
    if not ACTIVE_LINK.exists():
        return None

    if not ACTIVE_LINK.is_symlink():
        # Legacy: real directory, not symlinked yet
        return None

    target = ACTIVE_LINK.resolve()
    return read_dir_meta(target)


def is_legacy_data_dir() -> bool:
    """Check if ``neo4j/`` is a real directory (pre-migration)."""
    return ACTIVE_LINK.exists() and not ACTIVE_LINK.is_symlink()


def switch_active_graph(graph_hash: str) -> GraphDirInfo:
    """Repoint the ``neo4j/`` symlink to a different graph directory.

    Does NOT stop/start Neo4j — the caller is responsible for that.

    Args:
        graph_hash: Hash of the target graph directory.

    Returns:
        :class:`GraphDirInfo` for the newly active graph.

    Raises:
        FileNotFoundError: Target directory doesn't exist.
        ValueError: Target has no metadata or hash mismatch.
        FileExistsError: ``neo4j/`` exists as a real directory (not symlink).
    """
    target = GRAPH_STORE / graph_hash
    if not target.exists():
        raise FileNotFoundError(f"No graph directory: {target}")

    info = read_dir_meta(target)
    if info is None:
        raise ValueError(f"No {META_FILENAME} in {target}")

    # Validate hash consistency
    expected = compute_graph_hash(info.name, info.facilities)
    if expected != graph_hash:
        raise ValueError(
            f"Hash mismatch: dir={graph_hash}, computed={expected} "
            f"from name={info.name}, facilities={info.facilities}"
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
    logger.info("Switched active graph to %s (name=%s)", graph_hash, info.name)
    return info


def find_graph(identifier: str) -> GraphDirInfo:
    """Find a graph by name or hash prefix.

    Tries exact name match first, then hash prefix match.

    Args:
        identifier: Graph name (e.g. ``"codex"``) or hash prefix
            (e.g. ``"a3f8"``).

    Returns:
        :class:`GraphDirInfo` for the matched graph.

    Raises:
        LookupError: No match or ambiguous match.
    """
    graphs = list_local_graphs()

    # Try exact name match
    by_name = [g for g in graphs if g.name == identifier]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        lines = [f"  {g.hash}  [{', '.join(g.facilities)}]" for g in by_name]
        raise LookupError(
            f"Multiple graphs named '{identifier}':\n"
            + "\n".join(lines)
            + "\nUse hash prefix to disambiguate."
        )

    # Try hash prefix match
    by_hash = [g for g in graphs if g.hash.startswith(identifier)]
    if len(by_hash) == 1:
        return by_hash[0]
    if len(by_hash) > 1:
        lines = [f"  {g.hash}  {g.name}  [{', '.join(g.facilities)}]" for g in by_hash]
        raise LookupError(f"Ambiguous hash prefix '{identifier}':\n" + "\n".join(lines))

    raise LookupError(
        f"No graph found matching '{identifier}'.\n"
        "Run 'imas-codex graph list' to see available graphs."
    )


def rename_graph_dir(
    old_hash: str,
    name: str,
    new_facilities: list[str],
) -> GraphDirInfo:
    """Rename a graph directory when its hash changes (facility add/remove).

    Computes the new hash from ``name`` + ``new_facilities``, renames the
    directory, updates ``.meta.json``, and repoints the symlink if this
    was the active graph.

    Args:
        old_hash: Current directory hash.
        name: Graph name (unchanged).
        new_facilities: Updated facility list.

    Returns:
        :class:`GraphDirInfo` for the renamed directory.

    Raises:
        FileNotFoundError: Source directory doesn't exist.
        FileExistsError: Target hash already exists.
    """
    new_hash = compute_graph_hash(name, new_facilities)

    if old_hash == new_hash:
        # Hash unchanged — just update metadata
        old_path = GRAPH_STORE / old_hash
        write_dir_meta(old_path, name, new_facilities, new_hash)
        return read_dir_meta(old_path)  # type: ignore[return-value]

    old_path = GRAPH_STORE / old_hash
    new_path = GRAPH_STORE / new_hash

    if not old_path.exists():
        raise FileNotFoundError(f"Source graph dir not found: {old_path}")
    if new_path.exists():
        raise FileExistsError(f"Target graph dir already exists: {new_path}")

    # Check if active graph before renaming
    was_active = (
        ACTIVE_LINK.is_symlink() and ACTIVE_LINK.resolve() == old_path.resolve()
    )

    old_path.rename(new_path)
    write_dir_meta(new_path, name, new_facilities, new_hash)

    # Update symlink if this was the active graph
    if was_active:
        ACTIVE_LINK.unlink()
        ACTIVE_LINK.symlink_to(new_path)

    logger.info("Renamed graph dir: %s → %s", old_hash, new_hash)

    info = read_dir_meta(new_path)
    if info is None:
        raise RuntimeError(f"Failed to read metadata after rename: {new_path}")
    return info


def validate_graph_dir(graph_dir: Path) -> list[str]:
    """Validate a graph directory's hash matches its metadata.

    Returns:
        List of warning messages (empty = valid).
    """
    warnings: list[str] = []
    info = read_dir_meta(graph_dir)
    if info is None:
        warnings.append(f"No {META_FILENAME} in {graph_dir}")
        return warnings

    expected = compute_graph_hash(info.name, info.facilities)
    dir_hash = graph_dir.name

    if expected != dir_hash:
        warnings.append(
            f"Hash drift: dir={dir_hash}, computed={expected} "
            f"from name={info.name}, facilities={info.facilities}"
        )

    return warnings


def migrate_legacy_dir(name: str, facilities: list[str]) -> GraphDirInfo:
    """Migrate a legacy ``neo4j/`` real directory to the ``.neo4j/`` store.

    Moves ``neo4j/`` → ``.neo4j/<hash>/``, writes ``.meta.json``,
    creates the ``neo4j → .neo4j/<hash>/`` symlink.

    Args:
        name: Graph name for the existing data.
        facilities: Facility list for the existing data.

    Returns:
        :class:`GraphDirInfo` for the migrated directory.

    Raises:
        FileNotFoundError: ``neo4j/`` doesn't exist.
        FileExistsError: Target hash directory already exists.
        ValueError: ``neo4j/`` is already a symlink.
    """
    if not ACTIVE_LINK.exists():
        raise FileNotFoundError(f"No data directory at {ACTIVE_LINK}")

    if ACTIVE_LINK.is_symlink():
        raise ValueError(f"{ACTIVE_LINK} is already a symlink (already migrated?)")

    graph_hash = compute_graph_hash(name, facilities)
    store = ensure_graph_store()
    target = store / graph_hash

    if target.exists():
        raise FileExistsError(
            f"Target directory already exists: {target}\n"
            f"  hash={graph_hash} for name={name}, facilities={facilities}"
        )

    # Move the real directory into the store
    ACTIVE_LINK.rename(target)
    write_dir_meta(target, name, facilities, graph_hash)

    # Create symlink
    ACTIVE_LINK.symlink_to(target)

    logger.info(
        "Migrated legacy dir to %s (name=%s, facilities=%s)",
        graph_hash,
        name,
        facilities,
    )
    return GraphDirInfo(
        name=name,
        facilities=sorted(facilities),
        hash=graph_hash,
        path=target,
        created_at=datetime.now(UTC).isoformat(),
    )


__all__ = [
    "ACTIVE_LINK",
    "DATA_BASE_DIR",
    "GRAPH_STORE",
    "GraphDirInfo",
    "compute_graph_hash",
    "create_graph_dir",
    "ensure_graph_store",
    "find_graph",
    "get_active_graph",
    "is_legacy_data_dir",
    "list_local_graphs",
    "migrate_legacy_dir",
    "read_dir_meta",
    "rename_graph_dir",
    "switch_active_graph",
    "validate_graph_dir",
    "write_dir_meta",
]

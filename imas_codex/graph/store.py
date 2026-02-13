"""Local graph store for fast switching between named graphs.

Graphs are stored as tar.gz archives under
``~/.local/share/imas-codex/graphs/``.  Each graph name has exactly one
stored copy (not versioned locally — use GHCR for version history).

The store enables ``graph switch`` to dump the current graph and load
another without re-fetching from GHCR every time.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STORE_DIR = Path.home() / ".local" / "share" / "imas-codex" / "graphs"
NEO4J_IMAGE = Path.home() / "apptainer" / "neo4j_2025.11-community.sif"


def _require_apptainer() -> None:
    if not shutil.which("apptainer"):
        msg = "apptainer not found in PATH"
        raise RuntimeError(msg)


def store_dir() -> Path:
    """Return (and create) the local graph store directory."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    return STORE_DIR


def dump_to_store(
    name: str,
    data_dir: Path,
    image: Path | None = None,
) -> Path:
    """Dump the current Neo4j graph to the local store.

    Requires Neo4j to be **stopped** before calling.

    Args:
        name: Graph name for the archive (e.g. ``"codex"``).
        data_dir: Neo4j data directory (contains ``data/``).
        image: Apptainer image path.  Defaults to standard location.

    Returns:
        Path to the created ``{name}.tar.gz`` in the store.
    """
    _require_apptainer()
    img = image or NEO4J_IMAGE
    target = store_dir() / f"{name}.tar.gz"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        archive_dir = tmp / name
        archive_dir.mkdir()

        dumps_dir = data_dir / "dumps"
        dumps_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "apptainer",
            "exec",
            "--bind",
            f"{data_dir}/data:/data",
            "--bind",
            f"{dumps_dir}:/dumps",
            "--writable-tmpfs",
            str(img),
            "neo4j-admin",
            "database",
            "dump",
            "neo4j",
            "--to-path=/dumps",
            "--overwrite-destination=true",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            msg = f"neo4j-admin dump failed: {result.stderr}"
            raise RuntimeError(msg)

        dump_file = dumps_dir / "neo4j.dump"
        if not dump_file.exists():
            msg = "neo4j-admin dump did not produce neo4j.dump"
            raise RuntimeError(msg)

        shutil.move(str(dump_file), str(archive_dir / "graph.dump"))

        manifest = {
            "name": name,
            "dumped_at": datetime.now(UTC).isoformat(),
        }
        (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        with tarfile.open(target, "w:gz") as tar:
            tar.add(archive_dir, arcname=name)

    logger.info("Dumped graph '%s' to store: %s", name, target)
    return target


def load_from_store(
    name: str,
    data_dir: Path,
    password: str = "imas-codex",
    image: Path | None = None,
) -> None:
    """Load a named graph from the local store into Neo4j.

    Requires Neo4j to be **stopped** before calling.  Resets the Neo4j
    password after load (the dump replaces the auth database).

    Args:
        name: Graph name to load.
        data_dir: Neo4j data directory.
        password: Password to set after load.
        image: Apptainer image path.

    Raises:
        FileNotFoundError: If *name* is not in the local store.
    """
    _require_apptainer()
    img = image or NEO4J_IMAGE
    archive = store_dir() / f"{name}.tar.gz"

    if not archive.exists():
        msg = f"Graph '{name}' not in local store. Run: imas-codex graph fetch {name}"
        raise FileNotFoundError(msg)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp)

        # Find graph.dump inside extracted directory
        extracted_dirs = list(tmp.iterdir())
        if not extracted_dirs:
            msg = f"Empty archive: {archive}"
            raise RuntimeError(msg)

        dump_file = None
        for d in extracted_dirs:
            candidate = d / "graph.dump"
            if candidate.exists():
                dump_file = candidate
                break

        if dump_file is None:
            msg = f"No graph.dump found in archive: {archive}"
            raise RuntimeError(msg)

        dumps_dir = data_dir / "dumps"
        dumps_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(dump_file), str(dumps_dir / "neo4j.dump"))

        # Load into Neo4j
        cmd = [
            "apptainer",
            "exec",
            "--bind",
            f"{data_dir}/data:/data",
            "--bind",
            f"{dumps_dir}:/dumps",
            "--writable-tmpfs",
            str(img),
            "neo4j-admin",
            "database",
            "load",
            "neo4j",
            "--from-path=/dumps",
            "--overwrite-destination=true",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            msg = f"neo4j-admin load failed: {result.stderr}"
            raise RuntimeError(msg)

        # Reset password (dump replaces auth DB)
        pw_cmd = [
            "apptainer",
            "exec",
            "--bind",
            f"{data_dir}/data:/data",
            "--writable-tmpfs",
            str(img),
            "neo4j-admin",
            "dbms",
            "set-initial-password",
            password,
        ]
        subprocess.run(pw_cmd, capture_output=True, text=True)

    logger.info("Loaded graph '%s' from store into %s", name, data_dir)


def list_store() -> list[dict[str, Any]]:
    """List graphs in the local store with metadata.

    Returns:
        List of dicts: ``{"name": str, "size_mb": float, "dumped_at": str}``.
    """
    graphs: list[dict[str, Any]] = []
    sd = store_dir()

    for path in sorted(sd.glob("*.tar.gz")):
        name = path.stem.removesuffix(".tar")  # "codex.tar" → "codex"
        if "." in name:
            name = path.name.removesuffix(".tar.gz")  # handle weird names
        entry: dict[str, Any] = {
            "name": name,
            "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
            "path": str(path),
        }

        # Try to read manifest from archive
        try:
            with tarfile.open(path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith("manifest.json"):
                        f = tar.extractfile(member)
                        if f:
                            manifest = json.loads(f.read().decode())
                            entry["dumped_at"] = manifest.get("dumped_at")
                            entry["facilities"] = manifest.get("facilities")
                        break
        except Exception:
            pass

        graphs.append(entry)

    return graphs


def graph_in_store(name: str) -> bool:
    """Check whether a named graph exists in the local store."""
    return (store_dir() / f"{name}.tar.gz").exists()


def remove_from_store(name: str) -> bool:
    """Delete a graph from the local store.

    Returns:
        True if the file was deleted, False if it didn't exist.
    """
    path = store_dir() / f"{name}.tar.gz"
    if path.exists():
        path.unlink()
        logger.info("Removed '%s' from local store", name)
        return True
    return False


__all__ = [
    "STORE_DIR",
    "dump_to_store",
    "graph_in_store",
    "list_store",
    "load_from_store",
    "remove_from_store",
    "store_dir",
]

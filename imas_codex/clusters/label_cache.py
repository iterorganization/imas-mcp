"""
Content-addressed label cache for LLM-generated cluster labels.

Uses SQLite for efficient persistent storage with content-addressed keys.
Each cluster is identified by a hash of its sorted paths, enabling:
- Incremental cache growth as new clusters are encountered
- Cache reuse across different clustering runs with same paths
- Persistence of labels even when cluster IDs change

Labels can be exported to JSON for version control and sharing, and
imported to seed fresh caches without expensive LLM regeneration.
"""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from imas_codex import dd_version
from imas_codex.definitions.clusters import LABELS_FILE
from imas_codex.resource_path_accessor import ResourcePathAccessor
from imas_codex.settings import get_language_model

logger = logging.getLogger(__name__)


@dataclass
class CachedLabel:
    """A cached label for a cluster."""

    label: str
    description: str
    model: str
    created_at: str


def compute_cluster_hash(paths: list[str]) -> str:
    """Compute a content-addressed hash for a cluster based on its paths.

    Args:
        paths: List of paths in the cluster (will be sorted for consistency)

    Returns:
        SHA256 hash (first 16 characters) of the sorted paths
    """
    sorted_paths = sorted(paths)
    paths_str = "\n".join(sorted_paths)
    return hashlib.sha256(paths_str.encode()).hexdigest()[:16]


class LabelCache:
    """SQLite-based content-addressed cache for cluster labels.

    Labels are keyed by a hash of sorted cluster paths, allowing:
    - Cache hits even when cluster IDs change
    - Incremental growth as new clusters are added
    - Persistence across different clustering runs
    """

    def __init__(self, cache_file: Path | None = None):
        """Initialize the label cache.

        Args:
            cache_file: Path to SQLite database. If None, uses default location.
        """
        if cache_file is None:
            path_accessor = ResourcePathAccessor(dd_version=dd_version)
            cache_file = path_accessor.clusters_dir / "label_cache.db"

        self.cache_file = cache_file
        self._ensure_schema()
        self._seed_from_definitions()

    def _seed_from_definitions(self) -> None:
        """Seed cache from definitions file if cache is empty."""
        if not LABELS_FILE.exists():
            return

        with self._get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
            if count > 0:
                return  # Cache already populated

        try:
            with LABELS_FILE.open() as f:
                data = json.load(f)

            if not data:
                return

            imported = self.import_labels(data)
            logger.info(f"Seeded cache with {imported} labels from definitions")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to seed from definitions: {e}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(self.cache_file)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Ensure the database schema exists."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS labels (
                    path_hash TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    description TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    paths_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model ON labels(model)
            """)
            conn.commit()

    def get_label(
        self, paths: list[str], model: str | None = None
    ) -> CachedLabel | None:
        """Get a cached label for a cluster.

        Args:
            paths: List of paths in the cluster
            model: Optional model name to match. If None, any model matches.

        Returns:
            CachedLabel if found, None otherwise
        """
        path_hash = compute_cluster_hash(paths)

        with self._get_connection() as conn:
            if model:
                row = conn.execute(
                    "SELECT label, description, model, created_at "
                    "FROM labels WHERE path_hash = ? AND model = ?",
                    (path_hash, model),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT label, description, model, created_at "
                    "FROM labels WHERE path_hash = ?",
                    (path_hash,),
                ).fetchone()

            if row:
                return CachedLabel(
                    label=row["label"],
                    description=row["description"],
                    model=row["model"],
                    created_at=row["created_at"],
                )
            return None

    def set_label(
        self,
        paths: list[str],
        label: str,
        description: str,
        model: str | None = None,
    ) -> str:
        """Store a label for a cluster.

        Args:
            paths: List of paths in the cluster
            label: The cluster label
            description: The cluster description
            model: Model that generated the label (default from settings)

        Returns:
            The path hash used as key
        """
        path_hash = compute_cluster_hash(paths)
        model = model or get_language_model()
        created_at = datetime.now().isoformat()
        paths_json = json.dumps(sorted(paths))

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO labels
                (path_hash, label, description, model, created_at, paths_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (path_hash, label, description, model, created_at, paths_json),
            )
            conn.commit()

        logger.debug(f"Cached label for cluster {path_hash}: {label}")
        return path_hash

    def get_many(
        self, clusters: list[dict], model: str | None = None
    ) -> tuple[dict[int, CachedLabel], list[dict]]:
        """Get cached labels for multiple clusters at once.

        Args:
            clusters: List of cluster dicts with 'id' and 'paths' keys
            model: Optional model name to match

        Returns:
            Tuple of (cached labels by cluster ID, list of uncached clusters)
        """
        cached = {}
        uncached = []

        for cluster in clusters:
            cluster_id = cluster["id"]
            paths = cluster.get("paths", [])

            label = self.get_label(paths, model)
            if label:
                cached[cluster_id] = label
            else:
                uncached.append(cluster)

        logger.info(
            f"Label cache: {len(cached)} hits, {len(uncached)} misses "
            f"({len(cached) / len(clusters) * 100:.1f}% hit rate)"
        )

        return cached, uncached

    def set_many(
        self,
        labels: list[tuple[list[str], str, str]],
        model: str | None = None,
    ) -> int:
        """Store multiple labels at once.

        Args:
            labels: List of (paths, label, description) tuples
            model: Model that generated the labels (default from settings)

        Returns:
            Number of labels stored
        """
        model = model or get_language_model()
        created_at = datetime.now().isoformat()
        count = 0

        with self._get_connection() as conn:
            for paths, label, description in labels:
                path_hash = compute_cluster_hash(paths)
                paths_json = json.dumps(sorted(paths))

                conn.execute(
                    """
                    INSERT OR REPLACE INTO labels
                    (path_hash, label, description, model, created_at, paths_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (path_hash, label, description, model, created_at, paths_json),
                )
                count += 1

            conn.commit()

        logger.info(f"Cached {count} labels")
        return count

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
            by_model = conn.execute(
                "SELECT model, COUNT(*) as count FROM labels GROUP BY model"
            ).fetchall()

        return {
            "total_labels": total,
            "by_model": {row["model"]: row["count"] for row in by_model},
            "cache_file": str(self.cache_file),
            "cache_size_mb": (
                self.cache_file.stat().st_size / (1024 * 1024)
                if self.cache_file.exists()
                else 0
            ),
        }

    def clear(self, model: str | None = None) -> int:
        """Clear cached labels.

        Args:
            model: If provided, only clear labels from this model

        Returns:
            Number of labels deleted
        """
        with self._get_connection() as conn:
            if model:
                result = conn.execute("DELETE FROM labels WHERE model = ?", (model,))
            else:
                result = conn.execute("DELETE FROM labels")

            count = result.rowcount
            conn.commit()

        logger.info(f"Cleared {count} cached labels")
        return count

    def export_labels(self, output_file: Path | None = None) -> dict:
        """Export all labels to a flat dict keyed by path_hash.

        One entry per hash (latest wins if duplicates exist).

        Args:
            output_file: Optional path to write JSON. If None, uses LABELS_FILE.

        Returns:
            Dict of labels keyed by path_hash
        """
        output_file = output_file or LABELS_FILE

        # Load existing labels to merge with
        existing: dict = {}
        if output_file.exists():
            try:
                with output_file.open() as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}

        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT path_hash, label, description, model, created_at, paths_json "
                "FROM labels"
            ).fetchall()

        # Merge: cache entries overwrite existing (they're newer)
        for row in rows:
            existing[row["path_hash"]] = {
                "label": row["label"],
                "description": row["description"],
                "model": row["model"],
                "created_at": row["created_at"],
                "paths": json.loads(row["paths_json"]),
            }

        # Write atomically
        output_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = output_file.with_suffix(".json.tmp")
        with tmp_file.open("w") as f:
            json.dump(existing, f, indent=2, sort_keys=True)
        tmp_file.rename(output_file)

        logger.info(f"Exported {len(existing)} labels to {output_file}")
        return existing

    def import_labels(self, data: dict) -> int:
        """Import labels from a dict, additive only (skip existing hashes).

        Args:
            data: Dict of labels keyed by path_hash

        Returns:
            Number of labels imported
        """
        count = 0

        with self._get_connection() as conn:
            for path_hash, entry in data.items():
                # Skip if already exists
                existing = conn.execute(
                    "SELECT 1 FROM labels WHERE path_hash = ?", (path_hash,)
                ).fetchone()
                if existing:
                    continue

                paths_json = json.dumps(entry.get("paths", []))
                conn.execute(
                    """
                    INSERT INTO labels
                    (path_hash, label, description, model, created_at, paths_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        path_hash,
                        entry["label"],
                        entry["description"],
                        entry.get("model", "unknown"),
                        entry.get("created_at", datetime.now().isoformat()),
                        paths_json,
                    ),
                )
                count += 1

            conn.commit()

        logger.info(f"Imported {count} labels (skipped existing)")
        return count

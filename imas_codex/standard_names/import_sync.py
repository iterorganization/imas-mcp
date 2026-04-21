"""Import synchronisation primitives — lock and watermark.

Provides graph-backed advisory locking and compare-and-set watermark
advancement for ``sn import``.  Prevents concurrent imports from
corrupting graph state.

See plan 35 §Import concurrency: lock + CAS watermark.
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from datetime import UTC, datetime

from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)

#: Default stale-lock timeout in minutes.
STALE_LOCK_MINUTES = 30


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class WatermarkState:
    """Snapshot of the ImportWatermark singleton."""

    last_commit_sha: str | None
    last_import_at: str | None
    source_repo: str | None


@dataclass
class LockState:
    """Snapshot of the ImportLock singleton."""

    held: bool
    holder: str | None
    acquired_at: str | None


# ---------------------------------------------------------------------------
# Holder identity
# ---------------------------------------------------------------------------


def _holder_id() -> str:
    """Return a unique holder string: ``hostname:pid``."""
    return f"{socket.gethostname()}:{os.getpid()}"


# ---------------------------------------------------------------------------
# Lock operations
# ---------------------------------------------------------------------------


def acquire_import_lock(gc: GraphClient) -> bool:
    """Attempt to acquire the import lock singleton.

    Creates the ``ImportLock`` node if it doesn't exist.
    Returns True if the lock was acquired, False if another holder has it.

    Stale locks (older than ``STALE_LOCK_MINUTES``) are broken automatically
    with a warning.
    """
    holder = _holder_id()
    now_iso = datetime.now(UTC).isoformat()

    # First: try to break stale lock
    gc.query(
        """
        MERGE (l:ImportLock {id: 'singleton'})
        WITH l
        WHERE l.holder IS NOT NULL
          AND l.holder <> $holder
          AND l.acquired_at IS NOT NULL
          AND datetime(l.acquired_at) < datetime() - duration({minutes: $stale_minutes})
        SET l.holder = null,
            l.acquired_at = null
        """,
        holder=holder,
        stale_minutes=STALE_LOCK_MINUTES,
    )

    # Now try to acquire
    rows = gc.query(
        """
        MERGE (l:ImportLock {id: 'singleton'})
        WITH l
        WHERE l.holder IS NULL
        SET l.holder = $holder,
            l.acquired_at = $now
        RETURN true AS acquired
        """,
        holder=holder,
        now=now_iso,
    )

    if rows:
        logger.info("Import lock acquired by %s", holder)
        return True

    # Check who holds it
    rows = gc.query(
        """
        MATCH (l:ImportLock {id: 'singleton'})
        RETURN l.holder AS holder, l.acquired_at AS acquired_at
        """
    )
    if rows:
        current = rows[0]
        if current["holder"] == holder:
            # We already hold it (re-entrant)
            logger.info("Import lock already held by us (%s)", holder)
            return True
        logger.warning(
            "Import lock held by %s since %s — cannot acquire",
            current["holder"],
            current["acquired_at"],
        )
    return False


def release_import_lock(gc: GraphClient) -> None:
    """Release the import lock.

    Only releases if the current process holds it.
    """
    holder = _holder_id()
    gc.query(
        """
        MATCH (l:ImportLock {id: 'singleton'})
        WHERE l.holder = $holder
        SET l.holder = null,
            l.acquired_at = null
        """,
        holder=holder,
    )
    logger.info("Import lock released by %s", holder)


def force_release_import_lock(gc: GraphClient) -> None:
    """Unconditionally release the import lock (admin escape hatch)."""
    gc.query(
        """
        MERGE (l:ImportLock {id: 'singleton'})
        SET l.holder = null,
            l.acquired_at = null
        """
    )
    logger.warning("Import lock force-released")


# ---------------------------------------------------------------------------
# Watermark operations
# ---------------------------------------------------------------------------


def read_watermark(gc: GraphClient) -> WatermarkState:
    """Read the current ImportWatermark singleton.

    Creates the singleton if it doesn't exist (returns all-None state).
    """
    rows = gc.query(
        """
        MERGE (w:ImportWatermark {id: 'singleton'})
        RETURN w.last_commit_sha AS last_commit_sha,
               w.last_import_at AS last_import_at,
               w.source_repo AS source_repo
        """
    )
    if rows:
        row = rows[0]
        return WatermarkState(
            last_commit_sha=row.get("last_commit_sha"),
            last_import_at=str(row["last_import_at"])
            if row.get("last_import_at")
            else None,
            source_repo=row.get("source_repo"),
        )
    return WatermarkState(last_commit_sha=None, last_import_at=None, source_repo=None)


def advance_watermark(
    gc: GraphClient,
    *,
    expected_prev_sha: str | None,
    new_sha: str,
    source_repo: str | None = None,
) -> bool:
    """Compare-and-set advance of the watermark.

    Returns True if the watermark was advanced, False if another import
    moved it while we were running (CAS failure).
    """
    now_iso = datetime.now(UTC).isoformat()

    if expected_prev_sha is None:
        # First-ever import: watermark has no SHA yet
        rows = gc.query(
            """
            MATCH (w:ImportWatermark {id: 'singleton'})
            WHERE w.last_commit_sha IS NULL
            SET w.last_commit_sha = $new_sha,
                w.last_import_at = $now,
                w.source_repo = coalesce($source_repo, w.source_repo)
            RETURN w.last_commit_sha AS sha
            """,
            new_sha=new_sha,
            now=now_iso,
            source_repo=source_repo,
        )
    else:
        rows = gc.query(
            """
            MATCH (w:ImportWatermark {id: 'singleton'})
            WHERE w.last_commit_sha = $expected
            SET w.last_commit_sha = $new_sha,
                w.last_import_at = $now,
                w.source_repo = coalesce($source_repo, w.source_repo)
            RETURN w.last_commit_sha AS sha
            """,
            expected=expected_prev_sha,
            new_sha=new_sha,
            now=now_iso,
            source_repo=source_repo,
        )

    if rows:
        logger.info("Watermark advanced to %s", new_sha[:12])
        return True

    logger.warning(
        "Watermark CAS failed: expected %s but another import moved it",
        expected_prev_sha[:12] if expected_prev_sha else "NULL",
    )
    return False

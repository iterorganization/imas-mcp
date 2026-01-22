"""
Parallel command execution for local and remote facilities.

This module extends remote.tools.run() with async parallel execution.
Commands run either locally or via SSH depending on is_local_facility().

IMPORTANT: This interface must work identically whether:
1. Running on the target facility (local execution)
2. Running from a different machine (SSH execution)

The max_sessions parameter limits concurrent SSH connections to avoid overwhelming
remote systems.

Async Branch Isolation:
    For graph-led discovery, we need to ensure that parallel SSH processes
    don't interfere with each other. The strategy:
    
    1. Initial discovery from filesystem root runs SINGLE-THREADED to build
       the initial frontier without conflicts.
    
    2. Once the frontier has expanded (multiple independent branches), parallel
       processes can each claim a BRANCH exclusively via graph locking.
    
    3. A branch is a subtree rooted at a path. Once claimed, no other process
       can discover within that subtree until the claim is released.
    
    The BranchExecutor class implements this pattern.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from imas_codex.remote.tools import is_local_facility, run as sync_run

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of a command execution."""

    path: str
    stdout: str
    stderr: str
    returncode: int
    duration_ms: int


@dataclass
class ParallelExecutor:
    """Execute commands in parallel with session limiting.

    Uses asyncio with a semaphore to limit concurrent SSH sessions.
    Works transparently for both local and remote execution.

    Args:
        facility: Target facility ID
        max_sessions: Max concurrent executions (default: 4)
        timeout: Per-command timeout in seconds

    Example:
        executor = ParallelExecutor(facility="epfl", max_sessions=4)
        async for result in executor.run_batch(commands):
            print(f"{result.path}: {result.returncode}")
    """

    facility: str
    max_sessions: int = 4
    timeout: int = 30
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)
    _is_local: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.max_sessions)
        self._is_local = is_local_facility(self.facility)

    async def run_one(self, cmd: str, path: str) -> CommandResult:
        """Execute single command with semaphore limiting."""
        async with self._semaphore:
            start = time.monotonic()

            loop = asyncio.get_event_loop()
            try:
                output = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: sync_run(cmd, facility=self.facility, timeout=self.timeout),
                    ),
                    timeout=self.timeout + 5,
                )
                returncode = 0
                stderr = ""
                if "[stderr]:" in output:
                    parts = output.split("[stderr]:", 1)
                    output = parts[0].strip()
                    stderr = parts[1].strip() if len(parts) > 1 else ""

            except asyncio.TimeoutError:
                output = ""
                stderr = "Timeout"
                returncode = -1
            except Exception as e:
                output = ""
                stderr = str(e)
                returncode = -1

            duration_ms = int((time.monotonic() - start) * 1000)

            return CommandResult(
                path=path,
                stdout=output,
                stderr=stderr,
                returncode=returncode,
                duration_ms=duration_ms,
            )

    async def run_batch(
        self,
        commands: list[tuple[str, str]],  # [(cmd, path), ...]
    ) -> AsyncIterator[CommandResult]:
        """Execute commands in parallel, yielding results as they complete."""
        tasks = [asyncio.create_task(self.run_one(cmd, path)) for cmd, path in commands]

        for coro in asyncio.as_completed(tasks):
            yield await coro


@dataclass
class BranchClaim:
    """A claimed branch for exclusive processing.

    Branches are subtrees rooted at a path. Once claimed, no other
    process should discover within that subtree.
    """

    facility: str
    root_path: str
    session_id: str
    claimed_at: float = field(default_factory=time.time)
    child_count: int = 0

    @property
    def branch_id(self) -> str:
        """Unique identifier for this branch."""
        return f"{self.facility}:{self.root_path}"


class BranchExecutor:
    """Execute discovery on exclusive branches.

    This executor ensures that parallel discovery processes don't
    interfere with each other by claiming branches before scanning.

    Strategy:
        1. Query graph for unclaimed branches (paths with status=pending
           and expand_to > depth, not yet claimed by another session)
        2. Claim a branch atomically via graph transaction
        3. Scan all paths within that branch
        4. Release the claim when done

    Usage:
        async with BranchExecutor(facility="iter", session_id="scan-123") as executor:
            async for branch in executor.claim_branches(max_branches=4):
                await process_branch(branch)
    """

    def __init__(
        self,
        facility: str,
        session_id: str,
        max_sessions: int = 4,
        timeout: int = 30,
    ) -> None:
        self.facility = facility
        self.session_id = session_id
        self.max_sessions = max_sessions
        self.timeout = timeout
        self._parallel_executor = ParallelExecutor(
            facility=facility,
            max_sessions=max_sessions,
            timeout=timeout,
        )
        self._claimed_branches: list[BranchClaim] = []

    async def __aenter__(self) -> BranchExecutor:
        return self

    async def __aexit__(self, *args) -> None:
        # Release all claimed branches
        await self.release_all_claims()

    async def claim_branch(self, root_path: str) -> BranchClaim | None:
        """Attempt to claim a branch for exclusive processing.

        Returns None if the branch is already claimed by another session.
        Uses atomic graph update to prevent race conditions.
        """
        from imas_codex.graph import GraphClient

        branch_id = f"{self.facility}:{root_path}"

        with GraphClient() as gc:
            # Atomic claim: only succeeds if not already claimed
            result = gc.query(
                """
                MATCH (p:FacilityPath {id: $branch_id})
                WHERE p.claimed_by IS NULL OR p.claimed_by = $session_id
                SET p.claimed_by = $session_id,
                    p.claimed_at = datetime()
                RETURN p.id AS id, p.path AS path
                """,
                branch_id=branch_id,
                session_id=self.session_id,
            )

            if result:
                claim = BranchClaim(
                    facility=self.facility,
                    root_path=root_path,
                    session_id=self.session_id,
                )
                self._claimed_branches.append(claim)
                logger.debug(f"Claimed branch: {root_path}")
                return claim

        logger.debug(f"Branch already claimed: {root_path}")
        return None

    async def release_claim(self, claim: BranchClaim) -> None:
        """Release a branch claim."""
        from imas_codex.graph import GraphClient

        with GraphClient() as gc:
            gc.query(
                """
                MATCH (p:FacilityPath {id: $branch_id})
                WHERE p.claimed_by = $session_id
                REMOVE p.claimed_by, p.claimed_at
                """,
                branch_id=claim.branch_id,
                session_id=self.session_id,
            )

        if claim in self._claimed_branches:
            self._claimed_branches.remove(claim)
        logger.debug(f"Released branch: {claim.root_path}")

    async def release_all_claims(self) -> None:
        """Release all claims held by this executor."""
        for claim in list(self._claimed_branches):
            await self.release_claim(claim)

    async def get_claimable_branches(self, limit: int = 10) -> list[str]:
        """Get paths that can be claimed for parallel processing.

        Only returns paths that:
        1. Have status='pending' or expand_to > depth
        2. Are not claimed by another session
        3. Are at sufficient depth to allow parallel processing

        For initial discovery (depth=0), returns at most 1 path to
        ensure sequential processing until the frontier expands.
        """
        from imas_codex.graph import GraphClient

        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
                WHERE (p.status = 'pending' OR (p.expand_to IS NOT NULL AND p.expand_to > p.depth))
                  AND (p.claimed_by IS NULL OR p.claimed_by = $session_id)
                RETURN p.path AS path, p.depth AS depth
                ORDER BY p.depth ASC, p.path ASC
                LIMIT $limit
                """,
                facility=self.facility,
                session_id=self.session_id,
                limit=limit,
            )

            paths = [r["path"] for r in result]

            # If we have paths at depth 0, only return one to ensure sequential
            # initial discovery
            if paths and result[0]["depth"] == 0:
                return paths[:1]

            return paths

    @property
    def parallel_executor(self) -> ParallelExecutor:
        """Get the underlying parallel executor for running commands."""
        return self._parallel_executor

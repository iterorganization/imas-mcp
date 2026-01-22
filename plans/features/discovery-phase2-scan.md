# Phase 2: Scan Pipeline

**Goal**: Implement SSH-based directory scanning with parallel execution and graph persistence  
**Scope**: ~2500 lines (scanner implementation, executor, graph persistence, tests)  
**Testable Outputs**: DirectoryScanner class, parallel execution, Cypher integration, SSH handling  
**Duration**: 3-5 days (1-2 dev)

## Deliverables

### 1. Parallel Executor

Create `imas_codex/discovery/executor.py`:

```python
"""Parallel command execution for local and remote facilities."""

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator

from imas_codex.remote.tools import is_local_facility, run as sync_run


@dataclass
class CommandResult:
    """Result of a command execution."""
    path: str
    stdout: str
    stderr: str
    returncode: int
    duration_ms: int
    
    @property
    def success(self) -> bool:
        """Whether command executed successfully."""
        return self.returncode == 0


class AsyncExecutor:
    """Execute commands in parallel with SSH session limiting.
    
    Works transparently for:
    1. Local execution (direct subprocess)
    2. Remote execution via SSH (facility-aware)
    
    The max_sessions parameter limits concurrent SSH connections.
    Local execution is not limited (no resource constraint).
    
    Args:
        facility: Target facility ID
        max_sessions: Max concurrent executions (default: 4)
        timeout: Per-command timeout in seconds (default: 30)
    
    Example:
        executor = AsyncExecutor(facility="epfl", max_sessions=4)
        commands = [
            ("fd -e py", "/home/codes"),
            ("du -sb", "/home/data"),
        ]
        async for result in executor.run_batch(commands):
            print(f"{result.path}: {result.returncode}")
    """
    
    def __init__(
        self,
        facility: str,
        max_sessions: int = 4,
        timeout: int = 30,
    ):
        self.facility = facility
        self.max_sessions = max_sessions
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_sessions)
        self._is_local = is_local_facility(facility)
    
    async def run_one(self, cmd: str, path: str) -> CommandResult:
        """Execute single command with session limiting.
        
        Args:
            cmd: Shell command to execute
            path: Associated path (for context/logging)
        
        Returns:
            CommandResult with output and return code
        """
        async with self._semaphore:
            start = time.monotonic()
            
            loop = asyncio.get_event_loop()
            try:
                # Use run_in_executor for blocking sync_run call
                output = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: sync_run(
                            cmd,
                            facility=self.facility,
                            timeout=self.timeout,
                            check=False,
                        )
                    ),
                    timeout=self.timeout + 5,
                )
                
                # Parse stderr if present
                returncode = 0
                stderr = ""
                if "[stderr]:" in output:
                    parts = output.split("[stderr]:", 1)
                    output = parts[0].strip()
                    stderr = parts[1].strip() if len(parts) > 1 else ""
                
            except asyncio.TimeoutError:
                output = ""
                stderr = "Command timeout"
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
        commands: list[tuple[str, str]],
    ) -> AsyncIterator[CommandResult]:
        """Execute commands in parallel, yielding as they complete.
        
        Args:
            commands: List of (cmd, path) tuples
        
        Yields:
            CommandResult objects as commands complete (not in input order)
        """
        tasks = [
            asyncio.create_task(self.run_one(cmd, path))
            for cmd, path in commands
        ]
        
        for coro in asyncio.as_completed(tasks):
            yield await coro


# Convenience functions for direct execution

async def run_parallel_commands(
    facility: str,
    commands: list[tuple[str, str]],
    max_sessions: int = 4,
    timeout: int = 30,
) -> list[CommandResult]:
    """Execute multiple commands in parallel.
    
    Args:
        facility: Target facility
        commands: List of (cmd, path) tuples
        max_sessions: Max concurrent executions
        timeout: Per-command timeout
    
    Returns:
        List of CommandResult objects
    """
    executor = AsyncExecutor(facility, max_sessions, timeout)
    results = []
    async for result in executor.run_batch(commands):
        results.append(result)
    return results
```

### 2. Directory Scanner

Create `imas_codex/discovery/scanner.py`:

```python
"""Directory scanning with safe handling of large directories."""

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

from imas_codex.discovery.executor import AsyncExecutor, CommandResult
from imas_codex.discovery.models import DirStats, DiscoveryScanConfig


class DirectoryScanner:
    """Scan directories via SSH/local with DirStats collection.
    
    Handles:
    - Safe large directory enumeration (timeouts, file counting)
    - Pattern detection for IMAS/physics codes
    - File type classification
    - Quick metadata collection (README, Makefile, git)
    
    Args:
        config: DiscoveryScanConfig with facility and limits
    
    Example:
        config = DiscoveryScanConfig(facility="epfl", timeout=30)
        scanner = DirectoryScanner(config)
        stats = await scanner.scan_directory("/home/codes")
    """
    
    # Patterns to detect IMAS/physics content
    IMAS_PATTERNS = [
        "put_slice", "get_slice", "mdsplus", "imas",
        "equilibrium", "transport", "core_profiles",
    ]
    
    PHYSICS_PATTERNS = [
        "stellarator", "tokamak", "plasma", "fusion",
        "MHD", "turbulence", "gyrokinetic",
    ]
    
    def __init__(self, config: DiscoveryScanConfig):
        self.config = config
        self.executor = AsyncExecutor(
            facility=config.facility,
            max_sessions=config.max_sessions,
            timeout=config.timeout,
        )
    
    async def scan_directory(self, path: str, depth: int = 0) -> tuple[DirStats, list[str]]:
        """Scan directory and return stats + child paths.
        
        Collects:
        - File type distribution via fd
        - Total file/dir count via find (bounded)
        - Size via du (or skipped if >10k files)
        - README/Makefile/git presence
        - IMAS/physics pattern matches
        
        Returns:
            (DirStats, list_of_children)
        
        Raises:
            TimeoutError if scan exceeds timeout
            OSError if directory doesn't exist
        """
        # Build command pipeline for efficient scanning
        commands = [
            # Count files by extension (fast, bounded)
            (f"fd --type f . '{path}' | head -10001 | sed 's/.*\\.//' | sort | uniq -c",
             f"{path}:file_types"),
            
            # Count total files (head stops at 10001)
            (f"fd --type f . '{path}' | wc -l",
             f"{path}:file_count"),
            
            # Count directories
            (f"fd --type d . '{path}' | wc -l",
             f"{path}:dir_count"),
            
            # Check for special files
            (f"ls -la '{path}' | grep -E '(README|Makefile|.git)' | wc -l",
             f"{path}:special_files"),
            
            # List immediate children (for traversal)
            (f"ls -1 '{path}'",
             f"{path}:children"),
            
            # Quick pattern search (limited to avoid huge output)
            (f"rg -l '({'|'.join(self.IMAS_PATTERNS)})' '{path}' --max-count 5 2>/dev/null | head -5",
             f"{path}:imas_patterns"),
        ]
        
        # Try size calculation only if likely to succeed
        file_count_result = None
        size_skipped = False
        
        # Execute all commands in parallel
        results = await self._run_scan_commands(commands)
        
        # Parse results
        stats = DirStats()
        children = []
        
        for result in results:
            if result.returncode != 0 and result.path != f"{path}:children":
                # Non-fatal failure, continue
                continue
            
            if result.path == f"{path}:file_types":
                stats.file_type_counts = self._parse_file_types(result.stdout)
            
            elif result.path == f"{path}:file_count":
                file_count_result = int(result.stdout.strip())
                stats.total_files = file_count_result
            
            elif result.path == f"{path}:dir_count":
                stats.total_dirs = int(result.stdout.strip())
            
            elif result.path == f"{path}:special_files":
                special_count = int(result.stdout.strip())
                stats.has_readme = special_count > 0  # Proxy check
                stats.has_git = "git" in result.stdout.lower()
                stats.has_makefile = "makefile" in result.stdout.lower()
            
            elif result.path == f"{path}:children":
                if result.returncode == 0:
                    children = [
                        path.rstrip('/') + '/' + child.strip()
                        for child in result.stdout.strip().split('\n')
                        if child.strip()
                    ]
            
            elif result.path == f"{path}:imas_patterns":
                if result.returncode == 0:
                    patterns = result.stdout.strip().split('\n')
                    stats.patterns_detected = [p for p in patterns if p]
        
        # Size calculation (only if <10k files)
        if file_count_result and file_count_result < self.config.max_files_for_size:
            size_result = await self._get_directory_size(path)
            if size_result:
                stats.total_size_bytes = size_result
            else:
                stats.size_skipped = True
        else:
            stats.size_skipped = True
        
        return stats, children
    
    async def _run_scan_commands(self, commands: list[tuple[str, str]]) -> list[CommandResult]:
        """Execute all scan commands in parallel."""
        results = []
        async for result in self.executor.run_batch(commands):
            results.append(result)
        return results
    
    async def _get_directory_size(self, path: str) -> int | None:
        """Get directory size via du (only for small dirs)."""
        try:
            # Use -b for bytes, -s for summary
            result = await self.executor.run_one(
                f"du -sb '{path}' | awk '{{print $1}}'",
                f"{path}:size"
            )
            if result.success:
                return int(result.stdout.strip())
        except (ValueError, Exception):
            pass
        return None
    
    def _parse_file_types(self, output: str) -> dict[str, int]:
        """Parse file type counts from fd output.
        
        Input format:
            42 py
            12 f90
        """
        counts = {}
        for line in output.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    count = int(parts[0])
                    ext = parts[-1]
                    counts[ext] = count
                except ValueError:
                    continue
        return counts


class ScanPhase:
    """Execute complete scan phase for a facility.
    
    Orchestrates:
    1. Query graph for frontier (pending + expand_to > depth)
    2. Scan each path in parallel (with max_sessions limit)
    3. Persist results to graph
    4. Track progress and statistics
    """
    
    def __init__(self, config: DiscoveryScanConfig, graph_client):
        self.config = config
        self.graph = graph_client
        self.scanner = DirectoryScanner(config)
    
    async def run(self) -> dict:
        """Execute scan phase.
        
        Returns:
            Statistics dict with:
            {
              "scanned": 123,
              "children_created": 456,
              "errors": 2,
              "total_time_ms": 5000,
              "throughput": 25,  # paths/minute
            }
        """
        import time
        start = time.monotonic()
        
        # 1. Query frontier
        frontier = self.graph.get_frontier(self.config.facility, limit=self.config.limit)
        if not frontier:
            # Seed if empty
            frontier = self.graph.seed_facility(self.config.facility)
        
        scanned = 0
        children_created = 0
        errors = 0
        
        # 2. Scan in batches
        batch_size = self.config.max_sessions
        for i in range(0, len(frontier), batch_size):
            batch = frontier[i:i+batch_size]
            
            # Scan all paths in batch concurrently
            scan_tasks = [
                self.scanner.scan_directory(path)
                for path in batch
            ]
            
            for path, coro in zip(batch, scan_tasks):
                try:
                    stats, children = await coro
                    
                    # Persist to graph
                    self.graph.persist_scan_result(
                        facility=self.config.facility,
                        path=path,
                        dir_stats=stats,
                        children=children,
                    )
                    
                    scanned += 1
                    children_created += len(children)
                    
                except Exception as e:
                    errors += 1
                    self.graph.mark_skipped(
                        facility=self.config.facility,
                        path=path,
                        skip_reason=f"Scan error: {e}",
                    )
        
        duration_ms = int((time.monotonic() - start) * 1000)
        throughput = (scanned * 60000) // max(duration_ms, 1000)
        
        return {
            "scanned": scanned,
            "children_created": children_created,
            "errors": errors,
            "total_time_ms": duration_ms,
            "throughput": throughput,  # paths/minute
        }
```

### 3. Graph Persistence

Extend `imas_codex/discovery/graph_ops.py` with scan-specific operations:

```python
def persist_scan_result(
    self,
    facility: str,
    path: str,
    dir_stats: DirStats,
    children: list[str],
):
    """Persist scan result to graph.
    
    Cypher:
        MATCH (p:FacilityPath {id: $path_id})
        SET p.status = 'scanned'
        SET p.file_type_counts = $file_counts
        SET p.total_files = $total_files
        SET p.total_dirs = $total_dirs
        SET p.total_size_bytes = $total_size
        SET p.size_skipped = $size_skipped
        SET p.has_readme = $has_readme
        SET p.has_makefile = $has_makefile
        SET p.has_git = $has_git
        SET p.patterns_detected = $patterns
        SET p.scanned_at = datetime()
        WITH p
        UNWIND $children AS child_path
        CREATE (p)-[:CONTAINS]->(child:FacilityPath {
          id: $facility + ':' + child_path,
          path: child_path,
          depth: p.depth + 1,
          status: 'pending',
          created_at: datetime()
        })
    """
    pass

def mark_skipped(self, facility: str, path: str, skip_reason: str):
    """Mark path as skipped (error or low priority).
    
    Cypher:
        MATCH (p:FacilityPath {id: ...})
        SET p.status = 'skipped'
        SET p.skip_reason = $reason
        SET p.scanned_at = datetime()
    """
    pass

def mark_expanded(self, facility: str, path: str):
    """Mark path as expanded (children created).
    
    Cypher:
        MATCH (p:FacilityPath {id: ...})
        SET p.expand_to = NULL
    """
    pass
```

### 4. Integration Tests

Create `tests/discovery/test_scanner.py`:

```python
"""Integration tests for directory scanner."""

import asyncio
import pytest
from pathlib import Path
from imas_codex.discovery.scanner import DirectoryScanner
from imas_codex.discovery.models import DiscoveryScanConfig


@pytest.fixture
def temp_dir_structure(tmp_path):
    """Create test directory structure."""
    codes_dir = tmp_path / "codes"
    codes_dir.mkdir()
    
    # Create some Python files
    (codes_dir / "main.py").write_text("print('hello')")
    (codes_dir / "util.py").write_text("def helper(): pass")
    
    # Create subdirectory
    lib_dir = codes_dir / "lib"
    lib_dir.mkdir()
    (lib_dir / "lib.py").write_text("# library")
    
    # Create README
    (codes_dir / "README.md").write_text("# My Code")
    
    return codes_dir


@pytest.mark.asyncio
async def test_scan_directory(temp_dir_structure):
    """Test scanning a directory."""
    config = DiscoveryScanConfig(
        facility="local",  # Local execution
        timeout=10,
    )
    
    scanner = DirectoryScanner(config)
    stats, children = await scanner.scan_directory(str(temp_dir_structure))
    
    # Verify stats
    assert stats.total_files >= 3  # At least the files we created
    assert stats.total_dirs >= 1  # At least the lib dir
    assert stats.has_readme is True
    assert "py" in stats.file_type_counts
    assert stats.file_type_counts["py"] >= 2


@pytest.mark.asyncio
async def test_file_type_parsing():
    """Test file type count parsing."""
    scanner = DirectoryScanner(DiscoveryScanConfig(facility="test"))
    
    output = "   42 py\n   12 f90\n    5 sh"
    counts = scanner._parse_file_types(output)
    
    assert counts["py"] == 42
    assert counts["f90"] == 12
    assert counts["sh"] == 5


def test_scan_config_validation():
    """Test configuration validation."""
    config = DiscoveryScanConfig(
        facility="epfl",
        timeout=30,
        max_sessions=4,
    )
    config.validate()  # Should not raise
```

### 5. Unit Tests for Executor

Create `tests/discovery/test_executor.py`:

```python
"""Unit tests for async executor."""

import asyncio
import pytest
from unittest.mock import Mock, patch
from imas_codex.discovery.executor import AsyncExecutor, CommandResult


@pytest.mark.asyncio
async def test_executor_semaphore():
    """Test that executor respects max_sessions limit."""
    concurrent_count = 0
    max_concurrent = 0
    
    async def mock_run(cmd, facility, timeout, check):
        nonlocal concurrent_count, max_concurrent
        concurrent_count += 1
        max_concurrent = max(max_concurrent, concurrent_count)
        await asyncio.sleep(0.1)
        concurrent_count -= 1
        return f"output of {cmd}"
    
    with patch('imas_codex.discovery.executor.sync_run', side_effect=mock_run):
        executor = AsyncExecutor(facility="test", max_sessions=2)
        
        commands = [
            (f"cmd{i}", f"path{i}")
            for i in range(5)
        ]
        
        results = []
        async for result in executor.run_batch(commands):
            results.append(result)
        
        assert len(results) == 5
        assert max_concurrent <= 2  # Should never exceed max_sessions


@pytest.mark.asyncio
async def test_executor_command_result():
    """Test CommandResult object."""
    result = CommandResult(
        path="/home/test",
        stdout="output",
        stderr="",
        returncode=0,
        duration_ms=100,
    )
    
    assert result.success is True
    assert result.duration_ms == 100
    
    failed = CommandResult(
        path="/home/test",
        stdout="",
        stderr="not found",
        returncode=-1,
        duration_ms=50,
    )
    
    assert failed.success is False
```

## Success Criteria

- ✅ AsyncExecutor correctly limits concurrent executions
- ✅ DirectoryScanner handles large directories safely
- ✅ File type detection works correctly
- ✅ Pattern detection finds IMAS/physics indicators
- ✅ Graph persistence creates proper parent-child relationships
- ✅ All tests pass with >85% coverage

## Testing Checklist

```bash
# Unit tests
pytest tests/discovery/test_executor.py -v
pytest tests/discovery/test_scanner.py -v

# Integration with mock graph
pytest tests/discovery/ -v

# Real execution (if on facility)
# Need to be careful with this
```

## Continuation Points

- **Phase 3**: Implement scoring phase with LLM
- **Phase 4**: Implement discover command orchestration
- **Performance tuning**: Benchmark scan throughput, optimize fd commands
- **Error handling**: Add retry logic for transient SSH failures

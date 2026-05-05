"""
Persistent SSH Python workers for amortizing session overhead.

Problem: Each SSH session to lac5/TCV triggers PAM session setup
(pam_systemd.so scope creation) costing ~2.6 seconds. With batch sizes
of 20-25, the pipeline makes dozens of SSH calls, each paying this cost.

Solution: Maintain persistent SSH processes with Python workers that
accept script requests over stdin/stdout. Session setup cost is paid
once (~2.6s), then subsequent commands execute in ~12ms (network RTT).

IMPORTANT — Python version constraint:
    Workers use ``/usr/bin/python3`` (system Python, may be 3.9+) to
    avoid the 60-100s NFS venv startup penalty.  All scripts executed
    via ``pooled_run_python_script()`` / ``SSHWorkerPool`` must be:
      - Python 3.9+ compatible (no ``match``, no ``X | Y`` type unions)
      - stdlib-only (no third-party imports)
    Scripts requiring 3.12+ features or venv packages (e.g. MDSplus)
    must be dispatched via ``run_python_script()`` instead, which uses
    the imas-codex venv Python.

Architecture:
    SSHWorkerPool (per ssh_host)
    ├── Worker 0: ssh -T host python3 -c '<worker_loop>'
    ├── Worker 1: ssh -T host python3 -c '<worker_loop>'
    └── Worker N: (created on demand, up to max_workers)

Each worker:
    - Runs a Python event loop on the remote side reading JSON requests
    - Executes scripts via exec() in isolated namespaces
    - Returns JSON responses with stdout capture
    - Handles errors gracefully (malformed input, script exceptions)
    - Self-terminates on stdin EOF (parent process dies)

Lifecycle:
    1. Pool created on first use (lazy)
    2. Workers acquired via async context manager
    3. Released back to pool after use
    4. Pool shutdown on:
       - Explicit close() call
       - Process exit (atexit handler)
       - Signal (SIGINT/SIGTERM via asyncio shutdown)
       - Context manager exit (__aexit__)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Track all active pools for cleanup on process exit
_active_pools: weakref.WeakSet[SSHWorkerPool] = weakref.WeakSet()
_atexit_registered = False


# ============================================================================
# Remote Worker Script
# ============================================================================

# This script runs on the remote host inside a persistent SSH session.
# It reads JSON requests from stdin, executes Python scripts, and writes
# JSON responses to stdout. Requirements:
# - Python 3.9+ compatible (system Python on some facilities)
# - stdlib only (no pip packages available on system Python)
# - Must handle exec() safely with isolated namespaces
# - Must flush stdout after each response (critical for protocol)
# - Must exit cleanly on stdin EOF

_WORKER_SCRIPT = r"""
import sys, json, traceback, io, os, subprocess

# Signal readiness — pool will send warmup commands after startup
sys.stdout.write(json.dumps({"ready": True, "pid": os.getpid()}) + "\n")
sys.stdout.flush()

while True:
    try:
        line = sys.stdin.readline()
        if not line:
            break  # EOF — parent closed stdin
        line = line.strip()
        if not line:
            continue

        request = json.loads(line)
        action = request.get("action", "exec")

        if action == "ping":
            sys.stdout.write(json.dumps({"ok": True, "pong": True}) + "\n")
            sys.stdout.flush()
            continue

        if action == "exec":
            script = request["script"]
            stdin_data = request.get("stdin", "")

            # Execute in isolated namespace
            old_stdin = sys.stdin
            old_stdout = sys.stdout
            sys.stdin = io.StringIO(stdin_data)
            buf = io.StringIO()
            sys.stdout = buf

            try:
                ns = {"__name__": "__main__", "__builtins__": __builtins__}
                exec(compile(script, "<remote>", "exec"), ns)
                result = {"ok": True, "stdout": buf.getvalue()}
            except SystemExit as e:
                result = {"ok": True, "stdout": buf.getvalue(), "exit_code": e.code or 0}
            except Exception as e:
                result = {"ok": False, "error": str(e),
                          "type": type(e).__name__,
                          "tb": traceback.format_exc(),
                          "stdout": buf.getvalue()}
            finally:
                sys.stdout = old_stdout
                sys.stdin = old_stdin

            sys.stdout.write(json.dumps(result) + "\n")
            sys.stdout.flush()
            continue

        # Unknown action
        sys.stdout.write(json.dumps({"ok": False, "error": f"unknown action: {action}"}) + "\n")
        sys.stdout.flush()

    except json.JSONDecodeError as e:
        sys.stdout.write(json.dumps({"ok": False, "error": f"invalid JSON: {e}"}) + "\n")
        sys.stdout.flush()
    except Exception as e:
        try:
            sys.stdout.write(json.dumps({"ok": False, "error": f"worker error: {e}"}) + "\n")
            sys.stdout.flush()
        except Exception:
            break  # stdout broken, exit
"""


@dataclass
class WorkerStats:
    """Statistics for a persistent SSH worker."""

    requests_sent: int = 0
    requests_ok: int = 0
    requests_failed: int = 0
    total_rtt_ms: float = 0.0
    worker_pid: int | None = None  # Remote PID
    started_at: float = field(default_factory=time.monotonic)

    @property
    def avg_rtt_ms(self) -> float:
        if self.requests_ok == 0:
            return 0.0
        return self.total_rtt_ms / self.requests_ok


class SSHWorker:
    """A single persistent SSH Python worker process.

    Manages an SSH subprocess running the worker loop script.
    Communicates via JSON-over-stdin/stdout protocol.
    """

    def __init__(
        self, ssh_host: str, worker_id: int, setup_commands: list[str] | None = None
    ):
        self.ssh_host = ssh_host
        self.worker_id = worker_id
        self.setup_commands = setup_commands
        self._proc: asyncio.subprocess.Process | None = None
        self._ready = False
        self._lock = asyncio.Lock()
        self.stats = WorkerStats()
        self._reader_buffer = ""

    async def start(self, timeout: float = 30.0) -> None:
        """Start the SSH worker process and wait for ready signal."""
        import shlex

        from imas_codex.remote.executor import _get_host_nice_level

        # Use shlex.quote to safely embed the worker script without base64.
        # shlex.quote wraps the script in single-quotes with proper escaping,
        # which is the standard POSIX approach and avoids XDR malware triggers.
        python_cmd = f"/usr/bin/python3 -c {shlex.quote(_WORKER_SCRIPT)}"

        # Apply nice level if configured
        nice_level = _get_host_nice_level(self.ssh_host)
        if nice_level is not None and nice_level > 0:
            python_cmd = f"nice -n {nice_level} {python_cmd}"

        # Put /tmp/imas-codex-tools first so cached local copies are used
        path_prefix = (
            'export PATH="/tmp/imas-codex-tools:'
            "$HOME/.local/share/imas-codex/venv/bin:"
            '$HOME/bin:$HOME/.local/bin:$PATH"'
        )
        parts = [path_prefix]
        if self.setup_commands:
            parts.extend(self.setup_commands)
        parts.append(python_cmd)
        remote_cmd = " && ".join(parts)

        cmd = ["ssh", "-T", self.ssh_host, remote_cmd]

        logger.debug(
            "Starting persistent SSH worker %d for %s",
            self.worker_id,
            self.ssh_host,
        )

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=16 * 1024 * 1024,  # 16MB — scan output can exceed 64KB default
        )

        # Wait for ready signal
        assert self._proc.stdout is not None
        try:
            ready_line = await asyncio.wait_for(
                self._proc.stdout.readline(),
                timeout=timeout,
            )
            ready_data = json.loads(ready_line.decode())
            if ready_data.get("ready"):
                self._ready = True
                self.stats.worker_pid = ready_data.get("pid")
                logger.debug(
                    "SSH worker %d for %s ready (remote PID %s)",
                    self.worker_id,
                    self.ssh_host,
                    self.stats.worker_pid,
                )
            else:
                raise RuntimeError(f"Worker did not signal ready: {ready_data}")
        except TimeoutError as exc:
            await self._kill()
            raise RuntimeError(
                f"SSH worker {self.worker_id} for {self.ssh_host} "
                f"did not become ready within {timeout}s"
            ) from exc
        except Exception:
            await self._kill()
            raise

    @property
    def alive(self) -> bool:
        """Check if the worker process is still running."""
        return self._proc is not None and self._proc.returncode is None and self._ready

    async def execute(
        self,
        script: str,
        stdin_data: str = "{}",
        timeout: float = 60.0,
    ) -> str:
        """Execute a Python script on the remote worker.

        Args:
            script: Python source code to execute
            stdin_data: Data to provide as stdin to the script
            timeout: Maximum time to wait for response

        Returns:
            Script stdout output

        Raises:
            RuntimeError: If worker is dead or script failed
            TimeoutError: If response not received within timeout
        """
        if not self.alive:
            raise RuntimeError(
                f"SSH worker {self.worker_id} for {self.ssh_host} is not alive"
            )

        request = json.dumps(
            {
                "action": "exec",
                "script": script,
                "stdin": stdin_data,
            }
        )

        start = time.monotonic()

        assert self._proc is not None
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        async with self._lock:
            try:
                self._proc.stdin.write((request + "\n").encode())
                await self._proc.stdin.drain()

                response_line = await asyncio.wait_for(
                    self._proc.stdout.readline(),
                    timeout=timeout,
                )
            except TimeoutError:
                # Worker hung — kill it
                logger.warning(
                    "SSH worker %d for %s timed out after %.1fs, killing",
                    self.worker_id,
                    self.ssh_host,
                    timeout,
                )
                await self._kill()
                raise TimeoutError(
                    f"SSH worker {self.worker_id} timed out after {timeout}s"
                ) from None
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                logger.warning(
                    "SSH worker %d for %s pipe broken: %s",
                    self.worker_id,
                    self.ssh_host,
                    e,
                )
                self._ready = False
                raise RuntimeError(f"Worker pipe broken: {e}") from e

        elapsed_ms = (time.monotonic() - start) * 1000
        self.stats.requests_sent += 1

        if not response_line:
            self._ready = False
            raise RuntimeError(
                f"SSH worker {self.worker_id} returned empty response (process died?)"
            )

        try:
            result = json.loads(response_line.decode())
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid JSON from worker: {response_line.decode()[:200]}"
            ) from e

        if result.get("ok"):
            self.stats.requests_ok += 1
            self.stats.total_rtt_ms += elapsed_ms
            return result.get("stdout", "").strip()
        else:
            self.stats.requests_failed += 1
            error = result.get("error", "unknown error")
            tb = result.get("tb", "")
            # Include any partial stdout in the error for debugging
            partial = result.get("stdout", "")
            raise RuntimeError(
                f"Remote script error: {error}\n{tb}"
                + (f"\nPartial stdout: {partial[:500]}" if partial else "")
            )

    async def ping(self, timeout: float = 5.0) -> bool:
        """Check if worker is responsive."""
        if not self.alive:
            return False
        try:
            assert self._proc is not None
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None
            async with self._lock:
                request = json.dumps({"action": "ping"}) + "\n"
                self._proc.stdin.write(request.encode())
                await self._proc.stdin.drain()
                response = await asyncio.wait_for(
                    self._proc.stdout.readline(),
                    timeout=timeout,
                )
                data = json.loads(response.decode())
                return data.get("pong", False)
        except Exception:
            return False

    async def _kill(self) -> None:
        """Kill the worker process."""
        self._ready = False
        if self._proc is not None:
            try:
                self._proc.kill()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except Exception:
                pass
            self._proc = None

    async def close(self) -> None:
        """Gracefully shut down the worker."""
        self._ready = False
        if self._proc is not None:
            try:
                # Close stdin — worker exits on EOF
                if self._proc.stdin and not self._proc.stdin.is_closing():
                    self._proc.stdin.close()
                # Wait briefly for graceful exit
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=3.0)
                except TimeoutError:
                    self._proc.kill()
                    await asyncio.wait_for(self._proc.wait(), timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            finally:
                self._proc = None

    def force_kill(self) -> None:
        """Synchronous force-kill for use in atexit/signal handlers."""
        self._ready = False
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None


class SSHWorkerPool:
    """Pool of persistent SSH workers for a specific host.

    Manages worker lifecycle, provides workers to callers via an async
    semaphore, and handles cleanup on shutdown.
    """

    def __init__(
        self,
        ssh_host: str,
        max_workers: int = 4,
        setup_commands: list[str] | None = None,
        start_timeout: float = 60.0,
    ):
        self.ssh_host = ssh_host
        self.max_workers = max_workers
        self.setup_commands = setup_commands
        self.start_timeout = start_timeout
        self._workers: list[SSHWorker] = []
        self._available: asyncio.Queue[SSHWorker] | None = None
        self._lock = asyncio.Lock()
        self._started = False
        self._closed = False
        self._next_id = 0

        # Register for cleanup
        _active_pools.add(self)
        _ensure_atexit()

    async def start(self, initial_workers: int | None = None) -> None:
        """Start the pool with initial workers.

        Args:
            initial_workers: Number of workers to start immediately.
                Defaults to max_workers.
        """
        if self._started:
            return
        async with self._lock:
            if self._started:
                return

            n = initial_workers if initial_workers is not None else self.max_workers
            n = min(n, self.max_workers)

            self._available = asyncio.Queue(maxsize=self.max_workers)

            # Start tool caching AND ALL workers concurrently.
            # System Python is on local disk, so no NFS cache warming
            # needed. All SSH sessions pay PAM overhead in parallel.
            tasks: list[asyncio.Task] = []

            # Tool caching task (copies rg/fd/tokei to /tmp)
            cache_task = asyncio.create_task(self._cache_remote_tools())

            # Create and start all workers concurrently
            workers_starting: list[SSHWorker] = []
            for _ in range(n):
                worker = SSHWorker(self.ssh_host, self._next_id, self.setup_commands)
                self._next_id += 1
                workers_starting.append(worker)
                tasks.append(
                    asyncio.create_task(worker.start(timeout=self.start_timeout))
                )

            # Wait for everything in parallel
            all_results = await asyncio.gather(
                cache_task,
                *tasks,
                return_exceptions=True,
            )

            # First result is cache task
            cache_result = all_results[0]
            if isinstance(cache_result, BaseException):
                logger.debug("Tool caching failed (non-fatal): %s", cache_result)

            # Remaining results are worker starts
            for worker, result in zip(workers_starting, all_results[1:], strict=True):
                if isinstance(result, BaseException):
                    logger.warning(
                        "Failed to start SSH worker %d for %s: %s",
                        worker.worker_id,
                        self.ssh_host,
                        result,
                    )
                else:
                    self._workers.append(worker)
                    self._available.put_nowait(worker)

            if not self._workers:
                raise RuntimeError(
                    f"Failed to start any SSH workers for {self.ssh_host}"
                )

            self._started = True
            logger.info(
                "SSH worker pool for %s started: %d/%d workers ready",
                self.ssh_host,
                len(self._workers),
                n,
            )

    async def _start_worker(self, worker: SSHWorker) -> SSHWorker:
        """Start a single worker and return it."""
        await worker.start(timeout=self.start_timeout)
        return worker

    async def _cache_remote_tools(self) -> None:
        """Copy NFS-mounted tool binaries to /tmp on remote host.

        Runs as a one-shot SSH command. Copies rg/fd/tokei from ~/bin/
        (NFS, 16KB blocks, ~2-5s per binary cold load) to /tmp/imas-codex-tools/
        (local disk, ~4ms load). Idempotent — skips already-cached files.
        """
        cache_cmd = (
            "mkdir -p /tmp/imas-codex-tools && "
            "for t in rg fd tokei; do "
            '  src="$HOME/bin/$t"; '
            '  dst="/tmp/imas-codex-tools/$t"; '
            '  if [ -f "$src" ] && [ ! -f "$dst" ]; then '
            '    cp "$src" "$dst" && chmod 755 "$dst" && '
            '    echo "cached $t"; '
            '  elif [ -f "$dst" ]; then '
            '    echo "already cached $t"; '
            "  fi; "
            "done"
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-T",
                self.ssh_host,
                cache_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=120.0,
            )
            if stdout:
                logger.debug(
                    "Tool cache for %s: %s",
                    self.ssh_host,
                    stdout.decode().strip(),
                )
        except TimeoutError:
            logger.warning(
                "Tool caching for %s timed out (non-fatal)",
                self.ssh_host,
            )
        except Exception as e:
            logger.debug("Tool caching failed (non-fatal): %s", e)

    @asynccontextmanager
    async def acquire(self):
        """Acquire a worker from the pool.

        Usage:
            async with pool.acquire() as worker:
                result = await worker.execute(script, stdin_data)

        The worker is returned to the pool after use. If the worker
        died during use, it is replaced with a new one.
        """
        if not self._started or self._closed:
            raise RuntimeError("Pool not started or already closed")

        assert self._available is not None
        worker = await self._available.get()

        try:
            # Check if worker is still alive
            if not worker.alive:
                logger.debug("Worker %d dead, replacing", worker.worker_id)
                worker = await self._replace_worker(worker)
            yield worker
        except Exception:
            # If the worker died during use, try to replace it
            if not worker.alive:
                try:
                    worker = await self._replace_worker(worker)
                except Exception as e:
                    logger.warning("Failed to replace dead worker: %s", e)
                    # Put a dead worker back — next acquire will retry
            raise
        finally:
            # Return worker to pool (even if dead — acquire handles it)
            try:
                self._available.put_nowait(worker)  # type: ignore[union-attr]
            except asyncio.QueueFull:
                # Shouldn't happen, but don't lose the worker
                await worker.close()

    async def _replace_worker(self, dead_worker: SSHWorker) -> SSHWorker:
        """Replace a dead worker with a new one."""
        await dead_worker.close()
        if dead_worker in self._workers:
            self._workers.remove(dead_worker)

        new_worker = SSHWorker(self.ssh_host, self._next_id, self.setup_commands)
        self._next_id += 1
        await new_worker.start(timeout=self.start_timeout)
        self._workers.append(new_worker)
        return new_worker

    async def close(self) -> None:
        """Shut down all workers gracefully."""
        if self._closed:
            return
        self._closed = True

        logger.debug("Closing SSH worker pool for %s", self.ssh_host)

        # Close all workers concurrently
        close_tasks = [w.close() for w in self._workers]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        self._workers.clear()
        self._started = False

        # Log aggregate stats
        total_requests = (
            sum(w.stats.requests_sent for w in self._workers) if self._workers else 0
        )
        logger.debug(
            "SSH worker pool for %s closed (total requests: %d)",
            self.ssh_host,
            total_requests,
        )

    def force_kill_all(self) -> None:
        """Synchronous force-kill for all workers (atexit/signal handler)."""
        for w in self._workers:
            w.force_kill()
        self._workers.clear()
        self._closed = True
        self._started = False


# ============================================================================
# Global Pool Registry
# ============================================================================

_pools: dict[str, SSHWorkerPool] = {}
_registry_lock: asyncio.Lock | None = None


def _get_registry_lock() -> asyncio.Lock:
    """Get or create the registry lock (must be created in event loop context)."""
    global _registry_lock
    if _registry_lock is None:
        _registry_lock = asyncio.Lock()
    return _registry_lock


async def get_worker_pool(
    ssh_host: str,
    max_workers: int = 4,
    setup_commands: list[str] | None = None,
    start_timeout: float = 60.0,
) -> SSHWorkerPool:
    """Get or create a worker pool for an SSH host.

    Pools are cached per ssh_host. First call creates and starts the pool.
    Subsequent calls return the existing pool.

    Args:
        ssh_host: SSH host alias
        max_workers: Maximum concurrent workers
        setup_commands: Shell commands to run before Python on remote
        start_timeout: Timeout for worker startup

    Returns:
        Started SSHWorkerPool
    """
    lock = _get_registry_lock()
    async with lock:
        if ssh_host in _pools and not _pools[ssh_host]._closed:
            return _pools[ssh_host]

        pool = SSHWorkerPool(
            ssh_host=ssh_host,
            max_workers=max_workers,
            setup_commands=setup_commands,
            start_timeout=start_timeout,
        )
        await pool.start()
        _pools[ssh_host] = pool
        return pool


async def close_all_pools() -> None:
    """Close all active worker pools."""
    for pool in list(_pools.values()):
        await pool.close()
    _pools.clear()


def force_kill_all_pools() -> None:
    """Synchronous force-kill for all pools (atexit/signal handler).

    Safe to call from signal handlers — no async, no locks.
    Does NOT kill SSH ControlMaster sockets (those are shared).
    Only kills the worker SSH processes spawned by this module.
    """
    for pool in list(_pools.values()):
        pool.force_kill_all()
    _pools.clear()

    # Also kill any pools tracked via WeakSet
    for pool in list(_active_pools):
        pool.force_kill_all()


def _ensure_atexit() -> None:
    """Register atexit handler once."""
    global _atexit_registered
    if not _atexit_registered:
        import atexit

        atexit.register(force_kill_all_pools)
        _atexit_registered = True


# ============================================================================
# High-Level API — Drop-in replacement for async_run_python_script
# ============================================================================


async def pooled_run_python_script(
    script_name: str,
    input_data: dict | list | None = None,
    ssh_host: str | None = None,
    timeout: int = 60,
    python_command: str = "python3",
    setup_commands: list[str] | None = None,
    pool: SSHWorkerPool | None = None,
) -> str:
    """Execute a Python script using a persistent SSH worker.

    Drop-in replacement for async_run_python_script that uses a persistent
    SSH session instead of creating a new one per call. Falls back to
    async_run_python_script if no pool is available or for local execution.

    Args:
        script_name: Script filename in imas_codex/remote/scripts/
        input_data: Dict/list to pass as JSON on stdin
        ssh_host: SSH host (None = local execution)
        timeout: Command timeout in seconds
        python_command: Ignored (persistent worker uses its own Python)
        setup_commands: Ignored (configured at pool level)
        pool: Explicit pool to use (if None, uses global registry)

    Returns:
        Script output (stdout)

    Raises:
        subprocess.CalledProcessError: On script error
        TimeoutError: On timeout
    """
    import importlib.resources
    import subprocess

    from imas_codex.remote.executor import is_local_host

    # Local execution or no host — use standard async approach
    if ssh_host is None or is_local_host(ssh_host):
        from imas_codex.remote.executor import async_run_python_script

        return await async_run_python_script(
            script_name,
            input_data,
            ssh_host,
            timeout,
            python_command,
            setup_commands,
        )

    # Load script content
    script_path = importlib.resources.files("imas_codex.remote.scripts").joinpath(
        script_name
    )
    script_content = script_path.read_text()

    # Prepare JSON input
    json_input = json.dumps(input_data) if input_data is not None else "{}"

    # Get or create pool
    if pool is None:
        try:
            pool = await get_worker_pool(
                ssh_host,
                setup_commands=setup_commands,
            )
        except Exception as e:
            logger.warning(
                "Failed to get worker pool for %s, falling back: %s",
                ssh_host,
                e,
            )
            from imas_codex.remote.executor import async_run_python_script

            return await async_run_python_script(
                script_name,
                input_data,
                ssh_host,
                timeout,
                python_command,
                setup_commands,
            )

    # Execute via persistent worker
    try:
        async with pool.acquire() as worker:
            result = await worker.execute(
                script_content,
                stdin_data=json_input,
                timeout=float(timeout),
            )
            return result
    except (RuntimeError, TimeoutError) as e:
        # Worker failed — fall back to standard approach
        logger.warning(
            "Persistent worker failed for %s/%s, falling back: %s",
            ssh_host,
            script_name,
            e,
        )
        from imas_codex.remote.executor import async_run_python_script

        raise subprocess.CalledProcessError(1, script_name, str(e), str(e)) from e


# ============================================================================
# Context Manager for CLI Tools
# ============================================================================


@asynccontextmanager
async def ssh_worker_session(
    ssh_host: str,
    max_workers: int = 4,
    setup_commands: list[str] | None = None,
    start_timeout: float = 30.0,
):
    """Context manager that creates and cleans up an SSH worker pool.

    Use this in CLI commands to ensure proper cleanup even on Ctrl+C:

        async with ssh_worker_session("tcv", max_workers=3) as pool:
            # pool is ready, use pooled_run_python_script with pool=pool
            result = await pooled_run_python_script(
                "scan_directories.py", data, "tcv", pool=pool
            )

    On exit (normal, exception, or signal), all worker processes are killed.
    """
    pool = SSHWorkerPool(
        ssh_host=ssh_host,
        max_workers=max_workers,
        setup_commands=setup_commands,
        start_timeout=start_timeout,
    )

    try:
        await pool.start()
        yield pool
    finally:
        await pool.close()

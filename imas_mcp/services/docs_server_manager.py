"""
Docs Server Manager for persistent docs-mcp-server process management.

This module provides lifecycle management for a single persistent docs-mcp-server
process, handling port allocation, health monitoring, and integration with the
IMAS MCP server architecture.
"""

import asyncio
import logging
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imas_mcp.services.docs_proxy_service import DocsProxyService

import aiohttp
import anyio
import psutil
from dotenv import load_dotenv

from imas_mcp.exceptions import DocsServerError

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class PortAllocationError(Exception):
    """Raised when unable to allocate a port for the docs server."""

    pass


class DocsServerUnavailableError(DocsServerError):
    """Raised when docs-mcp-server is not accessible."""

    pass


class DocsServerManager:
    """
    Manages the lifecycle of a persistent docs-mcp-server process.

    This class handles:
    - Automatic process startup and shutdown
    - Port allocation with conflict detection
    - Health monitoring and auto-recovery
    - Library and version discovery for health endpoints
    - Integration with IMAS MCP server lifecycle
    - Process cleanup utilities for orphaned docs-mcp-server instances
    """

    # Port range for docs-mcp-server (narrow range for faster scanning)
    MIN_PORT = 6280
    MAX_PORT = 6290

    def __init__(
        self,
        default_port: int = 6280,
        store_path: Path | None = None,
        timeout: int = 30,
    ):
        """Initialize the docs server manager.

        Args:
            default_port: Port to try first (6280), fallback to available if occupied
            store_path: Path for docs server data storage
            timeout: Timeout for server operations
        """
        self.default_port = default_port
        self.timeout = timeout
        self.store_path = store_path or Path.cwd() / "docs-data"
        self.store_path.mkdir(parents=True, exist_ok=True)

        # Process management
        self.process: anyio.AsyncProcess | None = None
        self.allocated_port: int | None = None
        self._startup_task: asyncio.Task | None = None
        self._health_monitor_task: asyncio.Task | None = None
        self._log_stdout_task: asyncio.Task | None = None
        self._log_stderr_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None

        # Health and status
        self._start_time: float | None = None
        self._libraries_cache: list[dict[str, Any]] | None = None
        self._last_health_check: float | None = None
        self._shutdown_event = asyncio.Event()
        self._started: bool = False  # Track if we've attempted to start

    @property
    def base_url(self) -> str:
        """Get the base URL for the docs server API."""
        if self.allocated_port is None:
            raise DocsServerError("Server not started - no port allocated")
        return f"http://127.0.0.1:{self.allocated_port}"

    @property
    def is_running(self) -> bool:
        """Check if the docs server process is running."""
        return (
            self.process is not None
            and self.process.returncode is None
            and self.allocated_port is not None
        )

    @property
    def uptime(self) -> float:
        """Get server uptime in seconds."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    async def start_server(self) -> None:
        """Start the docs server with automatic port allocation."""
        # Log the store path being used
        logger.info(f"Using docs server store path: {self.store_path.absolute()}")

        # Always start our own fresh server for path control
        logger.info("Starting fresh docs server with controlled docs-data path...")

        # Allocate port
        logger.info("About to allocate port...")
        self.allocated_port = await self._allocate_port()
        logger.info(f"Port allocation complete. Allocated port: {self.allocated_port}")
        logger.info(f"Allocated port {self.allocated_port} for docs server")

        # Stop any existing process first
        if self.process is not None:
            logger.info("Stopping existing docs server process...")
            try:
                self.process.terminate()
                await self.process.wait()
            except Exception as e:
                logger.warning(f"Error stopping existing process: {e}")
                try:
                    self.process.kill()
                    await self.process.wait()
                except Exception as e2:
                    logger.warning(f"Error killing existing process: {e2}")
            self.process = None

        # Start the process
        await self._start_docs_server_process()

        # Wait for server to be ready
        await self._wait_for_server_ready()

        # Start health monitoring
        self._start_health_monitor()

        # Log startup details
        logger.info("=== Docs Server Started ===")
        logger.info(f"URL: {self.base_url}")
        logger.info(f"MCP Endpoint: {self.base_url}/mcp")
        logger.info(f"Store Path: {self.store_path.absolute()}")
        logger.info(f"Port: {self.allocated_port}")
        logger.info("===========================")

    async def stop_server(self) -> None:
        """Stop the docs server gracefully."""
        if not self.is_running:
            logger.info("Docs server not running")
            return

        logger.info("Stopping docs server...")

        # Signal shutdown
        self._shutdown_event.set()

        # Stop health monitoring
        if self._health_monitor_task and not self._health_monitor_task.done():
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass

        # Stop monitor task
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Terminate process
        if self.process:
            try:
                self.process.terminate()
                try:
                    with anyio.fail_after(10):  # 10 second grace period
                        await self.process.wait()
                except TimeoutError:
                    logger.warning(
                        "Docs server didn't stop gracefully, killing process"
                    )
                    self.process.kill()
                    await self.process.wait()
            except Exception as e:
                logger.warning(f"Error stopping process: {e}")

            # Ensure resource is closed (closes pipes etc)
            if hasattr(self.process, "aclose"):
                try:
                    await self.process.aclose()
                except Exception as e:
                    logger.debug(f"Error closing process resource: {e}")

        # Cancel logging tasks after process is dead
        for task in [self._log_stdout_task, self._log_stderr_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.process = None
        self.allocated_port = None
        self._start_time = None
        self._libraries_cache = None
        self._log_stdout_task = None
        self._log_stderr_task = None
        self._monitor_task = None

        logger.info("Docs server stopped")

    async def restart_server(self) -> None:
        """Restart the docs server."""
        logger.info("Restarting docs server...")
        await self.stop_server()
        await asyncio.sleep(1)  # Brief pause
        await self.start_server()

    async def health_check(self) -> dict[str, Any]:
        """Perform health check and return status information."""
        health_data = {
            "status": "unhealthy",
            "port": self.allocated_port,
            "base_url": None,
            "libraries": [],
            "total_libraries": 0,
            "last_check": time.time(),
            "uptime": self.uptime,
            "error": None,
        }

        if not self.is_running:
            health_data["error"] = "Server not running"
            return health_data

        try:
            health_data["base_url"] = self.base_url

            # Test server connectivity
            health_response = await self._make_health_request()
            health_data["status"] = health_response["status"]
            if "error" in health_response:
                health_data["error"] = health_response["error"]

            # Get libraries information
            if health_data["status"] == "healthy":
                try:
                    libraries = await self._get_libraries()
                    health_data["libraries"] = libraries
                    health_data["total_libraries"] = len(libraries)
                except Exception as e:
                    logger.warning(f"Failed to get libraries: {e}")
                    health_data["status"] = "degraded"
                    health_data["error"] = f"Library check failed: {e}"

        except Exception as e:
            health_data["error"] = str(e)
            health_data["status"] = "unhealthy"

        self._last_health_check = time.time()
        return health_data

    async def _allocate_port(self) -> int:
        """Allocate a fresh dynamic port (never reuse default 6280).

        Always finds a truly available port to ensure we start our own
        docs-mcp-server with controlled docs-data path.
        """
        logger.info("Allocating fresh dynamic port for docs-mcp-server")

        # Start search from default_port + 1 to avoid conflicts
        for port in range(self.default_port + 1, self.default_port + 1000):
            if await self._is_port_truly_available(port):
                logger.info(f"Allocated dynamic port {port} for fresh docs-mcp-server")
                return port

        raise PortAllocationError(
            f"No available ports found in range {self.default_port + 1} to {self.default_port + 1000}"
        )

    async def _is_port_truly_available(self, port: int) -> bool:
        """Check if a port is truly available (not just open, but not in use by docs-mcp-server)."""
        logger.debug(f"Checking if port {port} is truly available...")
        try:
            # First check if port is open
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(("127.0.0.1", port))
                if result == 0:
                    # Port is open - check if it's docs-mcp-server
                    try:
                        async with aiohttp.ClientSession(
                            timeout=aiohttp.ClientTimeout(total=2)
                        ) as session:
                            async with session.get(
                                f"http://127.0.0.1:{port}/api/ping"
                            ) as response:
                                if response.status == 200:
                                    # Port is in use by a running server
                                    logger.info(
                                        f"Port {port} in use by existing docs-mcp-server"
                                    )
                                    return False
                    except Exception as e:
                        # Port is open but not responding to docs-mcp-server API
                        # This could be another service, so we'll avoid it
                        logger.info(
                            f"Port {port} in use by non-docs-mcp-server service (error: {e})"
                        )
                        return False
                # Port is not open, so it's available
                logger.debug(f"Port {port} is available")
                return True
        except Exception as e:
            logger.debug(f"Error checking port {port}: {e}")
            return False

    async def _start_docs_server_process(self) -> None:
        """Start the docs-mcp-server process."""
        # Import helper to build command
        from imas_mcp.services.docs_cli_helpers import (
            build_docs_server_command,
            get_npx_executable,
        )

        try:
            npx_executable = get_npx_executable()
        except RuntimeError as e:
            raise DocsServerError(str(e)) from e

        # Build command using helper
        cmd = build_docs_server_command(
            npx_executable,
            command="",  # Server mode, no command
            port=self.allocated_port,
            host="127.0.0.1",
            store_path=self.store_path,
        )

        logger.info(f"Starting docs server with command: {' '.join(cmd)}")
        logger.info(f"Store path: {self.store_path.absolute()}")
        logger.info(f"Using npx from: {npx_executable}")
        logger.info(f"Platform: {platform.system()}")
        logger.info(f"Python version: {sys.version}")

        try:
            # Prepare environment variables
            env = os.environ.copy()

            # Ensure PATH includes the directory containing npx
            # This is critical in CI environments where Node.js was just installed
            npx_dir = Path(npx_executable).parent
            if "PATH" in env:
                # Prepend npx directory to PATH to ensure it's found
                env["PATH"] = f"{npx_dir}{os.pathsep}{env['PATH']}"
            else:
                env["PATH"] = str(npx_dir)

            # Set required environment variables with defaults
            env.update(
                {
                    "DOCS_MCP_EMBEDDING_MODEL": env.get(
                        "DOCS_MCP_EMBEDDING_MODEL", "openai/text-embedding-3-small"
                    ),
                    "DOCS_MCP_TELEMETRY": env.get("DOCS_MCP_TELEMETRY", "false"),
                    "DOCS_MCP_STORE_PATH": str(self.store_path),
                }
            )

            # Log environment variables for debugging (mask sensitive data)
            api_key_status = "SET" if env.get("OPENAI_API_KEY") else "NOT SET"
            logger.info(f"  OPENAI_API_KEY: {api_key_status}")
            logger.info(f"  OPENAI_BASE_URL: {env.get('OPENAI_BASE_URL', 'NOT SET')}")
            logger.info(
                f"  DOCS_MCP_EMBEDDING_MODEL: {env.get('DOCS_MCP_EMBEDDING_MODEL')}"
            )
            logger.debug(f"  PATH: {env.get('PATH', 'NOT SET')[:200]}...")

            # Start the process with stdio capture for logging
            self.process = await anyio.open_process(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,  # Capture for logging
                stderr=subprocess.PIPE,  # Capture for logging
                env=env,  # Pass environment variables
            )
            self._start_time = time.time()

            # Log process information for debugging
            logger.info(f"Process started with PID: {self.process.pid}")

            # Start output logging tasks
            self._log_stdout_task = asyncio.create_task(
                self._log_process_output(self.process.stdout, "stdout")
            )
            self._log_stderr_task = asyncio.create_task(
                self._log_process_output(self.process.stderr, "stderr")
            )

            # Monitor process for unexpected termination
            self._monitor_task = asyncio.create_task(self._monitor_process())

        except Exception as e:
            logger.error(f"Failed to start docs server process: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise DocsServerError(
                f"Failed to start docs-mcp-server process: {str(e)}"
            ) from e

    async def _wait_for_server_ready(self, max_wait: int = 60) -> None:
        """Wait for the server to be ready to accept connections."""
        start_time = time.time()
        last_error = None
        attempt_count = 0

        # Use exponential backoff: 0.5s, 0.5s, 1s, 1s, 2s, 2s, 2s...
        wait_time = 0.5

        while time.time() - start_time < max_wait:
            attempt_count += 1

            if not self.is_running:
                # Collect any available stderr output for debugging
                error_msg = "Process terminated during startup"
                if self.process and hasattr(self.process, "returncode"):
                    error_msg += f" (exit code: {self.process.returncode})"

                # Try to capture stderr output if available
                if self.process and hasattr(self.process, "stderr"):
                    try:
                        # Read with a short timeout to avoid hanging
                        with anyio.fail_after(0.5):
                            stderr_data = await self.process.stderr.read()
                            if stderr_data:
                                stderr_text = stderr_data.decode(
                                    "utf-8", errors="replace"
                                ).strip()
                                if stderr_text:
                                    error_msg += f"\nStderr output: {stderr_text[:500]}"
                    except (TimeoutError, Exception) as e:
                        logger.debug(f"Could not read stderr: {e}")

                raise DocsServerError(error_msg)

            try:
                # Use shorter timeout for individual attempts
                timeout = aiohttp.ClientTimeout(total=min(5, self.timeout))
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"{self.base_url}/api/ping") as response:
                        if response.status == 200:
                            logger.info(
                                f"Docs server ready after {attempt_count} attempts ({time.time() - start_time:.1f}s)"
                            )
                            return
                        else:
                            last_error = f"HTTP {response.status}"
            except Exception as e:
                last_error = str(e)
                # Log every 10th attempt to avoid spamming logs
                if attempt_count % 10 == 0:
                    logger.debug(
                        f"Attempt {attempt_count}: Server not ready yet ({last_error})"
                    )

            # Exponential backoff with cap at 2 seconds
            await asyncio.sleep(wait_time)
            wait_time = min(wait_time * 1.5, 2.0)

        # Provide detailed error message on timeout
        error_parts = [f"Server did not become ready within {max_wait} seconds"]
        if last_error:
            error_parts.append(f"Last error: {last_error}")
        error_parts.append(f"Attempts made: {attempt_count}")

        # Check if process is still running
        if self.is_running:
            error_parts.append(
                "Process is still running but not responding to health checks"
            )
        else:
            error_parts.append("Process has terminated")

        raise DocsServerError(". ".join(error_parts))

    async def _get_libraries(self) -> list[dict[str, Any]]:
        """Get list of available libraries and their versions."""
        if (
            self._libraries_cache
            and isinstance(self._libraries_cache, list)
            and len(self._libraries_cache) > 0
            and time.time() - self._libraries_cache[-1].get("_timestamp", 0) < 300
        ):
            # Return cached data if less than 5 minutes old
            cache_data = [
                item for item in self._libraries_cache if not item.get("_timestamp")
            ]
            return cache_data

        try:
            libraries = await self._get_libraries_from_api()

            # Cache the result
            self._libraries_cache = libraries + [{"_timestamp": time.time()}]
            return libraries

        except Exception as e:
            logger.warning(f"Failed to get libraries: {e}")

        return []

    async def _attempt_recovery(self) -> None:
        """Attempt to recover from unexpected process termination."""
        try:
            await asyncio.sleep(5)  # Brief pause before recovery
            await self.start_server()
            logger.info("Docs server recovered successfully")
        except Exception as e:
            logger.error(f"Failed to recover docs server: {e}")

    async def _log_process_output(self, stream, prefix: str) -> None:
        """Read process output stream and log it."""
        try:
            async for line in stream:
                line_str = line.decode("utf-8", errors="replace").rstrip()
                if line_str:
                    logger.info(f"[docs-mcp-server {prefix}] {line_str}")
        except Exception as e:
            logger.debug(f"Error reading {prefix} stream: {e}")

    async def _monitor_process(self) -> None:
        """Monitor the docs server process for unexpected termination."""
        if not self.process:
            return

        await self.process.wait()

        if not self._shutdown_event.is_set():
            logger.error("Docs server process terminated unexpectedly")
            # Attempt automatic recovery
            asyncio.create_task(self._attempt_recovery())

    def _start_health_monitor(self) -> None:
        """Start background health monitoring."""
        if self._health_monitor_task and not self._health_monitor_task.done():
            return

        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                if self.is_running:
                    health_data = await self.health_check()
                    if health_data["status"] == "unhealthy":
                        logger.warning("Docs server unhealthy, attempting restart")
                        await self.restart_server()
                else:
                    logger.warning("Docs server not running, attempting restart")
                    await self.start_server()
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

            await asyncio.sleep(30)  # Check every 30 seconds

    # =====================================
    # PROXY METHODS FOR TOOL INTEGRATION
    # =====================================

    def get_proxy_service(self) -> "DocsProxyService":
        """Get or create a docs proxy service for tool integration."""
        if not hasattr(self, "_proxy_service") or self._proxy_service is None:
            # Import here to avoid circular imports
            from imas_mcp.services.docs_proxy_service import DocsProxyService

            self._proxy_service = DocsProxyService(docs_manager=self)
        return self._proxy_service

    async def proxy_list_libraries(self) -> list[str]:
        """Proxy method for listing documentation libraries."""
        try:
            await self.ensure_started()  # Ensure server is started
            proxy_service = self.get_proxy_service()
            return await proxy_service.proxy_list_libraries()
        except Exception as e:
            logger.error(f"Failed to proxy list_libraries: {e}")
            raise

    async def proxy_search_docs(
        self,
        query: str,
        library: str | None = None,
        version: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Proxy method for searching documentation."""
        try:
            await self.ensure_started()  # Ensure server is started
            proxy_service = self.get_proxy_service()
            return await proxy_service.proxy_search_docs(query, library, version, limit)
        except Exception as e:
            logger.error(f"Failed to proxy search_docs: {e}")
            raise

    async def ensure_started(self) -> None:
        """Ensure the docs server is started, starting it if necessary."""
        if not self._started:
            self._started = True
            await self.start_server()

    async def _check_existing_server(self) -> bool:
        """Check if there's already a docs-mcp-server running on the default port."""
        try:
            timeout = aiohttp.ClientTimeout(total=3)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Try to connect to the default port first
                test_url = f"http://127.0.0.1:{self.default_port}/api/ping"
                async with session.get(test_url) as response:
                    if response.status == 200:
                        # Server is running, use it
                        self.allocated_port = self.default_port
                        logger.info(
                            f"Found existing docs server on port {self.default_port}"
                        )
                        return True
        except Exception:
            # No server found on default port
            pass

        return False

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get an aiohttp ClientSession with configured timeout."""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        return aiohttp.ClientSession(timeout=timeout)

    async def _make_health_request(self) -> dict[str, Any]:
        """Make a health check request to the docs server."""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{self.base_url}/api/ping") as response:
                if response.status == 200:
                    return {"status": "healthy"}
                else:
                    return {"status": "degraded", "error": f"HTTP {response.status}"}

    async def _get_libraries_from_api(self) -> list[dict[str, Any]]:
        """Get libraries data from the docs server API."""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{self.base_url}/api/list") as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_libraries_response(data)
        return []

    def _parse_libraries_response(self, data: Any) -> list[dict[str, Any]]:
        """Parse the response from the libraries endpoint."""
        libraries = []

        if isinstance(data, dict) and "libraries" in data:
            for lib_info in data["libraries"]:
                if isinstance(lib_info, dict) and "name" in lib_info:
                    libraries.append(
                        {
                            "name": lib_info["name"],
                            "versions": lib_info.get("versions", []),
                            "status": "available",
                        }
                    )
                elif isinstance(lib_info, str):
                    libraries.append(
                        {
                            "name": lib_info,
                            "versions": [],
                            "status": "available",
                        }
                    )

        return libraries

    @staticmethod
    def find_docs_server_processes() -> list[dict[str, Any]]:
        """Find all running docs-mcp-server processes by scanning ports 6280-6290.

        Returns:
            List of dicts with keys: pid, port, cmdline
        """
        found_processes = []

        # First, quickly check which ports are actually open using direct socket testing
        open_ports = []
        for port in range(DocsServerManager.MIN_PORT, DocsServerManager.MAX_PORT + 1):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)
                result = sock.connect_ex(("127.0.0.1", port))
                sock.close()

                if result == 0:  # Port is open
                    open_ports.append(port)
                    logger.debug(f"Port {port} is open")
            except Exception as e:
                logger.debug(f"Error checking port {port}: {e}")

        # If no ports are open, return early
        if not open_ports:
            return found_processes

        # Now get all connections ONCE and filter for our open ports
        try:
            all_connections = psutil.net_connections(kind="inet")

            for conn in all_connections:
                if (
                    conn.status == psutil.CONN_LISTEN
                    and conn.laddr.port in open_ports
                    and conn.pid
                ):
                    try:
                        proc = psutil.Process(conn.pid)
                        cmdline = " ".join(proc.cmdline())

                        # Check if this is a docs-mcp-server process
                        if "docs-mcp-server" in cmdline or (
                            "node" in proc.name().lower()
                            and "docs-mcp-server" in cmdline
                        ):
                            found_processes.append(
                                {
                                    "pid": conn.pid,
                                    "port": conn.laddr.port,
                                    "cmdline": cmdline,
                                }
                            )
                            logger.debug(
                                f"Found docs-mcp-server on port {conn.laddr.port}, PID {conn.pid}"
                            )
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
        except (psutil.AccessDenied, OSError) as e:
            logger.debug(f"Error getting network connections: {e}")

        return found_processes

    @staticmethod
    def kill_orphaned_servers(verbose: bool = True) -> int:
        """Kill all orphaned docs-mcp-server processes.

        Args:
            verbose: If True, log information about killed processes

        Returns:
            Number of processes terminated
        """
        processes = DocsServerManager.find_docs_server_processes()
        killed_count = 0

        for proc_info in processes:
            try:
                proc = psutil.Process(proc_info["pid"])
                if verbose:
                    logger.info(
                        f"Terminating docs-mcp-server on port {proc_info['port']}, PID {proc_info['pid']}"
                    )
                proc.terminate()

                # Wait up to 5 seconds for graceful termination
                try:
                    proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    if verbose:
                        logger.warning(
                            f"Process {proc_info['pid']} didn't terminate gracefully, killing..."
                        )
                    proc.kill()
                    proc.wait(timeout=2)

                killed_count += 1
                if verbose:
                    logger.info(f"Successfully terminated PID {proc_info['pid']}")

            except psutil.NoSuchProcess:
                if verbose:
                    logger.debug(f"Process {proc_info['pid']} already terminated")
            except psutil.AccessDenied:
                if verbose:
                    logger.error(
                        f"Access denied when trying to terminate PID {proc_info['pid']}"
                    )
            except Exception as e:
                if verbose:
                    logger.error(f"Error terminating process {proc_info['pid']}: {e}")

        return killed_count

    # Context manager support
    async def __aenter__(self):
        await self.start_server()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop_server()

    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down docs server...")
            asyncio.create_task(self.stop_server())
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

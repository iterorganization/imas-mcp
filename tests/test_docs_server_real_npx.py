import asyncio
import os
import random
import shutil
import subprocess

import pytest

from imas_mcp.services.docs_server_manager import DocsServerManager


def npx_is_functional() -> bool:
    """Check if npx can actually execute, not just if it exists in PATH.

    Returns:
        True if npx --version succeeds, False otherwise
    """
    try:
        result = subprocess.run(
            ["npx", "--version"],
            capture_output=True,
            timeout=5,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.mark.asyncio
@pytest.mark.skipif(
    not npx_is_functional(),
    reason="npx not functional (may exist but cannot execute)",
)
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required for docs-mcp-server startup",
)
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
async def test_docs_server_startup_real_npx():
    """
    Test that the docs server can actually start up if npx is available.
    This test is skipped if npx is not in the PATH or OPENAI_API_KEY is not set.

    Reliability improvements:
    - Cleans up orphaned processes before starting
    - Increased timeout to 120s for slower CI environments
    - Retries once on failure
    - Uses random high port to avoid conflicts
    - Exponential backoff in health checks
    """
    # Clean up any orphaned docs-mcp-server processes first
    killed = DocsServerManager.kill_orphaned_servers(verbose=False)
    if killed > 0:
        # Give OS time to fully release ports
        await asyncio.sleep(2)

    # Use a random high port to avoid conflicts
    test_port = random.randint(19200, 19299)

    # Increase timeout for slower environments (CI, heavy load)
    manager = DocsServerManager(default_port=test_port, timeout=120)
    max_retries = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            await manager.start_server()
            assert manager.is_running

            # Perform a health check
            health = await manager.health_check()
            assert health["status"] == "healthy"

            # If we get here, test passed
            return

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Cleanup before retry
                if manager.is_running:
                    await manager.stop_server()
                await asyncio.sleep(3)
            else:
                # Final attempt failed
                raise
        finally:
            if manager.is_running:
                await manager.stop_server()

    # If we exhausted retries, raise the last error
    if last_error:
        raise last_error

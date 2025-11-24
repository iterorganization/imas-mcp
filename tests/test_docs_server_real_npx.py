import asyncio
import os
import random
import shutil

import pytest

from imas_mcp.services.docs_server_manager import DocsServerManager


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("npx") is None, reason="npx not available")
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

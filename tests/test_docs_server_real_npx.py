import os
import shutil

import pytest

from imas_mcp.services.docs_server_manager import DocsServerManager


def _has_valid_openai_key() -> bool:
    """Check if a valid-looking OPENAI_API_KEY is present."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    # OpenAI keys start with 'sk-' and are at least 20 chars
    return bool(key and key.startswith("sk-") and len(key) > 20)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not shutil.which("npx"),
    reason="npx not found in PATH",
)
@pytest.mark.skipif(
    not _has_valid_openai_key(),
    reason="Valid OPENAI_API_KEY (starts with 'sk-') required for docs-mcp-server startup",
)
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
async def test_docs_server_startup_real_npx():
    """Test that the docs server can start up with npx and OPENAI_API_KEY.

    This is a simple integration test - if it fails, check:
    1. OPENAI_API_KEY is set and valid
    2. npx is in PATH
    3. Network connectivity for downloading docs-mcp-server
    """
    manager = DocsServerManager(default_port=6280)

    try:
        await manager.start_server()
        assert manager.is_running

        # Verify server responds
        health = await manager.health_check()
        assert health["status"] == "healthy"
    finally:
        await manager.stop_server()

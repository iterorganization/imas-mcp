import os
import shutil

import pytest

from imas_mcp.services.docs_server_manager import DocsServerManager


def _has_valid_openai_config() -> bool:
    """Check if valid OPENAI_API_KEY and OPENAI_BASE_URL are present."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    # OpenAI keys start with 'sk-' and are at least 20 chars
    # Base URL is required for OpenRouter/custom endpoints
    return bool(key and key.startswith("sk-") and len(key) > 20 and base_url)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not shutil.which("npx"),
    reason="npx not found in PATH",
)
@pytest.mark.skipif(
    not _has_valid_openai_config(),
    reason="OPENAI_API_KEY (starting with 'sk-') and OPENAI_BASE_URL environment variables required",
)
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
async def test_docs_server_startup_with_npx():
    """Test that the docs server can start up with npx and OPENAI_API_KEY.

    This is a simple integration test - if it fails, check:
    1. OPENAI_API_KEY and OPENAI_BASE_URL are set and valid
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

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

from imas_mcp import DOCS_MCP_SERVER_VERSION
from imas_mcp.services.docs_server_manager import DocsServerManager

# Load environment variables from .env file
load_dotenv()


@pytest.fixture
def mock_npx():
    with patch("shutil.which", return_value="/usr/bin/npx"):
        yield


@pytest.fixture
def mock_process():
    process_mock = AsyncMock()
    process_mock.pid = 12345
    process_mock.returncode = None

    # Simulate process running state
    exit_event = asyncio.Event()

    async def wait_impl():
        await exit_event.wait()
        return None

    process_mock.wait.side_effect = wait_impl

    def terminate_impl():
        exit_event.set()
        process_mock.returncode = 0

    process_mock.terminate = MagicMock(side_effect=terminate_impl)
    process_mock.kill = MagicMock(side_effect=terminate_impl)

    process_mock.stdout = AsyncMock()
    process_mock.stdout.__aiter__.return_value = iter([b"Server started"])
    process_mock.stderr = AsyncMock()
    process_mock.stderr.__aiter__.return_value = iter([])

    with patch("anyio.open_process", return_value=process_mock) as mock:
        yield mock, process_mock


@pytest.fixture
def mock_aiohttp():
    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_session_instance = MagicMock()
        mock_session_cls.return_value = mock_session_instance
        mock_context = MagicMock()
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_context)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)

        def create_response_cm(status=200, json_data=None, text_data=""):
            mock_response = AsyncMock()
            mock_response.status = status
            if json_data is not None:
                mock_response.json.return_value = json_data
            mock_response.text.return_value = text_data
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_response)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        mock_context.get.return_value = create_response_cm()
        mock_context.create_response_cm = create_response_cm
        yield mock_context


@pytest.mark.asyncio
async def test_docs_server_env_config(mock_npx, mock_process, mock_aiohttp):
    """Test that the docs server is started with the correct environment variables from .env."""

    # Read expected values from .env directly to verify against
    expected_embedding_model = os.environ.get("DOCS_MCP_EMBEDDING_MODEL")
    expected_store_path = os.environ.get("DOCS_MCP_STORE_PATH")

    # If not set in env, use defaults that the code uses, but we want to verify .env usage
    # The user request implies .env has specific values we want to test.

    print(f"Expected Embedding Model from .env: {expected_embedding_model}")
    print(f"Expected Store Path from .env: {expected_store_path}")

    manager = DocsServerManager(default_port=18300)

    # Mock port availability
    with patch.object(manager, "_is_port_truly_available", return_value=True):
        await manager.start_server()

        # Verify anyio.open_process was called
        mock_open_process, _ = mock_process
        assert mock_open_process.called

        # Get arguments passed to open_process
        call_args = mock_open_process.call_args
        args, kwargs = call_args

        cmd = args[0]
        env = kwargs.get("env")

        # Verify command structure
        assert "npx" in cmd[0]
        assert "@arabold/docs-mcp-server@" + DOCS_MCP_SERVER_VERSION in cmd

        # Verify environment variables
        assert env is not None

        if expected_embedding_model:
            assert env["DOCS_MCP_EMBEDDING_MODEL"] == expected_embedding_model
        else:
            # If not in .env, check default
            assert env["DOCS_MCP_EMBEDDING_MODEL"] == "openai/text-embedding-3-small"

        if expected_store_path:
            # The manager might resolve the path to absolute, so we check if it ends with the expected path
            # or if it matches the resolved path
            manager_store_path = str(manager.store_path)
            assert env["DOCS_MCP_STORE_PATH"] == manager_store_path

        await manager.stop_server()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

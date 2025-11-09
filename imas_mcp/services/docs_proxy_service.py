"""
Docs Proxy Service for docs-mcp-server communication.

This service handles communication with the docs-mcp-server via npx subprocess calls
to provide generic documentation search capabilities across multiple libraries.
"""

import json
import logging
import os
from typing import Any

import anyio

from imas_mcp.exceptions import DocsServerError

logger = logging.getLogger(__name__)


class Settings:
    """Simple settings class for documentation search configuration"""

    def __init__(self):
        # Docs server configuration
        self.docs_timeout: int = int(os.getenv("DOCS_TIMEOUT", "30"))
        self.default_docs_limit: int = int(os.getenv("DEFAULT_DOCS_LIMIT", "5"))
        self.max_docs_limit: int = int(os.getenv("MAX_DOCS_LIMIT", "20"))


class DocsServerUnavailableError(DocsServerError):
    """Raised when docs-mcp-server is not accessible"""

    def __init__(self):
        message = (
            "docs-mcp-server is not available via npx\n\n"
            "To install and use the docs server:\n"
            "1. Install: npm install -g @arabold/docs-mcp-server@latest\n"
            "2. Ensure Node.js and npm are installed\n"
            "3. The service will automatically use npx to run commands\n\n"
            "Note: This service delegates all operations to the docs-mcp-server via npx."
        )
        super().__init__(message)


class LibraryNotFoundError(DocsServerError):
    """Raised when requested documentation library is not found"""

    def __init__(self, library: str, available_libraries: list[str]):
        self.library = library
        self.available_libraries = available_libraries
        message = (
            f"Documentation library '{library}' not found\n\n"
            f"Available libraries: {', '.join(available_libraries)}\n\n"
            f"To add a new library:\n"
            f"1. Use the add_docs script: python scripts/add_docs.py <library> <url>\n"
            f"2. Or run: npx @arabold/docs-mcp-server@latest scrape <library> <url>"
        )
        super().__init__(message)


class DocsProxyService:
    """
    Service for communicating with docs-mcp-server to provide documentation search.

    This service handles:
    - Generic documentation search across multiple libraries via npx
    - Library management and version handling
    - Error handling with helpful setup instructions
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize the docs proxy service"""
        self.settings = settings or Settings()
        self.timeout = self.settings.docs_timeout

    async def _run_npx_command(self, command_args: list[str]) -> dict[str, Any]:
        """Run npx command with docs-mcp-server and return parsed JSON result"""
        cmd = ["npx", "@arabold/docs-mcp-server@latest"] + command_args

        try:
            # Use anyio's run_process for better async subprocess handling
            async with await anyio.open_process(cmd) as process:
                with anyio.fail_after(self.timeout):
                    stdout, stderr = await process.communicate()

                # Check the return code
                if process.returncode != 0:
                    error_msg = (
                        stderr.decode("utf-8").strip()
                        if stderr
                        else f"Command failed with code {process.returncode}"
                    )
                    raise DocsServerError(f"npx command failed: {error_msg}")

            # Parse JSON output
            stdout_text = stdout.decode("utf-8").strip()
            if stdout_text:
                try:
                    return json.loads(stdout_text)
                except json.JSONDecodeError as e:
                    raise DocsServerError(
                        f"Failed to parse JSON output: {str(e)}"
                    ) from e
            else:
                return {}

        except TimeoutError:
            raise DocsServerError(
                f"Command timed out after {self.timeout} seconds"
            ) from None
        except FileNotFoundError:
            raise DocsServerUnavailableError() from None
        except Exception as e:
            if isinstance(e, DocsServerError):
                raise
            raise DocsServerError(
                f"Unexpected error running npx command: {str(e)}"
            ) from e

    async def search_docs(
        self,
        query: str,
        library: str | None = None,
        limit: int | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        """
        Search documentation across indexed libraries.

        Args:
            query: Search query string
            library: Optional specific library to search
            limit: Maximum number of results (1-20)
            version: Specific version to search

        Returns:
            Dictionary containing search results and metadata
        """
        # Set defaults
        if limit is None:
            limit = self.settings.default_docs_limit
        limit = max(1, min(limit, self.settings.max_docs_limit))

        # If no library specified, we need to search all libraries
        # docs-mcp-server requires a library, so we'll need to handle this differently
        if not library:
            # For now, return an error since the CLI requires a library
            # In the future, we could implement a search across all libraries
            return {
                "error": "Library parameter is required for search. Use list_available_libraries() to see available libraries.",
                "query": query,
                "library": library,
                "version": version,
                "library_required": True,
            }

        # Build command arguments
        # docs-mcp-server search syntax: search [options] <library> <query>
        command_args = ["search", "--limit", str(limit)]

        if version:
            command_args.extend(["--version", version])

        # Add library and query as positional arguments (in correct order)
        command_args.extend([library, query])

        try:
            result = await self._run_npx_command(command_args)

            # Add query and library info to result
            if isinstance(result, dict):
                result["query"] = query
                result["library"] = library
                result["version"] = version

            return result

        except DocsServerError:
            # Re-raise docs server errors as-is
            raise
        except Exception as e:
            raise DocsServerError(f"Search failed: {str(e)}") from e

    async def search_imas_python_docs(
        self, query: str, limit: int | None = None, version: str | None = None
    ) -> dict[str, Any]:
        """
        Search specifically in IMAS-Python documentation.

        Args:
            query: Search query string
            limit: Maximum number of results (1-20)
            version: Specific IMAS-Python version to search

        Returns:
            Dictionary containing search results and metadata
        """
        return await self.search_docs(
            query=query, library="imas-python", limit=limit, version=version
        )

    async def list_available_libraries(self) -> list[str]:
        """
        List all indexed documentation libraries.

        Returns:
            List of available library names
        """
        try:
            result = await self._run_npx_command(["list"])

            if isinstance(result, dict) and "libraries" in result:
                # Extract library names from the result
                libraries = []
                for lib_info in result["libraries"]:
                    if isinstance(lib_info, dict) and "name" in lib_info:
                        libraries.append(lib_info["name"])
                    elif isinstance(lib_info, str):
                        libraries.append(lib_info)
                return libraries
            elif isinstance(result, list):
                return result
            else:
                logger.warning(f"Unexpected libraries response format: {type(result)}")
                return []

        except DocsServerError:
            # Re-raise docs server errors as-is
            raise
        except Exception as e:
            raise DocsServerError(f"Failed to list libraries: {str(e)}") from e

    async def get_library_versions(self, library: str) -> list[str]:
        """
        Get available versions for a specific library.

        Args:
            library: Library name to get versions for

        Returns:
            List of available versions (newest first)
        """
        try:
            # First check if library exists
            libraries = await self.list_available_libraries()
            if library not in libraries:
                raise LibraryNotFoundError(library, libraries)

            # Use find-version command to get version info
            # The CLI doesn't have a direct versions command, so we'll use find-version
            # and potentially need to adjust this based on actual output
            result = await self._run_npx_command(["find-version", library])

            if isinstance(result, dict):
                # The find-version command might return version information
                if "versions" in result:
                    return result["versions"]
                elif "version" in result:
                    return [result["version"]]
                elif "best_match" in result:
                    return [result["best_match"]]
                else:
                    # If we can't parse versions, return empty list
                    logger.warning(f"Unexpected versions response format: {result}")
                    return []
            elif isinstance(result, list):
                return result
            elif isinstance(result, str):
                return [result]
            else:
                logger.warning(f"Unexpected versions response format: {type(result)}")
                return []

        except LibraryNotFoundError:
            # Re-raise library not found errors as-is
            raise
        except DocsServerError:
            # Re-raise docs server errors as-is
            raise
        except Exception as e:
            raise DocsServerError(
                f"Failed to get versions for {library}: {str(e)}"
            ) from e

    async def trigger_scrape(
        self,
        library: str,
        version: str,
        url: str,
        max_pages: int | None = None,
        max_depth: int | None = None,
    ) -> str:
        """
        Trigger scraping of new documentation library/version.

        Args:
            library: Library name to create/update
            version: Version string
            url: Documentation URL to scrape
            max_pages: Optional maximum pages to scrape
            max_depth: Optional maximum depth to scrape

        Returns:
            Job ID for tracking scraping progress
        """
        command_args = ["scrape", library, url, "--version", version]

        if max_pages:
            command_args.extend(["--max-pages", str(max_pages)])
        if max_depth:
            command_args.extend(["--max-depth", str(max_depth)])

        try:
            result = await self._run_npx_command(command_args)

            if isinstance(result, dict) and "job_id" in result:
                return result["job_id"]
            elif isinstance(result, str):
                return result
            else:
                raise DocsServerError(f"Unexpected scrape response format: {result}")

        except DocsServerError:
            # Re-raise docs server errors as-is
            raise
        except Exception as e:
            raise DocsServerError(f"Failed to trigger scraping: {str(e)}") from e

    async def get_scrape_status(self, job_id: str) -> dict[str, Any]:
        """
        Get status of a scraping job.

        Args:
            job_id: Job ID returned from trigger_scrape

        Returns:
            Dictionary containing job status and progress
        """
        try:
            result = await self._run_npx_command(["scrape", "status", job_id])
            return (
                result
                if isinstance(result, dict)
                else {"status": "unknown", "job_id": job_id}
            )

        except DocsServerError:
            # Re-raise docs server errors as-is
            raise
        except Exception as e:
            raise DocsServerError(f"Failed to get scrape status: {str(e)}") from e

    async def validate_library_exists(self, library: str) -> bool:
        """
        Check if a library is available for searching.

        Args:
            library: Library name to check

        Returns:
            True if library exists, False otherwise
        """
        try:
            libraries = await self.list_available_libraries()
            return library in libraries
        except DocsServerError:
            return False

    async def find_best_version(self, library: str, target_version: str) -> str:
        """
        Find the best matching version for a library.

        Args:
            library: Library name
            target_version: Requested version (can be "latest" or specific version)

        Returns:
            Best matching version string
        """
        if target_version == "latest":
            versions = await self.get_library_versions(library)
            return versions[0] if versions else "latest"
        else:
            return target_version

    async def close(self) -> None:
        """Close method for compatibility - no persistent connections in npx approach"""
        pass

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

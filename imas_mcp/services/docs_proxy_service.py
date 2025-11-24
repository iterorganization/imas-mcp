"""
Docs Proxy Service for docs-mcp-server communication.

This service handles communication with the docs-mcp-server via HTTP client
to provide generic documentation search capabilities across multiple libraries.
"""

import json
import logging
import os
from typing import Any

import aiohttp

from imas_mcp import DOCS_MCP_SERVER_VERSION
from imas_mcp.exceptions import DocsServerError
from imas_mcp.services.docs_server_manager import (
    DocsServerManager,
    DocsServerUnavailableError,
    PortAllocationError,
)

logger = logging.getLogger(__name__)


class Settings:
    """Simple settings class for documentation search configuration"""

    def __init__(self):
        # Docs server configuration
        self.docs_timeout: int = int(os.getenv("DOCS_TIMEOUT", "30"))
        self.default_docs_limit: int = int(os.getenv("DEFAULT_DOCS_LIMIT", "5"))
        self.max_docs_limit: int = int(os.getenv("MAX_DOCS_LIMIT", "20"))


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
            f"2. Or run: npx @arabold/docs-mcp-server@{DOCS_MCP_SERVER_VERSION} scrape <library> <url>"
        )
        super().__init__(message)


class DocsProxyService:
    """
    Service for communicating with docs-mcp-server to provide documentation search.

    This service handles:
    - Generic documentation search across multiple libraries via HTTP
    - MCP protocol tool proxying via /mcp endpoint
    - Library management and version handling
    - Error handling with helpful setup instructions
    """

    def __init__(
        self,
        settings: Settings | None = None,
        docs_manager: DocsServerManager | None = None,
    ):
        """Initialize the docs proxy service"""
        self.settings = settings or Settings()
        self.timeout = self.settings.docs_timeout
        self.docs_manager = docs_manager or DocsServerManager()

    async def _make_http_request(
        self, endpoint: str, method: str = "GET", **kwargs
    ) -> dict[str, Any]:
        """Make HTTP request to docs server and return parsed JSON result"""
        try:
            if not self.docs_manager.is_running:
                await self.docs_manager.start_server()
        except Exception as e:
            # Provide more specific error messages for common startup issues
            error_msg = str(e)
            if (
                "WinError 2" in error_msg
                or "system cannot find the file" in error_msg.lower()
            ):
                raise DocsServerError(
                    f"Failed to start docs-mcp-server: {error_msg}. "
                    "This usually means Node.js/npx is not available. "
                    "Please install Node.js from https://nodejs.org/ and ensure npx is in your PATH."
                ) from e
            elif "npm" in error_msg.lower() or "node" in error_msg.lower():
                raise DocsServerError(
                    f"Node.js dependency issue: {error_msg}. "
                    "Please ensure Node.js and npm are properly installed and accessible."
                ) from e
            else:
                raise DocsServerError(
                    f"Failed to start documentation server: {str(e)}"
                ) from e

        url = f"{self.docs_manager.base_url}/api/{endpoint}"

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.request(method, url, **kwargs) as response:
                    # Check the return code
                    if response.status != 200:
                        error_text = await response.text()
                        raise DocsServerError(f"HTTP {response.status}: {error_text}")

                    # Parse JSON output
                    text = await response.text()
                    if text:
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError as e:
                            raise DocsServerError(
                                f"Failed to parse JSON response: {str(e)}"
                            ) from e
                    else:
                        return {}

        except aiohttp.ClientError as e:
            if "Connection refused" in str(e) or "Cannot connect" in str(e):
                raise DocsServerUnavailableError() from e
            raise DocsServerError(f"HTTP request failed: {str(e)}") from e
        except Exception as e:
            if isinstance(
                e, DocsServerError | DocsServerUnavailableError | PortAllocationError
            ):
                raise
            # Enhance error messages for common issues
            error_msg = str(e)
            if "WinError 2" in error_msg:
                raise DocsServerError(
                    f"Documentation server process failed to start: {error_msg}. "
                    "This is typically due to Node.js/npx not being available on Windows."
                ) from e
            raise DocsServerError(f"Unexpected error: {str(e)}") from e

    async def _make_mcp_request(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make MCP protocol request to docs-mcp-server via /mcp endpoint"""
        if not self.docs_manager.is_running:
            await self.docs_manager.start_server()

        mcp_url = f"{self.docs_manager.base_url}/mcp"

        # Construct MCP request
        mcp_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments or {}},
        }

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                headers = {
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                }
                async with session.post(
                    mcp_url, json=mcp_request, headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise DocsServerError(f"HTTP {response.status}: {error_text}")

                    # Check content type and handle accordingly
                    content_type = response.headers.get("content-type", "").lower()

                    if "application/json" in content_type:
                        result = await response.json()
                    elif "text/event-stream" in content_type:
                        # Handle SSE response by reading the first line
                        text = await response.text()
                        # Parse the first JSON line from SSE
                        for line in text.split("\n"):
                            line = line.strip()
                            if line.startswith("data: "):
                                try:
                                    result = json.loads(
                                        line[6:]
                                    )  # Remove 'data: ' prefix
                                    break
                                except json.JSONDecodeError:
                                    continue
                        else:
                            raise DocsServerError(
                                "No valid JSON data found in SSE response"
                            )
                    else:
                        # Try to parse as JSON anyway
                        text = await response.text()
                        try:
                            result = json.loads(text)
                        except json.JSONDecodeError as err:
                            raise DocsServerError(
                                f"Unexpected response content type: {content_type}"
                            ) from err

                    # Handle MCP protocol response
                    if "error" in result:
                        raise DocsServerError(f"MCP Error: {result['error']}")

                    if "result" in result and "content" in result["result"]:
                        # Extract the actual tool result
                        tool_result = result["result"]["content"]
                        if isinstance(tool_result, list) and len(tool_result) > 0:
                            # Extract the text content
                            text = tool_result[0].get("text", "")
                            # Return raw text for caller to parse appropriately
                            return {"text": text}
                        elif isinstance(tool_result, dict):
                            return tool_result
                        else:
                            return {"text": str(tool_result)}

                    return result

        except aiohttp.ClientError as e:
            if "Connection refused" in str(e) or "Cannot connect" in str(e):
                raise DocsServerUnavailableError() from e
            raise DocsServerError(f"HTTP request failed: {str(e)}") from e
        except Exception as e:
            if isinstance(e, DocsServerUnavailableError | PortAllocationError):
                raise
            raise DocsServerError(f"Unexpected error: {str(e)}") from e

    def _parse_search_results(self, text: str) -> dict[str, Any]:
        """Parse the formatted search results text to extract structured data."""
        # The docs-mcp-server formats results as:
        # "------------------------------------------------------------\nResult 1: URL\n\nContent\n"

        results = []
        lines = text.split("\n")
        current_result = None
        content_lines = []

        for line in lines:
            if line.startswith("Result ") and ":" in line:
                # Save previous result if exists
                if current_result is not None:
                    current_result["content"] = "\n".join(content_lines).strip()
                    results.append(current_result)

                # Start new result
                url_part = line.split(":", 1)[1].strip() if ":" in line else ""
                current_result = {
                    "url": url_part,
                    "content": "",
                    "score": None,
                    "mimeType": "text/plain",
                }
                content_lines = []
            elif current_result is not None and line.strip() == "":
                # Empty line - content separator
                continue
            elif current_result is not None and not line.startswith("---"):
                # Content line
                content_lines.append(line)

        # Save last result
        if current_result is not None:
            current_result["content"] = "\n".join(content_lines).strip()
            results.append(current_result)

        return {"results": results, "count": len(results)}

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

        # Build request parameters
        search_params = {"library": library, "query": query, "limit": limit}
        if version:
            search_params["version"] = version

        # Add library and query as positional arguments
        # docs-mcp-server search syntax: search [options] <library> <query>
        try:
            result = await self._make_http_request(
                "trpc/search", method="POST", json=search_params
            )

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

    async def list_available_libraries(self) -> list[str]:
        """
        List all indexed documentation libraries (names only).

        Returns:
            List of available library names
        """
        try:
            result = await self._make_http_request("trpc/listLibraries")

            if isinstance(result, dict) and "result" in result:
                data = result["result"]["data"]
                if isinstance(data, list):
                    # Extract library names from the result
                    libraries = []
                    for lib_info in data:
                        if isinstance(lib_info, dict) and "library" in lib_info:
                            libraries.append(lib_info["library"])
                        elif isinstance(lib_info, dict) and "name" in lib_info:
                            libraries.append(lib_info["name"])
                        elif isinstance(lib_info, str):
                            libraries.append(lib_info)
                    return libraries
                else:
                    logger.warning(f"Unexpected libraries data format: {type(data)}")
                    return []
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

    async def list_libraries_with_versions(self) -> list[dict[str, Any]]:
        """
        List all indexed documentation libraries with full version information.

        Returns:
            List of library info dicts with structure:
            {
                "name": str,
                "versions": [
                    {
                        "version": str,
                        "documentCount": int,
                        "uniqueUrlCount": int,
                        "indexedAt": str | None,
                        "status": str,
                        "progress": {"pages": int, "maxPages": int} | None,
                        "sourceUrl": str | None
                    }
                ]
            }
        """
        try:
            # The newer docs-mcp-server exposes MCP tools at /mcp endpoint
            # We need to call the list_docs tool via MCP protocol
            # For now, use direct HTTP to the database file if available
            # or return empty until proper MCP client integration is added

            # Try the ping endpoint to verify server is running
            await self._make_http_request("ping")

            # Return empty list with a note about the API change
            logger.info(
                "docs-mcp-server is running but library listing via HTTP API is not yet implemented for the new version"
            )
            return []

        except DocsServerError:
            # Re-raise docs server errors as-is
            raise
        except Exception as e:
            raise DocsServerError(
                f"Failed to list libraries with versions: {str(e)}"
            ) from e

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

            # Use findBestVersion tRPC endpoint to get version info
            result = await self._make_http_request(
                "trpc/findBestVersion", method="POST", json={"library": library}
            )

            if isinstance(result, dict) and "result" in result:
                data = result["result"]["data"]
                if isinstance(data, dict):
                    # The findBestVersion returns version information
                    if "versions" in data:
                        return data["versions"]
                    elif "best_match" in data:
                        return [data["best_match"]]
                    elif "version" in data:
                        return [data["version"]]
                    else:
                        # If we can't parse versions, return empty list
                        logger.warning(f"Unexpected versions response format: {data}")
                        return []
                elif isinstance(data, list):
                    return data
                elif isinstance(data, str):
                    return [data]
                else:
                    logger.warning(f"Unexpected versions data format: {type(data)}")
                    return []
            elif isinstance(result, list):
                return result
            elif isinstance(result, str):
                return [result]
            else:
                logger.warning(f"Unexpected versions response format: {type(result)}")
                return []

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
        params = {"version": version}
        if max_pages:
            params["max_pages"] = str(max_pages)
        if max_depth:
            params["max_depth"] = str(max_depth)

        try:
            result = await self._make_http_request(
                f"scrape/{library}/{url}", params=params
            )

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
            result = await self._make_http_request(f"scrape/status/{job_id}")
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

    async def proxy_list_libraries(self) -> list[str]:
        """Proxy call to docs-mcp-server list_libraries tool"""
        try:
            result = await self._make_mcp_request("list_libraries")

            # The MCP tool returns a text response with library names
            # Format: "Indexed libraries:\n\n- library1\n- library2"
            if "text" in result:
                text = result["text"]
                logger.debug(f"list_libraries text response: {text}")

                # Check for "No libraries indexed yet." message
                if "No libraries indexed yet" in text:
                    return []

                libraries = []
                # Look for lines that start with "- " (bullet points)
                for line in text.split("\n"):
                    line = line.strip()
                    if line.startswith("- "):
                        # Extract library name after the bullet point
                        lib_name = line[2:].strip()
                        if lib_name:
                            libraries.append(lib_name)

                logger.info(f"Found {len(libraries)} libraries: {libraries}")
                return libraries
            else:
                logger.warning(
                    f"Unexpected result format from list_libraries: {result}"
                )
                return []
        except Exception as e:
            logger.error(f"Failed to proxy list_libraries: {e}")
            raise DocsServerError(f"Failed to list libraries: {str(e)}") from e

    async def proxy_search_docs(
        self,
        query: str,
        library: str | None = None,
        version: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Proxy call to docs-mcp-server search_docs tool"""
        try:
            arguments = {
                "query": query,
                "limit": limit or 5,  # Provide default if None
            }
            if library:
                arguments["library"] = library
            if version:
                arguments["version"] = version

            result = await self._make_mcp_request("search_docs", arguments)

            # Parse the formatted text results from search_docs
            if "text" in result:
                logger.info(
                    f"Raw search results for query='{query}' library='{library}':\n{result['text'][:500]}..."
                )
                parsed = self._parse_search_results(result["text"])
                parsed["query"] = query
                parsed["library"] = library
                parsed["version"] = version
                parsed["success"] = True
                return parsed

            # Fallback for unexpected format
            return {
                "results": [
                    {"content": result.get("text", ""), "url": "docs-mcp-server"}
                ],
                "query": query,
                "library": library,
                "version": version,
                "count": 1,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Failed to proxy search_docs: {e}")
            raise DocsServerError(f"Search failed: {str(e)}") from e

    async def proxy_add_library(
        self, library: str, url: str, max_pages: int | None = None
    ) -> dict[str, Any]:
        """Add a new documentation library by scraping a URL.

        Args:
            library: Library name to create
            url: Documentation URL to scrape
            max_pages: Optional maximum number of pages to scrape

        Returns:
            Dictionary with job_id and status information
        """
        try:
            arguments = {"library": library, "url": url}
            if max_pages:
                arguments["maxPages"] = max_pages

            result = await self._make_mcp_request("add_docs", arguments)

            if "text" in result:
                # Parse the text response to extract job info
                text = result["text"]
                logger.info(f"Add library response: {text}")
                return {
                    "success": True,
                    "library": library,
                    "url": url,
                    "message": text,
                }

            return {
                "success": True,
                "library": library,
                "url": url,
                "result": result,
            }
        except Exception as e:
            logger.error(f"Failed to add library: {e}")
            raise DocsServerError(f"Failed to add library: {str(e)}") from e

    async def proxy_remove_library(self, library: str) -> dict[str, Any]:
        """Remove a documentation library.

        Args:
            library: Library name to remove

        Returns:
            Dictionary with success status and message
        """
        try:
            # First check if library exists
            libraries = await self.proxy_list_libraries()
            if library not in libraries:
                return {
                    "success": False,
                    "library": library,
                    "error": f"Library '{library}' not found",
                    "available_libraries": libraries,
                }

            result = await self._make_mcp_request("remove_docs", {"library": library})

            if "text" in result:
                text = result["text"]
                logger.info(f"Remove library response: {text}")
                return {
                    "success": True,
                    "library": library,
                    "message": text,
                }

            return {
                "success": True,
                "library": library,
                "result": result,
            }
        except Exception as e:
            logger.error(f"Failed to remove library: {e}")
            raise DocsServerError(f"Failed to remove library: {str(e)}") from e

    async def close(self) -> None:
        """Close method for compatibility - delegates to docs manager"""
        await self.docs_manager.stop_server()

    async def __aenter__(self):
        """Async context manager entry"""
        await self.docs_manager.start_server()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

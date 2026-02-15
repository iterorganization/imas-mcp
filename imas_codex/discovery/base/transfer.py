"""Remote file transfer infrastructure.

Provides common utilities for downloading files from remote sources via HTTP
or SSH, with support for:
- Size checking before transfer (avoid downloading oversized files)
- Content validation (magic bytes check before processing)
- Batch downloads with compression for SSH transfers
- Context managers for temporary file cleanup

This module supports both wiki artifact ingestion and remote facility file
ingestion. The key difference is the access method:
- Wiki artifacts: HTTP (direct or via Tequila auth) or SSH-proxied curl
- Facility files: SSH/SCP direct access

Example usage:

    # Single file download via SSH
    async with TransferClient(ssh_host="tcv") as client:
        size = await client.get_size(url)
        if size and size < MAX_SIZE:
            content = await client.download(url)

    # Batch download via SSH with compression
    async with TransferClient(ssh_host="tcv") as client:
        paths = ["/path/to/file1.py", "/path/to/file2.py"]
        async with client.batch_download(paths) as files:
            for path, content in files.items():
                process(content)

    # HTTP download with session (for auth)
    async with TransferClient(session=requests_session) as client:
        content = await client.download(url)
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

if TYPE_CHECKING:
    import requests

logger = logging.getLogger(__name__)

# Content type detection by magic bytes
MAGIC_BYTES = {
    b"%PDF": "application/pdf",
    b"PK\x03\x04": "application/zip",  # Also covers docx, pptx, xlsx
    b"\xd0\xcf\x11\xe0": "application/msword",  # Old Office format
    b"<!DOC": "text/html",
    b"<html": "text/html",
    b"<HTML": "text/html",
    b"{\n": "application/json",  # ipynb starts with {
    b'{"': "application/json",
    b"\x89PNG": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF8": "image/gif",
}


@dataclass
class TransferResult:
    """Result of a file transfer operation."""

    url: str
    content: bytes | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    error: str | None = None
    detected_type: str | None = None  # From magic bytes

    @property
    def success(self) -> bool:
        return self.content is not None and self.error is None

    def validate_expected_type(self, expected_ext: str) -> bool:
        """Check if content matches expected file type.

        Args:
            expected_ext: Expected extension (e.g., 'pdf', 'docx')

        Returns:
            True if content matches expected type, False otherwise
        """
        if not self.content:
            return False

        # Check magic bytes
        header = self.content[:16]
        self.detected_type = detect_content_type(header)

        # PDF check
        if expected_ext.lower() == "pdf":
            return b"%PDF" in self.content[:1024]

        # Office formats (docx, pptx, xlsx) are actually ZIP files
        if expected_ext.lower() in ("docx", "pptx", "xlsx"):
            return header[:4] == b"PK\x03\x04"

        # Old Office formats
        if expected_ext.lower() in ("doc", "ppt", "xls"):
            return header[:4] == b"\xd0\xcf\x11\xe0" or header[:4] == b"PK\x03\x04"

        # Jupyter notebooks are JSON
        if expected_ext.lower() == "ipynb":
            return self.content[:1].strip() in (b"{", b"[")

        # For other types, just check it's not obviously wrong
        # (not HTML error page)
        if self.detected_type == "text/html":
            return False

        return True


def detect_content_type(header: bytes) -> str | None:
    """Detect content type from file header (magic bytes).

    Args:
        header: First 16 bytes of file content

    Returns:
        MIME type string or None if unknown
    """
    for magic, mime_type in MAGIC_BYTES.items():
        if header.startswith(magic):
            return mime_type
    return None


def decode_url(url: str) -> str:
    """Decode URL-encoded characters for shell safety.

    MediaWiki encodes special chars in URLs. We need to decode them
    for curl commands.
    """
    return unquote(url)


# =============================================================================
# Transfer Backends
# =============================================================================


class TransferBackend(ABC):
    """Abstract base class for transfer backends."""

    @abstractmethod
    async def get_size(self, url: str, timeout: int = 30) -> int | None:
        """Get file size without downloading content.

        Args:
            url: URL or path to check
            timeout: Timeout in seconds

        Returns:
            Size in bytes, or None if cannot be determined
        """
        pass

    @abstractmethod
    async def download(self, url: str, timeout: int = 120) -> TransferResult:
        """Download file content.

        Args:
            url: URL or path to download
            timeout: Timeout in seconds

        Returns:
            TransferResult with content or error
        """
        pass

    @abstractmethod
    async def download_batch(
        self,
        urls: list[str],
        timeout: int = 300,
    ) -> dict[str, TransferResult]:
        """Download multiple files.

        For SSH backends, this can use compression and batch transfer.

        Args:
            urls: List of URLs or paths to download
            timeout: Total timeout for batch operation

        Returns:
            Dict mapping URL to TransferResult
        """
        pass


class SSHCurlBackend(TransferBackend):
    """Transfer backend using SSH-proxied curl commands.

    Used for wiki artifacts accessible only from within facility networks.
    """

    def __init__(self, ssh_host: str, verify_ssl: bool = False):
        """Initialize SSH curl backend.

        Args:
            ssh_host: SSH host alias (e.g., 'tcv', 'iter')
            verify_ssl: Whether to verify SSL certificates
        """
        self.ssh_host = ssh_host
        self.verify_ssl = verify_ssl
        self._ssl_flag = "" if verify_ssl else "-k"

    async def get_size(self, url: str, timeout: int = 30) -> int | None:
        """Get file size via HTTP HEAD request through SSH."""
        decoded = decode_url(url)
        cmd = f'curl -sI {self._ssl_flag} "{decoded}" 2>/dev/null | grep -i content-length | head -1'

        try:
            result = subprocess.run(
                ["ssh", self.ssh_host, cmd],
                capture_output=True,
                timeout=timeout,
                text=True,
            )

            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if "content-length" in line.lower():
                        parts = line.split(":")
                        if len(parts) == 2:
                            try:
                                return int(parts[1].strip())
                            except ValueError:
                                pass
        except subprocess.TimeoutExpired:
            logger.warning("Timeout getting size for %s via %s", url, self.ssh_host)

        return None

    async def download(self, url: str, timeout: int = 120) -> TransferResult:
        """Download file via SSH-proxied curl."""
        decoded = decode_url(url)
        cmd = f'curl -sL {self._ssl_flag} -o /dev/stdout "{decoded}"'

        try:
            result = subprocess.run(
                ["ssh", self.ssh_host, cmd],
                capture_output=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                error = result.stderr.decode(errors="replace")
                return TransferResult(url=url, error=f"curl failed: {error}")

            content = result.stdout
            content_type = self._guess_content_type(url, content)

            return TransferResult(
                url=url,
                content=content,
                content_type=content_type,
                size_bytes=len(content),
            )

        except subprocess.TimeoutExpired:
            return TransferResult(url=url, error=f"Timeout after {timeout}s")

    async def download_batch(
        self,
        urls: list[str],
        timeout: int = 300,
    ) -> dict[str, TransferResult]:
        """Download multiple URLs sequentially.

        For HTTP URLs, batch optimization is limited - we download sequentially.
        For local files via SSH, see SSHFileBackend.
        """
        results = {}
        per_file_timeout = max(30, timeout // len(urls)) if urls else timeout

        for url in urls:
            results[url] = await self.download(url, timeout=per_file_timeout)

        return results

    def _guess_content_type(self, url: str, content: bytes) -> str:
        """Guess content type from URL extension and content."""
        # Try magic bytes first
        if content:
            detected = detect_content_type(content[:16])
            if detected:
                return detected

        # Fall back to URL extension
        ext = url.rsplit(".", 1)[-1].lower()
        content_types = {
            "pdf": "application/pdf",
            "ppt": "application/vnd.ms-powerpoint",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "doc": "application/msword",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "xls": "application/vnd.ms-excel",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "ipynb": "application/x-ipynb+json",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
        }
        return content_types.get(ext, "application/octet-stream")


class SSHFileBackend(TransferBackend):
    """Transfer backend for direct file access via SSH/SCP.

    Used for facility files accessible via SSH. Supports batch transfers
    with compression for efficiency.
    """

    def __init__(self, ssh_host: str):
        """Initialize SSH file backend.

        Args:
            ssh_host: SSH host alias (e.g., 'tcv', 'iter')
        """
        self.ssh_host = ssh_host

    async def get_size(self, path: str, timeout: int = 30) -> int | None:
        """Get file size via SSH stat command."""
        cmd = f'stat -c %s "{path}" 2>/dev/null || stat -f %z "{path}" 2>/dev/null'

        try:
            result = subprocess.run(
                ["ssh", self.ssh_host, cmd],
                capture_output=True,
                timeout=timeout,
                text=True,
            )

            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip())

        except (subprocess.TimeoutExpired, ValueError):
            pass

        return None

    async def download(self, path: str, timeout: int = 120) -> TransferResult:
        """Download file via SSH cat command."""
        cmd = f'cat "{path}"'

        try:
            result = subprocess.run(
                ["ssh", self.ssh_host, cmd],
                capture_output=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                error = result.stderr.decode(errors="replace")
                return TransferResult(url=path, error=f"cat failed: {error}")

            content = result.stdout
            content_type = detect_content_type(content[:16])

            return TransferResult(
                url=path,
                content=content,
                content_type=content_type,
                size_bytes=len(content),
            )

        except subprocess.TimeoutExpired:
            return TransferResult(url=path, error=f"Timeout after {timeout}s")

    async def download_batch(
        self,
        paths: list[str],
        timeout: int = 300,
    ) -> dict[str, TransferResult]:
        """Download multiple files with compression.

        Creates a tar archive on remote, transfers compressed, extracts locally.
        Much more efficient for many small files than individual downloads.
        """
        if not paths:
            return {}

        # For small batches, just download sequentially
        if len(paths) <= 3:
            results = {}
            per_file_timeout = max(30, timeout // len(paths))
            for path in paths:
                results[path] = await self.download(path, timeout=per_file_timeout)
            return results

        # For larger batches, use tar+gzip compression
        results: dict[str, TransferResult] = {}

        try:
            # Create temp directory for extracted files
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                archive_path = tmpdir_path / "batch.tar.gz"

                # Build tar command for remote files
                # Use -P to preserve absolute paths
                quoted_paths = " ".join(f'"{p}"' for p in paths)
                remote_cmd = f"tar czf - -P {quoted_paths} 2>/dev/null"

                # Download compressed archive
                result = subprocess.run(
                    ["ssh", self.ssh_host, remote_cmd],
                    capture_output=True,
                    timeout=timeout,
                )

                if result.returncode != 0:
                    # Fall back to sequential download
                    logger.debug("Batch tar failed, falling back to sequential")
                    per_file_timeout = max(30, timeout // len(paths))
                    for path in paths:
                        results[path] = await self.download(
                            path, timeout=per_file_timeout
                        )
                    return results

                # Write archive to temp file
                archive_path.write_bytes(result.stdout)

                # Extract archive
                extract_result = subprocess.run(
                    ["tar", "xzf", str(archive_path), "-C", str(tmpdir_path)],
                    capture_output=True,
                    timeout=30,
                )

                if extract_result.returncode != 0:
                    logger.warning("Failed to extract batch archive")
                    per_file_timeout = max(30, timeout // len(paths))
                    for path in paths:
                        results[path] = await self.download(
                            path, timeout=per_file_timeout
                        )
                    return results

                # Read extracted files
                for path in paths:
                    # Handle both absolute and relative paths in archive
                    local_path = tmpdir_path / path.lstrip("/")
                    if local_path.exists():
                        content = local_path.read_bytes()
                        results[path] = TransferResult(
                            url=path,
                            content=content,
                            content_type=detect_content_type(content[:16]),
                            size_bytes=len(content),
                        )
                    else:
                        results[path] = TransferResult(
                            url=path,
                            error=f"File not found in archive: {path}",
                        )

        except subprocess.TimeoutExpired:
            logger.warning("Batch download timeout, falling back to sequential")
            per_file_timeout = max(30, timeout // len(paths))
            for path in paths:
                if path not in results:
                    results[path] = await self.download(path, timeout=per_file_timeout)

        return results


class HTTPBackend(TransferBackend):
    """Transfer backend for direct HTTP requests.

    Used when wiki is accessible directly (e.g., ITER Confluence).
    """

    def __init__(
        self, session: requests.Session | None = None, verify_ssl: bool = True
    ):
        """Initialize HTTP backend.

        Args:
            session: Requests session (for auth cookies, etc.)
            verify_ssl: Whether to verify SSL certificates
        """
        self.session = session
        self.verify_ssl = verify_ssl

    def _get_session(self):
        """Get or create requests session."""
        if self.session is None:
            import requests

            self.session = requests.Session()
        return self.session

    async def get_size(self, url: str, timeout: int = 30) -> int | None:
        """Get file size via HTTP HEAD request."""
        try:
            # Override Accept header for binary downloads — Confluence sessions
            # set Accept: application/json for REST API, but file downloads
            # need Accept: */*.  Content-Type is meaningless for GET/HEAD.
            response = self._get_session().head(
                url,
                timeout=timeout,
                verify=self.verify_ssl,
                allow_redirects=True,
                headers={"Accept": "*/*"},
            )
            if response.status_code == 200:
                content_length = response.headers.get("content-length")
                if content_length:
                    return int(content_length)
        except Exception as e:
            logger.debug("HEAD request failed for %s: %s", url, e)

        return None

    async def download(self, url: str, timeout: int = 120) -> TransferResult:
        """Download file via HTTP GET."""
        try:
            # Override Accept header for binary downloads — Confluence sessions
            # set Accept: application/json for REST API, but file downloads
            # need Accept: */*.  Per-request headers override session defaults
            # without mutating the shared session.
            response = self._get_session().get(
                url,
                timeout=timeout,
                verify=self.verify_ssl,
                allow_redirects=True,
                headers={"Accept": "*/*"},
            )

            if response.status_code != 200:
                return TransferResult(
                    url=url,
                    error=f"HTTP {response.status_code}: {response.reason}",
                )

            content = response.content
            content_type = response.headers.get(
                "content-type", "application/octet-stream"
            )

            return TransferResult(
                url=url,
                content=content,
                content_type=content_type,
                size_bytes=len(content),
            )

        except Exception as e:
            return TransferResult(url=url, error=str(e))

    async def download_batch(
        self,
        urls: list[str],
        timeout: int = 300,
    ) -> dict[str, TransferResult]:
        """Download multiple URLs sequentially.

        HTTP doesn't support batch optimization like SSH tar can.
        """
        results = {}
        per_file_timeout = max(30, timeout // len(urls)) if urls else timeout

        for url in urls:
            results[url] = await self.download(url, timeout=per_file_timeout)

        return results


# =============================================================================
# Transfer Client (Unified Interface)
# =============================================================================


@dataclass
class TransferClient:
    """High-level transfer client with automatic backend selection.

    Example:
        # SSH-proxied HTTP (for wiki behind firewall)
        async with TransferClient(ssh_host="tcv") as client:
            result = await client.download(wiki_url)

        # Direct HTTP (for public wiki or with auth session)
        async with TransferClient(session=my_session) as client:
            result = await client.download(wiki_url)

        # SSH file access (for remote facility files)
        async with TransferClient(ssh_host="tcv", mode="file") as client:
            result = await client.download("/path/to/file.py")
    """

    ssh_host: str | None = None
    session: Any = None  # requests.Session
    mode: str = "http"  # "http" for URLs, "file" for SSH file paths
    verify_ssl: bool = False
    _backend: TransferBackend | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Select appropriate backend."""
        if self.mode == "file" and self.ssh_host:
            self._backend = SSHFileBackend(self.ssh_host)
        elif self.ssh_host:
            self._backend = SSHCurlBackend(self.ssh_host, verify_ssl=self.verify_ssl)
        else:
            self._backend = HTTPBackend(
                session=self.session, verify_ssl=self.verify_ssl
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def get_size(self, url_or_path: str, timeout: int = 30) -> int | None:
        """Get file size without downloading."""
        return await self._backend.get_size(url_or_path, timeout=timeout)

    async def download(
        self,
        url_or_path: str,
        timeout: int = 120,
        expected_type: str | None = None,
    ) -> TransferResult:
        """Download file and optionally validate type.

        Args:
            url_or_path: URL or path to download
            timeout: Timeout in seconds
            expected_type: Expected file extension for validation (e.g., 'pdf')

        Returns:
            TransferResult with content or error
        """
        result = await self._backend.download(url_or_path, timeout=timeout)

        if result.success and expected_type:
            if not result.validate_expected_type(expected_type):
                detected = result.detected_type or "unknown"
                result.error = (
                    f"Content type mismatch: expected {expected_type}, got {detected}"
                )
                result.content = None

        return result

    async def download_batch(
        self,
        urls_or_paths: list[str],
        timeout: int = 300,
    ) -> dict[str, TransferResult]:
        """Download multiple files.

        For SSH file backend, uses compression for efficiency.
        """
        return await self._backend.download_batch(urls_or_paths, timeout=timeout)

    @asynccontextmanager
    async def batch_context(self, urls_or_paths: list[str], timeout: int = 300):
        """Context manager for batch downloads with cleanup.

        Yields a dict of {url: content_bytes} for successful downloads.
        Automatically handles temp file cleanup.

        Example:
            async with client.batch_context(paths) as files:
                for path, content in files.items():
                    process(content)
        """
        results = await self.download_batch(urls_or_paths, timeout=timeout)
        try:
            # Yield only successful downloads
            yield {url: r.content for url, r in results.items() if r.success}
        finally:
            pass  # Cleanup handled by TransferResult (no temp files)


@contextmanager
def temp_file_context(content: bytes, suffix: str = "") -> Path:
    """Context manager for temporary file with content.

    Writes content to temp file, yields path, cleans up on exit.

    Example:
        with temp_file_context(pdf_bytes, suffix=".pdf") as path:
            reader.load(path)
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        temp_path = Path(f.name)

    try:
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)


# =============================================================================
# Convenience Functions
# =============================================================================


async def download_with_size_check(
    url_or_path: str,
    max_size_bytes: int,
    ssh_host: str | None = None,
    session: Any = None,
    mode: str = "http",
    expected_type: str | None = None,
) -> TransferResult:
    """Download file with size check.

    Convenience function that checks size before downloading.

    Args:
        url_or_path: URL or path to download
        max_size_bytes: Maximum allowed size
        ssh_host: SSH host for proxied access
        session: HTTP session for auth
        mode: "http" or "file"
        expected_type: Expected file extension for validation

    Returns:
        TransferResult with content, or error if oversized/invalid
    """
    async with TransferClient(ssh_host=ssh_host, session=session, mode=mode) as client:
        # Check size first
        size = await client.get_size(url_or_path)
        if size is not None and size > max_size_bytes:
            size_mb = size / (1024 * 1024)
            max_mb = max_size_bytes / (1024 * 1024)
            return TransferResult(
                url=url_or_path,
                size_bytes=size,
                error=f"File size {size_mb:.1f} MB exceeds limit {max_mb:.1f} MB",
            )

        # Download
        return await client.download(url_or_path, expected_type=expected_type)

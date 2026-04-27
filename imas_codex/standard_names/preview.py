"""Preview a staging directory via the ISN catalog-site renderer.

Thin wrapper around the ``standard-names catalog-site serve`` CLI
command from the ``imas-standard-names`` package. Launches a local
MkDocs dev server so the user can browse the proposed catalog before
publishing.

See plan 35 §Phase 3 (3b).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PreviewHandle:
    """Handle to a running preview server process."""

    process: subprocess.Popen | None
    url: str | None
    staging_dir: str

    def stop(self) -> None:
        """Stop the preview server."""
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


def run_preview(
    staging_dir: str | Path,
    *,
    port: int | None = None,
    host: str | None = None,
) -> PreviewHandle:
    """Launch the ISN catalog-site server on a staging directory.

    Parameters
    ----------
    staging_dir:
        Path to the staging directory produced by ``sn export``.
    port:
        Optional port number for the dev server. If ``None``, uses
        the ISN default (typically 8000).
    host:
        Optional host to bind to. Defaults to ``127.0.0.1`` (loopback).
        Pass ``0.0.0.0`` to allow other machines on the network to
        reach the dev server (useful when previewing over an SSH tunnel
        with shared access for collaborators).

    Returns
    -------
    PreviewHandle with the subprocess and URL.
    """
    staging = Path(staging_dir)
    if not staging.is_dir():
        raise FileNotFoundError(f"Staging directory not found: {staging}")

    manifest = staging / "catalog.yml"
    if not manifest.is_file():
        raise FileNotFoundError(f"No catalog.yml in staging directory: {staging}")

    # Build the command
    cmd = ["uv", "run", "standard-names", "serve", str(staging)]
    if port is not None:
        cmd.extend(["--port", str(port)])
    if host is not None:
        cmd.extend(["--host", host])

    effective_port = port or 8000
    effective_host = host or "127.0.0.1"
    url = f"http://{effective_host}:{effective_port}"

    logger.info("Starting preview server: %s", " ".join(cmd))
    logger.info("Preview URL: %s", url)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.error(
            "Could not find 'standard-names' CLI. "
            "Ensure imas-standard-names is installed."
        )
        return PreviewHandle(process=None, url=None, staging_dir=str(staging))

    return PreviewHandle(
        process=process,
        url=url,
        staging_dir=str(staging),
    )

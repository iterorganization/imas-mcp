"""Preview a staging directory via the ISN catalog renderer.

Uses ISN's ``CatalogRenderer`` (public API) to generate a documentation
site, then serves it locally with MkDocs. The rendering toolbox lives in
ISN — codex imports and uses it directly (no subprocess delegation to the
``standard-names`` CLI).

Press Ctrl-C to stop the preview server.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PreviewHandle:
    """Handle to a running preview server process."""

    process: subprocess.Popen | None
    url: str | None
    staging_dir: str
    temp_dir: str | None = None

    def stop(self) -> None:
        """Stop the preview server and clean up temporary files."""
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        if self.temp_dir is not None:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None


def _generate_preview_site(
    catalog_path: Path,
    docs_dir: Path,
    *,
    site_name: str = "Standard Names Preview",
    site_url: str = "",
) -> int:
    """Generate MkDocs site content from catalog YAML using ISN's CatalogRenderer.

    Uses ISN's public ``CatalogRenderer`` class for rendering and the
    ``MKDOCS_SERVE_TEMPLATE`` / ``CATALOG_CSS`` constants for site
    structure. This mirrors ISN's ``_generate_site_content()`` logic
    using only public API.

    Parameters
    ----------
    catalog_path:
        Path to the ``standard_names/`` directory containing YAML files.
    docs_dir:
        Temporary directory to write the MkDocs project into.
    site_name:
        Title for the documentation site.
    site_url:
        Base URL for the site (used in mkdocs.yml).

    Returns
    -------
    Number of standard names found.
    """
    from imas_standard_names.cli.catalog_site import (
        CATALOG_CSS,
        MKDOCS_SERVE_TEMPLATE,
    )
    from imas_standard_names.rendering.catalog import CatalogRenderer

    renderer = CatalogRenderer(catalog_path)
    stats = renderer.get_stats()

    # Create docs structure
    docs_content_dir = docs_dir / "docs"
    docs_content_dir.mkdir(exist_ok=True)
    stylesheets_dir = docs_content_dir / "stylesheets"
    stylesheets_dir.mkdir(exist_ok=True)

    # Generate mkdocs.yml
    mkdocs_config = MKDOCS_SERVE_TEMPLATE.format(
        site_name=site_name,
        site_url=site_url,
    )
    (docs_dir / "mkdocs.yml").write_text(mkdocs_config)

    # Generate CSS
    (stylesheets_dir / "catalog.css").write_text(CATALOG_CSS)

    # Generate index.md
    readme_path = catalog_path / "README.md"
    if readme_path.exists():
        index_content = readme_path.read_text(encoding="utf-8")
    else:
        index_content = f"# {site_name}\n\n"
        index_content += renderer.render_overview(link_prefix="catalog.md")
    (docs_content_dir / "index.md").write_text(index_content)

    # Generate catalog.md
    catalog_content = "# Standard Names Catalog\n\n"
    catalog_content += renderer.render_overview()
    catalog_content += "\n---\n\n"
    catalog_content += renderer.render_catalog()
    (docs_content_dir / "catalog.md").write_text(catalog_content)

    return stats["total_names"]


def run_preview(
    staging_dir: str | Path,
    *,
    port: int | None = None,
    host: str | None = None,
) -> PreviewHandle:
    """Launch a local MkDocs preview of a staging directory.

    Uses ISN's ``CatalogRenderer`` to generate documentation, then
    serves it with ``mkdocs serve``.

    Parameters
    ----------
    staging_dir:
        Path to the staging directory produced by ``sn export``.
        Must contain ``catalog.yml`` and ``standard_names/``.
    port:
        Port number for the dev server (default: 8000).
    host:
        Host to bind to (default: ``127.0.0.1``). Pass ``0.0.0.0``
        to allow remote access (e.g., over an SSH tunnel).

    Returns
    -------
    PreviewHandle with the subprocess, URL, and temp directory.

    Raises
    ------
    FileNotFoundError
        If staging directory or catalog.yml is missing.
    ImportError
        If imas-standard-names is not installed.
    """
    staging = Path(staging_dir)
    if not staging.is_dir():
        raise FileNotFoundError(f"Staging directory not found: {staging}")

    catalog_yml = staging / "catalog.yml"
    if not catalog_yml.is_file():
        raise FileNotFoundError(f"No catalog.yml in staging directory: {staging}")

    sn_dir = staging / "standard_names"
    if not sn_dir.is_dir():
        raise FileNotFoundError(f"No standard_names/ directory in staging: {staging}")

    # Generate site content using ISN's CatalogRenderer
    temp_dir = tempfile.mkdtemp(prefix="sn-preview-")
    docs_dir = Path(temp_dir)

    try:
        total = _generate_preview_site(
            catalog_path=sn_dir,
            docs_dir=docs_dir,
        )
    except ImportError as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ImportError(
            "imas-standard-names is required for preview. "
            "Install with: uv add imas-standard-names"
        ) from exc
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    effective_port = port or 8000
    effective_host = host or "127.0.0.1"
    url = f"http://{effective_host}:{effective_port}"

    logger.info("Generated preview site for %d standard names", total)
    logger.info("Preview URL: %s", url)

    # Run mkdocs serve (inherits active Python environment)
    cmd = [
        sys.executable,
        "-m",
        "mkdocs",
        "serve",
        "--dev-addr",
        f"{effective_host}:{effective_port}",
    ]

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(docs_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ImportError(
            "mkdocs is required for preview. Install with: uv add mkdocs-material"
        ) from exc

    return PreviewHandle(
        process=process,
        url=url,
        staging_dir=str(staging),
        temp_dir=temp_dir,
    )

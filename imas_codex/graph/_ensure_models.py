"""Auto-regenerate graph models when schemas are newer.

This module checks whether the generated models.py is stale relative to
the source LinkML schemas (facility.yaml, common.yaml, imas_dd.yaml).
If stale and running from an editable/development install, it triggers
automatic regeneration via the build-models script.

This prevents the common failure mode where `git pull` brings new schema
definitions but models.py (gitignored, generated) remains stale.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA_FILES = ("facility.yaml", "common.yaml", "imas_dd.yaml")


def _is_editable_install() -> bool:
    """Check if imas_codex is installed in editable/development mode."""
    # In editable installs, the package directory is the source tree
    package_dir = Path(__file__).resolve().parent.parent
    # Check for pyproject.toml in the parent (project root)
    return (package_dir.parent / "pyproject.toml").exists()


def _schemas_dir() -> Path:
    """Return the schemas directory."""
    return Path(__file__).resolve().parent.parent / "schemas"


def _models_file() -> Path:
    """Return the generated models.py path."""
    return Path(__file__).resolve().parent / "models.py"


def ensure_models_fresh() -> None:
    """Regenerate models if any schema file is newer than models.py.

    Only acts in editable/development installs. In installed packages,
    models are baked into the wheel and this is a no-op.
    """
    if not _is_editable_install():
        return

    models = _models_file()
    schemas_dir = _schemas_dir()

    if not models.exists():
        _regenerate(reason="models.py missing")
        return

    models_mtime = models.stat().st_mtime
    for name in _SCHEMA_FILES:
        schema = schemas_dir / name
        if schema.exists() and schema.stat().st_mtime > models_mtime:
            _regenerate(reason=f"{name} is newer than models.py")
            return


def _regenerate(reason: str) -> None:
    """Run build-models --force to regenerate."""
    project_root = Path(__file__).resolve().parent.parent.parent
    logger.info("Auto-regenerating graph models (%s)", reason)

    try:
        # Use the build-models entry point via uv run
        # Falls back to direct gen-pydantic if uv is not available
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from scripts.build_models import build_models; "
                    "build_models.main(['--force', '--quiet'], standalone_mode=False)"
                ),
            ],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.info("Graph models regenerated successfully")
        else:
            logger.warning(
                "Model regeneration failed (exit %d): %s",
                result.returncode,
                result.stderr[:500] if result.stderr else "(no stderr)",
            )
    except Exception as e:
        logger.warning("Model regeneration failed: %s", e)

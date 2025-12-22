"""
Remote facility artifact capture.

This package provides artifact capture for remote fusion facility exploration.
The LLM explores facilities via direct SSH, then captures findings here.

Usage:
    # Explore via SSH directly (faster than CLI)
    ssh epfl "python --version; pip list | head"

    # Capture findings
    uv run imas-codex epfl --capture environment << 'EOF'
    python:
      version: "3.9.21"
    EOF

See imas_codex/config/README.md for comprehensive exploration guidance.
"""

from imas_codex.remote.capture import (
    capture_artifact,
    list_artifacts,
    load_artifact,
)

__all__ = [
    "capture_artifact",
    "list_artifacts",
    "load_artifact",
]

"""
Remote facility exploration utilities.

This package provides utilities for remote fusion facility exploration.
Facility knowledge is now stored in the graph database, not flat artifacts.

Key functions:
- run(cmd, facility): Execute command locally or via SSH
- check_all_tools(facility): Check fast tool availability
- install_tool(tool_key, facility): Install a specific tool
- install_all_tools(facility): Install all fast tools

See imas_codex/config/README.md for exploration guidance.
"""

from imas_codex.remote.tools import (
    FastTool,
    FastToolsConfig,
    check_all_tools,
    check_tool,
    detect_architecture,
    ensure_path,
    install_all_tools,
    install_tool,
    is_local_facility,
    load_fast_tools,
    run,
)

__all__ = [
    "FastTool",
    "FastToolsConfig",
    "check_all_tools",
    "check_tool",
    "detect_architecture",
    "ensure_path",
    "install_all_tools",
    "install_tool",
    "is_local_facility",
    "load_fast_tools",
    "run",
]

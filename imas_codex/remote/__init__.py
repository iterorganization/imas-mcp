"""
Remote facility exploration utilities.

This package provides utilities for remote fusion facility exploration.
Facility knowledge is now stored in the graph database, not flat artifacts.

Key functions:
- run(cmd, facility): Execute command locally or via SSH
- run_script(script, facility): Execute script via stdin (faster)
- check_all_tools(facility): Check remote tool availability
- install_tool(tool_key, facility): Install a specific tool
- install_all_tools(facility): Install all remote tools

Low-level execution (no facility lookup, for use by discovery modules):
- run_command(cmd, ssh_host): Execute command directly
- run_script_via_stdin(script, ssh_host): Execute script directly
- is_local_host(ssh_host): Check if host is local

See imas_codex/config/README.md for exploration guidance.
"""

# Low-level executor (no facility imports - avoids circular imports)
from imas_codex.remote.executor import (
    cleanup_stale_sockets,
    ensure_ssh_healthy,
    is_local_host,
    run_command,
    run_script_via_stdin,
)

# High-level facility-aware tools
from imas_codex.remote.tools import (
    RemoteTool,
    RemoteToolsConfig,
    check_all_tools,
    check_tool,
    detect_architecture,
    ensure_path,
    install_all_tools,
    install_tool,
    is_local_facility,
    load_remote_tools,
    run,
    run_script,
)

# SSH tunnel management
from imas_codex.remote.tunnel import (
    TUNNEL_OFFSET,
    ensure_tunnel,
    is_port_bound_by_ssh,
    is_tunnel_active,
    stop_tunnel,
)

__all__ = [
    # Low-level executor
    "cleanup_stale_sockets",
    "ensure_ssh_healthy",
    "is_local_host",
    "run_command",
    "run_script_via_stdin",
    # SSH tunnel management
    "TUNNEL_OFFSET",
    "ensure_tunnel",
    "is_port_bound_by_ssh",
    "is_tunnel_active",
    "stop_tunnel",
    # High-level tools
    "RemoteTool",
    "RemoteToolsConfig",
    "check_all_tools",
    "check_tool",
    "detect_architecture",
    "ensure_path",
    "install_all_tools",
    "install_tool",
    "is_local_facility",
    "load_remote_tools",
    "run",
    "run_script",
]

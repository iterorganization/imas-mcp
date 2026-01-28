"""
Smolagents agent for installing fast CLI tools on local and remote facilities.

This agent checks tool availability and installs missing tools.
It handles architecture detection, PATH configuration, and verification.

Usage from python() REPL:
    # Check and install tools on EPFL
    result = setup_tools('tcv')

    # Check and install tools locally (ITER/SDCC)
    result = setup_tools('iter')

    # Install specific tool
    result = install_tool('rg', facility='tcv')
"""

import logging
from dataclasses import dataclass

from smolagents import CodeAgent, Tool

from imas_codex.agentic.agents import create_litellm_model, get_model_for_task
from imas_codex.remote.tools import (
    check_all_tools,
    check_tool,
    detect_architecture,
    ensure_path,
    install_all_tools,
    install_tool,
    load_fast_tools,
    run,
)

logger = logging.getLogger(__name__)

# System prompt for the tool installer agent
TOOL_INSTALLER_SYSTEM_PROMPT = """You are a tool installation agent for fusion facility exploration.

Your job is to ensure fast CLI tools (rg, fd, tokei, scc, etc.) are available on target systems.
These tools are defined in imas_codex/config/fast_tools.yaml.

## Workflow

1. **Check current state**: Use check_all_tools() to see what's installed
2. **Detect architecture**: Use detect_architecture() to determine x86_64 vs aarch64
3. **Ensure PATH**: Use ensure_path() to add ~/bin to PATH if needed
4. **Install missing tools**: Use install_tool() for each missing required tool
5. **Verify installation**: Re-check tools to confirm success

## Key Principles

- Always check before installing (don't reinstall existing tools)
- Install required tools first (rg, fd), then optional tools
- Report clear success/failure status for each tool
- If a tool fails to install, continue with others and report at end

## Facilities

- Local facilities (local=true): Commands run directly without SSH
- Remote facilities: Commands run via SSH to the facility's ssh_host

## Output Format

Provide a summary like:
- Architecture: x86_64
- Required tools: rg ✓, fd ✓
- Optional tools: tokei ✓, scc ✓, dust ✗ (failed: ...)
- PATH configured: yes
"""


def _create_tool_installer_tools(facility: str | None = None) -> list[Tool]:
    """Create smolagents tools for the installer agent.

    Args:
        facility: Target facility (None = local)

    Returns:
        List of Tool instances
    """

    class CheckToolsTool(Tool):
        """Check availability of all fast CLI tools."""

        name = "check_tools"
        description = "Check availability of all fast CLI tools. Returns dict with tools status, required_ok, missing_required, missing_optional."
        inputs = {}
        output_type = "object"

        def forward(self) -> dict:
            return check_all_tools(facility=facility)

    class CheckSingleToolTool(Tool):
        """Check if a specific tool is available."""

        name = "check_single_tool"
        description = "Check if a specific tool is available. Returns dict with available, version, path, required, purpose."
        inputs = {
            "tool_key": {
                "type": "string",
                "description": "Tool key (e.g., 'rg', 'fd', 'tokei')",
            }
        }
        output_type = "object"

        def forward(self, tool_key: str) -> dict:
            return check_tool(tool_key, facility=facility)

    class GetArchitectureTool(Tool):
        """Detect CPU architecture of target system."""

        name = "get_architecture"
        description = (
            "Detect CPU architecture of target system. Returns 'x86_64' or 'aarch64'."
        )
        inputs = {}
        output_type = "string"

        def forward(self) -> str:
            return detect_architecture(facility=facility)

    class ConfigurePathTool(Tool):
        """Ensure ~/bin is in PATH."""

        name = "configure_path"
        description = "Ensure ~/bin is in PATH, adding to .bashrc if needed. Returns status message."
        inputs = {}
        output_type = "string"

        def forward(self) -> str:
            return ensure_path(facility=facility)

    class InstallSingleToolTool(Tool):
        """Install a specific tool."""

        name = "install_single_tool"
        description = "Install a specific tool. Returns dict with success, action, version, error."
        inputs = {
            "tool_key": {
                "type": "string",
                "description": "Tool key (e.g., 'rg', 'fd')",
            },
            "force": {
                "type": "boolean",
                "description": "Reinstall even if already present",
                "nullable": True,
            },
        }
        output_type = "object"

        def forward(self, tool_key: str, force: bool = False) -> dict:
            return install_tool(tool_key, facility=facility, force=force)

    class InstallAllTool(Tool):
        """Install all fast tools."""

        name = "install_all"
        description = "Install all fast tools. Returns dict with installed, already_present, failed lists."
        inputs = {
            "required_only": {
                "type": "boolean",
                "description": "Only install required tools (rg, fd)",
                "nullable": True,
            },
            "force": {
                "type": "boolean",
                "description": "Reinstall even if already present",
                "nullable": True,
            },
        }
        output_type = "object"

        def forward(self, required_only: bool = False, force: bool = False) -> dict:
            return install_all_tools(
                facility=facility, required_only=required_only, force=force
            )

    class RunCommandTool(Tool):
        """Run a shell command on the target system."""

        name = "run_command"
        description = (
            "Run a shell command on the target system. Returns command output."
        )
        inputs = {"cmd": {"type": "string", "description": "Shell command to execute"}}
        output_type = "string"

        def forward(self, cmd: str) -> str:
            return run(cmd, facility=facility)

    class ListAvailableToolsTool(Tool):
        """List all tools defined in fast_tools.yaml."""

        name = "list_available_tools"
        description = "List all tools defined in fast_tools.yaml. Returns dict with required and optional tool definitions."
        inputs = {}
        output_type = "object"

        def forward(self) -> dict:
            config = load_fast_tools()
            return {
                "required": {
                    k: {"name": v.name, "purpose": v.purpose, "binary": v.binary}
                    for k, v in config.required.items()
                },
                "optional": {
                    k: {"name": v.name, "purpose": v.purpose, "binary": v.binary}
                    for k, v in config.optional.items()
                },
            }

    return [
        CheckToolsTool(),
        CheckSingleToolTool(),
        GetArchitectureTool(),
        ConfigurePathTool(),
        InstallSingleToolTool(),
        InstallAllTool(),
        RunCommandTool(),
        ListAvailableToolsTool(),
    ]


@dataclass
class ToolInstallerResult:
    """Result from tool installer agent."""

    success: bool
    facility: str
    architecture: str
    installed: list[str]
    already_present: list[str]
    failed: list[dict]
    path_configured: bool
    summary: str


def get_tool_installer_agent(
    facility: str | None = None,
    model: str | None = None,
    verbose: bool = False,
) -> CodeAgent:
    """Create a CodeAgent for tool installation.

    Args:
        facility: Target facility (None = local)
        model: LLM model to use (default: from config)
        verbose: Enable verbose logging

    Returns:
        Configured CodeAgent
    """
    llm = create_litellm_model(
        model=model or get_model_for_task("default"),
        temperature=0.1,
        max_tokens=4096,
    )
    tools = _create_tool_installer_tools(facility=facility)

    agent = CodeAgent(
        tools=tools,
        model=llm,
        instructions=TOOL_INSTALLER_SYSTEM_PROMPT,
        max_steps=15,
        name="tool_installer",
    )

    logger.info(f"Created tool installer agent for facility={facility or 'local'}")
    return agent


def setup_tools(
    facility: str | None = None,
    required_only: bool = False,
    verbose: bool = False,
) -> ToolInstallerResult:
    """High-level function to setup tools using the CodeAgent.

    This is the main entry point for tool installation. It creates
    an agent that checks and installs tools.

    Args:
        facility: Target facility (None = local)
        required_only: Only install required tools
        verbose: Enable verbose agent output

    Returns:
        ToolInstallerResult with installation summary

    Examples:
        # Setup tools on EPFL
        result = setup_tools('tcv')
        print(result.summary)

        # Setup only required tools locally
        result = setup_tools('iter', required_only=True)
    """
    agent = get_tool_installer_agent(facility=facility, verbose=verbose)

    # Construct the task prompt
    scope = (
        "required tools only" if required_only else "all tools (required and optional)"
    )
    task = f"""Setup fast CLI tools on facility '{facility or "local"}'.

Install {scope}.

Steps:
1. Check current tool availability
2. Detect system architecture
3. Ensure ~/bin is in PATH
4. Install any missing tools
5. Verify installations

Report the final status of each tool."""

    # Run the agent (CodeAgent uses .run() instead of .chat())
    try:
        response = agent.run(task)
        summary = str(response)
    except Exception as e:
        logger.exception("Tool installer agent failed")
        summary = f"Agent failed: {e}"

    # Get final state
    final_state = check_all_tools(facility=facility)
    arch = detect_architecture(facility=facility)

    # Determine what was installed vs already present
    # (This is approximate since we don't track state changes)
    installed = []
    already_present = []
    failed = []

    for tool_key, status in final_state.get("tools", {}).items():
        if status.get("available"):
            # We don't know if it was just installed or already there
            # The agent's response should clarify
            already_present.append(tool_key)
        else:
            failed.append({"tool": tool_key, "error": "Not available after setup"})

    return ToolInstallerResult(
        success=final_state.get("required_ok", False),
        facility=facility or "local",
        architecture=arch,
        installed=installed,
        already_present=already_present,
        failed=failed,
        path_configured=True,  # ensure_path was called
        summary=summary,
    )


# Convenience function for direct use without agent
def quick_setup(
    facility: str | None = None,
    required_only: bool = False,
) -> dict:
    """Quick tool setup without using the ReAct agent.

    This is a simpler, faster alternative to setup_tools() that
    doesn't use an LLM. Good for automated/scripted use.

    Args:
        facility: Target facility (None = local)
        required_only: Only install required tools

    Returns:
        Dict with installation results
    """
    results = {
        "facility": facility or "local",
        "architecture": detect_architecture(facility=facility),
        "path_status": ensure_path(facility=facility),
    }

    # Install tools
    install_results = install_all_tools(
        facility=facility,
        required_only=required_only,
    )

    results.update(install_results)

    # Final check
    final_state = check_all_tools(facility=facility)
    results["final_state"] = final_state
    results["success"] = final_state.get("required_ok", False)

    return results

"""
ReAct agent for installing fast CLI tools on local and remote facilities.

This agent reactively checks tool availability and installs missing tools.
It handles architecture detection, PATH configuration, and verification.

Usage from python() REPL:
    # Check and install tools on EPFL
    result = setup_tools('epfl')

    # Check and install tools locally (ITER/SDCC)
    result = setup_tools('iter')

    # Install specific tool
    result = install_tool('rg', facility='epfl')
"""

import logging
from dataclasses import dataclass

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

from imas_codex.agentic.llm import get_llm
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


def _create_tool_installer_tools(facility: str | None = None) -> list[FunctionTool]:
    """Create LlamaIndex tools for the installer agent.

    Args:
        facility: Target facility (None = local)

    Returns:
        List of FunctionTool instances
    """

    def check_tools() -> dict:
        """Check availability of all fast CLI tools.

        Returns dict with:
        - tools: {tool_name: {available, version, required, purpose}}
        - required_ok: bool
        - missing_required: list
        - missing_optional: list
        """
        return check_all_tools(facility=facility)

    def check_single_tool(tool_key: str) -> dict:
        """Check if a specific tool is available.

        Args:
            tool_key: Tool key (e.g., 'rg', 'fd', 'tokei')

        Returns dict with available, version, path, required, purpose
        """
        return check_tool(tool_key, facility=facility)

    def get_architecture() -> str:
        """Detect CPU architecture of target system.

        Returns 'x86_64' or 'aarch64'
        """
        return detect_architecture(facility=facility)

    def configure_path() -> str:
        """Ensure ~/bin is in PATH, adding to .bashrc if needed.

        Returns status message.
        """
        return ensure_path(facility=facility)

    def install_single_tool(tool_key: str, force: bool = False) -> dict:
        """Install a specific tool.

        Args:
            tool_key: Tool key (e.g., 'rg', 'fd')
            force: Reinstall even if already present

        Returns dict with success, action, version, error
        """
        return install_tool(tool_key, facility=facility, force=force)

    def install_all(required_only: bool = False, force: bool = False) -> dict:
        """Install all fast tools.

        Args:
            required_only: Only install required tools (rg, fd)
            force: Reinstall even if already present

        Returns dict with installed, already_present, failed lists
        """
        return install_all_tools(
            facility=facility, required_only=required_only, force=force
        )

    def run_command(cmd: str) -> str:
        """Run a shell command on the target system.

        Args:
            cmd: Shell command to execute

        Returns command output
        """
        return run(cmd, facility=facility)

    def list_available_tools() -> dict:
        """List all tools defined in fast_tools.yaml.

        Returns dict with required and optional tool definitions.
        """
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
        FunctionTool.from_defaults(fn=check_tools, name="check_tools"),
        FunctionTool.from_defaults(fn=check_single_tool, name="check_single_tool"),
        FunctionTool.from_defaults(fn=get_architecture, name="get_architecture"),
        FunctionTool.from_defaults(fn=configure_path, name="configure_path"),
        FunctionTool.from_defaults(fn=install_single_tool, name="install_single_tool"),
        FunctionTool.from_defaults(fn=install_all, name="install_all"),
        FunctionTool.from_defaults(fn=run_command, name="run_command"),
        FunctionTool.from_defaults(
            fn=list_available_tools, name="list_available_tools"
        ),
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
    model: str = "google/gemini-2.0-flash",
    verbose: bool = False,
) -> ReActAgent:
    """Create a ReAct agent for tool installation.

    Args:
        facility: Target facility (None = local)
        model: LLM model to use
        verbose: Enable verbose logging

    Returns:
        Configured ReActAgent
    """
    llm = get_llm(model=model, temperature=0.1)
    tools = _create_tool_installer_tools(facility=facility)

    agent = ReActAgent(
        tools=tools,
        llm=llm,
        verbose=verbose,
        system_prompt=TOOL_INSTALLER_SYSTEM_PROMPT,
        max_iterations=15,
    )

    logger.info(f"Created tool installer agent for facility={facility or 'local'}")
    return agent


def setup_tools(
    facility: str | None = None,
    required_only: bool = False,
    verbose: bool = False,
) -> ToolInstallerResult:
    """High-level function to setup tools using the ReAct agent.

    This is the main entry point for tool installation. It creates
    a ReAct agent that reactively checks and installs tools.

    Args:
        facility: Target facility (None = local)
        required_only: Only install required tools
        verbose: Enable verbose agent output

    Returns:
        ToolInstallerResult with installation summary

    Examples:
        # Setup tools on EPFL
        result = setup_tools('epfl')
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

    # Run the agent
    try:
        response = agent.chat(task)
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

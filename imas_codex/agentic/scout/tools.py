"""Smolagents tools for stateless scout.

These tools are minimal and focused on schema-compliant graph operations.
The LLM already knows how to use rg, fd, shell commands via CodeAgent.

Tools here are for:
1. Querying the graph for frontier state
2. Persisting discoveries with proper schema
3. Recording interest reasons and skip decisions

DO NOT add tools for:
- grep/find (LLM uses shell directly)
- file reading (LLM uses head/cat directly)
- basic shell operations (LLM knows these)
"""

from __future__ import annotations

import logging

from smolagents import Tool

from .stateless import (
    InterestScore,
    advance_path_status,
    discover_path,
    get_exploration_summary,
    get_frontier,
    queue_source_file,
    skip_path,
)

logger = logging.getLogger(__name__)


class DiscoverPathTool(Tool):
    """Discover a new path and add to exploration frontier."""

    name = "discover_path"
    description = (
        "Add a discovered path to the exploration frontier. "
        "Use this when you find a new directory or file worth exploring. "
        "Provide an interest_reason explaining WHY this path is valuable."
    )
    inputs = {
        "path": {
            "type": "string",
            "description": "Absolute path on the facility",
        },
        "path_type": {
            "type": "string",
            "description": "Type: 'directory' or 'file'",
            "nullable": True,
        },
        "imas_relevance": {
            "type": "number",
            "description": "IMAS relevance score 0.0-1.0 (high if contains IDS code)",
            "nullable": True,
        },
        "physics_codes": {
            "type": "number",
            "description": "Physics code score 0.0-1.0 (high if simulation code)",
            "nullable": True,
        },
        "interest_reason": {
            "type": "string",
            "description": "Explain WHY this path is interesting (required for high scores)",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, facility: str):
        super().__init__()
        self.facility = facility

    def forward(
        self,
        path: str,
        path_type: str | None = None,
        imas_relevance: float | None = None,
        physics_codes: float | None = None,
        interest_reason: str | None = None,
    ) -> str:
        """Add the path to the frontier."""
        score = None
        if imas_relevance is not None or physics_codes is not None:
            score = InterestScore(
                imas_relevance=imas_relevance or 0.5,
                physics_codes=physics_codes or 0.5,
            )

        result = discover_path(
            facility=self.facility,
            path=path,
            path_type=path_type or "directory",
            interest_score=score,
            interest_reason=interest_reason,
        )

        status = result.get("status", "unknown")
        if status == "discovered":
            return f"✓ Discovered: {path} (score: {result.get('score', 0.5):.2f})"
        elif status == "skipped":
            return f"⊘ Skipped: {path} ({result.get('reason', 'dead-end')})"
        elif status == "already_exists":
            return (
                f"○ Already known: {path} (status: {result.get('current_status', '?')})"
            )
        else:
            return f"✗ Error: {result.get('error', 'unknown')}"


class QueueFileTool(Tool):
    """Queue a source file for ingestion into the code graph."""

    name = "queue_file"
    description = (
        "Queue a source file for ingestion. "
        "Use this for valuable files you want to analyze and add to the knowledge graph. "
        "Prioritize files with IMAS patterns, physics code, or interesting data access."
    )
    inputs = {
        "path": {
            "type": "string",
            "description": "Absolute path to the source file",
        },
        "imas_relevance": {
            "type": "number",
            "description": "IMAS relevance score 0.0-1.0",
            "nullable": True,
        },
        "physics_codes": {
            "type": "number",
            "description": "Physics code score 0.0-1.0",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, facility: str):
        super().__init__()
        self.facility = facility

    def forward(
        self,
        path: str,
        imas_relevance: float | None = None,
        physics_codes: float | None = None,
    ) -> str:
        """Queue the file."""
        score = None
        if imas_relevance is not None or physics_codes is not None:
            score = InterestScore(
                imas_relevance=imas_relevance or 0.5,
                physics_codes=physics_codes or 0.5,
            )

        result = queue_source_file(
            facility=self.facility,
            path=path,
            interest_score=score,
            discovered_by="scout_agent",
        )

        status = result.get("status", "unknown")
        if status == "queued":
            return f"✓ Queued: {path} (score: {result.get('score', 0.5):.2f})"
        elif status == "already_exists":
            return f"○ Already queued: {path} (status: {result.get('current_status', '?')})"
        else:
            return f"✗ Error: {result.get('error', 'unknown')}"


class SkipPathTool(Tool):
    """Skip a path as uninteresting with an explanation."""

    name = "skip_path"
    description = (
        "Mark a path as skipped (not worth exploring). "
        "Use this when you determine a path is a dead-end or irrelevant. "
        "Always provide a clear reason for skipping."
    )
    inputs = {
        "path": {
            "type": "string",
            "description": "Absolute path to skip",
        },
        "reason": {
            "type": "string",
            "description": "Why this path should be skipped",
        },
    }
    output_type = "string"

    def __init__(self, facility: str):
        super().__init__()
        self.facility = facility

    def forward(self, path: str, reason: str) -> str:
        """Skip the path."""
        result = skip_path(
            facility=self.facility,
            path=path,
            reason=reason,
        )

        status = result.get("status", "unknown")
        if status == "skipped":
            return f"⊘ Skipped: {path} (reason: {reason})"
        else:
            return f"✗ Error: {result.get('error', 'unknown')}"


class AdvanceStatusTool(Tool):
    """Advance a path's exploration status."""

    name = "advance_status"
    description = (
        "Update a path's status after exploring it. "
        "Status lifecycle: discovered → listed → scanned → analyzed → explored. "
        "Use after listing directory contents or analyzing files."
    )
    inputs = {
        "path": {
            "type": "string",
            "description": "Path to update",
        },
        "new_status": {
            "type": "string",
            "description": "New status: listed, scanned, analyzed, or explored",
        },
        "children_discovered": {
            "type": "integer",
            "description": "Number of child paths discovered",
            "nullable": True,
        },
        "files_discovered": {
            "type": "integer",
            "description": "Number of files discovered",
            "nullable": True,
        },
        "notes": {
            "type": "string",
            "description": "Notes about what was found",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, facility: str):
        super().__init__()
        self.facility = facility

    def forward(
        self,
        path: str,
        new_status: str,
        children_discovered: int | None = None,
        files_discovered: int | None = None,
        notes: str | None = None,
    ) -> str:
        """Update the status."""
        result = advance_path_status(
            facility=self.facility,
            path=path,
            new_status=new_status,
            children_discovered=children_discovered or 0,
            files_discovered=files_discovered or 0,
            notes=notes,
        )

        status = result.get("status", "unknown")
        if status == "updated":
            return f"✓ {path} → {new_status}"
        else:
            return f"✗ Error: {result.get('error', 'unknown')}"


class GetFrontierTool(Tool):
    """Get the current exploration frontier."""

    name = "get_frontier"
    description = (
        "Get the current exploration frontier - paths waiting to be explored. "
        "Returns paths sorted by interest score (highest first). "
        "Use this to decide what to explore next."
    )
    inputs = {
        "limit": {
            "type": "integer",
            "description": "Maximum paths to return (default: 10)",
            "nullable": True,
        },
        "min_score": {
            "type": "number",
            "description": "Minimum interest score filter (default: 0.0)",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, facility: str):
        super().__init__()
        self.facility = facility

    def forward(
        self,
        limit: int | None = None,
        min_score: float | None = None,
    ) -> str:
        """Get frontier paths."""
        frontier = get_frontier(
            facility=self.facility,
            limit=limit or 10,
            min_interest_score=min_score or 0.0,
        )

        if not frontier:
            return "No paths in frontier - exploration may be complete or needs seeding"

        lines = [f"Frontier ({len(frontier)} paths):"]
        for p in frontier:
            score = p.get("interest_score", 0.5)
            reason = p.get("interest_reason", "")
            line = f"  {p['path']} (score: {score:.2f})"
            if reason:
                line += f" - {reason}"
            lines.append(line)

        return "\n".join(lines)


class GetSummaryTool(Tool):
    """Get exploration progress summary."""

    name = "get_summary"
    description = (
        "Get a summary of exploration progress. "
        "Shows total paths, explored vs remaining, files queued, etc."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, facility: str):
        super().__init__()
        self.facility = facility

    def forward(self) -> str:
        """Get summary."""
        summary = get_exploration_summary(self.facility)

        if "error" in summary:
            return f"Error: {summary['error']}"

        lines = [
            f"Exploration Summary: {self.facility}",
            f"  Total paths: {summary.get('total_paths', 0)}",
            f"  Explored: {summary.get('explored', 0)} ({summary.get('coverage', 0):.1%})",
            f"  Remaining: {summary.get('remaining', 0)}",
            f"  Files queued: {summary.get('files_queued', 0)}",
            f"  Files ingested: {summary.get('files_ingested', 0)}",
        ]

        status_counts = summary.get("status_counts", {})
        if status_counts:
            lines.append("  Status breakdown:")
            for status, count in sorted(status_counts.items()):
                lines.append(f"    {status}: {count}")

        return "\n".join(lines)


class ShellTool(Tool):
    """Execute shell commands for file exploration."""

    name = "shell"
    description = (
        "Execute a shell command to explore the filesystem. "
        "Use this for: ls, rg (fast grep), fd (fast find), head, cat, wc -l. "
        "Commands run on the target facility. "
        "Examples: 'ls -la /home', 'rg -l IMAS /path', 'fd -e py /path', 'head -50 /file.py'"
    )
    inputs = {
        "command": {
            "type": "string",
            "description": "Shell command to execute (e.g., 'ls -la /path' or 'rg pattern /path')",
        },
    }
    output_type = "string"

    def __init__(self, facility: str):
        super().__init__()
        self.facility = facility

    def forward(self, command: str) -> str:
        """Execute the command."""
        import subprocess

        # Security: Only allow read-only exploration commands
        allowed_prefixes = (
            "ls",
            "rg",
            "fd",
            "head",
            "tail",
            "cat",
            "wc",
            "file",
            "stat",
            "find",
            "grep",
            "tree",
            "du",
        )
        cmd_start = command.strip().split()[0] if command.strip() else ""
        if not any(cmd_start.startswith(p) for p in allowed_prefixes):
            return f"Error: Command '{cmd_start}' not allowed. Use: ls, rg, fd, head, tail, cat, wc, file, stat, find, grep, tree, du"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"

            # Truncate if too long
            if len(output) > 10000:
                output = output[:10000] + "\n... (truncated)"

            return output if output.strip() else "(no output)"

        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds"
        except Exception as e:
            return f"Error: {e}"


def get_scout_tools(facility: str) -> list[Tool]:
    """Get the minimal tool set for stateless scout.

    Includes:
    - Shell tool for filesystem exploration
    - Graph tools for persisting discoveries

    Args:
        facility: Facility ID

    Returns:
        List of scout tools
    """
    return [
        ShellTool(facility),  # For filesystem exploration
        DiscoverPathTool(facility),
        QueueFileTool(facility),
        SkipPathTool(facility),
        AdvanceStatusTool(facility),
        GetFrontierTool(facility),
        GetSummaryTool(facility),
    ]

"""Graph-first stateless scout for facility exploration.

Implements a stateless exploration loop where:
1. Query graph for current frontier (discovered paths)
2. LLM decides ONE action based on frontier + goals
3. Execute action (shell command or graph mutation)
4. Persist results to graph
5. Done - next step starts fresh from graph

The graph IS the state - no session context needs to be carried between steps.
The LLM queries the graph each step to understand the exploration frontier.

This module provides:
- StatelessScout: Main class orchestrating the loop
- Interest scoring with multi-dimensional criteria
- Dead-end pattern detection
- Schema-compliant graph persistence via tools
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from imas_codex.graph import GraphClient

if TYPE_CHECKING:
    from smolagents import CodeAgent

logger = logging.getLogger(__name__)

# =============================================================================
# Dead-End Patterns - Skip these entirely
# =============================================================================

DEAD_END_PATTERNS = frozenset(
    {
        ".git",
        "site-packages",
        "dist-packages",
        "__pycache__",
        ".cache",
        ".venv",
        "venv/",
        "node_modules",
        ".npm",
        ".tox",
        ".nox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "build/",
        "dist/",
        ".eggs",
        ".egg-info",
        ".svn",
        ".hg",
        ".bzr",
        "CVS",
        ".sass-cache",
        ".cargo",
        ".rustup",
        "target/debug",
        "target/release",
        "__MACOSX",
        ".DS_Store",
        "Thumbs.db",
        ".ipynb_checkpoints",
    }
)


def is_dead_end(path: str) -> tuple[bool, str | None]:
    """Check if a path is a dead-end that should be skipped.

    Returns:
        (is_dead_end, reason) tuple
    """
    path_lower = path.lower()
    for pattern in DEAD_END_PATTERNS:
        if pattern in path_lower:
            return True, f"matches dead-end pattern: {pattern}"
    return False, None


# =============================================================================
# Interest Scoring - Multi-dimensional criteria
# =============================================================================


@dataclass
class InterestScore:
    """Multi-dimensional interest score for a path.

    Interest varies by what we're looking for:
    - imas_relevance: Likelihood of IMAS integration code
    - physics_codes: Likelihood of physics simulation codes
    - data_sources: Likelihood of data files (MDSplus, HDF5, IMAS)
    - language_value: Value based on programming language

    Overall score is computed from these dimensions based on exploration focus.
    """

    imas_relevance: float = 0.5
    physics_codes: float = 0.5
    data_sources: float = 0.5
    language_value: float = 0.5

    def overall(self, weights: dict[str, float] | None = None) -> float:
        """Compute weighted overall score.

        Args:
            weights: Optional dimension weights (default: equal)

        Returns:
            Weighted average score 0.0-1.0
        """
        if weights is None:
            weights = {
                "imas_relevance": 0.4,
                "physics_codes": 0.3,
                "data_sources": 0.2,
                "language_value": 0.1,
            }

        total = 0.0
        weight_sum = 0.0
        for dim, w in weights.items():
            total += getattr(self, dim, 0.5) * w
            weight_sum += w

        return total / weight_sum if weight_sum > 0 else 0.5

    def to_dict(self) -> dict[str, float]:
        """Convert to dict for graph storage."""
        return {
            "imas_relevance": self.imas_relevance,
            "physics_codes": self.physics_codes,
            "data_sources": self.data_sources,
            "language_value": self.language_value,
            "interest_score": self.overall(),
        }


# Patterns that suggest high IMAS relevance
IMAS_PATTERNS = {
    r"imas": 0.9,
    r"ids\.(py|f90|c|cpp)": 0.8,
    r"data_entry|data_dictionary": 0.8,
    r"imasdb|uda": 0.7,
    r"put_slice|get_slice": 0.7,
    r"hcd|core_profiles|equilibrium": 0.7,
}

# Patterns that suggest physics simulation codes
PHYSICS_PATTERNS = {
    r"chease|liuqe|astra|jetto": 0.9,
    r"helena|efit|transp": 0.9,
    r"equil|transport": 0.7,
    r"mhd|gyro|turb": 0.7,
    r"nbi|ech|icrh|lhcd": 0.7,
    r"(solver|simulation)": 0.6,
}

# Language value for file extensions
LANGUAGE_VALUES = {
    ".py": 0.9,  # Python - high value for IMAS work
    ".f90": 0.8,  # Fortran - common for physics codes
    ".f": 0.7,
    ".for": 0.7,
    ".c": 0.7,
    ".cpp": 0.7,
    ".cxx": 0.7,
    ".jl": 0.6,  # Julia
    ".m": 0.5,  # MATLAB
    ".pro": 0.5,  # IDL
    ".sh": 0.4,
    ".bash": 0.4,
}


def compute_interest_score(
    path: str,
    content_sample: str | None = None,
) -> InterestScore:
    """Compute multi-dimensional interest score for a path.

    Uses heuristic pattern matching on path and optional content sample.
    For full analysis, use the LLM agent to examine content.

    Args:
        path: File or directory path
        content_sample: Optional first N lines of content

    Returns:
        InterestScore with dimension scores
    """
    path_lower = path.lower()

    # IMAS relevance
    imas_score = 0.3  # Base
    for pattern, boost in IMAS_PATTERNS.items():
        if re.search(pattern, path_lower):
            imas_score = max(imas_score, boost)

    # Check content if provided
    if content_sample:
        content_lower = content_sample.lower()
        for pattern, boost in IMAS_PATTERNS.items():
            if re.search(pattern, content_lower):
                imas_score = max(imas_score, boost)

    # Physics codes
    physics_score = 0.3
    for pattern, boost in PHYSICS_PATTERNS.items():
        if re.search(pattern, path_lower):
            physics_score = max(physics_score, boost)

    if content_sample:
        content_lower = content_sample.lower()
        for pattern, boost in PHYSICS_PATTERNS.items():
            if re.search(pattern, content_lower):
                physics_score = max(physics_score, boost)

    # Data sources
    data_score = 0.3
    if any(ext in path_lower for ext in [".h5", ".hdf5", ".nc", ".netcdf"]):
        data_score = 0.8
    if "mdsplus" in path_lower or "mds+" in path_lower:
        data_score = 0.9
    if "shot" in path_lower or "pulse" in path_lower:
        data_score = 0.7

    # Language value
    lang_score = 0.3
    for ext, value in LANGUAGE_VALUES.items():
        if path_lower.endswith(ext):
            lang_score = max(lang_score, value)

    return InterestScore(
        imas_relevance=imas_score,
        physics_codes=physics_score,
        data_sources=data_score,
        language_value=lang_score,
    )


# =============================================================================
# Graph Operations - Schema-compliant persistence
# =============================================================================


def discover_path(
    facility: str,
    path: str,
    path_type: str = "directory",
    interest_score: InterestScore | None = None,
    interest_reason: str | None = None,
) -> dict[str, Any]:
    """Add a discovered path to the graph.

    Creates a FacilityPath node with status='discovered'.

    Args:
        facility: Facility ID
        path: Absolute path on the facility
        path_type: 'directory' or 'file'
        interest_score: Multi-dimensional interest score
        interest_reason: LLM-generated reason for interest level

    Returns:
        Dict with result info
    """
    # Check for dead-ends first
    is_dead, reason = is_dead_end(path)
    if is_dead:
        return skip_path(facility, path, reason)

    path_id = f"{facility}:{path}"
    score = interest_score or compute_interest_score(path)

    props = {
        "id": path_id,
        "path": path,
        "path_type": path_type,
        "status": "discovered",
        "discovered_at": datetime.now(UTC).isoformat(),
        **score.to_dict(),
    }

    if interest_reason:
        props["interest_reason"] = interest_reason

    try:
        with GraphClient() as client:
            # Check if already exists
            existing = client.query(
                "MATCH (p:FacilityPath {id: $id}) RETURN p.status AS status",
                id=path_id,
            )
            if existing and existing[0]["status"]:
                return {
                    "status": "already_exists",
                    "path": path,
                    "current_status": existing[0]["status"],
                }

            client.query(
                """
                MERGE (p:FacilityPath {id: $id})
                SET p += $props
                WITH p
                MATCH (f:Facility {id: $facility})
                MERGE (p)-[:FACILITY_ID]->(f)
                """,
                id=path_id,
                props=props,
                facility=facility,
            )

            return {"status": "discovered", "path": path, "score": score.overall()}

    except Exception as e:
        logger.exception("Failed to discover path: %s", e)
        return {"status": "error", "path": path, "error": str(e)}


def skip_path(facility: str, path: str, reason: str) -> dict[str, Any]:
    """Mark a path as skipped (dead-end).

    Creates or updates FacilityPath with status='skipped'.
    """
    path_id = f"{facility}:{path}"

    try:
        with GraphClient() as client:
            client.query(
                """
                MERGE (p:FacilityPath {id: $id})
                SET p.path = $path,
                    p.status = 'skipped',
                    p.skip_reason = $reason,
                    p.skipped_at = $ts
                WITH p
                MATCH (f:Facility {id: $facility})
                MERGE (p)-[:FACILITY_ID]->(f)
                """,
                id=path_id,
                path=path,
                reason=reason,
                ts=datetime.now(UTC).isoformat(),
                facility=facility,
            )

            return {"status": "skipped", "path": path, "reason": reason}

    except Exception as e:
        logger.exception("Failed to skip path: %s", e)
        return {"status": "error", "path": path, "error": str(e)}


def advance_path_status(
    facility: str,
    path: str,
    new_status: str,
    children_discovered: int = 0,
    files_discovered: int = 0,
    notes: str | None = None,
) -> dict[str, Any]:
    """Advance a path's status in the exploration lifecycle.

    Status lifecycle: discovered → listed → scanned → analyzed → explored

    Args:
        facility: Facility ID
        path: Path being updated
        new_status: New status value
        children_discovered: Number of child paths discovered
        files_discovered: Number of files discovered
        notes: Optional notes about exploration
    """
    path_id = f"{facility}:{path}"

    props = {
        "status": new_status,
        f"{new_status}_at": datetime.now(UTC).isoformat(),
    }

    if children_discovered > 0:
        props["children_discovered"] = children_discovered
    if files_discovered > 0:
        props["files_discovered"] = files_discovered
    if notes:
        props["exploration_notes"] = notes

    try:
        with GraphClient() as client:
            client.query(
                """
                MATCH (p:FacilityPath {id: $id})
                SET p += $props
                """,
                id=path_id,
                props=props,
            )

            return {"status": "updated", "path": path, "new_status": new_status}

    except Exception as e:
        logger.exception("Failed to advance path status: %s", e)
        return {"status": "error", "path": path, "error": str(e)}


def queue_source_file(
    facility: str,
    path: str,
    interest_score: InterestScore | None = None,
    discovered_by: str = "scout",
) -> dict[str, Any]:
    """Queue a source file for ingestion.

    Creates a SourceFile node with status='discovered'.
    """
    file_id = f"{facility}:{path}"
    score = interest_score or compute_interest_score(path)

    props = {
        "id": file_id,
        "path": path,
        "facility_id": facility,
        "status": "discovered",
        "discovered_at": datetime.now(UTC).isoformat(),
        "discovered_by": discovered_by,
        "interest_score": score.overall(),
    }

    try:
        with GraphClient() as client:
            # Check if already exists
            existing = client.query(
                """
                MATCH (sf:SourceFile {id: $id})
                RETURN sf.status AS status
                UNION ALL
                MATCH (ce:CodeExample {source_file: $path, facility_id: $facility})
                RETURN 'ingested' AS status
                """,
                id=file_id,
                path=path,
                facility=facility,
            )
            if existing:
                return {
                    "status": "already_exists",
                    "path": path,
                    "current_status": existing[0]["status"],
                }

            client.query(
                """
                MERGE (sf:SourceFile {id: $id})
                SET sf += $props
                WITH sf
                MATCH (f:Facility {id: $facility})
                MERGE (sf)-[:FACILITY_ID]->(f)
                """,
                id=file_id,
                props=props,
                facility=facility,
            )

            return {"status": "queued", "path": path, "score": score.overall()}

    except Exception as e:
        logger.exception("Failed to queue source file: %s", e)
        return {"status": "error", "path": path, "error": str(e)}


# =============================================================================
# Frontier Queries - What to explore next
# =============================================================================


def get_frontier(
    facility: str,
    limit: int = 10,
    min_interest_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Get the current exploration frontier.

    Returns paths with status='discovered' ordered by interest score.
    These are the candidates for the next exploration step.
    """
    try:
        with GraphClient() as client:
            result = client.query(
                """
                MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
                WHERE p.status = 'discovered'
                  AND coalesce(p.interest_score, 0.5) >= $min_score
                RETURN p.path AS path,
                       p.interest_score AS interest_score,
                       p.imas_relevance AS imas_relevance,
                       p.physics_codes AS physics_codes,
                       p.path_type AS path_type,
                       p.interest_reason AS interest_reason
                ORDER BY p.interest_score DESC
                LIMIT $limit
                """,
                facility=facility,
                min_score=min_interest_score,
                limit=limit,
            )
            return list(result)

    except Exception as e:
        logger.exception("Failed to get frontier: %s", e)
        return []


def get_exploration_summary(facility: str) -> dict[str, Any]:
    """Get summary of exploration progress.

    Returns counts by status and overall coverage.
    """
    try:
        with GraphClient() as client:
            # Path status counts
            path_result = client.query(
                """
                MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
                RETURN p.status AS status, count(*) AS count
                """,
                facility=facility,
            )

            status_counts = {row["status"]: row["count"] for row in path_result}
            total = sum(status_counts.values())
            explored = sum(
                status_counts.get(s, 0)
                for s in ["listed", "scanned", "analyzed", "explored", "skipped"]
            )

            # File counts
            file_result = client.query(
                """
                MATCH (sf:SourceFile)-[:FACILITY_ID]->(f:Facility {id: $facility})
                RETURN sf.status AS status, count(*) AS count
                """,
                facility=facility,
            )
            file_counts = {row["status"]: row["count"] for row in file_result}

            return {
                "facility": facility,
                "total_paths": total,
                "explored": explored,
                "remaining": status_counts.get("discovered", 0),
                "coverage": explored / total if total > 0 else 0.0,
                "status_counts": status_counts,
                "files_queued": sum(
                    file_counts.get(s, 0) for s in ["discovered", "queued"]
                ),
                "files_ingested": file_counts.get("ingested", 0),
            }

    except Exception as e:
        logger.exception("Failed to get exploration summary: %s", e)
        return {"error": str(e)}


# =============================================================================
# Stateless Scout - Main orchestration
# =============================================================================


@dataclass
class ScoutConfig:
    """Configuration for stateless scout."""

    facility: str
    max_steps: int = 50
    exploration_focus: str = "general"  # 'imas', 'physics', 'data', 'general'
    root_paths: list[str] = field(default_factory=list)
    verbose: bool = False

    def __post_init__(self) -> None:
        if not self.root_paths:
            self.root_paths = ["/home", "/work", "/projects"]

    @property
    def interest_weights(self) -> dict[str, float]:
        """Get interest dimension weights based on focus."""
        if self.exploration_focus == "imas":
            return {
                "imas_relevance": 0.6,
                "physics_codes": 0.2,
                "data_sources": 0.1,
                "language_value": 0.1,
            }
        elif self.exploration_focus == "physics":
            return {
                "imas_relevance": 0.2,
                "physics_codes": 0.6,
                "data_sources": 0.1,
                "language_value": 0.1,
            }
        elif self.exploration_focus == "data":
            return {
                "imas_relevance": 0.2,
                "physics_codes": 0.1,
                "data_sources": 0.6,
                "language_value": 0.1,
            }
        else:  # general
            return {
                "imas_relevance": 0.4,
                "physics_codes": 0.3,
                "data_sources": 0.2,
                "language_value": 0.1,
            }


SCOUT_SYSTEM_PROMPT = """You are an exploration agent discovering physics code at a fusion research facility.

## Your Task
Explore the file system to find:
1. **IMAS integration code** - Files that read/write IMAS IDS data structures
2. **Physics simulation codes** - Equilibrium solvers (CHEASE, LIUQE), transport codes (ASTRA, JETTO)
3. **Data access patterns** - MDSplus tree access, HDF5 files, IMAS databases

## Available Actions
You have shell access via the `run(command, facility)` function which auto-detects local vs SSH.
Use these fast tools:
- `rg pattern /path` - Fast grep (10x faster than grep -r)
- `fd -e py /path` - Fast find (5x faster than find)
- `dust -d 2 /path` - Disk usage visualization

## How to Explore
1. **Start with listing** - Use `fd` or `ls` to see directory contents
2. **Search for patterns** - Use `rg` to find IMAS/physics patterns
3. **Examine interesting files** - Use `head` or `cat` to see content
4. **Queue valuable files** - Call `queue_source_file()` for ingestion

## Interest Scoring
When you discover paths, explain WHY they're interesting:
- High IMAS relevance: "Contains put_slice/get_slice calls for IDS writing"
- High physics value: "CHEASE equilibrium solver with IMAS output"
- High data value: "MDSplus tree access routines for TCV shots"

Be specific about what makes each path valuable or not.

## Dead-End Detection
Skip these patterns immediately (they're auto-filtered):
- .git, __pycache__, site-packages, node_modules
- build/, dist/, .cache/, .venv/

When you hit a dead-end, explain why and move on.

## Your Goal
Find and queue the most valuable source files for IMAS code analysis.
Focus on files that bridge local data formats to IMAS standards.
"""


def build_frontier_prompt(
    facility: str,
    frontier: list[dict[str, Any]],
    summary: dict[str, Any],
    root_paths: list[str],
) -> str:
    """Build the prompt describing current exploration state.

    This is the only context the LLM receives - it queries the graph
    each step instead of carrying accumulated context.
    """
    lines = [
        f"# Exploration Frontier: {facility}",
        "",
        "## Progress",
        f"- Total paths: {summary.get('total_paths', 0)}",
        f"- Explored: {summary.get('explored', 0)} ({summary.get('coverage', 0):.1%})",
        f"- Remaining: {summary.get('remaining', 0)}",
        f"- Files queued: {summary.get('files_queued', 0)}",
        f"- Files ingested: {summary.get('files_ingested', 0)}",
        "",
    ]

    if frontier:
        lines.append("## Frontier - Paths to Explore (highest priority first)")
        for p in frontier[:10]:
            score = p.get("interest_score", 0.5)
            reason = p.get("interest_reason", "")
            lines.append(f"- `{p['path']}` (score: {score:.2f})")
            if reason:
                lines.append(f"  Reason: {reason}")
    else:
        lines.append("## No discovered paths in frontier")
        lines.append("")
        lines.append("Start by exploring these root paths:")
        for rp in root_paths:
            lines.append(f"- `{rp}`")

    lines.extend(
        [
            "",
            "## Your Action",
            "Choose ONE action:",
            "1. List a frontier path to discover children",
            "2. Search for patterns in a path",
            "3. Examine and queue interesting files",
            "4. Skip a path as uninteresting (explain why)",
            "",
            "After your action, I'll persist results to the graph and give you the updated frontier.",
        ]
    )

    return "\n".join(lines)


class StatelessScout:
    """Graph-first stateless exploration agent.

    Each step:
    1. Query graph for frontier
    2. LLM decides one action
    3. Execute and persist
    4. Return (next step is fresh)

    Usage:
        scout = StatelessScout(ScoutConfig(facility="epfl"))

        # Single step
        result = scout.step()

        # Multiple steps
        for i in range(10):
            result = scout.step()
            if result.get("status") == "complete":
                break
    """

    def __init__(self, config: ScoutConfig) -> None:
        self.config = config
        self._agent: CodeAgent | None = None
        self.steps_taken = 0

    def _get_agent(self) -> CodeAgent:
        """Get or create the CodeAgent."""
        if self._agent is not None:
            return self._agent

        from smolagents import CodeAgent, LiteLLMModel

        from ..llm import get_model_for_task
        from .tools import get_scout_tools

        model_id = get_model_for_task("scout")
        model = LiteLLMModel(model_id=model_id)

        tools = get_scout_tools(self.config.facility)

        self._agent = CodeAgent(
            model=model,
            tools=tools,
            instructions=SCOUT_SYSTEM_PROMPT,
            max_steps=3,  # Per invocation - we control outer loop
        )

        return self._agent

    def seed_frontier(self) -> dict[str, Any]:
        """Seed the frontier with root paths if empty.

        Call this before first step if starting fresh exploration.
        """
        summary = get_exploration_summary(self.config.facility)

        if summary.get("total_paths", 0) == 0:
            results = []
            for path in self.config.root_paths:
                result = discover_path(
                    self.config.facility,
                    path,
                    path_type="directory",
                    interest_reason="Root exploration path",
                )
                results.append(result)

            return {
                "status": "seeded",
                "paths_added": len(results),
                "results": results,
            }

        return {
            "status": "already_seeded",
            "total_paths": summary.get("total_paths", 0),
        }

    def step(self) -> dict[str, Any]:
        """Execute one exploration step.

        Queries graph for frontier, invokes agent, persists results.

        Returns:
            Dict with step results and status
        """
        if self.steps_taken >= self.config.max_steps:
            return {
                "status": "max_steps_reached",
                "steps_taken": self.steps_taken,
            }

        # 1. Query graph for current state
        frontier = get_frontier(self.config.facility, limit=10)
        summary = get_exploration_summary(self.config.facility)

        # Check if exploration is complete
        if summary.get("remaining", 0) == 0 and summary.get("total_paths", 0) > 0:
            return {
                "status": "complete",
                "summary": summary,
                "steps_taken": self.steps_taken,
            }

        # 2. Build prompt with frontier
        prompt = build_frontier_prompt(
            self.config.facility,
            frontier,
            summary,
            self.config.root_paths,
        )

        # 3. Invoke agent
        agent = self._get_agent()

        try:
            result = agent.run(prompt)
            self.steps_taken += 1

            return {
                "status": "step_complete",
                "step": self.steps_taken,
                "agent_result": result,
                "frontier_size": len(frontier),
                "summary": get_exploration_summary(self.config.facility),
            }

        except Exception as e:
            logger.exception("Scout step failed: %s", e)
            return {
                "status": "error",
                "error": str(e),
                "step": self.steps_taken,
            }

    def run(self, steps: int | None = None) -> dict[str, Any]:
        """Run multiple exploration steps.

        Args:
            steps: Number of steps to run (default: config.max_steps)

        Returns:
            Final summary and step count
        """
        max_steps = steps or self.config.max_steps

        # Seed frontier if needed
        seed_result = self.seed_frontier()
        if self.config.verbose:
            logger.info("Seed result: %s", seed_result)

        results = []
        for i in range(max_steps):
            result = self.step()
            results.append(result)

            if self.config.verbose:
                logger.info("Step %d: %s", i + 1, result.get("status"))

            if result.get("status") in ("complete", "max_steps_reached", "error"):
                break

        return {
            "status": "run_complete",
            "steps_taken": self.steps_taken,
            "results": results,
            "summary": get_exploration_summary(self.config.facility),
        }

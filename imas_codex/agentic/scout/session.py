"""Scout session with windowed execution and context management.

Implements the "moving frontier" exploration pattern:
1. Execute agent in fixed-step windows (e.g., 10 steps)
2. Summarize discoveries at window boundaries using cheap model
3. Start new window with summary + fresh context
4. Persist all discoveries to graph at each step
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from imas_codex.graph import GraphClient

from ..llm import get_model_for_task

if TYPE_CHECKING:
    from llama_index.core.agent import ReActAgent

logger = logging.getLogger(__name__)

# Patterns that indicate dead-ends - paths to skip entirely
DEAD_END_PATTERNS = frozenset(
    {
        ".git",
        ".git/",
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
        "*.egg-info",
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

# Additional patterns that should be explored carefully (low priority)
LOW_PRIORITY_PATTERNS = frozenset(
    {
        "examples/",
        "example/",
        "samples/",
        "test/",
        "tests/",
        "testing/",
        "doc/",
        "docs/",
        "documentation/",
        "backup/",
        "old/",
        "deprecated/",
        "archive/",
        "tmp/",
        "temp/",
    }
)


@dataclass
class ScoutConfig:
    """Configuration for scout session execution."""

    # Facility to explore
    facility: str

    # Window size in agent steps (smaller = more frequent summarization)
    window_size: int = 10

    # Maximum total steps across all windows
    max_steps: int = 100

    # Warning threshold (percentage of window)
    warning_threshold: float = 0.8

    # Root paths to start exploration from
    root_paths: list[str] = field(default_factory=list)

    # Whether to auto-persist discoveries to graph
    auto_persist: bool = True

    # Verbosity level for callbacks
    verbose: bool = False

    def __post_init__(self) -> None:
        if not self.root_paths:
            self.root_paths = ["/home", "/work", "/projects"]


@dataclass
class WindowState:
    """State for a single execution window."""

    window_num: int
    steps_in_window: int = 0
    discoveries: list[dict] = field(default_factory=list)
    skipped_paths: list[str] = field(default_factory=list)
    queued_files: list[str] = field(default_factory=list)
    summary: str = ""
    started_at: str = ""
    ended_at: str = ""


@dataclass
class SessionState:
    """Persistent state across all windows in a scout session."""

    session_id: str
    facility: str
    total_steps: int = 0
    total_discoveries: int = 0
    total_skipped: int = 0
    total_queued: int = 0
    current_window: int = 0
    windows: list[WindowState] = field(default_factory=list)
    accumulated_summary: str = ""
    started_at: str = ""
    last_checkpoint_at: str = ""
    status: str = "running"

    def to_checkpoint(self) -> dict[str, Any]:
        """Serialize state for graph storage."""
        return {
            "session_id": self.session_id,
            "facility": self.facility,
            "total_steps": self.total_steps,
            "total_discoveries": self.total_discoveries,
            "total_skipped": self.total_skipped,
            "total_queued": self.total_queued,
            "current_window": self.current_window,
            "accumulated_summary": self.accumulated_summary,
            "started_at": self.started_at,
            "last_checkpoint_at": datetime.now(UTC).isoformat(),
            "status": self.status,
        }

    @classmethod
    def from_checkpoint(cls, data: dict[str, Any]) -> "SessionState":
        """Restore state from graph storage."""
        return cls(
            session_id=data["session_id"],
            facility=data["facility"],
            total_steps=data.get("total_steps", 0),
            total_discoveries=data.get("total_discoveries", 0),
            total_skipped=data.get("total_skipped", 0),
            total_queued=data.get("total_queued", 0),
            current_window=data.get("current_window", 0),
            accumulated_summary=data.get("accumulated_summary", ""),
            started_at=data.get("started_at", ""),
            last_checkpoint_at=data.get("last_checkpoint_at", ""),
            status=data.get("status", "running"),
        )


StepCallback = Callable[[int, int, str], None]


class ScoutSession:
    """Manages windowed scout execution with context compression.

    The session runs in windows of fixed steps, summarizing between windows
    to keep context manageable. All discoveries are persisted to the graph
    immediately, so no data is lost if the session is interrupted.

    Usage:
        config = ScoutConfig(facility="epfl", window_size=10, max_steps=50)
        session = ScoutSession(config)

        # Run with automatic window management
        result = await session.run()

        # Or resume from a previous session
        result = await session.resume(session_id="...")
    """

    def __init__(
        self,
        config: ScoutConfig,
        step_callback: StepCallback | None = None,
    ) -> None:
        self.config = config
        self.step_callback = step_callback
        self.state: SessionState | None = None
        self._agent: ReActAgent | None = None

    def is_dead_end(self, path: str) -> bool:
        """Check if a path is a dead-end that should be skipped."""
        path_lower = path.lower()
        for pattern in DEAD_END_PATTERNS:
            if pattern in path_lower:
                return True
        return False

    def is_low_priority(self, path: str) -> bool:
        """Check if a path is low priority for exploration."""
        path_lower = path.lower()
        for pattern in LOW_PRIORITY_PATTERNS:
            if pattern in path_lower:
                return True
        return False

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return f"scout-{self.config.facility}-{ts}"

    def _init_session(self, session_id: str | None = None) -> None:
        """Initialize a new session state."""
        self.state = SessionState(
            session_id=session_id or self._generate_session_id(),
            facility=self.config.facility,
            started_at=datetime.now(UTC).isoformat(),
        )

    def _should_warn_limit(self) -> bool:
        """Check if approaching step limit within current window."""
        if self.state is None:
            return False
        current_window = self.state.windows[-1] if self.state.windows else None
        if current_window is None:
            return False
        threshold = int(self.config.window_size * self.config.warning_threshold)
        return current_window.steps_in_window >= threshold

    def _should_end_window(self) -> bool:
        """Check if current window should end."""
        if self.state is None:
            return True
        current_window = self.state.windows[-1] if self.state.windows else None
        if current_window is None:
            return True
        return current_window.steps_in_window >= self.config.window_size

    def _persist_discovery(self, discovery: dict[str, Any]) -> None:
        """Persist a single discovery to the graph immediately."""
        if not self.config.auto_persist:
            return

        try:
            with GraphClient() as client:
                node_type = discovery.get("type", "FacilityPath")
                if node_type == "FacilityPath":
                    client.query(
                        """
                        MERGE (p:FacilityPath {id: $id})
                        SET p += $props
                        WITH p
                        MATCH (f:Facility {id: $facility})
                        MERGE (p)-[:LOCATED_AT]->(f)
                        """,
                        id=discovery["id"],
                        props={
                            k: v
                            for k, v in discovery.items()
                            if k not in ("type", "facility")
                        },
                        facility=self.config.facility,
                    )
                elif node_type == "SourceFile":
                    client.query(
                        """
                        MERGE (sf:SourceFile {id: $id})
                        SET sf += $props
                        WITH sf
                        MATCH (f:Facility {id: $facility})
                        MERGE (sf)-[:FACILITY_ID]->(f)
                        """,
                        id=discovery["id"],
                        props={
                            k: v
                            for k, v in discovery.items()
                            if k not in ("type", "facility")
                        },
                        facility=self.config.facility,
                    )
        except Exception as e:
            logger.exception("Failed to persist discovery: %s", e)

    def _persist_skipped_path(self, path: str, reason: str) -> None:
        """Mark a path as skipped (dead-end) in the graph."""
        if not self.config.auto_persist:
            return

        path_id = f"{self.config.facility}:{path}"
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
                    MERGE (p)-[:LOCATED_AT]->(f)
                    """,
                    id=path_id,
                    path=path,
                    reason=reason,
                    ts=datetime.now(UTC).isoformat(),
                    facility=self.config.facility,
                )
        except Exception as e:
            logger.exception("Failed to persist skipped path: %s", e)

    def _save_checkpoint(self) -> None:
        """Save session checkpoint to the graph."""
        if self.state is None:
            return

        try:
            with GraphClient() as client:
                checkpoint = self.state.to_checkpoint()
                client.query(
                    """
                    MERGE (ar:AgentRun {id: $session_id})
                    SET ar += $props,
                        ar.run_type = 'scout',
                        ar.last_checkpoint_at = $ts
                    WITH ar
                    MATCH (f:Facility {id: $facility})
                    MERGE (ar)-[:EXPLORED]->(f)
                    """,
                    session_id=self.state.session_id,
                    props=checkpoint,
                    ts=datetime.now(UTC).isoformat(),
                    facility=self.config.facility,
                )
                logger.debug("Saved checkpoint for session %s", self.state.session_id)
        except Exception as e:
            logger.exception("Failed to save checkpoint: %s", e)

    def _load_checkpoint(self, session_id: str) -> SessionState | None:
        """Load session state from the graph."""
        try:
            with GraphClient() as client:
                result = client.query(
                    """
                    MATCH (ar:AgentRun {id: $session_id})
                    RETURN ar {.*} AS checkpoint
                    """,
                    session_id=session_id,
                )
                rows = list(result)
                if rows:
                    return SessionState.from_checkpoint(rows[0]["checkpoint"])
        except Exception as e:
            logger.exception("Failed to load checkpoint: %s", e)
        return None

    def get_model(self) -> str:
        """Get the appropriate model for scout discovery."""
        return get_model_for_task("scout")

    def get_summary_model(self) -> str:
        """Get the cheap model for summarization."""
        return get_model_for_task("summarization")

    def _on_agent_step(self, step_num: int, action: str) -> None:
        """Callback for each agent step - updates state and checks limits."""
        if self.state is None:
            return

        self.state.total_steps += 1

        # Update current window
        if self.state.windows:
            self.state.windows[-1].steps_in_window += 1

        # Fire callback if provided
        if self.step_callback:
            remaining = self.config.max_steps - self.state.total_steps
            self.step_callback(step_num, remaining, action)

        # Warn if approaching limit
        if self._should_warn_limit():
            logger.warning(
                "Approaching window limit (%d/%d steps)",
                self.state.windows[-1].steps_in_window if self.state.windows else 0,
                self.config.window_size,
            )

        # Save checkpoint periodically
        if self.state.total_steps % 5 == 0:
            self._save_checkpoint()

    def record_discovery(
        self,
        path: str,
        node_type: str = "FacilityPath",
        interest_score: float = 0.5,
        status: str = "discovered",
        **extra: Any,
    ) -> None:
        """Record a discovery and persist to graph.

        Args:
            path: The discovered path
            node_type: FacilityPath or SourceFile
            interest_score: Priority score 0.0-1.0
            status: Current status (discovered, listed, skipped, etc.)
            **extra: Additional fields to store
        """
        if self.state is None:
            self._init_session()

        # Check for dead-ends
        if self.is_dead_end(path):
            self._persist_skipped_path(path, "dead-end pattern match")
            if self.state and self.state.windows:
                self.state.windows[-1].skipped_paths.append(path)
            self.state.total_skipped += 1
            return

        # Reduce interest score for low-priority paths
        if self.is_low_priority(path):
            interest_score = max(0.1, interest_score * 0.5)

        discovery = {
            "type": node_type,
            "id": f"{self.config.facility}:{path}",
            "path": path,
            "facility": self.config.facility,
            "interest_score": interest_score,
            "status": status,
            "discovered_at": datetime.now(UTC).isoformat(),
            "session_id": self.state.session_id if self.state else None,
            **extra,
        }

        # Persist immediately
        self._persist_discovery(discovery)

        # Update state
        if self.state:
            self.state.total_discoveries += 1
            if self.state.windows:
                self.state.windows[-1].discoveries.append(discovery)
            if node_type == "SourceFile":
                self.state.total_queued += 1
                if self.state.windows:
                    self.state.windows[-1].queued_files.append(path)

    def record_skipped(self, path: str, reason: str = "dead-end") -> None:
        """Record a path that was intentionally skipped."""
        if self.state is None:
            self._init_session()

        self._persist_skipped_path(path, reason)

        if self.state:
            self.state.total_skipped += 1
            if self.state.windows:
                self.state.windows[-1].skipped_paths.append(path)

    def start_window(self) -> WindowState:
        """Start a new execution window."""
        if self.state is None:
            self._init_session()

        window = WindowState(
            window_num=self.state.current_window,
            started_at=datetime.now(UTC).isoformat(),
        )
        self.state.windows.append(window)
        self.state.current_window += 1

        logger.info(
            "Starting window %d for session %s",
            window.window_num,
            self.state.session_id,
        )
        return window

    def end_window(self, summary: str = "") -> None:
        """End current window and prepare for next."""
        if self.state is None or not self.state.windows:
            return

        current = self.state.windows[-1]
        current.ended_at = datetime.now(UTC).isoformat()
        current.summary = summary

        # Accumulate summary for context
        if summary:
            self.state.accumulated_summary += (
                f"\n\n## Window {current.window_num}\n{summary}"
            )

        self._save_checkpoint()
        logger.info(
            "Ended window %d: %d discoveries, %d skipped, %d files queued",
            current.window_num,
            len(current.discoveries),
            len(current.skipped_paths),
            len(current.queued_files),
        )

    def get_context_summary(self) -> str:
        """Get accumulated summary for use in new windows."""
        if self.state is None:
            return ""
        return self.state.accumulated_summary.strip()

    def get_frontier_prompt(self) -> str:
        """Generate a prompt describing the current frontier state."""
        if self.state is None:
            return ""

        return f"""## Scout Session Context

Session: {self.state.session_id}
Facility: {self.state.facility}
Total steps: {self.state.total_steps}/{self.config.max_steps}
Discoveries: {self.state.total_discoveries}
Skipped (dead-ends): {self.state.total_skipped}
Files queued: {self.state.total_queued}
Current window: {self.state.current_window}

### Previous Discoveries Summary
{self.state.accumulated_summary or "No previous windows."}

### Dead-End Detection
Skip paths containing: {", ".join(sorted(list(DEAD_END_PATTERNS)[:10]))}...

When you encounter a path matching these patterns, mark it as skipped and move on.
Focus on paths likely to contain physics code, IMAS integrations, or MDSplus usage.
"""

    def finalize(self) -> dict[str, Any]:
        """Finalize the session and return summary statistics."""
        if self.state is None:
            return {"error": "No session state"}

        self.state.status = "completed"
        self._save_checkpoint()

        return {
            "session_id": self.state.session_id,
            "facility": self.state.facility,
            "total_steps": self.state.total_steps,
            "total_discoveries": self.state.total_discoveries,
            "total_skipped": self.state.total_skipped,
            "total_queued": self.state.total_queued,
            "windows_completed": len(self.state.windows),
            "status": "completed",
        }


def get_scout_sessions(facility: str, limit: int = 10) -> list[dict[str, Any]]:
    """Get recent scout sessions for a facility."""
    try:
        with GraphClient() as client:
            result = client.query(
                """
                MATCH (ar:AgentRun {run_type: 'scout'})-[:EXPLORED]->(f:Facility {id: $facility})
                RETURN ar.id AS session_id,
                       ar.status AS status,
                       ar.total_steps AS total_steps,
                       ar.total_discoveries AS discoveries,
                       ar.total_queued AS files_queued,
                       ar.started_at AS started_at,
                       ar.last_checkpoint_at AS last_checkpoint
                ORDER BY ar.started_at DESC
                LIMIT $limit
                """,
                facility=facility,
                limit=limit,
            )
            return list(result)
    except Exception as e:
        logger.exception("Failed to get scout sessions: %s", e)
        return []


def resume_scout_session(session_id: str) -> ScoutSession | None:
    """Resume a scout session from its checkpoint."""
    try:
        with GraphClient() as client:
            result = client.query(
                """
                MATCH (ar:AgentRun {id: $session_id})
                RETURN ar {.*} AS checkpoint
                """,
                session_id=session_id,
            )
            rows = list(result)
            if not rows:
                logger.error("Session not found: %s", session_id)
                return None

            checkpoint = rows[0]["checkpoint"]
            facility = checkpoint.get("facility", "unknown")

            config = ScoutConfig(facility=facility)
            session = ScoutSession(config)
            session.state = SessionState.from_checkpoint(checkpoint)
            session.state.status = "running"

            logger.info(
                "Resumed session %s at step %d", session_id, session.state.total_steps
            )
            return session

    except Exception as e:
        logger.exception("Failed to resume session: %s", e)
        return None

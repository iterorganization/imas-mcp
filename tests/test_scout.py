"""Tests for the scout module."""

import pytest

from imas_codex.agentic.scout.session import (
    DEAD_END_PATTERNS,
    LOW_PRIORITY_PATTERNS,
    ScoutConfig,
    ScoutSession,
    SessionState,
    WindowState,
)


class TestScoutConfig:
    """Tests for ScoutConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ScoutConfig(facility="epfl")
        assert config.facility == "epfl"
        assert config.window_size == 10
        assert config.max_steps == 100
        assert config.warning_threshold == 0.8
        assert config.auto_persist is True
        assert "/home" in config.root_paths

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ScoutConfig(
            facility="tcv",
            window_size=5,
            max_steps=50,
            root_paths=["/custom/path"],
        )
        assert config.facility == "tcv"
        assert config.window_size == 5
        assert config.max_steps == 50
        assert config.root_paths == ["/custom/path"]


class TestDeadEndDetection:
    """Tests for dead-end path detection."""

    def test_git_directory_is_dead_end(self) -> None:
        """Test that .git directories are detected as dead-ends."""
        config = ScoutConfig(facility="test", auto_persist=False)
        session = ScoutSession(config)

        assert session.is_dead_end("/home/codes/.git")
        assert session.is_dead_end("/home/codes/.git/objects")
        assert session.is_dead_end("/path/to/.git/hooks")

    def test_site_packages_is_dead_end(self) -> None:
        """Test that site-packages are detected as dead-ends."""
        config = ScoutConfig(facility="test", auto_persist=False)
        session = ScoutSession(config)

        assert session.is_dead_end("/usr/lib/python3.11/site-packages")
        assert session.is_dead_end("/home/user/.local/lib/python/site-packages/numpy")

    def test_cache_directories_are_dead_ends(self) -> None:
        """Test that cache directories are detected as dead-ends."""
        config = ScoutConfig(facility="test", auto_persist=False)
        session = ScoutSession(config)

        assert session.is_dead_end("/home/user/__pycache__")
        assert session.is_dead_end("/project/.pytest_cache")
        assert session.is_dead_end("/project/.mypy_cache")
        assert session.is_dead_end("/project/.ruff_cache")

    def test_node_modules_is_dead_end(self) -> None:
        """Test that node_modules is detected as dead-end."""
        config = ScoutConfig(facility="test", auto_persist=False)
        session = ScoutSession(config)

        assert session.is_dead_end("/project/node_modules")
        assert session.is_dead_end("/project/node_modules/react")

    def test_normal_paths_not_dead_ends(self) -> None:
        """Test that normal paths are not detected as dead-ends."""
        config = ScoutConfig(facility="test", auto_persist=False)
        session = ScoutSession(config)

        assert not session.is_dead_end("/home/codes/equilibrium")
        assert not session.is_dead_end("/work/projects/imas_tools")
        assert not session.is_dead_end("/home/user/analysis.py")

    def test_dead_end_patterns_immutable(self) -> None:
        """Test that dead-end patterns are a frozen set."""
        assert isinstance(DEAD_END_PATTERNS, frozenset)
        # Verify key patterns are present
        assert ".git" in DEAD_END_PATTERNS
        assert "site-packages" in DEAD_END_PATTERNS
        assert "__pycache__" in DEAD_END_PATTERNS


class TestLowPriorityDetection:
    """Tests for low-priority path detection."""

    def test_test_directories_are_low_priority(self) -> None:
        """Test that test directories are low priority."""
        config = ScoutConfig(facility="test", auto_persist=False)
        session = ScoutSession(config)

        assert session.is_low_priority("/project/tests/")
        assert session.is_low_priority("/project/test/unit")

    def test_docs_directories_are_low_priority(self) -> None:
        """Test that docs directories are low priority."""
        config = ScoutConfig(facility="test", auto_persist=False)
        session = ScoutSession(config)

        assert session.is_low_priority("/project/docs/")
        assert session.is_low_priority("/project/documentation/")

    def test_backup_directories_are_low_priority(self) -> None:
        """Test that backup directories are low priority."""
        config = ScoutConfig(facility="test", auto_persist=False)
        session = ScoutSession(config)

        assert session.is_low_priority("/project/backup/")
        assert session.is_low_priority("/project/old/")
        assert session.is_low_priority("/project/archive/")

    def test_code_directories_not_low_priority(self) -> None:
        """Test that code directories are not low priority."""
        config = ScoutConfig(facility="test", auto_persist=False)
        session = ScoutSession(config)

        assert not session.is_low_priority("/home/codes/equilibrium")
        assert not session.is_low_priority("/work/projects/imas_tools")


class TestSessionState:
    """Tests for session state management."""

    def test_session_initialization(self) -> None:
        """Test session initialization."""
        config = ScoutConfig(facility="epfl", auto_persist=False)
        session = ScoutSession(config)

        session._init_session()

        assert session.state is not None
        assert session.state.facility == "epfl"
        assert session.state.total_steps == 0
        assert session.state.session_id.startswith("scout-epfl-")

    def test_session_id_format(self) -> None:
        """Test session ID format."""
        config = ScoutConfig(facility="tcv", auto_persist=False)
        session = ScoutSession(config)

        session_id = session._generate_session_id()

        assert session_id.startswith("scout-tcv-")
        # Format: scout-{facility}-YYYYMMDD-HHMMSS
        parts = session_id.split("-")
        assert len(parts) == 4
        assert len(parts[2]) == 8  # YYYYMMDD
        assert len(parts[3]) == 6  # HHMMSS

    def test_window_lifecycle(self) -> None:
        """Test window start and end."""
        config = ScoutConfig(facility="test", auto_persist=False)
        session = ScoutSession(config)
        session._init_session()

        # Start window
        window = session.start_window()
        assert window.window_num == 0
        assert window.started_at != ""

        # End window
        session.end_window(summary="Test summary")
        assert session.state.windows[0].ended_at != ""
        assert session.state.windows[0].summary == "Test summary"

    def test_checkpoint_serialization(self) -> None:
        """Test session state checkpoint serialization."""
        state = SessionState(
            session_id="test-123",
            facility="epfl",
            total_steps=42,
            total_discoveries=10,
            total_skipped=5,
            accumulated_summary="Test summary",
        )

        checkpoint = state.to_checkpoint()

        assert checkpoint["session_id"] == "test-123"
        assert checkpoint["facility"] == "epfl"
        assert checkpoint["total_steps"] == 42
        assert checkpoint["total_discoveries"] == 10

    def test_checkpoint_deserialization(self) -> None:
        """Test session state restoration from checkpoint."""
        checkpoint = {
            "session_id": "test-456",
            "facility": "tcv",
            "total_steps": 100,
            "total_discoveries": 25,
            "accumulated_summary": "Previous work",
        }

        state = SessionState.from_checkpoint(checkpoint)

        assert state.session_id == "test-456"
        assert state.facility == "tcv"
        assert state.total_steps == 100
        assert state.accumulated_summary == "Previous work"


class TestDiscoveryRecording:
    """Tests for discovery recording without graph persistence."""

    def test_record_normal_discovery(self) -> None:
        """Test recording a normal discovery."""
        config = ScoutConfig(facility="test", auto_persist=False)
        session = ScoutSession(config)
        session._init_session()
        session.start_window()

        session.record_discovery("/home/codes/equilibrium", interest_score=0.8)

        assert session.state.total_discoveries == 1
        assert len(session.state.windows[0].discoveries) == 1

    def test_record_dead_end_skips_discovery(self) -> None:
        """Test that dead-ends are automatically skipped."""
        config = ScoutConfig(facility="test", auto_persist=False)
        session = ScoutSession(config)
        session._init_session()
        session.start_window()

        session.record_discovery("/home/codes/.git")

        # Dead-end should be skipped, not discovered
        assert session.state.total_discoveries == 0
        assert session.state.total_skipped == 1

    def test_low_priority_reduces_score(self) -> None:
        """Test that low-priority paths get reduced interest scores."""
        config = ScoutConfig(facility="test", auto_persist=False)
        session = ScoutSession(config)
        session._init_session()
        session.start_window()

        session.record_discovery("/home/codes/tests/", interest_score=0.8)

        # Score should be reduced (0.8 * 0.5 = 0.4)
        discovery = session.state.windows[0].discoveries[0]
        assert discovery["interest_score"] < 0.8


class TestLimitAwareness:
    """Tests for step limit awareness."""

    def test_should_warn_at_threshold(self) -> None:
        """Test warning threshold detection."""
        config = ScoutConfig(
            facility="test",
            window_size=10,
            warning_threshold=0.8,
            auto_persist=False,
        )
        session = ScoutSession(config)
        session._init_session()
        window = session.start_window()

        # At 8 of 10 steps (80%), should warn
        window.steps_in_window = 8
        assert session._should_warn_limit()

        # At 7 steps, should not warn
        window.steps_in_window = 7
        assert not session._should_warn_limit()

    def test_should_end_window(self) -> None:
        """Test window end detection."""
        config = ScoutConfig(
            facility="test",
            window_size=10,
            auto_persist=False,
        )
        session = ScoutSession(config)
        session._init_session()
        window = session.start_window()

        # At 10 steps, should end
        window.steps_in_window = 10
        assert session._should_end_window()

        # At 9 steps, should not end
        window.steps_in_window = 9
        assert not session._should_end_window()

    def test_finalize_returns_summary(self) -> None:
        """Test session finalization."""
        config = ScoutConfig(facility="test", auto_persist=False)
        session = ScoutSession(config)
        session._init_session()
        session.start_window()

        result = session.finalize()

        assert result["facility"] == "test"
        assert result["status"] == "completed"
        assert "total_steps" in result
        assert "total_discoveries" in result


class TestFrontierPrompt:
    """Tests for frontier prompt generation."""

    def test_frontier_prompt_includes_context(self) -> None:
        """Test that frontier prompt includes session context."""
        config = ScoutConfig(facility="epfl", auto_persist=False)
        session = ScoutSession(config)
        session._init_session()

        prompt = session.get_frontier_prompt()

        assert "epfl" in prompt
        assert "Scout Session Context" in prompt
        assert "Dead-End Detection" in prompt

    def test_context_summary_accumulates(self) -> None:
        """Test that context summary accumulates across windows."""
        config = ScoutConfig(facility="test", auto_persist=False)
        session = ScoutSession(config)
        session._init_session()

        session.start_window()
        session.end_window(summary="Window 0 findings")

        session.start_window()
        session.end_window(summary="Window 1 findings")

        summary = session.get_context_summary()

        assert "Window 0" in summary
        assert "Window 1" in summary
        assert "Window 0 findings" in summary
        assert "Window 1 findings" in summary

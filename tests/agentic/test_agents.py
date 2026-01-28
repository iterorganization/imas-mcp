"""
Tests for CodeAgent infrastructure.

Tests:
- Agent creation and configuration
- Tool definitions and decorators
- Cost monitoring and budget limits
- Multi-agent orchestration
"""

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.agentic.agents import (
    AgentConfig,
    create_agent,
    create_litellm_model,
    get_model_for_task,
)
from imas_codex.agentic.monitor import (
    AgentMonitor,
    BudgetExhaustedError,
    create_step_callback,
    estimate_cost,
    estimate_task_cost,
)
from imas_codex.agentic.tools import (
    AddNoteTool,
    QueueFilesTool,
    RunCommandTool,
    get_enrichment_tools,
    get_exploration_tools,
)


class TestModelConfiguration:
    """Test model selection and configuration."""

    def test_task_models_defined(self):
        """Verify task models are defined via get_model_for_task."""
        assert get_model_for_task("default") is not None
        assert get_model_for_task("enrichment") is not None
        assert get_model_for_task("exploration") is not None

    def test_get_model_for_task_known(self):
        """Get model for known task."""
        model = get_model_for_task("enrichment")
        assert isinstance(model, str)
        assert "/" in model  # Model should have provider/name format

    def test_get_model_for_task_unknown(self):
        """Unknown task falls back to default."""
        model = get_model_for_task("unknown_task")
        default_model = get_model_for_task("default")
        assert model == default_model

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"})
    def test_create_litellm_model_adds_prefix(self):
        """Model ID gets openrouter/ prefix."""
        model = create_litellm_model(model="anthropic/claude-haiku-4.5")
        assert model.model_id.startswith("openrouter/")

    def test_create_litellm_model_requires_api_key(self):
        """Raises if API key missing."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove the key if it exists
            import os

            if "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                create_litellm_model(model="test")


class TestAgentMonitor:
    """Test cost monitoring and budget enforcement."""

    def test_record_step_updates_totals(self):
        """Recording a step updates all counters."""
        monitor = AgentMonitor(agent_name="test")
        cost = monitor.record_step(input_tokens=1000, output_tokens=500)

        assert cost > 0
        assert monitor.input_tokens == 1000
        assert monitor.output_tokens == 500
        assert monitor.step_count == 1
        assert monitor.total_cost_usd == cost

    def test_record_step_with_tool(self):
        """Tool calls are tracked."""
        monitor = AgentMonitor(agent_name="test")
        monitor.record_step(tool_name="query_neo4j")

        assert monitor.tool_calls == 1

    def test_record_step_with_error(self):
        """Errors are tracked."""
        monitor = AgentMonitor(agent_name="test")
        monitor.record_step(error="Test error")

        assert len(monitor.errors) == 1
        assert "Test error" in monitor.errors[0]

    def test_is_exhausted_no_limit(self):
        """No limit means never exhausted."""
        monitor = AgentMonitor(agent_name="test", cost_limit_usd=None)
        monitor.total_cost_usd = 1000.0

        assert not monitor.is_exhausted()

    def test_is_exhausted_under_limit(self):
        """Under limit is not exhausted."""
        monitor = AgentMonitor(agent_name="test", cost_limit_usd=5.0)
        monitor.total_cost_usd = 2.0

        assert not monitor.is_exhausted()

    def test_is_exhausted_at_limit(self):
        """At limit is exhausted."""
        monitor = AgentMonitor(agent_name="test", cost_limit_usd=5.0)
        monitor.total_cost_usd = 5.0

        assert monitor.is_exhausted()

    def test_remaining_usd(self):
        """Remaining budget calculation."""
        monitor = AgentMonitor(agent_name="test", cost_limit_usd=10.0)
        monitor.total_cost_usd = 3.5

        assert monitor.remaining_usd() == 6.5

    def test_remaining_usd_no_limit(self):
        """No limit returns None."""
        monitor = AgentMonitor(agent_name="test", cost_limit_usd=None)
        assert monitor.remaining_usd() is None

    def test_check_budget_raises_when_exhausted(self):
        """check_budget raises BudgetExhaustedError."""
        monitor = AgentMonitor(agent_name="test", cost_limit_usd=1.0)
        monitor.total_cost_usd = 2.0

        with pytest.raises(BudgetExhaustedError):
            monitor.check_budget()

    def test_summary_format(self):
        """Summary returns readable string."""
        monitor = AgentMonitor(agent_name="test", cost_limit_usd=5.0)
        monitor.record_step(input_tokens=100, output_tokens=50)

        summary = monitor.summary()
        assert "test" in summary
        assert "$" in summary
        assert "step" in summary

    def test_to_dict(self):
        """Export to dictionary."""
        monitor = AgentMonitor(agent_name="test", cost_limit_usd=5.0)
        monitor.record_step(input_tokens=100, output_tokens=50)

        data = monitor.to_dict()
        assert data["agent_name"] == "test"
        assert data["cost_limit_usd"] == 5.0
        assert data["step_count"] == 1
        assert "is_exhausted" in data


class TestEstimateCost:
    """Test cost estimation functions."""

    def test_estimate_cost_known_model(self):
        """Known models use their pricing."""
        cost = estimate_cost(1000000, 1000000, "google/gemini-3-flash-preview")
        # Flash: $0.10/1M input, $0.40/1M output = $0.50
        assert 0.49 < cost < 0.51

    def test_estimate_cost_unknown_model(self):
        """Unknown models use default pricing."""
        cost = estimate_cost(1000000, 1000000, "unknown/model")
        # Default: $1.00/1M input, $5.00/1M output = $6.00
        assert 5.99 < cost < 6.01

    def test_estimate_cost_strips_prefix(self):
        """openrouter/ prefix is stripped."""
        cost1 = estimate_cost(100000, 50000, "anthropic/claude-haiku-4.5")
        cost2 = estimate_cost(100000, 50000, "openrouter/anthropic/claude-haiku-4.5")
        assert cost1 == cost2

    def test_estimate_task_cost(self):
        """Estimate cost for batch enrichment."""
        result = estimate_task_cost(1000, batch_size=100)

        assert result["num_batches"] == 10
        assert result["input_tokens"] > 0
        assert result["output_tokens"] > 0
        assert result["estimated_cost_usd"] > 0


class TestTools:
    """Test tool creation and configuration."""

    def test_get_enrichment_tools_returns_list(self):
        """Enrichment tools returns list of tools."""
        tools = get_enrichment_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 4  # At least: neo4j, code search, wiki, imas

    def test_get_exploration_tools_returns_list(self):
        """Exploration tools returns list of tools."""
        tools = get_exploration_tools("tcv")
        assert isinstance(tools, list)
        assert len(tools) >= 4  # At least: run, neo4j, queue, note, info

    def test_run_command_tool_has_facility(self):
        """RunCommandTool is bound to facility."""
        tool = RunCommandTool("iter")
        assert tool.facility == "iter"

    def test_queue_files_tool_tracks_state(self):
        """QueueFilesTool tracks queued files."""
        tool = QueueFilesTool("tcv")
        assert tool.facility == "tcv"
        assert tool.files_queued == []

    def test_add_note_tool_tracks_state(self):
        """AddNoteTool tracks notes."""
        tool = AddNoteTool("tcv")
        assert tool.facility == "tcv"
        assert tool.notes == []


class TestAgentConfig:
    """Test agent configuration dataclass."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = AgentConfig(name="test")

        assert config.name == "test"
        assert config.task == "default"
        assert config.max_steps == 20
        assert config.temperature == 0.3
        assert config.cost_limit_usd is None

    def test_custom_values(self):
        """Config accepts custom values."""
        config = AgentConfig(
            name="enrichment",
            task="enrichment",
            max_steps=10,
            cost_limit_usd=5.0,
            planning_interval=3,
        )

        assert config.name == "enrichment"
        assert config.task == "enrichment"
        assert config.max_steps == 10
        assert config.cost_limit_usd == 5.0
        assert config.planning_interval == 3


class TestCreateStepCallback:
    """Test step callback creation."""

    def test_callback_updates_monitor(self):
        """Step callback updates the monitor."""
        monitor = AgentMonitor(agent_name="test")
        callback = create_step_callback(monitor)

        # Create mock step
        mock_step = MagicMock()
        mock_step.model_output = "Test output " * 100  # ~400 chars
        mock_step.tool_calls = None
        mock_step.error = None

        callback(mock_step)

        assert monitor.step_count == 1

    def test_callback_raises_on_budget_exhausted(self):
        """Callback raises when budget exhausted."""
        monitor = AgentMonitor(agent_name="test", cost_limit_usd=0.0001)
        monitor.total_cost_usd = 0.001  # Over limit
        callback = create_step_callback(monitor)

        mock_step = MagicMock()
        mock_step.model_output = "Test"
        mock_step.tool_calls = None
        mock_step.error = None

        with pytest.raises(BudgetExhaustedError):
            callback(mock_step)


# Integration test that requires API key (skip if not available)
@pytest.mark.skipif(
    not pytest.importorskip("os").environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
class TestAgentCreation:
    """Integration tests for agent creation (requires API key)."""

    def test_create_agent_basic(self):
        """Create a basic agent."""
        from smolagents import tool

        @tool
        def dummy_tool() -> str:
            """A dummy tool for testing."""
            return "test"

        config = AgentConfig(
            name="test",
            tools=[dummy_tool],
            max_steps=1,
        )

        agent = create_agent(config)
        assert agent is not None
        assert len(agent.tools) >= 1

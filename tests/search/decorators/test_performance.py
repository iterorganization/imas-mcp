"""Tests for performance.py - performance measurement decorator."""

import pytest

from imas_mcp.search.decorators.performance import (
    PerformanceMetrics,
    calculate_performance_score,
    get_performance_summary,
    measure_performance,
)


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics class."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = PerformanceMetrics()
        assert metrics.start_time is None
        assert metrics.end_time is None
        assert metrics.success is True

    def test_start_measurement(self):
        """Test starting performance measurement."""
        metrics = PerformanceMetrics()
        metrics.start("test_func", args_count=2, kwargs_count=1)

        assert metrics.start_time is not None
        assert metrics.function_name == "test_func"
        assert metrics.args_count == 2
        assert metrics.kwargs_count == 1

    def test_end_measurement_success(self):
        """Test ending measurement with success."""
        metrics = PerformanceMetrics()
        metrics.start("test_func", 0, 0)
        metrics.end(result={"data": "value"})

        assert metrics.end_time is not None
        assert metrics.execution_time is not None
        assert metrics.success is True
        assert metrics.error_type is None

    def test_end_measurement_with_error(self):
        """Test ending measurement with error."""
        metrics = PerformanceMetrics()
        metrics.start("test_func", 0, 0)
        metrics.end(error=ValueError("Test error"))

        assert metrics.success is False
        assert metrics.error_type == "ValueError"

    def test_result_size_for_dict(self):
        """Test result size calculation for dict."""
        metrics = PerformanceMetrics()
        metrics.start("test_func", 0, 0)
        metrics.end(result={"key": "value"})

        assert metrics.result_size is not None
        assert metrics.result_size > 0

    def test_result_size_for_list(self):
        """Test result size calculation for list."""
        metrics = PerformanceMetrics()
        metrics.start("test_func", 0, 0)
        metrics.end(result=[1, 2, 3, 4, 5])

        assert metrics.result_size == 5

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = PerformanceMetrics()
        metrics.start("test_func", 1, 2)
        metrics.end(result={"data": "value"})

        result = metrics.to_dict()

        assert result["function_name"] == "test_func"
        assert result["args_count"] == 1
        assert result["kwargs_count"] == 2
        assert result["success"] is True
        assert "execution_time_ms" in result
        assert result["execution_time_ms"] is not None


class TestCalculatePerformanceScore:
    """Tests for calculate_performance_score function."""

    def test_no_execution_time(self):
        """Test score calculation without execution time."""
        metrics = PerformanceMetrics()
        result = calculate_performance_score(metrics)

        assert result["score"] == 0
        assert result["classification"] == "unknown"

    def test_excellent_performance(self):
        """Test score for excellent performance (< 0.1s)."""
        metrics = PerformanceMetrics()
        metrics.execution_time = 0.05  # 50ms

        result = calculate_performance_score(metrics)

        assert result["score"] == 100
        assert result["classification"] == "excellent"

    def test_good_performance(self):
        """Test score for good performance (0.1s - 0.5s)."""
        metrics = PerformanceMetrics()
        metrics.execution_time = 0.3  # 300ms

        result = calculate_performance_score(metrics)

        assert result["score"] == 80
        assert result["classification"] == "good"

    def test_acceptable_performance(self):
        """Test score for acceptable performance (0.5s - 2.0s)."""
        metrics = PerformanceMetrics()
        metrics.execution_time = 1.0  # 1s

        result = calculate_performance_score(metrics)

        assert result["score"] == 60
        assert result["classification"] == "acceptable"

    def test_slow_performance(self):
        """Test score for slow performance (> 2.0s)."""
        metrics = PerformanceMetrics()
        metrics.execution_time = 5.0  # 5s

        result = calculate_performance_score(metrics)

        assert result["classification"] == "slow"
        assert result["score"] < 60


class TestMeasurePerformanceDecorator:
    """Tests for measure_performance decorator."""

    @pytest.mark.asyncio
    async def test_adds_performance_metrics_to_dict_result(self):
        """Test that performance metrics are added to dict results."""

        @measure_performance(include_metrics=True)
        async def test_func():
            return {"data": "value"}

        result = await test_func()

        assert "_performance" in result
        assert "metrics" in result["_performance"]
        assert "score" in result["_performance"]

    @pytest.mark.asyncio
    async def test_can_disable_metrics(self):
        """Test that metrics can be disabled."""

        @measure_performance(include_metrics=False)
        async def test_func():
            return {"data": "value"}

        result = await test_func()

        assert "_performance" not in result

    @pytest.mark.asyncio
    async def test_preserves_original_result(self):
        """Test that original result is preserved."""

        @measure_performance()
        async def test_func():
            return {"data": "value", "count": 42}

        result = await test_func()

        assert result["data"] == "value"
        assert result["count"] == 42

    @pytest.mark.asyncio
    async def test_handles_non_dict_results(self):
        """Test handling of non-dict results."""

        @measure_performance()
        async def test_func():
            return "string result"

        result = await test_func()

        assert result == "string result"

    @pytest.mark.asyncio
    async def test_propagates_exceptions(self):
        """Test that exceptions are propagated."""

        @measure_performance()
        async def test_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await test_func()

    @pytest.mark.asyncio
    async def test_sync_function_support(self):
        """Test support for sync functions (wrapped in async)."""

        @measure_performance()
        async def test_func():
            # This is async but executes synchronously
            return {"result": "sync-like"}

        result = await test_func()
        assert result["result"] == "sync-like"


class TestGetPerformanceSummary:
    """Tests for get_performance_summary function."""

    def test_empty_results(self):
        """Test summary with no results."""
        summary = get_performance_summary([])
        assert summary["message"] == "No performance data available"

    def test_results_without_performance_data(self):
        """Test summary with results lacking performance data."""
        results = [{"data": "value1"}, {"data": "value2"}]
        summary = get_performance_summary(results)
        assert summary["message"] == "No performance data available"

    def test_valid_performance_data(self):
        """Test summary with valid performance data."""
        results = [
            {
                "_performance": {
                    "metrics": {
                        "execution_time_ms": 100,
                        "success": True,
                    }
                }
            },
            {
                "_performance": {
                    "metrics": {
                        "execution_time_ms": 200,
                        "success": True,
                    }
                }
            },
            {
                "_performance": {
                    "metrics": {
                        "execution_time_ms": 300,
                        "success": False,
                    }
                }
            },
        ]

        summary = get_performance_summary(results)

        assert summary["total_operations"] == 3
        assert summary["successful_operations"] == 2
        assert summary["success_rate"] == pytest.approx(66.7, rel=0.1)
        assert "timing_stats" in summary
        assert summary["timing_stats"]["average_time_ms"] == pytest.approx(200, rel=0.1)
        assert summary["timing_stats"]["min_time_ms"] == 100
        assert summary["timing_stats"]["max_time_ms"] == 300

    def test_performance_distribution(self):
        """Test performance distribution calculation."""
        results = [
            {
                "_performance": {"metrics": {"execution_time_ms": 50, "success": True}}
            },  # excellent
            {
                "_performance": {"metrics": {"execution_time_ms": 200, "success": True}}
            },  # good
            {
                "_performance": {
                    "metrics": {"execution_time_ms": 1000, "success": True}
                }
            },  # acceptable
            {
                "_performance": {
                    "metrics": {"execution_time_ms": 5000, "success": True}
                }
            },  # slow
        ]

        summary = get_performance_summary(results)

        assert summary["performance_distribution"]["excellent"] == 1
        assert summary["performance_distribution"]["good"] == 1
        assert summary["performance_distribution"]["acceptable"] == 1
        assert summary["performance_distribution"]["slow"] == 1

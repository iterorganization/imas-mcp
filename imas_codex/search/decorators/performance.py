"""
Performance measurement decorator for tracking execution metrics.

Provides timing, memory usage, and performance analytics for tool functions.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Container for performance metrics."""

    def __init__(self):
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.execution_time: float | None = None
        self.function_name: str = ""
        self.args_count: int = 0
        self.kwargs_count: int = 0
        self.result_size: int | None = None
        self.success: bool = True
        self.error_type: str | None = None

    def start(self, func_name: str, args_count: int, kwargs_count: int) -> None:
        """Start performance measurement."""
        self.start_time = time.perf_counter()
        self.function_name = func_name
        self.args_count = args_count
        self.kwargs_count = kwargs_count

    def end(self, result: Any = None, error: Exception | None = None) -> None:
        """End performance measurement."""
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            self.execution_time = self.end_time - self.start_time

        if error:
            self.success = False
            self.error_type = type(error).__name__
        else:
            self.success = True
            if isinstance(result, dict):
                self.result_size = len(str(result))
            elif isinstance(result, list | tuple):
                self.result_size = len(result)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "function_name": self.function_name,
            "execution_time_ms": round(self.execution_time * 1000, 2)
            if self.execution_time
            else None,
            "args_count": self.args_count,
            "kwargs_count": self.kwargs_count,
            "result_size": self.result_size,
            "success": self.success,
            "error_type": self.error_type,
            "timestamp": self.end_time,
        }


def calculate_performance_score(metrics: PerformanceMetrics) -> dict[str, Any]:
    """
    Calculate performance score and classification.

    Args:
        metrics: Performance metrics

    Returns:
        Performance score and classification
    """
    if not metrics.execution_time:
        return {"score": 0, "classification": "unknown"}

    # Performance thresholds (in seconds)
    excellent_threshold = 0.1
    good_threshold = 0.5
    acceptable_threshold = 2.0

    execution_time = metrics.execution_time

    if execution_time <= excellent_threshold:
        classification = "excellent"
        score = 100
    elif execution_time <= good_threshold:
        classification = "good"
        score = 80
    elif execution_time <= acceptable_threshold:
        classification = "acceptable"
        score = 60
    else:
        classification = "slow"
        score = max(20, 60 - int((execution_time - acceptable_threshold) * 10))

    return {
        "score": score,
        "classification": classification,
        "execution_time_ms": round(execution_time * 1000, 2),
    }


def measure_performance(
    include_metrics: bool = True,
    log_slow_operations: bool = True,
    slow_threshold: float = 1.0,
) -> Callable[[F], F]:
    """
    Decorator to measure function performance.

    Args:
        include_metrics: Whether to include metrics in result
        log_slow_operations: Whether to log slow operations
        slow_threshold: Threshold in seconds for logging slow operations

    Returns:
        Decorated function with performance measurement
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Initialize metrics
            metrics = PerformanceMetrics()

            # Start measurement
            func_name = (
                f"{func.__module__}.{func.__name__}"
                if hasattr(func, "__module__")
                else func.__name__
            )
            metrics.start(func_name, len(args), len(kwargs))

            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # End measurement
                metrics.end(result=result)

                # Log slow operations
                if (
                    log_slow_operations
                    and metrics.execution_time
                    and metrics.execution_time > slow_threshold
                ):
                    logger.warning(
                        f"Slow operation detected: {func_name} took "
                        f"{metrics.execution_time:.2f}s (threshold: {slow_threshold}s)"
                    )

                # Add performance metrics to result
                if include_metrics and isinstance(result, dict):
                    result = result.copy()
                    performance_score = calculate_performance_score(metrics)

                    result["_performance"] = {
                        "metrics": metrics.to_dict(),
                        "score": performance_score,
                    }

                return result

            except Exception as e:
                # End measurement with error
                metrics.end(error=e)

                # Log the error performance
                if log_slow_operations:
                    logger.error(
                        f"Function {func_name} failed after "
                        f"{metrics.execution_time:.2f}s: {e}"
                    )

                # Re-raise the exception
                raise

        return wrapper  # type: ignore

    return decorator


def get_performance_summary(results: list) -> dict[str, Any]:
    """
    Generate performance summary from multiple results.

    Args:
        results: List of function results with performance metrics

    Returns:
        Performance summary statistics
    """
    performance_data = []

    for result in results:
        if isinstance(result, dict) and "_performance" in result:
            perf_data = result["_performance"]
            if "metrics" in perf_data:
                performance_data.append(perf_data["metrics"])

    if not performance_data:
        return {"message": "No performance data available"}

    # Calculate statistics
    execution_times = [
        data["execution_time_ms"] / 1000
        for data in performance_data
        if data.get("execution_time_ms") is not None
    ]

    if not execution_times:
        return {"message": "No timing data available"}

    avg_time = sum(execution_times) / len(execution_times)
    min_time = min(execution_times)
    max_time = max(execution_times)

    success_count = sum(1 for data in performance_data if data.get("success", False))
    total_count = len(performance_data)

    return {
        "total_operations": total_count,
        "successful_operations": success_count,
        "success_rate": round(success_count / total_count * 100, 1),
        "timing_stats": {
            "average_time_ms": round(avg_time * 1000, 2),
            "min_time_ms": round(min_time * 1000, 2),
            "max_time_ms": round(max_time * 1000, 2),
        },
        "performance_distribution": {
            "excellent": sum(1 for t in execution_times if t <= 0.1),
            "good": sum(1 for t in execution_times if 0.1 < t <= 0.5),
            "acceptable": sum(1 for t in execution_times if 0.5 < t <= 2.0),
            "slow": sum(1 for t in execution_times if t > 2.0),
        },
    }

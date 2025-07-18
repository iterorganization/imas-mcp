"""Performance targets for IMAS MCP tools."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class PerformanceTarget:
    """Performance target for a specific benchmark."""

    name: str
    target_time: float  # seconds
    max_time: float  # seconds (failure threshold)
    memory_limit: int  # MB
    description: str


# Current tool performance targets (baseline)
CURRENT_PERFORMANCE_TARGETS = {
    "search_imas_basic": PerformanceTarget(
        name="search_imas_basic",
        target_time=2.0,
        max_time=5.0,
        memory_limit=500,
        description="Basic search without AI enhancement",
    ),
    "search_imas_with_ai": PerformanceTarget(
        name="search_imas_with_ai",
        target_time=3.0,
        max_time=8.0,
        memory_limit=600,
        description="Search with AI enhancement",
    ),
    "search_imas_complex": PerformanceTarget(
        name="search_imas_complex",
        target_time=4.0,
        max_time=10.0,
        memory_limit=700,
        description="Complex multi-term search",
    ),
    "explain_concept_basic": PerformanceTarget(
        name="explain_concept_basic",
        target_time=1.5,
        max_time=4.0,
        memory_limit=400,
        description="Basic concept explanation",
    ),
    "analyze_ids_structure": PerformanceTarget(
        name="analyze_ids_structure",
        target_time=2.5,
        max_time=6.0,
        memory_limit=600,
        description="IDS structure analysis",
    ),
    "export_ids_bulk_single": PerformanceTarget(
        name="export_ids_bulk_single",
        target_time=1.0,
        max_time=3.0,
        memory_limit=400,
        description="Single IDS bulk export",
    ),
    "export_ids_bulk_multiple": PerformanceTarget(
        name="export_ids_bulk_multiple",
        target_time=3.0,
        max_time=8.0,
        memory_limit=800,
        description="Multiple IDS bulk export",
    ),
    "explore_relationships": PerformanceTarget(
        name="explore_relationships",
        target_time=2.0,
        max_time=5.0,
        memory_limit=500,
        description="Relationship exploration",
    ),
}

# Future performance targets (after optimization)
OPTIMIZED_PERFORMANCE_TARGETS = {
    "search_imas_fast": PerformanceTarget(
        name="search_imas_fast",
        target_time=0.5,
        max_time=1.0,
        memory_limit=300,
        description="Fast lexical search mode",
    ),
    "search_imas_adaptive": PerformanceTarget(
        name="search_imas_adaptive",
        target_time=1.0,
        max_time=2.0,
        memory_limit=400,
        description="Adaptive search mode",
    ),
    "search_imas_comprehensive": PerformanceTarget(
        name="search_imas_comprehensive",
        target_time=3.0,
        max_time=6.0,
        memory_limit=600,
        description="Comprehensive search mode",
    ),
    "export_bulk_raw": PerformanceTarget(
        name="export_bulk_raw",
        target_time=0.5,
        max_time=1.0,
        memory_limit=200,
        description="Raw format bulk export",
    ),
    "export_bulk_structured": PerformanceTarget(
        name="export_bulk_structured",
        target_time=1.0,
        max_time=2.0,
        memory_limit=300,
        description="Structured format bulk export",
    ),
    "export_bulk_enhanced": PerformanceTarget(
        name="export_bulk_enhanced",
        target_time=3.0,
        max_time=6.0,
        memory_limit=500,
        description="Enhanced format bulk export",
    ),
}


def validate_performance_results(
    results: Dict[str, Any], targets: Dict[str, PerformanceTarget]
) -> Dict[str, Any]:
    """Validate performance results against targets."""
    validation_results = {"passed": [], "failed": [], "warnings": []}

    for benchmark_name, target in targets.items():
        if benchmark_name in results:
            result_time = results[benchmark_name].get("time", float("inf"))

            if result_time <= target.target_time:
                validation_results["passed"].append(
                    {
                        "benchmark": benchmark_name,
                        "time": result_time,
                        "target": target.target_time,
                        "status": "excellent",
                    }
                )
            elif result_time <= target.max_time:
                validation_results["warnings"].append(
                    {
                        "benchmark": benchmark_name,
                        "time": result_time,
                        "target": target.target_time,
                        "max_time": target.max_time,
                        "status": "acceptable",
                    }
                )
            else:
                validation_results["failed"].append(
                    {
                        "benchmark": benchmark_name,
                        "time": result_time,
                        "target": target.target_time,
                        "max_time": target.max_time,
                        "status": "failed",
                    }
                )

    return validation_results

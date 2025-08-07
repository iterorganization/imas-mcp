#!/usr/bin/env python3
"""
Simple benchmark runner that doesn't require ASV.

This script runs the benchmarks directly in the current environment
and provides timing information.
"""

import time
from statistics import mean, stdev
from typing import Dict, Any

from benchmarks.benchmarks import (
    SearchBenchmarks,
    ExplainConceptBenchmarks,
    StructureAnalysisBenchmarks,
    BulkExportBenchmarks,
)


def time_function(func, runs: int = 3) -> Dict[str, Any]:
    """Time a function multiple times and return statistics."""
    times = []

    for i in range(runs):
        start = time.perf_counter()
        try:
            func()
            end = time.perf_counter()
            times.append(end - start)
            print(f"  Run {i + 1}: {times[-1]:.3f}s")
        except Exception as e:
            print(f"  Run {i + 1}: FAILED - {e}")
            return {"error": str(e)}

    if times:
        return {
            "mean": mean(times),
            "std": stdev(times) if len(times) > 1 else 0.0,
            "min": min(times),
            "max": max(times),
            "runs": len(times),
        }
    else:
        return {"error": "All runs failed"}


def run_search_benchmarks():
    """Run search benchmarks."""
    print("\n=== Search Benchmarks ===")
    sb = SearchBenchmarks()
    print("Setting up SearchBenchmarks...")
    sb.setup()

    benchmarks = [
        ("Basic Search", sb.time_search_imas_basic),
        ("Single IDS Search", sb.time_search_imas_single_ids),
        ("Complex Query Search", sb.time_search_imas_complex_query),
    ]

    results = {}
    for name, func in benchmarks:
        print(f"\nRunning {name}:")
        results[name] = time_function(func)
        if "error" not in results[name]:
            print(
                f"  Average: {results[name]['mean']:.3f}s ± {results[name]['std']:.3f}s"
            )

    return results


def run_explain_benchmarks():
    """Run explain concept benchmarks."""
    print("\n=== Explain Concept Benchmarks ===")
    eb = ExplainConceptBenchmarks()
    print("Setting up ExplainConceptBenchmarks...")
    eb.setup()

    benchmarks = [
        ("Basic Explanation", eb.time_explain_concept_basic),
        ("Advanced Explanation", eb.time_explain_concept_advanced),
    ]

    results = {}
    for name, func in benchmarks:
        print(f"\nRunning {name}:")
        results[name] = time_function(func)
        if "error" not in results[name]:
            print(
                f"  Average: {results[name]['mean']:.3f}s ± {results[name]['std']:.3f}s"
            )

    return results


def run_analysis_benchmarks():
    """Run structure analysis benchmarks."""
    print("\n=== Structure Analysis Benchmarks ===")
    ab = StructureAnalysisBenchmarks()
    print("Setting up StructureAnalysisBenchmarks...")
    ab.setup()

    benchmarks = [
        ("Single IDS Analysis", ab.time_analyze_ids_structure_single),
        ("Equilibrium Analysis", ab.time_analyze_ids_structure_equilibrium),
    ]

    results = {}
    for name, func in benchmarks:
        print(f"\nRunning {name}:")
        results[name] = time_function(func)
        if "error" not in results[name]:
            print(
                f"  Average: {results[name]['mean']:.3f}s ± {results[name]['std']:.3f}s"
            )

    return results


def run_export_benchmarks():
    """Run bulk export benchmarks."""
    print("\n=== Bulk Export Benchmarks ===")
    bb = BulkExportBenchmarks()
    print("Setting up BulkExportBenchmarks...")
    bb.setup()

    benchmarks = [
        ("Single IDS Export", bb.time_export_ids_single),
        ("Multiple IDS Export", bb.time_export_ids_multiple),
        ("Export with Relationships", bb.time_export_ids_with_relationships),
        ("Physics Domain Export", bb.time_export_physics_domain),
    ]

    results = {}
    for name, func in benchmarks:
        print(f"\nRunning {name}:")
        results[name] = time_function(func)
        if "error" not in results[name]:
            print(
                f"  Average: {results[name]['mean']:.3f}s ± {results[name]['std']:.3f}s"
            )

    return results


def main():
    """Run all benchmarks."""
    print("IMAS MCP Benchmark Suite")
    print("=" * 50)

    all_results = {}

    try:
        all_results["Search"] = run_search_benchmarks()
    except Exception as e:
        print(f"Search benchmarks failed: {e}")
        all_results["Search"] = {"error": str(e)}

    try:
        all_results["Explain"] = run_explain_benchmarks()
    except Exception as e:
        print(f"Explain benchmarks failed: {e}")
        all_results["Explain"] = {"error": str(e)}

    try:
        all_results["Analysis"] = run_analysis_benchmarks()
    except Exception as e:
        print(f"Analysis benchmarks failed: {e}")
        all_results["Analysis"] = {"error": str(e)}

    try:
        all_results["Export"] = run_export_benchmarks()
    except Exception as e:
        print(f"Export benchmarks failed: {e}")
        all_results["Export"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)

    for category, results in all_results.items():
        print(f"\n{category}:")
        if "error" in results:
            print(f"  FAILED: {results['error']}")
        else:
            for benchmark, stats in results.items():
                if "error" in stats:
                    print(f"  {benchmark}: FAILED - {stats['error']}")
                else:
                    print(f"  {benchmark}: {stats['mean']:.3f}s ± {stats['std']:.3f}s")


if __name__ == "__main__":
    main()

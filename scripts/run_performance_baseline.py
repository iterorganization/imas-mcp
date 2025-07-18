#!/usr/bin/env python
"""Script to establish performance baseline for current tools."""

import json
import sys
import time
from pathlib import Path

# Add the project root to the path so we can import from benchmarks
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_runner import BenchmarkRunner  # noqa: E402


def main() -> int:
    """Establish performance baseline for current MCP tools."""
    print("🚀 Establishing Performance Baseline for IMAS MCP Tools")
    print("=" * 60)

    runner = BenchmarkRunner()

    # Core benchmarks to establish baseline
    core_benchmarks = [
        "SearchBenchmarks.time_search_imas_basic",
        "SearchBenchmarks.time_search_imas_single_ids",
        "SearchBenchmarks.time_search_imas_complex_query",
        "ExplainConceptBenchmarks.time_explain_concept_basic",
        "StructureAnalysisBenchmarks.time_analyze_ids_structure_single",
        "BulkExportBenchmarks.time_export_ids_bulk_single",
        "BulkExportBenchmarks.time_export_ids_bulk_multiple",
        "RelationshipBenchmarks.time_explore_relationships_depth_1",
    ]

    # First, setup the ASV machine configuration
    print("\n⚙️  Setting up ASV machine configuration...")
    machine_result = runner.setup_machine()

    if machine_result["return_code"] != 0:
        print(f"❌ Machine setup failed: {machine_result['stderr']}")
        print(f"stdout: {machine_result['stdout']}")
        return 1

    print("✅ ASV machine configuration completed")

    # Run baseline benchmarks
    print("\n📊 Running baseline benchmarks...")
    baseline_results = runner.run_benchmarks(core_benchmarks)

    if baseline_results["return_code"] != 0:
        print(f"❌ Benchmark run failed: {baseline_results['stderr']}")
        print(f"stdout: {baseline_results['stdout']}")
        return 1

    print("✅ Baseline benchmarks completed")

    # Generate HTML report
    print("\n📈 Generating HTML report...")
    html_results = runner.generate_html_report()

    if html_results["return_code"] == 0:
        print(f"✅ HTML report generated: {html_results['html_dir']}")
    else:
        print(f"❌ HTML report generation failed: {html_results['stderr']}")
        print(f"stdout: {html_results['stdout']}")

    # Save baseline metadata
    baseline_metadata = {
        "timestamp": time.time(),
        "benchmarks_run": core_benchmarks,
        "execution_time": baseline_results["execution_time"],
        "status": "success" if baseline_results["return_code"] == 0 else "failed",
        "machine_setup": machine_result["return_code"] == 0,
        "html_report": html_results["return_code"] == 0,
    }

    baseline_file = Path("benchmarks/baseline_metadata.json")
    baseline_file.parent.mkdir(exist_ok=True)
    with open(baseline_file, "w") as f:
        json.dump(baseline_metadata, f, indent=2)

    print(f"\n📄 Baseline metadata saved to: {baseline_file}")
    print("\n🎉 Performance baseline established!")

    # Show latest results if available
    latest_results = runner.get_latest_results()
    if "error" not in latest_results:
        print(f"\n📋 Latest results available at: {latest_results['file']}")

    return 0


if __name__ == "__main__":
    exit(main())

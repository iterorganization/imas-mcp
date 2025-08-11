#!/usr/bin/env python
"""Script to establish performance baseline for current tools."""

import os
import sys
from importlib import resources
from pathlib import Path

import click

# Add the project root to the path so we can import from benchmarks
# Get project root using importlib.resources
try:
    imas_mcp_package = resources.files("imas_mcp")
    project_root = Path(str(imas_mcp_package)).parent
    sys.path.insert(0, str(project_root))
except ImportError:
    # Fallback for development
    sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from benchmarks.benchmark_runner import BenchmarkRunner  # noqa: E402


@click.command()
@click.option(
    "--filter",
    "-f",
    "benchmark_filter",
    help="Filter benchmarks by pattern (e.g., 'SearchBenchmarks')",
)
def main(benchmark_filter) -> int:
    """Establish performance baseline for current MCP tools."""

    # Detect CI environment
    is_ci = os.getenv("CI", "false").lower() == "true"

    if is_ci:
        print("🔧 Running in CI environment")

    print("🚀 Establishing Performance Baseline for IMAS MCP Tools")
    print("=" * 60)

    runner = BenchmarkRunner()

    # Determine which benchmarks to run
    if benchmark_filter:
        print(f"\n🔍 Running benchmarks matching filter: '{benchmark_filter}'")
    else:
        print("\n📊 Running all available benchmarks...")

    # First, setup the ASV machine configuration
    print("\n⚙️  Setting up ASV machine configuration...")
    machine_result = runner.setup_machine()

    if machine_result["return_code"] != 0:
        print(f"⚠️  Machine setup had issues: {machine_result['stderr']}")
        print(f"stdout: {machine_result['stdout']}")
        # Try to continue anyway - ASV might still work with default config
        print("🔄 Attempting to continue with default configuration...")
    else:
        print("✅ ASV machine configuration completed")

    # Run benchmarks
    print("\n📊 Running benchmarks...")

    # Choose progress display based on environment
    if is_ci:
        # Simple progress for CI logs
        print("📊 Running all benchmarks in CI mode...")

        if benchmark_filter:
            result = runner.run_benchmarks([benchmark_filter])
        else:
            result = runner.run_benchmarks(None)  # Run all benchmarks

        if result["return_code"] != 0:
            print(f"❌ Benchmarks failed: {result['stderr']}")
            return 1
        else:
            print("✅ All benchmarks completed")
    else:
        # Rich progress bar for local development
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
        ) as progress:
            # Add task with indeterminate progress since we don't know count upfront
            benchmark_task = progress.add_task("Running benchmarks...", total=None)

            # Run all benchmarks
            if benchmark_filter:
                result = runner.run_benchmarks([benchmark_filter])
            else:
                result = runner.run_benchmarks(None)  # Run all benchmarks

            if result["return_code"] != 0:
                progress.update(benchmark_task, description="❌ Benchmarks failed")
                print(f"❌ Benchmarks failed: {result['stderr']}")
                print(f"stdout: {result['stdout']}")
                return 1
            else:
                progress.update(
                    benchmark_task, description="✅ All benchmarks completed"
                )

    print("✅ Benchmarks completed")

    # Generate HTML report
    print("\n📈 Generating HTML report...")
    html_results = runner.generate_html_report()

    if html_results["return_code"] == 0:
        print(f"✅ HTML report generated: {html_results['html_dir']}")
    else:
        print(f"❌ HTML report generation failed: {html_results['stderr']}")
        print(f"stdout: {html_results['stdout']}")

    # Display performance summary using ASV's native tools
    print("\n📊 Performance Summary:")
    print("=" * 50)

    # Try to get recent results using ASV
    latest_results = runner.get_latest_results()
    if "error" not in latest_results and "data" in latest_results:
        results_data = latest_results["data"]
        if "results" in results_data:
            # Display all available benchmark results
            for full_name, result_data in results_data["results"].items():
                if result_data and len(result_data) > 0:
                    time_result = result_data[0]
                    if isinstance(time_result, list) and len(time_result) > 0:
                        time_ms = time_result[0] * 1000
                        # Extract method name from full benchmark name
                        method_name = (
                            full_name.split(".")[-1] if "." in full_name else full_name
                        )
                        print(f"  {method_name:<40} {time_ms:>8.2f} ms")
    else:
        print("  No performance data available. Check ASV results.")

    print("\n🎉 Performance benchmarks completed!")

    # Show HTML results link
    html_path = Path(".asv/html/index.html").absolute()
    print(f"\n📋 View results at: file:///{html_path}")
    print("💻 Or run the following commands to serve the results:")
    print("   cd .asv\\html")
    print("   python -m http.server 8000")
    print("   Then open: http://localhost:8000")

    if is_ci:
        print(
            "\n💡 In CI: Use 'asv show' or 'asv compare' commands for detailed analysis"
        )

    return 0


if __name__ == "__main__":
    main()

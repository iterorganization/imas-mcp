import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


class BenchmarkRunner:
    """Utility for running and managing ASV benchmarks."""

    def __init__(self, benchmark_dir: Path = Path("benchmarks")):
        self.benchmark_dir = benchmark_dir
        self.results_dir = Path(".asv/results")
        self.html_dir = Path(".asv/html")

    def run_benchmarks(
        self, benchmark_names: list[str] | None = None
    ) -> dict[str, Any]:
        """Run ASV benchmarks and return results."""
        cmd = ["asv", "run", "--python=3.12"]

        # Use the specific machine if we're in GitHub Actions
        if os.getenv("GITHUB_ACTIONS", "").lower() == "true":
            cmd.extend(["--machine", "github-actions"])

        if benchmark_names:
            # Use individual -b flags for each benchmark
            for benchmark in benchmark_names:
                cmd.extend(["-b", benchmark])

        print(f"Running benchmarks: {' '.join(cmd)}")

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()

        return {
            "command": " ".join(cmd),
            "execution_time": end_time - start_time,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def generate_html_report(self) -> dict[str, Any]:
        """Generate HTML benchmark report."""
        cmd = ["asv", "publish"]

        print(f"Generating HTML report: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        return {
            "command": " ".join(cmd),
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "html_dir": str(self.html_dir.absolute()),
        }

    def compare_benchmarks(self, commit1: str, commit2: str) -> dict[str, Any]:
        """Compare benchmarks between two commits."""
        cmd = ["asv", "compare", commit1, commit2]

        print(f"Comparing benchmarks: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        return {
            "command": " ".join(cmd),
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def get_latest_results(self) -> dict[str, Any]:
        """Get latest benchmark results with actual timing data."""
        if not self.results_dir.exists():
            return {"error": "No benchmark results found"}

        # Look for machine-specific results which contain actual timing data
        machine_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        if not machine_dirs:
            return {"error": "No machine-specific result directories found"}

        # Get the latest machine directory
        latest_machine_dir = max(machine_dirs, key=lambda d: d.stat().st_mtime)

        # Find the latest result file in that directory
        result_files = list(latest_machine_dir.glob("*.json"))
        if not result_files:
            return {"error": "No benchmark result files found"}

        # Filter out machine.json and get actual benchmark results
        benchmark_files = [
            f for f in result_files if not f.name.endswith("machine.json")
        ]
        if not benchmark_files:
            return {"error": "No benchmark timing files found"}

        latest_file = max(benchmark_files, key=lambda f: f.stat().st_mtime)

        try:
            with open(latest_file) as f:
                data = json.load(f)
            return {
                "file": str(latest_file),
                "data": data,
                "timestamp": latest_file.stat().st_mtime,
            }
        except Exception as e:
            return {"error": f"Failed to read results: {e}"}

    def setup_machine(self) -> dict[str, Any]:
        """Setup ASV machine configuration with fallback for CI environments."""
        import os

        # Check if we're in a CI environment
        is_ci = os.getenv("CI", "").lower() == "true"
        is_github_actions = os.getenv("GITHUB_ACTIONS", "").lower() == "true"

        if is_github_actions:
            # GitHub Actions - use predefined machine config
            machine_name = "github-actions"
            cmd = ["asv", "machine", "--machine", machine_name]
        elif is_ci:
            # Other CI environments - try with --yes flag
            cmd = ["asv", "machine", "--yes"]
        else:
            # Local development - interactive setup
            cmd = ["asv", "machine", "--yes"]

        print(f"Setting up ASV machine: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return {
                "command": " ".join(cmd),
                "return_code": 1,
                "stdout": "",
                "stderr": "Machine setup timed out after 60 seconds",
            }

        return {
            "command": " ".join(cmd),
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def list_benchmarks(self) -> dict[str, Any]:
        """List all available benchmarks by scanning benchmark files."""
        import ast

        benchmarks = []
        benchmark_files = list(self.benchmark_dir.glob("*.py"))

        for file_path in benchmark_files:
            if file_path.name.startswith("__"):
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Parse the AST to find classes and methods
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_name = node.name
                        if class_name.endswith("Benchmarks"):
                            # Find benchmark methods in this class
                            for item in node.body:
                                if isinstance(item, ast.FunctionDef):
                                    method_name = item.name
                                    if method_name.startswith(
                                        ("time_", "mem_", "peakmem_")
                                    ):
                                        benchmarks.append(f"{class_name}.{method_name}")

            except Exception as e:
                print(f"Warning: Could not parse {file_path}: {e}")
                continue

        return {
            "benchmarks": sorted(benchmarks),
            "total_count": len(benchmarks),
            "return_code": 0,
            "stdout": "\n".join(sorted(benchmarks)),
            "stderr": "",
        }

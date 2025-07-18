import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Any, List


class BenchmarkRunner:
    """Utility for running and managing ASV benchmarks."""

    def __init__(self, benchmark_dir: Path = Path("benchmarks")):
        self.benchmark_dir = benchmark_dir
        self.results_dir = Path(".asv/results")
        self.html_dir = Path(".asv/html")

    def run_benchmarks(
        self, benchmark_names: List[str] | None = None
    ) -> Dict[str, Any]:
        """Run ASV benchmarks and return results."""
        cmd = ["asv", "run", "--python=3.12"]

        if benchmark_names:
            cmd.extend(["-b", ",".join(benchmark_names)])

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

    def generate_html_report(self) -> Dict[str, Any]:
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

    def compare_benchmarks(self, commit1: str, commit2: str) -> Dict[str, Any]:
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

    def get_latest_results(self) -> Dict[str, Any]:
        """Get latest benchmark results."""
        if not self.results_dir.exists():
            return {"error": "No benchmark results found"}

        # Find the latest results file
        result_files = list(self.results_dir.glob("*.json"))
        if not result_files:
            return {"error": "No benchmark result files found"}

        latest_file = max(result_files, key=lambda f: f.stat().st_mtime)

        try:
            with open(latest_file, "r") as f:
                data = json.load(f)
            return {
                "file": str(latest_file),
                "data": data,
                "timestamp": latest_file.stat().st_mtime,
            }
        except Exception as e:
            return {"error": f"Failed to read results: {e}"}

    def setup_machine(self) -> Dict[str, Any]:
        """Setup ASV machine configuration."""
        cmd = ["asv", "machine", "--yes"]

        print(f"Setting up ASV machine: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        return {
            "command": " ".join(cmd),
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

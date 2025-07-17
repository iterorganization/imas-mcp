# IMAS MCP Tools Development Plan

## Executive Summary

This plan implements critical improvements to the IMAS MCP server based on comprehensive tool analysis, focusing on performance optimization, selective AI enhancement, and expanded capabilities with MCP resources and prompts.

## Phase 0: Testing Foundation & Performance Baseline (Weeks 1-2)

### 0.1 Pytest Unit Tests for Current MCP Tools

**Purpose**: Establish comprehensive test coverage for existing tools before refactoring

```python
# File: tests/conftest.py
import pytest
import asyncio
from typing import Generator, Dict, Any
from unittest.mock import Mock, AsyncMock
from functools import cached_property
from fastmcp.testing import MockContext
from imas_mcp.server import Server
from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.semantic_search import SemanticSearch
from imas_mcp.graph_analyzer import IMASGraphAnalyzer

class TestServerFixture:
    """Long-lived server fixture for expensive operations using cached_property."""

    @cached_property
    def server(self) -> Server:
        """Get or create server instance - cached for performance."""
        return Server()

    @cached_property
    def document_store(self) -> DocumentStore:
        """Get or create document store - cached for performance."""
        return self.server.document_store

    @cached_property
    def semantic_search(self) -> SemanticSearch:
        """Get or create semantic search - cached for performance."""
        return self.server.semantic_search

    @cached_property
    def graph_analyzer(self) -> IMASGraphAnalyzer:
        """Get or create graph analyzer - cached for performance."""
        return self.server.graph_analyzer

@pytest.fixture(scope="session")
def test_server() -> TestServerFixture:
    """Session-scoped server fixture to avoid costly recreations."""
    return TestServerFixture()

@pytest.fixture
def mock_context() -> MockContext:
    """Create mock MCP context for testing."""
    context = MockContext()
    context.sample = AsyncMock(return_value="Mock AI response")
    return context

@pytest.fixture
def sample_search_results() -> Dict[str, Any]:
    """Sample search results for testing."""
    return {
        "results": [
            {
                "path": "core_profiles/profiles_1d/electrons/temperature",
                "ids_name": "core_profiles",
                "score": 0.95,
                "documentation": "Electron temperature profile"
            },
            {
                "path": "equilibrium/time_slice/profiles_1d/psi",
                "ids_name": "equilibrium",
                "score": 0.88,
                "documentation": "Poloidal flux profile"
            }
        ],
        "total_results": 2
    }

# File: tests/test_current_tools.py
import pytest
from unittest.mock import AsyncMock, patch
from imas_mcp.server import Server

class TestCurrentSearchImas:
    """Test suite for current search_imas implementation."""

    @pytest.mark.asyncio
    async def test_search_imas_basic_query(self, test_server, mock_context):
        """Test basic search functionality."""
        result = await test_server.server.search_imas(
            query="plasma temperature",
            max_results=10,
            ctx=mock_context
        )

        assert isinstance(result, dict)
        assert "results" in result
        assert "total_results" in result
        assert len(result["results"]) <= 10

    @pytest.mark.asyncio
    async def test_search_imas_with_ids_filter(self, test_server, mock_context):
        """Test search with IDS filtering."""
        result = await test_server.server.search_imas(
            query="temperature",
            ids_name="core_profiles",
            max_results=5,
            ctx=mock_context
        )

        assert isinstance(result, dict)
        assert "results" in result
        # All results should be from core_profiles
        for item in result["results"]:
            assert item["ids_name"] == "core_profiles"

    @pytest.mark.asyncio
    async def test_search_imas_list_query(self, test_server, mock_context):
        """Test search with list of query terms."""
        result = await test_server.server.search_imas(
            query=["plasma", "temperature", "density"],
            max_results=15,
            ctx=mock_context
        )

        assert isinstance(result, dict)
        assert "results" in result
        assert len(result["results"]) <= 15

    @pytest.mark.asyncio
    async def test_search_imas_without_context(self, test_server):
        """Test search without AI context."""
        result = await test_server.server.search_imas(
            query="equilibrium",
            max_results=5
        )

        assert isinstance(result, dict)
        assert "results" in result
        # Should work without AI enhancement
        assert "ai_enhancement" not in result or result["ai_enhancement"] is None

class TestCurrentExplainConcept:
    """Test suite for current explain_concept implementation."""

    @pytest.mark.asyncio
    async def test_explain_concept_basic(self, test_server, mock_context):
        """Test basic concept explanation."""
        result = await test_server.server.explain_concept(
            concept="plasma temperature",
            ctx=mock_context
        )

        assert isinstance(result, dict)
        assert "concept" in result
        assert result["concept"] == "plasma temperature"

    @pytest.mark.asyncio
    async def test_explain_concept_detail_levels(self, test_server, mock_context):
        """Test different detail levels."""
        for level in ["basic", "intermediate", "advanced"]:
            result = await test_server.server.explain_concept(
                concept="transport coefficients",
                detail_level=level,
                ctx=mock_context
            )

            assert isinstance(result, dict)
            assert "detail_level" in result
            assert result["detail_level"] == level

class TestCurrentGetOverview:
    """Test suite for current get_overview implementation."""

    @pytest.mark.asyncio
    async def test_get_overview_general(self, test_server, mock_context):
        """Test general overview functionality."""
        result = await test_server.server.get_overview(ctx=mock_context)

        assert isinstance(result, dict)
        assert "available_ids" in result
        assert "total_paths" in result
        assert isinstance(result["available_ids"], list)
        assert isinstance(result["total_paths"], int)

    @pytest.mark.asyncio
    async def test_get_overview_with_question(self, test_server, mock_context):
        """Test overview with specific question."""
        result = await test_server.server.get_overview(
            question="How many paths are in core_profiles?",
            ctx=mock_context
        )

        assert isinstance(result, dict)
        assert "question" in result
        assert result["question"] == "How many paths are in core_profiles?"

class TestCurrentAnalyzeIDSStructure:
    """Test suite for current analyze_ids_structure implementation."""

    @pytest.mark.asyncio
    async def test_analyze_ids_structure_valid_ids(self, test_server, mock_context):
        """Test IDS structure analysis for valid IDS."""
        result = await test_server.server.analyze_ids_structure(
            ids_name="core_profiles",
            ctx=mock_context
        )

        assert isinstance(result, dict)
        assert "ids_name" in result
        assert result["ids_name"] == "core_profiles"
        assert "structure_analysis" in result

    @pytest.mark.asyncio
    async def test_analyze_ids_structure_invalid_ids(self, test_server, mock_context):
        """Test IDS structure analysis for invalid IDS."""
        result = await test_server.server.analyze_ids_structure(
            ids_name="nonexistent_ids",
            ctx=mock_context
        )

        assert isinstance(result, dict)
        assert "error" in result or "ids_name" in result

class TestCurrentExploreRelationships:
    """Test suite for current explore_relationships implementation."""

    @pytest.mark.asyncio
    async def test_explore_relationships_basic(self, test_server, mock_context):
        """Test basic relationship exploration."""
        result = await test_server.server.explore_relationships(
            path="core_profiles/profiles_1d/electrons/temperature",
            ctx=mock_context
        )

        assert isinstance(result, dict)
        assert "path" in result
        assert "relationships" in result

    @pytest.mark.asyncio
    async def test_explore_relationships_with_depth(self, test_server, mock_context):
        """Test relationship exploration with different depths."""
        for depth in [1, 2, 3]:
            result = await test_server.server.explore_relationships(
                path="equilibrium/time_slice/profiles_1d/psi",
                max_depth=depth,
                ctx=mock_context
            )

            assert isinstance(result, dict)
            assert "max_depth" in result
            assert result["max_depth"] == depth

class TestCurrentExploreIdentifiers:
    """Test suite for current explore_identifiers implementation."""

    @pytest.mark.asyncio
    async def test_explore_identifiers_basic(self, test_server, mock_context):
        """Test basic identifier exploration."""
        result = await test_server.server.explore_identifiers(
            ctx=mock_context
        )

        assert isinstance(result, dict)
        assert "identifier_schemas" in result

    @pytest.mark.asyncio
    async def test_explore_identifiers_with_query(self, test_server, mock_context):
        """Test identifier exploration with query."""
        result = await test_server.server.explore_identifiers(
            query="temperature",
            ctx=mock_context
        )

        assert isinstance(result, dict)
        assert "query" in result
        assert result["query"] == "temperature"

class TestCurrentExportIDSBulk:
    """Test suite for current export_ids_bulk implementation."""

    @pytest.mark.asyncio
    async def test_export_ids_bulk_single_ids(self, test_server, mock_context):
        """Test bulk export with single IDS."""
        result = await test_server.server.export_ids_bulk(
            ids_list=["core_profiles"],
            ctx=mock_context
        )

        assert isinstance(result, dict)
        assert "ids_list" in result
        assert result["ids_list"] == ["core_profiles"]

    @pytest.mark.asyncio
    async def test_export_ids_bulk_multiple_ids(self, test_server, mock_context):
        """Test bulk export with multiple IDS."""
        result = await test_server.server.export_ids_bulk(
            ids_list=["core_profiles", "equilibrium", "transport"],
            ctx=mock_context
        )

        assert isinstance(result, dict)
        assert "ids_list" in result
        assert len(result["ids_list"]) == 3

    @pytest.mark.asyncio
    async def test_export_ids_bulk_with_relationships(self, test_server, mock_context):
        """Test bulk export with relationships."""
        result = await test_server.server.export_ids_bulk(
            ids_list=["core_profiles", "equilibrium"],
            include_relationships=True,
            ctx=mock_context
        )

        assert isinstance(result, dict)
        assert "include_relationships" in result
        assert result["include_relationships"] is True

class TestCurrentExportPhysicsDomain:
    """Test suite for current export_physics_domain implementation."""

    @pytest.mark.asyncio
    async def test_export_physics_domain_basic(self, test_server, mock_context):
        """Test physics domain export."""
        result = await test_server.server.export_physics_domain(
            domain="equilibrium",
            ctx=mock_context
        )

        assert isinstance(result, dict)
        assert "domain" in result
        assert result["domain"] == "equilibrium"

    @pytest.mark.asyncio
    async def test_export_physics_domain_with_cross_domain(self, test_server, mock_context):
        """Test physics domain export with cross-domain analysis."""
        result = await test_server.server.export_physics_domain(
            domain="transport",
            include_cross_domain=True,
            ctx=mock_context
        )

        assert isinstance(result, dict)
        assert "include_cross_domain" in result
        assert result["include_cross_domain"] is True

# File: tests/test_current_integration.py
import pytest
from unittest.mock import patch, AsyncMock

class TestCurrentIntegration:
    """Integration tests for current MCP tool interactions."""

    @pytest.mark.asyncio
    async def test_search_to_explain_workflow(self, test_server, mock_context):
        """Test workflow: search -> explain concept."""
        # Step 1: Search for concepts
        search_result = await test_server.server.search_imas(
            query="plasma temperature",
            max_results=5,
            ctx=mock_context
        )

        assert "results" in search_result

        # Step 2: Explain a concept found in search
        if search_result["results"]:
            explain_result = await test_server.server.explain_concept(
                concept="plasma temperature",
                ctx=mock_context
            )

            assert "concept" in explain_result

    @pytest.mark.asyncio
    async def test_search_to_structure_analysis_workflow(self, test_server, mock_context):
        """Test workflow: search -> analyze IDS structure."""
        # Step 1: Search for paths
        search_result = await test_server.server.search_imas(
            query="equilibrium profiles",
            max_results=10,
            ctx=mock_context
        )

        assert "results" in search_result

        # Step 2: Analyze structure of found IDS
        if search_result["results"]:
            ids_name = search_result["results"][0]["ids_name"]
            structure_result = await test_server.server.analyze_ids_structure(
                ids_name=ids_name,
                ctx=mock_context
            )

            assert "ids_name" in structure_result

    @pytest.mark.asyncio
    async def test_bulk_export_workflow(self, test_server, mock_context):
        """Test bulk export workflow."""
        # Get overview first
        overview_result = await test_server.server.get_overview(ctx=mock_context)

        assert "available_ids" in overview_result

        # Export subset of available IDS
        if overview_result["available_ids"]:
            ids_subset = overview_result["available_ids"][:3]
            export_result = await test_server.server.export_ids_bulk(
                ids_list=ids_subset,
                ctx=mock_context
            )

            assert "ids_list" in export_result
```

### 0.2 ASV Performance Monitoring Setup

**Purpose**: Establish performance baselines and regression detection

```python
# File: benchmarks/asv.conf.json
{
    "version": 1,
    "project": "imas-mcp",
    "project_url": "https://github.com/user/imas-mcp",
    "repo": ".",
    "branches": ["main"],
    "environment_type": "conda",
    "pythons": ["3.12"],
    "matrix": {
        "python": ["3.12"],
        "req": {"pip": ["requirements.txt"]}
    },
    "benchmark_dir": "benchmarks",
    "env_dir": ".asv/env",
    "results_dir": ".asv/results",
    "html_dir": ".asv/html",
    "build_cache_size": 8
}

# File: benchmarks/benchmarks.py
import asyncio
import time
from typing import List, Dict, Any
from unittest.mock import AsyncMock
from functools import cached_property
from fastmcp.testing import MockContext

# Import using composition to avoid costly imports in benchmark setup
class BenchmarkFixture:
    """Composition-based benchmark fixture for performance testing."""

    @cached_property
    def server(self):
        """Lazy-loaded server instance."""
        from imas_mcp.server import Server
        return Server()

    @cached_property
    def mock_context(self):
        """Lazy-loaded mock context."""
        mock_context = MockContext()
        mock_context.sample = AsyncMock(return_value="Mock AI response")
        return mock_context

    @cached_property
    def sample_queries(self) -> List[str]:
        """Sample queries for benchmarking."""
        return [
            "plasma temperature",
            "equilibrium profiles",
            "transport coefficients",
            "electron density",
            "magnetic field"
        ]

    @cached_property
    def sample_ids_lists(self) -> List[List[str]]:
        """Sample IDS lists for benchmarking."""
        return [
            ["core_profiles"],
            ["core_profiles", "equilibrium"],
            ["core_profiles", "equilibrium", "transport"],
            ["core_profiles", "equilibrium", "transport", "ec_launchers"],
            ["core_profiles", "equilibrium", "transport", "ec_launchers", "nbi"]
        ]

# Global benchmark fixture
_benchmark_fixture = BenchmarkFixture()

class SearchBenchmarks:
    """Benchmark suite for search_imas tool."""

    def setup(self):
        """Setup benchmark environment."""
        self.fixture = _benchmark_fixture
        # Warm up the server components
        asyncio.run(self._warmup())

    async def _warmup(self):
        """Warm up server components to avoid cold start penalties."""
        # Initialize cached properties
        _ = self.fixture.server.document_store
        _ = self.fixture.server.semantic_search
        _ = self.fixture.server.graph_analyzer

        # Perform a small search to warm up indices
        await self.fixture.server.search_imas(
            query="test",
            max_results=1
        )

    def time_search_imas_basic(self):
        """Benchmark basic search performance."""
        async def run_search():
            return await self.fixture.server.search_imas(
                query="plasma temperature",
                max_results=10
            )

        return asyncio.run(run_search())

    def time_search_imas_with_ai(self):
        """Benchmark search with AI enhancement."""
        async def run_search():
            return await self.fixture.server.search_imas(
                query="plasma temperature",
                max_results=10,
                ctx=self.fixture.mock_context
            )

        return asyncio.run(run_search())

    def time_search_imas_complex_query(self):
        """Benchmark complex query performance."""
        async def run_search():
            return await self.fixture.server.search_imas(
                query=["plasma", "temperature", "density", "profile"],
                max_results=20
            )

        return asyncio.run(run_search())

    def time_search_imas_ids_filter(self):
        """Benchmark search with IDS filtering."""
        async def run_search():
            return await self.fixture.server.search_imas(
                query="temperature",
                ids_name="core_profiles",
                max_results=15
            )

        return asyncio.run(run_search())

    def peakmem_search_imas_basic(self):
        """Benchmark memory usage for basic search."""
        async def run_search():
            return await self.fixture.server.search_imas(
                query="plasma temperature",
                max_results=10
            )

        return asyncio.run(run_search())

class ExplainConceptBenchmarks:
    """Benchmark suite for explain_concept tool."""

    def setup(self):
        """Setup benchmark environment."""
        self.fixture = _benchmark_fixture
        asyncio.run(self._warmup())

    async def _warmup(self):
        """Warm up server components."""
        _ = self.fixture.server.document_store
        await self.fixture.server.explain_concept(
            concept="test",
            ctx=self.fixture.mock_context
        )

    def time_explain_concept_basic(self):
        """Benchmark basic concept explanation."""
        async def run_explain():
            return await self.fixture.server.explain_concept(
                concept="plasma temperature",
                ctx=self.fixture.mock_context
            )

        return asyncio.run(run_explain())

    def time_explain_concept_advanced(self):
        """Benchmark advanced concept explanation."""
        async def run_explain():
            return await self.fixture.server.explain_concept(
                concept="transport coefficients",
                detail_level="advanced",
                ctx=self.fixture.mock_context
            )

        return asyncio.run(run_explain())

class StructureAnalysisBenchmarks:
    """Benchmark suite for analyze_ids_structure tool."""

    def setup(self):
        """Setup benchmark environment."""
        self.fixture = _benchmark_fixture
        asyncio.run(self._warmup())

    async def _warmup(self):
        """Warm up server components."""
        _ = self.fixture.server.graph_analyzer
        await self.fixture.server.analyze_ids_structure(
            ids_name="core_profiles",
            ctx=self.fixture.mock_context
        )

    def time_analyze_ids_structure_small(self):
        """Benchmark structure analysis for small IDS."""
        async def run_analysis():
            return await self.fixture.server.analyze_ids_structure(
                ids_name="core_profiles",
                ctx=self.fixture.mock_context
            )

        return asyncio.run(run_analysis())

    def time_analyze_ids_structure_large(self):
        """Benchmark structure analysis for large IDS."""
        async def run_analysis():
            return await self.fixture.server.analyze_ids_structure(
                ids_name="equilibrium",
                ctx=self.fixture.mock_context
            )

        return asyncio.run(run_analysis())

class BulkExportBenchmarks:
    """Benchmark suite for bulk export tools."""

    def setup(self):
        """Setup benchmark environment."""
        self.fixture = _benchmark_fixture
        asyncio.run(self._warmup())

    async def _warmup(self):
        """Warm up server components."""
        _ = self.fixture.server.document_store
        await self.fixture.server.export_ids_bulk(
            ids_list=["core_profiles"],
            ctx=self.fixture.mock_context
        )

    def time_export_ids_bulk_single(self):
        """Benchmark bulk export with single IDS."""
        async def run_export():
            return await self.fixture.server.export_ids_bulk(
                ids_list=["core_profiles"],
                ctx=self.fixture.mock_context
            )

        return asyncio.run(run_export())

    def time_export_ids_bulk_multiple(self):
        """Benchmark bulk export with multiple IDS."""
        async def run_export():
            return await self.fixture.server.export_ids_bulk(
                ids_list=["core_profiles", "equilibrium", "transport"],
                ctx=self.fixture.mock_context
            )

        return asyncio.run(run_export())

    def time_export_ids_bulk_with_relationships(self):
        """Benchmark bulk export with relationships."""
        async def run_export():
            return await self.fixture.server.export_ids_bulk(
                ids_list=["core_profiles", "equilibrium"],
                include_relationships=True,
                ctx=self.fixture.mock_context
            )

        return asyncio.run(run_export())

    def time_export_physics_domain(self):
        """Benchmark physics domain export."""
        async def run_export():
            return await self.fixture.server.export_physics_domain(
                domain="equilibrium",
                ctx=self.fixture.mock_context
            )

        return asyncio.run(run_export())

    def peakmem_export_ids_bulk_large(self):
        """Benchmark memory usage for large bulk export."""
        async def run_export():
            return await self.fixture.server.export_ids_bulk(
                ids_list=self.fixture.sample_ids_lists[-1],  # Largest list
                include_relationships=True,
                ctx=self.fixture.mock_context
            )

        return asyncio.run(run_export())

class RelationshipBenchmarks:
    """Benchmark suite for relationship exploration."""

    def setup(self):
        """Setup benchmark environment."""
        self.fixture = _benchmark_fixture
        asyncio.run(self._warmup())

    async def _warmup(self):
        """Warm up server components."""
        _ = self.fixture.server.graph_analyzer
        await self.fixture.server.explore_relationships(
            path="core_profiles/profiles_1d/electrons/temperature",
            ctx=self.fixture.mock_context
        )

    def time_explore_relationships_depth_1(self):
        """Benchmark relationship exploration with depth 1."""
        async def run_explore():
            return await self.fixture.server.explore_relationships(
                path="core_profiles/profiles_1d/electrons/temperature",
                max_depth=1,
                ctx=self.fixture.mock_context
            )

        return asyncio.run(run_explore())

    def time_explore_relationships_depth_2(self):
        """Benchmark relationship exploration with depth 2."""
        async def run_explore():
            return await self.fixture.server.explore_relationships(
                path="equilibrium/time_slice/profiles_1d/psi",
                max_depth=2,
                ctx=self.fixture.mock_context
            )

        return asyncio.run(run_explore())

    def time_explore_relationships_depth_3(self):
        """Benchmark relationship exploration with depth 3."""
        async def run_explore():
            return await self.fixture.server.explore_relationships(
                path="transport/model/diffusion_coefficient",
                max_depth=3,
                ctx=self.fixture.mock_context
            )

        return asyncio.run(run_explore())

# File: benchmarks/benchmark_runner.py
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

    def run_benchmarks(self, benchmark_names: List[str] = None) -> Dict[str, Any]:
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
            "stderr": result.stderr
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
            "html_dir": str(self.html_dir.absolute())
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
            "stderr": result.stderr
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
            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to read results: {e}"}

# File: scripts/run_performance_baseline.py
#!/usr/bin/env python
"""Script to establish performance baseline for current tools."""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any
from benchmarks.benchmark_runner import BenchmarkRunner

def main():
    """Establish performance baseline for current MCP tools."""
    print("🚀 Establishing Performance Baseline for IMAS MCP Tools")
    print("=" * 60)

    runner = BenchmarkRunner()

    # Core benchmarks to establish baseline
    core_benchmarks = [
        "SearchBenchmarks.time_search_imas_basic",
        "SearchBenchmarks.time_search_imas_with_ai",
        "SearchBenchmarks.time_search_imas_complex_query",
        "ExplainConceptBenchmarks.time_explain_concept_basic",
        "StructureAnalysisBenchmarks.time_analyze_ids_structure_small",
        "BulkExportBenchmarks.time_export_ids_bulk_single",
        "BulkExportBenchmarks.time_export_ids_bulk_multiple",
        "RelationshipBenchmarks.time_explore_relationships_depth_1"
    ]

    # Run baseline benchmarks
    print("\n📊 Running baseline benchmarks...")
    baseline_results = runner.run_benchmarks(core_benchmarks)

    if baseline_results["return_code"] != 0:
        print(f"❌ Benchmark run failed: {baseline_results['stderr']}")
        return 1

    print("✅ Baseline benchmarks completed successfully")

    # Generate HTML report
    print("\n📈 Generating HTML report...")
    html_results = runner.generate_html_report()

    if html_results["return_code"] == 0:
        print(f"✅ HTML report generated: {html_results['html_dir']}")
    else:
        print(f"❌ HTML report generation failed: {html_results['stderr']}")

    # Save baseline metadata
    baseline_metadata = {
        "timestamp": time.time(),
        "benchmarks_run": core_benchmarks,
        "execution_time": baseline_results["execution_time"],
        "status": "success" if baseline_results["return_code"] == 0 else "failed"
    }

    baseline_file = Path("benchmarks/baseline_metadata.json")
    with open(baseline_file, 'w') as f:
        json.dump(baseline_metadata, f, indent=2)

    print(f"\n📄 Baseline metadata saved to: {baseline_file}")
    print("\n🎉 Performance baseline established successfully!")

    return 0

if __name__ == "__main__":
    exit(main())
```

### 0.3 Test and Performance Integration

**Purpose**: Integrate testing and performance monitoring into development workflow

```python
# File: pyproject.toml (additions)
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--verbose",
    "--tb=short",
    "--strict-markers",
    "--disable-warnings",
    "--cov=imas_mcp",
    "--cov-report=html:htmlcov",
    "--cov-report=term-missing",
    "--cov-fail-under=85"
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "performance: marks tests as performance tests",
    "unit: marks tests as unit tests"
]
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning"
]

# File: Makefile (additions)
.PHONY: test-baseline performance-baseline test-current performance-current

test-baseline:
	@echo "Running baseline tests for current tools..."
	python -m pytest tests/test_current_tools.py -v --tb=short

test-current: test-baseline
	@echo "Running all current tests..."
	python -m pytest tests/ -v --tb=short -m "not slow"

performance-baseline:
	@echo "Establishing performance baseline..."
	python scripts/run_performance_baseline.py

performance-current:
	@echo "Running current performance benchmarks..."
	asv run --python=3.12

performance-compare:
	@echo "Comparing performance against baseline..."
	asv compare HEAD~1 HEAD

test-and-performance: test-current performance-current
	@echo "Running tests and performance monitoring..."

# File: .github/workflows/test-and-performance.yml
name: Test and Performance Monitoring

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-current:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.12]

    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0  # Needed for ASV

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install pytest pytest-cov pytest-asyncio
        pip install asv

    - name: Run unit tests
      run: |
        python -m pytest tests/test_current_tools.py -v --cov=imas_mcp

    - name: Run integration tests
      run: |
        python -m pytest tests/test_current_integration.py -v

    - name: Establish performance baseline
      run: |
        python scripts/run_performance_baseline.py

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

### 0.4 Performance Targets and Monitoring

**Purpose**: Define performance targets and monitoring strategy

```python
# File: benchmarks/performance_targets.py
"""Performance targets for IMAS MCP tools."""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PerformanceTarget:
    """Performance target for a specific benchmark."""
    name: str
    target_time: float  # seconds
    max_time: float     # seconds (failure threshold)
    memory_limit: int   # MB
    description: str

# Current tool performance targets (baseline)
CURRENT_PERFORMANCE_TARGETS = {
    "search_imas_basic": PerformanceTarget(
        name="search_imas_basic",
        target_time=2.0,
        max_time=5.0,
        memory_limit=500,
        description="Basic search without AI enhancement"
    ),
    "search_imas_with_ai": PerformanceTarget(
        name="search_imas_with_ai",
        target_time=3.0,
        max_time=8.0,
        memory_limit=600,
        description="Search with AI enhancement"
    ),
    "search_imas_complex": PerformanceTarget(
        name="search_imas_complex",
        target_time=4.0,
        max_time=10.0,
        memory_limit=700,
        description="Complex multi-term search"
    ),
    "explain_concept_basic": PerformanceTarget(
        name="explain_concept_basic",
        target_time=1.5,
        max_time=4.0,
        memory_limit=400,
        description="Basic concept explanation"
    ),
    "analyze_ids_structure": PerformanceTarget(
        name="analyze_ids_structure",
        target_time=2.5,
        max_time=6.0,
        memory_limit=600,
        description="IDS structure analysis"
    ),
    "export_ids_bulk_single": PerformanceTarget(
        name="export_ids_bulk_single",
        target_time=1.0,
        max_time=3.0,
        memory_limit=400,
        description="Single IDS bulk export"
    ),
    "export_ids_bulk_multiple": PerformanceTarget(
        name="export_ids_bulk_multiple",
        target_time=3.0,
        max_time=8.0,
        memory_limit=800,
        description="Multiple IDS bulk export"
    ),
    "explore_relationships": PerformanceTarget(
        name="explore_relationships",
        target_time=2.0,
        max_time=5.0,
        memory_limit=500,
        description="Relationship exploration"
    )
}

# Future performance targets (after optimization)
OPTIMIZED_PERFORMANCE_TARGETS = {
    "search_imas_fast": PerformanceTarget(
        name="search_imas_fast",
        target_time=0.5,
        max_time=1.0,
        memory_limit=300,
        description="Fast lexical search mode"
    ),
    "search_imas_adaptive": PerformanceTarget(
        name="search_imas_adaptive",
        target_time=1.0,
        max_time=2.0,
        memory_limit=400,
        description="Adaptive search mode"
    ),
    "search_imas_comprehensive": PerformanceTarget(
        name="search_imas_comprehensive",
        target_time=3.0,
        max_time=6.0,
        memory_limit=600,
        description="Comprehensive search mode"
    ),
    "export_bulk_raw": PerformanceTarget(
        name="export_bulk_raw",
        target_time=0.5,
        max_time=1.0,
        memory_limit=200,
        description="Raw format bulk export"
    ),
    "export_bulk_structured": PerformanceTarget(
        name="export_bulk_structured",
        target_time=1.0,
        max_time=2.0,
        memory_limit=300,
        description="Structured format bulk export"
    ),
    "export_bulk_enhanced": PerformanceTarget(
        name="export_bulk_enhanced",
        target_time=3.0,
        max_time=6.0,
        memory_limit=500,
        description="Enhanced format bulk export"
    )
}

def validate_performance_results(results: Dict[str, Any], targets: Dict[str, PerformanceTarget]) -> Dict[str, Any]:
    """Validate performance results against targets."""
    validation_results = {
        "passed": [],
        "failed": [],
        "warnings": []
    }

    for benchmark_name, target in targets.items():
        if benchmark_name in results:
            result_time = results[benchmark_name].get("time", float('inf'))

            if result_time <= target.target_time:
                validation_results["passed"].append({
                    "benchmark": benchmark_name,
                    "target": target.target_time,
                    "actual": result_time,
                    "status": "excellent"
                })
            elif result_time <= target.max_time:
                validation_results["warnings"].append({
                    "benchmark": benchmark_name,
                    "target": target.target_time,
                    "actual": result_time,
                    "max_allowed": target.max_time,
                    "status": "acceptable"
                })
            else:
                validation_results["failed"].append({
                    "benchmark": benchmark_name,
                    "target": target.target_time,
                    "actual": result_time,
                    "max_allowed": target.max_time,
                    "status": "failed"
                })

    return validation_results
```

## Phase 1: Core Tool Optimization (Weeks 3-5)

### 1.1 Enhanced search_imas Tool (Priority 1)

**Problem**: Most-called tool with performance and consistency issues
**Solution**: Implement tiered search strategy with caching

```python
# File: imas_mcp/server.py
async def search_imas(
    self,
    query: Union[str, List[str]],
    ids_name: Optional[str] = None,
    max_results: int = 10,
    search_mode: str = "adaptive",  # fast, adaptive, comprehensive
    explain_ranking: bool = False,
    include_suggestions: bool = True,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """
    Adaptive search with intelligent performance optimization.

    Search modes:
    - fast: Cached + lexical search, no AI enhancement
    - adaptive: Smart caching + selective semantic search
    - comprehensive: Full semantic + AI enhancement
    """

    # Step 1: Query analysis and caching
    cache_key = self._generate_cache_key(query, ids_name, max_results, search_mode)

    if search_mode in ["fast", "adaptive"]:
        cached_result = self._search_cache.get(cache_key)
        if cached_result:
            cached_result["cache_hit"] = True
            return cached_result

    # Step 2: Stratified search execution
    if search_mode == "fast":
        results = self._fast_lexical_search(query, ids_name, max_results)
    elif search_mode == "adaptive":
        results = await self._adaptive_search(query, ids_name, max_results, ctx)
    else:  # comprehensive
        results = await self._comprehensive_search(query, ids_name, max_results, ctx)

    # Step 3: Add tool recommendations
    if include_suggestions:
        results["suggested_tools"] = self._suggest_follow_up_tools(results)

    # Step 4: Cache results
    if search_mode in ["fast", "adaptive"]:
        self._search_cache.set(cache_key, results, ttl=3600)

    return results

def _fast_lexical_search(self, query: str, ids_name: Optional[str], max_results: int) -> Dict[str, Any]:
    """Fast lexical search using existing DocumentStore methods."""
    # Use existing search_by_keywords for simple queries
    if " " not in query and not any(op in query for op in ["AND", "OR", "NOT", '"']):
        results = self.document_store.search_by_keywords([query], max_results)
    else:
        # Use SQLite FTS5 for complex queries
        results = self.document_store.search_full_text(query, max_results=max_results)

    # Filter by IDS if specified
    if ids_name:
        results = [r for r in results if r.metadata.ids_name == ids_name]

    return {
        "results": [self._document_to_search_result(doc) for doc in results],
        "total_results": len(results),
        "search_strategy": "lexical_search",
        "performance_mode": "fast"
    }

def _suggest_follow_up_tools(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Suggest relevant tools based on current results."""
    suggestions = []

    # Physics concept suggestions
    if results.get("physics_matches"):
        suggestions.append({
            "tool": "explain_concept",
            "reason": "Get detailed physics explanations for found concepts",
            "sample_call": f"explain_concept('{results['physics_matches'][0]['concept']}')"
        })

    # Relationship analysis suggestions
    if len(results.get("results", [])) > 1:
        unique_ids = set(r["ids_name"] for r in results["results"])
        if len(unique_ids) > 1:
            suggestions.append({
                "tool": "explore_relationships",
                "reason": "Analyze relationships between found paths across IDS",
                "sample_call": f"explore_relationships('{results['results'][0]['path']}')"
            })

    # Bulk export suggestions
    if len(results.get("results", [])) > 5:
        unique_ids = list(set(r["ids_name"] for r in results["results"]))
        if len(unique_ids) > 1:
            suggestions.append({
                "tool": "export_ids_bulk",
                "reason": f"Export bulk data for {len(unique_ids)} IDS with relationships",
                "sample_call": f"export_ids_bulk({unique_ids[:3]})"
            })

    return suggestions
```

### 1.2 Selective AI Enhancement Strategy

**Problem**: AI enhancement on all tools adds unnecessary latency
**Solution**: Implement conditional AI enhancement

```python
# File: imas_mcp/search/ai_enhancer.py
AI_ENHANCEMENT_STRATEGY = {
    "search_imas": "conditional",      # Only for complex queries or when requested
    "explain_concept": "always",       # Core value-add
    "get_overview": "always",          # Benefits from synthesis
    "analyze_ids_structure": "conditional",  # Only for complex IDS
    "explore_relationships": "conditional",  # Only for deep analysis
    "explore_identifiers": "never",    # Structured data doesn't need AI
    "export_ids_bulk": "conditional",  # Only for enhanced format
    "export_physics_domain": "conditional"  # Only for workflow guidance
}

def ai_enhancer(expert_prompt: str, task_description: str,
                strategy: str = "always", **kwargs):
    """Enhanced AI decorator with conditional enhancement."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check if AI enhancement should be applied
            ctx = kwargs.get('ctx')

            if strategy == "never" or not ctx:
                # Remove AI prompt setup, execute without enhancement
                return await func(*args, **kwargs)

            elif strategy == "conditional":
                # Apply AI enhancement based on specific conditions
                should_enhance = _should_apply_ai_enhancement(func.__name__, args, kwargs)
                if not should_enhance:
                    return await func(*args, **kwargs)

            # Apply AI enhancement
            result = await func(*args, **kwargs)

            if ctx and result.get("ai_prompt"):
                try:
                    ai_response = await ctx.sample(result["ai_prompt"], **kwargs)
                    result["ai_enhancement"] = ai_response
                except Exception as e:
                    logger.warning(f"AI enhancement failed: {e}")
                    result["ai_enhancement"] = {"error": "AI enhancement unavailable"}

            return result
        return wrapper
    return decorator

def _should_apply_ai_enhancement(func_name: str, args: tuple, kwargs: dict) -> bool:
    """Determine if AI enhancement should be applied based on context."""
    if func_name == "search_imas":
        search_mode = kwargs.get("search_mode", "adaptive")
        return search_mode == "comprehensive"

    elif func_name == "export_ids_bulk":
        output_format = kwargs.get("output_format", "structured")
        return output_format == "enhanced"

    elif func_name == "analyze_ids_structure":
        # Apply AI for complex IDS (>100 paths)
        ids_name = args[1] if len(args) > 1 else kwargs.get("ids_name")
        return _is_complex_ids(ids_name)

    return True  # Default to enhancement for conditional tools
```

### 1.3 Multi-Format Bulk Export Tools

**Problem**: Bulk tools always apply AI enhancement, but researchers need raw data
**Solution**: Implement format-based processing

```python
# File: imas_mcp/server.py
async def export_ids_bulk(
    self,
    ids_list: List[str],
    format: str = "structured",  # raw, structured, enhanced
    include_relationships: bool = True,
    include_physics_context: bool = True,
    ctx: Optional[Context] = None,
) -> Dict[str, Any]:
    """
    Multi-format bulk export with optional AI enhancement.

    Formats:
    - raw: Pure data export, no AI enhancement, fastest
    - structured: Organized data with relationships, medium speed
    - enhanced: AI-enhanced with insights, requires ctx, slowest
    """

    # Validate format
    if format not in ["raw", "structured", "enhanced"]:
        return {"error": f"Invalid format: {format}. Use: raw, structured, enhanced"}

    # Raw format bypasses AI entirely
    if format == "raw":
        return self._export_raw_bulk(ids_list, include_relationships)

    # Structured format provides organized data
    if format == "structured":
        return self._export_structured_bulk(ids_list, include_relationships, include_physics_context)

    # Enhanced format uses AI only if context available
    if format == "enhanced":
        if not ctx:
            logger.warning("Enhanced format requested but no AI context available, falling back to structured")
            return self._export_structured_bulk(ids_list, include_relationships, include_physics_context)
        return await self._export_enhanced_bulk(ids_list, include_relationships, include_physics_context, ctx)

def _export_raw_bulk(self, ids_list: List[str], include_relationships: bool) -> Dict[str, Any]:
    """Raw data export for maximum performance."""
    # Implementation focused on speed, minimal processing
    pass

def _export_structured_bulk(self, ids_list: List[str], include_relationships: bool, include_physics_context: bool) -> Dict[str, Any]:
    """Structured data export with relationships but no AI."""
    # Implementation with organized data and relationships
    pass
```

## Phase 2: MCP Resources Implementation (Weeks 6-8)

### 2.1 Static JSON IDS Data Resources

**Purpose**: Serve pre-computed IDS data for efficient access

```python
# File: imas_mcp/resources/ids_catalog.py
from fastmcp import FastMCP
from typing import Dict, Any, Optional
import json
from pathlib import Path

class IDSCatalogResource:
    """MCP resource for serving static IDS catalog data."""

    def __init__(self, server: 'Server'):
        self.server = server
        self.catalog_path = Path("imas_mcp/resources/data/ids_catalog.json")
        self.structure_path = Path("imas_mcp/resources/data/ids_structures/")
        self._ensure_catalog_exists()

    def register_resources(self, mcp: FastMCP):
        """Register IDS catalog resources with MCP server."""

        @mcp.resource("ids://catalog")
        async def get_ids_catalog() -> str:
            """Get complete IDS catalog with metadata."""
            return self._load_catalog()

        @mcp.resource("ids://structure/{ids_name}")
        async def get_ids_structure(ids_name: str) -> str:
            """Get detailed structure for specific IDS."""
            return self._load_ids_structure(ids_name)

        @mcp.resource("ids://summary")
        async def get_ids_summary() -> str:
            """Get high-level summary of all IDS."""
            return self._generate_summary()

        @mcp.resource("ids://physics-domains")
        async def get_physics_domains() -> str:
            """Get physics domain mappings."""
            return self._load_physics_domains()

    def _load_catalog(self) -> str:
        """Load pre-computed IDS catalog."""
        if not self.catalog_path.exists():
            self._build_catalog()

        with open(self.catalog_path, 'r') as f:
            return f.read()

    def _build_catalog(self) -> None:
        """Build comprehensive IDS catalog."""
        available_ids = self.server.document_store.get_available_ids()

        catalog = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "total_ids": len(available_ids),
                "version": "1.0"
            },
            "ids": {}
        }

        for ids_name in available_ids:
            ids_docs = self.server.document_store.get_documents_by_ids(ids_name)

            # Calculate statistics
            physics_domains = set()
            units_distribution = {}
            identifier_count = 0

            for doc in ids_docs:
                if doc.metadata.physics_domain:
                    physics_domains.add(doc.metadata.physics_domain)
                if doc.metadata.units:
                    units_distribution[doc.metadata.units] = units_distribution.get(doc.metadata.units, 0) + 1
                if doc.raw_data.get("identifier_schema"):
                    identifier_count += 1

            catalog["ids"][ids_name] = {
                "path_count": len(ids_docs),
                "physics_domains": list(physics_domains),
                "units_distribution": dict(sorted(units_distribution.items(), key=lambda x: x[1], reverse=True)[:10]),
                "identifier_paths": identifier_count,
                "max_depth": max(len(doc.metadata.path_name.split("/")) for doc in ids_docs) if ids_docs else 0,
                "sample_paths": [doc.metadata.path_name for doc in ids_docs[:5]]
            }

        # Save catalog
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2)

# File: imas_mcp/resources/measurement_data.py
class MeasurementDataResource:
    """MCP resource for physics measurement data."""

    def register_resources(self, mcp: FastMCP):
        """Register measurement data resources."""

        @mcp.resource("measurement://units/{unit_type}")
        async def get_unit_paths(unit_type: str) -> str:
            """Get all paths using specific unit type."""
            return self._get_paths_by_unit(unit_type)

        @mcp.resource("measurement://coordinates/{coordinate}")
        async def get_coordinate_paths(coordinate: str) -> str:
            """Get all paths using specific coordinate."""
            return self._get_paths_by_coordinate(coordinate)

        @mcp.resource("measurement://physics-domain/{domain}")
        async def get_domain_paths(domain: str) -> str:
            """Get all paths in specific physics domain."""
            return self._get_paths_by_domain(domain)
```

### 2.2 Usage Examples Resource

**Purpose**: Provide code examples and workflows

```python
# File: imas_mcp/resources/examples.py
class ExamplesResource:
    """MCP resource for usage examples and workflows."""

    def register_resources(self, mcp: FastMCP):
        """Register example resources."""

        @mcp.resource("examples://search-patterns")
        async def get_search_patterns() -> str:
            """Get common search patterns and examples."""
            return json.dumps({
                "basic_search": {
                    "example": "search_imas('plasma temperature')",
                    "description": "Simple concept search",
                    "expected_results": "Temperature-related paths across IDS"
                },
                "field_specific": {
                    "example": "search_imas('units:eV AND documentation:electron')",
                    "description": "Field-specific search with boolean operators",
                    "expected_results": "Electron-related paths with eV units"
                },
                "bulk_search": {
                    "example": "search_imas(['plasma', 'temperature', 'density'])",
                    "description": "Multi-term search",
                    "expected_results": "Paths related to all terms"
                }
            })

        @mcp.resource("examples://workflows/{workflow_type}")
        async def get_workflow_examples(workflow_type: str) -> str:
            """Get workflow examples for specific analysis types."""
            workflows = {
                "equilibrium_analysis": {
                    "steps": [
                        "search_imas('equilibrium profiles')",
                        "analyze_ids_structure('equilibrium')",
                        "explore_relationships('equilibrium/time_slice/profiles_1d')",
                        "export_physics_domain('equilibrium')"
                    ],
                    "description": "Complete equilibrium analysis workflow"
                },
                "transport_study": {
                    "steps": [
                        "search_imas('transport coefficients')",
                        "export_ids_bulk(['core_profiles', 'transport', 'core_transport'])",
                        "explore_relationships('transport/model/diffusion')"
                    ],
                    "description": "Transport physics study workflow"
                }
            }
            return json.dumps(workflows.get(workflow_type, {}))
```

## Phase 3: MCP Prompts Implementation (Weeks 9-10)

### 3.1 Physics Analysis Prompts

**Purpose**: Provide specialized prompts for physics analysis

```python
# File: imas_mcp/prompts/physics_analysis.py
class PhysicsAnalysisPrompts:
    """MCP prompts for physics analysis tasks."""

    def register_prompts(self, mcp: FastMCP):
        """Register physics analysis prompts."""

        @mcp.prompt("physics-explain")
        async def physics_explain_prompt(
            concept: str,
            context: str = "",
            level: str = "intermediate"
        ) -> str:
            """Generate physics explanation prompt."""
            return f"""
            Explain the plasma physics concept '{concept}' at {level} level.

            Context: {context}

            Provide:
            1. Physical definition and significance
            2. Mathematical formulation (if applicable)
            3. Typical measurement methods
            4. Relationship to other plasma phenomena
            5. Practical applications in fusion research

            Format as clear, educational explanation suitable for plasma physicists.
            """

        @mcp.prompt("measurement-workflow")
        async def measurement_workflow_prompt(
            measurement_type: str,
            ids_involved: str,
            analysis_goal: str
        ) -> str:
            """Generate measurement workflow analysis prompt."""
            return f"""
            Design a measurement workflow for {measurement_type} analysis.

            Available IDS: {ids_involved}
            Analysis Goal: {analysis_goal}

            Provide:
            1. Data collection strategy
            2. Required coordinate systems
            3. Preprocessing steps
            4. Analysis methodology
            5. Validation approaches
            6. Expected outputs and interpretation

            Focus on practical implementation using IMAS data structures.
            """

        @mcp.prompt("cross-ids-analysis")
        async def cross_ids_analysis_prompt(
            ids_list: str,
            research_question: str
        ) -> str:
            """Generate cross-IDS analysis prompt."""
            return f"""
            Analyze relationships between IDS: {ids_list}
            Research Question: {research_question}

            Provide:
            1. Data coupling mechanisms
            2. Coordinate transformation requirements
            3. Consistency checks needed
            4. Integration challenges
            5. Recommended analysis sequence
            6. Quality assurance procedures

            Focus on physics-consistent analysis across multiple IDS.
            """

# File: imas_mcp/prompts/code_generation.py
class CodeGenerationPrompts:
    """MCP prompts for code generation tasks."""

    def register_prompts(self, mcp: FastMCP):
        """Register code generation prompts."""

        @mcp.prompt("imas-python-code")
        async def imas_python_code_prompt(
            task_description: str,
            paths: str,
            analysis_type: str = "basic"
        ) -> str:
            """Generate Python code for IMAS data analysis."""
            return f"""
            Generate Python code for IMAS data analysis.

            Task: {task_description}
            IMAS Paths: {paths}
            Analysis Type: {analysis_type}

            Requirements:
            1. Use IMAS Python API appropriately
            2. Include proper error handling
            3. Add physics-aware data validation
            4. Include plotting/visualization if relevant
            5. Add documentation and comments

            Generate clean, production-ready Python code.
            """

        @mcp.prompt("data-validation")
        async def data_validation_prompt(
            data_description: str,
            physics_constraints: str
        ) -> str:
            """Generate data validation approach."""
            return f"""
            Design data validation for: {data_description}
            Physics Constraints: {physics_constraints}

            Provide:
            1. Range checks and physical limits
            2. Consistency checks across related quantities
            3. Unit conversion validation
            4. Coordinate system checks
            5. Temporal consistency validation
            6. Error detection strategies

            Focus on physics-based validation rules.
            """
```

### 3.2 Workflow Automation Prompts

**Purpose**: Automate complex analysis workflows

```python
# File: imas_mcp/prompts/workflow_automation.py
class WorkflowAutomationPrompts:
    """MCP prompts for workflow automation."""

    def register_prompts(self, mcp: FastMCP):
        """Register workflow automation prompts."""

        @mcp.prompt("analysis-pipeline")
        async def analysis_pipeline_prompt(
            objective: str,
            available_data: str,
            constraints: str = ""
        ) -> str:
            """Generate complete analysis pipeline."""
            return f"""
            Design analysis pipeline for: {objective}
            Available Data: {available_data}
            Constraints: {constraints}

            Generate:
            1. Data extraction strategy
            2. Preprocessing pipeline
            3. Analysis methodology
            4. Quality control steps
            5. Visualization approach
            6. Results interpretation
            7. Error propagation

            Provide step-by-step implementation plan.
            """

        @mcp.prompt("experiment-design")
        async def experiment_design_prompt(
            research_question: str,
            hypothesis: str,
            available_diagnostics: str
        ) -> str:
            """Generate experiment design recommendations."""
            return f"""
            Design experiment for research question: {research_question}
            Hypothesis: {hypothesis}
            Available Diagnostics: {available_diagnostics}

            Provide:
            1. Measurement requirements
            2. Temporal resolution needs
            3. Spatial coverage requirements
            4. Calibration procedures
            5. Data acquisition strategy
            6. Analysis plan
            7. Expected outcomes

            Focus on physics-driven experimental design.
            """
```

## Phase 4: Performance Optimization (Weeks 11-12)

### 4.1 Caching Strategy

**Purpose**: Implement intelligent caching for common operations

```python
# File: imas_mcp/performance/cache_manager.py
from typing import Dict, Any, Optional
import hashlib
import pickle
from pathlib import Path
import time

class CacheManager:
    """Intelligent caching for MCP operations."""

    def __init__(self, cache_dir: Path = Path("cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache: Dict[str, Any] = {}
        self.cache_stats = {"hits": 0, "misses": 0}

    def get(self, key: str, ttl: int = 3600) -> Optional[Any]:
        """Get cached item with TTL check."""
        # Check memory cache first
        if key in self.memory_cache:
            item = self.memory_cache[key]
            if time.time() - item["timestamp"] < ttl:
                self.cache_stats["hits"] += 1
                return item["data"]
            else:
                del self.memory_cache[key]

        # Check disk cache
        cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    item = pickle.load(f)
                    if time.time() - item["timestamp"] < ttl:
                        self.memory_cache[key] = item
                        self.cache_stats["hits"] += 1
                        return item["data"]
                    else:
                        cache_file.unlink()
            except Exception:
                pass

        self.cache_stats["misses"] += 1
        return None

    def set(self, key: str, data: Any, ttl: int = 3600) -> None:
        """Set cached item."""
        item = {"data": data, "timestamp": time.time()}

        # Store in memory cache
        self.memory_cache[key] = item

        # Store in disk cache for persistence
        cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(item, f)
        except Exception:
            pass

    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total if total > 0 else 0

        return {
            "hit_rate": hit_rate,
            "total_requests": total,
            "memory_items": len(self.memory_cache),
            "disk_items": len(list(self.cache_dir.glob("*.pkl")))
        }
```

### 4.2 Performance Monitoring

**Purpose**: Track and optimize tool performance

```python
# File: imas_mcp/performance/monitor.py
from dataclasses import dataclass
from typing import Dict, List, Any
import time
import functools
from collections import defaultdict

@dataclass
class PerformanceMetrics:
    """Performance metrics for MCP tools."""
    tool_name: str
    execution_time: float
    cache_hit: bool
    result_count: int
    error_occurred: bool
    timestamp: float

class PerformanceMonitor:
    """Monitor and analyze MCP tool performance."""

    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.tool_stats = defaultdict(list)

    def monitor_tool(self, tool_name: str):
        """Decorator to monitor tool performance."""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                error_occurred = False
                result_count = 0
                cache_hit = False

                try:
                    result = await func(*args, **kwargs)

                    # Extract metrics from result
                    if isinstance(result, dict):
                        result_count = len(result.get("results", []))
                        cache_hit = result.get("cache_hit", False)

                    return result

                except Exception as e:
                    error_occurred = True
                    raise e

                finally:
                    execution_time = time.time() - start_time

                    # Record metrics
                    metric = PerformanceMetrics(
                        tool_name=tool_name,
                        execution_time=execution_time,
                        cache_hit=cache_hit,
                        result_count=result_count,
                        error_occurred=error_occurred,
                        timestamp=time.time()
                    )

                    self.metrics.append(metric)
                    self.tool_stats[tool_name].append(metric)

            return wrapper
        return decorator

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "overall_stats": {
                "total_requests": len(self.metrics),
                "average_response_time": sum(m.execution_time for m in self.metrics) / len(self.metrics) if self.metrics else 0,
                "error_rate": sum(1 for m in self.metrics if m.error_occurred) / len(self.metrics) if self.metrics else 0,
                "cache_hit_rate": sum(1 for m in self.metrics if m.cache_hit) / len(self.metrics) if self.metrics else 0
            },
            "tool_breakdown": {}
        }

        for tool_name, metrics in self.tool_stats.items():
            report["tool_breakdown"][tool_name] = {
                "request_count": len(metrics),
                "avg_response_time": sum(m.execution_time for m in metrics) / len(metrics),
                "error_rate": sum(1 for m in metrics if m.error_occurred) / len(metrics),
                "cache_hit_rate": sum(1 for m in metrics if m.cache_hit) / len(metrics),
                "avg_result_count": sum(m.result_count for m in metrics) / len(metrics)
            }

        return report
```

## Phase 5: Testing Strategy (Weeks 13-14)

### 5.1 Tool Testing with FastMCP 2

**Purpose**: Comprehensive testing of MCP tools

```python
# File: tests/test_mcp_tools.py
import pytest
from fastmcp import FastMCP
from fastmcp.testing import MockContext
from imas_mcp.server import Server
import json

class TestMCPTools:
    """Test suite for MCP tools using FastMCP 2."""

    @pytest.fixture
    def server(self):
        """Create test server instance."""
        return Server()

    @pytest.fixture
    def mock_context(self):
        """Create mock MCP context."""
        return MockContext()

    @pytest.mark.asyncio
    async def test_search_imas_fast_mode(self, server, mock_context):
        """Test search_imas in fast mode."""
        result = await server.search_imas(
            query="plasma temperature",
            search_mode="fast",
            max_results=5,
            ctx=mock_context
        )

        assert result["search_strategy"] == "lexical_search"
        assert result["performance_mode"] == "fast"
        assert len(result["results"]) <= 5
        assert "suggested_tools" in result

    @pytest.mark.asyncio
    async def test_search_imas_adaptive_mode(self, server, mock_context):
        """Test search_imas in adaptive mode."""
        result = await server.search_imas(
            query="equilibrium profiles",
            search_mode="adaptive",
            max_results=10,
            ctx=mock_context
        )

        assert result["search_strategy"] in ["lexical_search", "semantic_search"]
        assert len(result["results"]) <= 10
        assert "suggested_tools" in result

    @pytest.mark.asyncio
    async def test_search_imas_comprehensive_mode(self, server, mock_context):
        """Test search_imas in comprehensive mode."""
        result = await server.search_imas(
            query="transport coefficients",
            search_mode="comprehensive",
            max_results=15,
            ctx=mock_context
        )

        assert result["search_strategy"] == "semantic_search"
        assert len(result["results"]) <= 15
        assert "ai_enhancement" in result
        assert "suggested_tools" in result

    @pytest.mark.asyncio
    async def test_export_ids_bulk_raw_format(self, server):
        """Test bulk export in raw format."""
        result = await server.export_ids_bulk(
            ids_list=["core_profiles", "equilibrium"],
            format="raw",
            include_relationships=True
        )

        assert "ai_enhancement" not in result
        assert result["export_format"] == "raw"
        assert len(result["valid_ids"]) == 2

    @pytest.mark.asyncio
    async def test_export_ids_bulk_enhanced_format(self, server, mock_context):
        """Test bulk export in enhanced format."""
        result = await server.export_ids_bulk(
            ids_list=["core_profiles"],
            format="enhanced",
            include_relationships=True,
            ctx=mock_context
        )

        assert result["export_format"] == "enhanced"
        assert "ai_enhancement" in result

    @pytest.mark.asyncio
    async def test_tool_suggestions(self, server, mock_context):
        """Test tool suggestion functionality."""
        # Search for physics concepts
        search_result = await server.search_imas(
            query="plasma temperature density",
            search_mode="adaptive",
            ctx=mock_context
        )

        suggestions = search_result["suggested_tools"]
        assert len(suggestions) > 0

        # Check suggestion structure
        for suggestion in suggestions:
            assert "tool" in suggestion
            assert "reason" in suggestion
            assert "sample_call" in suggestion

    @pytest.mark.asyncio
    async def test_conditional_ai_enhancement(self, server, mock_context):
        """Test conditional AI enhancement strategy."""
        # Test tool that should not have AI enhancement
        result = await server.explore_identifiers(
            query="temperature",
            scope="summary",
            ctx=mock_context
        )

        assert "ai_enhancement" not in result

        # Test tool that should have AI enhancement
        result = await server.explain_concept(
            concept="plasma temperature",
            ctx=mock_context
        )

        assert "ai_enhancement" in result

# File: tests/test_mcp_resources.py
import pytest
from fastmcp.testing import MockResourceClient
from imas_mcp.resources.ids_catalog import IDSCatalogResource
import json

class TestMCPResources:
    """Test suite for MCP resources."""

    @pytest.fixture
    def resource_client(self):
        """Create mock resource client."""
        return MockResourceClient()

    @pytest.mark.asyncio
    async def test_ids_catalog_resource(self, resource_client):
        """Test IDS catalog resource."""
        response = await resource_client.get("ids://catalog")

        catalog = json.loads(response)
        assert "metadata" in catalog
        assert "ids" in catalog
        assert catalog["metadata"]["total_ids"] > 0

    @pytest.mark.asyncio
    async def test_ids_structure_resource(self, resource_client):
        """Test IDS structure resource."""
        response = await resource_client.get("ids://structure/core_profiles")

        structure = json.loads(response)
        assert "path_count" in structure
        assert "physics_domains" in structure
        assert "sample_paths" in structure

    @pytest.mark.asyncio
    async def test_physics_domains_resource(self, resource_client):
        """Test physics domains resource."""
        response = await resource_client.get("ids://physics-domains")

        domains = json.loads(response)
        assert isinstance(domains, dict)
        assert len(domains) > 0

# File: tests/test_mcp_prompts.py
import pytest
from fastmcp.testing import MockPromptClient
from imas_mcp.prompts.physics_analysis import PhysicsAnalysisPrompts

class TestMCPPrompts:
    """Test suite for MCP prompts."""

    @pytest.fixture
    def prompt_client(self):
        """Create mock prompt client."""
        return MockPromptClient()

    @pytest.mark.asyncio
    async def test_physics_explain_prompt(self, prompt_client):
        """Test physics explanation prompt."""
        response = await prompt_client.get_prompt(
            "physics-explain",
            concept="plasma temperature",
            level="intermediate"
        )

        assert "plasma temperature" in response
        assert "intermediate" in response
        assert "Physical definition" in response

    @pytest.mark.asyncio
    async def test_measurement_workflow_prompt(self, prompt_client):
        """Test measurement workflow prompt."""
        response = await prompt_client.get_prompt(
            "measurement-workflow",
            measurement_type="temperature profile",
            ids_involved="core_profiles, equilibrium",
            analysis_goal="transport analysis"
        )

        assert "temperature profile" in response
        assert "core_profiles" in response
        assert "transport analysis" in response
```

### 5.2 Performance Testing

**Purpose**: Ensure tools meet performance requirements

```python
# File: tests/test_performance.py
import pytest
import time
from imas_mcp.server import Server
from imas_mcp.performance.monitor import PerformanceMonitor
from fastmcp.testing import MockContext

class TestPerformance:
    """Performance testing suite."""

    @pytest.fixture
    def server(self):
        """Create server with performance monitoring."""
        server = Server()
        server.performance_monitor = PerformanceMonitor()
        return server

    @pytest.mark.asyncio
    async def test_search_performance_fast_mode(self, server):
        """Test search performance in fast mode."""
        start_time = time.time()

        result = await server.search_imas(
            query="plasma temperature",
            search_mode="fast",
            max_results=10
        )

        execution_time = time.time() - start_time

        # Fast mode should complete in under 0.5 seconds
        assert execution_time < 0.5
        assert result["performance_mode"] == "fast"

    @pytest.mark.asyncio
    async def test_bulk_export_performance(self, server):
        """Test bulk export performance."""
        start_time = time.time()

        result = await server.export_ids_bulk(
            ids_list=["core_profiles", "equilibrium"],
            format="raw",
            include_relationships=False
        )

        execution_time = time.time() - start_time

        # Raw export should complete in under 1 second
        assert execution_time < 1.0
        assert result["export_format"] == "raw"

    @pytest.mark.asyncio
    async def test_cache_effectiveness(self, server):
        """Test caching effectiveness."""
        query = "plasma temperature"

        # First call - should miss cache
        start_time = time.time()
        result1 = await server.search_imas(query=query, search_mode="adaptive")
        first_time = time.time() - start_time

        # Second call - should hit cache
        start_time = time.time()
        result2 = await server.search_imas(query=query, search_mode="adaptive")
        second_time = time.time() - start_time

        # Cached call should be significantly faster
        assert second_time < first_time * 0.5
        assert result2.get("cache_hit", False)

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, server):
        """Test performance monitoring functionality."""
        # Execute several operations
        await server.search_imas("plasma", search_mode="fast")
        await server.search_imas("temperature", search_mode="adaptive")
        await server.explain_concept("plasma temperature")

        # Check performance report
        report = server.performance_monitor.get_performance_report()

        assert report["overall_stats"]["total_requests"] >= 3
        assert "tool_breakdown" in report
        assert "search_imas" in report["tool_breakdown"]
```

## Phase 6: Documentation and Deployment (Weeks 15-16)

### 6.1 Comprehensive Documentation

**Purpose**: Document all tools, resources, and prompts

```python
# File: docs/mcp_tools_guide.md
# IMAS MCP Tools Guide

## Tool Performance Modes

### search_imas
- **fast**: Lexical search, <0.5s response time, no AI enhancement
- **adaptive**: Intelligent search selection, <1s response time, conditional AI
- **comprehensive**: Full semantic search, <3s response time, AI enhancement

### export_ids_bulk
- **raw**: Pure data export, fastest, no AI enhancement
- **structured**: Organized with relationships, medium speed
- **enhanced**: AI-enhanced insights, slowest, requires AI context

## Resource Usage

### IDS Catalog Resources
- `ids://catalog`: Complete IDS catalog with metadata
- `ids://structure/{ids_name}`: Detailed structure for specific IDS
- `ids://physics-domains`: Physics domain mappings

### Usage Examples Resources
- `examples://search-patterns`: Common search patterns
- `examples://workflows/{type}`: Analysis workflow examples

## Prompt Templates

### Physics Analysis
- `physics-explain`: Generate physics explanations
- `measurement-workflow`: Create measurement workflows
- `cross-ids-analysis`: Analyze cross-IDS relationships

### Code Generation
- `imas-python-code`: Generate Python analysis code
- `data-validation`: Create validation approaches
```

### 6.2 Performance Benchmarks

**Purpose**: Establish performance baselines

```python
# File: benchmarks/performance_benchmarks.py
import asyncio
import time
from typing import Dict, List
from imas_mcp.server import Server

class PerformanceBenchmark:
    """Performance benchmarking suite."""

    def __init__(self):
        self.server = Server()
        self.results: Dict[str, List[float]] = {}

    async def benchmark_search_modes(self, queries: List[str], iterations: int = 10):
        """Benchmark different search modes."""
        modes = ["fast", "adaptive", "comprehensive"]

        for mode in modes:
            times = []

            for _ in range(iterations):
                for query in queries:
                    start_time = time.time()
                    await self.server.search_imas(query=query, search_mode=mode)
                    times.append(time.time() - start_time)

            self.results[f"search_{mode}"] = times

    async def benchmark_export_formats(self, ids_lists: List[List[str]], iterations: int = 5):
        """Benchmark different export formats."""
        formats = ["raw", "structured", "enhanced"]

        for format_type in formats:
            times = []

            for _ in range(iterations):
                for ids_list in ids_lists:
                    start_time = time.time()
                    await self.server.export_ids_bulk(
                        ids_list=ids_list,
                        format=format_type
                    )
                    times.append(time.time() - start_time)

            self.results[f"export_{format_type}"] = times

    def generate_report(self) -> Dict[str, Dict[str, float]]:
        """Generate performance benchmark report."""
        report = {}

        for test_name, times in self.results.items():
            report[test_name] = {
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "p95_time": sorted(times)[int(len(times) * 0.95)],
                "total_calls": len(times)
            }

        return report

# Performance targets
PERFORMANCE_TARGETS = {
    "search_fast": 0.5,      # seconds
    "search_adaptive": 1.0,   # seconds
    "search_comprehensive": 3.0,  # seconds
    "export_raw": 1.0,       # seconds
    "export_structured": 2.0,  # seconds
    "export_enhanced": 5.0    # seconds
}
```

## Implementation Timeline

| Phase       | Duration    | Key Deliverables                                                                    | Dependencies      |
| ----------- | ----------- | ----------------------------------------------------------------------------------- | ----------------- |
| **Phase 0** | Weeks 1-2   | Comprehensive pytest unit tests, ASV performance monitoring, baseline establishment | Existing codebase |
| **Phase 1** | Weeks 3-5   | Enhanced search_imas, selective AI enhancement, multi-format exports                | Phase 0 complete  |
| **Phase 2** | Weeks 6-8   | MCP resources for IDS data, examples, measurements                                  | Phase 1 complete  |
| **Phase 3** | Weeks 9-10  | MCP prompts for physics analysis, code generation, workflows                        | Phase 2 complete  |
| **Phase 4** | Weeks 11-12 | Caching system, performance monitoring, optimization                                | Phase 3 complete  |
| **Phase 5** | Weeks 13-14 | Comprehensive testing suite, performance validation                                 | Phase 4 complete  |
| **Phase 6** | Weeks 15-16 | Documentation, benchmarking, deployment preparation                                 | Phase 5 complete  |

## Success Metrics

### Performance Metrics

- **Search Fast Mode**: <0.5s response time, >90% cache hit rate
- **Search Adaptive Mode**: <1s response time, intelligent mode selection
- **Bulk Export Raw**: <1s for 2 IDS, <2s for 5 IDS
- **Overall Error Rate**: <1% across all tools

### Quality Metrics

- **Test Coverage**: >95% code coverage
- **Tool Suggestion Accuracy**: >80% relevant suggestions
- **Cache Effectiveness**: >70% hit rate for common queries
- **AI Enhancement Value**: Measurable improvement in result quality

### Usage Metrics

- **Tool Adoption**: Usage distribution across all 8 tools
- **Resource Utilization**: Regular access to MCP resources
- **Prompt Usage**: Active use of MCP prompts for workflows

## Risk Mitigation

### Technical Risks

- **Performance Degradation**: Comprehensive monitoring and benchmarking
- **Cache Invalidation**: Intelligent cache management with TTL
- **AI Service Availability**: Graceful degradation when AI unavailable

### Operational Risks

- **Backward Compatibility**: Maintain existing API contracts
- **Resource Consumption**: Monitor memory and CPU usage
- **Scalability**: Design for concurrent user access

This development plan provides a comprehensive roadmap for implementing the critical improvements identified in the tool analysis while expanding capabilities with

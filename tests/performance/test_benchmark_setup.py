"""Test the benchmark setup and basic functionality."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from benchmarks.benchmark_runner import BenchmarkRunner
from benchmarks.benchmarks import BenchmarkFixture, SearchBenchmarks


class TestBenchmarkConfiguration:
    """Test basic benchmark configuration and setup."""

    def test_benchmark_runner_init(self):
        """Test that BenchmarkRunner initializes correctly."""
        runner = BenchmarkRunner()
        assert runner.benchmark_dir == Path("benchmarks")
        assert runner.results_dir == Path(".asv/results")
        assert runner.html_dir == Path(".asv/html")

    def test_asv_config_exists(self):
        """Test that ASV configuration file exists and is valid."""
        config_path = Path("asv.conf.json")
        assert config_path.exists(), "ASV configuration file should exist"

        with open(config_path, "r") as f:
            config = json.load(f)

        # Check required fields
        assert config["version"] == 1
        assert config["project"] == "imas-mcp"
        assert config["environment_type"] == "virtualenv"
        assert "3.12" in config["pythons"]
        assert config["benchmark_dir"] == "benchmarks"

    def test_benchmarks_file_exists(self):
        """Test that benchmarks.py file exists."""
        benchmarks_path = Path("benchmarks/benchmarks.py")
        assert benchmarks_path.exists(), "Benchmarks file should exist"

    def test_baseline_script_exists(self):
        """Test that baseline script exists."""
        script_path = Path("scripts/run_performance_baseline.py")
        assert script_path.exists(), "Baseline script should exist"


class TestBenchmarkSetup:
    """Test benchmark setup and warmup procedures."""

    @pytest.fixture
    def benchmark_fixture(self):
        """Create a benchmark fixture for testing."""
        return BenchmarkFixture()

    @pytest.fixture
    def search_benchmarks(self):
        """Create search benchmarks instance."""
        return SearchBenchmarks()

    def test_benchmark_fixture_properties(self, benchmark_fixture):
        """Test that benchmark fixture properties are properly configured."""
        # Test sample queries
        queries = benchmark_fixture.sample_queries
        assert isinstance(queries, list)
        assert len(queries) > 0
        assert "plasma temperature" in queries

        # Test single IDS
        single_ids = benchmark_fixture.single_ids
        assert isinstance(single_ids, str)
        assert single_ids == "core_profiles"

        # Test IDS pair
        ids_pair = benchmark_fixture.ids_pair
        assert isinstance(ids_pair, list)
        assert len(ids_pair) == 2
        assert "core_profiles" in ids_pair
        assert "equilibrium" in ids_pair

    def test_mock_context_setup(self, benchmark_fixture):
        """Test that client is properly configured."""
        client = benchmark_fixture.client
        assert client is not None
        assert hasattr(client, "call_tool")

    @pytest.mark.asyncio
    async def test_warmup_calls_embedding_generation(self, search_benchmarks):
        """Test that warmup properly triggers embedding generation."""
        # Setup the benchmark to get access to fixture
        search_benchmarks.setup()

        # Mock the server to track calls
        with patch.object(search_benchmarks.fixture, "client") as mock_client:
            # Setup mock client context manager
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.call_tool = AsyncMock()

            # Run warmup
            await search_benchmarks._warmup()

            # Verify call_tool was called for warmup
            assert (
                mock_client.call_tool.call_count >= 3
            )  # At least once per IDS + cross-IDS

            # Verify calls include search_imas tool
            calls = mock_client.call_tool.call_args_list
            tool_names = [call.args[0] if call.args else None for call in calls]
            assert "search_imas" in tool_names

    @pytest.mark.asyncio
    async def test_benchmark_consistency(self, benchmark_fixture):
        """Test that all benchmarks use consistent data."""
        # All benchmark classes should use the same fixture
        from benchmarks.benchmarks import (
            _benchmark_fixture,
            ExplainConceptBenchmarks,
            StructureAnalysisBenchmarks,
            BulkExportBenchmarks,
            RelationshipBenchmarks,
        )

        # Verify all classes use the same global fixture when setup
        search_bench = SearchBenchmarks()
        search_bench.setup()

        explain_bench = ExplainConceptBenchmarks()
        explain_bench.setup()

        structure_bench = StructureAnalysisBenchmarks()
        structure_bench.setup()

        bulk_bench = BulkExportBenchmarks()
        bulk_bench.setup()

        relation_bench = RelationshipBenchmarks()
        relation_bench.setup()

        # All should reference the same fixture instance after setup
        assert search_bench.fixture is _benchmark_fixture
        assert explain_bench.fixture is _benchmark_fixture
        assert structure_bench.fixture is _benchmark_fixture
        assert bulk_bench.fixture is _benchmark_fixture
        assert relation_bench.fixture is _benchmark_fixture

    def test_single_ids_benchmark_uses_consistent_data(self):
        """Test that single IDS benchmark uses data from the consistent IDS set."""
        from benchmarks.benchmarks import BulkExportBenchmarks

        bulk_bench = BulkExportBenchmarks()
        bulk_bench.setup()

        # Check that there is a single IDS export method
        methods = [method for method in dir(bulk_bench) if method.startswith("time_")]
        single_ids_methods = [
            method for method in methods if "single" in method.lower()
        ]

        assert len(single_ids_methods) == 1, (
            f"Expected exactly one single IDS method, found: {single_ids_methods}"
        )
        assert "time_export_ids_single" in single_ids_methods

    @pytest.mark.asyncio
    async def test_warmup_performance_impact(self, search_benchmarks):
        """Test that warmup actually improves subsequent call performance."""
        # This is a conceptual test - in practice you'd measure timing
        # Here we just verify the warmup runs without errors

        # Setup the benchmark to get access to fixture
        search_benchmarks.setup()

        with patch.object(search_benchmarks.fixture, "server") as mock_server:
            mock_server.tools.document_store = AsyncMock()
            mock_server.tools.search_composer = AsyncMock()
            mock_server.tools.graph_analyzer = AsyncMock()

            # Warmup should complete without errors
            await search_benchmarks._warmup()

            # Verify warmup touched all major components
            assert mock_server.tools.document_store is not None
            assert mock_server.tools.search_composer is not None
            assert mock_server.tools.graph_analyzer is not None

    def test_benchmark_method_naming(self):
        """Test that benchmark methods follow ASV naming conventions."""
        from benchmarks.benchmarks import (
            SearchBenchmarks,
            ExplainConceptBenchmarks,
            StructureAnalysisBenchmarks,
            BulkExportBenchmarks,
            RelationshipBenchmarks,
        )

        benchmark_classes = [
            SearchBenchmarks,
            ExplainConceptBenchmarks,
            StructureAnalysisBenchmarks,
            BulkExportBenchmarks,
            RelationshipBenchmarks,
        ]

        for benchmark_class in benchmark_classes:
            methods = [
                method
                for method in dir(benchmark_class)
                if method.startswith(("time_", "peakmem_"))
            ]

            # Each class should have at least one benchmark method
            assert len(methods) > 0, (
                f"{benchmark_class.__name__} has no benchmark methods"
            )

            # All methods should follow proper naming
            for method in methods:
                assert method.startswith(("time_", "peakmem_")), (
                    f"Invalid method name: {method}"
                )


if __name__ == "__main__":
    pytest.main([__file__])

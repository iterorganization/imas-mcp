import asyncio
from functools import cached_property

from fastmcp import Client

# Standard test IDS set for consistency across all tests and benchmarks
STANDARD_TEST_IDS_SET = {"equilibrium", "core_profiles"}


class BenchmarkFixture:
    """Composition-based benchmark fixture for performance testing."""

    @cached_property
    def server(self):
        """Lazy-loaded server instance."""
        from imas_codex.server import Server

        return Server(ids_set=STANDARD_TEST_IDS_SET)

    @cached_property
    def client(self):
        """Lazy-loaded FastMCP client."""
        return Client(self.server.mcp)

    @cached_property
    def sample_queries(self) -> list[str]:
        """Sample queries for benchmarking."""
        return [
            "plasma temperature",
            "magnetic field",
            "electron density",
            "transport coefficients",
            "equilibrium",
        ]

    @cached_property
    def single_ids(self) -> str:
        """Single IDS for benchmarking - from the consistent IDS set."""
        return "core_profiles"

    @cached_property
    def ids_pair(self) -> list[str]:
        """IDS pair for benchmarking - from the consistent IDS set."""
        return ["core_profiles", "equilibrium"]


_benchmark_fixture = BenchmarkFixture()


class SearchBenchmarks:
    """Benchmark suite for search_imas_paths tool."""

    def setup(self):
        """Setup benchmark environment."""
        self.fixture = _benchmark_fixture
        asyncio.run(self._warmup())

    async def _warmup(self):
        """Warm up server components to avoid cold start penalties."""
        _ = self.fixture.server.tools.document_store

        async with self.fixture.client:
            for ids_name in self.fixture.ids_pair:
                await self.fixture.client.call_tool(
                    "search_imas_paths",
                    {
                        "query": "temperature",
                        "ids_filter": [ids_name],
                        "max_results": 1,
                    },
                )

            await self.fixture.client.call_tool(
                "search_imas_paths", {"query": "plasma", "max_results": 1}
            )

    def time_search_imas_paths_basic(self):
        """Benchmark basic search performance."""

        async def run_search():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "search_imas_paths",
                    {"query": self.fixture.sample_queries[0], "max_results": 5},
                )

        return asyncio.run(run_search())

    def time_search_imas_paths_single_ids(self):
        """Benchmark search with single IDS filtering."""

        async def run_search():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "search_imas_paths",
                    {
                        "query": self.fixture.sample_queries[1],
                        "ids_filter": [self.fixture.single_ids],
                        "max_results": 10,
                    },
                )

        return asyncio.run(run_search())

    def time_search_imas_paths_complex_query(self):
        """Benchmark complex query performance."""

        async def run_search():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "search_imas_paths",
                    {
                        "query": "plasma temperature AND magnetic field",
                        "max_results": 15,
                    },
                )

        return asyncio.run(run_search())

    def peakmem_search_imas_paths_basic(self):
        """Benchmark memory usage for basic search."""

        async def run_search():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "search_imas_paths",
                    {"query": self.fixture.sample_queries[0], "max_results": 5},
                )

        return asyncio.run(run_search())


class ClusterSearchBenchmarks:
    """Benchmark suite for search_imas_clusters tool."""

    def setup(self):
        """Setup benchmark environment."""
        self.fixture = _benchmark_fixture
        asyncio.run(self._warmup())

    async def _warmup(self):
        """Warm up server components."""
        _ = self.fixture.server.tools.document_store

        async with self.fixture.client:
            await self.fixture.client.call_tool(
                "search_imas_paths",
                {"query": "relationships", "max_results": 1},
            )
            for ids_name in self.fixture.ids_pair:
                await self.fixture.client.call_tool(
                    "search_imas_paths",
                    {
                        "query": "temperature",
                        "ids_filter": [ids_name],
                        "max_results": 1,
                    },
                )

    def time_search_clusters_depth_1(self):
        """Benchmark cluster search with depth 1."""

        async def run_explore():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "search_imas_clusters",
                    {
                        "path": "core_profiles/profiles_1d/electrons/temperature",
                        "max_depth": 1,
                    },
                )

        return asyncio.run(run_explore())

    def time_search_clusters_depth_2(self):
        """Benchmark cluster search with depth 2."""

        async def run_explore():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "search_imas_clusters",
                    {
                        "path": "core_profiles/profiles_1d/electrons/density",
                        "max_depth": 2,
                    },
                )

        return asyncio.run(run_explore())

    def time_search_clusters_depth_3(self):
        """Benchmark cluster search with depth 3."""

        async def run_explore():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "search_imas_clusters",
                    {"path": "equilibrium/time_slice/profiles_2d/psi", "max_depth": 3},
                )

        return asyncio.run(run_explore())

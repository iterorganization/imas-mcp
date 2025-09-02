import asyncio
from functools import cached_property

from fastmcp import Client

# Standard test IDS set for consistency across all tests and benchmarks
STANDARD_TEST_IDS_SET = {"equilibrium", "core_profiles"}


# Import using composition to avoid costly imports in benchmark setup
class BenchmarkFixture:
    """Composition-based benchmark fixture for performance testing."""

    @cached_property
    def server(self):
        """Lazy-loaded server instance."""
        from imas_mcp.server import Server

        # Use consistent IDS set to avoid multiple embeddings
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
        _ = self.fixture.server.tools.document_store

        # Ensure embeddings are generated for our sample IDS during warmup
        # This prevents embedding generation from being included in benchmark timing
        async with self.fixture.client:
            for ids_name in self.fixture.ids_pair:
                # Trigger embedding generation by doing a semantic search for each IDS
                await self.fixture.client.call_tool(
                    "search_imas",
                    {
                        "query": "temperature",  # Simple query to trigger embedding load
                        "ids_filter": [ids_name],
                        "max_results": 1,
                    },
                )

            # Perform a cross-IDS search to ensure all embeddings are loaded
            await self.fixture.client.call_tool(
                "search_imas", {"query": "plasma", "max_results": 1}
            )

    def time_search_imas_basic(self):
        """Benchmark basic search performance."""

        async def run_search():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "search_imas",
                    {"query": self.fixture.sample_queries[0], "max_results": 5},
                )

        return asyncio.run(run_search())

    def time_search_imas_single_ids(self):
        """Benchmark search with single IDS filtering."""

        async def run_search():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "search_imas",
                    {
                        "query": self.fixture.sample_queries[1],
                        "ids_filter": [self.fixture.single_ids],
                        "max_results": 10,
                    },
                )

        return asyncio.run(run_search())

    def time_search_imas_complex_query(self):
        """Benchmark complex query performance."""

        async def run_search():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "search_imas",
                    {
                        "query": "plasma temperature AND magnetic field",
                        "max_results": 15,
                    },
                )

        return asyncio.run(run_search())

    def peakmem_search_imas_basic(self):
        """Benchmark memory usage for basic search."""

        async def run_search():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "search_imas",
                    {"query": self.fixture.sample_queries[0], "max_results": 5},
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
        _ = self.fixture.server.tools.document_store

        # Warm up with a simple concept explanation to load any models/caches
        async with self.fixture.client:
            await self.fixture.client.call_tool(
                "explain_concept", {"concept": "plasma", "detail_level": "basic"}
            )

    def time_explain_concept_basic(self):
        """Benchmark basic concept explanation."""

        async def run_explain():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "explain_concept",
                    {"concept": "plasma temperature", "detail_level": "basic"},
                )

        return asyncio.run(run_explain())

    def time_explain_concept_advanced(self):
        """Benchmark advanced concept explanation."""

        async def run_explain():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "explain_concept",
                    {"concept": "transport coefficients", "detail_level": "advanced"},
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
        _ = self.fixture.server.tools.document_store

        # Warm up with search_imas to initialize embeddings and caches
        async with self.fixture.client:
            for ids_name in self.fixture.ids_pair:
                await self.fixture.client.call_tool(
                    "search_imas",
                    {
                        "query": "structure",  # Simple query to trigger initialization
                        "ids_filter": [ids_name],
                        "max_results": 1,
                    },
                )

    def time_analyze_ids_structure_single(self):
        """Benchmark structure analysis for single IDS."""

        async def run_analysis():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "analyze_ids_structure", {"ids_name": self.fixture.single_ids}
                )

        return asyncio.run(run_analysis())

    def time_analyze_ids_structure_equilibrium(self):
        """Benchmark structure analysis for equilibrium IDS."""

        async def run_analysis():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "analyze_ids_structure", {"ids_name": "equilibrium"}
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
        _ = self.fixture.server.tools.document_store

        async with self.fixture.client:
            await self.fixture.client.call_tool(
                "export_ids",
                {
                    "ids_list": self.fixture.ids_pair,
                    "include_relationships": False,
                    "include_physics": False,
                },
            )

    def time_export_ids_single(self):
        """Benchmark bulk export with single IDS."""

        async def run_export():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "export_ids",
                    {
                        "ids_list": [self.fixture.single_ids],
                        "include_relationships": False,
                        "include_physics": False,
                    },
                )

        return asyncio.run(run_export())

    def time_export_ids_multiple(self):
        """Benchmark bulk export with multiple IDS."""

        async def run_export():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "export_ids",
                    {"ids_list": self.fixture.ids_pair, "include_relationships": True},
                )

        return asyncio.run(run_export())

    def time_export_ids_with_relationships(self):
        """Benchmark bulk export with relationships."""

        async def run_export():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "export_ids",
                    {
                        "ids_list": self.fixture.ids_pair,
                        "include_relationships": True,
                        "include_physics": True,
                    },
                )

        return asyncio.run(run_export())

    def time_export_physics_domain(self):
        """Benchmark physics domain export."""

        async def run_export():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "export_physics_domain",
                    {"domain": "core_profiles", "include_cross_domain": True},
                )

        return asyncio.run(run_export())

    def peakmem_export_ids_large(self):
        """Benchmark memory usage for large bulk export."""

        async def run_export():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "export_ids",
                    {
                        "ids_list": self.fixture.ids_pair,
                        "include_relationships": True,
                        "include_physics": True,
                    },
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
        _ = self.fixture.server.tools.document_store

        # Warm up with search_imas to initialize embeddings and caches
        async with self.fixture.client:
            await self.fixture.client.call_tool(
                "search_imas",
                {
                    "query": "relationships",  # Simple query to trigger initialization
                    "max_results": 1,
                },
            )
            # Also warm up each IDS individually
            for ids_name in self.fixture.ids_pair:
                await self.fixture.client.call_tool(
                    "search_imas",
                    {
                        "query": "temperature",
                        "ids_filter": [ids_name],
                        "max_results": 1,
                    },
                )

    def time_explore_relationships_depth_1(self):
        """Benchmark relationship exploration with depth 1."""

        async def run_explore():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "explore_relationships",
                    {
                        "path": "core_profiles/profiles_1d/electrons/temperature",
                        "max_depth": 1,
                    },
                )

        return asyncio.run(run_explore())

    def time_explore_relationships_depth_2(self):
        """Benchmark relationship exploration with depth 2."""

        async def run_explore():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "explore_relationships",
                    {
                        "path": "core_profiles/profiles_1d/electrons/density",
                        "max_depth": 2,
                    },
                )

        return asyncio.run(run_explore())

    def time_explore_relationships_depth_3(self):
        """Benchmark relationship exploration with depth 3."""

        async def run_explore():
            async with self.fixture.client:
                return await self.fixture.client.call_tool(
                    "explore_relationships",
                    {"path": "equilibrium/time_slice/profiles_2d/psi", "max_depth": 3},
                )

        return asyncio.run(run_explore())

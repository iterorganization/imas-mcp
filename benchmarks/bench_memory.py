"""Memory profiling benchmarks.

Tracks peak memory for key operations to detect memory leaks
or regressions across releases.
"""

from __future__ import annotations

import asyncio

from benchmarks.conftest_bench import IDS_NAMES, SEARCH_QUERIES, _fixture


class MemoryBenchmarks:
    """Peak memory benchmarks for server and tool operations."""

    timeout = 180

    def setup(self):
        """Warmup: ensure server is initialised."""

        async def _warmup():
            async with _fixture.client:
                await _fixture.client.call_tool(
                    "search_imas", {"query": "warmup", "k": 1}
                )

        asyncio.run(_warmup())

    def peakmem_server_idle(self):
        """Base memory footprint of AgentsServer at rest."""
        from imas_codex.llm.server import AgentsServer

        AgentsServer()

    def peakmem_search_burst(self):
        """Memory under load: 20 sequential search_imas calls."""

        async def _burst():
            async with _fixture.client:
                for q in list(SEARCH_QUERIES.values()) * 4:
                    await _fixture.client.call_tool(
                        "search_imas", {"query": q, "k": 10}
                    )

        asyncio.run(_burst())

    def peakmem_export_large_ids(self):
        """Export serialization memory."""

        async def _export():
            async with _fixture.client:
                return await _fixture.client.call_tool(
                    "export_imas_ids", {"ids_name": IDS_NAMES["large"]}
                )

        asyncio.run(_export())

    def peakmem_encoder_loaded(self):
        """Embedding model memory footprint."""
        from imas_codex.embeddings.encoder import Encoder

        enc = Encoder()
        enc.embed_texts(["warmup"])

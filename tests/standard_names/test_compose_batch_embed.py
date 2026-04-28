"""Tests for batch embedding in hybrid-search neighbours.

Verifies that :func:`embed_query_texts` and
:func:`_hybrid_search_neighbours_batch` perform a single batch embed
call (not N+1 sequential calls) and pass ``prompt_name="query"``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# embed_query_texts — single round-trip, query prompt
# ---------------------------------------------------------------------------


class TestEmbedQueryTexts:
    """embed_query_texts calls encoder.embed_texts once with prompt_name='query'."""

    def test_single_call_for_batch(self) -> None:
        """A 5-text batch triggers exactly one encoder.embed_texts call."""
        mock_encoder = MagicMock()
        mock_encoder.embed_texts.return_value = np.random.randn(5, 256)

        with patch(
            "imas_codex.embeddings.description._get_encoder",
            return_value=mock_encoder,
        ):
            from imas_codex.embeddings.description import embed_query_texts

            texts = [f"text_{i}" for i in range(5)]
            result = embed_query_texts(texts)

        assert mock_encoder.embed_texts.call_count == 1
        assert len(result) == 5
        # Each result is a plain list[float]
        for emb in result:
            assert isinstance(emb, list)
            assert all(isinstance(v, float) for v in emb)

    def test_prompt_name_query(self) -> None:
        """embed_query_texts passes prompt_name='query' to the encoder."""
        mock_encoder = MagicMock()
        mock_encoder.embed_texts.return_value = np.random.randn(2, 256)

        with patch(
            "imas_codex.embeddings.description._get_encoder",
            return_value=mock_encoder,
        ):
            from imas_codex.embeddings.description import embed_query_texts

            embed_query_texts(["alpha", "beta"])

        _args, kwargs = mock_encoder.embed_texts.call_args
        assert kwargs.get("prompt_name") == "query" or (
            len(_args) >= 2 and _args[1] == "query"
        ), "prompt_name='query' not passed to encoder.embed_texts"

    def test_empty_input_returns_empty(self) -> None:
        """Empty list input returns [] without calling the encoder."""
        mock_encoder = MagicMock()
        with patch(
            "imas_codex.embeddings.description._get_encoder",
            return_value=mock_encoder,
        ):
            from imas_codex.embeddings.description import embed_query_texts

            result = embed_query_texts([])
        assert result == []
        mock_encoder.embed_texts.assert_not_called()


# ---------------------------------------------------------------------------
# _hybrid_search_neighbours_batch — batch embed + per-item search
# ---------------------------------------------------------------------------


def _make_mock_search_hit(path: str, score: float = 0.9):
    """Create a minimal mock SearchHit with required attributes."""
    hit = MagicMock()
    hit.path = path
    hit.score = score
    hit.ids_name = path.split("/")[0]
    hit.units = "eV"
    hit.physics_domain = "transport"
    hit.documentation = "Mock doc"
    hit.description = "Mock desc"
    hit.cocos_transformation_type = None
    return hit


class TestHybridSearchNeighboursBatch:
    """Batch helper embeds once then fans out hybrid_dd_search calls."""

    def test_single_embed_call_for_batch(self) -> None:
        """A 3-item batch triggers exactly one embed_query_texts call."""
        items = [
            (
                "core_profiles/profiles_1d/electrons/temperature",
                "Electron temperature",
                "transport",
            ),
            (
                "core_profiles/profiles_1d/electrons/density",
                "Electron density",
                "transport",
            ),
            ("equilibrium/time_slice/profiles_1d/psi", "Poloidal flux", "equilibrium"),
        ]

        mock_embeddings = [[0.1] * 256, [0.2] * 256, [0.3] * 256]

        mock_gc = MagicMock()
        mock_gc.query.return_value = []  # SN resolution returns empty

        mock_hit = _make_mock_search_hit("some/other/path", 0.85)

        with (
            patch(
                "imas_codex.embeddings.description.embed_query_texts",
                return_value=mock_embeddings,
            ) as mock_embed,
            patch(
                "imas_codex.graph.dd_search.hybrid_dd_search",
                return_value=[mock_hit],
            ) as mock_search,  # noqa: F841
        ):
            from imas_codex.standard_names.workers import (
                _hybrid_search_neighbours_batch,
            )

            results = _hybrid_search_neighbours_batch(mock_gc, items)

        # Exactly ONE embed call for all 3 unique description texts
        assert mock_embed.call_count == 1
        embedded_texts = mock_embed.call_args[0][0]
        assert len(embedded_texts) == 3

        # Results returned for all 3 items
        assert len(results) == 3
        for r in results:
            assert isinstance(r, list)

    def test_dedup_identical_descriptions(self) -> None:
        """Identical descriptions are embedded only once (dedup)."""
        items = [
            ("path/a", "Electron temperature", None),
            ("path/b", "Electron temperature", None),
            ("path/c", "Ion temperature", None),
        ]

        mock_gc = MagicMock()
        mock_gc.query.return_value = []

        with (
            patch(
                "imas_codex.embeddings.description.embed_query_texts",
                return_value=[[0.1] * 256, [0.2] * 256],
            ) as mock_embed,
            patch(
                "imas_codex.graph.dd_search.hybrid_dd_search",
                return_value=[],
            ),
        ):
            from imas_codex.standard_names.workers import (
                _hybrid_search_neighbours_batch,
            )

            _hybrid_search_neighbours_batch(mock_gc, items)

        # Only 2 unique texts embedded, not 3
        embedded_texts = mock_embed.call_args[0][0]
        assert len(embedded_texts) == 2
        assert "Electron temperature" in embedded_texts
        assert "Ion temperature" in embedded_texts

    def test_empty_items_returns_empty(self) -> None:
        """Empty items list returns empty results."""
        from imas_codex.standard_names.workers import (
            _hybrid_search_neighbours_batch,
        )

        mock_gc = MagicMock()
        results = _hybrid_search_neighbours_batch(mock_gc, [])
        assert results == []


# ---------------------------------------------------------------------------
# hybrid_dd_search — pre-computed embedding skips internal _embed
# ---------------------------------------------------------------------------


class TestHybridDdSearchEmbeddingParam:
    """hybrid_dd_search(embedding=[...]) skips internal _embed call."""

    def test_precomputed_embedding_skips_embed(self) -> None:
        """When embedding is provided, _embed is NOT called."""
        pre_emb = [0.1] * 256

        mock_gc = MagicMock()
        # Return empty for all queries
        mock_gc.query.return_value = []

        with (
            patch("imas_codex.graph.dd_search._embed") as mock_embed,
            patch(
                "imas_codex.tools.graph_search._text_search_dd_paths",
                return_value=[],
            ),
            patch(
                "imas_codex.tools.graph_search._dd_version_clause",
                return_value="",
            ),
        ):
            from imas_codex.graph.dd_search import hybrid_dd_search

            hybrid_dd_search(mock_gc, "electron temperature", embedding=pre_emb, k=5)

        mock_embed.assert_not_called()

    def test_none_embedding_calls_embed(self) -> None:
        """When embedding is None (default), _embed IS called."""
        mock_gc = MagicMock()
        mock_gc.query.return_value = []

        with (
            patch(
                "imas_codex.graph.dd_search._embed",
                return_value=[0.1] * 256,
            ) as mock_embed,
            patch(
                "imas_codex.tools.graph_search._text_search_dd_paths",
                return_value=[],
            ),
            patch(
                "imas_codex.tools.graph_search._dd_version_clause",
                return_value="",
            ),
        ):
            from imas_codex.graph.dd_search import hybrid_dd_search

            hybrid_dd_search(mock_gc, "electron temperature", k=5)

        mock_embed.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

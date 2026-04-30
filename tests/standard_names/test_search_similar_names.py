"""Tests for plan 39 Phase 0 (a) — `gc=` reuse and `include_superseded` flag.

Plan reference: ``plans/features/standard-names/39-structured-fanout.md`` §3.6.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _row(
    name: str,
    *,
    pipeline_status: str = "named",
    score: float = 0.9,
) -> dict:
    return {
        "id": name,
        "description": "doc",
        "kind": "scalar",
        "unit": "eV",
        "score": score,
        "pipeline_status": pipeline_status,
    }


class TestGCReuse:
    """When ``gc`` is supplied, no new ``GraphClient`` is instantiated."""

    @patch("imas_codex.graph.client.GraphClient")
    @patch("imas_codex.embeddings.encoder.Encoder")
    def test_gc_kwarg_reuses_session(self, MockEncoder, MockGC) -> None:
        from imas_codex.standard_names.search import search_standard_names_vector

        mock_enc = MagicMock()
        mock_enc.embed_texts.return_value = [[0.1] * 768]
        MockEncoder.return_value = mock_enc

        external_gc = MagicMock()
        external_gc.query.return_value = []

        result = search_standard_names_vector(
            "electron temperature", k=5, gc=external_gc
        )

        # Caller-supplied gc is used; constructor never invoked.
        assert result == []
        external_gc.query.assert_called_once()
        MockGC.assert_not_called()

    @patch("imas_codex.graph.client.GraphClient")
    @patch("imas_codex.embeddings.encoder.Encoder")
    def test_no_gc_opens_own_session(self, MockEncoder, MockGC) -> None:
        """Back-compat: when gc=None, the function opens its own session."""
        from imas_codex.standard_names.search import search_standard_names_vector

        mock_enc = MagicMock()
        mock_enc.embed_texts.return_value = [[0.1] * 768]
        MockEncoder.return_value = mock_enc

        gc_instance = MagicMock()
        gc_instance.query.return_value = []
        gc_ctx = MagicMock()
        gc_ctx.__enter__ = MagicMock(return_value=gc_instance)
        gc_ctx.__exit__ = MagicMock(return_value=False)
        MockGC.return_value = gc_ctx

        search_standard_names_vector("electron temperature", k=5)

        MockGC.assert_called_once()
        gc_ctx.__exit__.assert_called_once()


class TestIncludeSuperseded:
    """``include_superseded=True`` drops the superseded WHERE clause."""

    @patch("imas_codex.graph.client.GraphClient")
    @patch("imas_codex.embeddings.encoder.Encoder")
    def test_default_excludes_superseded(self, MockEncoder, MockGC) -> None:
        from imas_codex.standard_names.search import search_standard_names_vector

        mock_enc = MagicMock()
        mock_enc.embed_texts.return_value = [[0.1] * 768]
        MockEncoder.return_value = mock_enc

        gc = MagicMock()
        gc.query.return_value = []

        search_standard_names_vector("electron temperature", k=5, gc=gc)

        cypher = gc.query.call_args[0][0]
        assert "superseded" in cypher
        assert "quarantined" in cypher
        assert "exhausted" in cypher

    @patch("imas_codex.graph.client.GraphClient")
    @patch("imas_codex.embeddings.encoder.Encoder")
    def test_include_superseded_drops_filter(self, MockEncoder, MockGC) -> None:
        from imas_codex.standard_names.search import search_standard_names_vector

        mock_enc = MagicMock()
        mock_enc.embed_texts.return_value = [[0.1] * 768]
        MockEncoder.return_value = mock_enc

        gc = MagicMock()
        gc.query.return_value = []

        search_standard_names_vector(
            "electron temperature", k=5, gc=gc, include_superseded=True
        )

        cypher = gc.query.call_args[0][0]
        assert "superseded" not in cypher
        # Other lifecycle filters remain.
        assert "quarantined" in cypher
        assert "exhausted" in cypher

    @patch("imas_codex.graph.client.GraphClient")
    @patch("imas_codex.embeddings.encoder.Encoder")
    def test_include_superseded_returns_superseded_rows(
        self, MockEncoder, MockGC
    ) -> None:
        """When the flag is set and the graph returns superseded rows, they
        flow through to the caller (the WHERE no longer filters them)."""
        from imas_codex.standard_names.search import search_standard_names_vector

        mock_enc = MagicMock()
        mock_enc.embed_texts.return_value = [[0.1] * 768]
        MockEncoder.return_value = mock_enc

        rows = [
            _row("active_name", pipeline_status="named", score=0.95),
            _row("legacy_name", pipeline_status="superseded", score=0.85),
        ]
        gc = MagicMock()
        gc.query.return_value = rows

        result = search_standard_names_vector(
            "electron temperature", k=5, gc=gc, include_superseded=True
        )
        ids = [r["id"] for r in result]
        assert "active_name" in ids
        assert "legacy_name" in ids

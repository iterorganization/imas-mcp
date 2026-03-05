"""Tests for ingestion.search — direct Cypher vector search."""

from unittest.mock import MagicMock, patch

import numpy as np

from imas_codex.ingestion.search import ChunkSearchResult, search_code_chunks


class TestChunkSearchResult:
    def test_fields(self):
        r = ChunkSearchResult(
            chunk_id="tcv:file.py:chunk_0",
            content="def foo(): pass",
            function_name="foo",
            source_file="/home/codes/file.py",
            facility_id="tcv",
            related_ids=["equilibrium"],
            score=0.85,
            start_line=1,
            end_line=2,
        )
        assert r.chunk_id == "tcv:file.py:chunk_0"
        assert r.score == 0.85
        assert r.start_line == 1

    def test_optional_fields_default_none(self):
        r = ChunkSearchResult(
            chunk_id="x",
            content="c",
            function_name=None,
            source_file="f",
            facility_id="tcv",
            related_ids=[],
            score=0.5,
        )
        assert r.start_line is None
        assert r.end_line is None


class TestSearchCodeChunks:
    @patch("imas_codex.ingestion.search.GraphClient")
    @patch("imas_codex.ingestion.search.Encoder")
    def test_basic_search(self, mock_encoder_cls, mock_gc_cls):
        encoder = MagicMock()
        encoder.embed_texts.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_encoder_cls.return_value = encoder

        gc = MagicMock()
        gc.query.return_value = [
            {
                "chunk_id": "tcv:file.py:chunk_0",
                "content": "code here",
                "function_name": "my_func",
                "source_file": "/codes/file.py",
                "facility_id": "tcv",
                "related_ids": ["equilibrium"],
                "start_line": 10,
                "end_line": 20,
                "score": 0.9,
            }
        ]
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)

        results = search_code_chunks("equilibrium solver", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "tcv:file.py:chunk_0"
        assert results[0].score == 0.9
        encoder.embed_texts.assert_called_once_with(["equilibrium solver"])

    @patch("imas_codex.ingestion.search.GraphClient")
    @patch("imas_codex.ingestion.search.Encoder")
    def test_min_score_filtering(self, mock_encoder_cls, mock_gc_cls):
        encoder = MagicMock()
        encoder.embed_texts.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_encoder_cls.return_value = encoder

        gc = MagicMock()
        gc.query.return_value = [
            {
                "chunk_id": "a",
                "content": "hi",
                "function_name": None,
                "source_file": "f",
                "facility_id": "tcv",
                "related_ids": [],
                "start_line": None,
                "end_line": None,
                "score": 0.3,
            },
            {
                "chunk_id": "b",
                "content": "lo",
                "function_name": None,
                "source_file": "g",
                "facility_id": "tcv",
                "related_ids": [],
                "start_line": None,
                "end_line": None,
                "score": 0.8,
            },
        ]
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)

        results = search_code_chunks("query", min_score=0.5)
        assert len(results) == 1
        assert results[0].chunk_id == "b"

    @patch("imas_codex.ingestion.search.GraphClient")
    @patch("imas_codex.ingestion.search.Encoder")
    def test_facility_filter(self, mock_encoder_cls, mock_gc_cls):
        encoder = MagicMock()
        encoder.embed_texts.return_value = np.array([[0.1, 0.2]])
        mock_encoder_cls.return_value = encoder

        gc = MagicMock()
        gc.query.return_value = []
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)

        search_code_chunks("query", facility="tcv")

        # Verify the Cypher query contains facility filter
        call_args = gc.query.call_args
        cypher = call_args[0][0]
        assert "node.facility_id = $facility" in cypher
        assert call_args[1]["facility"] == "tcv"

    @patch("imas_codex.ingestion.search.GraphClient")
    @patch("imas_codex.ingestion.search.Encoder")
    def test_ids_filter(self, mock_encoder_cls, mock_gc_cls):
        encoder = MagicMock()
        encoder.embed_texts.return_value = np.array([[0.1, 0.2]])
        mock_encoder_cls.return_value = encoder

        gc = MagicMock()
        gc.query.return_value = []
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)

        search_code_chunks("query", ids_filter=["equilibrium", "core_profiles"])

        call_args = gc.query.call_args
        cypher = call_args[0][0]
        assert "related_ids" in cypher
        assert call_args[1]["ids_filter"] == ["equilibrium", "core_profiles"]

    @patch("imas_codex.ingestion.search.GraphClient")
    @patch("imas_codex.ingestion.search.Encoder")
    def test_empty_results(self, mock_encoder_cls, mock_gc_cls):
        encoder = MagicMock()
        encoder.embed_texts.return_value = np.array([[0.1, 0.2]])
        mock_encoder_cls.return_value = encoder

        gc = MagicMock()
        gc.query.return_value = []
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)

        results = search_code_chunks("no match")
        assert results == []

    @patch("imas_codex.ingestion.search.GraphClient")
    @patch("imas_codex.ingestion.search.Encoder")
    def test_combined_filters(self, mock_encoder_cls, mock_gc_cls):
        encoder = MagicMock()
        encoder.embed_texts.return_value = np.array([[0.1, 0.2]])
        mock_encoder_cls.return_value = encoder

        gc = MagicMock()
        gc.query.return_value = []
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)

        search_code_chunks("query", facility="tcv", ids_filter=["equilibrium"])

        call_args = gc.query.call_args
        cypher = call_args[0][0]
        assert "node.facility_id = $facility" in cypher
        assert "related_ids" in cypher

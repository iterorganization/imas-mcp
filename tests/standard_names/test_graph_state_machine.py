"""Tests for graph-state-machine claim/mark/release pattern.

Validates that:
- claim_names_for_validation atomically claims with token + ORDER BY rand()
- mark_names_validated writes results with token verification
- release_validation_claims releases on error
- claim_names_for_embedding claims unembedded validated names
- mark_names_embedded writes embeddings with token verification
- persist_generated_name_batch writes names immediately to graph
- get_validated_names reads validated names for consolidation
- mark_names_consolidated writes consolidated_at timestamp

All graph operations are mocked — these tests verify the Cypher logic
and claim/release protocol, not actual Neo4j behavior.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_candidates() -> list[dict]:
    """Sample compose output ready for persist_generated_name_batch."""
    return [
        {
            "id": "electron_temperature",
            "description": "Electron temperature profile",
            "documentation": "The electron temperature $T_e$.",
            "kind": "scalar",
            "unit": "eV",
            "tags": ["core_profiles"],
            "fields": {
                "physical_base": "temperature",
                "subject": "electron",
            },
        },
        {
            "id": "ion_density",
            "description": "Ion density",
            "documentation": "Total ion density $n_i$.",
            "kind": "scalar",
            "unit": "m^-3",
            "tags": ["core_profiles"],
            "fields": {"physical_base": "density", "subject": "ion"},
        },
    ]


# ---------------------------------------------------------------------------
# persist_generated_name_batch tests
# ---------------------------------------------------------------------------


class TestPersistGeneratedNameBatch:
    """Tests for immediate per-batch persist during compose."""

    def test_empty_candidates_returns_zero(self):
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        result = persist_generated_name_batch([], compose_model="test-model")
        assert result == 0

    @patch("imas_codex.standard_names.graph_ops._finalize_generated_name_stage")
    @patch("imas_codex.standard_names.graph_ops.write_standard_names")
    @patch("imas_codex.embeddings.description.embed_descriptions_batch")
    def test_enriches_with_provenance(
        self, mock_embed, mock_write, mock_finalize, sample_candidates
    ):
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        # Simulate successful embedding for each candidate
        def _embed_side_effect(
            items, text_field="description", embedding_field="embedding"
        ):
            for item in items:
                item[embedding_field] = [0.1, 0.2, 0.3]
            return items

        mock_embed.side_effect = _embed_side_effect
        mock_write.return_value = 2
        result = persist_generated_name_batch(
            sample_candidates, compose_model="claude-test"
        )
        assert result == 2

        written = mock_write.call_args[0][0]
        for entry in written:
            assert entry["model"] == "claude-test"
            assert entry["pipeline_status"] == "named"
            assert "generated_at" in entry

    @patch("imas_codex.standard_names.graph_ops._finalize_generated_name_stage")
    @patch("imas_codex.standard_names.graph_ops.write_standard_names")
    @patch("imas_codex.embeddings.description.embed_descriptions_batch")
    def test_extracts_grammar_fields(
        self, mock_embed, mock_write, mock_finalize, sample_candidates
    ):
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        def _embed_side_effect(
            items, text_field="description", embedding_field="embedding"
        ):
            for item in items:
                item[embedding_field] = [0.1, 0.2, 0.3]
            return items

        mock_embed.side_effect = _embed_side_effect
        mock_write.return_value = 2
        persist_generated_name_batch(sample_candidates, compose_model="test")

        written = mock_write.call_args[0][0]
        # Grammar fields (physical_base, subject, etc.) are no longer
        # extracted from compose `fields` dict — grammar decomposition
        # happens inside write_standard_names() via _grammar_decomposition().
        # Verify the compose output no longer carries legacy field names.
        assert "physical_base" not in written[0]
        assert "subject" not in written[0]

    @patch("imas_codex.standard_names.graph_ops._finalize_generated_name_stage")
    @patch("imas_codex.standard_names.graph_ops.write_standard_names")
    @patch("imas_codex.embeddings.description.embed_descriptions_batch")
    def test_embeds_name_string(
        self, mock_embed, mock_write, mock_finalize, sample_candidates
    ):
        """persist_generated_name_batch embeds the name (id) field, not description."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        def _embed_side_effect(
            items, text_field="description", embedding_field="embedding"
        ):
            for item in items:
                item[embedding_field] = [0.1, 0.2, 0.3]
            return items

        mock_embed.side_effect = _embed_side_effect
        mock_write.return_value = 2
        persist_generated_name_batch(sample_candidates, compose_model="test")

        # Must embed using text_field="_embed_text" (post-A1: name — description basis)
        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args
        assert (
            call_kwargs[1].get("text_field") == "_embed_text"
            or call_kwargs[0][0] is sample_candidates
        )
        # Verify text_field kwarg is "_embed_text"
        _, kwargs = mock_embed.call_args
        assert kwargs.get("text_field") == "_embed_text"

    @patch("imas_codex.standard_names.graph_ops._finalize_generated_name_stage")
    @patch("imas_codex.standard_names.graph_ops.write_standard_names")
    @patch("imas_codex.embeddings.description.embed_descriptions_batch")
    def test_writes_embedding_and_embedded_at(
        self, mock_embed, mock_write, mock_finalize, sample_candidates
    ):
        """Each persisted SN has embedding + embedded_at when embedding succeeds."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        def _embed_side_effect(
            items, text_field="description", embedding_field="embedding"
        ):
            for item in items:
                item[embedding_field] = [0.4, 0.5, 0.6]
            return items

        mock_embed.side_effect = _embed_side_effect
        mock_write.return_value = 2
        persist_generated_name_batch(sample_candidates, compose_model="test")

        written = mock_write.call_args[0][0]
        for entry in written:
            assert entry["embedding"] == [0.4, 0.5, 0.6]
            assert entry["embedded_at"] is not None
            assert entry["pipeline_status"] == "named"
            assert entry["validation_status"] != "quarantined"

    @patch("imas_codex.standard_names.graph_ops._finalize_generated_name_stage")
    @patch("imas_codex.standard_names.graph_ops.write_standard_names")
    @patch("imas_codex.embeddings.description.embed_descriptions_batch")
    def test_quarantines_on_embedding_failure(
        self, mock_embed, mock_write, mock_finalize, sample_candidates
    ):
        """Candidates with failed embeddings get quarantined."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        # Simulate partial failure — first succeeds, second fails
        def _embed_side_effect(
            items, text_field="description", embedding_field="embedding"
        ):
            items[0][embedding_field] = [0.1, 0.2]
            items[1][embedding_field] = None  # failed
            return items

        mock_embed.side_effect = _embed_side_effect
        mock_write.return_value = 2
        persist_generated_name_batch(sample_candidates, compose_model="test")

        written = mock_write.call_args[0][0]
        # First candidate: embedded successfully
        assert written[0]["embedding"] == [0.1, 0.2]
        assert written[0]["validation_status"] != "quarantined"
        # Second candidate: quarantined
        assert written[1]["embedding"] is None
        assert written[1]["validation_status"] == "quarantined"
        assert "embedding_failed" in written[1]["validation_issues"]

    @patch("imas_codex.standard_names.graph_ops._finalize_generated_name_stage")
    @patch("imas_codex.standard_names.graph_ops.write_standard_names")
    @patch(
        "imas_codex.embeddings.description.embed_descriptions_batch",
        side_effect=ConnectionError("embed server down"),
    )
    def test_quarantines_all_on_total_embed_failure(
        self, mock_embed, mock_write, mock_finalize, sample_candidates
    ):
        """When embed server is down, all candidates are quarantined."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        mock_write.return_value = 2
        persist_generated_name_batch(sample_candidates, compose_model="test")

        written = mock_write.call_args[0][0]
        for entry in written:
            assert entry["validation_status"] == "quarantined"
            assert "embedding_failed" in entry["validation_issues"]
            assert entry.get("embedding") is None

    @patch("imas_codex.standard_names.graph_ops._finalize_generated_name_stage")
    @patch("imas_codex.standard_names.graph_ops.write_standard_names")
    @patch("imas_codex.embeddings.description.embed_descriptions_batch")
    def test_idempotent_rerun(
        self, mock_embed, mock_write, mock_finalize, sample_candidates
    ):
        """Re-running persist on the same batch produces same result."""
        from copy import deepcopy

        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        def _embed_side_effect(
            items, text_field="description", embedding_field="embedding"
        ):
            for item in items:
                item[embedding_field] = [0.1, 0.2, 0.3]
            return items

        mock_embed.side_effect = _embed_side_effect
        mock_write.return_value = 2

        batch1 = deepcopy(sample_candidates)
        batch2 = deepcopy(sample_candidates)

        persist_generated_name_batch(batch1, compose_model="test")
        first_written = mock_write.call_args[0][0]

        persist_generated_name_batch(batch2, compose_model="test")
        second_written = mock_write.call_args[0][0]

        # Same embeddings, same pipeline_status, same structure
        for a, b in zip(first_written, second_written, strict=True):
            assert a["embedding"] == b["embedding"]
            assert a["pipeline_status"] == b["pipeline_status"]
            assert a["validation_status"] == b["validation_status"]


# ---------------------------------------------------------------------------
# claim_names_for_validation tests
# ---------------------------------------------------------------------------


class TestClaimNamesForValidation:
    """Tests for validation claim/mark/release protocol."""

    @patch("imas_codex.standard_names.graph_ops.GraphClient")
    def test_claim_returns_token_and_items(self, mock_client_cls):
        from imas_codex.standard_names.graph_ops import claim_names_for_validation

        gc = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        gc.query.side_effect = [
            None,  # SET query
            [
                {
                    "id": "electron_temperature",
                    "description": "Te",
                    "documentation": None,
                    "kind": "scalar",
                    "unit": "eV",
                    "tags": None,
                    "links": None,
                    "source_paths": None,
                    "physical_base": "temperature",
                    "subject": "electron",
                    "component": None,
                    "coordinate": None,
                    "position": None,
                    "process": None,
                    "geometric_base": None,
                    "object": None,
                    "source_ids": [],
                }
            ],
        ]

        token, items = claim_names_for_validation(limit=10)

        assert isinstance(token, str)
        assert len(token) == 36  # UUID
        assert len(items) == 1
        assert items[0]["id"] == "electron_temperature"
        assert gc.query.call_count == 2

    @patch("imas_codex.standard_names.graph_ops.GraphClient")
    def test_claim_uses_random_ordering(self, mock_client_cls):
        from imas_codex.standard_names.graph_ops import claim_names_for_validation

        gc = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        gc.query.side_effect = [None, []]

        claim_names_for_validation(limit=5)

        cypher = gc.query.call_args_list[0][0][0]
        assert "ORDER BY rand()" in cypher

    @patch("imas_codex.standard_names.graph_ops.GraphClient")
    def test_claim_empty_returns_empty(self, mock_client_cls):
        from imas_codex.standard_names.graph_ops import claim_names_for_validation

        gc = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        gc.query.side_effect = [None, []]

        token, items = claim_names_for_validation(limit=10)
        assert items == []

    @patch("imas_codex.standard_names.graph_ops.GraphClient")
    def test_claim_stale_claims_reclaimed(self, mock_client_cls):
        """Verify Cypher checks claimed_at against timeout for orphan recovery."""
        from imas_codex.standard_names.graph_ops import claim_names_for_validation

        gc = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        gc.query.side_effect = [None, []]

        claim_names_for_validation(limit=5)

        cypher = gc.query.call_args_list[0][0][0]
        assert "duration($timeout)" in cypher

    @patch("imas_codex.standard_names.graph_ops.GraphClient")
    def test_two_step_verify_pattern(self, mock_client_cls):
        """Verify claim uses SET then read-by-token (anti double-claim)."""
        from imas_codex.standard_names.graph_ops import claim_names_for_validation

        gc = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        gc.query.side_effect = [None, []]

        claim_names_for_validation(limit=5)

        # First query: SET claim_token
        first_cypher = gc.query.call_args_list[0][0][0]
        assert "SET" in first_cypher
        assert "claim_token" in first_cypher

        # Second query: read by claim_token
        second_cypher = gc.query.call_args_list[1][0][0]
        assert "claim_token: $token" in second_cypher


# ---------------------------------------------------------------------------
# mark_names_validated tests
# ---------------------------------------------------------------------------


class TestMarkNamesValidated:
    """Tests for writing validation results with token verification."""

    @patch("imas_codex.standard_names.graph_ops.GraphClient")
    def test_mark_with_token_verification(self, mock_client_cls):
        from imas_codex.standard_names.graph_ops import mark_names_validated

        gc = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        gc.query.return_value = [{"marked": 2}]

        results = [
            {
                "id": "electron_temperature",
                "validation_issues": ["semantic: minor issue"],
                "validation_layer_summary": '{"pydantic": {"passed": 5}}',
            },
            {
                "id": "ion_density",
                "validation_issues": [],
                "validation_layer_summary": "{}",
            },
        ]

        marked = mark_names_validated("test-token-123", results)
        assert marked == 2

        call_kwargs = gc.query.call_args[1]
        assert call_kwargs["token"] == "test-token-123"

        # Verify Cypher includes token check
        cypher = gc.query.call_args[0][0]
        assert "claim_token: $token" in cypher

    def test_mark_empty_returns_zero(self):
        from imas_codex.standard_names.graph_ops import mark_names_validated

        assert mark_names_validated("token", []) == 0

    @patch("imas_codex.standard_names.graph_ops.GraphClient")
    def test_mark_clears_claim(self, mock_client_cls):
        """Verify mark sets claimed_at and claim_token to null."""
        from imas_codex.standard_names.graph_ops import mark_names_validated

        gc = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        gc.query.return_value = [{"marked": 1}]

        mark_names_validated("tok", [{"id": "te", "validation_issues": []}])

        cypher = gc.query.call_args[0][0]
        assert "claimed_at = null" in cypher
        assert "claim_token = null" in cypher


# ---------------------------------------------------------------------------
# release_validation_claims tests
# ---------------------------------------------------------------------------


class TestReleaseValidationClaims:
    """Tests for releasing claims on error."""

    @patch("imas_codex.standard_names.graph_ops.GraphClient")
    def test_release_clears_claimed_at(self, mock_client_cls):
        from imas_codex.standard_names.graph_ops import release_validation_claims

        gc = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        gc.query.return_value = [{"released": 3}]

        released = release_validation_claims("test-token")
        assert released == 3

        cypher = gc.query.call_args[0][0]
        assert "claim_token: $token" in cypher
        assert "claimed_at = null" in cypher


# ---------------------------------------------------------------------------
# claim_names_for_embedding tests
# ---------------------------------------------------------------------------


class TestClaimNamesForEmbedding:
    """Tests for embedding claim protocol."""

    @patch("imas_codex.standard_names.graph_ops.GraphClient")
    def test_claim_embedding_requires_validated(self, mock_client_cls):
        """Embedding claims should only target validated names."""
        from imas_codex.standard_names.graph_ops import claim_names_for_embedding

        gc = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        gc.query.side_effect = [None, []]

        claim_names_for_embedding(limit=10)

        cypher = gc.query.call_args_list[0][0][0]
        assert "validated_at IS NOT NULL" in cypher
        assert "embedding IS NULL" in cypher

    @patch("imas_codex.standard_names.graph_ops.GraphClient")
    def test_claim_embedding_two_step(self, mock_client_cls):
        """Verify embedding claim uses SET then read-by-token."""
        from imas_codex.standard_names.graph_ops import claim_names_for_embedding

        gc = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        gc.query.side_effect = [None, [{"id": "te", "description": "Te"}]]

        token, items = claim_names_for_embedding(limit=5)
        assert len(items) == 1
        assert gc.query.call_count == 2


# ---------------------------------------------------------------------------
# mark_names_embedded tests
# ---------------------------------------------------------------------------


class TestMarkNamesEmbedded:
    """Tests for writing embeddings with token verification."""

    @patch("imas_codex.standard_names.graph_ops.GraphClient")
    def test_mark_embedded_with_token(self, mock_client_cls):
        from imas_codex.standard_names.graph_ops import mark_names_embedded

        gc = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        gc.query.return_value = [{"marked": 2}]

        batch = [
            {"id": "electron_temperature", "embedding": [0.1, 0.2, 0.3]},
            {"id": "ion_density", "embedding": [0.4, 0.5, 0.6]},
        ]

        marked = mark_names_embedded("emb-token", batch)
        assert marked == 2

        call_kwargs = gc.query.call_args[1]
        assert call_kwargs["token"] == "emb-token"

    def test_mark_embedded_empty_returns_zero(self):
        from imas_codex.standard_names.graph_ops import mark_names_embedded

        assert mark_names_embedded("token", []) == 0

    @patch("imas_codex.standard_names.graph_ops.GraphClient")
    def test_mark_embedded_skips_null_embeddings(self, mock_client_cls):
        from imas_codex.standard_names.graph_ops import mark_names_embedded

        gc = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        gc.query.return_value = [{"marked": 1}]

        batch = [
            {"id": "good", "embedding": [0.1]},
            {"id": "bad", "embedding": None},
        ]

        mark_names_embedded("token", batch)
        call_kwargs = gc.query.call_args[1]
        assert len(call_kwargs["batch"]) == 1


# ---------------------------------------------------------------------------
# get_validated_names tests
# ---------------------------------------------------------------------------


class TestGetValidatedNames:
    """Tests for read-only consolidation query."""

    @patch("imas_codex.standard_names.graph_ops.GraphClient")
    def test_returns_validated_names(self, mock_client_cls):
        from imas_codex.standard_names.graph_ops import get_validated_names

        gc = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        gc.query.return_value = [
            {
                "id": "electron_temperature",
                "description": "Te",
                "documentation": None,
                "kind": "scalar",
                "unit": "eV",
                "tags": None,
                "links": None,
                "source_paths": None,
                "source_ids": [],
            },
        ]

        names = get_validated_names()
        assert len(names) == 1
        assert names[0]["id"] == "electron_temperature"

    @patch("imas_codex.standard_names.graph_ops.GraphClient")
    def test_ids_filter(self, mock_client_cls):
        from imas_codex.standard_names.graph_ops import get_validated_names

        gc = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        gc.query.return_value = []

        get_validated_names(ids_filter="equilibrium")

        cypher = gc.query.call_args[0][0]
        assert "STARTS WITH" in cypher


# ---------------------------------------------------------------------------
# mark_names_consolidated tests
# ---------------------------------------------------------------------------


class TestMarkNamesConsolidated:
    """Tests for consolidation marking."""

    @patch("imas_codex.standard_names.graph_ops.GraphClient")
    def test_marks_consolidated(self, mock_client_cls):
        from imas_codex.standard_names.graph_ops import mark_names_consolidated

        gc = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=gc)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        gc.query.return_value = [{"marked": 3}]

        result = mark_names_consolidated(["a", "b", "c"])
        assert result == 3

        cypher = gc.query.call_args[0][0]
        assert "consolidated_at" in cypher

    def test_empty_returns_zero(self):
        from imas_codex.standard_names.graph_ops import mark_names_consolidated

        assert mark_names_consolidated([]) == 0


# ---------------------------------------------------------------------------
# Worker integration tests (async)
# ---------------------------------------------------------------------------


class TestValidateWorkerClaimLoop:
    """Integration tests for validate_worker claim loop."""

    @pytest.mark.asyncio()
    async def test_validate_exits_on_empty_claims(self):
        """validate_worker exits after MAX_IDLE empty claims."""
        from imas_codex.standard_names.workers import validate_worker

        state = _make_mock_state()

        with patch(
            "imas_codex.standard_names.graph_ops.claim_names_for_validation"
        ) as mock_claim:
            mock_claim.return_value = ("token", [])

            await validate_worker(state)

            assert state.validate_phase.mark_done.called
            assert state.stats["validate_valid"] == 0

    @pytest.mark.asyncio()
    async def test_validate_processes_claimed_names(self):
        """validate_worker processes claimed names and marks results."""
        from imas_codex.standard_names.workers import validate_worker

        state = _make_mock_state()

        claimed_items = [
            {
                "id": "electron_temperature",
                "description": "Te",
                "physical_base": "temperature",
                "subject": "electron",
                "component": None,
                "coordinate": None,
                "position": None,
                "process": None,
                "geometric_base": None,
                "object": None,
            },
        ]

        call_count = 0

        def _claim_side_effect(limit):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("token-1", claimed_items)
            return ("token-2", [])

        with (
            patch(
                "imas_codex.standard_names.graph_ops.claim_names_for_validation",
                side_effect=_claim_side_effect,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.mark_names_validated",
                return_value=1,
            ) as mock_mark,
            patch(
                "imas_standard_names.grammar.parse_standard_name",
            ) as mock_parse,
            patch(
                "imas_standard_names.grammar.compose_standard_name",
                return_value="electron_temperature",
            ),
            patch(
                "imas_codex.standard_names.workers._validate_via_isn",
                return_value=([], {"pydantic": {"passed": 5}}),
            ),
        ):
            mock_parse.return_value = MagicMock()

            await validate_worker(state)

            assert mock_mark.called
            mark_token = mock_mark.call_args[0][0]
            assert mark_token == "token-1"

    @pytest.mark.asyncio()
    async def test_validate_releases_on_error(self):
        """validate_worker releases claims when mark fails."""
        from imas_codex.standard_names.workers import validate_worker

        state = _make_mock_state()

        call_count = 0

        def _claim_side_effect(limit):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("token-err", [{"id": "bad_name"}])
            return ("token-2", [])

        with (
            patch(
                "imas_codex.standard_names.graph_ops.claim_names_for_validation",
                side_effect=_claim_side_effect,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.mark_names_validated",
                side_effect=RuntimeError("graph write failed"),
            ),
            patch(
                "imas_codex.standard_names.graph_ops.release_validation_claims",
                return_value=1,
            ) as mock_release,
            patch(
                "imas_standard_names.grammar.parse_standard_name",
                side_effect=ValueError("bad grammar"),
            ),
            patch(
                "imas_codex.standard_names.workers._validate_via_isn",
                return_value=([], {}),
            ),
        ):
            await validate_worker(state)

            assert mock_release.called
            assert mock_release.call_args[0][0] == "token-err"

    @pytest.mark.asyncio()
    async def test_validate_dry_run_skips(self):
        """validate_worker skips all processing in dry run."""
        from imas_codex.standard_names.workers import validate_worker

        state = _make_mock_state(dry_run=True)

        await validate_worker(state)

        assert state.validate_phase.mark_done.called
        assert state.stats.get("validate_skipped") is True


class TestPersistWorkerClaimLoop:
    """Integration tests for persist_worker (embedding) claim loop."""

    @pytest.mark.asyncio()
    async def test_persist_exits_on_empty(self):
        from imas_codex.standard_names.workers import persist_worker

        state = _make_mock_state()

        with patch(
            "imas_codex.standard_names.graph_ops.claim_names_for_embedding"
        ) as mock_claim:
            mock_claim.return_value = ("token", [])

            await persist_worker(state)

            assert state.persist_phase.mark_done.called

    @pytest.mark.asyncio()
    async def test_persist_embeds_and_marks(self):
        from imas_codex.standard_names.workers import persist_worker

        state = _make_mock_state()

        call_count = 0

        def _claim_side_effect(limit):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("emb-token", [{"id": "te", "description": "Te"}])
            return ("emb-2", [])

        with (
            patch(
                "imas_codex.standard_names.graph_ops.claim_names_for_embedding",
                side_effect=_claim_side_effect,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.mark_names_embedded",
                return_value=1,
            ) as mock_mark,
            patch(
                "imas_codex.embeddings.description.embed_descriptions_batch",
                return_value=[{"id": "te", "embedding": [0.1, 0.2]}],
            ),
        ):
            await persist_worker(state)

            assert mock_mark.called
            assert mock_mark.call_args[0][0] == "emb-token"

    @pytest.mark.asyncio()
    async def test_persist_dry_run_skips(self):
        from imas_codex.standard_names.workers import persist_worker

        state = _make_mock_state(dry_run=True)

        await persist_worker(state)
        assert state.persist_phase.mark_done.called


class TestConsolidateWorkerGraphPrimary:
    """Tests for consolidate_worker graph-primary pattern."""

    @pytest.mark.asyncio()
    async def test_consolidate_reads_from_graph(self):
        from imas_codex.standard_names.workers import consolidate_worker

        state = _make_mock_state()

        mock_result = MagicMock()
        mock_result.approved = [{"id": "te"}]
        mock_result.conflicts = []
        mock_result.coverage_gaps = []
        mock_result.reused = []
        mock_result.stats = {}

        with (
            patch(
                "imas_codex.standard_names.graph_ops.get_validated_names",
                return_value=[{"id": "te", "description": "Te"}],
            ),
            patch(
                "imas_codex.standard_names.consolidation.consolidate_candidates",
                return_value=mock_result,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.mark_names_consolidated",
                return_value=1,
            ) as mock_mark,
        ):
            await consolidate_worker(state)

            assert mock_mark.called
            assert state.consolidate_phase.mark_done.called

    @pytest.mark.asyncio()
    async def test_consolidate_skips_when_empty(self):
        from imas_codex.standard_names.workers import consolidate_worker

        state = _make_mock_state()

        with patch(
            "imas_codex.standard_names.graph_ops.get_validated_names",
            return_value=[],
        ):
            await consolidate_worker(state)

            assert state.consolidate_phase.mark_done.called


# ---------------------------------------------------------------------------
# Pipeline should_stop_fn tests
# ---------------------------------------------------------------------------


class TestPipelineShouldStopFn:
    """Verify downstream workers use stop_requested, not budget."""

    def test_downstream_workers_have_should_stop_fn(self):
        """Validate/consolidate/persist should ignore budget exhaustion."""
        import inspect

        from imas_codex.standard_names.pipeline import run_sn_pipeline

        source = inspect.getsource(run_sn_pipeline)
        assert "should_stop_fn" in source
        assert "_downstream_should_stop" in source

    def test_engine_stop_fn_ignores_budget(self):
        """Supervised loop must use stop_requested, not budget_exhausted.

        Without this, the supervised loop exits on cost-limit and cancel_all()
        kills downstream workers before they process composed names.
        """
        import inspect

        from imas_codex.standard_names.pipeline import run_sn_pipeline

        source = inspect.getsource(run_sn_pipeline)
        # Must pass stop_fn to run_discovery_engine
        assert "stop_fn=" in source
        # stop_fn must check stop_requested only, not should_stop (budget)
        assert "stop_fn=lambda: state.stop_requested" in source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_state(dry_run: bool = False):
    """Create a minimal mock StandardNameBuildState for worker testing."""
    state = MagicMock()
    state.dry_run = dry_run
    state.stop_requested = False
    state.force = False
    state.extracted = []
    state.ids_filter = None
    state.source = "dd"
    state.compose_model = None
    state.stats = {}

    for phase in (
        "validate_phase",
        "consolidate_phase",
        "persist_phase",
        "compose_phase",
    ):
        phase_mock = MagicMock()
        phase_mock.mark_done = MagicMock()
        setattr(state, phase, phase_mock)

    for stat in (
        "validate_stats",
        "consolidate_stats",
        "persist_stats",
        "compose_stats",
        "finalize_stats",
    ):
        stat_mock = MagicMock()
        stat_mock.total = 0
        stat_mock.processed = 0
        stat_mock.errors = 0
        stat_mock.cost = 0.0
        stat_mock.freeze_rate = MagicMock()
        stat_mock.record_batch = MagicMock()
        stat_mock.stream_queue = MagicMock()
        stat_mock.stream_queue.add = MagicMock()
        setattr(state, stat, stat_mock)

    return state

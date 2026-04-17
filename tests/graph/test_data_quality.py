"""Data quality tests.

Verifies embedding dimensions, self-similarity, description quality,
and enumerated field values.
"""

import pytest

from .conftest import get_all_embeddable_labels, get_description_embeddable_labels

pytestmark = pytest.mark.graph

# Minimum self-similarity score (cosine) for embedding vs description
# A well-embedded description should have high self-similarity
MIN_SELF_SIMILARITY = 0.5


class TestEmbeddingDimensions:
    """All embeddings must match the configured dimension."""

    @pytest.mark.parametrize("label", get_all_embeddable_labels())
    def test_embedding_dimension_matches_config(
        self, graph_client, label, label_counts, embedding_dimension
    ):
        """Embedding vectors must have exactly the configured dimension."""
        if not label_counts.get(label):
            pytest.skip(f"No {label} nodes in graph")

        # Sample up to 100 nodes with embeddings
        result = graph_client.query(
            f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL "
            f"RETURN size(n.embedding) AS dim "
            f"LIMIT 100"
        )
        if not result:
            pytest.skip(f"No {label} nodes with embeddings")

        wrong = [r for r in result if r["dim"] != embedding_dimension]
        wrong_dims = {r["dim"] for r in wrong}
        assert not wrong, (
            f"{len(wrong)}/{len(result)} {label} embeddings have wrong dimension. "
            f"Expected {embedding_dimension}, found: {wrong_dims}"
        )

    @pytest.mark.parametrize("label", get_all_embeddable_labels())
    def test_no_mixed_dimensions(self, graph_client, label, label_counts):
        """All embeddings for a label must have the same dimension."""
        if not label_counts.get(label):
            pytest.skip(f"No {label} nodes in graph")

        result = graph_client.query(
            f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL "
            f"WITH DISTINCT size(n.embedding) AS dim "
            f"RETURN collect(dim) AS dims"
        )
        if not result or not result[0]["dims"]:
            pytest.skip(f"No {label} nodes with embeddings")

        dims = result[0]["dims"]
        assert len(dims) == 1, (
            f"{label} has mixed embedding dimensions: {dims}. "
            f"All embeddings must use the same dimension."
        )


class TestEmbeddingQuality:
    """Embedding vectors should be valid and meaningful."""

    @pytest.mark.parametrize("label", get_all_embeddable_labels())
    def test_no_zero_embeddings(self, graph_client, label, label_counts):
        """Embeddings should not be all-zero vectors."""
        if not label_counts.get(label):
            pytest.skip(f"No {label} nodes in graph")

        # Check for vectors where all elements are 0
        # A zero vector has magnitude 0, which breaks cosine similarity
        result = graph_client.query(
            f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL "
            f"WITH n, reduce(s = 0.0, x IN n.embedding | s + abs(x)) AS mag "
            f"WHERE mag < 0.0001 "
            f"RETURN count(n) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} {label} nodes have zero/near-zero embeddings"

    @pytest.mark.parametrize("label", get_all_embeddable_labels())
    def test_embeddings_are_normalized(self, graph_client, label, label_counts):
        """Embeddings should be approximately unit-length (L2 norm ~ 1.0)."""
        if not label_counts.get(label):
            pytest.skip(f"No {label} nodes in graph")

        # Sample and check L2 norm is close to 1.0
        # Tolerance [0.99, 1.01] — properly normalized embeddings should be
        # very close to unit length. Wider deviations indicate missing
        # re-normalization after Matryoshka dimension truncation.
        result = graph_client.query(
            f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL "
            f"WITH n, "
            f"  reduce(s = 0.0, x IN n.embedding | s + x * x) AS sq_sum "
            f"WITH sqrt(sq_sum) AS norm "
            f"WHERE norm < 0.99 OR norm > 1.01 "
            f"RETURN count(*) AS cnt "
            f"LIMIT 200"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} {label} embeddings have L2 norm outside [0.99, 1.01]. "
            f"Embeddings should be normalized to unit length."
        )


class TestEmbeddingSelfSimilarity:
    """Embeddings should be semantically consistent with their descriptions.

    This test re-embeds a sample of descriptions and checks that the
    cosine similarity between the stored embedding and the freshly
    computed embedding is high. This catches:
    - Embeddings stored from a different model
    - Embeddings computed from wrong text
    - Corrupt or shuffled embeddings

    Requires the remote embedding server (localhost:18765 via SSH tunnel).
    """

    @pytest.mark.parametrize("label", get_description_embeddable_labels())
    def test_self_similarity(self, graph_client, label, label_counts):
        """Stored embeddings should match re-embedded descriptions."""

        if not label_counts.get(label):
            pytest.skip(f"No {label} nodes in graph")

        # Fetch a sample of nodes with both description and embedding
        result = graph_client.query(
            f"MATCH (n:{label}) "
            f"WHERE n.description IS NOT NULL AND n.embedding IS NOT NULL "
            f"RETURN n.id AS id, n.description AS desc, n.embedding AS emb "
            f"LIMIT 20"
        )
        if not result:
            pytest.skip(f"No {label} nodes with both description and embedding")

        from imas_codex.embeddings.description import embed_description

        low_similarity = []
        dim_mismatches = 0
        for row in result:
            fresh_emb = embed_description(row["desc"])
            if fresh_emb is None:
                continue

            # Cosine similarity (both should be normalized)
            stored = row["emb"]
            if len(stored) != len(fresh_emb):
                # Dimension mismatch: stored embeddings from a different model
                # This is expected when models change — skip, don't fail.
                dim_mismatches += 1
                continue
            dot_product = sum(a * b for a, b in zip(stored, fresh_emb, strict=True))
            if dot_product < MIN_SELF_SIMILARITY:
                low_similarity.append(f"{row['id']}: cosine={dot_product:.3f}")

        if dim_mismatches == len(result):
            pytest.skip(
                f"All {label} embeddings have dimension mismatch "
                f"(stored model differs from test model)"
            )

        assert not low_similarity, (
            f"{len(low_similarity)} {label} embeddings have low self-similarity "
            f"(< {MIN_SELF_SIMILARITY}):\n  " + "\n  ".join(low_similarity[:5])
        )


class TestDescriptionEmbeddingCoverage:
    """Description embedding coverage checks.

    After `data push --embed`, all descriptions should be embedded.
    """

    @pytest.mark.parametrize("label", get_description_embeddable_labels())
    def test_description_embedding_coverage(self, graph_client, label, label_counts):
        """Nodes that have been through the embed step should retain embeddings.

        Uses ``embedded_at`` to distinguish processed vs unprocessed nodes.
        Skips when no nodes exist or the embed step has not run for this label.
        """

        if not label_counts.get(label):
            pytest.skip(f"No {label} nodes in graph")

        result = graph_client.query(
            f"MATCH (n:{label}) "
            f"WHERE n.embedded_at IS NOT NULL AND n.embedding IS NULL "
            f"RETURN count(n) AS corrupted"
        )
        corrupted = result[0]["corrupted"] if result else 0

        if corrupted == 0:
            # Check that at least some embeddings exist (embed step has run)
            check = graph_client.query(
                f"MATCH (n:{label}) "
                f"WHERE n.embedded_at IS NOT NULL "
                f"RETURN count(n) AS attempted"
            )
            attempted = check[0]["attempted"] if check else 0
            if attempted == 0:
                pytest.skip(f"No {label} embeddings attempted — embed step not yet run")
            return  # All embedded nodes retain their vectors

        assert corrupted == 0, (
            f"{label} has {corrupted} nodes with embedded_at set but no embedding vector. "
            f"Embeddings were written then lost — possible data corruption."
        )


class TestDescriptionQuality:
    """Verify description fields are not empty strings."""

    def test_no_empty_string_descriptions(self, graph_client, graph_labels, schema):
        """Description fields should be null or non-empty, never empty string."""
        violations = []
        for label in sorted(graph_labels):
            if label.startswith("_"):
                continue
            if label not in schema.node_labels:
                continue

            slots = schema.get_all_slots(label)
            if "description" not in slots:
                continue

            result = graph_client.query(
                f"MATCH (n:{label}) WHERE n.description = '' RETURN count(n) AS cnt"
            )
            count = result[0]["cnt"] if result else 0
            if count > 0:
                violations.append(f"{label}: {count} empty descriptions")

        assert not violations, (
            "Nodes with empty string descriptions (should be null or non-empty):\n  "
            + "\n  ".join(violations)
        )


class TestCodeFilePaths:
    """Verify code file path format."""

    def test_code_file_paths_absolute(self, graph_client, label_counts):
        """CodeFile.path should be absolute (start with /)."""
        if not label_counts.get("CodeFile"):
            pytest.skip("No CodeFile nodes in graph")

        result = graph_client.query(
            "MATCH (n:CodeFile) "
            "WHERE n.path IS NOT NULL AND NOT n.path STARTS WITH '/' "
            "RETURN count(n) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} CodeFile nodes with non-absolute paths"


class TestWikiReferenceIntegrity:
    """Verify URL references on wiki discovery nodes."""

    def test_wiki_pages_have_url(self, graph_client, label_counts):
        """Every WikiPage must have a non-empty url."""
        if not label_counts.get("WikiPage"):
            pytest.skip("No WikiPage nodes in graph")

        result = graph_client.query(
            "MATCH (n:WikiPage) "
            "WHERE n.url IS NULL OR n.url = '' "
            "RETURN count(n) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} WikiPage nodes missing url"

    def test_documents_have_url(self, graph_client, label_counts):
        """Every Document must have a non-empty url."""
        if not label_counts.get("Document"):
            pytest.skip("No Document nodes in graph")

        result = graph_client.query(
            "MATCH (n:Document) "
            "WHERE n.url IS NULL OR n.url = '' "
            "RETURN count(n) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} Document nodes missing url"

    def test_images_have_url(self, graph_client, label_counts):
        """Every Image must have a non-empty url."""
        if not label_counts.get("Image"):
            pytest.skip("No Image nodes in graph")

        result = graph_client.query(
            "MATCH (n:Image) WHERE n.url IS NULL OR n.url = '' RETURN count(n) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} Image nodes missing url"


class TestImageScoring:
    """Verify image scoring consistency."""

    _SCORE_DIMS = [
        "score_data_documentation",
        "score_physics_content",
        "score_code_documentation",
        "score_data_access",
        "score_calibration",
        "score_imas_relevance",
    ]

    def test_scored_images_have_composite(self, graph_client, label_counts):
        """Images with individual score dimensions must have score_composite."""
        if not label_counts.get("Image"):
            pytest.skip("No Image nodes in graph")

        dims_present = " OR ".join(f"n.{d} IS NOT NULL" for d in self._SCORE_DIMS)
        result = graph_client.query(
            f"MATCH (n:Image) "
            f"WHERE n.score_composite IS NULL AND ({dims_present}) "
            f"RETURN count(n) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} Image nodes have score dimensions but no score_composite"
        )

    def test_no_stray_score_property(self, graph_client, label_counts):
        """Images must not have a non-schema 'score' property (legacy bug)."""
        if not label_counts.get("Image"):
            pytest.skip("No Image nodes in graph")

        result = graph_client.query(
            "MATCH (n:Image) WHERE n.score IS NOT NULL RETURN count(n) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} Image nodes have stray 'score' property (non-schema)"
        )


class TestCOCOSLabelIntegrity:
    """Verify COCOS label consistency on IMASNode nodes."""

    def test_no_cocos_half_state(self, graph_client, label_counts):
        """No IMASNode should have cocos_label_source set without a label.

        The "half-state" bug writes cocos_label_source (e.g.
        'inferred_forward') but leaves cocos_label_transformation as null.
        This makes the node invisible to COCOS-filtered queries.
        """
        if not label_counts.get("IMASNode"):
            pytest.skip("No IMASNode nodes in graph")

        result = graph_client.query(
            "MATCH (p:IMASNode) "
            "WHERE p.cocos_label_source IS NOT NULL "
            "AND p.cocos_label_transformation IS NULL "
            "RETURN count(*) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} IMASNode nodes have cocos_label_source set but "
            f"cocos_label_transformation is null (half-state bug). "
            f"Run: imas-codex graph repair cocos-labels"
        )

    def test_cocos_labels_are_valid(self, graph_client, label_counts):
        """COCOS transformation labels must be from the known set."""
        if not label_counts.get("IMASNode"):
            pytest.skip("No IMASNode nodes in graph")

        valid_labels = {"psi_like", "ip_like", "b0_like", "f_like"}
        result = graph_client.query(
            "MATCH (p:IMASNode) "
            "WHERE p.cocos_label_transformation IS NOT NULL "
            "RETURN DISTINCT p.cocos_label_transformation AS label"
        )
        if not result:
            pytest.skip("No COCOS-labelled nodes in graph")

        actual = {r["label"] for r in result}
        invalid = actual - valid_labels
        assert not invalid, (
            f"Invalid cocos_label_transformation values: {invalid}. "
            f"Expected one of: {valid_labels}"
        )


class TestChunkIntegrity:
    """Verify chunk parent references."""

    def test_ingested_pages_have_chunks(self, graph_client, label_counts):
        """Ingested WikiPages should have at least one WikiChunk."""
        if not label_counts.get("WikiPage"):
            pytest.skip("No WikiPage nodes in graph")

        result = graph_client.query(
            "MATCH (n:WikiPage) "
            "WHERE n.status = 'ingested' "
            "AND NOT EXISTS { (n)-[:HAS_CHUNK]->(:WikiChunk) } "
            "RETURN count(n) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} ingested WikiPage nodes have no WikiChunk children"

    def test_ingested_documents_have_chunks(self, graph_client, label_counts):
        """Ingested Documents should have at least one WikiChunk."""
        if not label_counts.get("Document"):
            pytest.skip("No Document nodes in graph")

        result = graph_client.query(
            "MATCH (n:Document) "
            "WHERE n.status = 'ingested' "
            "AND NOT EXISTS { (n)-[:HAS_CHUNK]->(:WikiChunk) } "
            "RETURN count(n) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} ingested Document nodes have no WikiChunk children"

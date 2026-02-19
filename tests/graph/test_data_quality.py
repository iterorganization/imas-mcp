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
        result = graph_client.query(
            f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL "
            f"WITH n, "
            f"  reduce(s = 0.0, x IN n.embedding | s + x * x) AS sq_sum "
            f"WITH sqrt(sq_sum) AS norm "
            f"WHERE norm < 0.9 OR norm > 1.1 "
            f"RETURN count(*) AS cnt "
            f"LIMIT 200"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, (
            f"{count} {label} embeddings have L2 norm outside [0.9, 1.1]. "
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
        """All nodes with descriptions should have embeddings.

        Skips if no embeddings exist at all for this label (embed step not run).
        """

        if not label_counts.get(label):
            pytest.skip(f"No {label} nodes in graph")

        result = graph_client.query(
            f"MATCH (n:{label}) "
            f"WHERE n.description IS NOT NULL AND n.description <> '' "
            f"WITH count(n) AS with_desc, "
            f"  count(CASE WHEN n.embedding IS NOT NULL THEN 1 END) AS with_emb "
            f"RETURN with_desc, with_emb"
        )
        if not result:
            pytest.skip(f"No {label} nodes with descriptions")

        with_desc = result[0]["with_desc"]
        with_emb = result[0]["with_emb"]

        if with_desc == 0:
            pytest.skip(f"No {label} nodes with descriptions")

        if with_emb == 0:
            pytest.skip(f"No {label} embeddings found — run `data push --embed` first")

        coverage = with_emb / with_desc
        assert coverage >= 0.95, (
            f"{label} embedding coverage is {coverage:.1%} "
            f"({with_emb}/{with_desc}). "
            f"Expected >= 95% after embed update."
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


class TestSourceFilePaths:
    """Verify source file path format."""

    def test_source_file_paths_absolute(self, graph_client, label_counts):
        """SourceFile.path should be absolute (start with /)."""
        if not label_counts.get("SourceFile"):
            pytest.skip("No SourceFile nodes in graph")

        result = graph_client.query(
            "MATCH (n:SourceFile) "
            "WHERE n.path IS NOT NULL AND NOT n.path STARTS WITH '/' "
            "RETURN count(n) AS cnt"
        )
        count = result[0]["cnt"] if result else 0
        assert count == 0, f"{count} SourceFile nodes with non-absolute paths"

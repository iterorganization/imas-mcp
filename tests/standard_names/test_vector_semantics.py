"""Integration tests for vector-based semantic groupings.

Exercises the three vector indexes that underpin StandardName semantic
search:

* ``standard_name_desc_embedding`` on :class:`StandardName`
* ``standard_name_source_desc_embedding`` on :class:`StandardNameSource`
* ``cluster_label_embedding`` / ``cluster_embedding`` on
  :class:`IMASSemanticCluster`

Tests assert empirical coherence — within-group cosine similarity should
exceed a random baseline, top-K search should surface domain-matching
results, and cluster labels should align with their member names.

Findings are recorded in
``plans/research/standard-names/43-vector-grouping-assertions.md``.

All tests require a live graph and are marked ``integration``; they are
auto-skipped by the top-level conftest when Neo4j is unreachable.
"""

from __future__ import annotations

import math
import os
import random
from typing import Any

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.graph]


# ---------------------------------------------------------------------------
# Session fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _use_production_embedding_model():
    """Use the production embedding model for queries.

    The top-level conftest forces ``all-MiniLM-L6-v2`` for speed on unit
    tests, but stored SN embeddings are produced by the model configured
    in ``pyproject.toml`` (``Qwen/Qwen3-Embedding-0.6B`` at 256-dim).
    Mismatched models produce mismatched vectors — override for this
    module so query embeddings are comparable with stored ones.
    """
    # Read production settings directly from pyproject.toml, bypassing
    # env vars that the top-level conftest sets for other tests.
    from imas_codex.settings import _MODEL_DEFAULTS, _get_section

    section = _get_section("embedding")
    prod_model = section.get("model", _MODEL_DEFAULTS["embedding"])
    prod_location = str(section.get("location") or "local").lower()

    saved = {
        k: os.environ.get(k)
        for k in ("IMAS_CODEX_EMBEDDING_MODEL", "IMAS_CODEX_EMBEDDING_LOCATION")
    }
    os.environ["IMAS_CODEX_EMBEDDING_MODEL"] = prod_model
    os.environ["IMAS_CODEX_EMBEDDING_LOCATION"] = prod_location
    try:
        yield prod_model
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def gc():
    """Module-scoped GraphClient, skipped when graph is unavailable."""
    from imas_codex.graph.client import GraphClient

    try:
        client = GraphClient()
        client.get_stats()
    except Exception as e:  # pragma: no cover - covered by collection hook
        pytest.skip(f"Neo4j not available: {e}")
    yield client
    client.close()


@pytest.fixture(scope="module")
def encoder():
    """Shared encoder using the production model so vectors match stored ones."""
    from imas_codex.embeddings.config import EncoderConfig
    from imas_codex.embeddings.encoder import Encoder

    try:
        return Encoder(EncoderConfig())
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Embedding backend unavailable: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    return dot / (math.sqrt(na) * math.sqrt(nb) + 1e-12)


def _embed(encoder: Any, text: str) -> list[float]:
    vec = encoder.embed_texts([text])[0]
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def _vector_search_sn(gc: Any, embedding: list[float], k: int) -> list[dict]:
    rows = gc.query(
        """
        CALL db.index.vector.queryNodes(
            'standard_name_desc_embedding', $k, $embedding
        ) YIELD node AS sn, score
        WHERE sn.id IS NOT NULL
        RETURN sn.id AS id,
               sn.description AS description,
               sn.physics_domain AS physics_domain,
               sn.tags AS tags,
               score
        ORDER BY score DESC
        """,
        embedding=embedding,
        k=k,
    )
    return [dict(r) for r in rows]


# Five well-known concepts mapped to a rich natural-language query and the
# keyword tokens we expect to see in at least one top-10 result id.
CONCEPT_QUERIES = [
    pytest.param(
        "Electron number density profile measured by Thomson scattering or interferometry",
        ("electron_density", "electron_number_density"),
        id="electron_density",
    ),
    pytest.param(
        "Safety factor q profile on flux surfaces representing pitch of magnetic field lines",
        ("safety_factor",),
        id="safety_factor",
    ),
    pytest.param(
        "Total toroidal plasma current flowing in the tokamak",
        ("plasma_current",),
        id="plasma_current",
    ),
    pytest.param(
        "Current density parallel to the magnetic field",
        ("parallel_current", "parallel_component_of_current"),
        id="parallel_current_density",
    ),
    pytest.param(
        "Major radius coordinate of magnetic axis location",
        ("magnetic_axis",),
        id="magnetic_axis",
    ),
]


# ---------------------------------------------------------------------------
# Test A — StandardName semantic search round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query, expected_tokens", CONCEPT_QUERIES)
def test_sn_semantic_search_surfaces_expected_concept(
    gc, encoder, query: str, expected_tokens: tuple[str, ...]
):
    """Each concept query must surface at least one id containing the
    expected keyword token within the top-10 results."""
    embedding = _embed(encoder, query)
    rows = _vector_search_sn(gc, embedding, k=10)
    assert rows, "Expected non-empty vector search result"

    ids = [r["id"] for r in rows]
    hit = any(any(tok in sid for tok in expected_tokens) for sid in ids)
    assert hit, (
        f"No id in top-10 for {query!r} matched any of {expected_tokens}. Got: {ids}"
    )


@pytest.mark.parametrize("query, expected_tokens", CONCEPT_QUERIES)
def test_sn_semantic_search_domain_coherence(
    gc, encoder, request, query: str, expected_tokens: tuple[str, ...]
):
    """Top-5 results should not be *fully* spread across physics domains.

    A coherent vector search should return at least one dominant domain
    — we require that the modal domain cover ≥ 2 of the top-5 results.
    The SN vocabulary is still incomplete and cross-domain noise is
    expected at the margins, so we do not demand strict single-domain
    coherence.

    Known finding: "plasma_current" results span five different
    physics domains (current_drive, equilibrium, plasma_control,
    transport, waves) because the taxonomy splits related current
    concepts very finely.  See
    plans/research/standard-names/43-vector-grouping-assertions.md.
    """
    from collections import Counter

    # Known-incoherent cases flagged in the research doc.
    known_domain_incoherent = {"plasma_current"}
    case_id = request.node.callspec.id if hasattr(request.node, "callspec") else ""

    embedding = _embed(encoder, query)
    rows = _vector_search_sn(gc, embedding, k=5)
    domains = [r.get("physics_domain") for r in rows if r.get("physics_domain")]
    assert domains, f"No physics_domain values in top-5 for {query!r}"

    counts = Counter(domains)
    modal_count = counts.most_common(1)[0][1]

    if case_id in known_domain_incoherent and modal_count < 2:
        pytest.xfail(
            f"Known physics_domain spread for {case_id!r}: {sorted(set(domains))}"
        )

    assert modal_count >= 2, (
        f"Top-5 for {query!r} have no modal domain (all {len(domains)} "
        f"results in distinct domains): {sorted(set(domains))}"
    )


# ---------------------------------------------------------------------------
# Test B — Cluster membership groups names semantically
# ---------------------------------------------------------------------------


def test_cluster_membership_groups_sns_semantically(gc):
    """Within-cluster SN similarity must exceed random baseline.

    Picks the top-5 largest clusters by embedded-SN member count and
    compares pairwise cosine similarity with a random SN baseline.
    """
    cluster_rows = gc.query(
        """
        MATCH (c:IMASSemanticCluster)<-[:IN_CLUSTER]-(n:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
        WHERE sn.embedding IS NOT NULL
        WITH c, collect(DISTINCT sn.id) AS ids
        WHERE size(ids) >= 5
        RETURN c.id AS cid, size(ids) AS n
        ORDER BY n DESC LIMIT 5
        """
    )
    if not cluster_rows:
        pytest.skip("No clusters with >=5 embedded SN members")

    within_means: list[float] = []
    for row in cluster_rows:
        emb_rows = gc.query(
            """
            MATCH (c:IMASSemanticCluster {id: $cid})<-[:IN_CLUSTER]-
                  (n:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
            WHERE sn.embedding IS NOT NULL
            RETURN DISTINCT sn.id AS id, sn.embedding AS emb
            """,
            cid=row["cid"],
        )
        embs = [r["emb"] for r in emb_rows]
        sims = [
            _cosine(embs[i], embs[j])
            for i in range(len(embs))
            for j in range(i + 1, len(embs))
        ]
        if sims:
            within_means.append(sum(sims) / len(sims))

    assert within_means, "No pairwise similarities could be computed"
    within_overall = sum(within_means) / len(within_means)

    # Random baseline: sample up to 500 SN embeddings and compute 200 random pairs.
    baseline_rows = gc.query(
        """
        MATCH (sn:StandardName) WHERE sn.embedding IS NOT NULL
        RETURN sn.embedding AS emb LIMIT 500
        """
    )
    rng = random.Random(42)
    pairs = 200
    baseline = []
    pool = [r["emb"] for r in baseline_rows]
    for _ in range(pairs):
        a, b = rng.sample(pool, 2)
        baseline.append(_cosine(a, b))
    baseline_mean = sum(baseline) / len(baseline)

    assert within_overall > baseline_mean + 0.1, (
        f"Within-cluster mean ({within_overall:.3f}) does not exceed "
        f"random baseline ({baseline_mean:.3f}) by 0.1"
    )


# ---------------------------------------------------------------------------
# Test C — StandardNameSource embeddings group by producing name
# ---------------------------------------------------------------------------


def test_standard_name_source_embeddings_group_by_producer(gc):
    """StandardNameSource embeddings should cluster by producing SN.

    Skipped when StandardNameSource embeddings are not populated — the
    ``standard_name_source_desc_embedding`` index is online but the
    underlying property is not yet filled by any pipeline step. This
    fact is flagged as an infrastructure health finding.
    """
    producers = gc.query(
        """
        MATCH (sns:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
        WHERE sns.embedding IS NOT NULL
        WITH sn, collect(sns.embedding) AS embs
        WHERE size(embs) >= 3
        RETURN sn.id AS sn_id, embs
        LIMIT 10
        """
    )
    if not producers:
        pytest.skip(
            "StandardNameSource.embedding is unpopulated — no group-by test possible"
        )

    within_means: list[float] = []
    for row in producers:
        embs = row["embs"]
        sims = [
            _cosine(embs[i], embs[j])
            for i in range(len(embs))
            for j in range(i + 1, len(embs))
        ]
        if sims:
            within_means.append(sum(sims) / len(sims))

    within_overall = sum(within_means) / len(within_means)
    assert within_overall > 0.7, (
        f"Mean within-SN source similarity {within_overall:.3f} below 0.7"
    )


# ---------------------------------------------------------------------------
# Test D — MCP ``search_standard_names`` returns coherent results
# ---------------------------------------------------------------------------


def test_mcp_search_standard_names_returns_coherent_results(gc):
    """The public search tool must rank electron-density hits above
    wildly unrelated quantities (e.g. ``magnetic_shear_at_magnetic_axis``)
    for an electron-density query.
    """
    from imas_codex.llm.sn_tools import _search_standard_names

    report = _search_standard_names("electron density profile at pedestal", k=15, gc=gc)
    assert report, "Empty search report"
    # Expect at least one electron-density style hit in the top 15
    assert any(
        tok in report for tok in ("electron_density", "electron_number_density")
    ), f"No electron_density hit in top-15 report:\n{report}"

    # And no wildly-unrelated top-ranked hit
    forbidden = ("magnetic_shear_at_magnetic_axis",)
    for tok in forbidden:
        assert tok not in report, (
            f"Unrelated result {tok!r} returned by electron-density query"
        )


# ---------------------------------------------------------------------------
# Test E — Cluster labels match member names
# ---------------------------------------------------------------------------


def test_cluster_labels_match_member_sns(gc):
    """For clusters with a label_embedding, the label vector should be
    reasonably similar to the embeddings of its member StandardNames
    (per-cluster mean ≥ 0.5).
    """
    rows = gc.query(
        """
        MATCH (c:IMASSemanticCluster)
        WHERE c.label IS NOT NULL AND c.label_embedding IS NOT NULL
        MATCH (c)<-[:IN_CLUSTER]-(n:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
        WHERE sn.embedding IS NOT NULL
        WITH c, collect({id: sn.id, emb: sn.embedding})[..3] AS members
        WHERE size(members) >= 3
        RETURN c.id AS cid, c.label AS label, c.label_embedding AS lemb, members
        LIMIT 30
        """
    )
    if not rows:
        pytest.skip("No clusters with label_embedding and ≥3 embedded members")

    low_coherence: list[tuple[str, float]] = []
    per_cluster_means: list[float] = []
    for row in rows:
        lemb = row["lemb"]
        sims = [_cosine(lemb, m["emb"]) for m in row["members"]]
        mean = sum(sims) / len(sims)
        per_cluster_means.append(mean)
        if mean < 0.5:
            low_coherence.append((row["cid"], mean))

    overall = sum(per_cluster_means) / len(per_cluster_means)
    # Require overall mean above 0.5 and no more than 10% of clusters
    # below that threshold.
    assert overall >= 0.5, f"Overall label↔member similarity {overall:.3f} below 0.5"
    tolerance = max(1, len(rows) // 10)
    assert len(low_coherence) <= tolerance, (
        f"{len(low_coherence)} clusters with mean similarity < 0.5 "
        f"(tolerance {tolerance}): {low_coherence[:5]}"
    )

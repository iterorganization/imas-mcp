"""Unit tests for the ``sn review`` pipeline (Layer 1–3 + budget + CLI dry-run).

All tests run without live graph or embedding services — external calls are
mocked where needed.
"""

from __future__ import annotations

# =============================================================================
# 1. Embedding preflight
# =============================================================================


def test_audit_embedding_preflight():
    """run_embedding_preflight detects missing and stale embeddings."""
    from imas_codex.standard_names.review.audits import (
        compute_review_input_hash,
        run_embedding_preflight,
    )

    names = [
        # Missing embedding
        {
            "id": "electron_temperature",
            "description": "Te",
            "embedding": None,
            "review_input_hash": None,
        },
        # Stale: has an embedding but hash doesn't match current content
        {
            "id": "plasma_current",
            "description": "Ip",
            "embedding": [0.1, 0.2],
            "review_input_hash": "stale_hash",
        },
        # Fresh: embedding present and hash matches current content
        {
            "id": "electron_density",
            "description": "ne",
            "embedding": [0.3, 0.4],
            "review_input_hash": compute_review_input_hash(
                {"id": "electron_density", "description": "ne"}
            ),
        },
    ]

    # The embedding service may be unavailable — the audit should still
    # classify missing/stale entries deterministically.
    report = run_embedding_preflight(names)

    assert report.total == 3
    assert report.missing_count == 1
    assert "electron_temperature" in report.missing_ids
    assert report.stale_count == 1
    assert "plasma_current" in report.stale_ids


# =============================================================================
# 2. Duplicate detection
# =============================================================================


def test_audit_duplicate_candidates():
    """Multi-pass blocking groups lexically-similar names into components.

    The lexical threshold is Jaccard token-overlap > 0.8.  Names must share
    at least 9 of 11 distinct tokens to exceed the threshold.  The two
    ``electron_temperature_profile_core_*`` names each have 9 shared tokens
    and one unique terminal token, giving overlap ≈ 0.818.
    """
    from imas_codex.standard_names.review.audits import run_duplicate_detection

    # These two names share 9 of 11 tokens → Jaccard ≈ 0.818 > 0.8 threshold
    names = [
        {
            "id": "electron_temperature_profile_core_radial_ion_flux_energy_pressure_alpha",
            "description": "Electron temperature profile at core (alpha variant)",
            "unit": "eV",
            "kind": "scalar",
            "physics_domain": "transport",
            "physical_base": "temperature",
        },
        {
            "id": "electron_temperature_profile_core_radial_ion_flux_energy_pressure_beta",
            "description": "Electron temperature profile at core (beta variant)",
            "unit": "eV",
            "kind": "scalar",
            "physics_domain": "transport",
            "physical_base": "temperature",
        },
        {
            "id": "plasma_current",
            "description": "Plasma current",
            "unit": "A",
            "kind": "scalar",
            "physics_domain": "magnetics",
            "physical_base": "current",
        },
    ]

    # Graph search unavailable → lexical-only detection
    components = run_duplicate_detection(names)

    assert len(components) >= 1
    name_a = "electron_temperature_profile_core_radial_ion_flux_energy_pressure_alpha"
    name_b = "electron_temperature_profile_core_radial_ion_flux_energy_pressure_beta"
    temp_component = next((c for c in components if name_a in c.names), None)
    assert temp_component is not None
    assert name_b in temp_component.names


# =============================================================================
# 3. Link integrity
# =============================================================================


def test_audit_link_integrity():
    """Dead links, unresolved status, and missing reverses are all detected."""
    from imas_codex.standard_names.review.audits import run_link_integrity

    names = [
        {
            "id": "electron_temperature",
            "links": ["ion_temperature", "nonexistent_name"],
            "link_status": None,
        },
        {
            "id": "ion_temperature",
            "links": ["electron_temperature"],
            "link_status": None,
        },
        {
            "id": "plasma_current",
            "links": [],
            "link_status": "unresolved",
        },
    ]

    findings = run_link_integrity(names)

    dead_links = [f for f in findings if f.finding_type == "dead_link"]
    unresolved = [f for f in findings if f.finding_type == "unresolved"]

    assert len(dead_links) >= 1
    assert any(f.target == "nonexistent_name" for f in dead_links)
    assert len(unresolved) >= 1
    assert any(f.name_id == "plasma_current" for f in unresolved)


# =============================================================================
# 4. Cluster reconstruction
# =============================================================================


def test_review_cluster_reconstruction():
    """reconstruct_clusters_batch selects dominant cluster per name."""
    from unittest.mock import MagicMock

    from imas_codex.standard_names.review.enrichment import reconstruct_clusters_batch

    gc = MagicMock()
    gc.query = MagicMock(
        return_value=[
            {
                "name_id": "electron_temperature",
                "cluster_id": "c1",
                "cluster_label": "Electron thermal",
                "cluster_description": "Electron thermal properties",
                "scope": "ids",
                "source_count": 3,
            },
            {
                "name_id": "electron_temperature",
                "cluster_id": "c2",
                "cluster_label": "Temperature profiles",
                "cluster_description": "All temperature profiles",
                "scope": "domain",
                "source_count": 5,
            },
            {
                "name_id": "plasma_current",
                "cluster_id": "c3",
                "cluster_label": "Currents",
                "cluster_description": "Plasma currents",
                "scope": "global",
                "source_count": 2,
            },
        ]
    )

    names = [
        {"id": "electron_temperature"},
        {"id": "plasma_current"},
    ]
    result = reconstruct_clusters_batch(names, gc)

    assert "electron_temperature" in result
    assert "plasma_current" in result
    # Dominant cluster selected (not None)
    assert result["electron_temperature"] is not None
    assert result["plasma_current"] is not None
    # plasma_current maps to c3
    assert result["plasma_current"]["cluster_id"] == "c3"


# =============================================================================
# 5. Batch token budget
# =============================================================================


def test_review_batch_token_budget():
    """group_into_review_batches splits on token budget, not just item count."""
    from imas_codex.standard_names.review.enrichment import (
        estimate_name_tokens,
        group_into_review_batches,
    )

    names = []
    for i in range(30):
        doc_length = 2000 if i < 5 else 100  # First 5 have very long docs
        names.append(
            {
                "id": f"name_{i}",
                "description": f"Description {i}",
                "documentation": "x" * doc_length,
                "unit": "eV",
                "kind": "scalar",
                "physics_domain": "transport",
            }
        )

    clusters = {n["id"]: None for n in names}

    batches = group_into_review_batches(
        names, clusters, max_batch_size=25, token_budget=4000
    )

    # Token budget forces multiple batches despite ≤25 hard cap
    assert len(batches) > 1

    for batch in batches:
        assert len(batch["names"]) <= 25

    # Sanity-check token estimator
    long_name = {"description": "Short", "documentation": "x" * 2000}
    short_name = {"description": "Short", "documentation": "x" * 100}
    assert estimate_name_tokens(long_name) > estimate_name_tokens(short_name)


# =============================================================================
# 6. Neighborhood enrichment
# =============================================================================


def test_review_neighborhood_enrichment():
    """build_neighborhood_context excludes batch members from results."""
    from unittest.mock import patch

    from imas_codex.standard_names.review.enrichment import build_neighborhood_context

    batch = {
        "names": [
            {"id": "electron_temperature", "description": "Electron temperature"},
            {"id": "ion_temperature", "description": "Ion temperature"},
        ],
        "cluster": {"cluster_label": "Temperature profiles"},
        "group_key": "temp/eV",
    }

    all_names = batch["names"] + [
        {
            "id": "electron_density",
            "description": "Electron density",
            "kind": "scalar",
            "unit": "m^-3",
            "review_tier": "good",
        },
    ]

    mock_results = [
        {
            "id": "electron_temperature",
            "description": "Electron temperature",
            "kind": "scalar",
            "unit": "eV",
            "score": 0.99,
        },
        {
            "id": "electron_density",
            "description": "Electron density",
            "kind": "scalar",
            "unit": "m^-3",
            "score": 0.85,
        },
        {
            "id": "plasma_pressure",
            "description": "Plasma pressure",
            "kind": "scalar",
            "unit": "Pa",
            "score": 0.70,
        },
    ]

    with patch(
        "imas_codex.standard_names.search.search_similar_names",
        return_value=mock_results,
    ):
        neighborhood = build_neighborhood_context(batch, all_names, k=10)

    neighbor_ids = {n["id"] for n in neighborhood}
    # Batch members must be excluded
    assert "electron_temperature" not in neighbor_ids
    assert "ion_temperature" not in neighbor_ids
    # At least one non-member should be present
    assert "electron_density" in neighbor_ids or "plasma_pressure" in neighbor_ids


# =============================================================================
# 7. Budget manager
# =============================================================================


def test_review_budget_manager():
    """ReviewBudgetManager tracks reservations, reconciliations, and exhaustion."""
    from imas_codex.standard_names.review.budget import ReviewBudgetManager

    mgr = ReviewBudgetManager(total_budget=1.0)

    assert mgr.remaining == 1.0
    assert not mgr.exhausted()

    # Reserve with internal 1.3× headroom
    assert mgr.reserve(0.5)  # Internally reserves 0.65
    assert mgr.remaining < 1.0

    # Reconcile — return unused portion
    mgr.reconcile(reserved=0.65, actual=0.3)
    # Unused = 0.35 returned to the pool

    # A reservation larger than the remaining budget must fail
    remaining_before = mgr.remaining
    assert not mgr.reserve(remaining_before * 2)

    summary = mgr.summary
    assert summary["total_budget"] == 1.0
    assert summary["batch_count"] == 1  # Only the one successful reservation
    assert summary["total_actual"] == 0.3


def test_review_budget_manager_reconcile_in_finally():
    """Budget reconciliation in a finally block fully restores the pool."""
    from imas_codex.standard_names.review.budget import ReviewBudgetManager

    mgr = ReviewBudgetManager(total_budget=1.0)

    reserved = 0.0
    try:
        assert mgr.reserve(0.3)
        reserved = 0.3 * 1.3
        raise ValueError("Simulated failure")
    except ValueError:
        pass
    finally:
        mgr.reconcile(reserved, actual=0.0)

    # Full reconciliation → budget fully restored
    assert abs(mgr.remaining - 1.0) < 1e-9


# =============================================================================
# 8. Consolidation — convention drift
# =============================================================================


def test_review_consolidation_drift():
    """detect_convention_drift finds mixed position suffixes in a domain."""
    from imas_codex.standard_names.review.consolidation import detect_convention_drift

    names = [
        {
            "id": "temperature_at_magnetic_axis",
            "physics_domain": "equilibrium",
            "physical_base": "temperature",
            "documentation": "x" * 100,
        },
        {
            "id": "temperature_on_axis",
            "physics_domain": "equilibrium",
            "physical_base": "temperature",
            "documentation": "x" * 100,
        },
        {
            "id": "density_reconstructed",
            "physics_domain": "equilibrium",
            "physical_base": "density",
            "documentation": "x" * 100,
        },
        {
            "id": "pressure_equilibrium",
            "physics_domain": "equilibrium",
            "physical_base": "pressure",
            "documentation": "x" * 100,
        },
    ]

    warnings = detect_convention_drift(names)

    position_warnings = [w for w in warnings if w.drift_type == "mixed_position"]
    assert len(position_warnings) >= 1
    assert position_warnings[0].physics_domain == "equilibrium"


# =============================================================================
# 9. Staleness policy — hash-based invalidation
# =============================================================================


def test_review_staleness_policy():
    """compute_review_input_hash is deterministic and content-sensitive."""
    from imas_codex.standard_names.review.audits import compute_review_input_hash

    name = {
        "id": "electron_temperature",
        "description": "Electron temperature profile",
        "documentation": "The electron temperature $T_e$.",
        "kind": "scalar",
        "unit": "eV",
        "tags": ["core_profiles"],
        "links": ["ion_temperature"],
        "physical_base": "temperature",
        "subject": "electron",
        "cocos_transformation_type": None,
        "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
    }

    hash1 = compute_review_input_hash(name)

    # Same content → same hash
    hash2 = compute_review_input_hash(name)
    assert hash1 == hash2

    # Changed description → different hash
    modified = dict(name)
    modified["description"] = "Updated electron temperature profile"
    hash3 = compute_review_input_hash(modified)
    assert hash3 != hash1

    # Changed tags → different hash
    modified2 = dict(name)
    modified2["tags"] = ["core_profiles", "kinetics"]
    hash4 = compute_review_input_hash(modified2)
    assert hash4 != hash1

    # Tag order must not matter (tags are sorted)
    modified3 = dict(name)
    modified3["tags"] = ["kinetics", "core_profiles"]
    hash5 = compute_review_input_hash(modified3)
    assert hash5 == hash4


# =============================================================================
# 10. CLI dry-run
# =============================================================================


def test_review_cli_dry_run():
    """sn review --dry-run runs audits and shows batch plan without LLM calls."""
    from unittest.mock import MagicMock, patch

    from click.testing import CliRunner

    from imas_codex.cli.sn import sn

    mock_names = [
        {
            "id": "electron_temperature",
            "description": "Te",
            "pipeline_status": "drafted",
            "documentation": "Electron temperature.",
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
            "cocos_transformation_type": None,
            "physics_domain": "transport",
            "reviewer_score": None,
            "review_input_hash": None,
            "embedding": [0.1, 0.2],
            "review_tier": None,
            "link_status": None,
            "source_types": ["dd"],
            "geometric_base": None,
            "reviewer_comments": None,
            "source_id": "core_profiles/profiles_1d/electrons/temperature",
            "generated_at": None,
            "reviewed_at": None,
        },
    ]

    mock_gc = MagicMock()
    mock_gc.query = MagicMock(return_value=mock_names)
    mock_gc.__enter__ = MagicMock(return_value=mock_gc)
    mock_gc.__exit__ = MagicMock(return_value=False)

    with patch("imas_codex.graph.client.GraphClient", return_value=mock_gc):
        runner = CliRunner()
        result = runner.invoke(sn, ["review", "--dry-run"])

    # Should either succeed or gracefully report no names (not a hard crash)
    assert result.exit_code == 0 or "No standard names" in (result.output or "")


# =============================================================================
# Additional: hash determinism
# =============================================================================


def test_compute_review_input_hash_deterministic():
    """Hash is deterministic across calls and handles missing fields."""
    from imas_codex.standard_names.review.audits import compute_review_input_hash

    name = {"id": "test_name", "description": "A test", "kind": "scalar"}

    # Multiple calls → same result
    assert compute_review_input_hash(name) == compute_review_input_hash(name)

    # Missing fields → consistent (empty-string default)
    name_minimal = {"id": "test_name"}
    hash_minimal = compute_review_input_hash(name_minimal)
    assert isinstance(hash_minimal, str)
    assert len(hash_minimal) == 64  # SHA-256 hex digest


# =============================================================================
# 11. 1:1 scoring invariant — _match_reviews_to_entries
# =============================================================================


def test_match_reviews_all_matched():
    """When LLM returns reviews for all entries, unmatched list is empty."""
    import logging

    from imas_codex.standard_names.models import (
        StandardNameQualityReview,
        StandardNameQualityScore,
        StandardNameReviewVerdict,
    )
    from imas_codex.standard_names.review.pipeline import _match_reviews_to_entries

    wlog = logging.getLogger("test")

    names = [
        {"id": "electron_temperature", "source_id": "core/te"},
        {"id": "ion_temperature", "source_id": "core/ti"},
    ]
    reviews = [
        StandardNameQualityReview(
            source_id="core/te",
            standard_name="electron_temperature",
            scores=StandardNameQualityScore(
                grammar=18,
                semantic=18,
                documentation=18,
                convention=18,
                completeness=18,
                compliance=18,
            ),
            verdict=StandardNameReviewVerdict.accept,
            reasoning="Good name.",
        ),
        StandardNameQualityReview(
            source_id="core/ti",
            standard_name="ion_temperature",
            scores=StandardNameQualityScore(
                grammar=16,
                semantic=16,
                documentation=16,
                convention=16,
                completeness=16,
                compliance=16,
            ),
            verdict=StandardNameReviewVerdict.accept,
            reasoning="Good name.",
        ),
    ]

    scored, unmatched, revised = _match_reviews_to_entries(reviews, names, wlog)

    assert len(scored) == 2
    assert len(unmatched) == 0
    assert revised == 0
    # All entries have scores
    for entry in scored:
        assert entry["reviewer_score"] is not None
        assert entry["review_tier"] is not None


def test_match_reviews_partial_response():
    """When LLM omits entries, they appear in the unmatched list."""
    import logging

    from imas_codex.standard_names.models import (
        StandardNameQualityReview,
        StandardNameQualityScore,
        StandardNameReviewVerdict,
    )
    from imas_codex.standard_names.review.pipeline import _match_reviews_to_entries

    wlog = logging.getLogger("test")

    names = [
        {"id": "electron_temperature", "source_id": "core/te"},
        {"id": "ion_temperature", "source_id": "core/ti"},
        {"id": "plasma_current", "source_id": "mag/ip"},
    ]
    # LLM only returns 1 of 3 reviews (simulates truncation)
    reviews = [
        StandardNameQualityReview(
            source_id="core/te",
            standard_name="electron_temperature",
            scores=StandardNameQualityScore(
                grammar=18,
                semantic=18,
                documentation=18,
                convention=18,
                completeness=18,
                compliance=18,
            ),
            verdict=StandardNameReviewVerdict.accept,
            reasoning="Good name.",
        ),
    ]

    scored, unmatched, revised = _match_reviews_to_entries(reviews, names, wlog)

    assert len(scored) == 1
    assert len(unmatched) == 2
    assert revised == 0

    unmatched_ids = {e["id"] for e in unmatched}
    assert "ion_temperature" in unmatched_ids
    assert "plasma_current" in unmatched_ids

    # Scored entry has a score, unmatched do NOT
    assert scored[0]["reviewer_score"] is not None
    for entry in unmatched:
        assert entry.get("reviewer_score") is None


def test_match_reviews_fallback_to_standard_name():
    """Matching falls back to standard_name when source_id is None."""
    import logging

    from imas_codex.standard_names.models import (
        StandardNameQualityReview,
        StandardNameQualityScore,
        StandardNameReviewVerdict,
    )
    from imas_codex.standard_names.review.pipeline import _match_reviews_to_entries

    wlog = logging.getLogger("test")

    # source_id is None — common for many StandardName nodes
    names = [
        {"id": "electron_temperature", "source_id": None},
    ]
    reviews = [
        StandardNameQualityReview(
            source_id="None",  # LLM renders None as string
            standard_name="electron_temperature",
            scores=StandardNameQualityScore(
                grammar=18,
                semantic=18,
                documentation=18,
                convention=18,
                completeness=18,
                compliance=18,
            ),
            verdict=StandardNameReviewVerdict.accept,
            reasoning="Good name.",
        ),
    ]

    scored, unmatched, revised = _match_reviews_to_entries(reviews, names, wlog)

    # Should match via standard_name fallback
    assert len(scored) == 1
    assert len(unmatched) == 0


# =============================================================================
# 12. Retry logic — _review_single_batch with incomplete response
# =============================================================================


def test_review_single_batch_retries_unmatched():
    """_review_single_batch retries unmatched entries and reports unscored count."""
    import asyncio
    import logging
    from unittest.mock import AsyncMock, patch

    from imas_codex.standard_names.models import (
        StandardNameQualityReview,
        StandardNameQualityReviewBatch,
        StandardNameQualityScore,
        StandardNameReviewVerdict,
    )
    from imas_codex.standard_names.review.pipeline import _review_single_batch

    wlog = logging.getLogger("test")

    names = [
        {"id": "electron_temperature", "source_id": "core/te"},
        {"id": "ion_temperature", "source_id": "core/ti"},
        {"id": "plasma_current", "source_id": "mag/ip"},
    ]

    score_18 = StandardNameQualityScore(
        grammar=18,
        semantic=18,
        documentation=18,
        convention=18,
        completeness=18,
        compliance=18,
    )

    # First call: LLM returns only 1 of 3 reviews
    first_response = StandardNameQualityReviewBatch(
        reviews=[
            StandardNameQualityReview(
                source_id="core/te",
                standard_name="electron_temperature",
                scores=score_18,
                verdict=StandardNameReviewVerdict.accept,
                reasoning="Good.",
            ),
        ]
    )
    # Retry call: LLM returns 1 of 2 remaining (1 still missing)
    retry_response = StandardNameQualityReviewBatch(
        reviews=[
            StandardNameQualityReview(
                source_id="core/ti",
                standard_name="ion_temperature",
                scores=score_18,
                verdict=StandardNameReviewVerdict.accept,
                reasoning="Good.",
            ),
        ]
    )

    call_count = 0

    async def mock_acall_llm_structured(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return first_response, 0.01, 100
        return retry_response, 0.005, 50

    with (
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            return_value="mocked prompt",
        ),
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            side_effect=mock_acall_llm_structured,
        ),
    ):
        result = asyncio.run(
            _review_single_batch(
                names=names,
                model="test-model",
                grammar_enums={},
                compose_ctx={},
                batch_context="test",
                neighborhood=[],
                audit_findings=[],
                wlog=wlog,
            )
        )

    # 2 scored (first + retry), 1 unscored (plasma_current)
    assert len(result["_items"]) == 2
    assert result["_unscored"] == 1
    # Costs accumulated from both calls
    assert result["_cost"] == 0.015
    assert result["_tokens"] == 150
    # LLM was called twice (initial + retry)
    assert call_count == 2

    # All scored items have reviewer_score set
    for item in result["_items"]:
        assert item["reviewer_score"] is not None


def test_review_single_batch_no_retry_when_all_matched():
    """No retry when all entries are matched on first try."""
    import asyncio
    import logging
    from unittest.mock import patch

    from imas_codex.standard_names.models import (
        StandardNameQualityReview,
        StandardNameQualityReviewBatch,
        StandardNameQualityScore,
        StandardNameReviewVerdict,
    )
    from imas_codex.standard_names.review.pipeline import _review_single_batch

    wlog = logging.getLogger("test")

    names = [
        {"id": "electron_temperature", "source_id": "core/te"},
    ]

    response = StandardNameQualityReviewBatch(
        reviews=[
            StandardNameQualityReview(
                source_id="core/te",
                standard_name="electron_temperature",
                scores=StandardNameQualityScore(
                    grammar=18,
                    semantic=18,
                    documentation=18,
                    convention=18,
                    completeness=18,
                    compliance=18,
                ),
                verdict=StandardNameReviewVerdict.accept,
                reasoning="Good.",
            ),
        ]
    )

    call_count = 0

    async def mock_acall_llm_structured(**kwargs):
        nonlocal call_count
        call_count += 1
        return response, 0.01, 100

    with (
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            return_value="mocked prompt",
        ),
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            side_effect=mock_acall_llm_structured,
        ),
    ):
        result = asyncio.run(
            _review_single_batch(
                names=names,
                model="test-model",
                grammar_enums={},
                compose_ctx={},
                batch_context="test",
                neighborhood=[],
                audit_findings=[],
                wlog=wlog,
            )
        )

    assert len(result["_items"]) == 1
    assert result["_unscored"] == 0
    # Only one LLM call — no retry needed
    assert call_count == 1


# =============================================================================
# 13. Consolidation — accurate scored/unscored reporting
# =============================================================================


def test_build_summary_report_scored_unscored_split():
    """build_summary_report separates scored and unscored entries correctly."""
    from imas_codex.standard_names.review.consolidation import build_summary_report

    all_names = [{"id": f"name_{i}"} for i in range(100)]

    reviewed = [
        # 3 scored entries with tiers
        {
            "id": "electron_temperature",
            "reviewer_score": 0.9,
            "review_tier": "outstanding",
        },
        {
            "id": "ion_temperature",
            "reviewer_score": 0.7,
            "review_tier": "good",
        },
        {
            "id": "plasma_current",
            "reviewer_score": 0.45,
            "review_tier": "inadequate",
        },
        # 2 unscored entries (the bug case)
        {
            "id": "electron_density",
            "reviewer_score": None,
            "review_tier": None,
        },
        {
            "id": "plasma_pressure",
            "reviewer_score": None,
            "review_tier": None,
        },
    ]

    summary = build_summary_report(
        all_names=all_names,
        reviewed_names=reviewed,
        duplicate_reports=[],
        drift_warnings=[],
        outliers=[],
        total_cost=0.5,
    )

    # Total includes all entries
    assert summary.total_reviewed == 5
    # Scored/unscored split is accurate
    assert summary.total_scored == 3
    assert summary.total_unscored == 2
    # Tier distribution only counts scored entries (no "unknown" tier from unscored)
    assert "unknown" not in summary.tier_distribution
    assert summary.tier_distribution.get("outstanding") == 1
    assert summary.tier_distribution.get("good") == 1
    assert summary.tier_distribution.get("inadequate") == 1
    assert sum(summary.tier_distribution.values()) == 3
    # Coverage based on scored entries only
    assert summary.coverage_pct == 3.0  # 3/100 * 100
    # Lowest scorers only from scored entries
    assert len(summary.lowest_scorers) == 3
    assert all(s["score"] is not None for s in summary.lowest_scorers)


def test_build_summary_report_all_scored():
    """When all entries are scored, total_unscored is 0."""
    from imas_codex.standard_names.review.consolidation import build_summary_report

    all_names = [{"id": f"name_{i}"} for i in range(10)]

    reviewed = [
        {
            "id": "electron_temperature",
            "reviewer_score": 0.9,
            "review_tier": "outstanding",
        },
        {
            "id": "ion_temperature",
            "reviewer_score": 0.7,
            "review_tier": "good",
        },
    ]

    summary = build_summary_report(
        all_names=all_names,
        reviewed_names=reviewed,
        duplicate_reports=[],
        drift_warnings=[],
        outliers=[],
    )

    assert summary.total_reviewed == 2
    assert summary.total_scored == 2
    assert summary.total_unscored == 0
    assert sum(summary.tier_distribution.values()) == summary.total_scored

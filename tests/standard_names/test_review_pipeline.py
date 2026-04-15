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
            "review_status": "drafted",
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

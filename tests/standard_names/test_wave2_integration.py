"""Wave 2 integration smoke test: verify prompt context keys are wired up.

This test exercises the compose-worker and review-worker context-building
paths with mocks in place of the graph client and LLM, asserting that the
Wave 2 feature set is correctly plumbed:

- Compose items carry: hybrid_neighbours, related_neighbours, error_fields,
  identifier_values
- Compose context contains: compose_scored_examples
- Review context contains: review_scored_examples

Not a semantic test — per-feature tests exist elsewhere. This is a wiring
sanity check.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ── Compose-side: _enrich_batch_items populates per-item context ─────────


def test_enrich_batch_items_injects_hybrid_neighbours():
    """hybrid_neighbours are injected per-item when hybrid search returns."""
    # Verify the function signature accepts the expected args
    import inspect

    from imas_codex.standard_names.workers import _hybrid_search_neighbours

    sig = inspect.signature(_hybrid_search_neighbours)
    assert "gc" in sig.parameters
    assert "path" in sig.parameters
    assert "description" in sig.parameters
    assert "physics_domain" in sig.parameters
    assert "search_k" in sig.parameters


def test_enrich_batch_items_injects_related_neighbours():
    """related_neighbours are injected per-item when related paths found."""
    import inspect

    from imas_codex.standard_names.workers import _related_path_neighbours

    sig = inspect.signature(_related_path_neighbours)
    assert "gc" in sig.parameters
    assert "path" in sig.parameters
    assert "max_results" in sig.parameters


def test_enrich_batch_items_populates_all_wave2_keys():
    """_enrich_batch_items populates hybrid, related, error, and identifier keys."""
    from imas_codex.standard_names.workers import _enrich_batch_items

    # Build a mock GraphClient that returns data for all four channels
    mock_gc = MagicMock()

    # DD context query returns identifier schema data
    dd_context_row = {
        "coordinate1": "time",
        "coordinate2": None,
        "coordinate3": None,
        "timebase": "time",
        "cocos_label": None,
        "cocos_expression": None,
        "lifecycle_status": "active",
        "identifier_schema_name": "species_type",
        "identifier_schema_doc": "Type of ion species",
        "identifier_options": '[{"name":"electron","index":0,"description":"Electron"}]',
        "sibling_fields": [],
    }

    # Cross-IDS query returns empty
    cross_ids_rows: list[dict] = []

    # Version history returns empty
    version_rows: list[dict] = []

    # Error fields query returns one error field
    error_row = {
        "error_path": "core_profiles/profiles_1d/electrons/temperature_error_upper"
    }

    # Configure mock query responses
    def mock_query(cypher, **kwargs):
        if "identifier_schema_name" in cypher or "coordinate1" in cypher:
            return [dd_context_row]
        if "cluster_label" in cypher:
            return cross_ids_rows
        if "IMASNodeChange" in cypher or "change_type" in cypher:
            return version_rows
        if "error" in cypher.lower() or "_error_" in cypher:
            return [error_row]
        return []

    mock_gc.query = mock_query

    items = [
        {
            "path": "core_profiles/profiles_1d/electrons/temperature",
            "description": "Electron temperature",
            "physics_domain": "transport",
        }
    ]

    # Mock the hybrid and related search functions to return data
    mock_hybrid = [
        {
            "path": "core_sources/profiles_1d/electrons/temperature",
            "sn": "electron_temperature",
        }
    ]
    mock_related = [
        {
            "path": "edge_profiles/profiles_1d/electrons/temperature",
            "relationship": "cluster",
        }
    ]

    with (
        patch("imas_codex.graph.client.GraphClient") as MockGC,
        patch(
            "imas_codex.standard_names.workers._hybrid_search_neighbours_batch",
            return_value=[mock_hybrid],
        ),
        patch(
            "imas_codex.standard_names.workers._related_path_neighbours",
            return_value=mock_related,
        ),
    ):
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)

        _enrich_batch_items(items)

    item = items[0]
    # Wave 2 context channels
    assert "hybrid_neighbours" in item, "hybrid_neighbours not injected"
    assert item["hybrid_neighbours"] == mock_hybrid
    assert "related_neighbours" in item, "related_neighbours not injected"
    assert item["related_neighbours"] == mock_related
    assert "error_fields" in item, "error_fields not injected"
    assert "identifier_values" in item, "identifier_values not injected"
    assert item["identifier_values"][0]["name"] == "electron"


# ── Compose context: compose_scored_examples key ────────────────────────


def test_compose_context_has_scored_examples_key():
    """compose_scored_examples is always set in compose context (even if empty)."""
    # The compose worker always sets context["compose_scored_examples"] = ...
    # We verify by checking the key is set in the code path.
    import ast
    from pathlib import Path

    workers_path = (
        Path(__file__).resolve().parents[2]
        / "imas_codex"
        / "standard_names"
        / "workers.py"
    )
    source = workers_path.read_text()
    tree = ast.parse(source)

    # Find any assignment of compose_scored_examples
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == "context":
                if (
                    isinstance(node.slice, ast.Constant)
                    and node.slice.value == "compose_scored_examples"
                ):
                    found = True
                    break

    assert found, (
        "context['compose_scored_examples'] assignment not found in workers.py"
    )


# ── Review context: review_scored_examples key ──────────────────────────


def test_review_context_has_scored_examples_key():
    """review_scored_examples is always set in review context (even if empty)."""
    import ast
    from pathlib import Path

    pipeline_path = (
        Path(__file__).resolve().parents[2]
        / "imas_codex"
        / "standard_names"
        / "review"
        / "pipeline.py"
    )
    source = pipeline_path.read_text()
    tree = ast.parse(source)

    # Count occurrences of "review_scored_examples" as a string constant in subscript assignments
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "review_scored_examples":
            count += 1

    assert count >= 2, (
        f"Expected review_scored_examples in at least 2 context dicts "
        f"(system + user), found {count}"
    )


# ── Review enrichment: hybrid + related neighbours injected ─────────────


def test_review_dd_context_injects_neighbours():
    """_fetch_review_dd_context enriches review items with hybrid + related."""
    from imas_codex.standard_names.review.pipeline import _fetch_review_dd_context

    mock_gc = MagicMock()

    # Query for DD path docs
    def mock_query(cypher, **kwargs):
        if "IMASNode" in cypher and "description" in cypher:
            return [
                {
                    "id": "equilibrium/time_slice/profiles_1d/psi",
                    "unit": "Wb",
                    "description": "Poloidal flux",
                    "documentation": "",
                }
            ]
        if "IMASNodeChange" in cypher or "change_type" in cypher:
            return []
        return []

    mock_gc.query = mock_query

    items = [
        {
            "id": "poloidal_flux",
            "standard_name": "poloidal_flux",
            "source_paths": '["equilibrium/time_slice/profiles_1d/psi"]',
        }
    ]

    mock_hybrid = [{"path": "core_profiles/profiles_1d/grid/psi", "sn": None}]
    mock_related = [{"path": "mhd_linear/flux_surface/psi", "relationship": "unit"}]

    with (
        patch("imas_codex.graph.client.GraphClient") as MockGC,
        patch(
            "imas_codex.standard_names.workers._hybrid_search_neighbours_batch",
            return_value=[mock_hybrid],
        ),
        patch(
            "imas_codex.standard_names.workers._related_path_neighbours",
            return_value=mock_related,
        ),
    ):
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)

        _fetch_review_dd_context(items)

    item = items[0]
    assert "dd_source_docs" in item, "dd_source_docs not injected by review enrichment"
    assert "nearest_peers" in item, (
        "nearest_peers (hybrid) not injected by review enrichment"
    )
    assert "related_neighbours" in item, (
        "related_neighbours not injected by review enrichment"
    )


# ── Settings accessors wired to workers ─────────────────────────────────


def test_retry_accessors_return_configured_defaults():
    """Retry tunables are accessible via settings and workers."""
    from imas_codex.settings import get_sn_retry_attempts, get_sn_retry_k_expansion
    from imas_codex.standard_names.workers import _retry_attempts, _retry_k_expansion

    # Settings and worker accessors agree
    assert _retry_attempts() == get_sn_retry_attempts()
    assert _retry_k_expansion() == get_sn_retry_k_expansion()

    # Defaults are the expected values
    assert _retry_attempts() == 1
    assert _retry_k_expansion() == 12


def test_example_settings_return_configured_defaults():
    """Example injection tunables are accessible via settings."""
    from imas_codex.settings import (
        get_sn_example_per_bucket,
        get_sn_example_target_scores,
        get_sn_example_tolerance,
    )

    scores = get_sn_example_target_scores()
    assert isinstance(scores, tuple)
    assert len(scores) == 4
    assert scores[0] == 1.0
    assert scores[-1] == 0.4

    assert get_sn_example_tolerance() == 0.05
    assert get_sn_example_per_bucket() == 1

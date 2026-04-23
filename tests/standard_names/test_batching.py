"""Tests for the batching module.

Covers:
- Token estimation heuristics
- Pre-flight token check for extraction batches
- Pre-flight token check for enrich batches
- Grouping strategies produce expected partitions
- pyproject configuration reading
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.batching import (
    DEFAULT_MAX_TOKENS,
    estimate_batch_tokens,
    estimate_enrich_batch_tokens,
    estimate_tokens,
    get_enrich_batch_config,
    get_generate_batch_config,
    pre_flight_enrich_token_check,
    pre_flight_token_check,
)
from imas_codex.standard_names.enrichment import (
    group_by_concept_and_unit,
    group_for_name_only,
)
from imas_codex.standard_names.sources.base import ExtractionBatch

# ============================================================================
# Helpers
# ============================================================================


def _make_item(
    path: str = "core_profiles/profiles_1d/electrons/temperature",
    unit: str = "eV",
    description: str = "Electron temperature",
    documentation: str = "",
    physics_domain: str = "transport",
    ids_name: str = "core_profiles",
    cluster_id: str | None = "c1",
    cluster_label: str | None = "Electron temperature",
    cluster_description: str | None = "Temperature of electrons",
    cluster_scope: str | None = "global",
    parent_path: str | None = "core_profiles/profiles_1d/electrons",
    parent_description: str | None = "Electrons node",
) -> dict:
    """Build an enriched item dict for testing."""
    return {
        "path": path,
        "unit": unit,
        "description": description,
        "documentation": documentation,
        "physics_domain": physics_domain,
        "ids_name": ids_name,
        "grouping_cluster_id": cluster_id,
        "grouping_cluster_label": cluster_label,
        "primary_cluster_id": cluster_id,
        "primary_cluster_label": cluster_label,
        "primary_cluster_description": cluster_description,
        "cluster_scope": cluster_scope,
        "parent_path": parent_path,
        "parent_description": parent_description,
        "cluster_siblings": [],
        "all_clusters": [],
        "data_type": "FLT_1D",
    }


def _make_batch(
    n_items: int = 5,
    context: str = "Test context",
    mode: str = "default",
    **item_overrides,
) -> ExtractionBatch:
    """Build an ExtractionBatch with *n_items* items."""
    items = [
        _make_item(
            path=f"test/path_{i}",
            description=f"Description for path {i}" * 10,
            **item_overrides,
        )
        for i in range(n_items)
    ]
    return ExtractionBatch(
        source="dd",
        group_key="test/group",
        items=items,
        context=context,
        mode=mode,
    )


def _make_enrich_item(
    name: str = "electron_temperature",
    description: str = "Electron temperature profile",
    documentation: str = "The electron temperature T_e.",
    unit: str = "eV",
    kind: str = "scalar",
    physics_domain: str = "transport",
    context: str | None = None,
) -> dict:
    """Build an enrich item dict."""
    item: dict = {
        "name": name,
        "description": description,
        "documentation": documentation,
        "unit": unit,
        "kind": kind,
        "physics_domain": physics_domain,
        "tags": ["core_profiles"],
        "links": ["ion_temperature"],
        "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
    }
    if context:
        item["context"] = context
    return item


def _make_enrich_batch(
    n_items: int = 5,
    token: str | None = "claim-123",
    **item_overrides,
) -> dict:
    """Build an enrich batch dict with *n_items* items."""
    items = [
        _make_enrich_item(
            name=f"test_name_{i}",
            description=f"Description {i}" * 20,
            **item_overrides,
        )
        for i in range(n_items)
    ]
    return {
        "items": items,
        "claim_token": token,
        "batch_index": 0,
    }


# ============================================================================
# Token estimation
# ============================================================================


class TestEstimateTokens:
    """Tests for the token estimation heuristic."""

    def test_empty_string(self):
        assert estimate_tokens("") == 1  # min 1

    def test_short_text(self):
        # "hello" = 5 chars → 5 // 4 = 1
        assert estimate_tokens("hello") == 1

    def test_longer_text(self):
        text = "a" * 400
        assert estimate_tokens(text) == 100  # 400 / 4

    def test_proportional(self):
        short = estimate_tokens("short text")
        long = estimate_tokens("short text" * 100)
        assert long > short


class TestEstimateBatchTokens:
    """Tests for extraction batch token estimation."""

    def test_empty_batch(self):
        batch = ExtractionBatch(
            source="dd",
            group_key="test",
            items=[],
            context="",
        )
        assert estimate_batch_tokens(batch) >= 0

    def test_includes_context_and_items(self):
        batch = _make_batch(n_items=3, context="System context " * 50)
        tokens = estimate_batch_tokens(batch)
        assert tokens > 0
        # Should include context + item fields
        context_only = estimate_tokens("System context " * 50)
        assert tokens > context_only

    def test_more_items_more_tokens(self):
        small = estimate_batch_tokens(_make_batch(n_items=2))
        large = estimate_batch_tokens(_make_batch(n_items=20))
        assert large > small

    def test_siblings_contribute(self):
        item = _make_item()
        item["cluster_siblings"] = [
            {"path": f"sibling/{i}", "description": f"Sibling desc {i}" * 10}
            for i in range(5)
        ]
        batch = ExtractionBatch(
            source="dd",
            group_key="test",
            items=[item],
            context="ctx",
        )
        batch_no_sibs = ExtractionBatch(
            source="dd",
            group_key="test",
            items=[_make_item()],
            context="ctx",
        )
        assert estimate_batch_tokens(batch) > estimate_batch_tokens(batch_no_sibs)


class TestEstimateEnrichBatchTokens:
    """Tests for enrich batch token estimation."""

    def test_empty_batch(self):
        assert estimate_enrich_batch_tokens({"items": []}) == 0

    def test_includes_item_fields(self):
        batch = _make_enrich_batch(n_items=3)
        tokens = estimate_enrich_batch_tokens(batch)
        assert tokens > 0

    def test_context_string_counted(self):
        batch = _make_enrich_batch(n_items=1, context="A" * 4000)
        tokens = estimate_enrich_batch_tokens(batch)
        assert tokens >= 1000  # 4000 chars / 4

    def test_more_items_more_tokens(self):
        small = estimate_enrich_batch_tokens(_make_enrich_batch(n_items=2))
        large = estimate_enrich_batch_tokens(_make_enrich_batch(n_items=10))
        assert large > small


# ============================================================================
# Pre-flight token check — extraction batches
# ============================================================================


class TestPreFlightTokenCheck:
    """Tests for pre_flight_token_check on ExtractionBatches."""

    def test_no_split_when_under_budget(self):
        batch = _make_batch(n_items=3)
        result = pre_flight_token_check([batch], max_tokens=999_999)
        assert len(result) == 1
        assert len(result[0].items) == 3

    def test_splits_oversized_batch(self):
        # Create a batch with many large items to exceed a low token limit
        batch = _make_batch(n_items=10)
        tokens = estimate_batch_tokens(batch)
        # Set max_tokens to less than the batch's estimated tokens
        result = pre_flight_token_check([batch], max_tokens=tokens // 3)
        assert len(result) > 1
        # All items preserved
        total_items = sum(len(b.items) for b in result)
        assert total_items == 10

    def test_singleton_never_split(self):
        batch = _make_batch(n_items=1)
        result = pre_flight_token_check([batch], max_tokens=1)
        assert len(result) == 1

    def test_preserves_metadata(self):
        batch = _make_batch(n_items=4)
        batch.dd_version = "3.39.0"
        batch.cocos_version = 11
        batch.mode = "names"
        tokens = estimate_batch_tokens(batch)
        result = pre_flight_token_check([batch], max_tokens=tokens // 3)
        for b in result:
            assert b.dd_version == "3.39.0"
            assert b.cocos_version == 11
            assert b.mode == "names"
            assert b.source == "dd"

    def test_split_group_keys_have_suffix(self):
        batch = _make_batch(n_items=6)
        tokens = estimate_batch_tokens(batch)
        result = pre_flight_token_check([batch], max_tokens=tokens // 3)
        for b in result:
            assert "#split-" in b.group_key

    def test_empty_list_returns_empty(self):
        result = pre_flight_token_check([])
        assert result == []


# ============================================================================
# Pre-flight token check — enrich batches
# ============================================================================


class TestPreFlightEnrichTokenCheck:
    """Tests for pre_flight_enrich_token_check on enrich batch dicts."""

    def test_no_split_when_under_budget(self):
        batch = _make_enrich_batch(n_items=3)
        result = pre_flight_enrich_token_check([batch], max_tokens=999_999)
        assert len(result) == 1

    def test_splits_oversized_batch(self):
        batch = _make_enrich_batch(n_items=10)
        tokens = estimate_enrich_batch_tokens(batch)
        result = pre_flight_enrich_token_check([batch], max_tokens=tokens // 3)
        assert len(result) > 1
        total_items = sum(len(b["items"]) for b in result)
        assert total_items == 10

    def test_singleton_never_split(self):
        batch = _make_enrich_batch(n_items=1)
        result = pre_flight_enrich_token_check([batch], max_tokens=1)
        assert len(result) == 1

    def test_preserves_claim_token(self):
        batch = _make_enrich_batch(n_items=6, token="tok-abc")
        tokens = estimate_enrich_batch_tokens(batch)
        result = pre_flight_enrich_token_check([batch], max_tokens=tokens // 3)
        for b in result:
            assert b["claim_token"] == "tok-abc"

    def test_empty_list_returns_empty(self):
        result = pre_flight_enrich_token_check([])
        assert result == []


# ============================================================================
# Grouping strategy: cluster × unit (full compose)
# ============================================================================


class TestGroupByClusterAndUnit:
    """Tests for cluster × unit grouping (full compose mode)."""

    def test_same_cluster_same_unit_one_batch(self):
        items = [
            _make_item(path="a/te", cluster_id="c1", unit="eV"),
            _make_item(path="b/te", cluster_id="c1", unit="eV"),
        ]
        batches = group_by_concept_and_unit(items, max_batch_size=25)
        assert len(batches) == 1
        assert len(batches[0].items) == 2

    def test_different_units_different_batches(self):
        items = [
            _make_item(path="a/te", cluster_id="c1", unit="eV"),
            _make_item(path="a/ne", cluster_id="c1", unit="m^-3"),
        ]
        batches = group_by_concept_and_unit(items, max_batch_size=25)
        assert len(batches) == 2

    def test_different_clusters_different_batches(self):
        items = [
            _make_item(path="a/te", cluster_id="c1", unit="eV"),
            _make_item(path="a/ne", cluster_id="c2", unit="eV"),
        ]
        batches = group_by_concept_and_unit(items, max_batch_size=25)
        assert len(batches) == 2

    def test_oversized_group_splits(self):
        items = [
            _make_item(path=f"test/p{i}", cluster_id="c1", unit="eV") for i in range(10)
        ]
        batches = group_by_concept_and_unit(items, max_batch_size=3)
        # 10 items / 3 per batch = 4 batches (3+3+3+1)
        assert len(batches) == 4
        total = sum(len(b.items) for b in batches)
        assert total == 10

    def test_unclustered_paths_grouped_by_parent(self):
        items = [
            _make_item(
                path="ids/parent1/x",
                cluster_id=None,
                unit="eV",
                ids_name="ids",
                parent_path="ids/parent1",
            ),
            _make_item(
                path="ids/parent2/y",
                cluster_id=None,
                unit="eV",
                ids_name="ids",
                parent_path="ids/parent2",
            ),
        ]
        batches = group_by_concept_and_unit(items, max_batch_size=25)
        assert len(batches) == 2

    def test_empty_returns_empty(self):
        assert group_by_concept_and_unit([]) == []

    def test_max_tokens_triggers_split(self):
        """Verify that max_tokens parameter causes oversized batches to split."""
        items = [
            _make_item(
                path=f"test/p{i}",
                cluster_id="c1",
                unit="eV",
                description="x" * 500,
            )
            for i in range(20)
        ]
        # Without max_tokens
        batches_no_check = group_by_concept_and_unit(items, max_batch_size=20)
        assert len(batches_no_check) == 1

        # With a very tight max_tokens — forces split
        tokens = estimate_batch_tokens(batches_no_check[0])
        batches_with_check = group_by_concept_and_unit(
            items, max_batch_size=20, max_tokens=tokens // 3
        )
        assert len(batches_with_check) > 1


# ============================================================================
# Grouping strategy: domain × unit (name-only mode)
# ============================================================================


class TestGroupByDomainAndUnit:
    """Tests for domain × unit grouping (name-only mode)."""

    def test_same_domain_same_unit_one_batch(self):
        items = [
            _make_item(path="a/te", physics_domain="transport", unit="eV"),
            _make_item(path="b/te", physics_domain="transport", unit="eV"),
        ]
        batches = group_for_name_only(items, batch_size=50)
        assert len(batches) == 1
        assert batches[0].mode == "names"

    def test_different_domains_different_batches(self):
        items = [
            _make_item(path="a/te", physics_domain="transport", unit="eV"),
            _make_item(path="b/ip", physics_domain="magnetics", unit="A"),
        ]
        batches = group_for_name_only(items, batch_size=50)
        assert len(batches) == 2

    def test_cross_ids_merged(self):
        """Items from different IDSs but same domain+unit → one batch."""
        items = [
            _make_item(
                path="core_profiles/te",
                ids_name="core_profiles",
                physics_domain="transport",
                unit="eV",
            ),
            _make_item(
                path="edge_profiles/te",
                ids_name="edge_profiles",
                physics_domain="transport",
                unit="eV",
            ),
        ]
        batches = group_for_name_only(items, batch_size=50)
        assert len(batches) == 1
        assert len(batches[0].items) == 2

    def test_oversized_chunks(self):
        items = [
            _make_item(path=f"test/p{i}", physics_domain="transport", unit="eV")
            for i in range(15)
        ]
        batches = group_for_name_only(items, batch_size=4)
        # 15 / 4 = 4 batches (4+4+4+3)
        assert len(batches) == 4
        total = sum(len(b.items) for b in batches)
        assert total == 15

    def test_all_batches_name_only_mode(self):
        items = [_make_item(physics_domain="eq", unit="m")]
        batches = group_for_name_only(items, batch_size=50)
        for b in batches:
            assert b.mode == "names"

    def test_empty_returns_empty(self):
        assert group_for_name_only([]) == []

    def test_max_tokens_triggers_split(self):
        items = [
            _make_item(
                path=f"test/p{i}",
                physics_domain="transport",
                unit="eV",
                description="x" * 500,
            )
            for i in range(20)
        ]
        batches_no_check = group_for_name_only(items, batch_size=20)
        assert len(batches_no_check) == 1

        tokens = estimate_batch_tokens(batches_no_check[0])
        batches_with_check = group_for_name_only(
            items, batch_size=20, max_tokens=tokens // 3
        )
        assert len(batches_with_check) > 1


# ============================================================================
# Pyproject configuration
# ============================================================================


class TestBatchConfig:
    """Tests for pyproject.toml batch configuration reading."""

    def test_generate_config_has_expected_keys(self):
        cfg = get_generate_batch_config()
        assert "batch_size" in cfg
        assert "name_only_batch_size" in cfg
        assert "max_tokens" in cfg

    def test_generate_config_values_are_ints(self):
        cfg = get_generate_batch_config()
        assert isinstance(cfg["batch_size"], int)
        assert isinstance(cfg["name_only_batch_size"], int)
        assert isinstance(cfg["max_tokens"], int)

    def test_generate_config_defaults(self):
        """Defaults match pyproject.toml [tool.imas-codex.sn-generate]."""
        cfg = get_generate_batch_config()
        assert cfg["batch_size"] == 25
        assert cfg["name_only_batch_size"] == 50
        assert cfg["max_tokens"] == 150_000

    def test_enrich_config_has_expected_keys(self):
        cfg = get_enrich_batch_config()
        assert "batch_size" in cfg
        assert "max_tokens" in cfg

    def test_enrich_config_values_are_ints(self):
        cfg = get_enrich_batch_config()
        assert isinstance(cfg["batch_size"], int)
        assert isinstance(cfg["max_tokens"], int)

    def test_enrich_config_defaults(self):
        """Defaults match pyproject.toml [tool.imas-codex.sn-enrich]."""
        cfg = get_enrich_batch_config()
        assert cfg["batch_size"] == 12
        assert cfg["max_tokens"] == 150_000

    def test_default_max_tokens_constant(self):
        """Module constant matches the safety margin design."""
        assert DEFAULT_MAX_TOKENS == 150_000

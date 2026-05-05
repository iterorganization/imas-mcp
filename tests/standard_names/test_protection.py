"""Tests for pipeline protection helper (plan 35 §Pipeline protection enforcement)."""

from __future__ import annotations

import pytest

from imas_codex.standard_names.protection import PROTECTED_FIELDS, filter_protected

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(name: str, **kwargs) -> dict:
    """Build a minimal item dict with optional extra fields."""
    item = {"id": name}
    item.update(kwargs)
    return item


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFilterProtectedOverrideFalse:
    """override=False should strip protected fields from catalog_edit items."""

    def test_strips_protected_fields_from_catalog_edit(self):
        """Protected fields are removed when origin=catalog_edit."""
        items = [
            _make_item(
                "electron_temperature",
                description="new desc",
                documentation="new doc",
                kind="scalar",
                pipeline_status="enriched",  # NOT protected
            )
        ]
        filtered, skipped = filter_protected(
            items,
            override=False,
            protected_names={"electron_temperature"},
        )
        assert len(filtered) == 1
        result = filtered[0]
        # Protected fields stripped
        assert "description" not in result
        assert "documentation" not in result
        assert "kind" not in result
        # Non-protected fields preserved
        assert result["id"] == "electron_temperature"
        assert result["pipeline_status"] == "enriched"
        # Skipped list
        assert skipped == ["electron_temperature"]

    def test_non_protected_fields_pass_through(self):
        """Fields not in PROTECTED_FIELDS pass through even for catalog_edit."""
        items = [
            _make_item(
                "plasma_current",
                pipeline_status="named",
                model="test-model",
            )
        ]
        filtered, skipped = filter_protected(
            items,
            override=False,
            protected_names={"plasma_current"},
        )
        assert len(filtered) == 1
        result = filtered[0]
        assert result["pipeline_status"] == "named"
        assert result["model"] == "test-model"
        # No protected fields present → not in skipped
        assert skipped == []

    def test_pipeline_origin_passes_through(self):
        """Items NOT in protected_names (i.e. origin=pipeline) pass unchanged."""
        items = [
            _make_item(
                "major_radius",
                description="some desc",
                documentation="some doc",
                kind="scalar",
            )
        ]
        filtered, skipped = filter_protected(
            items,
            override=False,
            protected_names=set(),  # No catalog_edit names
        )
        assert len(filtered) == 1
        result = filtered[0]
        assert result["description"] == "some desc"
        assert result["documentation"] == "some doc"
        assert result["kind"] == "scalar"
        assert skipped == []

    def test_mixed_batch(self):
        """Batch with both pipeline and catalog_edit items."""
        items = [
            _make_item("name_a", description="desc_a", kind="scalar"),
            _make_item("name_b", description="desc_b", kind="vector"),
        ]
        filtered, skipped = filter_protected(
            items,
            override=False,
            protected_names={"name_b"},  # Only name_b is catalog_edit
        )
        assert len(filtered) == 2
        # name_a passes through
        assert filtered[0]["description"] == "desc_a"
        # name_b has protected fields stripped
        assert "description" not in filtered[1]
        assert "kind" not in filtered[1]
        assert filtered[1]["id"] == "name_b"
        assert skipped == ["name_b"]


class TestFilterProtectedOverrideTrue:
    """override=True should pass all fields through unchanged."""

    def test_override_passes_through(self):
        """All fields pass through when override=True."""
        items = [
            _make_item(
                "electron_temperature",
                description="new desc",
                documentation="new doc",
                kind="scalar",
                status="published",
            )
        ]
        filtered, skipped = filter_protected(
            items,
            override=True,
            protected_names={"electron_temperature"},
        )
        assert len(filtered) == 1
        result = filtered[0]
        assert result["description"] == "new desc"
        assert result["documentation"] == "new doc"
        assert result["kind"] == "scalar"
        assert result["status"] == "published"
        assert skipped == []


class TestFilterProtectedEdgeCases:
    """Edge cases and correctness."""

    def test_empty_list(self):
        """Empty items list returns empty."""
        filtered, skipped = filter_protected([], override=False, protected_names=set())
        assert filtered == []
        assert skipped == []

    def test_does_not_mutate_input(self):
        """Input items list and dicts are not modified."""
        items = [_make_item("x", description="desc", kind="scalar")]
        original = [dict(item) for item in items]
        filter_protected(items, override=False, protected_names={"x"})
        assert items == original

    def test_all_protected_fields_covered(self):
        """All PROTECTED_FIELDS are actually stripped when present."""
        item = _make_item("test_name")
        for field in PROTECTED_FIELDS:
            item[field] = "test_value"
        item["non_protected"] = "keep"

        filtered, skipped = filter_protected(
            [item], override=False, protected_names={"test_name"}
        )
        result = filtered[0]
        for field in PROTECTED_FIELDS:
            assert field not in result, f"{field} should have been stripped"
        assert result["non_protected"] == "keep"
        assert result["id"] == "test_name"
        assert skipped == ["test_name"]

    def test_returns_correct_skipped_list(self):
        """Skipped list only includes names that actually had fields stripped."""
        items = [
            _make_item("a", description="desc"),  # has protected field
            _make_item("b", model="m"),  # no protected fields
        ]
        filtered, skipped = filter_protected(
            items, override=False, protected_names={"a", "b"}
        )
        assert skipped == ["a"]  # Only "a" had fields stripped


class TestProtectedFieldsConstant:
    """Verify the PROTECTED_FIELDS frozenset contents."""

    def test_expected_fields(self):
        expected = {
            "description",
            "documentation",
            "kind",
            "links",
            "status",
            "deprecates",
            "superseded_by",
            "validity_domain",
            "constraints",
        }
        assert PROTECTED_FIELDS == expected

    def test_is_frozenset(self):
        assert isinstance(PROTECTED_FIELDS, frozenset)

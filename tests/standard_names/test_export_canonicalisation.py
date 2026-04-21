"""Tests for export canonicalisation — stable round-trip output.

Plan 35 §3d: whitespace + list-sort produces stable output; round-trip
twice gives byte-identical files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from imas_codex.standard_names.canonical import canonicalise_entry
from imas_codex.standard_names.export import _write_entry_yaml


class TestCanonicaliseEntry:
    """canonicalise_entry applies deterministic normalisation."""

    def test_whitespace_trimmed(self) -> None:
        entry = {
            "name": "  electron_temperature  ",
            "description": "  Some description  \r\n  with windows newlines\r\n",
            "documentation": "  Rich docs\n\n\n",
        }
        result = canonicalise_entry(entry)
        assert result["name"] == "electron_temperature"
        assert not result["description"].startswith(" ")
        assert not result["description"].endswith("\n")
        assert "\r\n" not in result["description"]
        assert not result["documentation"].endswith("\n")

    def test_lists_sorted(self) -> None:
        entry = {
            "tags": ["zebra", "alpha", "middle"],
            "constraints": ["c > 0", "a > 0", "b > 0"],
        }
        result = canonicalise_entry(entry)
        assert result["tags"] == ["alpha", "middle", "zebra"]
        assert result["constraints"] == ["a > 0", "b > 0", "c > 0"]

    def test_links_not_sorted(self) -> None:
        """Link order is editorial — should NOT be sorted."""
        entry = {
            "links": ["zebra_quantity", "alpha_quantity"],
        }
        result = canonicalise_entry(entry)
        assert result["links"] == ["zebra_quantity", "alpha_quantity"]

    def test_missing_nullable_fields_default_to_none(self) -> None:
        entry = {"name": "test_name"}
        result = canonicalise_entry(entry)
        assert result.get("deprecates") is None
        assert result.get("superseded_by") is None
        assert result.get("validity_domain") is None
        assert result.get("cocos_transformation_type") is None

    def test_missing_list_fields_default_to_empty(self) -> None:
        entry = {"name": "test_name"}
        result = canonicalise_entry(entry)
        assert result.get("tags") == []
        assert result.get("links") == []
        assert result.get("constraints") == []

    def test_idempotent(self) -> None:
        """Applying canonicalise twice gives the same result."""
        entry = {
            "name": "  messy_name  ",
            "tags": ["z", "a"],
            "description": "  desc \r\n trailing \n\n",
            "documentation": "docs\n\n",
            "links": ["b", "a"],
            "constraints": ["y", "x"],
        }
        first = canonicalise_entry(entry)
        second = canonicalise_entry(first)
        assert first == second

    def test_does_not_mutate_input(self) -> None:
        entry = {"tags": ["z", "a"], "name": "test"}
        original_tags = entry["tags"].copy()
        canonicalise_entry(entry)
        assert entry["tags"] == original_tags


class TestYamlRoundTripStability:
    """Writing the same entry twice produces byte-identical files."""

    def test_double_write_identical(self, tmp_path: Path) -> None:
        entry = {
            "name": "electron_temperature",
            "description": "Electron temperature profile",
            "documentation": "Detailed documentation",
            "kind": "scalar",
            "unit": "eV",
            "tags": ["kinetics", "core_profiles"],
            "links": ["electron_density"],
            "constraints": ["T_e > 0"],
            "validity_domain": "core plasma",
            "status": "draft",
        }
        canon = canonicalise_entry(entry)

        # Write once
        dir1 = tmp_path / "round1"
        p1 = _write_entry_yaml(dir1, canon, "equilibrium")

        # Write twice
        dir2 = tmp_path / "round2"
        p2 = _write_entry_yaml(dir2, canon, "equilibrium")

        content1 = p1.read_text(encoding="utf-8")
        content2 = p2.read_text(encoding="utf-8")
        assert content1 == content2, "Two writes of the same entry differ"

    def test_canonicalise_then_write_stable(self, tmp_path: Path) -> None:
        """Canonicalise messy input → write → read → canonicalise → write.
        Both writes must be identical.
        """
        messy = {
            "name": "  electron_temperature  ",
            "description": "  desc \r\n",
            "documentation": "docs\n\n\n",
            "kind": "scalar",
            "unit": "eV",
            "tags": ["z_tag", "a_tag"],
            "links": [],
            "constraints": ["b > 0", "a > 0"],
            "status": "draft",
        }

        # First pass
        canon1 = canonicalise_entry(messy)
        dir1 = tmp_path / "pass1"
        p1 = _write_entry_yaml(dir1, canon1, "test")
        content1 = p1.read_text(encoding="utf-8")

        # Read back and canonicalise again
        loaded = yaml.safe_load(content1)
        canon2 = canonicalise_entry(loaded)

        dir2 = tmp_path / "pass2"
        p2 = _write_entry_yaml(dir2, canon2, "test")
        content2 = p2.read_text(encoding="utf-8")

        assert content1 == content2, "Round-trip through YAML is not stable"

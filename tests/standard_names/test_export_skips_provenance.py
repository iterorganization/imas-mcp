"""Tests that exported YAML contains neither source_paths nor dd_paths.

Plan 35 §3d: provenance stays in the graph — never serialised to YAML.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from imas_codex.standard_names.export import (
    ExportReport,
    _graph_node_to_entry_dict,
    _write_entry_yaml,
    canonicalise_entry,
)


@pytest.fixture()
def sample_graph_node() -> dict:
    """A graph node with provenance fields that must NOT leak."""
    return {
        "id": "electron_temperature",
        "description": "Electron temperature profile",
        "documentation": "Detailed docs about Te.",
        "kind": "scalar",
        "unit": "eV",
        "tags": ["equilibrium"],
        "links": ["electron_density"],
        "constraints": ["T_e > 0"],
        "validity_domain": "core plasma",
        "cocos_transformation_type": None,
        "status": "draft",
        "physics_domain": "equilibrium",
        # ── Graph-only provenance fields ──
        "source_paths": [
            "dd:core_profiles/profiles_1d/electrons/temperature",
            "tcv:DIAG:thomson:te",
        ],
        "dd_paths": [
            "core_profiles/profiles_1d/electrons/temperature",
        ],
        "source_types": ["dd", "signals"],
        # ── Pipeline metadata (also graph-only) ──
        "pipeline_status": "published",
        "reviewer_score": 0.85,
        "origin": "pipeline",
        "cocos": 17,
    }


def test_graph_node_to_entry_excludes_provenance(sample_graph_node: dict) -> None:
    """_graph_node_to_entry_dict must NOT include source_paths or dd_paths."""
    entry = _graph_node_to_entry_dict(sample_graph_node)

    assert "source_paths" not in entry, "source_paths leaked into entry dict"
    assert "dd_paths" not in entry, "dd_paths leaked into entry dict"
    assert "source_types" not in entry, "source_types leaked into entry dict"
    assert "pipeline_status" not in entry, "pipeline_status leaked"
    assert "reviewer_score" not in entry, "reviewer_score leaked"
    assert "origin" not in entry, "origin leaked"
    assert "cocos" not in entry, "cocos (graph FK) leaked"


def test_written_yaml_excludes_provenance(
    sample_graph_node: dict, tmp_path: Path
) -> None:
    """YAML file written to staging must NOT contain provenance keys."""
    entry = _graph_node_to_entry_dict(sample_graph_node)
    entry = canonicalise_entry(entry)

    filepath = _write_entry_yaml(tmp_path, entry, "equilibrium")

    content = filepath.read_text(encoding="utf-8")
    parsed = yaml.safe_load(content)

    assert "source_paths" not in parsed, "source_paths in YAML file"
    assert "dd_paths" not in parsed, "dd_paths in YAML file"
    assert "source_types" not in parsed, "source_types in YAML file"


def test_entry_dict_has_expected_catalog_fields(sample_graph_node: dict) -> None:
    """Entry dict should contain all catalog-owned fields."""
    entry = _graph_node_to_entry_dict(sample_graph_node)

    assert entry["name"] == "electron_temperature"
    assert entry["description"] == "Electron temperature profile"
    assert entry["kind"] == "scalar"
    assert entry["unit"] == "eV"
    assert "tags" in entry
    assert "links" in entry
    assert "constraints" in entry

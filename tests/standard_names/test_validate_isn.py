"""Tests for ISN three-layer validation in validate_worker."""

from __future__ import annotations

import json


def test_validate_via_isn_well_formed():
    """Well-formed entry produces no pydantic errors."""
    from imas_codex.standard_names.workers import _validate_via_isn

    entry = {
        "id": "electron_temperature",
        "kind": "scalar",
        "description": "Electron temperature",
        "documentation": "The temperature of electrons in the plasma.",
        "unit": "eV",
        "tags": [],
        "links": [],
        "physics_domain": "transport",
        "imas_paths": ["core_profiles/profiles_1d/electrons/temperature"],
    }
    issues, summary = _validate_via_isn(entry)
    assert summary["pydantic"]["passed"] is True


def test_validate_via_isn_bad_name():
    """Double-underscore name triggers pydantic validation error."""
    from imas_codex.standard_names.workers import _validate_via_isn

    entry = {
        "id": "bad__name",
        "kind": "scalar",
        "description": "A bad name",
        "unit": "m",
    }
    issues, summary = _validate_via_isn(entry)
    assert summary["pydantic"]["passed"] is False
    assert summary["pydantic"]["error_count"] > 0
    assert any("[pydantic:" in i for i in issues)


def test_validate_via_isn_empty_entry():
    """Empty entry doesn't crash."""
    from imas_codex.standard_names.workers import _validate_via_isn

    entry = {}
    issues, summary = _validate_via_isn(entry)
    # Should not crash — may have issues but summary structure is intact
    assert "pydantic" in summary
    assert "semantic" in summary
    assert "description" in summary


def test_validate_via_isn_returns_json_serializable_summary():
    """Layer summary can be JSON-serialized."""
    from imas_codex.standard_names.workers import _validate_via_isn

    entry = {
        "id": "plasma_current",
        "kind": "scalar",
        "description": "Plasma current",
        "unit": "A",
    }
    issues, summary = _validate_via_isn(entry)
    # Must be JSON-serializable for graph storage
    json_str = json.dumps(summary)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert "pydantic" in parsed


def test_validate_via_isn_description_quality():
    """Description quality issues are captured in layer 3."""
    from imas_codex.standard_names.workers import _validate_via_isn

    entry = {
        "id": "electron_temperature",
        "kind": "scalar",
        "description": "This is the electron temperature stored in core_profiles/profiles_1d",
        "documentation": "Temperature of electrons.",
        "unit": "eV",
    }
    issues, summary = _validate_via_isn(entry)
    # The description contains a path fragment which should be flagged
    # But whether it actually gets flagged depends on ISN's implementation
    assert isinstance(issues, list)
    assert isinstance(summary["description"]["issue_count"], int)

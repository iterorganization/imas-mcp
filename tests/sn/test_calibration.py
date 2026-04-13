"""Tests for centralized calibration loading."""

from imas_codex.sn.calibration import (
    clear_calibration_cache,
    get_calibration_for_prompt,
    load_calibration,
)


def test_load_calibration_returns_list():
    clear_calibration_cache()
    entries = load_calibration()
    assert isinstance(entries, list)
    assert len(entries) > 0


def test_load_calibration_caching():
    clear_calibration_cache()
    first = load_calibration()
    second = load_calibration()
    assert first is second  # Same object (cached)


def test_calibration_entry_has_required_keys():
    entries = load_calibration()
    required = {"name", "tier", "expected_score"}
    for entry in entries:
        assert required <= set(entry.keys()), f"Missing keys in {entry.get('name')}"


def test_get_calibration_for_prompt_format():
    entries = get_calibration_for_prompt()
    assert isinstance(entries, list)
    assert len(entries) > 0
    for entry in entries:
        assert "name" in entry
        assert "tier" in entry
        assert "score" in entry
        assert "reason" in entry
        assert 0.0 <= entry["score"] <= 1.0


def test_calibration_score_normalization():
    raw = load_calibration()
    prompt = get_calibration_for_prompt()
    for r, p in zip(raw, prompt, strict=False):
        expected = round(r["expected_score"] / 120.0, 2)
        assert p["score"] == expected, f"Score mismatch for {r['name']}"

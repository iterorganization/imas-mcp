"""Tests for export gate B — COCOS manifest consistency.

Plan 35 §3d: manifest COCOS mismatch triggers gate B failure.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.export import _run_gate_b


def _make_candidate(
    name: str,
    cocos: int | None = None,
    links: list[str] | None = None,
) -> dict:
    return {
        "id": name,
        "cocos": cocos,
        "links": links or [],
    }


class TestGateBCocos:
    """Gate B checks that per-name COCOS matches the manifest."""

    def test_matching_cocos_passes(self) -> None:
        candidates = [
            _make_candidate("psi_name", cocos=17),
            _make_candidate("ip_name", cocos=17),
        ]
        result = _run_gate_b(candidates, cocos_convention=17)
        cocos_issues = [i for i in result.issues if i["type"] == "cocos_mismatch"]
        assert len(cocos_issues) == 0

    def test_null_cocos_passes(self) -> None:
        """Names without COCOS (e.g. non-COCOS quantities) pass the check."""
        candidates = [_make_candidate("temperature", cocos=None)]
        result = _run_gate_b(candidates, cocos_convention=17)
        cocos_issues = [i for i in result.issues if i["type"] == "cocos_mismatch"]
        assert len(cocos_issues) == 0

    def test_mismatched_cocos_fails(self) -> None:
        candidates = [_make_candidate("wrong_cocos", cocos=11)]
        result = _run_gate_b(candidates, cocos_convention=17)
        cocos_issues = [i for i in result.issues if i["type"] == "cocos_mismatch"]
        assert len(cocos_issues) == 1
        assert cocos_issues[0]["name"] == "wrong_cocos"
        assert cocos_issues[0]["expected"] == 17
        assert cocos_issues[0]["actual"] == 11

    def test_mixed_cocos(self) -> None:
        candidates = [
            _make_candidate("good_cocos", cocos=17),
            _make_candidate("bad_cocos", cocos=11),
            _make_candidate("no_cocos", cocos=None),
        ]
        result = _run_gate_b(candidates, cocos_convention=17)
        cocos_issues = [i for i in result.issues if i["type"] == "cocos_mismatch"]
        assert len(cocos_issues) == 1
        assert cocos_issues[0]["name"] == "bad_cocos"
        assert not result.passed  # gate fails on any COCOS mismatch


class TestGateBDanglingLinks:
    """Gate B checks that links resolve to known names."""

    def test_valid_links_pass(self) -> None:
        candidates = [
            _make_candidate("alpha", links=["beta"]),
            _make_candidate("beta", links=["alpha"]),
        ]
        result = _run_gate_b(candidates, cocos_convention=17)
        link_issues = [i for i in result.issues if i["type"] == "dangling_link"]
        assert len(link_issues) == 0

    def test_dangling_link_fails(self) -> None:
        candidates = [
            _make_candidate("alpha", links=["nonexistent"]),
        ]
        result = _run_gate_b(candidates, cocos_convention=17)
        link_issues = [i for i in result.issues if i["type"] == "dangling_link"]
        assert len(link_issues) == 1
        assert link_issues[0]["link_target"] == "nonexistent"

"""Tests for the deterministic name-key duplicate guard (plan 39 §5.2).

Phase 1.5 is independently shippable: helpers in
``imas_codex.standard_names.canonical`` enumerate the lexical-variant
set and look up colliding ``StandardName.id`` values via Cypher.
No LLM, no vector search.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.canonical import (
    find_name_key_duplicate,
    lexical_variants,
    name_key_normalise,
)

# ---------------------------------------------------------------------------
# name_key_normalise
# ---------------------------------------------------------------------------


class TestNameKeyNormalise:
    """Documented variant rules: case-fold, collapse ``_`` runs, strip edges."""

    def test_casefold(self) -> None:
        assert name_key_normalise("Electron_Temperature") == "electron_temperature"

    def test_collapse_underscores(self) -> None:
        assert name_key_normalise("electron__temperature") == "electron_temperature"
        assert name_key_normalise("electron___temperature") == "electron_temperature"
        assert name_key_normalise("a____b") == "a_b"

    def test_strip_edge_underscores(self) -> None:
        assert name_key_normalise("__electron_temperature_") == "electron_temperature"
        assert name_key_normalise("_") == ""
        assert name_key_normalise("___") == ""

    def test_idempotent(self) -> None:
        for raw in [
            "Electron__Temperature",
            "_Te_",
            "PLASMA_CURRENT_density",
            "x__y___z",
        ]:
            once = name_key_normalise(raw)
            twice = name_key_normalise(once)
            assert once == twice, f"non-idempotent on {raw!r}"

    def test_non_string(self) -> None:
        assert name_key_normalise(None) == ""  # type: ignore[arg-type]
        assert name_key_normalise(42) == ""  # type: ignore[arg-type]

    def test_empty(self) -> None:
        assert name_key_normalise("") == ""

    def test_no_change_for_canonical(self) -> None:
        assert name_key_normalise("electron_temperature") == "electron_temperature"
        assert name_key_normalise("plasma_current") == "plasma_current"


# ---------------------------------------------------------------------------
# lexical_variants
# ---------------------------------------------------------------------------


class TestLexicalVariants:
    def test_includes_self_and_canonical(self) -> None:
        variants = lexical_variants("Electron_Temperature")
        assert "Electron_Temperature" in variants
        assert "electron_temperature" in variants

    def test_collapsed_variant_present(self) -> None:
        variants = lexical_variants("electron__temperature")
        assert "electron_temperature" in variants

    def test_empty_input(self) -> None:
        assert lexical_variants("") == set()
        assert lexical_variants(None) == set()  # type: ignore[arg-type]

    def test_deterministic_set(self) -> None:
        # Set is a value type; equality across calls is required.
        a = lexical_variants("Plasma_Current")
        b = lexical_variants("Plasma_Current")
        assert a == b


# ---------------------------------------------------------------------------
# find_name_key_duplicate (graph-mocked)
# ---------------------------------------------------------------------------


def _gc_returning(rows_per_call: list[list[dict]]):
    """Build a mock GraphClient whose ``.query`` returns successive lists."""
    gc = MagicMock()
    rows_iter = iter(rows_per_call + [[]] * 8)  # pad for fallback queries
    gc.query.side_effect = lambda *a, **kw: next(rows_iter)
    return gc


class TestFindNameKeyDuplicate:
    def test_no_match_returns_none(self) -> None:
        gc = _gc_returning([[]])
        assert find_name_key_duplicate(gc, "electron_temperature") is None

    def test_exact_match_returns_id(self) -> None:
        gc = _gc_returning([[{"id": "electron_temperature"}]])
        result = find_name_key_duplicate(gc, "Electron_Temperature")
        assert result == "electron_temperature"

    def test_case_only_collision(self) -> None:
        gc = _gc_returning([[{"id": "electron_temperature"}]])
        result = find_name_key_duplicate(gc, "ELECTRON_TEMPERATURE")
        assert result == "electron_temperature"

    def test_underscore_collapse_collision(self) -> None:
        # Cypher branch may already match; the Python re-check confirms
        # name_key_normalise agreement.
        gc = _gc_returning([[{"id": "electron_temperature"}]])
        result = find_name_key_duplicate(gc, "electron__temperature")
        assert result == "electron_temperature"

    def test_fallback_finds_long_underscore_runs(self) -> None:
        """Cypher's bounded ``__`` collapse may miss 4+ underscore runs;
        the Python fallback uses CONTAINS + name_key_normalise re-check
        and catches them."""
        gc = MagicMock()
        # 1st query: tight Cypher returns no match.
        # 2nd query: fallback CONTAINS query returns the candidate.
        gc.query.side_effect = [
            [],
            [{"id": "electron_temperature"}],
        ]
        result = find_name_key_duplicate(gc, "electron________temperature")
        assert result == "electron_temperature"

    def test_exclude_skips_self(self) -> None:
        """When refining ``old`` → ``new``, ``old`` is excluded so its
        own row never poses as a self-collision."""
        gc = _gc_returning([[]])
        result = find_name_key_duplicate(
            gc, "electron_temperature", exclude="electron_temperature"
        )
        # Cypher will be parametrised with $exclude; mock returns no match.
        assert result is None
        # Verify the parameter actually flowed through.
        kwargs = gc.query.call_args.kwargs
        assert kwargs.get("exclude") == "electron_temperature"

    def test_empty_candidate(self) -> None:
        gc = MagicMock()
        assert find_name_key_duplicate(gc, "") is None
        assert find_name_key_duplicate(gc, "   ") is None
        gc.query.assert_not_called()

    def test_zero_llm_cost(self) -> None:
        """The dup-guard MUST never reach for an LLM. We verify by
        importing the module's namespace and confirming no llm/litellm
        attribute is referenced. (Compile-time / signature check — the
        tests above already exercise behaviour with only a graph mock.)"""
        import imas_codex.standard_names.canonical as canon

        src = open(canon.__file__).read()
        assert "litellm" not in src
        assert (
            "openai" not in src.lower()
            or "openai" in src.lower()
            and ("import openai" not in src and "from openai" not in src)
        )


# ---------------------------------------------------------------------------
# Cypher contract — defensive
# ---------------------------------------------------------------------------


def test_first_cypher_filters_exclude_and_self() -> None:
    """The primary Cypher pass must exclude ``$exclude`` and the
    candidate id itself (so a freshly-MERGEd row can't shadow itself)."""
    gc = _gc_returning([[]])
    find_name_key_duplicate(gc, "electron_temperature", exclude="legacy")
    cypher = gc.query.call_args_list[0][0][0]
    assert "$exclude IS NULL OR sn.id <> $exclude" in cypher
    assert "sn.id <> $candidate_name" in cypher


def test_first_cypher_uses_candidate_key_param() -> None:
    gc = _gc_returning([[]])
    find_name_key_duplicate(gc, "Electron_Temperature")
    kwargs = gc.query.call_args_list[0].kwargs
    assert kwargs["candidate_key"] == "electron_temperature"


# ---------------------------------------------------------------------------
# Integration with re-normalisation invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "candidate,existing,should_match",
    [
        ("Electron_Temperature", "electron_temperature", True),
        ("electron__temperature", "electron_temperature", True),
        ("ELECTRON_TEMPERATURE", "electron_temperature", True),
        ("electron_density", "electron_temperature", False),
        ("plasma_current", "electron_temperature", False),
    ],
)
def test_dup_decision_matches_normaliser(candidate, existing, should_match) -> None:
    a = name_key_normalise(candidate)
    b = name_key_normalise(existing)
    assert (a == b) is should_match

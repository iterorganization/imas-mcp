"""W2: Closed-vocabulary completeness for the SN compose system prompt.

Verifies that the rendered ``sn/generate_name_system`` prompt — and the
shared ``_grammar_reference.md`` partial it embeds — contains EVERY token
from the canonical closed vocabulary segments declared by
``imas_standard_names.grammar.constants.SEGMENT_TOKEN_MAP``.

Background: the dominant LLM failure mode in W0/W1 reviews was
closed-vocabulary tokens (toroidal, parallel, thermal, e_cross_b_drift,
normalized, fast_ion, …) being absorbed into ``physical_base`` instead of
placed in their correct grammar segment slot.  W2 makes that error
completely fixable with prompting alone by injecting EVERY closed token
verbatim into the system prompt.  These tests guard the injection from
silent regressions.
"""

from __future__ import annotations

import pytest

from imas_codex.llm.prompt_loader import render_prompt
from imas_codex.standard_names.context import build_compose_context

# Aliased segments are duplicated in SEGMENT_TOKEN_MAP under multiple keys
# (component=coordinate, device=object, geometry=position).  We emit only the
# canonical names to keep the prompt cacheable; the alias names are noted in
# parentheses next to the canonical heading.
_ALIASES = {"coordinate", "object", "position"}

# Tokens shorter than this are skipped because single-letter coordinate
# tokens (x, y, z) are guaranteed to clash with arbitrary substrings.  The
# ``find_absorbed_closed_tokens`` primitive uses the same threshold.
_MIN_TOKEN_LEN = 3


def _closed_segments() -> dict[str, list[str]]:
    """Canonical closed segments (alias-deduped, open ones excluded)."""
    from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP

    out: dict[str, list[str]] = {}
    for seg, toks in SEGMENT_TOKEN_MAP.items():
        if seg in _ALIASES:
            continue
        if not toks:  # skip open segments (physical_base)
            continue
        out[seg] = sorted(toks)
    return out


@pytest.fixture(scope="module")
def rendered_system_prompt() -> str:
    """Render the SN compose system prompt with the production context."""
    ctx = build_compose_context()
    return render_prompt("sn/generate_name_system", context=ctx)


@pytest.fixture(scope="module")
def closed_segments() -> dict[str, list[str]]:
    return _closed_segments()


class TestClosedVocabFull:
    """Exercise the ``_load_closed_vocab_full`` context builder directly."""

    def test_returns_nonempty_list(self):
        from imas_codex.standard_names.context import _load_closed_vocab_full

        data = _load_closed_vocab_full()
        assert isinstance(data, list)
        assert len(data) >= 6, "expected ≥6 canonical closed segments"

    def test_alias_segments_omitted_at_top_level(self):
        from imas_codex.standard_names.context import _load_closed_vocab_full

        seg_names = {entry["segment"] for entry in _load_closed_vocab_full()}
        assert _ALIASES.isdisjoint(seg_names), (
            f"alias segments must not appear as top-level entries; got {seg_names & _ALIASES}"
        )

    def test_aliases_attached_to_canonicals(self):
        from imas_codex.standard_names.context import _load_closed_vocab_full

        by_seg = {e["segment"]: e for e in _load_closed_vocab_full()}
        # component is canonical for coordinate; device for object; geometry for position
        assert "coordinate" in by_seg.get("component", {}).get("aliases", [])
        assert "object" in by_seg.get("device", {}).get("aliases", [])
        assert "position" in by_seg.get("geometry", {}).get("aliases", [])

    def test_physical_base_excluded(self):
        from imas_codex.standard_names.context import _load_closed_vocab_full

        seg_names = {entry["segment"] for entry in _load_closed_vocab_full()}
        assert "physical_base" not in seg_names

    def test_tokens_sorted_alphabetically(self):
        from imas_codex.standard_names.context import _load_closed_vocab_full

        for entry in _load_closed_vocab_full():
            toks = entry["tokens"]
            assert toks == sorted(toks), (
                f"tokens for segment {entry['segment']!r} not sorted"
            )


class TestSystemPromptContainsAllClosedTokens:
    """The rendered prompt must contain every token from every closed segment."""

    def test_all_tokens_appear_in_rendered_prompt(
        self, rendered_system_prompt, closed_segments
    ):
        missing: list[str] = []
        for seg, toks in closed_segments.items():
            for tok in toks:
                if len(tok) < _MIN_TOKEN_LEN:
                    continue
                if tok not in rendered_system_prompt:
                    missing.append(f"{seg}:{tok}")
        assert not missing, (
            f"closed-vocab tokens missing from rendered system prompt "
            f"({len(missing)} missing): {missing[:25]}"
        )

    @pytest.mark.parametrize(
        "token",
        [
            "toroidal",
            "parallel",
            "thermal_electron",
            "e_cross_b_drift",
            "fast_ion",
            "volume_averaged",
            "pfirsch_schlueter",
            "scrape_off_layer",
            "edge_region",
            "diamagnetic_drift",
        ],
    )
    def test_high_signal_tokens_present(self, rendered_system_prompt, token):
        """Tokens cited in mid-tier reviewer comments must appear verbatim."""
        assert token in rendered_system_prompt, (
            f"high-signal closed-vocab token {token!r} missing from prompt — "
            "the W0 reviewer corpus singled this token out as a recurring "
            "decomposition-failure absorber."
        )

    def test_decomposition_checklist_present(self, rendered_system_prompt):
        """The numbered checklist that drives self-correction must render."""
        assert "Decomposition Checklist" in rendered_system_prompt
        # Look for the action-verb cues so the checklist is recognisable
        assert "Tokenise the candidate" in rendered_system_prompt
        assert "physical_base" in rendered_system_prompt

    def test_w2_anti_pattern_gallery_renders(self, rendered_system_prompt):
        """Anti-pattern gallery section must render with at least 6 entries."""
        assert "W2 DECOMPOSITION-FAILURE GALLERY" in rendered_system_prompt
        # Each entry uses the W2-D{n} prefix
        marker_count = sum(
            1 for n in range(1, 16) if f"W2-D{n}" in rendered_system_prompt
        )
        assert marker_count >= 6, (
            f"expected ≥6 W2 anti-pattern entries, found {marker_count}"
        )


class TestStaleGuidanceStripped:
    """The legacy 'physical_base is OPEN — no decomposition' wording must be gone."""

    def test_review_names_no_stale_open_section(self):
        """``review_names.md`` no longer has the stale 'OPEN vocabulary' header."""
        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        text = (PROMPTS_DIR / "sn" / "review_names.md").read_text(encoding="utf-8")
        assert "## `physical_base` is OPEN vocabulary" not in text
        # New replacement section must be present
        assert "SINGLE open grammar segment" in text

    def test_l6_retry_no_open_vocab_phrasing(self):
        """L6 grammar-retry helper no longer claims physical_base is open."""
        import inspect

        from imas_codex.standard_names.workers import _grammar_retry

        src = inspect.getsource(_grammar_retry)
        assert "open vocabulary" not in src, (
            "stale 'physical_base is open vocabulary' wording must be removed "
            "from the L6 retry helper"
        )

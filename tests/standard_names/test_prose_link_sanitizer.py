"""Tests for inline-markdown link sanitization in enrich validate phase.

Bug: The LLM enrichment can produce ``[anchor](name:foo)`` markdown links
inside ``enriched_description`` / ``enriched_documentation`` prose where
``foo`` is not a valid StandardName id.  Previously the validator recorded
``link_not_found`` issues but did not strip the broken links from the
prose, so the catalog rendered dead hyperlinks.

These tests pin the new behaviour:
  1. Markdown link extraction picks up ``name:foo`` targets in prose.
  2. ``_strip_broken_prose_links`` rewrites broken links to plain anchor
     text while leaving valid links intact.
  3. ``_check_links_batch`` validates prose links against batch + graph
     and mutates the item dict in-place to remove broken hyperlinks.
"""

from __future__ import annotations

from imas_codex.standard_names.enrich_workers import (
    _check_links_batch,
    _extract_prose_link_targets,
    _strip_broken_prose_links,
)


class TestExtractProseLinkTargets:
    def test_empty(self):
        assert _extract_prose_link_targets("") == []
        assert _extract_prose_link_targets(None) == []

    def test_single_link(self):
        text = "The [ohmic current](name:ohmic_current) drives ..."
        assert _extract_prose_link_targets(text) == ["ohmic_current"]

    def test_multiple_links(self):
        text = (
            "Linked to [psi](name:poloidal_flux) and [T_e](name:electron_temperature)."
        )
        assert _extract_prose_link_targets(text) == [
            "poloidal_flux",
            "electron_temperature",
        ]

    def test_ignores_url_links(self):
        text = "See [docs](https://example.com) and [name](name:foo_bar)."
        assert _extract_prose_link_targets(text) == ["foo_bar"]

    def test_invalid_target_format_ignored(self):
        # Targets must be lowercase snake_case starting with a letter.
        text = "Bad [a](name:Foo) [b](name:1bad) [c](name:_bad) good [d](name:ok_name)"
        assert _extract_prose_link_targets(text) == ["ok_name"]


class TestStripBrokenProseLinks:
    def test_none_input(self):
        assert _strip_broken_prose_links(None, set()) is None

    def test_empty_string(self):
        assert _strip_broken_prose_links("", set()) == ""

    def test_no_links_unchanged(self):
        text = "Plain prose with no links at all."
        assert _strip_broken_prose_links(text, set()) == text

    def test_valid_link_preserved(self):
        text = "Linked to [poloidal flux](name:poloidal_flux)."
        out = _strip_broken_prose_links(text, {"poloidal_flux"})
        assert out == text

    def test_broken_link_collapsed_to_anchor(self):
        text = "Driven by the [ohmic current](name:ohmic_current_density)."
        out = _strip_broken_prose_links(text, set())
        assert out == "Driven by the ohmic current."

    def test_mixed_valid_and_broken(self):
        text = (
            "Coupled to [T_e](name:electron_temperature) "
            "and [j_oh](name:ohmic_current_density)."
        )
        out = _strip_broken_prose_links(text, {"electron_temperature"})
        assert out == "Coupled to [T_e](name:electron_temperature) and j_oh."

    def test_idempotent(self):
        text = "Driven by the [ohmic current](name:ohmic_current_density)."
        once = _strip_broken_prose_links(text, set())
        twice = _strip_broken_prose_links(once, set())
        assert once == twice


class TestCheckLinksBatchProseAware:
    """Without graph: in-batch resolution and broken-link prose stripping."""

    def test_in_batch_link_valid_no_strip(self):
        items = [
            {
                "id": "current_a",
                "enriched_description": "Linked to [B](name:current_b).",
                "enriched_documentation": None,
                "enriched_links": [],
            },
            {
                "id": "current_b",
                "enriched_description": "Sibling of A.",
                "enriched_documentation": None,
                "enriched_links": [],
            },
        ]
        batch_ids = {"current_a", "current_b"}
        result = _check_links_batch(items, batch_ids)
        assert result["current_a"] == []
        # Prose was preserved
        assert items[0]["enriched_description"] == "Linked to [B](name:current_b)."

    def test_broken_prose_link_stripped_and_reported(self, monkeypatch):
        # Force the graph query to find nothing
        from imas_codex.standard_names import enrich_workers as ew

        class _FakeGC:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def query(self, *_args, **_kwargs):
                return []

        monkeypatch.setattr(ew, "_VALID_LINK_STATUSES", frozenset({"named"}))
        monkeypatch.setattr(
            "imas_codex.graph.client.GraphClient", lambda *a, **kw: _FakeGC()
        )

        items = [
            {
                "id": "alpha",
                "enriched_description": "Driven by [j_oh](name:ohmic_current_density).",
                "enriched_documentation": (
                    "See [psi](name:poloidal_flux) for context."
                ),
                "enriched_links": [],
            },
        ]
        result = _check_links_batch(items, batch_ids={"alpha"})

        # Both broken links were reported
        assert "link_not_found:name:ohmic_current_density" in result["alpha"]
        assert "link_not_found:name:poloidal_flux" in result["alpha"]

        # Prose was rewritten to plain anchor text
        assert items[0]["enriched_description"] == "Driven by j_oh."
        assert items[0]["enriched_documentation"] == "See psi for context."

    def test_mixed_links_only_broken_stripped(self, monkeypatch):
        from imas_codex.standard_names import enrich_workers as ew

        class _FakeGC:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def query(self, *_args, link_ids=None, **_kwargs):
                # Pretend "name:electron_temperature" exists in graph.
                if link_ids and "name:electron_temperature" in link_ids:
                    return [{"id": "name:electron_temperature"}]
                return []

        monkeypatch.setattr(ew, "_VALID_LINK_STATUSES", frozenset({"named"}))
        monkeypatch.setattr(
            "imas_codex.graph.client.GraphClient", lambda *a, **kw: _FakeGC()
        )

        items = [
            {
                "id": "alpha",
                "enriched_description": (
                    "Coupled to [T_e](name:electron_temperature) "
                    "and [j_oh](name:ohmic_current_density)."
                ),
                "enriched_documentation": None,
                "enriched_links": [],
            },
        ]
        result = _check_links_batch(items, batch_ids={"alpha"})

        assert "link_not_found:name:ohmic_current_density" in result["alpha"]
        assert "link_not_found:name:electron_temperature" not in result["alpha"]

        # Only the broken link was stripped; the valid one was kept.
        assert (
            items[0]["enriched_description"]
            == "Coupled to [T_e](name:electron_temperature) and j_oh."
        )

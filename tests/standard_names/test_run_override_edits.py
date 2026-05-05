"""Test ``--override-edits`` bypasses protection for named items only.

Verifies that ``override_edits`` on TurnConfig:
  - Passes the override set to resolve_links_batch
  - Bypasses protection for listed names
  - Other names remain protected

Uses the protection.filter_protected helper directly to verify selective
override semantics.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from imas_codex.standard_names.protection import PROTECTED_FIELDS, filter_protected
from imas_codex.standard_names.turn import TurnConfig, run_turn, skip_flags_from_only

# ── filter_protected selective override tests ────────────────────────


class TestFilterProtectedOverrideNames:
    """filter_protected with override_names parameter."""

    def test_override_names_bypasses_protection_for_listed(self):
        """Names in override_names pass through even if catalog-edited."""
        items = [
            {"id": "bar", "description": "new bar desc"},
        ]
        filtered, skipped = filter_protected(
            items,
            override=False,
            override_names={"bar"},
            protected_names={"bar"},  # bar is catalog-edited
        )
        assert len(filtered) == 1
        # bar should keep its protected fields
        assert "description" in filtered[0]
        assert skipped == []

    def test_override_names_does_not_affect_others(self):
        """Names NOT in override_names remain protected."""
        items = [
            {"id": "bar", "description": "bar desc"},
            {"id": "baz", "description": "baz desc"},
        ]
        filtered, skipped = filter_protected(
            items,
            override=False,
            override_names={"bar"},
            protected_names={"bar", "baz"},  # both catalog-edited
        )

        bar_item = next(f for f in filtered if f["id"] == "bar")
        baz_item = next(f for f in filtered if f["id"] == "baz")

        # bar: overridden, all fields pass
        assert "description" in bar_item

        # baz: still protected, fields stripped
        assert "description" not in baz_item
        assert "baz" in skipped

    def test_override_names_with_no_protected(self):
        """override_names has no effect when no names are protected."""
        items = [{"id": "foo", "description": "foo desc"}]
        filtered, skipped = filter_protected(
            items,
            override=False,
            override_names={"foo"},
            protected_names=set(),
        )
        assert len(filtered) == 1
        assert "description" in filtered[0]
        assert skipped == []


# ── TurnConfig + link integration ───────────────────────────


@pytest.mark.asyncio
class TestOverrideEditsInTurn:
    """--override-edits flows through to resolve_links_batch."""

    async def test_override_edits_passed_to_resolve_links(self):
        """override_edits from TurnConfig reaches resolve_links_batch."""
        flags = skip_flags_from_only("link")
        cfg = TurnConfig(
            domain="equilibrium",
            only="link",
            override_edits=["foo", "bar"],
            **flags,
        )

        with (
            patch(
                "imas_codex.standard_names.turn._fetch_unresolved_links",
                return_value=[
                    {"id": "foo", "links": ["dd:some/path"], "retry_count": 0},
                ],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.resolve_links_batch",
                return_value={"resolved": 1, "unresolved": 0, "failed": 0},
            ) as mock_resolve,
        ):
            await run_turn(cfg)

        mock_resolve.assert_called_once()
        call_kwargs = mock_resolve.call_args[1]
        assert call_kwargs["override_names"] == {"foo", "bar"}

    async def test_override_edits_none_when_empty(self):
        """When no override_edits, override_names is None."""
        flags = skip_flags_from_only("link")
        cfg = TurnConfig(
            domain="equilibrium",
            only="link",
            **flags,
        )

        with (
            patch(
                "imas_codex.standard_names.turn._fetch_unresolved_links",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.resolve_links_batch",
            ) as mock_resolve,
        ):
            await run_turn(cfg)

        # Should not have been called (no items)
        mock_resolve.assert_not_called()

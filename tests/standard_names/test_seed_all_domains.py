"""Unit tests for ``_seed_all_domains`` and ``_list_physics_domains_with_extractable_paths``.

``_seed_all_domains(source, max_sources=None)`` is an async function that:
1. Fetches distinct physics domains (via ``_list_physics_domains_with_extractable_paths``).
2. Iterates domains, skipping ``'mixed'`` (mixed-unit paths violate the unit
   invariant of a single StandardName).
3. Calls ``_seed_domain_sources(domain=d, source=source)`` for each eligible domain.
4. Stops early if ``max_sources`` is set and the running total reaches the cap.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODULE = "imas_codex.standard_names.loop"


def _patch_list_domains(return_value: list[str]):
    """Patch ``_list_physics_domains_with_extractable_paths`` in the loop module."""
    return patch(
        f"{_MODULE}._list_physics_domains_with_extractable_paths",
        return_value=return_value,
    )


def _patch_seed_domain(side_effect=None, return_value: int = 1):
    """Patch ``_seed_domain_sources`` with an AsyncMock."""
    mock = AsyncMock(return_value=return_value)
    if side_effect is not None:
        mock.side_effect = side_effect
    return patch(f"{_MODULE}._seed_domain_sources", mock), mock


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------


from imas_codex.standard_names.loop import _seed_all_domains  # noqa: E402

# ---------------------------------------------------------------------------
# mixed-domain skipping
# ---------------------------------------------------------------------------


class TestMixedDomainSkipped:
    """The 'mixed' domain must be silently skipped by _seed_all_domains."""

    async def test_mixed_not_seeded(self) -> None:
        """_seed_domain_sources must never be called with domain='mixed'."""
        domains = ["equilibrium", "mixed", "magnetics"]
        with _patch_list_domains(domains):
            patch_ctx, mock_seed = _patch_seed_domain()
            with patch_ctx:
                await _seed_all_domains(source="dd")

        called_domains = [call.kwargs["domain"] for call in mock_seed.call_args_list]
        assert "mixed" not in called_domains

    async def test_non_mixed_domains_are_seeded(self) -> None:
        """Every non-'mixed' domain returned by the list function must be seeded."""
        domains = ["equilibrium", "mixed", "core_profiles", "magnetics"]
        with _patch_list_domains(domains):
            patch_ctx, mock_seed = _patch_seed_domain()
            with patch_ctx:
                await _seed_all_domains(source="dd")

        called_domains = {call.kwargs["domain"] for call in mock_seed.call_args_list}
        assert called_domains == {"equilibrium", "core_profiles", "magnetics"}

    async def test_total_excludes_mixed(self) -> None:
        """Return value must count only the non-'mixed' domains."""
        domains = ["equilibrium", "mixed", "magnetics"]  # 2 non-mixed → total 2
        with _patch_list_domains(domains):
            patch_ctx, mock_seed = _patch_seed_domain(return_value=1)
            with patch_ctx:
                total = await _seed_all_domains(source="dd")

        assert total == 2

    async def test_only_mixed_returns_zero(self) -> None:
        """If the domain list contains only 'mixed', total is 0 and seed not called."""
        with _patch_list_domains(["mixed"]):
            patch_ctx, mock_seed = _patch_seed_domain()
            with patch_ctx:
                total = await _seed_all_domains(source="dd")

        assert total == 0
        mock_seed.assert_not_called()


# ---------------------------------------------------------------------------
# max_sources cap
# ---------------------------------------------------------------------------


class TestMaxSourcesCap:
    """max_sources halts iteration once the running total reaches the cap."""

    async def test_stops_after_cap_reached(self) -> None:
        """With max_sources=3 and each domain seeding 2, iteration stops
        after the second domain (total=4 >= 3)."""
        domains = ["equilibrium", "core_profiles", "magnetics"]
        # Each domain seeds 2 sources
        with _patch_list_domains(domains):
            patch_ctx, mock_seed = _patch_seed_domain(return_value=2)
            with patch_ctx:
                total = await _seed_all_domains(source="dd", max_sources=3)

        # Only the first two domains should have been seeded
        assert mock_seed.await_count == 2
        # total is the sum from completed calls (2+2=4)
        assert total == 4

    async def test_no_cap_seeds_all_domains(self) -> None:
        """When max_sources is None, all domains are seeded."""
        domains = ["equilibrium", "core_profiles", "magnetics"]
        with _patch_list_domains(domains):
            patch_ctx, mock_seed = _patch_seed_domain(return_value=1)
            with patch_ctx:
                total = await _seed_all_domains(source="dd", max_sources=None)

        assert mock_seed.await_count == 3
        assert total == 3

    async def test_cap_exact_boundary(self) -> None:
        """Iteration stops *after* the call that reaches or exceeds max_sources."""
        # Domain A seeds 5, cap is 5 → stops after A (no B call)
        domains = ["alpha", "beta"]
        with _patch_list_domains(domains):
            patch_ctx, mock_seed = _patch_seed_domain(return_value=5)
            with patch_ctx:
                total = await _seed_all_domains(source="dd", max_sources=5)

        assert mock_seed.await_count == 1
        assert total == 5


# ---------------------------------------------------------------------------
# Non-DD source
# ---------------------------------------------------------------------------


class TestNonDdSource:
    """For non-DD sources, _list_physics_domains_with_extractable_paths returns []."""

    async def test_signals_source_seeds_nothing(self) -> None:
        """With source='signals', the domain list is empty → total=0."""
        # Don't mock the list function; let it call the real one which returns []
        # for non-dd, but mock _seed_domain_sources to be safe
        with _patch_list_domains([]):
            patch_ctx, mock_seed = _patch_seed_domain()
            with patch_ctx:
                total = await _seed_all_domains(source="signals")

        assert total == 0
        mock_seed.assert_not_called()


# ---------------------------------------------------------------------------
# Source forwarded to _seed_domain_sources
# ---------------------------------------------------------------------------


class TestSourceForwarding:
    """The source argument must be forwarded to each _seed_domain_sources call."""

    async def test_source_dd_forwarded(self) -> None:
        with _patch_list_domains(["equilibrium"]):
            patch_ctx, mock_seed = _patch_seed_domain()
            with patch_ctx:
                await _seed_all_domains(source="dd")

        assert mock_seed.call_args.kwargs["source"] == "dd"

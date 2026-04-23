"""Tests for the DOCUMENT worker (Phase C.3).

Covers:
- Unit: mocked LLM returns valid enrichment → enriched fields set on items.
- Unit: cost / tokens accumulated on state.
- Unit: LLM parse error → batch marked failed, errors incremented, pipeline continues.
- Unit: LLM response tries to change unit → original unit preserved.
- Unit: budget exhausted between batches → remaining batches skipped.
- Unit: stop_requested set between batches → stops cleanly.
- Unit: dry_run skips LLM calls.
- Unit: empty batches → worker completes without error.
- Unit: LLM response missing item → skipped gracefully.
- Integration: single real batch of 2 items (requires OPENROUTER_API_KEY).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from imas_codex.standard_names.enrich_state import StandardNameEnrichState
from imas_codex.standard_names.models import (
    StandardNameEnrichBatch,
    StandardNameEnrichItem,
)

# =============================================================================
# Fixtures / helpers
# =============================================================================


@pytest.fixture(autouse=True)
def _mock_valid_names():
    """Isolate tests from live graph: treat all referenced names as valid."""
    with patch(
        "imas_codex.standard_names.enrich_workers._fetch_existing_sn_names",
        return_value={"electron_temperature", "ion_temperature"},
    ):
        yield


def _make_item(name: str, **overrides: Any) -> dict[str, Any]:
    """Build a mock SN item as returned by the contextualise worker."""
    base = {
        "id": name,
        "description": f"Description of {name}",
        "documentation": None,
        "kind": "scalar",
        "unit": "eV",
        "tags": ["equilibrium"],
        "links": None,
        "source_paths": [f"equilibrium/time_slice/profiles_1d/{name}"],
        "physical_base": "temperature",
        "subject": "electron",
        "component": None,
        "coordinate": None,
        "position": None,
        "process": None,
        "physics_domain": "equilibrium",
        "confidence": 0.9,
        "model": "test-model",
        # Contextualise fields
        "dd_paths": [
            {
                "path": f"core_profiles/profiles_1d/{name}",
                "ids": "core_profiles",
                "description": f"Profile for {name}",
                "documentation": None,
                "unit": "eV",
            }
        ],
        "nearby": [{"name": "ion_temperature", "description": "Ion temperature"}],
        "siblings": [{"name": "plasma_current", "description": "Plasma current"}],
        "grammar": {"physical_base": "temperature", "subject": "electron"},
        "cocos": None,
        "current": {
            "description": None,
            "documentation": None,
            "tags": None,
            "links": None,
        },
    }
    base.update(overrides)
    return base


def _make_batch(items: list[dict], batch_index: int = 0) -> dict[str, Any]:
    """Build a batch dict matching contextualise worker output."""
    return {
        "items": items,
        "claim_token": "test-token",
        "batch_index": batch_index,
    }


def _make_state(**overrides: Any) -> StandardNameEnrichState:
    """Create a test enrich state."""
    defaults: dict[str, Any] = {
        "facility": "dd",
        "cost_limit": 10.0,
    }
    defaults.update(overrides)
    return StandardNameEnrichState(**defaults)


def _make_enrich_result(*names: str) -> StandardNameEnrichBatch:
    """Build a valid StandardNameEnrichBatch for the given names."""
    items = [
        StandardNameEnrichItem(
            standard_name=name,
            description=f"One-sentence description of {name}.",
            documentation=f"# {name}\n\nDetailed documentation for {name} with $T_e$ LaTeX.",
            tags=["measured", "time-dependent"],
            links=["ion_temperature"],
            validity_domain="Tokamak core plasma",
            constraints=["Must be positive"],
        )
        for name in names
    ]
    return StandardNameEnrichBatch(items=items)


class _FakeLLMResult:
    """Mimics LLMResult with 3-tuple unpacking."""

    def __init__(self, parsed, cost=0.01, tokens=500):
        self.parsed = parsed
        self.cost = cost
        self.tokens = tokens

    def __iter__(self):
        return iter((self.parsed, self.cost, self.tokens))

    def __len__(self):
        return 3


# =============================================================================
# Unit tests
# =============================================================================


@pytest.mark.asyncio
async def test_document_enriches_items():
    """LLM returns valid enrichment → enriched fields set on items."""
    items = [_make_item("electron_temperature"), _make_item("ion_temperature")]
    state = _make_state(batches=[_make_batch(items)])

    enrichment = _make_enrich_result("electron_temperature", "ion_temperature")

    with (
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            new_callable=AsyncMock,
            return_value=_FakeLLMResult(enrichment, cost=0.05, tokens=1200),
        ),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            return_value="mock prompt",
        ),
        patch(
            "imas_codex.settings.get_model",
            return_value="openrouter/anthropic/claude-opus-4.6",
        ),
    ):
        from imas_codex.standard_names.enrich_workers import enrich_document_worker

        await enrich_document_worker(state)

    # Check enriched fields set on first item
    assert (
        items[0]["enriched_description"]
        == "One-sentence description of electron_temperature."
    )
    assert "LaTeX" in items[0]["enriched_documentation"]
    assert items[0]["enriched_tags"] == ["measured", "time-dependent"]
    assert items[0]["enriched_links"] == ["name:ion_temperature"]
    assert items[0]["enriched_validity_domain"] == "Tokamak core plasma"
    assert items[0]["enriched_constraints"] == ["Must be positive"]

    # Second item also enriched
    assert (
        items[1]["enriched_description"]
        == "One-sentence description of ion_temperature."
    )

    # Phase completed
    assert state.document_phase.done


@pytest.mark.asyncio
async def test_document_accumulates_cost_and_tokens():
    """Cost and tokens accumulated on state after LLM call."""
    items = [_make_item("electron_temperature")]
    state = _make_state(batches=[_make_batch(items)])

    enrichment = _make_enrich_result("electron_temperature")

    with (
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            new_callable=AsyncMock,
            return_value=_FakeLLMResult(enrichment, cost=0.123, tokens=4567),
        ),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            return_value="mock prompt",
        ),
        patch(
            "imas_codex.settings.get_model",
            return_value="test-model",
        ),
    ):
        from imas_codex.standard_names.enrich_workers import enrich_document_worker

        await enrich_document_worker(state)

    assert state.cost == pytest.approx(0.123)
    assert state.tokens_in == 4567
    assert state.document_stats.cost == pytest.approx(0.123)
    assert state.document_stats.processed == 1


@pytest.mark.asyncio
async def test_document_parse_error_marks_batch_failed():
    """LLM parse error → batch marked failed, errors incremented."""
    items = [_make_item("electron_temperature"), _make_item("ion_temperature")]
    state = _make_state(
        batches=[
            _make_batch(items, batch_index=0),
            _make_batch([_make_item("plasma_current")], batch_index=1),
        ]
    )

    # First batch raises parse error, second succeeds
    enrichment_2 = _make_enrich_result("plasma_current")
    call_count = 0

    async def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Failed to parse LLM response after 3 attempts")
        return _FakeLLMResult(enrichment_2, cost=0.01, tokens=300)

    with (
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            side_effect=_side_effect,
        ),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            return_value="mock prompt",
        ),
        patch(
            "imas_codex.settings.get_model",
            return_value="test-model",
        ),
    ):
        from imas_codex.standard_names.enrich_workers import enrich_document_worker

        await enrich_document_worker(state)

    # First batch marked as failed
    assert state.batches[0].get("failed") is True
    assert state.document_stats.errors == 2  # 2 items in first batch

    # Second batch still processed successfully
    assert state.batches[1]["items"][0].get("enriched_description") is not None

    # Pipeline didn't abort — phase completed
    assert state.document_phase.done


@pytest.mark.asyncio
async def test_document_preserves_readonly_unit():
    """LLM response cannot overwrite unit — original DD unit preserved."""
    items = [_make_item("electron_temperature", unit="eV")]
    state = _make_state(batches=[_make_batch(items)])

    enrichment = _make_enrich_result("electron_temperature")

    with (
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            new_callable=AsyncMock,
            return_value=_FakeLLMResult(enrichment),
        ),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            return_value="mock prompt",
        ),
        patch(
            "imas_codex.settings.get_model",
            return_value="test-model",
        ),
    ):
        from imas_codex.standard_names.enrich_workers import enrich_document_worker

        await enrich_document_worker(state)

    # unit is NOT overwritten — still the original DD value
    assert items[0]["unit"] == "eV"
    # Enriched fields are set
    assert items[0]["enriched_description"] is not None


@pytest.mark.asyncio
async def test_document_budget_exhausted_skips_remaining():
    """Budget exhausted between batches → remaining batches skipped."""
    batch_1_items = [_make_item("electron_temperature")]
    batch_2_items = [_make_item("ion_temperature")]

    # Set a very low cost limit so budget exhausts after first batch
    state = _make_state(
        batches=[
            _make_batch(batch_1_items, batch_index=0),
            _make_batch(batch_2_items, batch_index=1),
        ],
        cost_limit=0.05,  # $0.05 limit
    )

    enrichment = _make_enrich_result("electron_temperature")

    with (
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            new_callable=AsyncMock,
            return_value=_FakeLLMResult(enrichment, cost=0.10, tokens=500),
        ),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            return_value="mock prompt",
        ),
        patch(
            "imas_codex.settings.get_model",
            return_value="test-model",
        ),
    ):
        from imas_codex.standard_names.enrich_workers import enrich_document_worker

        await enrich_document_worker(state)

    # First batch processed (cost check is before LLM call, so first batch runs)
    assert batch_1_items[0].get("enriched_description") is not None

    # Second batch NOT processed (budget exhausted after first)
    assert batch_2_items[0].get("enriched_description") is None

    # Cost accumulated from first batch
    assert state.cost == pytest.approx(0.10)

    # Phase still completed
    assert state.document_phase.done


@pytest.mark.asyncio
async def test_document_stop_requested_between_batches():
    """stop_requested=True set between batches → stops cleanly."""
    batch_1_items = [_make_item("electron_temperature")]
    batch_2_items = [_make_item("ion_temperature")]
    state = _make_state(
        batches=[
            _make_batch(batch_1_items, batch_index=0),
            _make_batch(batch_2_items, batch_index=1),
        ]
    )

    enrichment = _make_enrich_result("electron_temperature")

    call_count = 0

    async def _side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # After first batch, set stop_requested
        state.stop_requested = True
        return _FakeLLMResult(enrichment, cost=0.01, tokens=300)

    with (
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            side_effect=_side_effect,
        ),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            return_value="mock prompt",
        ),
        patch(
            "imas_codex.settings.get_model",
            return_value="test-model",
        ),
    ):
        from imas_codex.standard_names.enrich_workers import enrich_document_worker

        await enrich_document_worker(state)

    # First batch processed
    assert batch_1_items[0].get("enriched_description") is not None

    # Second batch NOT processed (stop requested)
    assert batch_2_items[0].get("enriched_description") is None

    # Only one LLM call made
    assert call_count == 1


@pytest.mark.asyncio
async def test_document_dry_run_skips_llm():
    """Dry run mode skips LLM calls entirely."""
    items = [_make_item("electron_temperature")]
    state = _make_state(batches=[_make_batch(items)], dry_run=True)

    with patch(
        "imas_codex.discovery.base.llm.acall_llm_structured",
        new_callable=AsyncMock,
    ) as mock_llm:
        from imas_codex.standard_names.enrich_workers import enrich_document_worker

        await enrich_document_worker(state)

    # LLM never called
    mock_llm.assert_not_called()

    # Phase completed
    assert state.document_phase.done
    assert state.document_stats.total == 1
    assert state.document_stats.processed == 1


@pytest.mark.asyncio
async def test_document_empty_batches():
    """Empty batch list → worker completes without error."""
    state = _make_state(batches=[])

    from imas_codex.standard_names.enrich_workers import enrich_document_worker

    await enrich_document_worker(state)

    assert state.document_phase.done
    assert state.document_stats.processed == 0


@pytest.mark.asyncio
async def test_document_llm_missing_item_graceful():
    """LLM response missing one item → other items still enriched."""
    items = [_make_item("electron_temperature"), _make_item("ion_temperature")]
    state = _make_state(batches=[_make_batch(items)])

    # LLM only returns one of two items
    enrichment = _make_enrich_result("electron_temperature")

    with (
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            new_callable=AsyncMock,
            return_value=_FakeLLMResult(enrichment),
        ),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            return_value="mock prompt",
        ),
        patch(
            "imas_codex.settings.get_model",
            return_value="test-model",
        ),
    ):
        from imas_codex.standard_names.enrich_workers import enrich_document_worker

        await enrich_document_worker(state)

    # First item enriched
    assert items[0]["enriched_description"] is not None

    # Second item NOT enriched (missing from LLM response), but no error
    assert items[1].get("enriched_description") is None

    # Phase completed
    assert state.document_phase.done


@pytest.mark.asyncio
async def test_document_name_alias_injected():
    """Items get a ``name`` alias from ``id`` for template compatibility."""
    items = [_make_item("electron_temperature")]
    state = _make_state(batches=[_make_batch(items)])

    enrichment = _make_enrich_result("electron_temperature")

    with (
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            new_callable=AsyncMock,
            return_value=_FakeLLMResult(enrichment),
        ),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            return_value="mock prompt",
        ),
        patch(
            "imas_codex.settings.get_model",
            return_value="test-model",
        ),
    ):
        from imas_codex.standard_names.enrich_workers import enrich_document_worker

        await enrich_document_worker(state)

    # name alias exists and matches id
    assert items[0]["name"] == "electron_temperature"


# =============================================================================
# Integration test (requires OPENROUTER_API_KEY)
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
async def test_document_integration_real_llm():
    """Integration: single real batch of 2 items, real LLM call.

    Asserts cost > 0, description non-empty, unit unchanged.
    """
    items = [
        _make_item("electron_temperature", unit="eV"),
        _make_item("poloidal_magnetic_flux", unit="Wb", physics_domain="equilibrium"),
    ]
    state = _make_state(
        batches=[_make_batch(items)],
        cost_limit=5.0,
    )

    from imas_codex.standard_names.enrich_workers import enrich_document_worker

    await enrich_document_worker(state)

    # Cost accumulated
    assert state.cost > 0

    # Descriptions non-empty
    for item in items:
        assert item.get("enriched_description"), (
            f"Missing enriched_description for {item['id']}"
        )
        assert len(item["enriched_description"]) > 10

    # Units unchanged (DD-authoritative)
    assert items[0]["unit"] == "eV"
    assert items[1]["unit"] == "Wb"

    # Phase completed
    assert state.document_phase.done

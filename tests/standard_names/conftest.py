"""Shared fixtures for standard name tests."""

from __future__ import annotations

import pytest


@pytest.fixture()
def sample_standard_names() -> list[dict]:
    """Sample standard name dicts for write_standard_names testing."""
    return [
        {
            "id": "electron_temperature",
            "source_type": "dd",
            "source_id": "core_profiles/profiles_1d/electrons/temperature",
            "physical_base": "temperature",
            "subject": "electron",
            "description": "Electron temperature profile",
            "documentation": "The electron temperature $T_e$ is measured by Thomson scattering.",
            "kind": "scalar",
            "tags": ["core_profiles", "kinetics"],
            "links": ["ion_temperature", "electron_density"],
            "imas_paths": ["core_profiles/profiles_1d/electrons/temperature"],
            "validity_domain": "core plasma",
            "constraints": ["T_e > 0"],
            "unit": "eV",
            "model": "test/model",
            "review_status": "drafted",
            "generated_at": "2024-01-01T00:00:00Z",
            "confidence": 0.95,
        },
        {
            "id": "plasma_current",
            "source_type": "signals",
            "source_id": "tcv:ip/measured",
            "physical_base": "current",
            "description": "Plasma current",
            "unit": "A",
            "kind": "scalar",
            "tags": ["magnetics"],
            "model": "test/model",
            "review_status": "drafted",
            "generated_at": "2024-01-01T00:00:00Z",
            "confidence": 0.88,
        },
    ]


@pytest.fixture()
def mock_graph_client():
    """A mock GraphClient that records query calls."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.query = MagicMock(return_value=[])
    return client

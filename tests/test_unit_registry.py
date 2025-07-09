"""
Test the Data Dictionary unit registry with the provided unit aliases.
"""

import pytest

from imas_mcp.units import unit_registry


@pytest.mark.parametrize("dd_unit", ["mixed", "UTC", "m__pow__dimension"])
def test_unit_registry(dd_unit):
    assert f"{unit_registry.Unit(dd_unit)}" == "unit_error"


if __name__ == "__main__":
    pytest.main([__file__])

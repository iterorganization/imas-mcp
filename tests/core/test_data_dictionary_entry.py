import pytest

from imas_mcp.search_result import DataDictionaryEntry


def test_units_validator():
    dd_entry = DataDictionaryEntry(path="ids/path", documentation="docs", units="m/s^2")
    assert dd_entry.units == "m.s^-2"


def test_no_units_validator():
    dd_entry = DataDictionaryEntry(path="ids/path", documentation="docs")
    assert dd_entry.units == ""


if __name__ == "__main__":
    pytest.main([__file__])

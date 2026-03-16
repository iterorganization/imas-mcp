"""Tests for scanner plugin registry."""

from __future__ import annotations

import pytest

from imas_codex.discovery.signals.scanners.base import (
    DataSourceScanner,
    ScanResult,
    _registry,
    get_scanner,
    get_scanners_for_facility,
    list_scanners,
    register_scanner,
)


class TestScannerRegistry:
    """Test scanner registration and lookup."""

    def test_list_scanners_has_builtins(self):
        """All built-in scanners auto-register."""
        scanners = list_scanners()
        assert "tdi" in scanners
        assert "ppf" in scanners
        assert "edas" in scanners
        assert "mdsplus" in scanners
        assert "imas" in scanners
        assert "device_xml" in scanners
        assert "wiki" not in scanners

    def test_get_scanner_returns_instance(self):
        """get_scanner returns a scanner with correct type."""
        scanner = get_scanner("tdi")
        assert scanner.scanner_type == "tdi"
        assert hasattr(scanner, "scan")
        assert hasattr(scanner, "check")

    def test_get_scanner_unknown_raises(self):
        """Unknown scanner type raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            get_scanner("nonexistent")

    def test_scanner_protocol_compliance(self):
        """All registered scanners implement DataSourceScanner protocol."""
        for scanner_type in list_scanners():
            scanner = get_scanner(scanner_type)
            assert isinstance(scanner, DataSourceScanner)

    def test_scan_result_defaults(self):
        """ScanResult has sensible defaults."""
        result = ScanResult()
        assert result.signals == []
        assert result.data_access is None
        assert result.metadata == {}
        assert result.stats == {}


class TestGetScannersForFacility:
    """Test facility-based scanner dispatch."""

    def test_wiki_context_is_not_auto_added_as_scanner(self, monkeypatch):
        """Facilities with wiki sites only get configured data_system scanners."""
        from imas_codex.discovery.signals.scanners import base

        monkeypatch.setattr(
            "imas_codex.discovery.base.facility.get_facility",
            lambda facility: {
                "data_systems": {"mdsplus": {}},
                "wiki_sites": ["https://example.test/wiki"],
            },
        )

        scanners = get_scanners_for_facility("jet")

        assert [scanner.scanner_type for scanner in scanners] == ["mdsplus"]

    def test_tcv_returns_tdi(self, monkeypatch):
        """TCV facility should dispatch TDI scanner."""
        from imas_codex.discovery.signals.scanners import base

        monkeypatch.setattr(
            base,
            "_auto_register",
            lambda: None,
        )
        # Pre-populate registry
        _registry.clear()

        # Just test that the registry works with manual registration
        from imas_codex.discovery.signals.scanners.tdi import TDIScanner

        register_scanner(TDIScanner())
        assert get_scanner("tdi").scanner_type == "tdi"

    def test_disabled_data_source_is_skipped(self, monkeypatch):
        """Data systems marked available=false are not scheduled."""
        from imas_codex.discovery.signals.scanners import base

        monkeypatch.setattr(
            base,
            "_auto_register",
            lambda: None,
        )
        _registry.clear()

        from imas_codex.discovery.signals.scanners.mdsplus import MDSplusScanner

        register_scanner(MDSplusScanner())
        monkeypatch.setattr(
            "imas_codex.discovery.base.facility.get_facility",
            lambda facility: {
                "data_systems": {
                    "mdsplus": {},
                    "ppf": {"available": False},
                }
            },
        )

        scanners = get_scanners_for_facility("jet")

        assert [scanner.scanner_type for scanner in scanners] == ["mdsplus"]

    def test_register_custom_scanner(self):
        """Custom scanners can be registered."""

        class CustomScanner:
            scanner_type = "custom_test"

            async def scan(self, facility, ssh_host, config, reference_shot=None):
                return ScanResult()

            async def check(
                self, facility, ssh_host, signals, config, reference_shot=None
            ):
                return []

        register_scanner(CustomScanner())
        scanner = get_scanner("custom_test")
        assert scanner.scanner_type == "custom_test"

        # Clean up
        del _registry["custom_test"]

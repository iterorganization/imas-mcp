"""E2E tests for DeviceXMLScanner — JET machine description geometry.

Tests the full scan → persist → query cycle using mock SSH data.
Validates DataSource, SignalEpoch, SignalNode, and FacilitySignal creation.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from imas_codex.discovery.signals.scanners.base import get_scanner
from imas_codex.discovery.signals.scanners.device_xml import (
    JEC2020_SYSTEM_MAP,
    PPF_GEOMETRY_CROSSREFS,
    SECTION_METADATA,
    DeviceXMLScanner,
    _build_data_access,
    _build_jec2020_data_access,
    _make_signal_id,
    _make_signal_name,
    _persist_graph_nodes,
    _persist_jec2020_nodes,
    _persist_mcfg_nodes,
    _persist_ppf_static_nodes,
)

# Minimal parsed output matching what parse_device_xml.py returns
MOCK_PARSED_OUTPUT = {
    "versions": {
        "p89440": {
            "magprobes": [
                {
                    "id": "1",
                    "r": 4.292,
                    "z": 0.604,
                    "angle": -74.1,
                    "abs_error": 0.003,
                    "rel_error": 0.001,
                    "file": "MAGN",
                    "signal": "BPME(1)",
                },
                {
                    "id": "2",
                    "r": 4.281,
                    "z": 0.724,
                    "angle": -73.5,
                    "abs_error": 0.003,
                    "rel_error": 0.001,
                    "file": "MAGN",
                    "signal": "BPME(2)",
                },
            ],
            "flux": [
                {
                    "id": "1",
                    "r": 2.087,
                    "z": 1.795,
                    "dphi": 360.0,
                    "abs_error": 0.001,
                    "rel_error": 0.001,
                },
            ],
            "pfcoils": [
                {
                    "id": "1",
                    "r": 2.150,
                    "z": 1.780,
                    "dr": 0.264,
                    "dz": 0.574,
                    "turnsperelement": 120.0,
                    "abs_error": 0.001,
                    "rel_error": 0.001,
                },
                {
                    "id": "3",
                    "r": [3.827, 3.864, 3.901],
                    "z": [-1.457, -1.457, -1.457],
                    "dr": [0.068, 0.068, 0.068],
                    "dz": [0.378, 0.378, 0.378],
                    "turnsperelement": [6.0, 6.0, 6.0],
                },
            ],
            "pfcircuits": [
                {
                    "id": "1",
                    "coil_connect": "P1+P2",
                    "supply_connect": "S1",
                },
            ],
            "pfpassive": [
                {
                    "id": "1",
                    "r": 3.95,
                    "z": 1.60,
                    "dr": 0.05,
                    "dz": 0.40,
                    "ang1": 0.0,
                    "ang2": 360.0,
                    "resistance": 1.5e-4,
                    "abs_error": 0.001,
                    "rel_error": 0.001,
                },
            ],
            "enabled_probes": ["BPME(1)", "BPME(2)"],
            "disabled_probes": [],
        },
    },
    "limiters": {
        "Mk2ILW": {
            "r": [2.0, 2.5, 3.0, 3.5, 3.0, 2.5, 2.0],
            "z": [1.5, 1.8, 1.5, 0.0, -1.5, -1.8, -1.5],
            "n_points": 7,
        },
    },
}

MOCK_JET_CONFIG = {
    "git_repo": "/home/chain1/git/efit_f90.git",
    "input_prefix": "JET/input",
    "versions": [
        {
            "version": "p89440",
            "first_shot": 89440,
            "last_shot": 90539,
            "description": "P802B→P802A replacement",
            "device_xml": "Devices/device_p89440.xml",
            "snap_file": "Snap_files/EFITSNAP/efitsnap_p89440_bound0",
        },
    ],
    "limiter_versions": [
        {
            "name": "Mk2ILW",
            "first_shot": 79854,
            "description": "ITER-Like Wall (beryllium/tungsten)",
            "file": "Limiters/limiter.mk2ilw_cc",
        },
    ],
    "systems": [
        {"symbol": "PF", "name": "Poloidal field coils", "size": 22},
        {"symbol": "MP", "name": "Magnetic probes"},
    ],
}


class TestSignalNaming:
    """Test signal ID and name generation."""

    def test_make_signal_name(self):
        assert _make_signal_name("magprobes", "1", "r") == "bpme_1_r"
        assert _make_signal_name("flux", "3", "z") == "flux_loop_3_z"
        assert _make_signal_name("pfcoils", "2", "dr") == "pf_coil_2_dr"
        assert (
            _make_signal_name("pfpassive", "5", "resistance") == "passive_5_resistance"
        )

    def test_make_signal_id(self):
        sig_id = _make_signal_id("jet", "magprobes", "1", "r")
        assert sig_id == "jet:magnetic_field_diagnostics/bpme_1_r"

    def test_signal_id_includes_physics_domain(self):
        for section, meta in SECTION_METADATA.items():
            sig_id = _make_signal_id("jet", section, "1", "r")
            assert meta["physics_domain"] in sig_id


class TestDataAccess:
    """Test DataAccess node construction."""

    def test_build_data_access(self):
        da = _build_data_access("jet", MOCK_JET_CONFIG)
        assert da.id == "jet:device_xml:git"
        assert da.facility_id == "jet"
        assert da.method_type == "device_xml"
        assert da.library == "xml.etree.ElementTree"
        assert da.access_type == "local"
        assert da.data_source == "/home/chain1/git/efit_f90.git"
        assert "git" in da.connection_template
        assert "xml_bytes" in da.connection_template

    def test_full_example_is_executable_python(self):
        da = _build_data_access("jet", MOCK_JET_CONFIG)
        assert da.full_example is not None
        # Verify it's valid Python syntax
        compile(da.full_example, "<test>", "exec")


class TestScannerRegistration:
    """Test scanner protocol compliance."""

    def test_device_xml_in_registry(self):
        scanner = get_scanner("device_xml")
        assert scanner.scanner_type == "device_xml"

    def test_scanner_protocol(self):
        from imas_codex.discovery.signals.scanners.base import DataSourceScanner

        scanner = DeviceXMLScanner()
        assert isinstance(scanner, DataSourceScanner)


class TestScannerScan:
    """Test scan() method with mocked SSH."""

    @pytest.mark.anyio
    async def test_scan_returns_scan_result(self):
        """scan() should return ScanResult with stats."""
        scanner = DeviceXMLScanner()

        mock_output = json.dumps(MOCK_PARSED_OUTPUT)

        with (
            patch(
                "imas_codex.discovery.signals.scanners.device_xml.run_python_script",
                return_value=mock_output,
            ),
            patch(
                "imas_codex.discovery.signals.scanners.device_xml.GraphClient",
            ) as mock_gc_cls,
        ):
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = []
            mock_gc.create_nodes.return_value = {"processed": 0, "relationships": {}}

            result = await scanner.scan(
                facility="jet",
                ssh_host="jet",
                config=MOCK_JET_CONFIG,
            )

        assert result.data_access is not None
        assert result.data_access.id == "jet:device_xml:git"
        assert result.stats.get("epochs") is not None

    @pytest.mark.anyio
    async def test_scan_no_git_repo_returns_error(self):
        """scan() with no git_repo returns error stat."""
        scanner = DeviceXMLScanner()
        result = await scanner.scan(
            facility="jet",
            ssh_host="jet",
            config={},
        )
        assert "error" in result.stats

    @pytest.mark.anyio
    async def test_scan_ssh_failure_returns_error(self):
        """scan() handles SSH failures gracefully."""
        scanner = DeviceXMLScanner()

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.run_python_script",
            side_effect=RuntimeError("SSH connection refused"),
        ):
            result = await scanner.scan(
                facility="jet",
                ssh_host="jet",
                config=MOCK_JET_CONFIG,
            )

        assert "error" in result.stats


class TestPersistGraphNodes:
    """Test graph node creation logic."""

    def test_persist_creates_expected_node_types(self):
        """_persist_graph_nodes creates DataSource, epochs, nodes, signals."""
        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = []
            mock_gc.create_nodes.return_value = {"processed": 0, "relationships": {}}

            stats = _persist_graph_nodes(
                "jet",
                MOCK_JET_CONFIG,
                MOCK_PARSED_OUTPUT["versions"],
                MOCK_PARSED_OUTPUT["limiters"],
            )

        assert stats["epochs"] == 1
        assert stats["data_nodes"] > 0
        assert stats["signals"] > 0
        assert stats["limiter_nodes"] == 1

    def test_signal_count_matches_expected(self):
        """Number of signals matches instances × fields."""
        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = []
            mock_gc.create_nodes.return_value = {"processed": 0, "relationships": {}}

            stats = _persist_graph_nodes(
                "jet",
                MOCK_JET_CONFIG,
                MOCK_PARSED_OUTPUT["versions"],
                MOCK_PARSED_OUTPUT["limiters"],
            )

        # Expected signals:
        # 2 magprobes × 3 fields (r,z,angle) = 6
        # 1 flux × 3 fields (r,z,dphi) = 3
        # 2 pfcoils × 5 fields = 10
        # 1 pfcircuit × 2 fields = 2
        # 1 pfpassive × 7 fields = 7
        # 1 limiter = 1
        # Total = 29
        assert stats["signals"] == 29

    def test_data_node_has_geometry_values(self):
        """SignalNode dicts include r, z, angle etc. as properties."""
        created_nodes: list[list[dict]] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = []
            mock_gc.create_nodes.side_effect = lambda label, items, **kw: (
                created_nodes.append(items)
                or {"processed": len(items), "relationships": {}}
            )

            _persist_graph_nodes(
                "jet",
                MOCK_JET_CONFIG,
                MOCK_PARSED_OUTPUT["versions"],
                MOCK_PARSED_OUTPUT["limiters"],
            )

        # Find SignalNode calls (exclude FacilitySignal and DataAccess)
        data_node_batches = [
            batch
            for batch in created_nodes
            if batch and "path" in batch[0] and "node_type" in batch[0]
        ]
        assert len(data_node_batches) > 0

        # Check first magprobe node has geometry values
        all_dns = [dn for batch in data_node_batches for dn in batch]
        magprobe_dns = [dn for dn in all_dns if "magprobes:1" in dn["path"]]
        assert len(magprobe_dns) == 1
        dn = magprobe_dns[0]
        assert dn["r"] == 4.292
        assert dn["z"] == 0.604
        assert dn["angle"] == -74.1

    def test_multi_element_coil_has_array_properties(self):
        """Multi-element PF coil SignalNode stores R/Z as float arrays."""
        created_nodes: list[list[dict]] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = []
            mock_gc.create_nodes.side_effect = lambda label, items, **kw: (
                created_nodes.append(items)
                or {"processed": len(items), "relationships": {}}
            )

            _persist_graph_nodes(
                "jet",
                MOCK_JET_CONFIG,
                MOCK_PARSED_OUTPUT["versions"],
                MOCK_PARSED_OUTPUT["limiters"],
            )

        data_node_batches = [
            batch
            for batch in created_nodes
            if batch and "path" in batch[0] and "node_type" in batch[0]
        ]
        all_dns = [dn for batch in data_node_batches for dn in batch]
        coil3_dns = [dn for dn in all_dns if "pfcoils:3" in dn["path"]]
        assert len(coil3_dns) == 1
        dn = coil3_dns[0]
        assert isinstance(dn["r"], list)
        assert dn["r"] == [3.827, 3.864, 3.901]
        assert isinstance(dn["z"], list)
        assert dn["z"] == [-1.457, -1.457, -1.457]

    def test_data_node_has_system_property(self):
        """SignalNode includes system property for domain filtering."""
        created_nodes: list[list[dict]] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = []
            mock_gc.create_nodes.side_effect = lambda label, items, **kw: (
                created_nodes.append(items)
                or {"processed": len(items), "relationships": {}}
            )

            _persist_graph_nodes(
                "jet",
                MOCK_JET_CONFIG,
                MOCK_PARSED_OUTPUT["versions"],
                MOCK_PARSED_OUTPUT["limiters"],
            )

        data_node_batches = [
            batch
            for batch in created_nodes
            if batch and "path" in batch[0] and "node_type" in batch[0]
        ]
        all_dns = [dn for batch in data_node_batches for dn in batch]

        # Magprobes have system=MP
        mp_dns = [dn for dn in all_dns if "magprobes:" in dn["path"]]
        assert all(dn["system"] == "MP" for dn in mp_dns)

        # PF coils have system=PF
        pf_dns = [dn for dn in all_dns if "pfcoils:" in dn["path"]]
        assert all(dn["system"] == "PF" for dn in pf_dns)

        # Passive structures have system=PS
        ps_dns = [dn for dn in all_dns if "pfpassive:" in dn["path"]]
        assert all(dn["system"] == "PS" for dn in ps_dns)

    def test_introduced_in_relationships_created(self):
        """INTRODUCED_IN relationships are created for all DataNodes."""
        query_calls: list[tuple] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.create_nodes.return_value = {"processed": 0, "relationships": {}}

            def capture_query(cypher, **kwargs):
                query_calls.append((cypher, kwargs))
                return []

            mock_gc.query.side_effect = capture_query

            _persist_graph_nodes(
                "jet",
                MOCK_JET_CONFIG,
                MOCK_PARSED_OUTPUT["versions"],
                MOCK_PARSED_OUTPUT["limiters"],
            )

        # Find INTRODUCED_IN query calls
        intro_calls = [(q, kw) for q, kw in query_calls if "INTRODUCED_IN" in q]
        assert len(intro_calls) >= 1
        # All records should point to the correct epoch
        for _, kw in intro_calls:
            for rec in kw["records"]:
                assert rec["epoch_id"] == "jet:device_xml:p89440"

    def test_limiter_node_has_contour_data(self):
        """Limiter SignalNode includes R,Z contour arrays."""
        created_nodes: list[list[dict]] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = []
            mock_gc.create_nodes.side_effect = lambda label, items, **kw: (
                created_nodes.append(items)
                or {"processed": len(items), "relationships": {}}
            )

            _persist_graph_nodes(
                "jet",
                MOCK_JET_CONFIG,
                MOCK_PARSED_OUTPUT["versions"],
                MOCK_PARSED_OUTPUT["limiters"],
            )

        all_dns = [dn for batch in created_nodes for dn in batch]
        limiter_dns = [dn for dn in all_dns if "limiter:" in dn.get("path", "")]
        assert len(limiter_dns) == 1
        assert limiter_dns[0]["r_contour"] == [2.0, 2.5, 3.0, 3.5, 3.0, 2.5, 2.0]
        assert limiter_dns[0]["z_contour"] == [1.5, 1.8, 1.5, 0.0, -1.5, -1.8, -1.5]
        assert limiter_dns[0]["n_points"] == 7


class TestScannerCheck:
    """Test check() validation method."""

    @pytest.mark.anyio
    async def test_check_valid_signal(self):
        """check() returns valid for signals with SignalNode data."""
        from imas_codex.graph.models import FacilitySignal

        scanner = DeviceXMLScanner()
        signal = FacilitySignal(
            id="jet:magnetic_field_diagnostics/bpme_1_r",
            facility_id="jet",
            physics_domain="magnetic_field_diagnostics",
            accessor="device_xml:magprobes/1/r",
            data_access="jet:device_xml:git",
            data_source_node="jet:device_xml:p89440:magprobes:1",
        )

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = [
                {"r": 4.292, "z": 0.604, "path": "jet:device_xml:p89440:magprobes:1"}
            ]

            results = await scanner.check(
                facility="jet",
                ssh_host="jet",
                signals=[signal],
                config=MOCK_JET_CONFIG,
            )

        assert len(results) == 1
        assert results[0]["valid"] is True

    @pytest.mark.anyio
    async def test_check_missing_data_source_node(self):
        """check() returns invalid for signals without data_source_node."""
        from imas_codex.graph.models import FacilitySignal

        scanner = DeviceXMLScanner()
        signal = FacilitySignal(
            id="jet:magnetic_field_diagnostics/bpme_1_r",
            facility_id="jet",
            physics_domain="magnetic_field_diagnostics",
            accessor="device_xml:magprobes/1/r",
            data_access="jet:device_xml:git",
        )

        with patch("imas_codex.discovery.signals.scanners.device_xml.GraphClient"):
            results = await scanner.check(
                facility="jet",
                ssh_host="jet",
                signals=[signal],
                config=MOCK_JET_CONFIG,
            )

        assert len(results) == 1
        assert results[0]["valid"] is False


class TestSectionMetadata:
    """Test section metadata configuration."""

    def test_all_sections_have_physics_domain(self):
        for section, meta in SECTION_METADATA.items():
            assert "physics_domain" in meta, f"Missing physics_domain for {section}"

    def test_all_sections_have_system(self):
        """Every section has a system code for domain filtering."""
        for section, meta in SECTION_METADATA.items():
            assert "system" in meta, f"Missing system for {section}"
            assert len(meta["system"]) > 0, f"Empty system for {section}"

    def test_all_sections_have_fields(self):
        for section, meta in SECTION_METADATA.items():
            assert "fields" in meta, f"Missing fields for {section}"
            assert len(meta["fields"]) > 0, f"Empty fields for {section}"

    def test_all_fields_have_unit_and_desc(self):
        for section, meta in SECTION_METADATA.items():
            for field, field_meta in meta["fields"].items():
                assert "unit" in field_meta, f"Missing unit for {section}.{field}"
                assert "desc" in field_meta, f"Missing desc for {section}.{field}"


class TestMultiVersionDedup:
    """Test signal deduplication across epochs sharing the same XML."""

    def test_shared_xml_produces_unique_signals(self):
        """Multiple epochs using same XML should not duplicate signals."""
        multi_version_config = {
            "git_repo": "/repo",
            "input_prefix": "JET/input",
            "versions": [
                {
                    "version": "p68613",
                    "first_shot": 68613,
                    "last_shot": 78358,
                    "description": "Baseline",
                    "device_xml": "Devices/device_p68613.xml",
                },
                {
                    "version": "p78359",
                    "first_shot": 78359,
                    "last_shot": 79853,
                    "description": "New DMSS config",
                    "device_xml": "Devices/device_p68613.xml",  # Same XML!
                },
            ],
            "limiter_versions": [],
        }

        # Both versions return identical geometry
        parsed = {
            "p68613": {
                "magprobes": [{"id": "1", "r": 4.0, "z": 0.5, "angle": -70.0}],
                "flux": [],
                "pfcoils": [],
                "pfcircuits": [],
                "pfpassive": [],
            },
            "p78359": {
                "magprobes": [{"id": "1", "r": 4.0, "z": 0.5, "angle": -70.0}],
                "flux": [],
                "pfcoils": [],
                "pfcircuits": [],
                "pfpassive": [],
            },
        }

        signal_calls: list[list[dict]] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = []

            def capture_create(label, items, **kw):
                if label == "FacilitySignal":
                    signal_calls.append(items)
                return {"processed": len(items), "relationships": {}}

            mock_gc.create_nodes.side_effect = capture_create

            stats = _persist_graph_nodes("jet", multi_version_config, parsed, {})

        # Should have exactly 3 signals (1 probe × 3 fields), not 6
        assert stats["signals"] == 3
        assert stats["epochs"] == 2
        # DataNodes should be created for BOTH epochs though
        assert stats["data_nodes"] == 2  # One SignalNode per epoch per instance


class TestParseDeviceXMLScript:
    """Test the remote parse script in isolation."""

    def test_parse_device_xml_function(self):
        """Test parse_device_xml() with inline XML."""
        # Import the parse function from the remote script
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "parse_device_xml",
            "imas_codex/remote/scripts/parse_device_xml.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Minimal device XML
        xml_bytes = b"""<?xml version="1.0"?>
<device>
  <magprobes>
    <instance id="1" file="MAGN" signal="BPME(1)">
      <r>4.292</r>
      <z>0.604</z>
      <angle>-74.1</angle>
      <abs_error>0.003</abs_error>
      <rel_error>0.001</rel_error>
    </instance>
  </magprobes>
  <pfcoils>
    <instance id="1">
      <r>2.15</r>
      <z>1.78</z>
      <dr>0.264</dr>
      <dz>0.574</dz>
      <turnsperelement>120</turnsperelement>
    </instance>
  </pfcoils>
</device>"""
        result = mod.parse_device_xml(xml_bytes)
        assert "magprobes" in result
        assert len(result["magprobes"]) == 1
        assert result["magprobes"][0]["r"] == 4.292
        assert result["magprobes"][0]["z"] == 0.604
        assert "pfcoils" in result
        assert result["pfcoils"][0]["r"] == 2.15

    def test_parse_limiter_file(self):
        """Test limiter contour parsing."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "parse_device_xml",
            "imas_codex/remote/scripts/parse_device_xml.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        data = b"3\n2.0 1.5\n2.5 1.8\n3.0 1.5\n"
        result = mod.parse_limiter_file(data)
        assert result["r"] == [2.0, 2.5, 3.0]
        assert result["z"] == [1.5, 1.8, 1.5]
        assert result["n_points"] == 3

    def test_parse_limiter_file_with_segment_header(self):
        """Test limiter contour parsing with segment count header (chain1 format)."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "parse_device_xml",
            "imas_codex/remote/scripts/parse_device_xml.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Format matching the real mk2a file: segment count + R,Z pairs
        # then second segment (should be ignored)
        data = (
            b"  3\n"
            b" 1.82396       -.02263\n"
            b" 1.83565       -.12204\n"
            b" 1.85062       -.22101\n"
            b"    2\n"
            b"2.3921      -0.3922\n"
            b"2.4123      -0.4011\n"
        )
        result = mod.parse_limiter_file(data)
        # Only first segment extracted (3 points)
        assert result["n_points"] == 3
        assert result["r"][0] == pytest.approx(1.82396)
        assert result["z"][0] == pytest.approx(-0.02263)
        assert len(result["r"]) == 3

    def test_parse_multi_element_coil(self):
        """Multi-value R/Z text in PF coils is parsed as float arrays."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "parse_device_xml",
            "imas_codex/remote/scripts/parse_device_xml.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        xml_bytes = b"""<?xml version="1.0"?>
<device>
  <pfcoils>
    <instance id="3">
      <r>3.827 3.864 3.901</r>
      <z>-1.457 -1.457 -1.457</z>
      <dr>0.068 0.068 0.068</dr>
      <dz>0.378 0.378 0.378</dz>
      <turnsperelement>6 6 6</turnsperelement>
    </instance>
    <instance id="1">
      <r>2.15</r>
      <z>1.78</z>
      <dr>0.264</dr>
      <dz>0.574</dz>
      <turnsperelement>120</turnsperelement>
    </instance>
  </pfcoils>
</device>"""
        result = mod.parse_device_xml(xml_bytes)
        coils = result["pfcoils"]
        assert len(coils) == 2

        # Multi-element coil: arrays
        c3 = coils[0]
        assert c3["id"] == "3"
        assert isinstance(c3["r"], list)
        assert c3["r"] == [3.827, 3.864, 3.901]
        assert isinstance(c3["z"], list)
        assert c3["z"] == [-1.457, -1.457, -1.457]

        # Single-element coil: scalar float
        c1 = coils[1]
        assert c1["id"] == "1"
        assert isinstance(c1["r"], float)
        assert c1["r"] == 2.15


class TestChain1LimiterParsing:
    """Test filesystem-based limiter file reading (chain1 directory)."""

    def test_chain1_limiter_produces_same_format_as_git(self):
        """Limiter files from chain1 use identical R,Z format as git files."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "parse_device_xml",
            "imas_codex/remote/scripts/parse_device_xml.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Both sources use the same segment-count + R,Z format
        git_data = b"3\n2.0 1.5\n2.5 1.8\n3.0 1.5\n"
        git_result = mod.parse_limiter_file(git_data)

        chain1_data = b"  3\n2.0 1.5\n2.5 1.8\n3.0 1.5\n"
        chain1_result = mod.parse_limiter_file(chain1_data)

        # Both should produce identical R,Z contours
        assert git_result["r"] == chain1_result["r"]
        assert git_result["z"] == chain1_result["z"]
        assert git_result["n_points"] == chain1_result["n_points"]


class TestLimiterProvenance:
    """Test limiter SignalNode provenance tracking (file_source, file_path)."""

    def test_limiter_node_has_provenance(self):
        """Limiter SignalNode includes file_source and file_path properties."""
        parsed_limiters = {
            "Mk2A": {
                "r": [1.8, 1.9, 2.0],
                "z": [0.0, -0.1, -0.2],
                "n_points": 3,
                "file_source": "filesystem",
                "file_path": "/home/chain1/input/efit/Limiters/limiter.mk2a",
            },
            "Mk2ILW": {
                "r": [2.0, 2.5, 3.0],
                "z": [1.5, 0.0, -1.5],
                "n_points": 3,
                "file_source": "git",
                "file_path": "JET/input/Limiters/limiter.mk2ilw_cc",
            },
        }
        config = {
            **MOCK_JET_CONFIG,
            "limiter_versions": [
                {
                    "name": "Mk2A",
                    "first_shot": 1,
                    "last_shot": 44414,
                    "file": "limiter.mk2a",
                    "source_dir": "/home/chain1/input/efit/Limiters",
                },
                {
                    "name": "Mk2ILW",
                    "first_shot": 79854,
                    "file": "Limiters/limiter.mk2ilw_cc",
                },
            ],
        }

        created_nodes: list[list[dict]] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = []
            mock_gc.create_nodes.side_effect = lambda label, items, **kw: (
                created_nodes.append(items)
                or {"processed": len(items), "relationships": {}}
            )

            _persist_graph_nodes(
                "jet", config, MOCK_PARSED_OUTPUT["versions"], parsed_limiters
            )

        all_dns = [dn for batch in created_nodes for dn in batch]
        limiter_dns = [dn for dn in all_dns if "limiter:" in dn.get("path", "")]
        assert len(limiter_dns) == 2

        mk2a = next(dn for dn in limiter_dns if "Mk2A" in dn["path"])
        assert mk2a["file_source"] == "filesystem"
        assert mk2a["file_path"] == "/home/chain1/input/efit/Limiters/limiter.mk2a"

        mk2ilw = next(dn for dn in limiter_dns if "Mk2ILW" in dn["path"])
        assert mk2ilw["file_source"] == "git"
        assert mk2ilw["file_path"] == "JET/input/Limiters/limiter.mk2ilw_cc"


class TestUsesLimiterRelationship:
    """Test SignalEpoch → Limiter SignalNode USES_LIMITER relationships."""

    def test_uses_limiter_relationships_created(self):
        """USES_LIMITER graph queries are invoked with correct epoch/limiter pairs."""
        config = {
            "git_repo": "/repo",
            "input_prefix": "JET/input",
            "versions": [
                {
                    "version": "p68613",
                    "first_shot": 68613,
                    "last_shot": 79853,
                    "description": "Mk2HD era",
                    "device_xml": "Devices/device_p68613.xml",
                    "limiter": "Limiters/limiter.mk2hd_cc",
                },
                {
                    "version": "p79854",
                    "first_shot": 79854,
                    "description": "ILW era",
                    "device_xml": "Devices/device_p68613.xml",
                    "limiter": "Limiters/limiter.mk2ilw_cc",
                },
            ],
            "limiter_versions": [
                {
                    "name": "Mk2HD",
                    "first_shot": 63446,
                    "last_shot": 79853,
                    "file": "Limiters/limiter.mk2hd_cc",
                },
                {
                    "name": "Mk2ILW",
                    "first_shot": 79854,
                    "file": "Limiters/limiter.mk2ilw_cc",
                },
            ],
        }
        parsed = {
            "p68613": {
                "magprobes": [{"id": "1", "r": 4.0, "z": 0.5, "angle": -70.0}],
                "flux": [],
                "pfcoils": [],
                "pfcircuits": [],
                "pfpassive": [],
            },
        }
        parsed_limiters = {
            "Mk2HD": {"r": [2.0, 2.5], "z": [1.0, 0.0], "n_points": 2},
            "Mk2ILW": {"r": [2.0, 3.0], "z": [1.5, -1.5], "n_points": 2},
        }

        query_calls: list[tuple] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = []
            mock_gc.create_nodes.return_value = {"processed": 0, "relationships": {}}

            def capture_query(cypher, **kwargs):
                query_calls.append((cypher, kwargs))
                return []

            mock_gc.query.side_effect = capture_query

            _persist_graph_nodes("jet", config, parsed, parsed_limiters)

        # Find the USES_LIMITER query call
        ul_calls = [(q, kw) for q, kw in query_calls if "USES_LIMITER" in q]
        assert len(ul_calls) == 1
        records = ul_calls[0][1]["records"]
        assert len(records) == 2
        assert records[0]["epoch_id"] == "jet:device_xml:p68613"
        assert records[0]["limiter_path"] == "jet:device_xml:limiter:Mk2HD"
        assert records[1]["epoch_id"] == "jet:device_xml:p79854"
        assert records[1]["limiter_path"] == "jet:device_xml:limiter:Mk2ILW"

    def test_uses_limiter_direct_name_matching(self):
        """Pre-EFIT++ epochs use uses_limiter for direct name matching."""
        config = {
            "git_repo": "/repo",
            "input_prefix": "JET/input",
            "versions": [
                {
                    "version": "limiter_era",
                    "first_shot": 1,
                    "last_shot": 28791,
                    "description": "Pre-divertor limiter era",
                    "uses_limiter": "Limiter",
                    "wall_configuration": "limiter",
                },
                {
                    "version": "p68613",
                    "first_shot": 68613,
                    "last_shot": 74386,
                    "description": "EFIT++ baseline",
                    "device_xml": "Devices/device_p68613.xml",
                    "limiter": "Limiters/limiter.mk2hd_cc",
                },
            ],
            "limiter_versions": [
                {
                    "name": "Limiter",
                    "first_shot": 1,
                    "last_shot": 28791,
                    "file": "L91NEW",
                    "source_dir": "/usr/local/data/flush",
                },
                {
                    "name": "Mk2HD",
                    "first_shot": 63446,
                    "last_shot": 79853,
                    "file": "Limiters/limiter.mk2hd_cc",
                },
            ],
        }
        parsed = {
            "p68613": {
                "magprobes": [],
                "flux": [],
                "pfcoils": [],
                "pfcircuits": [],
                "pfpassive": [],
            },
        }
        parsed_limiters = {
            "Limiter": {
                "r": [2.0, 2.5],
                "z": [1.0, 0.0],
                "n_points": 2,
                "segments": [
                    {"r": [2.0, 2.5], "z": [1.0, 0.0], "n_points": 2},
                ],
            },
            "Mk2HD": {"r": [2.0, 2.4], "z": [-0.5, -1.3], "n_points": 2},
        }

        query_calls: list[tuple] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = []
            mock_gc.create_nodes.return_value = {"processed": 0, "relationships": {}}

            def capture_query(cypher, **kwargs):
                query_calls.append((cypher, kwargs))
                return []

            mock_gc.query.side_effect = capture_query

            _persist_graph_nodes("jet", config, parsed, parsed_limiters)

        ul_calls = [(q, kw) for q, kw in query_calls if "USES_LIMITER" in q]
        assert len(ul_calls) == 1
        records = ul_calls[0][1]["records"]
        assert len(records) == 2
        # Pre-EFIT++ epoch uses direct name matching
        assert records[0]["epoch_id"] == "jet:device_xml:limiter_era"
        assert records[0]["limiter_path"] == "jet:device_xml:limiter:Limiter"
        # EFIT++ epoch uses file-based matching
        assert records[1]["epoch_id"] == "jet:device_xml:p68613"
        assert records[1]["limiter_path"] == "jet:device_xml:limiter:Mk2HD"


class TestPreDivertorEpochs:
    """Test pre-divertor epoch handling — epochs without device_xml."""

    def test_epochs_without_device_xml_create_structural_epochs(self):
        """Pre-EFIT++ epochs are created as SignalEpoch nodes."""
        config = {
            "git_repo": "/repo",
            "input_prefix": "JET/input",
            "versions": [
                {
                    "version": "limiter_era",
                    "first_shot": 1,
                    "last_shot": 28791,
                    "description": "Pre-divertor limiter era",
                    "uses_limiter": "Limiter",
                    "wall_configuration": "limiter",
                },
                {
                    "version": "p68613",
                    "first_shot": 68613,
                    "last_shot": 74386,
                    "description": "EFIT++ baseline",
                    "device_xml": "Devices/device_p68613.xml",
                    "limiter": "Limiters/limiter.mk2hd_cc",
                    "wall_configuration": "divertor",
                },
            ],
            "limiter_versions": [
                {
                    "name": "Limiter",
                    "first_shot": 1,
                    "last_shot": 28791,
                    "file": "L91NEW",
                    "source_dir": "/usr/local/data/flush",
                },
            ],
        }
        parsed = {
            "p68613": {
                "magprobes": [{"id": "1", "r": 4.0, "z": 0.5, "angle": -70.0}],
                "flux": [],
                "pfcoils": [],
                "pfcircuits": [],
                "pfpassive": [],
            }
        }
        parsed_limiters = {
            "Limiter": {"r": [2.0, 2.5], "z": [1.0, 0.0], "n_points": 2},
        }

        query_calls: list[tuple] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.create_nodes.return_value = {"processed": 0, "relationships": {}}

            def capture_query(cypher, **kwargs):
                query_calls.append((cypher, kwargs))
                return []

            mock_gc.query.side_effect = capture_query

            stats = _persist_graph_nodes("jet", config, parsed, parsed_limiters)

        assert stats["epochs"] == 2

        # Find the epoch creation query
        epoch_calls = [
            (q, kw) for q, kw in query_calls if "SignalEpoch" in q and "UNWIND" in q
        ]
        assert len(epoch_calls) >= 1
        records = epoch_calls[0][1]["records"]
        assert len(records) == 2
        assert records[0]["id"] == "jet:device_xml:limiter_era"
        assert records[0]["wall_configuration"] == "limiter"
        assert records[1]["id"] == "jet:device_xml:p68613"
        assert records[1]["wall_configuration"] == "divertor"

    def test_epochs_without_device_xml_skip_data_node_creation(self):
        """Pre-EFIT++ epochs produce no DataNodes from XML parsing."""
        config = {
            "git_repo": "/repo",
            "input_prefix": "JET/input",
            "versions": [
                {
                    "version": "limiter_era",
                    "first_shot": 1,
                    "last_shot": 28791,
                    "description": "Pre-divertor",
                    "uses_limiter": "Limiter",
                    "wall_configuration": "limiter",
                },
            ],
            "limiter_versions": [
                {
                    "name": "Limiter",
                    "first_shot": 1,
                    "last_shot": 28791,
                    "file": "L91NEW",
                    "source_dir": "/usr/local/data/flush",
                },
            ],
        }
        parsed_limiters = {
            "Limiter": {"r": [2.0, 2.5], "z": [1.0, 0.0], "n_points": 2},
        }

        created_nodes: list[list[dict]] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = []
            mock_gc.create_nodes.side_effect = lambda label, items, **kw: (
                created_nodes.append(items)
                or {"processed": len(items), "relationships": {}}
            )

            stats = _persist_graph_nodes("jet", config, {}, parsed_limiters)

        assert stats["data_nodes"] == 0
        # Only limiter + signal + data_access batches, no SignalNode from XML
        all_dns = [dn for batch in created_nodes for dn in batch]
        xml_dns = [
            dn
            for dn in all_dns
            if dn.get("path", "").startswith("jet:device_xml:")
            and "limiter:" not in dn.get("path", "")
        ]
        assert len(xml_dns) == 0


class TestContourSections:
    """Test limiter contour_sections segment selection."""

    def test_contour_sections_selects_specific_segments(self):
        """contour_sections [1, 2] selects specified auxiliary segments."""
        config = {
            "git_repo": "/repo",
            "input_prefix": "JET/input",
            "versions": [],
            "limiter_versions": [
                {
                    "name": "Limiter",
                    "first_shot": 1,
                    "last_shot": 26087,
                    "file": "limiter.mk2a",
                    "source_dir": "/chain1/Limiters",
                    "contour_sections": [1, 2],
                },
            ],
        }
        parsed_limiters = {
            "Limiter": {
                "r": [2.0, 2.5, 3.0],
                "z": [1.0, 0.0, -1.0],
                "n_points": 3,
                "segments": [
                    {"r": [2.0, 2.5, 3.0], "z": [1.0, 0.0, -1.0], "n_points": 3},
                    {"r": [0.96, 1.5, 2.43], "z": [-1.42, 1.42, -1.42], "n_points": 3},
                    {"r": [2.38, 2.39], "z": [-0.39, 0.39], "n_points": 2},
                ],
            },
        }

        created_nodes: list[list[dict]] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = []
            mock_gc.create_nodes.side_effect = lambda label, items, **kw: (
                created_nodes.append(items)
                or {"processed": len(items), "relationships": {}}
            )

            _persist_graph_nodes("jet", config, {}, parsed_limiters)

        all_dns = [dn for batch in created_nodes for dn in batch]
        limiter_dns = [dn for dn in all_dns if "limiter:" in dn.get("path", "")]
        assert len(limiter_dns) == 1
        dn = limiter_dns[0]
        # Should concatenate segments 1 and 2 (not segment 0)
        assert dn["r_contour"] == [0.96, 1.5, 2.43, 2.38, 2.39]
        assert dn["z_contour"] == [-1.42, 1.42, -1.42, -0.39, 0.39]
        assert dn["n_points"] == 5

    def test_contour_sections_absent_uses_primary_segment(self):
        """Without contour_sections, uses primary segment (top-level r, z)."""
        config = {
            "git_repo": "/repo",
            "input_prefix": "JET/input",
            "versions": [],
            "limiter_versions": [
                {
                    "name": "Mk2A",
                    "first_shot": 1,
                    "file": "limiter.mk2a",
                    "source_dir": "/chain1/Limiters",
                },
            ],
        }
        parsed_limiters = {
            "Mk2A": {
                "r": [2.0, 2.5, 3.0],
                "z": [1.0, 0.0, -1.0],
                "n_points": 3,
                "segments": [
                    {"r": [2.0, 2.5, 3.0], "z": [1.0, 0.0, -1.0], "n_points": 3},
                    {"r": [0.96, 1.5], "z": [-1.42, 1.42], "n_points": 2},
                ],
            },
        }

        created_nodes: list[list[dict]] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = []
            mock_gc.create_nodes.side_effect = lambda label, items, **kw: (
                created_nodes.append(items)
                or {"processed": len(items), "relationships": {}}
            )

            _persist_graph_nodes("jet", config, {}, parsed_limiters)

        all_dns = [dn for batch in created_nodes for dn in batch]
        limiter_dns = [dn for dn in all_dns if "limiter:" in dn.get("path", "")]
        assert len(limiter_dns) == 1
        dn = limiter_dns[0]
        # Falls back to top-level r, z (primary segment)
        assert dn["r_contour"] == [2.0, 2.5, 3.0]
        assert dn["z_contour"] == [1.0, 0.0, -1.0]
        assert dn["n_points"] == 3

    def test_parse_limiter_file_returns_all_segments(self):
        """parse_limiter_file returns all segments in the segments list."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "parse_device_xml",
            "imas_codex/remote/scripts/parse_device_xml.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Three segments: 3 points, 2 points, 2 points
        data = (
            b"3\n2.0 1.0\n2.5 0.0\n3.0 -1.0\n"
            b"2\n0.96 -1.42\n2.43 1.42\n"
            b"2\n2.38 -0.39\n2.39 0.39\n"
        )
        result = mod.parse_limiter_file(data)

        # Top-level is segment 0
        assert result["n_points"] == 3
        assert result["r"] == [2.0, 2.5, 3.0]

        # All segments available
        assert len(result["segments"]) == 3
        assert result["segments"][0]["n_points"] == 3
        assert result["segments"][1]["n_points"] == 2
        assert result["segments"][1]["r"] == [0.96, 2.43]
        assert result["segments"][2]["n_points"] == 2
        assert result["segments"][2]["r"] == [2.38, 2.39]


class TestCompleteLimiterCoverage:
    """Test that all 5 limiter versions produce DataNodes when both sources used."""

    def test_all_five_limiters_ingested(self):
        """All 5 limiter contour files (Mk2A-Mk2ILW) produce DataNodes with R,Z data."""
        config = {
            "git_repo": "/repo",
            "input_prefix": "JET/input",
            "versions": [
                {
                    "version": "p68613",
                    "first_shot": 68613,
                    "device_xml": "Devices/device_p68613.xml",
                    "limiter": "Limiters/limiter.mk2hd_cc",
                },
            ],
            "limiter_versions": [
                {
                    "name": "Mk2A",
                    "first_shot": 1,
                    "last_shot": 44414,
                    "file": "limiter.mk2a",
                    "source_dir": "/chain1/Limiters",
                },
                {
                    "name": "Mk2GB",
                    "first_shot": 44415,
                    "last_shot": 54351,
                    "file": "limiter.mk2gb",
                    "source_dir": "/chain1/Limiters",
                },
                {
                    "name": "Mk2GB-NS",
                    "first_shot": 54352,
                    "last_shot": 63445,
                    "file": "limiter.mk2gb_ns_cc",
                },
                {
                    "name": "Mk2HD",
                    "first_shot": 63446,
                    "last_shot": 79853,
                    "file": "limiter.mk2hd_cc",
                },
                {"name": "Mk2ILW", "first_shot": 79854, "file": "limiter.mk2ilw_cc"},
            ],
        }
        parsed_limiters = {
            "Mk2A": {
                "r": [1.8, 1.9],
                "z": [0.0, -0.1],
                "n_points": 2,
                "file_source": "filesystem",
                "file_path": "/chain1/Limiters/limiter.mk2a",
            },
            "Mk2GB": {
                "r": [2.0, 2.1],
                "z": [-0.9, -1.1],
                "n_points": 2,
                "file_source": "filesystem",
                "file_path": "/chain1/Limiters/limiter.mk2gb",
            },
            "Mk2GB-NS": {
                "r": [2.0, 2.3],
                "z": [-0.9, -1.3],
                "n_points": 2,
                "file_source": "git",
                "file_path": "JET/input/Limiters/limiter.mk2gb_ns_cc",
            },
            "Mk2HD": {
                "r": [2.0, 2.4],
                "z": [-0.5, -1.3],
                "n_points": 2,
                "file_source": "git",
                "file_path": "JET/input/Limiters/limiter.mk2hd_cc",
            },
            "Mk2ILW": {
                "r": [2.0, 3.0],
                "z": [1.5, -1.5],
                "n_points": 2,
                "file_source": "git",
                "file_path": "JET/input/Limiters/limiter.mk2ilw_cc",
            },
        }

        created_nodes: list[list[dict]] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = []
            mock_gc.create_nodes.side_effect = lambda label, items, **kw: (
                created_nodes.append(items)
                or {"processed": len(items), "relationships": {}}
            )

            stats = _persist_graph_nodes(
                "jet",
                config,
                {
                    "p68613": {
                        "magprobes": [],
                        "flux": [],
                        "pfcoils": [],
                        "pfcircuits": [],
                        "pfpassive": [],
                    }
                },
                parsed_limiters,
            )

        assert stats["limiter_nodes"] == 5
        all_dns = [dn for batch in created_nodes for dn in batch]
        limiter_dns = [dn for dn in all_dns if "limiter:" in dn.get("path", "")]
        assert len(limiter_dns) == 5

        names = {dn["path"].split(":")[-1] for dn in limiter_dns}
        assert names == {"Mk2A", "Mk2GB", "Mk2GB-NS", "Mk2HD", "Mk2ILW"}


# =========================================================================
# JEC2020 Tests
# =========================================================================

# Minimal JEC2020 source config matching jet.yaml static_sources entry
MOCK_JEC2020_CONFIG = {
    "name": "jec2020_geometry",
    "description": "EFIT++ equilibrium geometry files",
    "format": "xml",
    "base_dir": "/home/chain1/jec2020",
    "reference_shot": 79951,
    "files": [
        {"path": "magnetics.xml", "role": "magnetics"},
        {"path": "pfSystems.xml", "role": "pf_coils"},
        {"path": "ironBoundaries3.xml", "role": "iron_core"},
        {"path": "limiter.xml", "role": "limiter"},
    ],
}

# Mock parsed output from parse_jec2020.py
MOCK_JEC2020_PARSED = {
    "magnetics": {
        "probes": [
            {
                "id": "1",
                "description": "Internal Discrete Coil, Oct.3",
                "rCentre": 4.292,
                "zCentre": 0.604,
                "poloidalOrientation": -74.1,
                "angle_units": "degrees",
                "ppf_signal": "BPME(1)",
                "jpf_signal": "DA/C2-CX01",
                "ppf_data_source": "JET::PPF::/magn/$pulseNumber$/0/jetppf",
                "jpf_data_source": "JPF::$pulseNumber$",
                "error_type": "relativeAbsolute",
                "rel_error": 0.02,
                "abs_error": 0.005,
            },
            {
                "id": "2",
                "description": "Internal Discrete Coil, Oct.3",
                "rCentre": 4.281,
                "zCentre": 0.724,
                "poloidalOrientation": -73.5,
                "ppf_signal": "BPME(2)",
                "jpf_signal": "DA/C2-CX02",
                "ppf_data_source": "JET::PPF::/magn/$pulseNumber$/0/jetppf",
                "jpf_data_source": "JPF::$pulseNumber$",
                "rel_error": 0.02,
                "abs_error": 0.005,
            },
        ],
        "flux_loops": [
            {
                "id": "1",
                "description": "Flux loop at inboard midplane",
                "rCentre": 2.0,
                "zCentre": 0.0,
                "ppf_signal": "FLME(1)",
                "jpf_signal": "DA/C2-FL01",
                "ppf_data_source": "JET::PPF",
                "jpf_data_source": "JPF",
            },
        ],
    },
    "pf_coils": {
        "coils": [
            {
                "id": "1",
                "name": "P1/ME",
                "rCentre": 0.897,
                "zCentre": 0.0,
                "dR": 0.337,
                "dZ": 5.427,
                "angle1": 0.0,
                "angle2": 0.0,
                "turnCount": 710.0,
            },
            {
                "id": "3",
                "name": "P2/SUI/8",
                # Multi-element coil with comma-separated arrays
                "rCentre": [1.967, 2.005, 2.043, 2.081],
                "zCentre": [3.871, 3.871, 3.871, 3.871],
                "dR": [0.035, 0.035, 0.035, 0.035],
                "dZ": [0.035, 0.035, 0.035, 0.035],
                "turnCount": [0.5, 0.5, 0.5, 0.5],
            },
        ],
        "circuits": [
            {
                "id": "1",
                "name": "P1U_circuit",
                "coil_ids": ["1"],
            },
        ],
    },
    "iron_core": {
        "material_id": "3",
        "material2_id": "1",
        "r": [6.512, 4.952, 3.392],
        "z": [4.45, 4.45, 4.45],
        "permeabilities": [852.82, 724.86, 887.43],
        "segment_lengths": [1.56, 1.56, 1.56],
        "n_segments": 3.0,
        "boundary_length": 4.68,
    },
    "limiter": {
        "r": [2.0, 2.5, 3.0, 3.5],
        "z": [1.0, 0.5, -0.5, -1.0],
        "n_points": 4,
    },
}


class TestJEC2020DataAccess:
    """Test JEC2020 DataAccess node construction."""

    def test_build_jec2020_data_access(self):
        da = _build_jec2020_data_access("jet", "/home/chain1/jec2020")
        assert da.id == "jet:jec2020:xml"
        assert da.method_type == "static_xml"
        assert da.data_source == "jec2020_geometry"
        assert da.library == "xml.etree.ElementTree"

    def test_full_example_is_valid_python(self):
        da = _build_jec2020_data_access("jet", "/home/chain1/jec2020")
        # Should be parseable Python (syntax check)
        compile(da.full_example, "<jec2020>", "exec")


class TestJEC2020SystemMap:
    """Test JEC2020 system code definitions."""

    def test_all_roles_have_system_codes(self):
        expected_roles = {
            "magnetics_probe",
            "magnetics_flux",
            "pf_coils",
            "pf_circuits",
            "iron_core",
            "limiter",
        }
        assert set(JEC2020_SYSTEM_MAP.keys()) == expected_roles


class TestJEC2020Persist:
    """Test _persist_jec2020_nodes with mock graph."""

    def _run_persist(self, parsed=None):
        """Run persist with mocked GraphClient and return captured data."""
        if parsed is None:
            parsed = MOCK_JEC2020_PARSED

        created_nodes: dict[str, list[dict]] = {}
        query_calls: list[tuple] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value

            def capture_query(cypher, **kwargs):
                query_calls.append((cypher, kwargs))
                return []

            mock_gc.query.side_effect = capture_query

            def capture_create(label, items, **kw):
                created_nodes.setdefault(label, []).extend(items)
                return {"processed": len(items), "relationships": {}}

            mock_gc.create_nodes.side_effect = capture_create

            stats = _persist_jec2020_nodes("jet", MOCK_JEC2020_CONFIG, parsed)

        return stats, created_nodes, query_calls

    def test_persist_creates_probe_nodes(self):
        """Magnetics probes create DataNodes with R,Z and dual data sources."""
        stats, nodes, _ = self._run_persist()
        assert stats["probes"] == 2

        probe_dns = [
            dn for dn in nodes.get("SignalNode", []) if "probe:" in dn.get("path", "")
        ]
        assert len(probe_dns) == 2

        probe1 = next(dn for dn in probe_dns if dn["path"].endswith(":1"))
        assert probe1["r"] == 4.292
        assert probe1["z"] == 0.604
        assert probe1["ppf_signal"] == "BPME(1)"
        assert probe1["jpf_signal"] == "DA/C2-CX01"
        assert probe1["system"] == "MP"

    def test_persist_creates_flux_loop_nodes(self):
        """Flux loops create DataNodes with geometry and signal references."""
        stats, nodes, _ = self._run_persist()
        assert stats["flux_loops"] == 1

        loop_dns = [
            dn
            for dn in nodes.get("SignalNode", [])
            if "flux_loop:" in dn.get("path", "")
        ]
        assert len(loop_dns) == 1
        assert loop_dns[0]["r"] == 2.0
        assert loop_dns[0]["system"] == "FL"

    def test_persist_creates_pf_coil_nodes(self):
        """PF coils create DataNodes with geometry (including multi-element)."""
        stats, nodes, _ = self._run_persist()
        assert stats["pf_coils"] == 2

        coil_dns = [
            dn for dn in nodes.get("SignalNode", []) if "pf_coil:" in dn.get("path", "")
        ]
        assert len(coil_dns) == 2

        # Single-element coil
        coil1 = next(dn for dn in coil_dns if dn["path"].endswith(":1"))
        assert coil1["rCentre"] == 0.897
        assert coil1["system"] == "PF"

        # Multi-element coil stores array and first element
        coil3 = next(dn for dn in coil_dns if dn["path"].endswith(":3"))
        assert coil3["rCentre"] == 1.967  # First element as scalar
        assert coil3["rCentre_array"] == [1.967, 2.005, 2.043, 2.081]

    def test_persist_creates_pf_circuits(self):
        """PF circuits create DataNodes and IN_CIRCUIT relationships."""
        stats, nodes, queries = self._run_persist()
        assert stats["pf_circuits"] == 1

        circuit_dns = [
            dn
            for dn in nodes.get("SignalNode", [])
            if "pf_circuit:" in dn.get("path", "")
        ]
        assert len(circuit_dns) == 1

        # Check IN_CIRCUIT relationship query
        circuit_queries = [(q, kw) for q, kw in queries if "IN_CIRCUIT" in q]
        assert len(circuit_queries) == 1

    def test_persist_creates_iron_boundary(self):
        """Iron core boundary creates SignalNode with R,Z contour and permeabilities."""
        stats, nodes, _ = self._run_persist()
        assert stats["iron_segments"] == 3

        iron_dns = [
            dn
            for dn in nodes.get("SignalNode", [])
            if "iron_boundary" in dn.get("path", "")
        ]
        assert len(iron_dns) == 1
        assert iron_dns[0]["r_contour"] == [6.512, 4.952, 3.392]
        assert iron_dns[0]["permeabilities"] == [852.82, 724.86, 887.43]
        assert iron_dns[0]["boundary_length"] == 4.68

    def test_persist_creates_limiter(self):
        """JEC2020 limiter creates SignalNode with R,Z contour."""
        stats, nodes, _ = self._run_persist()
        assert stats["limiter_points"] == 4

        lim_dns = [
            dn
            for dn in nodes.get("SignalNode", [])
            if dn.get("path", "") == "jet:jec2020:limiter"
        ]
        assert len(lim_dns) == 1
        assert lim_dns[0]["r_contour"] == [2.0, 2.5, 3.0, 3.5]
        assert lim_dns[0]["system"] == "LIM"

    def test_persist_creates_same_geometry_link(self):
        """JEC2020 limiter links to device_xml Mk2ILW via SAME_GEOMETRY."""
        _, _, queries = self._run_persist()
        geom_queries = [(q, kw) for q, kw in queries if "SAME_GEOMETRY" in q]
        assert len(geom_queries) == 1
        assert geom_queries[0][1]["dx_path"] == "jet:device_xml:limiter:Mk2ILW"

    def test_persist_creates_signals(self):
        """JEC2020 ingestion creates FacilitySignal nodes."""
        stats, nodes, _ = self._run_persist()
        # 2 probes + 1 flux loop + 2 PF coils + 1 iron + 1 limiter = 7
        assert stats["signals"] == 7

        signals = nodes.get("FacilitySignal", [])
        assert len(signals) == 7

        # Check probe signal has correct data access
        probe_sigs = [s for s in signals if "jec2020_probe_" in s["id"]]
        assert len(probe_sigs) == 2
        assert probe_sigs[0]["data_access"] == "jet:jec2020:xml"
        assert probe_sigs[0]["discovery_source"] == "jec2020_xml"

    def test_persist_creates_data_access(self):
        """JEC2020 ingestion creates DataAccess node."""
        _, nodes, _ = self._run_persist()
        da_nodes = nodes.get("DataAccess", [])
        assert len(da_nodes) == 1
        assert da_nodes[0]["id"] == "jet:jec2020:xml"


class TestParseJEC2020Script:
    """Test the JEC2020 remote parse script in isolation."""

    def _load_module(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "parse_jec2020",
            "imas_codex/remote/scripts/parse_jec2020.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_parse_magnetics(self):
        """Parse magnetics XML with probes and flux loops."""
        mod = self._load_module()
        xml = b"""<?xml version="1.0"?>
<magnetics>
  <magneticProbe id="1" description="Probe 1">
    <geometry rCentre="4.292" zCentre="0.604" angleUnits="degrees"
              poloidalOrientation="-74.1"/>
    <timeTrace signalName="BPME(1)" signalName2="DA/C2-CX01"
               dataSource="JET::PPF" dataSource2="JPF"
               errorType="relativeAbsolute"
               errorRelativeAbsolute="[0.02,0.005]"/>
  </magneticProbe>
  <fluxLoop id="1" description="Loop 1">
    <geometry rCentre="2.0" zCentre="0.0"/>
    <timeTrace signalName="FLME(1)" signalName2="DA/C2-FL01"
               dataSource="JET::PPF" dataSource2="JPF"/>
  </fluxLoop>
</magnetics>"""
        result = mod.parse_magnetics(xml)
        assert len(result["probes"]) == 1
        assert result["probes"][0]["rCentre"] == 4.292
        assert result["probes"][0]["ppf_signal"] == "BPME(1)"
        assert result["probes"][0]["jpf_signal"] == "DA/C2-CX01"
        assert result["probes"][0]["rel_error"] == pytest.approx(0.02)
        assert result["probes"][0]["abs_error"] == pytest.approx(0.005)
        assert len(result["flux_loops"]) == 1
        assert result["flux_loops"][0]["rCentre"] == 2.0

    def test_parse_pf_systems(self):
        """Parse PF coils with single and multi-element geometry."""
        mod = self._load_module()
        xml = b"""<?xml version="1.0"?>
<pfSystems>
  <pfCoil id="1" name="P1/ME">
    <geometry rCentre="0.897" zCentre="0" dR="0.337" dZ="5.427"
              angle1="0" angle2="0" turnCount="710"/>
  </pfCoil>
  <pfCoil id="3" name="P2/SUI/8">
    <geometry rCentre="1.967,2.005" zCentre="3.871,3.871"
              dR="0.035,0.035" dZ="0.035,0.035" turnCount="0.500,0.500"/>
  </pfCoil>
  <pfCircuit id="1" name="P1U_circuit">
    <connections>
      <connection coilId="1"/>
    </connections>
  </pfCircuit>
</pfSystems>"""
        result = mod.parse_pf_systems(xml)
        assert len(result["coils"]) == 2
        assert result["coils"][0]["rCentre"] == 0.897
        # Multi-element coil
        assert result["coils"][1]["rCentre"] == [1.967, 2.005]
        assert len(result["circuits"]) == 1
        assert result["circuits"][0]["coil_ids"] == ["1"]

    def test_parse_iron_boundaries(self):
        """Parse iron core boundary with coordinate arrays."""
        mod = self._load_module()
        xml = b"""<?xml version="1.0"?>
<ironBoundaries>
  <ironBoundary material2Id="1" materialId="3">
    <knotSet basisFunctionCount="3"/>
    <observationPoints initialPermeabilities="852.82, 724.86, 887.43"/>
    <geometry boundaryCoordsR="6.512, 4.952, 3.392"
              boundaryCoordsZ="4.45, 4.45, 4.45"
              segmentLengths="1.56, 1.56, 1.56"
              boundaryLength="4.68"/>
  </ironBoundary>
</ironBoundaries>"""
        result = mod.parse_iron_boundaries(xml)
        assert result["r"] == [6.512, 4.952, 3.392]
        assert result["z"] == [4.45, 4.45, 4.45]
        assert result["permeabilities"] == [852.82, 724.86, 887.43]
        assert result["boundary_length"] == 4.68

    def test_parse_limiter(self):
        """Parse limiter XML with comma-separated R,Z arrays."""
        mod = self._load_module()
        xml = b"""<?xml version="1.0"?>
<limiter rValues="2.0, 2.5, 3.0" zValues="1.0, 0.5, -0.5"/>"""
        result = mod.parse_limiter(xml)
        assert result["r"] == [2.0, 2.5, 3.0]
        assert result["z"] == [1.0, 0.5, -0.5]

    def test_parse_limiter_whitespace_separated(self):
        """Parse limiter XML with whitespace-separated R,Z arrays."""
        mod = self._load_module()
        xml = b"""<?xml version="1.0"?>
<limiter name="JET first wall" rValues="2.0  2.5  3.0" zValues="1.0  0.5  -0.5"/>"""
        result = mod.parse_limiter(xml)
        assert result["r"] == [2.0, 2.5, 3.0]
        assert result["z"] == [1.0, 0.5, -0.5]


# ────────────────── MCFG sensor calibration fixtures ──────────────────

MOCK_MCFG_CONFIG = {
    "name": "sensor_calibration",
    "description": "MCFG sensor positions and calibration epochs",
    "format": "text",
    "base_dir": "/home/MAGNW/chain1/input",
    "files": [
        {
            "path": "PPFcfg/sensors_200c_2019-03-11.txt",
            "role": "sensors",
        },
        {
            "path": "magn_ep_2019-05-14/MCFG.ix",
            "role": "calibration_index",
        },
    ],
}

MOCK_MCFG_PARSED = {
    "sensors": {
        "coils": [
            {
                "id": 1,
                "r": 4.292,
                "z": 0.604,
                "angle": -74.1,
                "gain": 0.986,
                "rel_error": 0.02,
                "abs_error": 0.005,
                "jpf_name": "DA/C2-CX01",
                "description": "outboard midplane",
            },
            {
                "id": 2,
                "r": 4.281,
                "z": 0.724,
                "angle": -73.5,
                "gain": 0.992,
                "rel_error": 0.02,
                "abs_error": 0.005,
                "jpf_name": "DA/C2-CX02",
                "description": "outboard upper",
            },
        ],
        "hall_probes": [
            {
                "id": 1,
                "r": 6.2,
                "z": 0.0,
                "offset": 0.0,
                "gain": 1.0,
                "rel_error": 0.01,
                "abs_error": 0.001,
                "jpf_name": "DA/C2-HX01",
                "description": "ex-vessel hall 1",
            },
        ],
        "other": [],
    },
    "calibration_index": {
        "epochs": [
            {
                "date": "19960127",
                "first_shot": 35680,
                "user": "gchuyler",
                "config_id": "MCFG:0001/MAGNW",
                "config_type": "MAGNETICS",
                "description": "Initial calibration",
            },
            {
                "date": "20190314",
                "first_shot": 95350,
                "user": "operato",
                "config_id": "MCFG:0077/MAGNW",
                "config_type": "MAGNETICS",
                "description": "Latest calibration 2019",
            },
        ],
    },
}


class TestMCFGPersist:
    """Test _persist_mcfg_nodes with mock graph."""

    def _run_persist(self, parsed=None):
        """Run persist with mocked GraphClient and return captured data."""
        if parsed is None:
            parsed = MOCK_MCFG_PARSED

        created_nodes: dict[str, list[dict]] = {}
        query_calls: list[tuple] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value

            def capture_query(cypher, **kwargs):
                query_calls.append((cypher, kwargs))
                return []

            mock_gc.query.side_effect = capture_query

            def capture_create(label, items, **kw):
                created_nodes.setdefault(label, []).extend(items)
                return {"processed": len(items), "relationships": {}}

            mock_gc.create_nodes.side_effect = capture_create

            stats = _persist_mcfg_nodes("jet", MOCK_MCFG_CONFIG, parsed)

        return stats, created_nodes, query_calls

    def test_persist_returns_correct_stats(self):
        """Stats reflect coil, hall probe, and calibration epoch counts."""
        stats, _, _ = self._run_persist()
        assert stats["coil_sensors"] == 2
        assert stats["hall_probes"] == 1
        assert stats["calibration_epochs"] == 2

    def test_persist_creates_data_source(self):
        """Creates DataSource node with MCFG metadata."""
        _, _, queries = self._run_persist()
        ds_queries = [(q, kw) for q, kw in queries if "MERGE (ds:DataSource" in q]
        assert len(ds_queries) == 1
        assert ds_queries[0][1]["name"] == "sensor_calibration"
        assert ds_queries[0][1]["source_format"] == "text"

    def test_persist_creates_coil_sensor_nodes(self):
        """Coil sensors create SignalNode with R/Z/angle/gain/error fields."""
        _, nodes, _ = self._run_persist()
        coil_dns = [
            dn
            for dn in nodes.get("SignalNode", [])
            if "mcfg:sensor:" in dn.get("path", "")
        ]
        assert len(coil_dns) == 2

        sensor1 = next(
            dn for dn in coil_dns if dn["path"] == "jet:mcfg:sensor:DA/C2-CX01"
        )
        assert sensor1["r"] == 4.292
        assert sensor1["z"] == 0.604
        assert sensor1["angle"] == -74.1
        assert sensor1["gain"] == 0.986
        assert sensor1["rel_error"] == 0.02
        assert sensor1["abs_error"] == 0.005
        assert sensor1["jpf_name"] == "DA/C2-CX01"
        assert sensor1["system"] == "MP"
        assert sensor1["facility_id"] == "jet"

    def test_persist_creates_hall_probe_nodes(self):
        """Hall probes create SignalNode with reduced field set."""
        _, nodes, _ = self._run_persist()
        hall_dns = [
            dn
            for dn in nodes.get("SignalNode", [])
            if "mcfg:hall:" in dn.get("path", "")
        ]
        assert len(hall_dns) == 1
        assert hall_dns[0]["r"] == 6.2
        assert hall_dns[0]["z"] == 0.0
        assert hall_dns[0]["jpf_name"] == "DA/C2-HX01"
        assert hall_dns[0]["system"] == "MP"

    def test_persist_creates_matches_sensor_crossref(self):
        """MATCHES_SENSOR relationships link MCFG → JEC2020 by R,Z proximity."""
        _, _, queries = self._run_persist()
        xref_queries = [(q, kw) for q, kw in queries if "MATCHES_SENSOR" in q]
        assert len(xref_queries) == 1
        # Verify it matches on R,Z tolerance < 0.001m
        assert "abs(mcfg.r - jec.r) < 0.001" in xref_queries[0][0]
        assert "abs(mcfg.z - jec.z) < 0.001" in xref_queries[0][0]
        assert xref_queries[0][1]["mcfg_source"] == "sensor_calibration"
        assert xref_queries[0][1]["jec_source"] == "jec2020_geometry"

    def test_persist_stores_calibration_epochs(self):
        """Calibration epochs update DataSource with epoch count and shot range."""
        _, _, queries = self._run_persist()
        cal_queries = [(q, kw) for q, kw in queries if "calibration_epoch_count" in q]
        assert len(cal_queries) == 1
        assert cal_queries[0][1]["count"] == 2
        assert cal_queries[0][1]["first_shot"] == 35680
        assert cal_queries[0][1]["last_shot"] == 95350

    def test_persist_creates_calibration_epoch_nodes(self):
        """Individual CalibrationEpoch graph nodes are created for each epoch."""
        _, _, queries = self._run_persist()
        ce_queries = [(q, kw) for q, kw in queries if "CalibrationEpoch" in q]
        assert len(ce_queries) == 1
        records = ce_queries[0][1]["records"]
        assert len(records) == 2

        # First epoch
        epoch1 = records[0]
        assert epoch1["id"] == "jet:sensor_calibration:MCFG:0001/MAGNW"
        assert epoch1["facility_id"] == "jet"
        assert epoch1["data_source_name"] == "jet:sensor_calibration"
        assert epoch1["date"] == "19960127"
        assert epoch1["first_shot"] == 35680
        assert epoch1["config_id"] == "MCFG:0001/MAGNW"
        assert epoch1["config_type"] == "MAGNETICS"
        assert epoch1["user"] == "gchuyler"
        assert epoch1["description"] == "Initial calibration"

        # Second epoch
        epoch2 = records[1]
        assert epoch2["id"] == "jet:sensor_calibration:MCFG:0077/MAGNW"
        assert epoch2["first_shot"] == 95350
        assert epoch2["description"] == "Latest calibration 2019"

    def test_persist_calibration_epoch_relationships(self):
        """CalibrationEpoch nodes have AT_FACILITY and IN_DATA_SOURCE relationships."""
        _, _, queries = self._run_persist()
        ce_queries = [(q, kw) for q, kw in queries if "CalibrationEpoch" in q]
        assert len(ce_queries) == 1
        cypher = ce_queries[0][0]
        assert "AT_FACILITY" in cypher
        assert "IN_DATA_SOURCE" in cypher

    def test_persist_no_sensors_returns_zeros(self):
        """Empty sensor data returns zero counts without errors."""
        parsed = {
            "sensors": {"coils": [], "hall_probes": []},
            "calibration_index": {"epochs": []},
        }
        stats, nodes, _ = self._run_persist(parsed)
        assert stats["coil_sensors"] == 0
        assert stats["hall_probes"] == 0
        assert stats["calibration_epochs"] == 0
        assert not nodes.get("SignalNode")


class TestParseMCFGScript:
    """Test the MCFG remote parse script in isolation."""

    def _load_module(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "parse_mcfg_sensors",
            "imas_codex/remote/scripts/parse_mcfg_sensors.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_parse_sensors_coils(self):
        """Parse coil section with standard format fields."""
        mod = self._load_module()
        text = """\
# MCFG sensor file
  1   4.292   0.604  -74.1  0.986  0.02  0.005  DA/C2-CX01  -74.1  outboard midplane
  2   4.281   0.724  -73.5  0.992  0.02  0.005  DA/C2-CX02  -73.5  outboard upper
#END
#END
"""
        result = mod.parse_sensors(text)
        assert len(result["coils"]) == 2
        assert result["coils"][0]["r"] == 4.292
        assert result["coils"][0]["z"] == 0.604
        assert result["coils"][0]["angle"] == -74.1
        assert result["coils"][0]["jpf_name"] == "DA/C2-CX01"
        assert result["coils"][0]["gain"] == 0.986
        assert result["coils"][0]["rel_error"] == 0.02
        assert result["coils"][0]["abs_error"] == 0.005

    def test_parse_sensors_hall_probes(self):
        """Parse hall probe section after first #END marker."""
        mod = self._load_module()
        text = """\
# coils
  1   4.292   0.604  -74.1  0.986  0.02  0.005  DA/C2-CX01  -74.1
#END
  1   6.200   0.000   0.000  1.000  0.01  0.001  DA/C2-HX01  ex-vessel hall 1
#END
"""
        result = mod.parse_sensors(text)
        assert len(result["coils"]) == 1
        assert len(result["hall_probes"]) == 1
        assert result["hall_probes"][0]["r"] == 6.2
        assert result["hall_probes"][0]["jpf_name"] == "DA/C2-HX01"

    def test_parse_calibration_index(self):
        """Parse MCFG.ix calibration epoch index."""
        mod = self._load_module()
        text = """\
$:19960127 0035680 gchuyler MCFG:0001/MAGNW MAGNETICS ! Initial calibration
$:20190314 0095350 operato  MCFG:0077/MAGNW MAGNETICS ! Latest calibration 2019
"""
        result = mod.parse_calibration_index(text)
        assert len(result["epochs"]) == 2
        assert result["epochs"][0]["date"] == "19960127"
        assert result["epochs"][0]["first_shot"] == 35680
        assert result["epochs"][0]["user"] == "gchuyler"
        assert result["epochs"][0]["config_id"] == "MCFG:0001/MAGNW"
        assert result["epochs"][1]["first_shot"] == 95350
        assert result["epochs"][1]["description"] == "Latest calibration 2019"

    def test_parse_sensors_comments_skipped(self):
        """Comment lines starting with ! or * are skipped."""
        mod = self._load_module()
        text = """\
! Header comment
* Another comment
  1   4.292   0.604  -74.1  0.986  0.02  0.005  DA/C2-CX01  -74.1
#END
#END
"""
        result = mod.parse_sensors(text)
        assert len(result["coils"]) == 1
        assert len(result["hall_probes"]) == 0
        assert len(result["other"]) == 0


# Mock data for sensor version tests
MOCK_MCFG_CONFIG_WITH_VERSIONS = {
    **MOCK_MCFG_CONFIG,
    "sensor_versions": [
        {
            "file": "sensors-200c-12-05-04.txt",
            "path": "/home/chain1/input/magn90/",
            "date": "2005-10-12",
        },
        {
            "file": "sensors_200c_2019-03-11.txt",
            "path": "/home/MAGNW/chain1/input/PPFcfg/",
            "date": "2019-03-11",
        },
    ],
}

MOCK_MCFG_PARSED_WITH_VERSIONS = {
    **MOCK_MCFG_PARSED,
    "sensor_versions": [
        {
            "date": "2005-10-12",
            "file": "sensors-200c-12-05-04.txt",
            "error": None,
            "sensors": {
                "coils": [
                    {
                        "id": 1,
                        "r": 4.290,
                        "z": 0.602,
                        "angle": -74.0,
                        "gain": 0.980,
                        "rel_error": 0.03,
                        "abs_error": 0.006,
                        "jpf_name": "DA/C2-CX01",
                    },
                ],
                "hall_probes": [],
                "other": [],
            },
        },
        {
            "date": "2019-03-11",
            "file": "sensors_200c_2019-03-11.txt",
            "error": None,
            "sensors": {
                "coils": [
                    {
                        "id": 1,
                        "r": 4.292,
                        "z": 0.604,
                        "angle": -74.1,
                        "gain": 0.986,
                        "rel_error": 0.02,
                        "abs_error": 0.005,
                        "jpf_name": "DA/C2-CX01",
                    },
                ],
                "hall_probes": [
                    {
                        "id": 1,
                        "r": 6.2,
                        "z": 0.0,
                        "gain": 1.0,
                        "rel_error": 0.01,
                        "abs_error": 0.001,
                        "jpf_name": "DA/C2-HX01",
                    },
                ],
                "other": [],
            },
        },
    ],
}


class TestMCFGSensorVersions:
    """Test versioned sensor file persistence and SUPERSEDES chains."""

    def _run_persist(self, parsed=None, config=None):
        """Run persist with mocked GraphClient for version tests."""
        if parsed is None:
            parsed = MOCK_MCFG_PARSED_WITH_VERSIONS
        if config is None:
            config = MOCK_MCFG_CONFIG_WITH_VERSIONS

        created_nodes: dict[str, list[dict]] = {}
        query_calls: list[tuple] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value

            def capture_query(cypher, **kwargs):
                query_calls.append((cypher, kwargs))
                return []

            mock_gc.query.side_effect = capture_query

            def capture_create(label, items, **kw):
                created_nodes.setdefault(label, []).extend(items)
                return {"processed": len(items), "relationships": {}}

            mock_gc.create_nodes.side_effect = capture_create

            stats = _persist_mcfg_nodes("jet", config, parsed)

        return stats, created_nodes, query_calls

    def test_sensor_version_count_in_stats(self):
        """Stats include sensor_versions count for successfully parsed versions."""
        stats, _, _ = self._run_persist()
        assert stats["sensor_versions"] == 2

    def test_versioned_sensor_nodes_created(self):
        """Versioned sensors create SignalNode with version_date tags."""
        _, nodes, _ = self._run_persist()
        version_nodes = [
            dn for dn in nodes.get("SignalNode", []) if "version_date" in dn
        ]
        # 2005 version has 1 coil, 2019 has 1 coil + 1 hall probe = 3 total
        assert len(version_nodes) == 3

        v2005 = [n for n in version_nodes if n["version_date"] == "2005-10-12"]
        assert len(v2005) == 1
        assert v2005[0]["r"] == 4.290
        assert v2005[0]["path"] == "jet:mcfg:sensor:DA/C2-CX01:v2005-10-12"

        v2019 = [n for n in version_nodes if n["version_date"] == "2019-03-11"]
        assert len(v2019) == 2

    def test_versioned_data_sources_created(self):
        """Each version creates its own DataSource node."""
        _, _, queries = self._run_persist()
        ds_queries = [
            (q, kw)
            for q, kw in queries
            if "MERGE (ds:DataSource" in q and "ds_id" in kw
        ]
        assert len(ds_queries) == 2
        ds_ids = [q[1]["ds_id"] for q in ds_queries]
        assert "jet:sensor_calibration:v2005-10-12" in ds_ids
        assert "jet:sensor_calibration:v2019-03-11" in ds_ids

    def test_supersedes_chain_created(self):
        """SUPERSEDES relationship links newer → older versions."""
        _, _, queries = self._run_persist()
        sup_queries = [(q, kw) for q, kw in queries if "SUPERSEDES" in q]
        assert len(sup_queries) == 1
        records = sup_queries[0][1]["records"]
        assert len(records) == 1
        assert records[0]["newer_id"] == "jet:sensor_calibration:v2019-03-11"
        assert records[0]["older_id"] == "jet:sensor_calibration:v2005-10-12"

    def test_no_versions_skips_version_persistence(self):
        """Without sensor_versions in parsed data, no version nodes are created."""
        stats, nodes, queries = self._run_persist(
            parsed=MOCK_MCFG_PARSED, config=MOCK_MCFG_CONFIG
        )
        assert stats.get("sensor_versions", 0) == 0
        version_nodes = [
            dn for dn in nodes.get("SignalNode", []) if "version_date" in dn
        ]
        assert len(version_nodes) == 0
        sup_queries = [(q, kw) for q, kw in queries if "SUPERSEDES" in q]
        assert len(sup_queries) == 0

    def test_failed_version_skipped(self):
        """Versions with errors are not persisted."""
        parsed = {
            **MOCK_MCFG_PARSED,
            "sensor_versions": [
                {
                    "date": "2005-10-12",
                    "file": "sensors-200c-12-05-04.txt",
                    "error": "File not found",
                    "sensors": None,
                },
                {
                    "date": "2019-03-11",
                    "file": "sensors_200c_2019-03-11.txt",
                    "error": None,
                    "sensors": {"coils": [], "hall_probes": [], "other": []},
                },
            ],
        }
        stats, _, queries = self._run_persist(parsed=parsed)
        # Only 1 version succeeded (the empty one)
        assert stats["sensor_versions"] == 1
        # No SUPERSEDES — need at least 2 valid versions
        sup_queries = [(q, kw) for q, kw in queries if "SUPERSEDES" in q]
        assert len(sup_queries) == 0


class TestMCFGHandlerInput:
    """Test MCFGHandler.build_script_input with sensor_versions."""

    def test_build_script_input_forwards_sensor_versions(self):
        """sensor_versions from config are forwarded to script input."""
        from imas_codex.discovery.signals.scanners.device_xml import MCFGHandler

        handler = MCFGHandler()
        result = handler.build_script_input(MOCK_MCFG_CONFIG_WITH_VERSIONS)
        assert "sensor_versions" in result
        assert len(result["sensor_versions"]) == 2
        assert result["sensor_versions"][0]["file"] == "sensors-200c-12-05-04.txt"
        assert result["sensor_versions"][0]["date"] == "2005-10-12"

    def test_build_script_input_no_versions(self):
        """Without sensor_versions, key is absent from script input."""
        from imas_codex.discovery.signals.scanners.device_xml import MCFGHandler

        handler = MCFGHandler()
        result = handler.build_script_input(MOCK_MCFG_CONFIG)
        assert "sensor_versions" not in result


# ────────────────── PPF static geometry signal fixtures ──────────────────

MOCK_PPF_CONFIG = {
    "sal_endpoint": "https://sal.jet.uk",
    "reference_pulse": 99896,
    "default_owner": "jetppf",
    "setup_commands": ["source /etc/profile.d/modules.sh", "module load jet/1.0"],
    "static_signals": [
        {
            "name": "EFIT/RLIM",
            "description": "Limiter R coordinates (251 points, static across ILW)",
            "static": True,
            "expected_shape": [1, 251],
        },
        {
            "name": "EFIT/ZLIM",
            "description": "Limiter Z coordinates (251 points, static across ILW)",
            "static": True,
            "expected_shape": [1, 251],
        },
        {
            "name": "EFIT/RBND",
            "description": "Plasma boundary R (time-resolved, 947×105)",
            "static": False,  # Not static — should be skipped
            "expected_shape": [947, 105],
        },
        {
            "name": "LIDR/Z",
            "description": "LIDAR Thomson scattering Z positions",
            "static": True,
            "expected_shape": [50],
        },
        {
            "name": "VESL/CROS",
            "description": "Vessel cross-section contour",
            "static": True,
        },
    ],
}


class TestPPFStaticPersist:
    """Test _persist_ppf_static_nodes with mock graph."""

    def _run_persist(self, ppf_config=None):
        """Run persist with mocked GraphClient and return captured data."""
        if ppf_config is None:
            ppf_config = MOCK_PPF_CONFIG

        created_nodes: dict[str, list[dict]] = {}
        query_calls: list[tuple] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value

            def capture_query(cypher, **kwargs):
                query_calls.append((cypher, kwargs))
                return []

            mock_gc.query.side_effect = capture_query

            def capture_create(label, items, **kw):
                created_nodes.setdefault(label, []).extend(items)
                return {"processed": len(items), "relationships": {}}

            mock_gc.create_nodes.side_effect = capture_create

            stats = _persist_ppf_static_nodes("jet", ppf_config)

        return stats, created_nodes, query_calls

    def test_persist_returns_correct_stats(self):
        """Stats reflect static-only DataAccess count and cross-references."""
        stats, _, _ = self._run_persist()
        # 4 static signals (EFIT/RLIM, EFIT/ZLIM, LIDR/Z, VESL/CROS)
        # EFIT/RBND is not static so excluded
        assert stats["data_access_nodes"] == 4
        # EFIT/RLIM and EFIT/ZLIM have cross-refs in PPF_GEOMETRY_CROSSREFS
        assert stats["cross_references"] == 2

    def test_persist_creates_data_source(self):
        """Creates ppf_static DataSource node."""
        _, _, queries = self._run_persist()
        ds_queries = [(q, kw) for q, kw in queries if "MERGE (ds:DataSource" in q]
        assert len(ds_queries) == 1
        assert ds_queries[0][1]["name"] == "ppf_static"

    def test_persist_creates_data_access_for_static_only(self):
        """Only static PPF signals get DataAccess nodes."""
        _, nodes, _ = self._run_persist()
        da_nodes = nodes.get("DataAccess", [])
        assert len(da_nodes) == 4

        da_ids = {da["id"] for da in da_nodes}
        assert "jet:ppf:EFIT/RLIM" in da_ids
        assert "jet:ppf:EFIT/ZLIM" in da_ids
        assert "jet:ppf:LIDR/Z" in da_ids
        assert "jet:ppf:VESL/CROS" in da_ids
        # RBND is time-resolved, should NOT be present
        assert "jet:ppf:EFIT/RBND" not in da_ids

    def test_data_access_has_ppf_method(self):
        """DataAccess nodes have method_type=ppf and MDSplus library."""
        _, nodes, _ = self._run_persist()
        rlim = next(da for da in nodes["DataAccess"] if da["id"] == "jet:ppf:EFIT/RLIM")
        assert rlim["method_type"] == "ppf"
        assert rlim["library"] == "MDSplus"
        assert rlim["access_type"] == "remote"
        assert rlim["data_source"] == "ppf_static"

    def test_data_access_has_full_example(self):
        """DataAccess full_example is valid Python."""
        _, nodes, _ = self._run_persist()
        rlim = next(da for da in nodes["DataAccess"] if da["id"] == "jet:ppf:EFIT/RLIM")
        compile(rlim["full_example"], "<ppf_rlim>", "exec")
        assert "ppf" in rlim["full_example"]
        assert "99896" in rlim["full_example"]

    def test_persist_creates_accesses_geometry_crossrefs(self):
        """ACCESSES_GEOMETRY links PPF DataAccess to device_xml DataNodes."""
        _, _, queries = self._run_persist()
        geom_queries = [(q, kw) for q, kw in queries if "ACCESSES_GEOMETRY" in q]
        assert len(geom_queries) == 2

        # Both EFIT/RLIM and EFIT/ZLIM link to Mk2ILW limiter
        dx_paths = {kw["dx_path"] for _, kw in geom_queries}
        assert dx_paths == {"jet:device_xml:limiter:Mk2ILW"}

        da_ids = {kw["da_id"] for _, kw in geom_queries}
        assert "jet:ppf:EFIT/RLIM" in da_ids
        assert "jet:ppf:EFIT/ZLIM" in da_ids

    def test_persist_no_static_returns_zeros(self):
        """No static signals returns zero counts."""
        ppf_config = {
            "static_signals": [
                {"name": "EFIT/RBND", "static": False},
            ]
        }
        stats, _, _ = self._run_persist(ppf_config)
        assert stats["data_access_nodes"] == 0
        assert stats["cross_references"] == 0


class TestPPFGeometryCrossrefs:
    """Test PPF_GEOMETRY_CROSSREFS mapping."""

    def test_rlim_zlim_map_to_mk2ilw(self):
        """EFIT/RLIM and EFIT/ZLIM both reference Mk2ILW limiter."""
        assert PPF_GEOMETRY_CROSSREFS["EFIT/RLIM"] == "jet:device_xml:limiter:Mk2ILW"
        assert PPF_GEOMETRY_CROSSREFS["EFIT/ZLIM"] == "jet:device_xml:limiter:Mk2ILW"

    def test_no_crossref_for_unique_signals(self):
        """VESL/CROS has no device_xml equivalent (unique PPF data)."""
        assert "VESL/CROS" not in PPF_GEOMETRY_CROSSREFS


class TestProbeStatusPersistence:
    """Test that EFITSNAP probe enable/disable data is persisted to epochs."""

    def test_epoch_has_probes_enabled_disabled(self):
        """Epoch records include probes_enabled/probes_disabled from snap data."""
        query_calls: list[tuple] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.create_nodes.return_value = {"processed": 0, "relationships": {}}

            def capture_query(cypher, **kwargs):
                query_calls.append((cypher, kwargs))
                return []

            mock_gc.query.side_effect = capture_query

            _persist_graph_nodes(
                "jet",
                MOCK_JET_CONFIG,
                MOCK_PARSED_OUTPUT["versions"],
                MOCK_PARSED_OUTPUT["limiters"],
            )

        # Find epoch creation query
        epoch_calls = [
            (q, kw) for q, kw in query_calls if "SignalEpoch" in q and "UNWIND" in q
        ]
        assert len(epoch_calls) >= 1
        records = epoch_calls[0][1]["records"]
        epoch = records[0]
        assert epoch["probes_enabled"] == ["BPME(1)", "BPME(2)"]
        # Empty disabled list is omitted from record
        assert "probes_disabled" not in epoch

    def test_epoch_without_snap_has_no_probe_data(self):
        """Epochs with no snap file data omit probe fields from record."""
        config = {
            "git_repo": "/repo",
            "input_prefix": "JET/input",
            "versions": [
                {
                    "version": "limiter_era",
                    "first_shot": 1,
                    "last_shot": 28791,
                    "description": "Pre-divertor limiter",
                    "uses_limiter": "Limiter",
                    "wall_configuration": "limiter",
                },
            ],
            "limiter_versions": [],
        }
        query_calls: list[tuple] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.create_nodes.return_value = {"processed": 0, "relationships": {}}

            def capture_query(cypher, **kwargs):
                query_calls.append((cypher, kwargs))
                return []

            mock_gc.query.side_effect = capture_query

            _persist_graph_nodes("jet", config, {}, {})

        epoch_calls = [
            (q, kw) for q, kw in query_calls if "SignalEpoch" in q and "UNWIND" in q
        ]
        records = epoch_calls[0][1]["records"]
        epoch = records[0]
        # Pre-divertor epoch has no snap data → no probe fields
        assert "probes_enabled" not in epoch
        assert "probes_disabled" not in epoch


class TestPFConfigurationPersistence:
    """Test that pf_configuration is persisted from config to epochs."""

    def test_epoch_has_pf_configuration(self):
        """Epoch record includes pf_configuration when configured."""
        config = {
            "git_repo": "/repo",
            "input_prefix": "JET/input",
            "versions": [
                {
                    "version": "p68613",
                    "first_shot": 68613,
                    "last_shot": 74386,
                    "description": "Baseline DMSS=91",
                    "device_xml": "Devices/device_p68613.xml",
                    "pf_configuration": "DMSS=091",
                    "wall_configuration": "divertor",
                },
            ],
            "limiter_versions": [],
        }
        parsed = {
            "p68613": {
                "magprobes": [{"id": "1", "r": 4.0, "z": 0.5, "angle": -70.0}],
                "flux": [],
                "pfcoils": [],
                "pfcircuits": [],
                "pfpassive": [],
            }
        }
        query_calls: list[tuple] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.create_nodes.return_value = {"processed": 0, "relationships": {}}
            mock_gc.query.side_effect = lambda q, **kw: (
                query_calls.append((q, kw)) or []
            )

            _persist_graph_nodes("jet", config, parsed, {})

        epoch_calls = [
            (q, kw) for q, kw in query_calls if "SignalEpoch" in q and "UNWIND" in q
        ]
        records = epoch_calls[0][1]["records"]
        assert records[0]["pf_configuration"] == "DMSS=091"

    def test_pre_divertor_epoch_has_no_pf_configuration(self):
        """Pre-divertor epochs have no pf_configuration."""
        config = {
            "git_repo": "/repo",
            "input_prefix": "JET/input",
            "versions": [
                {
                    "version": "limiter_era",
                    "first_shot": 1,
                    "last_shot": 28791,
                    "description": "Pre-divertor",
                    "uses_limiter": "Limiter",
                    "wall_configuration": "limiter",
                },
            ],
            "limiter_versions": [],
        }
        query_calls: list[tuple] = []
        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.create_nodes.return_value = {"processed": 0, "relationships": {}}
            mock_gc.query.side_effect = lambda q, **kw: (
                query_calls.append((q, kw)) or []
            )

            _persist_graph_nodes("jet", config, {}, {})

        epoch_calls = [
            (q, kw) for q, kw in query_calls if "SignalEpoch" in q and "UNWIND" in q
        ]
        records = epoch_calls[0][1]["records"]
        assert "pf_configuration" not in records[0]


class TestDeviceXMLProvenance:
    """Test file_source/file_path provenance on SignalNode dicts."""

    def test_device_xml_data_node_has_provenance(self):
        """DataNodes from device XML include file_source=git and file_path."""
        created_nodes: list[list[dict]] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.query.return_value = []
            mock_gc.create_nodes.side_effect = lambda label, items, **kw: (
                created_nodes.append(items)
                or {"processed": len(items), "relationships": {}}
            )

            _persist_graph_nodes(
                "jet",
                MOCK_JET_CONFIG,
                MOCK_PARSED_OUTPUT["versions"],
                MOCK_PARSED_OUTPUT["limiters"],
            )

        # Find SignalNode calls (exclude signals and access)
        data_node_batches = [
            batch
            for batch in created_nodes
            if batch
            and isinstance(batch[0], dict)
            and "path" in batch[0]
            and "node_type" in batch[0]
        ]
        all_dns = [dn for batch in data_node_batches for dn in batch]
        magprobe_dns = [dn for dn in all_dns if "magprobes:1" in dn["path"]]
        assert len(magprobe_dns) == 1
        dn = magprobe_dns[0]
        assert dn["file_source"] == "git"
        assert dn["file_path"] == "Devices/device_p89440.xml"

    def test_epoch_has_device_xml_path(self):
        """Epoch records include device_xml_path for provenance."""
        query_calls: list[tuple] = []

        with patch(
            "imas_codex.discovery.signals.scanners.device_xml.GraphClient"
        ) as mock_gc_cls:
            mock_gc = mock_gc_cls.return_value.__enter__.return_value
            mock_gc.create_nodes.return_value = {"processed": 0, "relationships": {}}
            mock_gc.query.side_effect = lambda q, **kw: (
                query_calls.append((q, kw)) or []
            )

            _persist_graph_nodes(
                "jet",
                MOCK_JET_CONFIG,
                MOCK_PARSED_OUTPUT["versions"],
                MOCK_PARSED_OUTPUT["limiters"],
            )

        epoch_calls = [
            (q, kw) for q, kw in query_calls if "SignalEpoch" in q and "UNWIND" in q
        ]
        records = epoch_calls[0][1]["records"]
        epoch = records[0]
        assert epoch["device_xml_path"] == "Devices/device_p89440.xml"
        assert epoch["snap_file_path"] == "Snap_files/EFITSNAP/efitsnap_p89440_bound0"

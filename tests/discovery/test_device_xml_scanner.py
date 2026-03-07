"""E2E tests for DeviceXMLScanner — JET machine description geometry.

Tests the full scan → persist → query cycle using mock SSH data.
Validates DataSource, StructuralEpoch, DataNode, and FacilitySignal creation.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from imas_codex.discovery.signals.scanners.base import get_scanner
from imas_codex.discovery.signals.scanners.device_xml import (
    SECTION_METADATA,
    DeviceXMLScanner,
    _build_data_access,
    _make_signal_id,
    _make_signal_name,
    _persist_graph_nodes,
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
        # 1 pfcoil × 5 fields = 5
        # 1 pfcircuit × 2 fields = 2
        # 1 pfpassive × 7 fields = 7
        # 1 limiter = 1
        # Total = 24
        assert stats["signals"] == 24

    def test_data_node_has_geometry_values(self):
        """DataNode dicts include r, z, angle etc. as properties."""
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

        # Find DataNode calls (exclude FacilitySignal and DataAccess)
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

    def test_data_node_has_system_property(self):
        """DataNode includes system property for domain filtering."""
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
        """Limiter DataNode includes R,Z contour arrays."""
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
        """check() returns valid for signals with DataNode data."""
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
        assert stats["data_nodes"] == 2  # One DataNode per epoch per instance


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
    """Test limiter DataNode provenance tracking (file_source, file_path)."""

    def test_limiter_node_has_provenance(self):
        """Limiter DataNode includes file_source and file_path properties."""
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
    """Test StructuralEpoch → Limiter DataNode USES_LIMITER relationships."""

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

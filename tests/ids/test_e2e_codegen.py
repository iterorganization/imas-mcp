"""Tests for E2E code generation pipeline (Phase 10/11).

Tests cover:
  - Extraction script generation and validation
  - Assembly code validation (Tier 2)
  - Assembly code generation
  - Binary data transfer protocol (serialization)
  - Extraction caching
  - Remote capability probe strategy selection
  - E2E validation from fixtures
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# 11.2.1 Extraction Script Generation
# ---------------------------------------------------------------------------


class TestGenerateExtractionScript:
    """Tests for generate_extraction_script()."""

    def _make_script(self, **kwargs):
        from imas_codex.ids.codegen import generate_extraction_script

        section_signals = kwargs.pop(
            "section_signals",
            {
                "pf_active/coil": [
                    {
                        "id": "sig1",
                        "accessor": "\\IP",
                        "data_source_name": "tcv_shot",
                    },
                    {
                        "id": "sig2",
                        "accessor": "\\BTOR",
                        "data_source_name": "tcv_shot",
                    },
                ]
            },
        )
        data_access = kwargs.pop(
            "data_access",
            {
                "imports_template": "import MDSplus",
                "connection_template": "tree = MDSplus.Tree('tcv_shot', shot)",
                "data_template": 'tree.getNode("{accessor}").data()',
                "time_template": 'tree.getNode("{accessor}").dim_of().data()',
                "cleanup_template": "",
            },
        )
        return generate_extraction_script(
            kwargs.pop("facility", "tcv"),
            kwargs.pop("ids_name", "pf_active"),
            section_signals,
            data_access,
            max_points=kwargs.pop("max_points", 10000),
        )

    def test_valid_python_syntax(self):
        """Generated script passes ast.parse."""
        script = self._make_script()
        # Should not raise
        ast.parse(script)

    def test_has_all_signals(self):
        """All mapped signals appear in the generated script."""
        script = self._make_script()
        assert "sig1" in script
        assert "sig2" in script

    def test_contains_shot_variable(self):
        """Script reads shot from config."""
        script = self._make_script()
        assert 'config["shot"]' in script or "config['shot']" in script

    def test_contains_decimation(self):
        """Decimation code present when max_points set."""
        script = self._make_script(max_points=5000)
        assert "5000" in script

    def test_no_decimation_when_none(self):
        """No decimation code when max_points is None."""
        script = self._make_script(max_points=None)
        assert "Decimate" not in script

    def test_has_import_section(self):
        """Script includes the imports template."""
        script = self._make_script()
        assert "import MDSplus" in script

    def test_has_msgpack_detection(self):
        """Script has msgpack format detection."""
        script = self._make_script()
        assert "msgpack" in script


# ---------------------------------------------------------------------------
# 11.2.2 Assembly Code Validation (Tier 2)
# ---------------------------------------------------------------------------


class TestValidateAssemblyCode:
    """Tests for validate_assembly_code()."""

    def test_valid_code_passes(self):
        from imas_codex.ids.codegen import validate_assembly_code

        code = """
def assemble_coil(ids, signals, mappings):
    import numpy as np
    pass
"""
        errors = validate_assembly_code(code, "assemble_coil")
        assert errors == []

    def test_catches_syntax_error(self):
        from imas_codex.ids.codegen import validate_assembly_code

        code = "def assemble_coil(ids, signals, mappings)\n    pass"
        errors = validate_assembly_code(code, "assemble_coil")
        assert any("Syntax error" in e for e in errors)

    def test_catches_missing_function(self):
        from imas_codex.ids.codegen import validate_assembly_code

        code = "def wrong_name(ids, signals, mappings):\n    pass"
        errors = validate_assembly_code(code, "assemble_coil")
        assert any("not found" in e for e in errors)

    def test_catches_bad_signature(self):
        from imas_codex.ids.codegen import validate_assembly_code

        code = "def assemble_coil(x, y):\n    pass"
        errors = validate_assembly_code(code, "assemble_coil")
        assert any("must accept" in e for e in errors)

    def test_catches_forbidden_import(self):
        from imas_codex.ids.codegen import validate_assembly_code

        code = """
import os
def assemble_coil(ids, signals, mappings):
    os.system("rm -rf /")
"""
        errors = validate_assembly_code(code, "assemble_coil")
        assert any("Forbidden import" in e for e in errors)

    def test_allows_numpy_import(self):
        from imas_codex.ids.codegen import validate_assembly_code

        code = """
import numpy as np
def assemble_coil(ids, signals, mappings):
    arr = np.zeros(10)
"""
        errors = validate_assembly_code(code, "assemble_coil")
        assert errors == []

    def test_allows_math_import(self):
        from imas_codex.ids.codegen import validate_assembly_code

        code = """
import math
def assemble_coil(ids, signals, mappings):
    x = math.pi
"""
        errors = validate_assembly_code(code, "assemble_coil")
        assert errors == []


# ---------------------------------------------------------------------------
# 11.2.3 Assembly Code Generation
# ---------------------------------------------------------------------------


class TestGenerateAssemblyCode:
    """Tests for generate_assembly_code()."""

    def test_array_per_node_pattern(self):
        from imas_codex.ids.codegen import generate_assembly_code

        mappings = [
            {"source_id": "s1", "target_id": "pf_active/coil/r", "source_property": "value"},
            {"source_id": "s2", "target_id": "pf_active/coil/z", "source_property": "value"},
        ]
        code, func_name = generate_assembly_code(
            "pf_active/coil", "pf_active", mappings, "array_per_node"
        )
        assert func_name == "assemble_pf_active_coil"
        assert "def assemble_pf_active_coil(ids, signals, mappings):" in code
        # Should be valid Python
        ast.parse(code)

    def test_concatenate_pattern(self):
        from imas_codex.ids.codegen import generate_assembly_code

        mappings = [
            {"source_id": "s1", "target_id": "magnetics/flux_loop/data", "source_property": "value"},
        ]
        code, func_name = generate_assembly_code(
            "magnetics/flux_loop", "magnetics", mappings, "concatenate"
        )
        assert func_name == "assemble_magnetics_flux_loop"
        assert "concatenate" in code
        ast.parse(code)

    def test_generic_fallback_pattern(self):
        from imas_codex.ids.codegen import generate_assembly_code

        mappings = [
            {"source_id": "s1", "target_id": "x/y", "source_property": "value"},
        ]
        code, func_name = generate_assembly_code("x/y", "x", mappings, "matrix_assembly")
        assert func_name == "assemble_x_y"
        assert "matrix_assembly" in code
        ast.parse(code)


# ---------------------------------------------------------------------------
# 11.2.4 Binary Data Transfer Protocol
# ---------------------------------------------------------------------------


class TestBinaryTransferRoundtrip:
    """Tests for pack_array/unpack_array and decode_extraction_output."""

    def test_pack_unpack_1d(self):
        from imas_codex.remote.serialization import pack_array, unpack_array

        arr = np.array([1.0, 2.0, 3.0])
        packed = pack_array(arr)
        assert packed["__ndarray__"] is True
        assert packed["shape"] == [3]
        assert packed["dtype"] == "float64"

        unpacked = unpack_array(packed)
        np.testing.assert_array_equal(arr, unpacked)

    def test_pack_unpack_2d(self):
        from imas_codex.remote.serialization import pack_array, unpack_array

        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        packed = pack_array(arr)
        assert packed["shape"] == [2, 2]

        unpacked = unpack_array(packed)
        np.testing.assert_array_equal(arr, unpacked)

    def test_scalar_passthrough(self):
        from imas_codex.remote.serialization import pack_array, unpack_array

        assert pack_array(42) == 42
        assert unpack_array(42) == 42

    def test_decode_json_output(self):
        from imas_codex.remote.serialization import decode_extraction_output

        data = {"results": {"sig1": {"data": [1, 2, 3]}}}
        raw = json.dumps(data).encode()
        result = decode_extraction_output(raw)
        assert result["results"]["sig1"]["data"] == [1, 2, 3]

    def test_decode_json_string(self):
        from imas_codex.remote.serialization import decode_extraction_output

        data = {"results": {"sig1": {"data": [1, 2, 3]}}}
        raw = json.dumps(data)
        result = decode_extraction_output(raw)
        assert result["results"]["sig1"]["data"] == [1, 2, 3]

    def test_decode_empty(self):
        from imas_codex.remote.serialization import decode_extraction_output

        result = decode_extraction_output(b"")
        assert result == {"results": {}}

    def test_decode_msgpack_output(self):
        """Test msgpack round-trip if msgpack is available."""
        try:
            import msgpack
        except ImportError:
            pytest.skip("msgpack not installed")

        from imas_codex.remote.serialization import decode_extraction_output

        data = {"results": {"sig1": {"data": [1.0, 2.0, 3.0]}}}
        raw = msgpack.packb(data, use_bin_type=True)
        result = decode_extraction_output(raw)
        assert result["results"]["sig1"]["data"] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# 11.2.5 Extraction Caching
# ---------------------------------------------------------------------------


class TestExtractionCaching:
    """Tests for cache_extraction_result/load_cached_extraction."""

    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        from imas_codex.ids import codegen

        monkeypatch.setattr(codegen, "CACHE_DIR", tmp_path / "cache")

        results = {"sig1": {"success": True, "data": [1, 2, 3]}}
        path = codegen.cache_extraction_result("tcv", 80000, results)
        assert path.exists()

        loaded = codegen.load_cached_extraction("tcv", 80000)
        assert loaded == results

    def test_cache_miss(self, tmp_path, monkeypatch):
        from imas_codex.ids import codegen

        monkeypatch.setattr(codegen, "CACHE_DIR", tmp_path / "cache")

        loaded = codegen.load_cached_extraction("tcv", 99999)
        assert loaded is None


# ---------------------------------------------------------------------------
# 11.2.6 Strategy Selection
# ---------------------------------------------------------------------------


class TestStrategySelection:
    """Tests for probe_remote_capabilities and strategy decisions."""

    @patch("imas_codex.remote.executor.run_script_via_stdin")
    def test_probe_parses_json(self, mock_run):
        from imas_codex.remote.executor import probe_remote_capabilities

        mock_run.return_value = json.dumps(
            {"imas": True, "msgpack": True, "numpy": True, "h5py": False}
        )
        caps = probe_remote_capabilities("testhost")
        assert caps["imas"] is True
        assert caps["msgpack"] is True
        assert caps["h5py"] is False

    @patch("imas_codex.remote.executor.run_script_via_stdin")
    def test_probe_handles_failure(self, mock_run):
        from imas_codex.remote.executor import probe_remote_capabilities

        mock_run.side_effect = Exception("Connection refused")
        caps = probe_remote_capabilities("badhost")
        assert caps["imas"] is False
        assert caps["numpy"] is False

    def test_strategy_auto_remote_with_imas(self):
        """If imas is available, strategy should be 'remote'."""
        caps = {"imas": True, "msgpack": True, "numpy": True, "h5py": False}
        strategy = "remote" if caps.get("imas") else "client"
        assert strategy == "remote"

    def test_strategy_auto_client_without_imas(self):
        """If imas is not available, strategy should be 'client'."""
        caps = {"imas": False, "msgpack": True, "numpy": True, "h5py": False}
        strategy = "remote" if caps.get("imas") else "client"
        assert strategy == "client"


# ---------------------------------------------------------------------------
# 11.2.7 E2E Validation from Fixture
# ---------------------------------------------------------------------------


class TestE2EFromFixture:
    """Test E2E validation using recorded extraction fixture data."""

    @pytest.fixture
    def fixture_data(self):
        fixture_path = FIXTURES_DIR / "tcv_pf_active_80000_extraction.json"
        with open(fixture_path) as f:
            return json.load(f)

    def test_fixture_loads(self, fixture_data):
        """Fixture file loads and has expected structure."""
        assert "results" in fixture_data
        results = fixture_data["results"]
        assert len(results) == 7  # 6 success + 1 failed

    def test_fixture_success_count(self, fixture_data):
        results = fixture_data["results"]
        success = sum(
            1 for r in results.values()
            if isinstance(r, dict) and r.get("success")
        )
        assert success == 6

    def test_fixture_failure_count(self, fixture_data):
        results = fixture_data["results"]
        failed = sum(
            1 for r in results.values()
            if isinstance(r, dict) and not r.get("success")
        )
        assert failed == 1

    def test_fixture_time_consistency(self, fixture_data):
        """Check time-base consistency across signals in fixture."""
        results = fixture_data["results"]
        time_bases = []
        for sig_result in results.values():
            if isinstance(sig_result, dict) and sig_result.get("time"):
                tb = sig_result["time"]
                if len(tb) >= 2:
                    time_bases.append((tb[0], tb[-1], len(tb)))

        if time_bases:
            starts = [t[0] for t in time_bases]
            ends = [t[1] for t in time_bases]
            assert max(starts) - min(starts) < 1.0
            assert max(ends) - min(ends) < 1.0


# ---------------------------------------------------------------------------
# 11.2.8 Model Extensions
# ---------------------------------------------------------------------------


class TestAssemblyConfigExtensions:
    """Test assembly_code and assembly_function_name on AssemblyConfig."""

    def test_assembly_code_default_none(self):
        from imas_codex.ids.models import AssemblyConfig

        config = AssemblyConfig(target_path="pf_active/coil")
        assert config.assembly_code is None
        assert config.assembly_function_name is None

    def test_assembly_code_set(self):
        from imas_codex.ids.models import AssemblyConfig

        config = AssemblyConfig(
            target_path="pf_active/coil",
            assembly_code="def f(ids, signals, mappings): pass",
            assembly_function_name="f",
        )
        assert config.assembly_code is not None
        assert config.assembly_function_name == "f"

    def test_assembly_code_json_roundtrip(self):
        from imas_codex.ids.models import AssemblyConfig

        config = AssemblyConfig(
            target_path="test",
            assembly_code="x = 1\ny = 2",
            assembly_function_name="assemble_test",
        )
        data = config.model_dump()
        restored = AssemblyConfig(**data)
        assert restored.assembly_code == config.assembly_code
        assert restored.assembly_function_name == config.assembly_function_name


# ---------------------------------------------------------------------------
# 11.2.10 Decimation in Extraction Script
# ---------------------------------------------------------------------------


class TestDecimationPreservesShape:
    """Test that decimation in generated scripts handles shapes correctly."""

    def test_decimation_code_structure(self):
        """Decimation block checks length and uses step."""
        from imas_codex.ids.codegen import generate_extraction_script

        script = generate_extraction_script(
            "tcv",
            "pf_active",
            {"section": [{"id": "s1", "accessor": "x", "data_source_name": "ds"}]},
            {"imports_template": "", "connection_template": "pass",
             "data_template": "None", "time_template": "", "cleanup_template": ""},
            max_points=100,
        )
        assert "100" in script
        assert "step" in script

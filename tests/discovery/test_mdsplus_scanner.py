"""Tests for the unified MDSplusScanner plugin (Phase 6).

Verifies: thin scan() loop, subtree expansion, version resolution,
TDI linkage integration, check() delegation, and ScanResult shape.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.discovery.signals.scanners.mdsplus import MDSplusScanner

# Patch targets — imports are inside scan() method body
_RTD = "imas_codex.discovery.mdsplus.pipeline.run_tree_discovery"
_TDI = "imas_codex.discovery.mdsplus.tdi_linkage.link_tdi_to_tree_nodes"


@pytest.fixture
def scanner():
    return MDSplusScanner()


@pytest.fixture
def static_config():
    """Config with a single versioned (static) tree."""
    return {
        "connection_tree": "tcv_shot",
        "reference_shot": 85000,
        "setup_commands": ["source /etc/profile.d/mds.sh"],
        "trees": [
            {
                "tree_name": "tcv_machconfig",
                "versions": [
                    {"version": 1, "description": "Initial"},
                ],
            },
        ],
    }


@pytest.fixture
def dynamic_config():
    """Config with a dynamic (shot-scoped) parent tree with subtrees."""
    return {
        "connection_tree": "tcv_shot",
        "reference_shot": 85000,
        "setup_commands": ["source /etc/profile.d/mds.sh"],
        "trees": [
            {
                "tree_name": "tcv_shot",
                "subtrees": [
                    {"tree_name": "results", "node_usages": ["NUMERIC", "SIGNAL"]},
                    {"tree_name": "magnetics", "node_usages": ["SIGNAL"]},
                ],
            },
        ],
    }


@pytest.fixture
def mixed_config():
    """Config with both versioned and dynamic trees."""
    return {
        "connection_tree": "tcv_shot",
        "reference_shot": 85000,
        "setup_commands": ["source /etc/profile.d/mds.sh"],
        "trees": [
            {
                "tree_name": "tcv_machconfig",
                "versions": [
                    {"version": 1, "description": "Initial"},
                ],
            },
            {
                "tree_name": "tcv_shot",
                "subtrees": [
                    {"tree_name": "results"},
                ],
            },
        ],
    }


# --- scan() tests ---


@pytest.mark.asyncio
async def test_scan_versioned_tree(scanner, static_config):
    """Versioned tree calls run_tree_discovery with version list."""
    mock_stats = {"signals_promoted": 42, "versions_extracted": 1}

    with (
        patch(
            _RTD,
            new_callable=AsyncMock,
            return_value=mock_stats,
        ) as mock_rtd,
        patch(
            _TDI,
            return_value=0,
        ),
    ):
        result = await scanner.scan("tcv", "tcv-ssh", static_config)

    mock_rtd.assert_called_once()
    call_kw = mock_rtd.call_args.kwargs
    assert call_kw["facility"] == "tcv"
    assert call_kw["ssh_host"] == "tcv-ssh"
    assert call_kw["data_source_name"] == "tcv_machconfig"
    assert call_kw["ver_list"] == [1]
    assert result.signals == []
    assert result.stats["signals_promoted"] == 42
    assert result.stats["tcv_machconfig"]["versions_extracted"] == 1


@pytest.mark.asyncio
async def test_scan_dynamic_subtrees(scanner, dynamic_config):
    """Dynamic trees expand subtrees and use reference_shot as version."""
    call_log = []

    async def fake_rtd(**kwargs):
        call_log.append(kwargs)
        return {"signals_promoted": 10, "versions_extracted": 1}

    with (
        patch(
            _RTD,
            side_effect=fake_rtd,
        ),
        patch(
            _TDI,
            return_value=0,
        ),
    ):
        result = await scanner.scan("tcv", "tcv-ssh", dynamic_config)

    # Two subtrees should be processed
    assert len(call_log) == 2
    assert call_log[0]["data_source_name"] == "results"
    assert call_log[0]["ver_list"] == [85000]
    assert call_log[1]["data_source_name"] == "magnetics"
    assert call_log[1]["ver_list"] == [85000]

    # node_usages from subtree config should be in tree_config
    assert call_log[0]["tree_config"]["node_usages"] == ["NUMERIC", "SIGNAL"]
    assert call_log[1]["tree_config"]["node_usages"] == ["SIGNAL"]

    assert result.stats["signals_promoted"] == 20


@pytest.mark.asyncio
async def test_scan_mixed_trees(scanner, mixed_config):
    """Mixed config processes both static and dynamic trees."""
    call_log = []

    async def fake_rtd(**kwargs):
        call_log.append(kwargs["data_source_name"])
        return {"signals_promoted": 5, "versions_extracted": 1}

    with (
        patch(
            _RTD,
            side_effect=fake_rtd,
        ),
        patch(
            _TDI,
            return_value=3,
        ),
    ):
        result = await scanner.scan("tcv", "tcv-ssh", mixed_config)

    assert call_log == ["tcv_machconfig", "results"]
    assert result.stats["signals_promoted"] == 10
    assert result.stats["tdi_links"] == 3


@pytest.mark.asyncio
async def test_scan_no_versions_or_ref_shot(scanner):
    """Tree with no versions and no reference_shot logs warning."""
    config = {
        "trees": [{"tree_name": "orphan_tree"}],
    }

    with (
        patch(
            _RTD,
            new_callable=AsyncMock,
        ) as mock_rtd,
        patch(
            _TDI,
            return_value=0,
        ),
    ):
        result = await scanner.scan("tcv", "tcv-ssh", config)

    mock_rtd.assert_not_called()
    assert "error" in result.stats["orphan_tree"]


@pytest.mark.asyncio
async def test_scan_empty_trees(scanner):
    """Empty trees list produces empty result."""
    config = {"trees": []}

    with patch(
        _TDI,
        return_value=0,
    ):
        result = await scanner.scan("tcv", "tcv-ssh", config)

    assert result.signals == []
    assert result.stats["signals_promoted"] == 0


@pytest.mark.asyncio
async def test_scan_tree_failure_continues(scanner, mixed_config):
    """Failure in one tree doesn't prevent processing others."""
    call_count = 0

    async def fail_first(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("SSH timeout")
        return {"signals_promoted": 7, "versions_extracted": 1}

    with (
        patch(
            _RTD,
            side_effect=fail_first,
        ),
        patch(
            _TDI,
            return_value=0,
        ),
    ):
        result = await scanner.scan("tcv", "tcv-ssh", mixed_config)

    assert "error" in result.stats["tcv_machconfig"]
    assert result.stats["results"]["signals_promoted"] == 7
    assert result.stats["signals_promoted"] == 7


@pytest.mark.asyncio
async def test_scan_tdi_linkage_runs(scanner, static_config):
    """TDI linkage runs after tree processing."""
    with (
        patch(
            _RTD,
            new_callable=AsyncMock,
            return_value={"signals_promoted": 0},
        ),
        patch(
            _TDI,
            return_value=12,
        ) as mock_tdi,
    ):
        result = await scanner.scan("tcv", "tcv-ssh", static_config)

    mock_tdi.assert_called_once_with("tcv")
    assert result.stats["tdi_links"] == 12


@pytest.mark.asyncio
async def test_scan_tdi_linkage_failure_nonfatal(scanner, static_config):
    """TDI linkage failure doesn't fail the scan."""
    with (
        patch(
            _RTD,
            new_callable=AsyncMock,
            return_value={"signals_promoted": 5},
        ),
        patch(
            _TDI,
            side_effect=RuntimeError("graph down"),
        ),
    ):
        result = await scanner.scan("tcv", "tcv-ssh", static_config)

    assert result.stats["signals_promoted"] == 5
    assert result.stats["tdi_links"] == 0


@pytest.mark.asyncio
async def test_scan_data_access_created(scanner, static_config):
    """ScanResult includes DataAccess for connection_tree."""
    with (
        patch(
            _RTD,
            new_callable=AsyncMock,
            return_value={"signals_promoted": 0},
        ),
        patch(
            _TDI,
            return_value=0,
        ),
    ):
        result = await scanner.scan("tcv", "tcv-ssh", static_config)

    assert result.data_access is not None
    assert result.data_access.id == "tcv:mdsplus:tree_tdi"
    assert "tcv_shot" in result.data_access.connection_template


@pytest.mark.asyncio
async def test_scan_data_access_fallback(scanner):
    """DataAccess falls back to first tree when no connection_tree."""
    config = {
        "reference_shot": 85000,
        "trees": [{"tree_name": "first_tree", "versions": [{"version": 1}]}],
    }

    with (
        patch(
            _RTD,
            new_callable=AsyncMock,
            return_value={"signals_promoted": 0},
        ),
        patch(
            _TDI,
            return_value=0,
        ),
    ):
        result = await scanner.scan("tcv", "tcv-ssh", config)

    assert result.data_access is not None
    assert "first_tree" in result.data_access.connection_template


@pytest.mark.asyncio
async def test_scan_setup_commands_merged(scanner):
    """setup_commands from parent config are merged into tree_config."""
    config = {
        "reference_shot": 85000,
        "setup_commands": ["source /etc/mds.sh"],
        "trees": [{"tree_name": "mytest", "versions": [{"version": 1}]}],
    }

    with (
        patch(
            _RTD,
            new_callable=AsyncMock,
            return_value={"signals_promoted": 0},
        ) as mock_rtd,
        patch(
            _TDI,
            return_value=0,
        ),
    ):
        await scanner.scan("tcv", "tcv-ssh", config)

    tc = mock_rtd.call_args.kwargs["tree_config"]
    assert tc["setup_commands"] == ["source /etc/mds.sh"]


@pytest.mark.asyncio
async def test_scan_tree_level_setup_not_overridden(scanner):
    """Tree-level setup_commands are NOT overridden by parent."""
    config = {
        "reference_shot": 85000,
        "setup_commands": ["source /etc/mds.sh"],
        "trees": [
            {
                "tree_name": "mytest",
                "versions": [{"version": 1}],
                "setup_commands": ["source /opt/mds.sh"],
            }
        ],
    }

    with (
        patch(
            _RTD,
            new_callable=AsyncMock,
            return_value={"signals_promoted": 0},
        ) as mock_rtd,
        patch(
            _TDI,
            return_value=0,
        ),
    ):
        await scanner.scan("tcv", "tcv-ssh", config)

    tc = mock_rtd.call_args.kwargs["tree_config"]
    assert tc["setup_commands"] == ["source /opt/mds.sh"]


@pytest.mark.asyncio
async def test_scan_reference_shot_override(scanner, static_config):
    """reference_shot parameter overrides config value."""
    with (
        patch(
            _RTD,
            new_callable=AsyncMock,
            return_value={"signals_promoted": 0},
        ),
        patch(
            _TDI,
            return_value=0,
        ),
    ):
        # static_config has reference_shot=85000, but we override
        result = await scanner.scan(
            "tcv", "tcv-ssh", static_config, reference_shot=99000
        )

    assert result.metadata == {"connection_tree": "tcv_shot"}


# --- check() tests ---


@pytest.mark.asyncio
async def test_check_no_reference_shot(scanner):
    """check() returns invalid when no reference_shot."""
    from imas_codex.graph.models import FacilitySignal, FacilitySignalStatus

    signals = [
        FacilitySignal(
            id="tcv:results/top/ip",
            facility_id="tcv",
            status=FacilitySignalStatus.discovered,
            physics_domain="general",
            data_access="tcv:mdsplus:tree_tdi",
            accessor="\\RESULTS::TOP:IP",
        ),
    ]
    results = await scanner.check("tcv", "tcv-ssh", signals, {})
    assert len(results) == 1
    assert results[0]["valid"] is False
    assert "no reference_shot" in results[0]["error"]


@pytest.mark.asyncio
async def test_check_delegates_to_remote(scanner):
    """check() calls check_signals_batch.py via SSH."""
    from imas_codex.graph.models import FacilitySignal, FacilitySignalStatus

    signals = [
        FacilitySignal(
            id="tcv:results/top/ip",
            facility_id="tcv",
            status=FacilitySignalStatus.discovered,
            physics_domain="general",
            data_access="tcv:mdsplus:tree_tdi",
            accessor="\\RESULTS::TOP:IP",
            data_source_name="results",
        ),
    ]
    config = {"reference_shot": 85000, "setup_commands": ["source /mds.sh"]}

    mock_output = '{"results": [{"id": "tcv:results/top/ip", "success": true, "shape": [100], "dtype": "float64"}]}'
    with patch(
        "imas_codex.discovery.signals.scanners.mdsplus.run_python_script",
        return_value=mock_output,
    ) as mock_rps:
        results = await scanner.check("tcv", "tcv-ssh", signals, config)

    mock_rps.assert_called_once()
    assert mock_rps.call_args[0][0] == "check_signals_batch.py"
    assert len(results) == 1
    assert results[0]["valid"] is True
    assert results[0]["shape"] == [100]


@pytest.mark.asyncio
async def test_check_handles_remote_error(scanner):
    """check() handles SSH failures gracefully."""
    from imas_codex.graph.models import FacilitySignal, FacilitySignalStatus

    signals = [
        FacilitySignal(
            id="tcv:results/top/ip",
            facility_id="tcv",
            status=FacilitySignalStatus.discovered,
            physics_domain="general",
            data_access="tcv:mdsplus:tree_tdi",
            accessor="\\RESULTS::TOP:IP",
        ),
    ]
    config = {"reference_shot": 85000}

    with patch(
        "imas_codex.discovery.signals.scanners.mdsplus.run_python_script",
        side_effect=RuntimeError("SSH failed"),
    ):
        results = await scanner.check("tcv", "tcv-ssh", signals, config)

    assert len(results) == 1
    assert results[0]["valid"] is False


# --- registration ---


def test_scanner_type():
    """Scanner type is 'mdsplus'."""
    assert MDSplusScanner().scanner_type == "mdsplus"


def test_scanner_registered():
    """MDSplusScanner is auto-registered."""
    from imas_codex.discovery.signals.scanners.base import _registry

    assert "mdsplus" in _registry

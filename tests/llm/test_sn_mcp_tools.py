"""Plan 40 Phase 3 — MCP tool wrapper tests for the SN search facility.

Covers §9.7 (dd_only gate suppresses SN tools), §9.8 (3 new tools
registered when SN tools enabled), and basic format-report behaviour
of the wrappers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _registered_tool_names(server) -> set[str]:
    return {
        key.split("tool:", 1)[1].split("@", 1)[0]
        for key in server.mcp._local_provider._components
        if key.startswith("tool:")
    }


# ---------------------------------------------------------------------------
# §9.7 — dd_only gate
# ---------------------------------------------------------------------------


def test_dd_only_true_suppresses_standard_name_tools() -> None:
    """dd_only=True must hide ALL StandardName MCP tools (plan 40 §9.7)."""
    from imas_codex.llm.server import AgentsServer

    server = AgentsServer(dd_only=True, include_standard_names=True)
    tools = _registered_tool_names(server)

    sn_tool_names = {
        "search_standard_names",
        "fetch_standard_names",
        "list_standard_names",
        "list_grammar_vocabulary",
        "list_promotion_candidates",
        "find_related_standard_names",
        "check_standard_names",
        "get_standard_name_summary",
    }
    overlap = tools & sn_tool_names
    assert overlap == set(), f"SN tools leaked when dd_only=True: {overlap}"


def test_dd_only_false_registers_sn_tools() -> None:
    """When dd_only=False, all 8 SN tools must be registered (§9.8)."""
    from imas_codex.llm.server import AgentsServer

    server = AgentsServer(dd_only=False, include_standard_names=True)
    tools = _registered_tool_names(server)

    expected = {
        "search_standard_names",
        "fetch_standard_names",
        "list_standard_names",
        "list_grammar_vocabulary",
        "list_promotion_candidates",
        "find_related_standard_names",
        "check_standard_names",
        "get_standard_name_summary",
    }
    missing = expected - tools
    assert missing == set(), f"Missing SN tools: {missing}"


def test_include_standard_names_false_suppresses_all() -> None:
    """include_standard_names=False also suppresses SN tools."""
    from imas_codex.llm.server import AgentsServer

    server = AgentsServer(dd_only=False, include_standard_names=False)
    tools = _registered_tool_names(server)
    assert "search_standard_names" not in tools
    assert "find_related_standard_names" not in tools


# ---------------------------------------------------------------------------
# §9.8 — find_related_standard_names wrapper
# ---------------------------------------------------------------------------


def test_find_related_wrapper_formats_buckets() -> None:
    """_find_related_standard_names renders buckets in deterministic order."""
    from imas_codex.llm import sn_tools

    fake_buckets = {
        "Grammar Family": [{"name": "electron_density", "description": "n_e"}],
        "Unit Companions": [{"name": "ion_temperature", "description": "T_i"}],
    }
    gc = MagicMock()
    with patch.object(sn_tools, "_find_related_backing", return_value=fake_buckets):
        out = sn_tools._find_related_standard_names("electron_temperature", gc=gc)
    assert "Grammar Family" in out
    assert "Unit Companions" in out
    assert "electron_density" in out
    # Empty bucket suppression: 'COCOS Companions' should NOT be in the output
    assert "COCOS Companions" not in out


def test_find_related_wrapper_empty_buckets() -> None:
    from imas_codex.llm import sn_tools

    gc = MagicMock()
    with patch.object(sn_tools, "_find_related_backing", return_value={}):
        out = sn_tools._find_related_standard_names("foo", gc=gc)
    assert "No related names" in out


# ---------------------------------------------------------------------------
# check_standard_names wrapper
# ---------------------------------------------------------------------------


def test_check_standard_names_formats_table() -> None:
    from imas_codex.llm import sn_tools

    fake = [
        {"name": "good_name", "exists": True, "suggestion": "", "reason": ""},
        {
            "name": "typo_name",
            "exists": False,
            "suggestion": "type_name",
            "reason": "levenshtein",
        },
    ]
    gc = MagicMock()
    with patch.object(sn_tools, "_check_names_backing", return_value=fake):
        out = sn_tools._check_standard_names("good_name typo_name", gc=gc)
    assert "good_name" in out and "typo_name" in out
    assert "✓" in out and "✗" in out
    assert "type_name" in out


def test_check_standard_names_empty_input() -> None:
    from imas_codex.llm import sn_tools

    gc = MagicMock()
    out = sn_tools._check_standard_names("   ", gc=gc)
    assert "No names" in out


# ---------------------------------------------------------------------------
# get_standard_name_summary wrapper
# ---------------------------------------------------------------------------


def test_get_summary_wrapper_renders_sections() -> None:
    from imas_codex.llm import sn_tools

    summary = {
        "physical_base": "temperature",
        "count": 12,
        "segment_distinct": {
            "subject": ["electron", "ion"],
            "transformation": [],
            "component": [],
            "position": [],
            "process": [],
            "geometric_base": [],
        },
        "unit_distinct": ["K", "eV"],
        "cocos_distinct": [],
        "physics_domain_distinct": ["core_profiles"],
        "sample_names": ["electron_temperature", "ion_temperature"],
        "lineage": {
            "predecessors_count": 1,
            "predecessors_max_depth": 1,
            "successors_count": 0,
            "successors_max_depth": 0,
            "refined_from_count": 2,
            "refined_from_max_depth": 1,
            "total_edges": 3,
        },
    }
    gc = MagicMock()
    with patch.object(sn_tools, "_summarise_family_backing", return_value=summary):
        out = sn_tools._get_standard_name_summary("temperature", gc=gc)
    assert "temperature" in out
    assert "12 members" in out
    assert "electron_temperature" in out
    assert "Lineage Counts" in out
    assert "Refined-from" in out


def test_get_summary_empty_family() -> None:
    from imas_codex.llm import sn_tools

    gc = MagicMock()
    with patch.object(
        sn_tools,
        "_summarise_family_backing",
        return_value={"physical_base": "x", "count": 0},
    ):
        out = sn_tools._get_standard_name_summary("x", gc=gc)
    assert "No standard names" in out

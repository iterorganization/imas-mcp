"""Regression tests for Tools facade delegation consistency.

Ensures that every facade method on the Tools class delegates to
a real method on its backend tool instance. Catches rename mismatches
where the facade was renamed but the backend was not (or vice versa).
"""

import inspect

import pytest

from imas_codex.tools import Tools
from imas_codex.tools.graph_search import (
    GraphClustersTool,
    GraphIdentifiersTool,
    GraphListTool,
    GraphOverviewTool,
    GraphPathContextTool,
    GraphPathTool,
    GraphSearchTool,
    GraphStructureTool,
)
from imas_codex.tools.version_tool import VersionTool

# Canonical mapping: facade method -> (backend attribute, backend method)
FACADE_DELEGATION_MAP = {
    "search_dd_paths": ("search_tool", "search_dd_paths"),
    "check_dd_paths": ("path_tool", "check_dd_paths"),
    "fetch_dd_paths": ("path_tool", "fetch_dd_paths"),
    "fetch_dd_error_fields": ("path_tool", "fetch_dd_error_fields"),
    "list_dd_paths": ("list_tool", "list_dd_paths"),
    "get_dd_overview": ("overview_tool", "get_dd_overview"),
    "get_dd_identifiers": ("identifiers_tool", "get_dd_identifiers"),
    "get_dd_path_context": ("path_context_tool", "get_dd_path_context"),
    "analyze_dd_structure": ("structure_tool", "analyze_dd_structure"),
    "get_dd_cocos_fields": ("structure_tool", "get_cocos_fields"),
    "search_dd_clusters": ("clusters_tool", "search_dd_clusters"),
    "get_dd_versions": ("version_tool", "get_dd_versions"),
    "get_dd_version_context": ("version_tool", "get_dd_version_context"),
    "get_dd_changelog": ("version_tool", "get_dd_changelog"),
    "export_imas_ids": ("structure_tool", "export_imas_ids"),
    "export_imas_domain": ("structure_tool", "export_imas_domain"),
    "analyze_dd_coverage": ("dd_analytics_tool", "analyze_dd_coverage"),
    "check_dd_units": ("dd_analytics_tool", "check_dd_units"),
    "analyze_dd_changes": ("dd_analytics_tool", "analyze_dd_changes"),
}

# Backend class for each tool attribute
BACKEND_CLASSES = {
    "search_tool": GraphSearchTool,
    "path_tool": GraphPathTool,
    "list_tool": GraphListTool,
    "overview_tool": GraphOverviewTool,
    "clusters_tool": GraphClustersTool,
    "identifiers_tool": GraphIdentifiersTool,
    "path_context_tool": GraphPathContextTool,
    "structure_tool": GraphStructureTool,
    "version_tool": VersionTool,
}


class TestFacadeDelegation:
    """Verify facade methods exist and delegate to real backend methods."""

    @pytest.mark.parametrize("facade_method", sorted(FACADE_DELEGATION_MAP.keys()))
    def test_facade_method_exists(self, facade_method):
        """Every mapped facade method must exist on Tools."""
        assert hasattr(Tools, facade_method), f"Tools.{facade_method} does not exist"
        method = getattr(Tools, facade_method)
        assert inspect.iscoroutinefunction(method), (
            f"Tools.{facade_method} must be async"
        )

    @pytest.mark.parametrize(
        "facade_method,backend_info",
        sorted(FACADE_DELEGATION_MAP.items()),
    )
    def test_backend_method_exists(self, facade_method, backend_info):
        """The backend method that the facade delegates to must exist."""
        backend_attr, backend_method = backend_info
        backend_class = BACKEND_CLASSES.get(backend_attr)

        if backend_class is None:
            # dd_analytics_tool — imported separately, just check it exists
            from imas_codex.tools.dd_analytics_tool import DDAnalyticsTool

            backend_class = DDAnalyticsTool

        assert hasattr(backend_class, backend_method), (
            f"Facade {facade_method} delegates to "
            f"{backend_attr}.{backend_method}() but "
            f"{backend_class.__name__}.{backend_method} does not exist. "
            f"This is a rename mismatch — check that both layers use "
            f"consistent method names."
        )

    @pytest.mark.parametrize(
        "facade_method,backend_info",
        sorted(FACADE_DELEGATION_MAP.items()),
    )
    def test_facade_source_references_backend(self, facade_method, backend_info):
        """The facade method source must reference the correct backend call."""
        backend_attr, backend_method = backend_info
        source = inspect.getsource(getattr(Tools, facade_method))
        expected_call = f"self.{backend_attr}.{backend_method}"
        assert expected_call in source, (
            f"Tools.{facade_method} source does not contain "
            f"'{expected_call}'. The delegation may be broken."
        )

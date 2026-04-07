"""Regression tests for Tools registered tool name consistency.

Ensures that all expected DD tool methods exist on the backend
tool instances and are discoverable via the Tools container.
No facade layer — callers access tool instances directly.
"""

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

# Canonical mapping: tool_instance_attr -> (backend_class, expected_methods)
TOOL_METHOD_MAP = {
    "search_tool": (GraphSearchTool, ["search_dd_paths"]),
    "path_tool": (
        GraphPathTool,
        ["check_dd_paths", "fetch_dd_paths", "fetch_dd_error_fields"],
    ),
    "list_tool": (GraphListTool, ["list_dd_paths"]),
    "overview_tool": (GraphOverviewTool, ["get_dd_overview"]),
    "clusters_tool": (GraphClustersTool, ["search_dd_clusters"]),
    "identifiers_tool": (GraphIdentifiersTool, ["get_dd_identifiers"]),
    "path_context_tool": (GraphPathContextTool, ["get_dd_path_context"]),
    "structure_tool": (
        GraphStructureTool,
        [
            "analyze_dd_structure",
            "get_cocos_fields",
            "export_imas_ids",
            "export_imas_domain",
        ],
    ),
    "version_tool": (
        VersionTool,
        ["get_dd_versions", "get_dd_version_context", "get_dd_changelog"],
    ),
}


def _all_method_params():
    """Yield (tool_attr, class, method_name) for parametrize."""
    for tool_attr, (cls, methods) in TOOL_METHOD_MAP.items():
        for method in methods:
            yield tool_attr, cls, method


class TestToolMethodExistence:
    """Verify backend tool methods exist and are async."""

    @pytest.mark.parametrize(
        "tool_attr,backend_class,method_name",
        list(_all_method_params()),
        ids=[f"{a}.{m}" for a, _, m in _all_method_params()],
    )
    def test_backend_method_exists(self, tool_attr, backend_class, method_name):
        """Every expected method must exist on its backend class."""
        assert hasattr(backend_class, method_name), (
            f"{backend_class.__name__}.{method_name} does not exist. "
            f"Check that the method was renamed correctly."
        )

    def test_no_facade_methods_on_tools(self):
        """Tools class must not have async facade delegation methods."""
        import inspect

        facade_names = {
            "search_dd_paths",
            "check_dd_paths",
            "fetch_dd_paths",
            "list_dd_paths",
            "get_dd_overview",
            "get_dd_identifiers",
            "get_dd_path_context",
            "get_dd_cocos_fields",
            "export_imas_ids",
            "export_imas_domain",
            "get_dd_versions",
            "search_dd_clusters",
            "get_dd_version_context",
            "get_dd_changelog",
            "fetch_dd_error_fields",
        }
        for name in facade_names:
            if hasattr(Tools, name):
                method = getattr(Tools, name)
                assert not inspect.iscoroutinefunction(method), (
                    f"Tools.{name} is an async facade method — "
                    f"these should be removed. Callers should use "
                    f"tools.<tool_instance>.{name}() directly."
                )

    def test_no_imas_named_methods(self):
        """No backend tool class should have _imas_ named methods (old naming)."""
        old_names = [
            "search_imas_paths",
            "check_imas_paths",
            "fetch_imas_paths",
            "list_imas_paths",
            "get_imas_overview",
            "search_imas_clusters",
            "get_imas_identifiers",
            "get_imas_path_context",
            "analyze_imas_structure",
        ]
        for _tool_attr, (cls, _) in TOOL_METHOD_MAP.items():
            for old_name in old_names:
                assert not hasattr(cls, old_name), (
                    f"{cls.__name__} still has old method {old_name}. "
                    f"Rename to _dd_ convention."
                )

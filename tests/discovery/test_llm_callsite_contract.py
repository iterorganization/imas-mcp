"""AST-based contract test: every LLM call site must set service= explicitly."""

import ast
import sys
from pathlib import Path

import pytest

# Import valid services from the production module
from imas_codex.discovery.base.llm import _VALID_SERVICES

# Target functions that need service= tagging
_TARGET_FUNCTIONS = {
    "call_llm_structured",
    "acall_llm_structured",
    "call_llm",
    "acall_llm",
}

# Files that DEFINE the functions (not call sites)
_DEFINITION_FILES = {
    "imas_codex/discovery/base/llm.py",
}

# Wrapper funnels — these files define internal wrappers that call the target
# functions. The wrapper itself should be tagged, not each caller of the wrapper.
_WRAPPER_FUNNELS = {
    "imas_codex/ids/mapping.py": {"_call_llm", "_acall_llm"},
}


def _collect_llm_call_sites() -> list[tuple[str, int, str]]:
    """Walk the source tree and find all calls to target LLM functions.

    Returns list of (file_path, line_number, function_name) tuples
    for calls that are MISSING the service= keyword argument.
    """
    root = Path("imas_codex")
    violations = []

    for py_file in sorted(root.rglob("*.py")):
        rel_path = str(py_file)

        # Skip definition files
        if rel_path in _DEFINITION_FILES:
            continue

        try:
            source = py_file.read_text()
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            # Get the function name being called
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name not in _TARGET_FUNCTIONS:
                continue

            # Check if this is inside a wrapper funnel
            # (We only care if the wrapper function itself tags service=)
            wrapper_names = _WRAPPER_FUNNELS.get(rel_path, set())
            if wrapper_names:
                # This file has known wrapper functions — check if this call
                # is at the top-level (inside a wrapper function definition)
                # The wrapper should have service= tagged
                pass

            # Check for service= keyword argument
            has_service = any(kw.arg == "service" for kw in node.keywords)

            if not has_service:
                violations.append((rel_path, node.lineno, func_name))

    return violations


def _collect_service_values() -> list[tuple[str, int, str, str]]:
    """Find all service= keyword values and check they are valid.

    Returns list of (file_path, line_number, function_name, value) tuples
    for calls with invalid service= values.
    """
    root = Path("imas_codex")
    invalid = []

    for py_file in sorted(root.rglob("*.py")):
        rel_path = str(py_file)
        if rel_path in _DEFINITION_FILES:
            continue

        try:
            source = py_file.read_text()
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name not in _TARGET_FUNCTIONS:
                continue

            for kw in node.keywords:
                if kw.arg == "service" and isinstance(kw.value, ast.Constant):
                    if kw.value.value not in _VALID_SERVICES:
                        invalid.append(
                            (rel_path, node.lineno, func_name, kw.value.value)
                        )

    return invalid


class TestLLMCallSiteContract:
    """Every production LLM call site must have service= with a valid value."""

    def test_all_call_sites_have_service_tag(self):
        """No call to call_llm_structured/acall_llm_structured/call_llm/acall_llm
        should be missing a service= keyword argument."""
        violations = _collect_llm_call_sites()
        if violations:
            msg = "LLM call sites missing service= tag:\n"
            for path, line, func in violations:
                msg += f"  {path}:{line} — {func}()\n"
            pytest.fail(msg)

    def test_all_service_values_are_valid(self):
        """All service= values must be in _VALID_SERVICES."""
        invalid = _collect_service_values()
        if invalid:
            msg = "LLM call sites with invalid service= values:\n"
            for path, line, func, value in invalid:
                msg += f"  {path}:{line} — {func}(service={value!r})\n"
            msg += f"Valid values: {sorted(_VALID_SERVICES)}"
            pytest.fail(msg)

    def test_valid_services_matches_type(self):
        """_VALID_SERVICES should match LLM_SERVICE Literal args."""
        from imas_codex.discovery.base.llm import LLM_SERVICE

        expected = set(LLM_SERVICE.__args__)
        assert _VALID_SERVICES == expected

#!/usr/bin/env python3
"""Fix test files to use Pydantic models instead of dict."""

import re
from pathlib import Path


def fix_test_file(file_path: Path) -> None:
    """Fix a test file to use Pydantic models."""
    content = file_path.read_text()

    # Add imports at top of file if not present
    if "from imas_mcp.models.response_models import" not in content:
        # Find the import section
        import_match = re.search(r"(import pytest\n)", content)
        if import_match:
            imports = """
from imas_mcp.models.response_models import (
    ConceptResult,
    StructureResult, 
    IdentifierResult,
    RelationshipResult,
    SearchResponse,
    OverviewResult,
    ExportData
)
"""
            content = content.replace(
                import_match.group(1), import_match.group(1) + imports
            )

    # Fix isinstance checks
    content = re.sub(
        r"assert isinstance\(result, dict\)",
        'assert hasattr(result, "__dict__")  # Pydantic model',
        content,
    )
    content = re.sub(
        r"isinstance\(result, dict\)", 'hasattr(result, "__dict__")', content
    )

    # Fix error checks
    content = re.sub(
        r'"error" not in result',
        'not hasattr(result.ai_insights, "error") if hasattr(result, "ai_insights") else True',
        content,
    )
    content = re.sub(
        r'"error" in result',
        'hasattr(result.ai_insights, "error") if hasattr(result, "ai_insights") else False',
        content,
    )

    # Fix field access patterns
    replacements = [
        (r'result\["(\w+)"\]', r"result.\1"),  # result["field"] -> result.field
        (
            r'assert "(\w+)" in result',
            r'assert hasattr(result, "\1")',
        ),  # "field" in result -> hasattr(result, "field")
        (r'if "(\w+)" in result', r'if hasattr(result, "\1")'),
        (
            r'isinstance\(result\["(\w+)"\]',
            r"isinstance(result.\1",
        ),  # isinstance(result["field"] -> isinstance(result.field
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    # Fix specific tool result type checks
    tool_checks = [
        ("await tools.explain_concept", "ConceptResult"),
        ("await tools.analyze_ids_structure", "StructureResult"),
        ("await tools.explore_identifiers", "IdentifierResult"),
        ("await tools.explore_relationships", "RelationshipResult"),
        ("await tools.search_imas", "SearchResponse"),
        ("await tools.get_overview", "OverviewResult"),
        ("await tools.export_", "ExportData"),
    ]

    for tool_call, result_type in tool_checks:
        # Find function calls and update isinstance checks
        pattern = rf'({re.escape(tool_call)}[^;]*?)(\n\s*assert hasattr\(result, "__dict__"\)\s*# Pydantic model)'
        replacement = rf"\1\n        assert isinstance(result, {result_type})"
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    file_path.write_text(content)
    print(f"Fixed {file_path}")


def main():
    """Fix all test files."""
    test_dir = Path("tests/features")

    for test_file in test_dir.glob("test_*.py"):
        print(f"Processing {test_file}")
        try:
            fix_test_file(test_file)
        except Exception as e:
            print(f"Error fixing {test_file}: {e}")


if __name__ == "__main__":
    main()

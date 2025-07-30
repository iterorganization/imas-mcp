#!/usr/bin/env python3
"""Quick fix for syntax errors in test files."""

import re
from pathlib import Path


def fix_syntax_errors(file_path: Path) -> None:
    """Fix syntax errors in test file."""
    content = file_path.read_text()

    # Fix the malformed conditional expressions
    content = re.sub(
        r'if not hasattr\(result\.ai_insights, "error"\) if hasattr\(result, "ai_insights"\) else True:',
        'if not (hasattr(result, "ai_insights") and hasattr(result.ai_insights, "error")):',
        content,
    )

    file_path.write_text(content)
    print(f"Fixed syntax errors in {file_path}")


def main():
    """Fix syntax errors in test files."""
    for test_file in Path("tests/features").glob("test_*.py"):
        print(f"Processing {test_file}")
        try:
            fix_syntax_errors(test_file)
        except Exception as e:
            print(f"Error fixing {test_file}: {e}")


if __name__ == "__main__":
    main()

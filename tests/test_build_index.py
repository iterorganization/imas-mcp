"""Test for the build_index script."""

import subprocess
import sys
from pathlib import Path


def test_build_index_script():
    """Test that the build_index script runs without errors."""
    # Get the project root
    project_root = Path(__file__).parent.parent

    # Test that the script can be called
    result = subprocess.run(
        [sys.executable, "-c", "from scripts.build_index import main; main()"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    # Check that it runs without error
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    # Check that it outputs an index name
    output = result.stdout.strip().split("\n")[
        -1
    ]  # Get last line which should be index name
    assert output.startswith("lexicographic_"), f"Expected index name, got: {output}"

    print(f"✓ Build script test passed. Index name: {output}")


if __name__ == "__main__":
    test_build_index_script()

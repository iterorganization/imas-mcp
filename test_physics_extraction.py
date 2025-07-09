"""
Simple test script for physics extraction system.

This demonstrates the basic functionality of the physics extraction system
using the existing IMAS JSON data.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import after path setup
from imas_mcp.physics_extraction.__main__ import (  # noqa: E402
    setup_extraction_environment,
    run_extraction_session,
)


def test_basic_extraction():
    """Test basic extraction functionality."""
    print("=== Physics Extraction System Test ===")

    # Set up the extraction environment
    print("Setting up extraction environment...")
    coordinator = setup_extraction_environment()

    # Show initial status
    print("\\nInitial status:")
    status = coordinator.get_extraction_status()
    print(f"  Available IDS: {status['available_ids']}")
    print(f"  Remaining IDS: {status['remaining_ids']}")
    print(f"  Current quantities in DB: {status['database_stats']['total_quantities']}")

    if status["remaining_ids"] == 0:
        print("\\nNo IDS remaining to process.")
        return

    # Extract from a small number of IDS for testing
    print("\\nStarting extraction (limited to 2 IDS, 5 paths each)...")
    try:
        session_id = run_extraction_session(
            coordinator=coordinator,
            max_ids=2,  # Only process 2 IDS for testing
            paths_per_ids=5,  # Only 5 paths per IDS
        )

        print(f"\\nExtraction completed successfully! Session ID: {session_id}")

        # Show final status
        final_status = coordinator.get_extraction_status()
        print("\\nFinal status:")
        print(
            f"  Total quantities found: {final_status['database_stats']['total_quantities']}"
        )
        print(
            f"  Verified quantities: {final_status['database_stats']['verified_quantities']}"
        )
        print(f"  Remaining IDS: {final_status['remaining_ids']}")
        print(
            f"  Completion: {final_status['extraction_progress']['completion_percentage']:.1f}%"
        )

        # Show some example quantities
        if coordinator.database.quantities:
            print("\\nExample extracted quantities:")
            for i, (qid, quantity) in enumerate(
                coordinator.database.quantities.items()
            ):
                if i >= 3:  # Show only first 3
                    break
                print(f"  - {quantity.name}: {quantity.description[:100]}...")
                print(
                    f"    Unit: {quantity.unit}, Confidence: {quantity.extraction_confidence:.2f}"
                )
                print(f"    IDS sources: {', '.join(quantity.ids_sources)}")
                print()

    except Exception as e:
        print(f"\\nExtraction failed: {e}")
        import traceback

        traceback.print_exc()


def test_status_display():
    """Test status display functionality."""
    print("\\n=== Status Display Test ===")

    coordinator = setup_extraction_environment()
    status = coordinator.get_extraction_status()

    print("\\nDetailed status information:")
    print("Database Statistics:")
    for key, value in status["database_stats"].items():
        print(f"  {key}: {value}")

    print("\\nExtraction Progress:")
    progress = status["extraction_progress"]
    for key, value in progress.items():
        print(f"  {key}: {value}")

    print("\\nConflicts:")
    conflicts = status["conflicts"]
    for key, value in conflicts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # Run basic tests
    test_status_display()
    test_basic_extraction()

    print("\\n=== Test Completed ===")
    print("You can now use the physics extraction system with:")
    print("  python -m imas_mcp.physics_extraction status")
    print("  python -m imas_mcp.physics_extraction extract --max-ids 5")

#!/usr/bin/env python3
"""
Main entry point for physics extraction system.

Provides CLI and programmatic interfaces for running physics extraction
on IMAS data dictionary JSON files.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .coordination import ExtractionCoordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_extraction_environment(
    json_data_dir: Optional[Path] = None,
    storage_dir: Optional[Path] = None,
    catalog_file: Optional[Path] = None,
) -> ExtractionCoordinator:
    """
    Set up the extraction environment with default directories.

    Args:
        json_data_dir: Directory containing IMAS JSON data
        storage_dir: Directory for storing extraction results
        catalog_file: Path to IDS catalog file

    Returns:
        Configured ExtractionCoordinator
    """
    # Default to standard project structure
    if json_data_dir is None:
        # Look for imas_mcp/resources/json_data/detailed relative to current working directory
        json_data_dir = Path("imas_mcp/resources/json_data/detailed")
        if not json_data_dir.exists():
            # Try relative to this file
            current_file = Path(__file__).parent
            project_root = current_file.parent.parent  # Go up to project root
            json_data_dir = (
                project_root / "imas_mcp" / "resources" / "json_data" / "detailed"
            )

    if storage_dir is None:
        # Use storage directory in project
        current_file = Path(__file__).parent
        project_root = current_file.parent.parent
        storage_dir = project_root / "storage" / "physics_extraction"

    # Ensure directories exist
    json_data_dir = Path(json_data_dir)
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    if not json_data_dir.exists():
        logger.warning(f"JSON data directory not found: {json_data_dir}")
        logger.info("You may need to run the data preparation scripts first")

    # Handle catalog file
    if catalog_file is None:
        # Default catalog location
        if json_data_dir:
            catalog_file = json_data_dir.parent / "ids_catalog.json"

        if not catalog_file or not catalog_file.exists():
            current_file = Path(__file__).parent
            project_root = current_file.parent.parent
            catalog_file = (
                project_root
                / "imas_mcp"
                / "resources"
                / "json_data"
                / "ids_catalog.json"
            )

    catalog_file = Path(catalog_file)
    if not catalog_file.exists():
        logger.warning(f"Catalog file not found: {catalog_file}")
        logger.info("Advanced features will be limited without catalog metadata")

    # Create coordinator
    coordinator = ExtractionCoordinator(
        json_data_dir=json_data_dir,
        storage_dir=storage_dir,
        catalog_file=catalog_file,
        ai_model="gpt-4",  # Can be configured
        confidence_threshold=0.5,
    )

    logger.info("Physics extraction environment initialized:")
    logger.info(f"  JSON data: {json_data_dir}")
    logger.info(f"  Catalog: {catalog_file}")
    logger.info(f"  Storage: {storage_dir}")

    return coordinator


def run_extraction_session(
    coordinator: ExtractionCoordinator,
    ids_list: Optional[List[str]] = None,
    paths_per_ids: int = 10,
    max_ids: Optional[int] = None,
) -> str:
    """
    Run a complete extraction session.

    Args:
        coordinator: ExtractionCoordinator instance
        ids_list: Specific IDS to process (None for all remaining)
        paths_per_ids: Number of paths to process per IDS
        max_ids: Maximum number of IDS to process

    Returns:
        Session ID
    """
    # Get status
    status = coordinator.get_extraction_status()
    logger.info("Current extraction status:")
    logger.info(
        f"  Database: {status['database_stats']['total_quantities']} quantities"
    )
    logger.info(f"  Remaining IDS: {status['remaining_ids']}")
    logger.info(
        f"  Completion: {status['extraction_progress']['completion_percentage']:.1f}%"
    )

    if status["is_locked"]:
        raise RuntimeError("Extraction is already in progress")

    # Determine IDS to process
    if ids_list is None:
        ids_list = status["remaining_ids_list"]

    if max_ids and ids_list and len(ids_list) > max_ids:
        ids_list = ids_list[:max_ids]

    if not ids_list:
        logger.info("No IDS to process")
        return ""

    logger.info(
        f"Starting extraction for {len(ids_list)} IDS: {ids_list[:5]}{'...' if len(ids_list) > 5 else ''}"
    )

    # Start extraction session
    session_id = coordinator.start_extraction(
        ids_list=ids_list, paths_per_ids=paths_per_ids, auto_resolve_conflicts=True
    )

    logger.info(f"Started extraction session: {session_id}")

    # Process all IDS in the session
    processed_count = 0
    while True:
        try:
            result = coordinator.process_next_ids(session_id)
            if result is None:
                break

            processed_count += 1
            logger.info(
                f"Processed {result.ids_name} ({processed_count}/{len(ids_list)}): "
                f"{len(result.quantities_found)} quantities found"
            )

            if result.warnings:
                for warning in result.warnings:
                    logger.warning(f"  {warning}")

        except Exception as e:
            logger.error(f"Error processing IDS: {e}")
            break

    # Complete session
    summary = coordinator.complete_extraction(session_id)

    logger.info("Extraction session completed:")
    logger.info(f"  Session: {session_id}")
    logger.info(
        f"  Processed: {summary['successful_ids']}/{summary['total_ids_processed']} IDS"
    )
    logger.info(f"  Quantities found: {summary['total_quantities_found']}")
    logger.info(f"  Conflicts: {summary['total_conflicts']}")
    logger.info(f"  Processing time: {summary['total_processing_time']:.1f}s")

    return session_id


def show_status(coordinator: ExtractionCoordinator):
    """Show current extraction status."""
    status = coordinator.get_extraction_status()

    print("\\n=== Physics Extraction Status ===")
    print(f"Database: {status['database_stats']['total_quantities']} quantities")
    print(f"  - Verified: {status['database_stats']['verified_quantities']}")
    print(f"  - Verification rate: {status['database_stats']['verification_rate']:.1%}")

    print("\nIDS Processing:")
    print(f"  - Available: {status['available_ids']}")
    print(f"  - Remaining: {status['remaining_ids']}")
    print(
        f"  - Completion: {status['extraction_progress']['completion_percentage']:.1f}%"
    )

    if status["remaining_ids"] > 0:
        print(f"  - Next to process: {', '.join(status['remaining_ids_list'][:5])}")

    print("\nConflicts:")
    print(f"  - Unresolved: {status['conflicts']['total_unresolved']}")
    print(f"  - Need review: {status['conflicts']['requiring_review']}")

    print("\nSystem:")
    print(f"  - Locked: {'Yes' if status['is_locked'] else 'No'}")
    print(f"  - Last updated: {status['last_updated']}")


def list_conflicts(coordinator: ExtractionCoordinator):
    """List conflicts requiring human review."""
    conflicts = coordinator.get_conflicts_for_review()

    if not conflicts:
        print("No conflicts requiring review.")
        return

    print(f"\\n=== Conflicts Requiring Review ({len(conflicts)}) ===")
    for conflict in conflicts:
        print(f"\\nConflict ID: {conflict['conflict_id']}")
        print(f"Quantity: {conflict['quantity_name']}")
        print(f"Fields: {', '.join(conflict['conflict_fields'])}")
        print(f"Requires review: {conflict['requires_human_review']}")


def export_database(coordinator: ExtractionCoordinator, output_file: Path):
    """Export physics database."""
    success = coordinator.export_database(output_file, format="json")
    if success:
        print(f"Database exported to: {output_file}")
    else:
        print(f"Failed to export database to: {output_file}")


def show_domains(coordinator: ExtractionCoordinator):
    """Show physics domains breakdown with categorization."""
    print("=== Physics Domains Summary ===")

    # Import the categorizer
    from ..core.physics_categorization import physics_categorizer

    # Get domain summary from categorizer
    domain_summary = physics_categorizer.get_domain_summary()

    # Display domain information
    for domain_name, info in domain_summary.items():
        if info["ids_count"] > 0:  # Only show domains with IDS
            print(f"\n{domain_name.upper()}:")
            print(f"  Description: {info['description']}")
            print(f"  IDS count: {info['ids_count']}")
            print(f"  Complexity: {info['complexity_level']}")
            print(f"  Primary phenomena: {', '.join(info['primary_phenomena'][:3])}")
            if len(info["primary_phenomena"]) > 3:
                print(f"    ... and {len(info['primary_phenomena']) - 3} more")
            print(f"  Typical units: {', '.join(info['typical_units'][:5])}")
            print(f"  Sample IDS: {', '.join(info['ids_list'][:3])}")
            if len(info["ids_list"]) > 3:
                print(f"    ... and {len(info['ids_list']) - 3} more")
            if info["related_domains"]:
                print(f"  Related domains: {', '.join(info['related_domains'][:3])}")

    # Get legacy domain summary from batch processor if available
    if (
        hasattr(coordinator, "catalog_batch_processor")
        and coordinator.catalog_batch_processor
    ):
        print("\n=== Legacy Domain Distribution ===")
        physics_summary = (
            coordinator.catalog_batch_processor.get_physics_domain_summary()
        )

        for domain, info in physics_summary.items():
            print(f"\n{domain.upper()}:")
            print(f"  IDS count: {info['ids_count']}")
            print(f"  Total paths: {info['total_paths']}")

    # Get processing progress by domain if available
    if hasattr(coordinator, "get_domain_progress"):
        domains = coordinator.get_domain_progress()
        if domains:
            print("\n=== Processing Progress by Domain ===")
            for domain, stats in domains.items():
                print(f"\n{domain}:")
                print(f"  Total IDS: {stats.get('total_ids', 0)}")
                print(f"  Completed: {stats.get('completed_ids', 0)}")
                print(f"  Failed: {stats.get('failed_ids', 0)}")
                print(f"  Total paths: {stats.get('total_paths', 0)}")
                print(f"  Processed paths: {stats.get('processed_paths', 0)}")
    else:
        print("\nNo processing progress available yet.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Physics Extraction System for IMAS Data Dictionary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current status
  python -m imas_mcp.physics_extraction status
  
  # Extract from 5 IDS with 20 paths each
  python -m imas_mcp.physics_extraction extract --max-ids 5 --paths-per-ids 20
  
  # Extract specific IDS
  python -m imas_mcp.physics_extraction extract --ids equilibrium core_profiles
  
  # Export database
  python -m imas_mcp.physics_extraction export --output physics_quantities.json
        """,
    )

    parser.add_argument(
        "--json-data-dir", type=Path, help="Directory containing IMAS JSON data"
    )

    parser.add_argument(
        "--storage-dir", type=Path, help="Directory for storing extraction results"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status command
    subparsers.add_parser("status", help="Show extraction status")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Run extraction")
    extract_parser.add_argument("--ids", nargs="+", help="Specific IDS to process")
    extract_parser.add_argument(
        "--max-ids", type=int, help="Maximum number of IDS to process"
    )
    extract_parser.add_argument(
        "--paths-per-ids",
        type=int,
        default=10,
        help="Number of paths to process per IDS",
    )

    # Conflicts command
    subparsers.add_parser("conflicts", help="List conflicts requiring review")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export database")
    export_parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output file for export"
    )

    # Domains command
    subparsers.add_parser("domains", help="Show physics domains breakdown")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.command:
        parser.print_help()
        return

    try:
        # Setup extraction environment
        coordinator = setup_extraction_environment(
            json_data_dir=args.json_data_dir, storage_dir=args.storage_dir
        )

        # Execute command
        if args.command == "status":
            show_status(coordinator)

        elif args.command == "extract":
            run_extraction_session(
                coordinator=coordinator,
                ids_list=args.ids,
                paths_per_ids=args.paths_per_ids,
                max_ids=args.max_ids,
            )

        elif args.command == "conflicts":
            list_conflicts(coordinator)

        elif args.command == "export":
            export_database(coordinator, args.output)

        elif args.command == "domains":
            show_domains(coordinator)

    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()

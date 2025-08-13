"""
AI-powered physics quantity extractors.

This module handles the AI-assisted extraction of physics quantities
from IMAS data dictionary JSON files.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from imas_mcp.physics.models import ExtractionResult, PhysicsQuantity

logger = logging.getLogger(__name__)


class AIPhysicsExtractor:
    """
    AI-powered extractor for physics quantities from IMAS data.

    Uses AI prompting to identify and extract physics quantities
    from IMAS data dictionary entries.
    """

    def __init__(
        self,
        ai_model: str = "gpt-4",
        confidence_threshold: float = 0.5,
        batch_size: int = 10,
    ):
        self.ai_model = ai_model
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size

    def extract_from_paths(
        self, ids_name: str, paths_data: dict[str, Any], max_paths: int | None = None
    ) -> ExtractionResult:
        """
        Extract physics quantities from IMAS paths.

        Args:
            ids_name: Name of the IDS being processed
            paths_data: Dictionary mapping paths to their data
            max_paths: Maximum number of paths to process

        Returns:
            ExtractionResult with found quantities
        """
        start_time = datetime.utcnow()
        result = ExtractionResult(
            ids_name=ids_name,
            ai_model=self.ai_model,
            confidence_threshold=self.confidence_threshold,
            started_at=start_time,
        )

        try:
            # Limit paths if requested
            paths_to_process = list(paths_data.keys())
            if max_paths:
                paths_to_process = paths_to_process[:max_paths]

            result.paths_processed = paths_to_process

            # Process paths in batches
            quantities = []
            for i in range(0, len(paths_to_process), self.batch_size):
                batch_paths = paths_to_process[i : i + self.batch_size]
                batch_data = {path: paths_data[path] for path in batch_paths}

                batch_quantities = self._extract_batch(ids_name, batch_data)
                quantities.extend(batch_quantities)

            # Filter by confidence
            high_confidence_quantities = [
                q
                for q in quantities
                if q.extraction_confidence >= self.confidence_threshold
            ]

            result.quantities_found = high_confidence_quantities
            result.mark_completed()

            # Calculate processing time
            end_time = datetime.utcnow()
            result.processing_time = (end_time - start_time).total_seconds()

            logger.info(
                f"Extracted {len(high_confidence_quantities)} quantities "
                f"from {len(paths_to_process)} paths in {ids_name}"
            )

        except Exception as e:
            logger.error(f"Extraction failed for {ids_name}: {e}")
            result.mark_failed(str(e))

        return result

    def _extract_batch(
        self, ids_name: str, batch_data: dict[str, Any]
    ) -> list[PhysicsQuantity]:
        """
        Extract physics quantities from a batch of paths.

        This is where the AI magic happens. For now, this is a placeholder
        that demonstrates the structure. In a real implementation, this would:
        1. Format the data for AI analysis
        2. Send to AI model with appropriate prompts
        3. Parse AI response into PhysicsQuantity objects
        4. Assign confidence scores

        Args:
            ids_name: IDS name
            batch_data: Path data to analyze

        Returns:
            List of extracted PhysicsQuantity objects
        """
        quantities = []

        for path, data in batch_data.items():
            # Simulate AI extraction logic
            # In practice, this would involve:
            # - Formatting data for AI analysis
            # - Sending to AI model with physics extraction prompts
            # - Parsing structured AI response

            extracted_quantities = self._analyze_path_with_ai(ids_name, path, data)
            quantities.extend(extracted_quantities)

        return quantities

    def _analyze_path_with_ai(
        self, ids_name: str, path: str, data: Any
    ) -> list[PhysicsQuantity]:
        """
        Analyze a single path with AI to extract physics quantities.

        This is a placeholder for actual AI integration.

        Args:
            ids_name: IDS name
            path: IMAS path
            data: Path data

        Returns:
            List of PhysicsQuantity objects
        """
        # Placeholder implementation
        # In reality, this would format prompts and call AI models

        # Example heuristic-based extraction for demonstration
        quantities = []

        # Extract from path structure and data
        if isinstance(data, dict):
            # Look for physics-related indicators
            description = data.get("description", "")
            unit = data.get("unit", data.get("units", ""))

            # Simple heuristics to identify physics quantities
            physics_indicators = [
                "temperature",
                "pressure",
                "density",
                "velocity",
                "energy",
                "magnetic",
                "electric",
                "plasma",
                "current",
                "voltage",
                "power",
                "frequency",
                "wavelength",
                "time",
                "mass",
            ]

            path_lower = path.lower()
            desc_lower = description.lower() if description else ""

            for indicator in physics_indicators:
                if indicator in path_lower or indicator in desc_lower:
                    # Create a physics quantity
                    quantity = PhysicsQuantity(
                        name=self._extract_quantity_name(path, data),
                        description=description or f"Physics quantity from {path}",
                        unit=unit if unit else None,
                        imas_paths=[path],
                        ids_sources={ids_name},
                        physics_context=self._infer_physics_context(path, data),
                        extraction_confidence=self._calculate_confidence(
                            path, data, indicator
                        ),
                        ai_source=self.ai_model,
                    )
                    quantities.append(quantity)
                    break  # Only create one quantity per path for now

        return quantities

    def _extract_quantity_name(self, path: str, data: dict) -> str:
        """Extract a human-readable name for the quantity."""
        # Try to get name from data
        if isinstance(data, dict):
            if "name" in data:
                return data["name"]
            if "description" in data:
                desc = data["description"]
                # Take first sentence as name
                if "." in desc:
                    return desc.split(".")[0].strip()
                return desc[:50] + "..." if len(desc) > 50 else desc

        # Fall back to path-based name
        path_parts = path.split("/")
        return path_parts[-1].replace("_", " ").title()

    def _infer_physics_context(self, path: str, data: dict) -> str | None:
        """Infer the physics context from path and data."""
        # Simple context inference
        context_map = {
            "equilibrium": "MHD Equilibrium",
            "core_profiles": "Core Plasma",
            "edge_profiles": "Edge Plasma",
            "summary": "Global Parameters",
            "wall": "Plasma-Wall Interaction",
            "disruption": "Disruptions",
            "transport": "Transport Physics",
            "current_drive": "Current Drive",
            "heating": "Heating & Current Drive",
        }

        path_lower = path.lower()
        for key, context in context_map.items():
            if key in path_lower:
                return context

        return "General Plasma Physics"

    def _calculate_confidence(self, path: str, data: dict, indicator: str) -> float:
        """Calculate extraction confidence score."""
        confidence = 0.3  # Base confidence

        # Boost confidence for explicit indicators
        if isinstance(data, dict):
            if data.get("unit") or data.get("units"):
                confidence += 0.3
            if data.get("description"):
                confidence += 0.2
                # Check if description mentions physics concepts
                desc_lower = data["description"].lower()
                physics_terms = [
                    "temperature",
                    "magnetic",
                    "plasma",
                    "current",
                    "field",
                ]
                if any(term in desc_lower for term in physics_terms):
                    confidence += 0.2

        # Path-based confidence
        if indicator in path.lower():
            confidence += 0.2

        return min(confidence, 1.0)


class BatchProcessor:
    """
    Manages batch processing of physics extraction across multiple IDS.

    Handles coordination, progress tracking, and resumable processing.
    """

    def __init__(
        self,
        json_data_dir: Path,
        extractor: AIPhysicsExtractor,
        default_paths_per_ids: int = 10,
    ):
        self.json_data_dir = Path(json_data_dir)
        self.extractor = extractor
        self.default_paths_per_ids = default_paths_per_ids

    def get_available_ids(self) -> list[str]:
        """Get list of available IDS from JSON data directory."""
        ids_list = []

        if not self.json_data_dir.exists():
            logger.warning(f"JSON data directory not found: {self.json_data_dir}")
            return ids_list

        # Look for JSON files in the data directory
        for json_file in self.json_data_dir.glob("*.json"):
            if json_file.stem not in ["index", "metadata"]:  # Skip special files
                ids_list.append(json_file.stem)

        return sorted(ids_list)

    def load_ids_data(self, ids_name: str) -> dict[str, Any] | None:
        """Load JSON data for a specific IDS."""
        json_file = self.json_data_dir / f"{ids_name}.json"

        if not json_file.exists():
            logger.error(f"JSON file not found for IDS: {ids_name}")
            return None

        try:
            with open(json_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON for {ids_name}: {e}")
            return None

    def extract_paths_from_ids_data(self, ids_data: dict[str, Any]) -> dict[str, Any]:
        """
        Extract path data from IDS JSON structure.

        The JSON structure varies, so we need to be flexible
        in extracting meaningful path information.
        """
        paths = {}

        def extract_recursive(data, current_path=""):
            """Recursively extract paths from nested data structure."""
            if isinstance(data, dict):
                for key, value in data.items():
                    new_path = f"{current_path}/{key}" if current_path else key

                    # If this looks like a terminal node with metadata, store it
                    if isinstance(value, dict) and (
                        "description" in value or "unit" in value
                    ):
                        paths[new_path] = value
                    else:
                        # Continue recursing
                        extract_recursive(value, new_path)
            elif isinstance(data, list) and len(data) > 0:
                # For arrays, examine the first element
                extract_recursive(data[0], current_path)

        extract_recursive(ids_data)
        return paths

    def process_ids(
        self, ids_name: str, max_paths: int | None = None
    ) -> ExtractionResult:
        """
        Process a single IDS for physics extraction.

        Args:
            ids_name: Name of the IDS to process
            max_paths: Maximum paths to process (None for all)

        Returns:
            ExtractionResult
        """
        logger.info(f"Starting extraction for IDS: {ids_name}")

        # Load IDS data
        ids_data = self.load_ids_data(ids_name)
        if not ids_data:
            result = ExtractionResult(
                ids_name=ids_name,
                ai_model=self.extractor.ai_model
                if hasattr(self, "extractor")
                else None,
            )
            result.mark_failed(f"Could not load data for IDS: {ids_name}")
            return result

        # Extract paths
        paths_data = self.extract_paths_from_ids_data(ids_data)
        logger.info(f"Found {len(paths_data)} paths in {ids_name}")

        if not paths_data:
            result = ExtractionResult(
                ids_name=ids_name, ai_model=self.extractor.ai_model
            )
            result.mark_failed(f"No extractable paths found in {ids_name}")
            return result

        # Use default if no max specified
        if max_paths is None:
            max_paths = self.default_paths_per_ids

        # Extract physics quantities
        result = self.extractor.extract_from_paths(ids_name, paths_data, max_paths)

        logger.info(
            f"Completed extraction for {ids_name}: "
            f"{len(result.quantities_found)} quantities found"
        )

        return result

    def process_multiple_ids(
        self, ids_list: list[str], paths_per_ids: int | None = None
    ) -> list[ExtractionResult]:
        """
        Process multiple IDS in sequence.

        Args:
            ids_list: List of IDS names to process
            paths_per_ids: Paths to process per IDS

        Returns:
            List of ExtractionResult objects
        """
        results = []

        for ids_name in ids_list:
            try:
                result = self.process_ids(ids_name, paths_per_ids)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {ids_name}: {e}")
                failed_result = ExtractionResult(
                    ids_name=ids_name, ai_model=self.extractor.ai_model
                )
                failed_result.mark_failed(str(e))
                results.append(failed_result)

        return results

    async def process_multiple_ids_async(
        self,
        ids_list: list[str],
        paths_per_ids: int | None = None,
        max_concurrent: int = 3,
    ) -> list[ExtractionResult]:
        """
        Process multiple IDS concurrently with limited parallelism.

        Args:
            ids_list: List of IDS names to process
            paths_per_ids: Paths to process per IDS
            max_concurrent: Maximum concurrent processing

        Returns:
            List of ExtractionResult objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single(ids_name: str) -> ExtractionResult:
            async with semaphore:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self.process_ids, ids_name, paths_per_ids
                )

        tasks = [process_single(ids_name) for ids_name in ids_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {ids_list[i]}: {result}")
                failed_result = ExtractionResult(
                    ids_name=ids_list[i], ai_model=self.extractor.ai_model
                )
                failed_result.mark_failed(str(result))
                final_results.append(failed_result)
            else:
                final_results.append(result)

        return final_results


class CatalogBatchProcessor(BatchProcessor):
    """
    Batch processor with catalog-based progress tracking.

    Uses IDS catalog metadata to provide detailed progress information
    including per-IDS path counts and physics domain categorization.
    """

    def __init__(
        self,
        json_data_dir: Path,
        catalog_file: Path,
        extractor: AIPhysicsExtractor,
        default_paths_per_ids: int = 10,
    ):
        super().__init__(json_data_dir, extractor, default_paths_per_ids)

        self.catalog_file = Path(catalog_file)
        self._load_catalog()

    def _load_catalog(self):
        """Load IDS catalog for enhanced tracking."""
        try:
            with open(self.catalog_file, encoding="utf-8") as f:
                catalog_data = json.load(f)

            self.catalog_metadata = catalog_data.get("metadata", {})
            self.ids_catalog = catalog_data.get("ids_catalog", {})

            logger.info(
                f"Enhanced processor loaded catalog with {len(self.ids_catalog)} IDS"
            )

        except Exception as e:
            logger.error(f"Failed to load catalog for enhanced processor: {e}")
            self.catalog_metadata = {}
            self.ids_catalog = {}

    def get_catalog_stats(self) -> dict[str, Any]:
        """Get catalog statistics."""
        return {
            "total_ids": self.catalog_metadata.get("total_ids", 0),
            "total_paths": self.catalog_metadata.get("total_paths", 0),
            "total_leaf_nodes": self.catalog_metadata.get("total_leaf_nodes", 0),
            "catalog_version": self.catalog_metadata.get("version", "unknown"),
        }

    def get_ids_with_metadata(self) -> dict[str, dict[str, Any]]:
        """Get available IDS with their catalog metadata."""
        available_ids = self.get_available_ids()

        ids_with_metadata = {}
        for ids_name in available_ids:
            catalog_info = self.ids_catalog.get(ids_name, {})
            ids_with_metadata[ids_name] = {
                "name": ids_name,
                "description": catalog_info.get("description", ""),
                "path_count": catalog_info.get("path_count", 0),
                "physics_domain": catalog_info.get("physics_domain", "general"),
                "has_json_data": True,  # Since we got it from get_available_ids()
            }

        return ids_with_metadata

    def get_physics_domain_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary by physics domain."""
        available_ids = self.get_available_ids()
        domains = {}

        for ids_name in available_ids:
            catalog_info = self.ids_catalog.get(ids_name, {})
            domain = catalog_info.get("physics_domain", "general")

            if domain not in domains:
                domains[domain] = {
                    "ids_count": 0,
                    "total_paths": 0,
                    "ids_list": [],
                }

            domains[domain]["ids_count"] += 1
            domains[domain]["total_paths"] += catalog_info.get("path_count", 0)
            domains[domain]["ids_list"].append(ids_name)

        return domains

    def process_ids_with_catalog_info(
        self, ids_name: str, max_paths: int | None = None
    ) -> ExtractionResult:
        """
        Process IDS with enhanced catalog-based information.

        Args:
            ids_name: Name of the IDS to process
            max_paths: Maximum paths to process (uses catalog info if None)

        Returns:
            ExtractionResult with enhanced metadata
        """
        # Get catalog information
        catalog_info = self.ids_catalog.get(ids_name, {})
        catalog_path_count = catalog_info.get("path_count", 0)

        # Determine paths to process
        if max_paths is None:
            max_paths = (
                min(self.default_paths_per_ids, catalog_path_count)
                if catalog_path_count > 0
                else self.default_paths_per_ids
            )

        logger.info(
            f"Processing {ids_name}: {max_paths} paths (catalog has {catalog_path_count})"
        )

        # Process using parent method
        result = super().process_ids(ids_name, max_paths)

        # Enhance result with catalog metadata
        if hasattr(result, "catalog_metadata"):
            result.catalog_metadata = {
                "catalog_path_count": catalog_path_count,
                "physics_domain": catalog_info.get("physics_domain", "general"),
                "description": catalog_info.get("description", ""),
                "coverage_percentage": (
                    len(result.paths_processed) / max(1, catalog_path_count)
                )
                * 100.0,
            }

        return result

    def estimate_processing_time(
        self, ids_list: list[str], paths_per_ids: int = 10
    ) -> dict[str, Any]:
        """
        Estimate processing time based on catalog metadata.

        Args:
            ids_list: List of IDS to process
            paths_per_ids: Paths to process per IDS

        Returns:
            Processing time estimates
        """
        total_paths = 0
        total_catalog_paths = 0

        for ids_name in ids_list:
            catalog_info = self.ids_catalog.get(ids_name, {})
            catalog_paths = catalog_info.get("path_count", 0)
            processing_paths = (
                min(paths_per_ids, catalog_paths)
                if catalog_paths > 0
                else paths_per_ids
            )

            total_paths += processing_paths
            total_catalog_paths += catalog_paths

        # Rough estimates (adjust based on actual performance)
        estimated_seconds_per_path = 0.5  # Conservative estimate
        total_estimated_time = total_paths * estimated_seconds_per_path

        return {
            "total_ids": len(ids_list),
            "total_processing_paths": total_paths,
            "total_catalog_paths": total_catalog_paths,
            "coverage_percentage": (total_paths / max(1, total_catalog_paths)) * 100.0,
            "estimated_time_seconds": total_estimated_time,
            "estimated_time_minutes": total_estimated_time / 60.0,
            "estimated_time_hours": total_estimated_time / 3600.0,
        }

"""
Processors for IMAS MCP tool data.

This module provides classes to process and convert data between different formats,
ensuring consistent use of Pydantic models throughout the codebase.
"""

from typing import Any, Dict, List

from imas_mcp.models.response_models import DataPath


class SearchResultProcessor:
    """Processor for converting search results to standardized Pydantic models."""

    @staticmethod
    def convert_to_data_path(search_result: Any) -> DataPath:
        """
        Convert a search result to DataPath Pydantic model.

        Args:
            search_result: Search result object with to_dict() method

        Returns:
            DataPath model with standardized fields
        """
        result_dict = search_result.to_dict()

        return DataPath(
            path=result_dict["path"],
            ids_name=result_dict["ids_name"],
            documentation=result_dict["documentation"],
            relevance_score=result_dict["relevance_score"],
            physics_domain=result_dict.get("physics_domain"),
            units=result_dict.get("units"),
            data_type=result_dict.get("data_type"),
            identifier=IdentifierProcessor.extract_identifier_info(
                search_result.document
            ),
        )

    @staticmethod
    def convert_results_list(search_results: List[Any]) -> List[DataPath]:
        """
        Convert a list of search results to DataPath models.

        Args:
            search_results: List of search result objects

        Returns:
            List of DataPath models
        """
        return [
            SearchResultProcessor.convert_to_data_path(result)
            for result in search_results
        ]


class IdentifierProcessor:
    """Processor for extracting and formatting identifier information."""

    @staticmethod
    def extract_identifier_info(document: Any) -> Dict[str, Any]:
        """
        Extract identifier information from a document.

        Args:
            document: Document object with raw_data attribute

        Returns:
            Dictionary with identifier information
        """
        if not hasattr(document, "raw_data") or not document.raw_data:
            return {"has_identifier": False}

        identifier_schema = document.raw_data.get("identifier_schema")
        if not identifier_schema:
            return {"has_identifier": False}

        # Extract identifier options and metadata
        options = []
        option_count = 0

        if isinstance(identifier_schema, dict):
            schema_options = identifier_schema.get("options", [])
            if isinstance(schema_options, list):
                options = schema_options[:5]  # Limit to first 5 for performance
                option_count = len(schema_options)

        # Determine branching significance
        if option_count > 10:
            significance = "Critical branching point with many options"
        elif option_count > 3:
            significance = "Important data structure branch"
        elif option_count > 0:
            significance = "Simple identifier branch"
        else:
            significance = "Identifier present but no options detected"

        return {
            "has_identifier": True,
            "schema_path": document.raw_data.get("path", ""),
            "option_count": option_count,
            "branching_significance": significance,
            "sample_options": options,
        }

    @staticmethod
    def summarize_identifier_analysis(
        identifier_schemas: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Create summary analysis of identifier schemas.

        Args:
            identifier_schemas: List of identifier schema dictionaries

        Returns:
            Dictionary with analysis summary
        """
        if not identifier_schemas:
            return {
                "branching_paths": 0,
                "schemas": [],
                "significance": "No branching logic identified",
            }

        total_options = sum(
            schema.get("option_count", 0) for schema in identifier_schemas
        )

        if total_options > 50:
            significance = "Complex data structure with extensive branching"
        elif total_options > 20:
            significance = "Moderate complexity with significant branching options"
        elif total_options > 5:
            significance = "Some branching complexity present"
        else:
            significance = "Simple branching structure"

        return {
            "branching_paths": len(identifier_schemas),
            "schemas": identifier_schemas,
            "significance": significance,
            "total_options": total_options,
        }
